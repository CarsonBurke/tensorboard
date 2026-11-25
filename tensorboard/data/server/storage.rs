// Copyright 2026 The TensorFlow Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0

//! Persistent, request-owned data storage. SQLite owns the catalog and reservoir
//! contents; no loader or query cache retains a catalog-sized Rust collection.
//! Each committed source record atomically advances its offset and all affected
//! reservoirs. WAL readers see usable prefixes while the writer continues.

use bytes::Bytes;
use prost::Message;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rusqlite::{config::DbConfig, params, Connection, OptionalExtension};
use std::borrow::Borrow;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::convert::TryInto;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::RwLock;
use std::time::Instant;
use tonic::Status;

use crate::commit::{BlobSequenceValue, Commit, ScalarValue, TimeSeries};
use crate::data_compat::{EventValue, GraphDefValue, SummaryValue, TaggedRunMetadataValue};
use crate::logdir::{EventFileBuf, FileFingerprint, Logdir};
use crate::proto::tensorboard::{self as pb, data};
use crate::reservoir::{Basin, Capacity, ReservoirControl};
use crate::types::{PluginSamplingHint, Run, Step, Tag, WallTime};

type Result<T, E = Box<dyn std::error::Error + Send + Sync>> = std::result::Result<T, E>;

/// Schema `user_version` at which the current [`compare_tag_names`] was
/// introduced. Future comparator changes need a new collation name and migration.
const COLLATION_VERSION: i64 = 3;

/// Schema `user_version` at which `samples` and `blobs` became rowid tables.
const ROWID_SAMPLES_VERSION: i64 = 4;

/// Sample and blob tables. They are rowid tables on purpose: in a WITHOUT
/// ROWID table every B-tree cell, interior ones included, stores the whole row,
/// so searches compare against (and read the overflow pages of) large values.
/// `blobs` names `samples` as its parent even when created beside a migration
/// copy, so the reference resolves once the copy takes the name.
fn sample_tables_ddl(if_not_exists: &str, samples: &str, blobs: &str) -> String {
    format!(
        "CREATE TABLE {if_not_exists} {samples}(series INTEGER NOT NULL REFERENCES series(id) ON DELETE CASCADE,
            step INTEGER NOT NULL, wall REAL NOT NULL, value BLOB, length INTEGER, UNIQUE(series,step));
        CREATE TABLE {if_not_exists} {blobs}(series INTEGER NOT NULL, step INTEGER NOT NULL, idx INTEGER NOT NULL,
            value BLOB NOT NULL, UNIQUE(series,step,idx),
            FOREIGN KEY(series,step) REFERENCES samples(series,step) ON DELETE CASCADE);"
    )
}

#[derive(Debug)]
pub struct DiskStore {
    path: PathBuf,
    identity: String,
    epoch: u128,
    /// The reloader's connection, kept between cycles so an unchanged index
    /// is answered from its page cache (SQLite revalidates it against the
    /// WAL on each transaction) instead of re-read from disk every cycle.
    /// Paired with the file it opened, in case another process's repair
    /// replaced the index.
    loader: std::sync::Mutex<Option<(Connection, Option<FileId>)>>,
}

/// Identifies the file behind a path, where the platform can say.
type FileId = (u64, u64);

fn file_id(path: &Path) -> Option<FileId> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        std::fs::metadata(path).ok().map(|m| (m.dev(), m.ino()))
    }
    // On Windows open SQLite files cannot be unlinked.
    #[cfg(not(unix))]
    {
        let _ = path;
        None
    }
}

/// Set when a request hits a damaged index; the reloader repairs it.
static CORRUPTION_REPORTED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Whether a request reported index corruption since the last call.
pub fn take_corruption_report() -> bool {
    CORRUPTION_REPORTED.swap(false, std::sync::atomic::Ordering::Relaxed)
}

/// Maps a poisoned request-local lock; never an index problem.
fn lock_status<T>(e: std::sync::PoisonError<T>) -> Status {
    log::error!("Data index: {}", e);
    Status::internal("failed to access data index")
}

fn status(e: impl std::fmt::Display + 'static) -> Status {
    log::error!("Data index: {}", e);
    if let Some(e) = (&e as &dyn std::any::Any).downcast_ref::<rusqlite::Error>() {
        if is_corruption(e) {
            CORRUPTION_REPORTED.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }
    Status::internal("failed to access data index")
}

impl DiskStore {
    pub fn open(logdir: &str, hints: &PluginSamplingHint, checksum: bool) -> Result<Self> {
        let location = if logdir.contains("://") {
            logdir.to_owned()
        } else {
            let path = Path::new(logdir);
            let absolute = if path.is_absolute() {
                path.to_owned()
            } else {
                std::env::current_dir()?.join(path)
            };
            absolute
                .canonicalize()
                .unwrap_or(absolute)
                .to_string_lossy()
                .into_owned()
        };
        let mut sampling: Vec<_> = hints.0.iter().collect();
        sampling.sort_by_key(|(name, _)| *name);
        // The format version is part of the identity: incompatible schemas and
        // sampling algorithms never reinterpret an existing index.
        let identity = format!(
            "rustboard-index-v2\n{}\n{:?}\n{}",
            location, sampling, checksum
        );
        let hash = ring::digest::digest(&ring::digest::SHA256, identity.as_bytes());
        let key: String = hash.as_ref().iter().map(|b| format!("{:02x}", b)).collect();
        let cache = std::env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .filter(|p| p.is_absolute())
            .or_else(|| {
                std::env::var_os("LOCALAPPDATA")
                    .map(PathBuf::from)
                    .filter(|p| p.is_absolute())
            })
            .or_else(|| {
                std::env::var_os("HOME")
                    .map(|h| PathBuf::from(h).join(".cache"))
                    .filter(|p| p.is_absolute())
            })
            .ok_or("cannot determine absolute user cache directory")?;
        let dir = cache.join("tensorboard").join(key);
        let resolved = dir.canonicalize().unwrap_or_else(|_| {
            cache
                .canonicalize()
                .unwrap_or(cache)
                .join("tensorboard")
                .join(dir.file_name().unwrap())
        });
        if !logdir.contains("://") && resolved.starts_with(Path::new(&location)) {
            return Err("user cache directory must be outside the source logdir".into());
        }
        std::fs::create_dir_all(&dir)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700))?;
        }
        Self::at(dir.join("index.sqlite"), &identity)
    }

    fn at(path: PathBuf, identity: &str) -> Result<Self> {
        let store = Self {
            path,
            identity: identity.to_owned(),
            epoch: rand::random(),
            loader: Default::default(),
        };
        match store.initialize() {
            Err(e) if is_corruption(&*e) => store.repair()?,
            result => result?,
        }
        Ok(store)
    }

    fn initialize(&self) -> Result<()> {
        let c = self.connect()?;
        c.execute_batch("PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS info(identity TEXT NOT NULL, revision INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS runs(name TEXT PRIMARY KEY, start REAL);
            CREATE INDEX IF NOT EXISTS runs_start ON runs(start,name);
            CREATE TABLE IF NOT EXISTS files(path BLOB PRIMARY KEY, run TEXT NOT NULL REFERENCES runs(name) ON DELETE CASCADE,
                identity TEXT, modified TEXT, length INTEGER, offset INTEGER NOT NULL DEFAULT 0,
                dead INTEGER NOT NULL DEFAULT 0, present INTEGER NOT NULL DEFAULT 1);
            CREATE INDEX IF NOT EXISTS files_run ON files(run,path);
            CREATE TABLE IF NOT EXISTS series(id INTEGER PRIMARY KEY, run TEXT NOT NULL REFERENCES runs(name) ON DELETE CASCADE,
                tag TEXT NOT NULL, plugin TEXT NOT NULL, class INTEGER NOT NULL, metadata BLOB NOT NULL,
                capacity INTEGER NOT NULL, seen INTEGER NOT NULL DEFAULT 0, rng TEXT NOT NULL DEFAULT '0',
                sample_count INTEGER NOT NULL DEFAULT 0, valid INTEGER NOT NULL DEFAULT 0, max_step INTEGER, max_wall REAL, max_length INTEGER,
                UNIQUE(run,tag));
            CREATE INDEX IF NOT EXISTS series_plugin_tag ON series(plugin,class,tag,run);
            CREATE INDEX IF NOT EXISTS series_run_plugin_tag ON series(run,plugin,class,tag);
            CREATE TABLE IF NOT EXISTS plugins(name TEXT PRIMARY KEY, count INTEGER NOT NULL);
            CREATE TRIGGER IF NOT EXISTS add_series AFTER INSERT ON series WHEN NEW.class IN (1,2,3) BEGIN
                INSERT INTO plugins VALUES(NEW.plugin,1) ON CONFLICT(name) DO UPDATE SET count=count+1;
                UPDATE info SET revision=revision+1;
            END;
            CREATE TRIGGER IF NOT EXISTS remove_series AFTER DELETE ON series WHEN OLD.class IN (1,2,3) BEGIN
                UPDATE plugins SET count=count-1 WHERE name=OLD.plugin;
                DELETE FROM plugins WHERE count=0;
                UPDATE info SET revision=revision+1;
            END;
            CREATE TRIGGER IF NOT EXISTS series_shape AFTER UPDATE OF valid,max_length ON series
                WHEN (OLD.valid=0)!=(NEW.valid=0) OR OLD.max_length IS NOT NEW.max_length BEGIN
                UPDATE info SET revision=revision+1;
            END;
            CREATE TRIGGER IF NOT EXISTS add_run AFTER INSERT ON runs BEGIN UPDATE info SET revision=revision+1; END;
            CREATE TRIGGER IF NOT EXISTS remove_run AFTER DELETE ON runs BEGIN UPDATE info SET revision=revision+1; END;")?;
        c.execute_batch(&sample_tables_ddl("IF NOT EXISTS", "samples", "blobs"))?;
        let created = c.execute(
            // A random origin keeps revision tokens unique across index
            // recreation (see `repair`); tokens are only compared for equality.
            "INSERT INTO info SELECT ?1,abs(random()>>16) WHERE NOT EXISTS(SELECT 1 FROM info)",
            params![self.identity],
        )? == 1;
        let actual: String = c.query_row("SELECT identity FROM info", [], |r| r.get(0))?;
        if actual != self.identity {
            return Err("data index identity mismatch".into());
        }
        if created {
            // Migrating an empty index is instant; requests then never see a
            // partial schema before the first reload.
            Self::migrate_on(&c)?;
        }
        Ok(())
    }

    /// Brings an index written by an older binary up to date. Runs in the
    /// reloader rather than at open: rebuilding a large index takes minutes,
    /// and requests keep reading the previous WAL snapshot meanwhile, whereas a
    /// server that has not started serving fails TensorBoard's startup probe.
    ///
    /// Returns whether the schema version changed.
    #[cfg(test)]
    fn migrate(&self) -> Result<bool> {
        Self::migrate_on(&self.connect()?)
    }

    fn migrate_on(c: &Connection) -> Result<bool> {
        let version =
            |c: &Connection| c.query_row("PRAGMA user_version", [], |r| r.get::<_, i64>(0));
        let before = version(c)?;
        Self::migrate_catalog_counts(c)?;
        Self::migrate_metrics_catalog(c)?;
        Self::migrate_collation(c)?;
        Self::migrate_rowid_samples(c)?;
        // Older binaries sharing this index recreate these at open.
        Self::drop_legacy_sample_indexes(c)?;
        Ok(version(c)? != before)
    }

    /// Loaders track valid-sample extrema themselves and reads use the
    /// (series,step) key, so these partial indexes only cost writes. Checking
    /// first keeps the usual case read-only.
    fn drop_legacy_sample_indexes(c: &Connection) -> rusqlite::Result<()> {
        let names = [
            "samples_valid_step",
            "samples_valid_wall",
            "samples_valid_length",
        ];
        for name in names {
            let exists = c
                .query_row(
                    "SELECT 1 FROM sqlite_schema WHERE type='index' AND name=?1",
                    [name],
                    |_| Ok(()),
                )
                .optional()?
                .is_some();
            if exists {
                c.execute_batch(&format!("DROP INDEX IF EXISTS {}", name))?;
            }
        }
        Ok(())
    }

    /// Versioned, atomic derived-index migration. Counts are maintained by the
    /// same transactions as series validity, not by an in-process query cache.
    fn migrate_catalog_counts(c: &Connection) -> rusqlite::Result<()> {
        let version: i64 = c.query_row("PRAGMA user_version", [], |row| row.get(0))?;
        if version >= 1 {
            return Ok(());
        }
        let tx = rusqlite::Transaction::new_unchecked(c, rusqlite::TransactionBehavior::Immediate)?;
        let version: i64 = tx.query_row("PRAGMA user_version", [], |row| row.get(0))?;
        if version < 1 {
            tx.execute_batch("
                CREATE INDEX series_list_plugin_tag ON series(plugin,class,tag,run) WHERE valid>0;
                CREATE INDEX series_list_run_plugin_tag ON series(run,plugin,class,tag) WHERE valid>0;
                CREATE TABLE series_counts(run TEXT NOT NULL,plugin TEXT NOT NULL,class INTEGER NOT NULL,
                    count INTEGER NOT NULL,PRIMARY KEY(run,plugin,class)) WITHOUT ROWID;
                INSERT INTO series_counts SELECT run,plugin,class,count(*) FROM series WHERE valid>0 GROUP BY run,plugin,class;
                CREATE TRIGGER count_insert AFTER INSERT ON series WHEN NEW.valid>0 BEGIN
                    INSERT INTO series_counts VALUES(NEW.run,NEW.plugin,NEW.class,1)
                        ON CONFLICT(run,plugin,class) DO UPDATE SET count=count+1;
                END;
                CREATE TRIGGER count_remove AFTER DELETE ON series WHEN OLD.valid>0 BEGIN
                    UPDATE series_counts SET count=count-1 WHERE run=OLD.run AND plugin=OLD.plugin AND class=OLD.class;
                    DELETE FROM series_counts WHERE run=OLD.run AND plugin=OLD.plugin AND class=OLD.class AND count=0;
                END;
                CREATE TRIGGER count_valid AFTER UPDATE OF valid ON series WHEN OLD.valid=0 AND NEW.valid>0 BEGIN
                    INSERT INTO series_counts VALUES(NEW.run,NEW.plugin,NEW.class,1)
                        ON CONFLICT(run,plugin,class) DO UPDATE SET count=count+1;
                END;
                CREATE TRIGGER count_invalid AFTER UPDATE OF valid ON series WHEN OLD.valid>0 AND NEW.valid=0 BEGIN
                    UPDATE series_counts SET count=count-1 WHERE run=OLD.run AND plugin=OLD.plugin AND class=OLD.class;
                    DELETE FROM series_counts WHERE run=OLD.run AND plugin=OLD.plugin AND class=OLD.class AND count=0;
                END;
                PRAGMA user_version=1;
            ")?;
        }
        tx.commit()
    }

    fn migrate_metrics_catalog(c: &Connection) -> rusqlite::Result<()> {
        let version: i64 = c.query_row("PRAGMA user_version", [], |r| r.get(0))?;
        if version >= 2 {
            return Ok(());
        }
        let tx = rusqlite::Transaction::new_unchecked(c, rusqlite::TransactionBehavior::Immediate)?;
        let version: i64 = tx.query_row("PRAGMA user_version", [], |r| r.get(0))?;
        if version < 2 {
            create_metrics_catalog(&tx)?;
            tx.execute_batch("
                INSERT INTO metrics_catalog
                    SELECT id,run,tag,plugin,
                        CASE instr(tag,'/') WHEN 0 THEN tag ELSE substr(tag,1,instr(tag,'/')-1) END,
                        CASE plugin WHEN 'images' THEN max(0,coalesce(max_length,0)-2) ELSE 1 END
                    FROM series WHERE valid>0 AND supported_metrics_metadata(metadata) AND
                        ((plugin='scalars' AND class=1) OR (plugin='histograms' AND class=2) OR (plugin='images' AND class=3));
                INSERT INTO metrics_catalog_groups
                    SELECT run,plugin,category,sum(cards) FROM metrics_catalog
                    WHERE cards>0 GROUP BY run,plugin,category;
                CREATE TRIGGER metrics_insert AFTER INSERT ON series WHEN NEW.valid>0 AND supported_metrics_metadata(NEW.metadata) AND
                    ((NEW.plugin='scalars' AND NEW.class=1) OR (NEW.plugin='histograms' AND NEW.class=2) OR (NEW.plugin='images' AND NEW.class=3)) BEGIN
                    INSERT INTO metrics_catalog VALUES(NEW.id,NEW.run,NEW.tag,NEW.plugin,
                        CASE instr(NEW.tag,'/') WHEN 0 THEN NEW.tag ELSE substr(NEW.tag,1,instr(NEW.tag,'/')-1) END,
                        CASE NEW.plugin WHEN 'images' THEN max(0,coalesce(NEW.max_length,0)-2) ELSE 1 END);
                END;
                CREATE TRIGGER metrics_remove AFTER DELETE ON series BEGIN
                    DELETE FROM metrics_catalog WHERE id=OLD.id;
                END;
                CREATE TRIGGER metrics_shape AFTER UPDATE OF valid,max_length ON series
                    WHEN (OLD.valid=0)!=(NEW.valid=0) OR OLD.max_length IS NOT NEW.max_length BEGIN
                    DELETE FROM metrics_catalog WHERE id=OLD.id;
                    INSERT INTO metrics_catalog SELECT NEW.id,NEW.run,NEW.tag,NEW.plugin,
                        CASE instr(NEW.tag,'/') WHEN 0 THEN NEW.tag ELSE substr(NEW.tag,1,instr(NEW.tag,'/')-1) END,
                        CASE NEW.plugin WHEN 'images' THEN max(0,coalesce(NEW.max_length,0)-2) ELSE 1 END
                    WHERE NEW.valid>0 AND supported_metrics_metadata(NEW.metadata) AND ((NEW.plugin='scalars' AND NEW.class=1) OR
                        (NEW.plugin='histograms' AND NEW.class=2) OR (NEW.plugin='images' AND NEW.class=3));
                END;
                CREATE TRIGGER metrics_group_insert AFTER INSERT ON metrics_catalog WHEN NEW.cards>0 BEGIN
                    INSERT INTO metrics_catalog_groups VALUES(NEW.run,NEW.plugin,NEW.category,NEW.cards)
                        ON CONFLICT(run,plugin,category) DO UPDATE SET cards=cards+NEW.cards;
                END;
                CREATE TRIGGER metrics_group_remove AFTER DELETE ON metrics_catalog WHEN OLD.cards>0 BEGIN
                    UPDATE metrics_catalog_groups SET cards=cards-OLD.cards
                        WHERE run=OLD.run AND plugin=OLD.plugin AND category=OLD.category;
                    DELETE FROM metrics_catalog_groups
                        WHERE run=OLD.run AND plugin=OLD.plugin AND category=OLD.category AND cards=0;
                END;
                PRAGMA user_version=2;
            ")?;
        }
        tx.commit()
    }

    /// Switches to a new collation name so old binaries sharing this cache
    /// cannot write keys into the new index using the old order.
    fn migrate_collation(c: &Connection) -> rusqlite::Result<()> {
        let version: i64 = c.query_row("PRAGMA user_version", [], |r| r.get(0))?;
        if version >= COLLATION_VERSION {
            return Ok(());
        }
        let tx = rusqlite::Transaction::new_unchecked(c, rusqlite::TransactionBehavior::Immediate)?;
        let version: i64 = tx.query_row("PRAGMA user_version", [], |r| r.get(0))?;
        if version < COLLATION_VERSION {
            tx.execute_batch(&format!(
                "DROP INDEX metrics_catalog_scope;
                 CREATE INDEX metrics_catalog_scope ON metrics_catalog(
                     run,plugin,category,tag COLLATE TAG_NAME_V2,cards
                 );
                 PRAGMA user_version={};",
                COLLATION_VERSION
            ))?;
        }
        tx.commit()
    }

    /// Rebuilds `samples` and `blobs` from WITHOUT ROWID tables, whose B-tree
    /// cells hold whole rows: every key comparison on a cell with a large value
    /// (text, images) read its full overflow chain, so one event could read
    /// megabytes. Rowid tables keep values out of every search path. Also
    /// drops the partial `samples` indexes that the loader no longer needs.
    fn migrate_rowid_samples(c: &Connection) -> rusqlite::Result<()> {
        let version: i64 = c.query_row("PRAGMA user_version", [], |r| r.get(0))?;
        if version >= ROWID_SAMPLES_VERSION {
            return Ok(());
        }
        // Enforcement cannot change inside a transaction, and dropping the
        // parent `samples` table with it on would cascade through `blobs`.
        c.execute_batch("PRAGMA foreign_keys=OFF")?;
        let migrated = (|| {
            let tx =
                rusqlite::Transaction::new_unchecked(c, rusqlite::TransactionBehavior::Immediate)?;
            let version: i64 = tx.query_row("PRAGMA user_version", [], |r| r.get(0))?;
            if version >= ROWID_SAMPLES_VERSION {
                return tx.commit();
            }
            let without_rowid = |table: &str| {
                tx.query_row(
                    "SELECT wr FROM pragma_table_list WHERE schema='main' AND name=?1",
                    [table],
                    |r| r.get::<_, bool>(0),
                )
            };
            // Dropping these first lets the copies reuse their pages.
            Self::drop_legacy_sample_indexes(&tx)?;
            if without_rowid("samples")? || without_rowid("blobs")? {
                log::info!("Migrating data index samples to rowid tables");
                // The copy appends to the new trees' right edges while reading
                // the old ones; with the default 2 MiB cache those pages were
                // spilled to the WAL and re-read throughout. Lasts only as long
                // as this connection.
                tx.execute_batch("PRAGMA cache_size=-262144")?;
                tx.execute_batch(&format!(
                    "{}
                    INSERT INTO samples_rowid(series,step,wall,value,length)
                        SELECT series,step,wall,value,length FROM samples ORDER BY series,step;
                    INSERT INTO blobs_rowid(series,step,idx,value)
                        SELECT series,step,idx,value FROM blobs ORDER BY series,step,idx;
                    DROP TABLE blobs;
                    DROP TABLE samples;
                    ALTER TABLE samples_rowid RENAME TO samples;
                    ALTER TABLE blobs_rowid RENAME TO blobs;",
                    sample_tables_ddl("", "samples_rowid", "blobs_rowid"),
                ))?;
                let violation: Option<String> = tx
                    .query_row("PRAGMA foreign_key_check", [], |r| r.get(0))
                    .optional()?;
                if let Some(table) = violation {
                    log::warn!("Data index has dangling {} rows after migration", table);
                }
            }
            tx.execute_batch(&format!("PRAGMA user_version={};", ROWID_SAMPLES_VERSION))?;
            tx.commit()
        })();
        c.execute_batch("PRAGMA foreign_keys=ON")?;
        migrated
    }

    fn connect(&self) -> rusqlite::Result<Connection> {
        // Every connection, including loader writers, must know index collations.
        let c = Connection::open(&self.path)?;
        c.create_collation("TAG_NAME", compare_legacy_tag_names)?;
        c.create_collation("TAG_NAME_V2", compare_tag_names)?;
        c.create_scalar_function(
            "supported_metrics_metadata",
            1,
            rusqlite::functions::FunctionFlags::SQLITE_DETERMINISTIC,
            |ctx| {
                Ok(ctx
                    .get_raw(0)
                    .as_blob()
                    .ok()
                    .and_then(|bytes| pb::SummaryMetadata::decode(bytes).ok())
                    .map_or(false, |md| supported_metrics_metadata(&md)))
            },
        )?;
        // Wait for short transactions, never fail a selected request because a
        // different process is committing a source record.
        c.busy_handler(Some(|_| {
            std::thread::sleep(std::time::Duration::from_millis(10));
            true
        }))?;
        c.execute_batch(
            "PRAGMA foreign_keys=ON; PRAGMA temp_store=FILE; PRAGMA synchronous=NORMAL;",
        )?;
        Ok(c)
    }

    pub fn revision(&self) -> std::result::Result<String, Status> {
        let c = self.connect().map_err(status)?;
        let n: i64 = c
            .query_row("SELECT revision FROM info", [], |r| r.get(0))
            .map_err(status)?;
        Ok(format!("{:032x}:{}", self.epoch, n))
    }

    /// Repairs an index that SQLite reported as corrupt. The index is derived
    /// from the logdir, so repair never needs the damaged contents: REINDEX
    /// rebuilds secondary indexes from their tables (the common case, since
    /// only collated indexes depend on application code), and if the tables
    /// themselves are damaged SQLite resets the index for a later reload.
    pub fn repair(&self) -> Result<()> {
        let lock = self.reload_lock()?;
        // Its cached pages and schema may be the damaged ones.
        *self
            .loader
            .lock()
            .map_err(|_| "loader connection lock poisoned")? = None;
        let reindexed = self.connect().and_then(|c| {
            c.execute_batch("REINDEX")?;
            // quick_check does not compare index order or contents.
            c.query_row("PRAGMA integrity_check(1)", [], |r| r.get::<_, String>(0))
        });
        match reindexed {
            Ok(result) if result == "ok" => return Ok(()),
            Ok(result) => {
                log::error!("Data index still damaged after REINDEX: {}", result);
            }
            Err(e) => {
                log::error!("Cannot REINDEX data index: {}", e);
            }
        }
        log::warn!("Resetting data index {} to rebuild it", self.path.display());
        // A reset VACUUM truncates the index through SQLite's own locking, so
        // connections in this and other processes stay consistent. Unlinking
        // the files instead can pair a new main file with the old WAL, and
        // leaves other processes' kept loaders writing to the deleted file.
        // A bare connection, since `connect()`'s pragmas need a readable
        // schema; and a bounded wait, since a damaged schema can drop the
        // reset to rollback mode, which needs other connections gone.
        Connection::open(&self.path).and_then(|c| {
            c.busy_timeout(std::time::Duration::from_secs(10))?;
            // SQLite's reset recipe: try to load the schema, ignoring damage.
            let _ = c.query_row("SELECT count(*) FROM sqlite_schema", [], |_| Ok(()));
            c.set_db_config(DbConfig::SQLITE_DBCONFIG_RESET_DATABASE, true)?;
            c.execute_batch("VACUUM")?;
            c.set_db_config(DbConfig::SQLITE_DBCONFIG_RESET_DATABASE, false)?;
            Ok(())
        })?;
        self.initialize()?;
        drop(lock);
        Ok(())
    }

    /// A separate SQLite database supplies a crash-released process lock that
    /// serializes index writers without locking the data database or readers.
    fn reload_lock(&self) -> Result<Connection> {
        let lock = Connection::open(self.path.with_extension("reload.sqlite"))?;
        lock.busy_handler(Some(|_| {
            std::thread::sleep(std::time::Duration::from_millis(10));
            true
        }))?;
        lock.execute_batch("BEGIN EXCLUSIVE")?;
        Ok(lock)
    }

    pub fn reload(
        &self,
        logdir: &impl Logdir,
        hints: &PluginSamplingHint,
        checksum: bool,
    ) -> Result<()> {
        let _lock = self.reload_lock()?;
        let mut loader = self
            .loader
            .lock()
            .map_err(|_| "loader connection lock poisoned")?;
        let current = file_id(&self.path);
        let (mut c, id) = match loader.take() {
            Some((c, id)) if id == current => (c, id),
            _ => {
                // Identify the file before opening it: if it is replaced in
                // between, the next cycle sees a mismatch and reconnects.
                let c = self.connect()?;
                // Without this a kept connection, which never closes the
                // index, would leave the WAL at its largest size (a full
                // copy of the samples after a migration).
                c.execute_batch("PRAGMA journal_size_limit=67108864;")?;
                (c, current)
            }
        };
        let migrated = Self::migrate_on(&c)?;
        self.reload_on(&mut c, logdir, hints, checksum)?;
        if migrated {
            // A static logdir may not write again to trigger the WAL size
            // limit. Close this connection so SQLite can remove the WAL.
            return Ok(());
        }
        // Keep the connection only after a clean cycle, with a cache trimmed
        // to what an idle cycle touches (`runs`, `files`, interior pages).
        c.execute_batch("PRAGMA cache_size=-4096;")?;
        *loader = Some((c, id));
        Ok(())
    }

    fn reload_on(
        &self,
        c: &mut Connection,
        logdir: &impl Logdir,
        hints: &PluginSamplingHint,
        checksum: bool,
    ) -> Result<()> {
        // Discovery and the diff below only read the index: an unchanged logdir
        // costs a directory walk and one ordered scan of `files`, never a write.
        let mut discovered = Vec::new();
        logdir.visit(&mut |run, file, fingerprint| {
            discovered.push(Discovered {
                path: encode_path(&file.0),
                run: run.0,
                fingerprint,
            });
            Ok(())
        })?;
        // BLOB primary keys order by memcmp, exactly like `Vec<u8>`.
        discovered.sort_unstable_by(|a, b| a.path.cmp(&b.path));
        discovered.dedup_by(|a, b| a.path == b.path);
        // A 64 MiB cache (default 2 MiB) holds the B-tree interiors that every
        // insert descends, which cut ingest reads 4x and writes a further 2.5x.
        c.execute_batch("PRAGMA cache_size=-65536; PRAGMA temp_store=FILE;")?;
        let changes = diff_files(c, &discovered)?;
        if !changes.is_empty() {
            // A failed discovery returns above without deleting any data. Replaced
            // or shortened files require replay of their run in canonical file order.
            let tx = c.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
            for run in &changes.new_runs {
                tx.prepare_cached("INSERT OR IGNORE INTO runs(name) VALUES(?1)")?
                    .execute(params![run])?;
            }
            for run in &changes.reset_runs {
                tx.prepare_cached("DELETE FROM series WHERE run=?1")?
                    .execute(params![run])?;
                tx.prepare_cached("UPDATE runs SET start=NULL WHERE name=?1")?
                    .execute(params![run])?;
                tx.prepare_cached("UPDATE files SET offset=0,dead=0 WHERE run=?1")?
                    .execute(params![run])?;
            }
            for &i in &changes.new_files {
                let d = &discovered[i];
                let (identity, modified, length) = fingerprint_columns(&d.fingerprint);
                tx.prepare_cached(
                    "INSERT INTO files(path,run,identity,modified,length,present) VALUES(?1,?2,?3,?4,?5,1)",
                )?
                .execute(params![d.path, d.run, identity, modified, length])?;
            }
            for &i in &changes.reappeared {
                tx.prepare_cached("UPDATE files SET present=1 WHERE path=?1")?
                    .execute(params![discovered[i].path])?;
            }
            // Keep missing-file tombstones while their run exists, matching
            // RunLoader: removed files must not be re-opened on a later reload.
            for (path, run) in &changes.missing {
                if changes.reset_runs.contains(run) || !changes.tombstoned.contains(path) {
                    tx.prepare_cached("UPDATE files SET dead=1,present=0 WHERE path=?1")?
                        .execute(params![path])?;
                }
            }
            for run in &changes.removed_runs {
                tx.prepare_cached("DELETE FROM runs WHERE name=?1")?
                    .execute(params![run])?;
            }
            tx.commit()?;
        }
        // Ingest statements fire triggers and cascades, so SQLite gives each a
        // statement journal; on disk those wrote ~40x the index size. Each one
        // touches a few pages, so they stay in memory, unlike the run-wide
        // deletes above, whose journals may spill to disk.
        c.execute_batch("PRAGMA temp_store=MEMORY;")?;
        for i in changes.pending {
            let d = &discovered[i];
            let path = &d.path;
            let run = &d.run;
            let (identity, modified, length) = fingerprint_columns(&d.fingerprint);
            let row = c
                .prepare_cached(
                    "SELECT offset,dead,identity,modified,length FROM files WHERE path=?1",
                )?
                .query_row(params![path], |r| {
                    Ok((
                        r.get::<_, i64>(0)?,
                        r.get::<_, bool>(1)?,
                        r.get::<_, Option<String>>(2)?,
                        r.get::<_, Option<String>>(3)?,
                        r.get::<_, Option<i64>>(4)?,
                    ))
                })
                .optional()?;
            let Some((offset, dead, old_identity, old_modified, old_length)) = row else {
                continue;
            };
            if dead
                || is_current(
                    offset,
                    &d.fingerprint,
                    &old_identity,
                    &old_modified,
                    old_length,
                )
            {
                continue;
            }
            let file = EventFileBuf(decode_path(&path)?);
            let source = match logdir.open_at(&file, offset as u64) {
                Ok(x) => x,
                Err(e) => {
                    log::warn!("Cannot open indexed event file {:?}: {}", file, e);
                    return Err(e.into());
                }
            };
            let mut reader = crate::tf_record::TfRecordReader::new(source);
            let mut offset = offset;
            let mut ingest = Ingest::new(run);
            // Publish at the existing loader cadence instead of committing a
            // disk transaction for every scalar event. Offsets and samples stay
            // atomic; readers keep seeing the previous WAL snapshot meanwhile.
            let mut tx = c.unchecked_transaction()?;
            let mut last_commit = Instant::now();
            loop {
                let record = match reader.read_record() {
                    Ok(x) => x,
                    Err(crate::tf_record::ReadRecordError::Truncated) => break,
                    Err(crate::tf_record::ReadRecordError::Io(e)) => {
                        checkpoint(&tx, &mut ingest, path, offset)?;
                        tx.commit()?;
                        return Err(e.into());
                    }
                    Err(e) => {
                        log::warn!("Event read error {:?}: {}", file, e);
                        tx.execute("UPDATE files SET dead=1 WHERE path=?1", params![path])?;
                        break;
                    }
                };
                let event = (|| -> Result<pb::Event> {
                    if checksum {
                        record.checksum()?;
                    }
                    let event = match pb::Event::decode(record.data.as_slice()) {
                        Ok(e) => e,
                        Err(e) => {
                            record.checksum()?;
                            return Err(e.into());
                        }
                    };
                    if event.wall_time.is_nan() {
                        return Err("NaN event wall time".into());
                    }
                    Ok(event)
                })();
                let event = match event {
                    Ok(e) => e,
                    Err(e) => {
                        log::warn!("Invalid event {:?}: {}", file, e);
                        tx.execute("UPDATE files SET dead=1 WHERE path=?1", params![path])?;
                        break;
                    }
                };
                offset += record.data.len() as i64 + 16;
                ingest.event(&tx, event, hints)?;
                if last_commit.elapsed() >= crate::run::COMMIT_INTERVAL {
                    checkpoint(&tx, &mut ingest, path, offset)?;
                    tx.commit()?;
                    tx = c.unchecked_transaction()?;
                    last_commit = Instant::now();
                }
            }
            checkpoint(&tx, &mut ingest, path, offset)?;
            tx.commit()?;
            // Remote logdirs without fingerprints re-read every cycle; skip the
            // write when nothing about the file changed.
            c.prepare_cached(
                "UPDATE files SET identity=?2,modified=?3,length=?4 WHERE path=?1
                    AND (identity IS NOT ?2 OR modified IS NOT ?3 OR length IS NOT ?4)",
            )?
            .execute(params![path, identity, modified, length])?;
        }
        Ok(())
    }
}

/// Persists loader state and the read position together, before a commit.
fn checkpoint(c: &Connection, ingest: &mut Ingest, path: &[u8], offset: i64) -> Result<()> {
    ingest.flush(c)?;
    c.prepare_cached("UPDATE files SET offset=?2 WHERE path=?1 AND offset IS NOT ?2")?
        .execute(params![path, offset])?;
    Ok(())
}

/// Whether `e`, or any error it wraps, reports a damaged index file.
pub fn is_corruption(e: &(dyn std::error::Error + 'static)) -> bool {
    let mut next = Some(e);
    while let Some(e) = next {
        if let Some(rusqlite::Error::SqliteFailure(failure, _)) = e.downcast_ref() {
            if matches!(
                failure.code,
                rusqlite::ErrorCode::DatabaseCorrupt | rusqlite::ErrorCode::NotADatabase
            ) {
                return true;
            }
        }
        next = e.source();
    }
    false
}

struct Discovered {
    path: Vec<u8>,
    run: String,
    fingerprint: Option<FileFingerprint>,
}

/// Index writes needed to match one discovery pass. Every field is
/// proportional to what changed, except `pending`, which lists only files
/// whose content may have changed.
#[derive(Default)]
struct FileChanges {
    new_runs: Vec<String>,
    removed_runs: Vec<String>,
    reset_runs: HashSet<String>,
    /// Indices into the discovery list.
    new_files: Vec<usize>,
    reappeared: Vec<usize>,
    /// Indexed but undiscovered files, with their runs.
    missing: Vec<(Vec<u8>, String)>,
    /// Missing files that are already `dead=1,present=0`.
    tombstoned: HashSet<Vec<u8>>,
    /// Discovered files to read, ascending by path.
    pending: Vec<usize>,
}

impl FileChanges {
    fn is_empty(&self) -> bool {
        self.new_runs.is_empty()
            && self.removed_runs.is_empty()
            && self.reset_runs.is_empty()
            && self.new_files.is_empty()
            && self.reappeared.is_empty()
            && self.missing.len() == self.tombstoned.len()
    }
}

fn fingerprint_columns(
    fingerprint: &Option<FileFingerprint>,
) -> (Option<&str>, Option<&str>, Option<i64>) {
    match fingerprint {
        Some(f) => (
            Some(f.identity.as_str()),
            Some(f.modified.as_str()),
            Some(f.len as i64),
        ),
        None => (None, None, None),
    }
}

/// Whether an indexed file was fully read at exactly this fingerprint.
fn is_current(
    offset: i64,
    fingerprint: &Option<FileFingerprint>,
    identity: &Option<String>,
    modified: &Option<String>,
    length: Option<i64>,
) -> bool {
    let (new_identity, new_modified, new_length) = fingerprint_columns(fingerprint);
    offset != 0
        && new_identity.is_some()
        && new_identity == identity.as_deref()
        && new_modified == modified.as_deref()
        && new_length == length
}

/// Merges sorted discovery against the indexed files in one ordered scan.
fn diff_files(c: &Connection, discovered: &[Discovered]) -> Result<FileChanges> {
    let mut changes = FileChanges::default();
    let mut indexed_runs = HashSet::new();
    {
        let mut runs = c.prepare("SELECT name FROM runs")?;
        let mut rows = runs.query([])?;
        while let Some(row) = rows.next()? {
            indexed_runs.insert(row.get::<_, String>(0)?);
        }
    }
    let mut discovered_runs = HashSet::new();
    for d in discovered {
        if discovered_runs.insert(d.run.as_str()) && !indexed_runs.contains(&d.run) {
            changes.new_runs.push(d.run.clone());
        }
    }
    changes.removed_runs = indexed_runs
        .into_iter()
        .filter(|run| !discovered_runs.contains(run.as_str()))
        .collect();

    let mut files = c.prepare(
        "SELECT path,run,offset,dead,present,identity,modified,length FROM files ORDER BY path",
    )?;
    let mut rows = files.query([])?;
    let mut next = 0;
    while let Some(row) = rows.next()? {
        let path: Vec<u8> = row.get(0)?;
        while next < discovered.len() && discovered[next].path < path {
            changes.new_files.push(next);
            changes.pending.push(next);
            next += 1;
        }
        if next == discovered.len() || discovered[next].path != path {
            let run: String = row.get(1)?;
            if row.get::<_, bool>(3)? && !row.get::<_, bool>(4)? {
                changes.tombstoned.insert(path.clone());
            }
            changes.missing.push((path, run));
            continue;
        }
        let d = &discovered[next];
        let offset: i64 = row.get(2)?;
        let dead: bool = row.get(3)?;
        let identity: Option<String> = row.get(5)?;
        let modified: Option<String> = row.get(6)?;
        let length: Option<i64> = row.get(7)?;
        let (new_identity, new_modified, new_length) = fingerprint_columns(&d.fingerprint);
        let replaced = identity.is_some() && new_identity != identity.as_deref();
        let shortened = new_length.map_or(false, |n| n < offset);
        let rewritten = new_modified != modified.as_deref()
            && matches!((new_length, length), (Some(n), Some(old)) if n <= old);
        if replaced || shortened || rewritten {
            let run: String = row.get(1)?;
            changes.reset_runs.insert(run);
        }
        if !row.get::<_, bool>(4)? {
            changes.reappeared.push(next);
        }
        if !dead && !is_current(offset, &d.fingerprint, &identity, &modified, length) {
            changes.pending.push(next);
        }
        next += 1;
    }
    for i in next..discovered.len() {
        changes.new_files.push(i);
        changes.pending.push(i);
    }
    if !changes.reset_runs.is_empty() {
        // Replay every discovered file of a reset run, in path order.
        let reset_runs = &changes.reset_runs;
        changes
            .pending
            .extend((0..discovered.len()).filter(|&i| reset_runs.contains(&discovered[i].run)));
        changes.pending.sort_unstable();
        changes.pending.dedup();
    }
    Ok(changes)
}
#[cfg(unix)]
fn encode_path(path: &Path) -> Vec<u8> {
    use std::os::unix::ffi::OsStrExt;
    path.as_os_str().as_bytes().to_vec()
}
#[cfg(unix)]
fn decode_path(path: &[u8]) -> Result<PathBuf> {
    use std::os::unix::ffi::OsStrExt;
    Ok(std::ffi::OsStr::from_bytes(path).into())
}
#[cfg(not(unix))]
fn encode_path(path: &Path) -> Vec<u8> {
    path.to_string_lossy().as_bytes().to_vec()
}
#[cfg(not(unix))]
fn decode_path(path: &[u8]) -> Result<PathBuf> {
    Ok(std::str::from_utf8(path)?.into())
}

/// Loader state for the events of one run within one transaction.
///
/// Each event used to re-read its `series` row (decoding the metadata proto),
/// re-write it with three freshly recomputed extrema, and re-write the run's
/// start time. Keeping that state here and writing each touched row once per
/// commit leaves a typical scalar event one INSERT, plus one DELETE once its
/// reservoir is full. [`Ingest::flush`] must run before every commit.
struct Ingest<'a> {
    run: &'a str,
    series: HashMap<String, SeriesState>,
    /// Earliest event wall time of `run`, once read (`Some(None)`: unset).
    start: Option<Option<f64>>,
    start_dirty: bool,
}

struct SeriesState {
    id: i64,
    metadata: pb::SummaryMetadata,
    capacity: i64,
    seen: i64,
    rng: ChaCha20Rng,
    len: i64,
    valid: i64,
    /// Greatest stored step, valid or not. Only an event at or below it can
    /// preempt samples, so later events skip the preemption queries.
    max_step: Option<i64>,
    /// The `series` extrema over valid samples.
    valid_step: Max<i64>,
    valid_wall: Max<f64>,
    valid_length: Max<i64>,
    dirty: bool,
}

impl SeriesState {
    fn remove_valid(&mut self, step: i64, wall: f64, length: Option<i64>) {
        self.valid -= 1;
        self.valid_step.remove(step);
        self.valid_wall.remove(wall);
        if let Some(length) = length {
            self.valid_length.remove(length);
        }
    }
}

/// The greatest of a changing set of values. Removing the maximum leaves an
/// upper bound, which a value at least as large makes exact again; only a
/// bound still inexact at flush needs a scan of the stored samples.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Max<T> {
    value: Option<T>,
    exact: bool,
}

impl<T: PartialOrd + Copy> Max<T> {
    fn exact(value: Option<T>) -> Self {
        Self { value, exact: true }
    }

    fn insert(&mut self, x: T) {
        if self.value.map_or(true, |v| x >= v) {
            *self = Self::exact(Some(x));
        }
    }

    fn remove(&mut self, x: T) {
        if self.value.map_or(false, |v| x >= v) {
            self.exact = false;
        }
    }
}

impl<'a> Ingest<'a> {
    fn new(run: &'a str) -> Self {
        Self {
            run,
            series: HashMap::new(),
            start: None,
            start_dirty: false,
        }
    }

    fn event(
        &mut self,
        c: &Connection,
        event: pb::Event,
        hints: &PluginSamplingHint,
    ) -> Result<()> {
        if WallTime::new(event.wall_time).is_none() {
            return Ok(());
        }
        let step = event.step;
        let wall = event.wall_time;
        self.observe_wall(c, wall)?;
        match event.what {
            Some(pb::event::What::GraphDef(bytes)) => self.offer(
                c,
                GraphDefValue::TAG_NAME,
                step,
                wall,
                EventValue::GraphDef(GraphDefValue(bytes)),
                None,
                hints,
            )?,
            Some(pb::event::What::TaggedRunMetadata(v)) => self.offer(
                c,
                &v.tag,
                step,
                wall,
                EventValue::TaggedRunMetadata(TaggedRunMetadataValue(v.run_metadata)),
                None,
                hints,
            )?,
            Some(pb::event::What::Summary(summary)) => {
                for v in summary.value {
                    let payload = v.value.unwrap_or_else(|| {
                        pb::summary::value::Value::Tensor(crate::run::null_tensor_proto())
                    });
                    self.offer(
                        c,
                        &v.tag,
                        step,
                        wall,
                        EventValue::Summary(SummaryValue(Box::new(payload))),
                        v.metadata,
                        hints,
                    )?;
                }
            }
            _ => (),
        }
        Ok(())
    }

    fn observe_wall(&mut self, c: &Connection, wall: f64) -> Result<()> {
        let start = match self.start {
            Some(start) => start,
            None => c
                .prepare_cached("SELECT start FROM runs WHERE name=?1")?
                .query_row(params![self.run], |r| r.get::<_, Option<f64>>(0))
                .optional()?
                .flatten(),
        };
        if start.map_or(true, |start| start > wall) {
            self.start = Some(Some(wall));
            self.start_dirty = true;
        } else {
            self.start = Some(start);
        }
        Ok(())
    }

    fn offer(
        &mut self,
        c: &Connection,
        tag: &str,
        step: i64,
        wall: f64,
        value: EventValue,
        initial: Option<pb::SummaryMetadata>,
        hints: &PluginSamplingHint,
    ) -> Result<()> {
        if !self.series.contains_key(tag) {
            let state = self.load(c, tag, &value, initial, hints)?;
            self.series.insert(tag.to_owned(), state);
        }
        let s = self.series.get_mut(tag).expect("series just loaded");
        if s.capacity == 0 {
            return Ok(());
        }
        if s.max_step.map_or(false, |max| step <= max) {
            let mut preempted = 0;
            // Read the doomed rows first: `RETURNING value IS NOT NULL` reports
            // NULL values as non-NULL in bundled SQLite 3.39 when the delete
            // cascades to `blobs`. Validity is tested with typeof(), which
            // SQLite answers from the record header; 3.39's IS NOT NULL loads
            // the whole value, overflow pages included. `length` is NULL (also
            // header-only) except for blob sequences, whose `value` is empty.
            let mut select = c.prepare_cached(
                "SELECT typeof(value)<>'null',step,wall,length FROM samples WHERE series=?1 AND step>=?2",
            )?;
            let mut rows = select.query(params![s.id, step])?;
            while let Some(row) = rows.next()? {
                preempted += 1;
                if row.get(0)? {
                    s.remove_valid(row.get(1)?, row.get(2)?, row.get(3)?);
                }
            }
            // A resumed run can have an arbitrarily long suffix. Bound the
            // number of samples journaled per statement, retaining one
            // transaction so readers see the deletion and replay together.
            drop(rows);
            drop(select);
            const DELETE_BATCH_SIZE: usize = 256;
            let mut delete = c.prepare_cached(
                "DELETE FROM samples WHERE rowid IN (
                    SELECT rowid FROM samples WHERE series=?1 AND step>=?2 LIMIT ?3
                )",
            )?;
            while delete.execute(params![s.id, step, DELETE_BATCH_SIZE])? == DELETE_BATCH_SIZE {}
            let old_len = s.len;
            s.len -= preempted;
            if preempted != 0 {
                s.seen = ((s.seen as u64 * s.len as u64) / old_len as u64) as i64;
            }
        }
        s.seen += 1;
        if s.capacity >= 0 && s.seen > s.capacity {
            let destination = s.rng.destination(s.seen as usize) as i64;
            let evict = if destination >= s.capacity {
                Some(s.len - 1)
            } else if s.len >= s.capacity {
                Some(destination)
            } else {
                None
            };
            if let Some(index) = evict.filter(|&index| index >= 0) {
                // Count from the nearer end; the usual eviction is the latest
                // sample. Skipped rows touch only the (series,step) index; the
                // table is read for the chosen row alone (deferred seek), and
                // the same RETURNING bug rules out a single DELETE.
                let (sql, offset) = if index >= s.len / 2 {
                    (
                        "SELECT rowid,typeof(value)<>'null',step,wall,length FROM samples
                            WHERE series=?1 ORDER BY step DESC LIMIT 1 OFFSET ?2",
                        s.len - 1 - index,
                    )
                } else {
                    (
                        "SELECT rowid,typeof(value)<>'null',step,wall,length FROM samples
                            WHERE series=?1 ORDER BY step LIMIT 1 OFFSET ?2",
                        index,
                    )
                };
                let (rowid, valid, step, wall, length): (i64, bool, i64, f64, Option<i64>) = c
                    .prepare_cached(sql)?
                    .query_row(params![s.id, offset], |r| {
                        Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?))
                    })?;
                c.prepare_cached("DELETE FROM samples WHERE rowid=?1")?
                    .execute(params![rowid])?;
                if valid {
                    s.remove_valid(step, wall, length);
                }
                s.len -= 1;
            }
        }
        let (payload, blobs, length): (Option<Vec<u8>>, Option<BlobSequenceValue>, Option<i64>) =
            match pb::DataClass::from_i32(s.metadata.data_class) {
                Some(pb::DataClass::Scalar) => (
                    value.into_scalar().ok().map(|v| v.0.to_le_bytes().to_vec()),
                    None,
                    None,
                ),
                Some(pb::DataClass::Tensor) => (
                    value
                        .into_tensor(&s.metadata)
                        .ok()
                        .map(|v| v.encode_to_vec()),
                    None,
                    None,
                ),
                Some(pb::DataClass::BlobSequence) => {
                    let blobs = value.into_blob_sequence(&s.metadata).ok();
                    let length = blobs.as_ref().map(|v| v.0.len() as i64);
                    (length.map(|_| Vec::new()), blobs, length)
                }
                _ => (None, None, None),
            };
        if payload.is_some() {
            s.valid += 1;
            s.valid_step.insert(step);
            s.valid_wall.insert(wall);
            if let Some(length) = length {
                s.valid_length.insert(length);
            }
        }
        c.prepare_cached(
            "INSERT INTO samples(series,step,wall,value,length) VALUES(?1,?2,?3,?4,?5)",
        )?
        .execute(params![s.id, step, wall, payload, length])?;
        if let Some(blobs) = blobs {
            for (index, blob) in blobs.0.into_iter().enumerate() {
                c.prepare_cached("INSERT INTO blobs VALUES(?1,?2,?3,?4)")?
                    .execute(params![s.id, step, index as i64, blob.as_ref()])?;
            }
        }
        // Preemption removed every sample at or after `step`.
        s.len += 1;
        s.max_step = Some(step);
        s.dirty = true;
        Ok(())
    }

    fn load(
        &self,
        c: &Connection,
        tag: &str,
        value: &EventValue,
        initial: Option<pb::SummaryMetadata>,
        hints: &PluginSamplingHint,
    ) -> Result<SeriesState> {
        let existing = c
            .prepare_cached(
                "SELECT id,metadata,capacity,seen,rng,sample_count,valid,max_step,max_wall,max_length
                    FROM series WHERE run=?1 AND tag=?2",
            )?
            .query_row(params![self.run, tag], |r| {
                Ok((
                    (
                        r.get::<_, i64>(0)?,
                        r.get::<_, Vec<u8>>(1)?,
                        r.get::<_, i64>(2)?,
                        r.get::<_, i64>(3)?,
                        r.get::<_, String>(4)?,
                        r.get::<_, i64>(5)?,
                        r.get::<_, i64>(6)?,
                    ),
                    (r.get(7)?, r.get(8)?, r.get(9)?),
                ))
            })
            .optional()?;
        let mut rng = ChaCha20Rng::seed_from_u64(0);
        Ok(match existing {
            Some((
                (id, metadata, capacity, seen, position, len, valid),
                (valid_step, valid_wall, valid_length),
            )) => {
                rng.set_word_pos(position.parse()?);
                let max_step = c
                    .prepare_cached("SELECT max(step) FROM samples WHERE series=?1")?
                    .query_row(params![id], |r| r.get(0))?;
                SeriesState {
                    id,
                    metadata: pb::SummaryMetadata::decode(metadata.as_slice())?,
                    capacity,
                    seen,
                    rng,
                    len,
                    valid,
                    max_step,
                    valid_step: Max::exact(valid_step),
                    valid_wall: Max::exact(valid_wall),
                    valid_length: Max::exact(valid_length),
                    dirty: false,
                }
            }
            None => {
                let metadata = match value {
                    EventValue::GraphDef(_) => GraphDefValue::initial_metadata(),
                    EventValue::TaggedRunMetadata(_) => TaggedRunMetadataValue::initial_metadata(),
                    EventValue::Summary(v) => v.initial_metadata(initial),
                };
                let capacity = match crate::run::StageTimeSeries::capacity(&metadata, hints) {
                    Capacity::Unbounded => -1,
                    Capacity::Bounded(n) => n.try_into()?,
                };
                let plugin = metadata
                    .plugin_data
                    .as_ref()
                    .map(|p| p.plugin_name.as_str())
                    .unwrap_or("");
                c.prepare_cached(
                    "INSERT INTO series(run,tag,plugin,class,metadata,capacity) VALUES(?1,?2,?3,?4,?5,?6)",
                )?
                .execute(params![
                    self.run,
                    tag,
                    plugin,
                    metadata.data_class,
                    metadata.encode_to_vec(),
                    capacity
                ])?;
                SeriesState {
                    id: c.last_insert_rowid(),
                    metadata: *metadata,
                    capacity,
                    seen: 0,
                    rng,
                    len: 0,
                    valid: 0,
                    max_step: None,
                    valid_step: Max::exact(None),
                    valid_wall: Max::exact(None),
                    valid_length: Max::exact(None),
                    dirty: false,
                }
            }
        })
    }

    /// Writes the state of every series and the run start changed since the
    /// last flush. Triggers on `series` then fire once per series per commit.
    fn flush(&mut self, c: &Connection) -> Result<()> {
        for s in self.series.values_mut().filter(|s| s.dirty) {
            if !(s.valid_step.exact && s.valid_wall.exact && s.valid_length.exact) {
                let (step, wall, length) = c
                    .prepare_cached(
                        "SELECT max(step),max(wall),max(length) FROM samples
                            WHERE series=?1 AND typeof(value)<>'null'",
                    )?
                    .query_row(params![s.id], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?;
                s.valid_step = Max::exact(step);
                s.valid_wall = Max::exact(wall);
                s.valid_length = Max::exact(length);
            }
            c.prepare_cached(
                "UPDATE series SET seen=?2,rng=?3,valid=?4,sample_count=?5,
                    max_step=?6,max_wall=?7,max_length=?8 WHERE id=?1",
            )?
            .execute(params![
                s.id,
                s.seen,
                s.rng.get_word_pos().to_string(),
                s.valid,
                s.len,
                s.valid_step.value,
                s.valid_wall.value,
                s.valid_length.value,
            ])?;
            s.dirty = false;
        }
        if let (true, Some(Some(start))) = (self.start_dirty, self.start) {
            c.prepare_cached(
                "UPDATE runs SET start=?2 WHERE name=?1 AND (start IS NULL OR start>?2)",
            )?
            .execute(params![self.run, start])?;
            self.start_dirty = false;
        }
        Ok(())
    }
}

/// Indexes one event in its own [`Ingest`].
#[cfg(test)]
fn index_event(
    c: &Connection,
    run: &str,
    event: pb::Event,
    hints: &PluginSamplingHint,
) -> Result<()> {
    let mut ingest = Ingest::new(run);
    ingest.event(c, event, hints)?;
    ingest.flush(c)
}

pub(crate) struct Listing {
    pub total: u64,
    pub rows: Vec<SeriesMetadata>,
}

pub(crate) struct SeriesMetadata {
    pub run: String,
    pub tag: String,
    pub metadata: pb::SummaryMetadata,
    pub max_step: i64,
    pub max_wall: f64,
    pub max_length: i64,
}

pub(crate) fn page_number(n: u64) -> std::result::Result<i64, Status> {
    n.try_into()
        .map_err(|_| Status::invalid_argument("pagination value exceeds signed 64-bit range"))
}

/// Request-local byte trie: lookup visits each name byte at most once, without
/// scanning session prefixes or allocating a run name. UTF-8 prefixes remain
/// ordinary string prefixes, not path-component matches.
pub(crate) struct SessionRanks {
    ranks: Vec<Option<i64>>,
    edges: std::collections::HashMap<(usize, u8), usize>,
    default_rank: i64,
}

impl SessionRanks {
    pub(crate) fn new(entries: &[data::RunSessionRank], default_rank: i64) -> Self {
        let mut trie = Self {
            ranks: vec![None],
            edges: std::collections::HashMap::new(),
            default_rank,
        };
        for entry in entries {
            let mut node = 0;
            for byte in entry.prefix.bytes() {
                node = match trie.edges.get(&(node, byte)) {
                    Some(&next) => next,
                    None => {
                        let next = trie.ranks.len();
                        trie.ranks.push(None);
                        trie.edges.insert((node, byte), next);
                        next
                    }
                };
            }
            // Repeated prefixes have the same last-value-wins semantics as
            // the frontend's session map.
            trie.ranks[node] = Some(entry.rank);
        }
        trie
    }

    pub(crate) fn rank(&self, name: &str) -> i64 {
        let mut node = 0;
        let mut rank = self.ranks[0].unwrap_or(self.default_rank);
        for byte in name.bytes() {
            node = match self.edges.get(&(node, byte)) {
                Some(&next) => next,
                None => break,
            };
            if let Some(value) = self.ranks[node] {
                rank = value;
            }
        }
        rank
    }
}

pub(crate) fn matches_query(regex: &regex::Regex, prefix: &str, name: &str) -> bool {
    regex.is_match(name)
        || (!prefix.is_empty()
            && (regex.is_match(prefix) || regex.is_match(&format!("{}/{}", prefix, name))))
}

fn register_query(c: &Connection, query: &str, prefix: &str) -> std::result::Result<(), Status> {
    let regex = regex::Regex::new(query)
        .map_err(|e| Status::invalid_argument(format!("invalid query regex: {}", e)))?;
    let prefix = prefix.to_owned();
    c.create_scalar_function(
        "matches_query",
        1,
        rusqlite::functions::FunctionFlags::SQLITE_DETERMINISTIC,
        move |ctx| {
            let name = ctx.get::<String>(0)?;
            Ok(matches_query(&regex, &prefix, &name))
        },
    )
    .map_err(status)
}

fn names_table(c: &Connection, table: &str, names: &[String]) -> rusqlite::Result<()> {
    // `table` is a static internal identifier, never a request field.
    c.execute_batch(&format!(
        "CREATE TEMP TABLE {}(name TEXT PRIMARY KEY) WITHOUT ROWID;",
        table
    ))?;
    let mut insert = c.prepare(&format!("INSERT OR IGNORE INTO {} VALUES(?1)", table))?;
    for name in names {
        insert.execute(params![name])?;
    }
    Ok(())
}
// All three bundled metric plugin protos use int32 version=1. Decode that
// common header while ignoring other fields, just like their full parsers.
#[derive(Clone, PartialEq, Message)]
struct MetricPluginVersion {
    #[prost(int32, tag = "1")]
    version: i32,
}

fn supported_metrics_metadata(metadata: &pb::SummaryMetadata) -> bool {
    let plugin = match &metadata.plugin_data {
        Some(plugin) => plugin,
        None => return false,
    };
    if !matches!(
        plugin.plugin_name.as_str(),
        "scalars" | "histograms" | "images"
    ) {
        return false;
    }
    if plugin.plugin_name == "histograms" && plugin.content.as_ref() == b"{}" {
        return true;
    }
    MetricPluginVersion::decode(plugin.content.as_ref())
        .map_or(false, |version| version.version == 0)
}

fn create_metrics_catalog(c: &Connection) -> rusqlite::Result<()> {
    // Covering, metadata-free indexes keep both unselected runs and closed
    // categories out of member queries. Group totals are maintained at ingest.
    c.execute_batch("
        CREATE TABLE metrics_catalog(id INTEGER PRIMARY KEY,run TEXT NOT NULL,
            tag TEXT NOT NULL,plugin TEXT NOT NULL,category TEXT NOT NULL,cards INTEGER NOT NULL);
        CREATE INDEX metrics_catalog_scope ON metrics_catalog(run,plugin,category,tag COLLATE TAG_NAME_V2,cards);
        CREATE UNIQUE INDEX metrics_catalog_tag ON metrics_catalog(run,tag,plugin);
        CREATE TABLE metrics_catalog_groups(run TEXT NOT NULL,plugin TEXT NOT NULL,
            category TEXT NOT NULL,cards INTEGER NOT NULL,
            PRIMARY KEY(run,plugin,category)) WITHOUT ROWID;
    ")
}

/// One comparison unit of a tag name; see [`compare_tag_names`].
#[derive(Clone, Copy, Debug)]
enum TagToken {
    Slash,
    Number(f64),
    Unit(u16),
}

impl TagToken {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Self::Number(a), Self::Number(b)) => a.total_cmp(b),
            (Self::Unit(a), Self::Unit(b)) => a.cmp(b),
            _ => self.rank().cmp(&other.rank()),
        }
    }

    fn rank(&self) -> u8 {
        match self {
            Self::Slash => 0,
            Self::Number(_) => 1,
            Self::Unit(_) => 2,
        }
    }
}

/// Splits a tag into a fixed token sequence. Tokenization depends only on the
/// tag itself, never on the tag it is compared against.
struct TagTokens<'a> {
    rest: &'a str,
    low_surrogate: Option<u16>,
}

impl Iterator for TagTokens<'_> {
    type Item = TagToken;

    fn next(&mut self) -> Option<TagToken> {
        if let Some(unit) = self.low_surrogate.take() {
            return Some(TagToken::Unit(unit));
        }
        let bytes = self.rest.as_bytes();
        let first = *bytes.first()?;
        if first == b'/' {
            self.rest = &self.rest[1..];
            return Some(TagToken::Slash);
        }
        if first.is_ascii_digit() {
            let end = number_end(bytes);
            let (number, rest) = self.rest.split_at(end);
            self.rest = rest;
            // The lexer only accepts complete decimal literals, so parsing
            // cannot fail; `total_cmp` keeps the order total regardless.
            return Some(TagToken::Number(number.parse().unwrap_or(f64::NAN)));
        }
        let c = self.rest.chars().next()?;
        self.rest = &self.rest[c.len_utf8()..];
        let mut units = [0; 2];
        let units = c.encode_utf16(&mut units);
        self.low_surrogate = units.get(1).copied();
        Some(TagToken::Unit(units[0]))
    }
}

/// Length of the decimal literal at the start of `s`: `digits [. digits*]
/// [(e|E) [+|-] digits+]`. A dangling exponent marker is not consumed.
fn number_end(s: &[u8]) -> usize {
    let digits = |from: usize| from + s[from..].iter().take_while(|b| b.is_ascii_digit()).count();
    let mut end = digits(0);
    if s.get(end) == Some(&b'.') {
        end = digits(end + 1);
    }
    if matches!(s.get(end), Some(b'e' | b'E')) {
        let sign = usize::from(matches!(s.get(end + 1), Some(b'+' | b'-')));
        let exponent = end + 1 + sign;
        if s.get(exponent).map_or(false, u8::is_ascii_digit) {
            end = digits(exponent);
        }
    }
    end
}

/// Frontend compareTagNames: numeric chunks (including decimals/exponents),
/// slash components before numbers before ordinary characters, and UTF-16
/// character order. Lexical ties make equivalent numeric spellings
/// deterministic.
///
/// This collates a persisted SQLite index, so it must be a strict total order:
/// an inconsistent comparator corrupts the B-tree ("database disk image is
/// malformed"). It is a lexicographic comparison of per-tag token sequences
/// under a total token order, broken by byte order, which is total by
/// construction. See `COLLATION_VERSION` for schema compatibility.
pub(crate) fn compare_tag_names(a: &str, b: &str) -> Ordering {
    let tokens = |rest| TagTokens {
        rest,
        low_surrogate: None,
    };
    let (mut a_tokens, mut b_tokens) = (tokens(a), tokens(b));
    loop {
        match (a_tokens.next(), b_tokens.next()) {
            (None, None) => return a.cmp(b),
            (None, Some(_)) => return Ordering::Less,
            (Some(_), None) => return Ordering::Greater,
            (Some(x), Some(y)) => match x.cmp(&y) {
                Ordering::Equal => (),
                order => return order,
            },
        }
    }
}

// Only used to read old indexes until the background migration replaces them.
fn compare_legacy_tag_names(a: &str, b: &str) -> Ordering {
    fn number_end(s: &str, mut i: usize) -> usize {
        let mut state = 0;
        while let Some(&c) = s.as_bytes().get(i) {
            match (state, c) {
                (0, b'.') => state = 1,
                (0 | 1, b'e' | b'E') => state = 2,
                (2, b'+' | b'-' | b'0'..=b'9') => state = 3,
                (_, b'0'..=b'9') => (),
                _ => break,
            }
            i += 1;
        }
        i
    }
    let (mut ai, mut bi) = (0, 0);
    while ai < a.len() && bi < b.len() {
        let ac = a[ai..].chars().next().unwrap();
        let bc = b[bi..].chars().next().unwrap();
        if ac.is_ascii_digit() && bc.is_ascii_digit() {
            let ae = number_end(a, ai + 1);
            let be = number_end(b, bi + 1);
            let an = a[ai..ae].parse::<f64>().unwrap_or(f64::NAN);
            let bn = b[bi..be].parse::<f64>().unwrap_or(f64::NAN);
            if an < bn {
                return Ordering::Less;
            }
            if an > bn {
                return Ordering::Greater;
            }
            ai = ae;
            bi = be;
            continue;
        }
        let ab = ac == '/' || ac.is_ascii_digit();
        let bb = bc == '/' || bc.is_ascii_digit();
        let order = match (ab, bb) {
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            (true, true) => Ordering::Equal,
            (false, false) => {
                let (mut au, mut bu) = ([0; 2], [0; 2]);
                ac.encode_utf16(&mut au)
                    .iter()
                    .cmp(bc.encode_utf16(&mut bu).iter())
            }
        };
        if order != Ordering::Equal {
            return order;
        }
        ai += ac.len_utf8();
        bi += bc.len_utf8();
    }
    (a.len() - ai).cmp(&(b.len() - bi)).then_with(|| a.cmp(b))
}

fn catalog_runs_table(
    c: &Connection,
    table: &str,
    runs: impl IntoIterator<Item = impl Borrow<data::MetricsCatalogRun>>,
) -> rusqlite::Result<()> {
    c.execute_batch(&format!(
        "CREATE TEMP TABLE {}(experiment TEXT NOT NULL,name TEXT NOT NULL,run_id TEXT NOT NULL,
            PRIMARY KEY(experiment,name)) WITHOUT ROWID;",
        table
    ))?;
    let mut insert = c.prepare(&format!("INSERT OR IGNORE INTO {} VALUES(?1,?2,?3)", table))?;
    for run in runs {
        let run = run.borrow();
        insert.execute(params![
            run.experiment_id,
            run.name,
            format!("{}/{}", run.experiment_id, run.name)
        ])?;
    }
    Ok(())
}

/// One row per scalar tag or histogram/image run-tag span. Images are weighted
/// spans, never a recursive SQL sample expansion or an in-memory sample array.
fn catalog_spans(group: bool, filtered: bool) -> String {
    format!(
        "SELECT m.plugin,m.tag,CASE m.plugin WHEN 'scalars' THEN '' ELSE r.run_id END AS run_id,
            max(m.cards) AS cards,p.rank
         FROM requested_runs r CROSS JOIN catalog_plugins p
         CROSS JOIN metrics_catalog m INDEXED BY metrics_catalog_scope
         WHERE m.run=r.name AND m.plugin=p.name AND m.cards>0 {} {}
         GROUP BY m.plugin,m.tag,CASE m.plugin WHEN 'scalars' THEN '' ELSE r.run_id END",
        if group { "AND m.category=?1" } else { "" },
        if filtered {
            "AND matches_query(m.tag)"
        } else {
            ""
        },
    )
}

fn catalog_window(
    c: &Connection,
    category: Option<&str>,
    offset: u64,
    limit: u64,
    response: &mut data::MetricsCatalogResponse,
) -> std::result::Result<(), Status> {
    let start = page_number(offset)?;
    let end = page_number(
        offset
            .checked_add(limit)
            .ok_or_else(|| Status::invalid_argument("card window overflows"))?,
    )?;
    if limit == 0 {
        return Ok(());
    }
    let spans = catalog_spans(category.is_some(), category.is_none());
    let sql = format!(
        "WITH spans AS ({}), positions AS (
            SELECT plugin,tag,run_id,cards,
                sum(cards) OVER (ORDER BY tag COLLATE TAG_NAME_V2,rank,run_id ROWS UNBOUNDED PRECEDING) AS end
            FROM spans)
         SELECT plugin,tag,run_id,cards,max(0,?2-(end-cards)),min(cards,?3-(end-cards))
         FROM positions WHERE end>?2 AND end-cards<?3
         ORDER BY tag COLLATE TAG_NAME_V2,
             CASE plugin WHEN 'scalars' THEN 0 WHEN 'histograms' THEN 1 ELSE 2 END,run_id", spans
    );
    let mut stmt = c.prepare(&sql).map_err(status)?;
    // ?1 is reserved for the category even in filtered mode.
    let mut rows = stmt.query(params![category, start, end]).map_err(status)?;
    let mut metadata = c.prepare(
        "INSERT OR IGNORE INTO catalog_selected_metadata
         SELECT m.id,r.run_id FROM requested_runs r CROSS JOIN metrics_catalog m INDEXED BY metrics_catalog_tag
         WHERE m.run=r.name AND m.tag=?1 AND m.plugin=?2 AND (?3='' OR r.run_id=?3)"
    ).map_err(status)?;
    while let Some(row) = rows.next().map_err(status)? {
        let plugin: String = row.get(0).map_err(status)?;
        let tag: String = row.get(1).map_err(status)?;
        let run_id: String = row.get(2).map_err(status)?;
        let count: i64 = row.get(3).map_err(status)?;
        let first: i64 = row.get(4).map_err(status)?;
        let last: i64 = row.get(5).map_err(status)?;
        metadata
            .execute(params![tag, plugin, run_id])
            .map_err(status)?;
        for sample in first..last {
            response.cards.push(data::MetricCard {
                plugin: plugin.clone(),
                tag: tag.clone(),
                run_id: run_id.clone(),
                sample: if plugin == "images" { sample as u64 } else { 0 },
                num_sample: if plugin == "images" { count as u64 } else { 0 },
            });
        }
    }
    Ok(())
}

fn query_metrics_catalog(
    c: &Connection,
    req: data::MetricsCatalogRequest,
) -> std::result::Result<data::MetricsCatalogResponse, Status> {
    register_query(c, &req.query, "")?;
    let group_offset = page_number(req.group_offset)?;
    let group_limit = page_number(req.group_limit)?;
    c.execute_batch("BEGIN;
        CREATE TEMP TABLE catalog_plugins(name TEXT PRIMARY KEY,rank INTEGER NOT NULL) WITHOUT ROWID;
        CREATE TEMP TABLE catalog_selected_metadata(id INTEGER NOT NULL,run_id TEXT NOT NULL,
            PRIMARY KEY(id,run_id)) WITHOUT ROWID;
        CREATE TEMP TABLE catalog_totals(category TEXT PRIMARY KEY,cards INTEGER NOT NULL) WITHOUT ROWID;"
    ).map_err(status)?;
    catalog_runs_table(c, "requested_runs", &req.runs).map_err(status)?;
    catalog_runs_table(c, "pinned_runs", req.runs.iter().chain(&req.pinned_runs))
        .map_err(status)?;
    names_table(c, "pinned_tags", &req.pinned_tags).map_err(status)?;
    for (rank, plugin) in ["scalars", "histograms", "images"].iter().enumerate() {
        if req.plugins.is_empty() || req.plugins.iter().any(|p| p == plugin) {
            c.execute(
                "INSERT INTO catalog_plugins VALUES(?1,?2)",
                params![plugin, rank as i64],
            )
            .map_err(status)?;
        }
    }
    let mut response = data::MetricsCatalogResponse {
        group_offset: req.group_offset,
        ..Default::default()
    };
    if req.query.is_empty() {
        c.execute_batch(
            "CREATE TEMP TABLE requested_names(name TEXT PRIMARY KEY) WITHOUT ROWID;
            INSERT INTO requested_names SELECT DISTINCT name FROM requested_runs;",
        )
        .map_err(status)?;
        let run_count: i64 = c
            .query_row("SELECT count(*) FROM requested_names", [], |r| r.get(0))
            .map_err(status)?;
        // For multiple selected runs scalar union requires an index-only tag
        // distinct. All other counts, and single-run scalars, use group totals:
        // opening a category never reads another category's members.
        let scalars = if run_count <= 1 {
            "SELECT g.category,g.cards FROM requested_names r
             CROSS JOIN metrics_catalog_groups g
             WHERE g.run=r.name AND g.plugin='scalars'
             AND EXISTS(SELECT 1 FROM catalog_plugins WHERE name='scalars')"
        } else {
            "SELECT category,count(*) AS cards FROM (
                SELECT m.category,m.tag FROM requested_names r
                CROSS JOIN metrics_catalog m INDEXED BY metrics_catalog_scope
                WHERE m.run=r.name AND m.plugin='scalars' AND m.cards>0
                AND EXISTS(SELECT 1 FROM catalog_plugins WHERE name='scalars')
                GROUP BY m.category,m.tag) GROUP BY category"
        };
        c.execute_batch(&format!(
            "INSERT INTO catalog_totals
             SELECT category,sum(cards) FROM (
                SELECT g.category,g.cards FROM requested_runs r CROSS JOIN catalog_plugins p
                CROSS JOIN metrics_catalog_groups g
                WHERE g.run=r.name AND g.plugin=p.name AND p.name!='scalars'
                UNION ALL {})
             GROUP BY category;",
            scalars
        ))
        .map_err(status)?;
        let (groups, cards): (i64, i64) = c
            .query_row(
                "SELECT count(*),coalesce(sum(cards),0) FROM catalog_totals",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .map_err(status)?;
        response.total_groups = groups as u64;
        response.total_cards = cards as u64;
        let mut stmt = c.prepare(
            "SELECT category,cards FROM catalog_totals ORDER BY category COLLATE TAG_NAME_V2 LIMIT ?1 OFFSET ?2"
        ).map_err(status)?;
        let groups = stmt
            .query_map(params![group_limit, group_offset], |r| {
                Ok(data::MetricsCatalogGroup {
                    name: r.get(0)?,
                    total_cards: r.get::<_, i64>(1)? as u64,
                })
            })
            .map_err(status)?;
        response.groups = groups.collect::<rusqlite::Result<_>>().map_err(status)?;
        for group in &req.groups {
            catalog_window(
                c,
                Some(&group.name),
                group.offset,
                group.limit,
                &mut response,
            )?;
        }
    } else {
        let count: i64 = c
            .query_row(
                &format!(
                    "SELECT coalesce(sum(cards),0) FROM ({})",
                    catalog_spans(false, true)
                ),
                [],
                |r| r.get(0),
            )
            .map_err(status)?;
        response.total_cards = count as u64;
        catalog_window(
            c,
            None,
            req.filtered_offset,
            req.filtered_limit,
            &mut response,
        )?;
    }
    // Pins are independent of visible plugins, regex, and both windows. Exact
    // indexed run/tag lookups include explicitly pinned runs outside selection.
    c.execute_batch(
        "INSERT OR IGNORE INTO catalog_selected_metadata
         SELECT m.id,r.run_id FROM pinned_runs r CROSS JOIN pinned_tags t
         CROSS JOIN metrics_catalog m INDEXED BY metrics_catalog_tag
         WHERE m.run=r.name AND m.tag=t.name;",
    )
    .map_err(status)?;
    let mut stmt = c
        .prepare(
            "SELECT m.plugin,m.tag,x.run_id,s.metadata,m.cards
         FROM catalog_selected_metadata x CROSS JOIN metrics_catalog m CROSS JOIN series s
         WHERE m.id=x.id AND s.id=x.id
         ORDER BY m.tag COLLATE TAG_NAME_V2,
            CASE m.plugin WHEN 'scalars' THEN 0 WHEN 'histograms' THEN 1 ELSE 2 END,x.run_id",
        )
        .map_err(status)?;
    let mut rows = stmt.query([]).map_err(status)?;
    while let Some(row) = rows.next().map_err(status)? {
        let plugin: String = row.get(0).map_err(status)?;
        let bytes: Vec<u8> = row.get(3).map_err(status)?;
        let metadata = pb::SummaryMetadata::decode(bytes.as_slice()).map_err(status)?;
        response.series.push(data::MetricSeries {
            max_samples: if plugin == "images" {
                row.get::<_, i64>(4).map_err(status)? as u64
            } else {
                0
            },
            plugin,
            tag: row.get(1).map_err(status)?,
            run_id: row.get(2).map_err(status)?,
            description: metadata.summary_description,
        });
    }
    response.cards.sort_by(|a, b| {
        compare_tag_names(&a.tag, &b.tag)
            .then_with(|| catalog_plugin_rank(&a.plugin).cmp(&catalog_plugin_rank(&b.plugin)))
            .then_with(|| a.run_id.cmp(&b.run_id))
            .then_with(|| a.sample.cmp(&b.sample))
    });
    response.cards.dedup();
    Ok(response)
}

fn catalog_plugin_rank(plugin: &str) -> u8 {
    match plugin {
        "scalars" => 0,
        "histograms" => 1,
        _ => 2,
    }
}

/// Memory mode shares the disk query semantics but may derive its catalog from
/// scoped metadata. Only disk mode maintains the persistent load-bearing index.
pub(crate) fn memory_metrics_catalog(
    req: data::MetricsCatalogRequest,
    rows: Vec<SeriesMetadata>,
) -> std::result::Result<data::MetricsCatalogResponse, Status> {
    let c = Connection::open_in_memory().map_err(status)?;
    c.create_collation("TAG_NAME_V2", compare_tag_names)
        .map_err(status)?;
    c.execute_batch("CREATE TABLE series(id INTEGER PRIMARY KEY,metadata BLOB NOT NULL);")
        .map_err(status)?;
    create_metrics_catalog(&c).map_err(status)?;
    {
        let tx = c.unchecked_transaction().map_err(status)?;
        let mut series = tx
            .prepare("INSERT INTO series(metadata) VALUES(?1)")
            .map_err(status)?;
        let mut catalog = tx
            .prepare("INSERT INTO metrics_catalog VALUES(?1,?2,?3,?4,?5,?6)")
            .map_err(status)?;
        for row in rows {
            if !supported_metrics_metadata(&row.metadata) {
                continue;
            }
            let plugin = row
                .metadata
                .plugin_data
                .as_ref()
                .map(|p| p.plugin_name.as_str())
                .unwrap_or("");
            let cards = if plugin == "images" {
                (row.max_length - 2).max(0)
            } else {
                1
            };
            series
                .execute(params![row.metadata.encode_to_vec()])
                .map_err(status)?;
            catalog
                .execute(params![
                    tx.last_insert_rowid(),
                    row.run,
                    row.tag,
                    plugin,
                    row.tag.split('/').next().unwrap_or(""),
                    cards
                ])
                .map_err(status)?;
        }
        drop(series);
        drop(catalog);
        tx.execute_batch(
            "INSERT INTO metrics_catalog_groups SELECT run,plugin,category,sum(cards)
            FROM metrics_catalog WHERE cards>0 GROUP BY run,plugin,category;",
        )
        .map_err(status)?;
        tx.commit().map_err(status)?;
    }
    query_metrics_catalog(&c, req)
}
impl DiskStore {
    pub(crate) fn metrics_catalog(
        &self,
        req: data::MetricsCatalogRequest,
    ) -> std::result::Result<data::MetricsCatalogResponse, Status> {
        let c = self.connect().map_err(status)?;
        query_metrics_catalog(&c, req)
    }

    pub fn list_runs(
        &self,
        req: data::ListRunsRequest,
    ) -> std::result::Result<data::ListRunsResponse, Status> {
        let c = self.connect().map_err(status)?;
        register_query(&c, &req.query, &req.query_prefix)?;
        let has_session_filter =
            req.default_rank < 0 || req.session_ranks.iter().any(|entry| entry.rank < 0);
        let has_session_order = req.sort_by == "session_rank" && !req.session_ranks.is_empty();
        if has_session_filter || has_session_order {
            let ranks = SessionRanks::new(&req.session_ranks, req.default_rank);
            c.create_scalar_function(
                "session_rank",
                1,
                rusqlite::functions::FunctionFlags::SQLITE_DETERMINISTIC,
                move |ctx| Ok(ranks.rank(ctx.get_raw(0).as_str()?)),
            )
            .map_err(status)?;
        }
        let mut predicate = "start IS NOT NULL".to_owned();
        if has_session_filter {
            predicate.push_str(" AND session_rank(name)>=0");
        }
        if !req.query.is_empty() {
            predicate.push_str(" AND matches_query(name)");
        }
        if let Some(names) = req.names {
            names_table(&c, "requested_runs", &names.names).map_err(status)?;
            predicate.push_str(" AND name IN (SELECT name FROM requested_runs)");
        }
        let order = match req.sort_by.as_str() {
            "" | "start_time" => "start",
            "name" => "name",
            "session_rank" if has_session_order => "session_rank(name)",
            "session_rank" => "name",
            _ => {
                return Err(Status::invalid_argument(
                    "sort_by must be name, start_time, or session_rank",
                ))
            }
        };
        let direction = if req.descending { "DESC" } else { "ASC" };
        let limit = if req.limit == 0 {
            -1
        } else {
            page_number(req.limit)?
        };
        let offset = page_number(req.offset)?;
        c.execute_batch("BEGIN").map_err(status)?;
        let total: i64 = c
            .query_row(
                &format!("SELECT count(*) FROM runs WHERE {}", predicate),
                [],
                |r| r.get(0),
            )
            .map_err(status)?;
        let mut stmt = c
            .prepare(&format!(
                "SELECT name,start FROM runs WHERE {} ORDER BY {} {},name {} LIMIT ?1 OFFSET ?2",
                predicate, order, direction, direction
            ))
            .map_err(status)?;
        let rows = stmt
            .query_map(params![limit, offset], |r| {
                Ok(data::Run {
                    name: r.get(0)?,
                    start_time: r.get(1)?,
                })
            })
            .map_err(status)?;
        Ok(data::ListRunsResponse {
            runs: rows.collect::<rusqlite::Result<_>>().map_err(status)?,
            total: total as u64,
        })
    }

    pub fn plugins(&self) -> std::result::Result<data::ListPluginsResponse, Status> {
        let c = self.connect().map_err(status)?;
        let mut stmt = c
            .prepare("SELECT name FROM plugins ORDER BY name")
            .map_err(status)?;
        let rows = stmt
            .query_map([], |r| Ok(data::Plugin { name: r.get(0)? }))
            .map_err(status)?;
        Ok(data::ListPluginsResponse {
            plugins: rows.collect::<rusqlite::Result<_>>().map_err(status)?,
        })
    }

    fn selected(
        &self,
        plugin: &str,
        class: pb::DataClass,
        filter: Option<&data::RunTagFilter>,
        list: bool,
    ) -> std::result::Result<(Connection, u64), Status> {
        let c = self.connect().map_err(status)?;
        let default = data::RunTagFilter::default();
        let filter = filter.unwrap_or(&default);
        register_query(&c, &filter.tag_query, "")?;
        let mut predicate = "plugin=?1 AND class=?2".to_owned();
        if !filter.tag_query.is_empty() {
            predicate.push_str(" AND matches_query(tag)");
        }
        if list {
            predicate.push_str(" AND valid>0");
        }
        let single_run = filter.runs.as_ref().and_then(|runs| {
            runs.names
                .first()
                .filter(|first| runs.names.iter().all(|name| name == *first))
        });
        if let Some(runs) = &filter.runs {
            names_table(&c, "requested_runs", &runs.names).map_err(status)?;
            predicate.push_str(if single_run.is_some() {
                " AND run=(SELECT name FROM requested_runs)"
            } else {
                " AND run IN (SELECT name FROM requested_runs)"
            });
        }
        if let Some(tags) = &filter.tags {
            names_table(&c, "requested_tags", &tags.names).map_err(status)?;
            predicate.push_str(" AND tag IN (SELECT name FROM requested_tags)");
        }
        let limit = if filter.tag_limit == 0 {
            -1
        } else {
            page_number(filter.tag_limit)?
        };
        let offset = page_number(filter.tag_offset)?;
        let source = match (list, filter.runs.is_some()) {
            (true, true) => "series INDEXED BY series_list_run_plugin_tag",
            (true, false) => "series INDEXED BY series_list_plugin_tag",
            (false, true) => "series INDEXED BY series_run_plugin_tag",
            (false, false) => "series INDEXED BY series_plugin_tag",
        };
        c.execute_batch("BEGIN; CREATE TEMP TABLE selected_tags(name TEXT PRIMARY KEY) WITHOUT ROWID; CREATE TEMP TABLE selected_series(id INTEGER PRIMARY KEY);").map_err(status)?;
        let total: i64 = if list
            && single_run.is_some()
            && filter.tag_query.is_empty()
            && filter.tags.is_none()
        {
            c.query_row(
                "SELECT coalesce((SELECT count FROM series_counts WHERE run=?3 AND plugin=?1 AND class=?2),0)",
                params![plugin, class as i32, single_run],
                |row| row.get(0),
            ).map_err(status)?
        } else if list {
            c.query_row(
                &format!(
                    "SELECT count(DISTINCT tag) FROM {} WHERE {}",
                    source, predicate
                ),
                params![plugin, class as i32],
                |r| r.get(0),
            )
            .map_err(status)?
        } else {
            0
        };
        let distinct = if single_run.is_some() {
            ""
        } else {
            "DISTINCT "
        };
        c.execute(&format!("INSERT INTO selected_tags SELECT {}tag FROM {} WHERE {} ORDER BY tag LIMIT ?3 OFFSET ?4",distinct,source,predicate),params![plugin,class as i32,limit,offset]).map_err(status)?;
        c.execute(&format!("INSERT INTO selected_series SELECT id FROM {} WHERE {} AND tag IN (SELECT name FROM selected_tags)",source,predicate),params![plugin,class as i32]).map_err(status)?;
        Ok((c, total as u64))
    }

    pub(crate) fn listing(
        &self,
        plugin: &str,
        class: pb::DataClass,
        filter: Option<&data::RunTagFilter>,
    ) -> std::result::Result<Listing, Status> {
        let (c, total) = self.selected(plugin, class, filter, true)?;
        let mut stmt = c.prepare("SELECT run,tag,metadata,max_step,max_wall,coalesce(max_length,0) FROM series WHERE id IN (SELECT id FROM selected_series) ORDER BY run,tag").map_err(status)?;
        let mut rows = stmt.query([]).map_err(status)?;
        let mut result = Vec::new();
        while let Some(r) = rows.next().map_err(status)? {
            let encoded: Vec<u8> = r.get(2).map_err(status)?;
            result.push(SeriesMetadata {
                run: r.get(0).map_err(status)?,
                tag: r.get(1).map_err(status)?,
                metadata: pb::SummaryMetadata::decode(encoded.as_slice()).map_err(status)?,
                max_step: r.get(3).map_err(status)?,
                max_wall: r.get(4).map_err(status)?,
                max_length: r.get(5).map_err(status)?,
            });
        }
        Ok(Listing {
            total,
            rows: result,
        })
    }

    /// Materialize only explicitly requested sampled series, owned by this RPC.
    /// The legacy response/downsampling code consumes this short-lived snapshot.
    pub(crate) fn read(
        &self,
        plugin: &str,
        class: pb::DataClass,
        filter: Option<&data::RunTagFilter>,
        num_points: usize,
    ) -> std::result::Result<Commit, Status> {
        if class != pb::DataClass::Scalar && class != pb::DataClass::Tensor {
            return Err(Status::invalid_argument(
                "payload snapshots require scalar or tensor data",
            ));
        }
        let (c, _) = self.selected(plugin, class, filter, false)?;
        let result = Commit::new();
        let mut runs = result.runs.write().map_err(lock_status)?;
        let mut stmt = c.prepare("SELECT id,run,tag,metadata FROM series WHERE id IN (SELECT id FROM selected_series)").map_err(status)?;
        let mut rows = stmt.query([]).map_err(status)?;
        while let Some(r) = rows.next().map_err(status)? {
            let id: i64 = r.get(0).map_err(status)?;
            let run: String = r.get(1).map_err(status)?;
            let tag = Tag(r.get(2).map_err(status)?);
            let bytes: Vec<u8> = r.get(3).map_err(status)?;
            let metadata = Box::new(pb::SummaryMetadata::decode(bytes.as_slice()).map_err(status)?);
            let run = runs
                .entry(Run(run))
                .or_insert_with(|| RwLock::new(result.new_run_data()));
            let mut run = run.write().map_err(lock_status)?;
            let order = match num_points {
                0 => "ASC LIMIT 0",
                1 => "DESC LIMIT 1",
                _ => "ASC",
            };
            let mut samples = c.prepare(&format!("SELECT step,wall,value FROM samples WHERE series=?1 AND value IS NOT NULL ORDER BY step {}",order)).map_err(status)?;
            let mut samples = samples.query(params![id]).map_err(status)?;
            let mut scalars = Vec::new();
            let mut tensors = Vec::new();
            while let Some(sample) = samples.next().map_err(status)? {
                let step = Step(sample.get(0).map_err(status)?);
                let wall = WallTime::new(sample.get(1).map_err(status)?)
                    .ok_or_else(|| status("invalid stored wall time"))?;
                let payload: Vec<u8> = sample.get(2).map_err(status)?;
                match class {
                    pb::DataClass::Scalar => {
                        let bytes: [u8; 4] = payload.as_slice().try_into().map_err(status)?;
                        scalars.push((step, (wall, Ok(ScalarValue(f32::from_le_bytes(bytes))))));
                    }
                    pb::DataClass::Tensor => tensors.push((
                        step,
                        (
                            wall,
                            Ok(pb::TensorProto::decode(payload.as_slice()).map_err(status)?),
                        ),
                    )),
                    _ => (),
                }
            }
            match class {
                pb::DataClass::Scalar => {
                    run.scalars.insert(
                        tag,
                        TimeSeries {
                            metadata,
                            basin: Basin::from_sorted(scalars),
                        },
                    );
                }
                pb::DataClass::Tensor => {
                    run.tensors.insert(
                        tag,
                        TimeSeries {
                            metadata,
                            basin: Basin::from_sorted(tensors),
                        },
                    );
                }
                _ => (),
            }
        }
        drop(runs);
        Ok(result)
    }

    pub(crate) fn blob_sequences(
        &self,
        plugin: &str,
        experiment: &str,
        filter: Option<&data::RunTagFilter>,
        num_points: usize,
    ) -> std::result::Result<data::ReadBlobSequencesResponse, Status> {
        let (c, _) = self.selected(plugin, pb::DataClass::BlobSequence, filter, false)?;
        let mut stmt = c.prepare("SELECT id,run,tag FROM series WHERE id IN (SELECT id FROM selected_series) ORDER BY run,tag").map_err(status)?;
        let mut rows = stmt.query([]).map_err(status)?;
        let mut result = data::ReadBlobSequencesResponse::default();
        while let Some(row) = rows.next().map_err(status)? {
            let id: i64 = row.get(0).map_err(status)?;
            let run: String = row.get(1).map_err(status)?;
            let tag: String = row.get(2).map_err(status)?;
            let mut points = Vec::<(i64, f64, i64)>::new();
            if num_points > 0 {
                let order = if num_points == 1 {
                    "DESC LIMIT 1"
                } else {
                    "ASC"
                };
                let mut samples = c.prepare(&format!("SELECT step,wall,length FROM samples WHERE series=?1 AND value IS NOT NULL ORDER BY step {}",order)).map_err(status)?;
                let samples = samples
                    .query_map(params![id], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
                    .map_err(status)?;
                points = samples.collect::<rusqlite::Result<_>>().map_err(status)?;
                crate::downsample::downsample(&mut points, num_points);
            }
            let mut series = data::BlobSequenceData::default();
            for (step, wall, length) in points {
                series.step.push(step);
                series.wall_time.push(wall);
                let refs = (0..length)
                    .map(|index| {
                        let key = crate::blob_key::BlobKey {
                            experiment_id: experiment.into(),
                            run: run.as_str().into(),
                            tag: tag.as_str().into(),
                            step: Step(step),
                            index: index as usize,
                        };
                        data::BlobReference {
                            blob_key: key.to_string(),
                            url: String::new(),
                        }
                    })
                    .collect();
                series
                    .values
                    .push(data::BlobReferenceSequence { blob_refs: refs });
            }
            if result.runs.last().map_or(true, |r| r.run_name != run) {
                result
                    .runs
                    .push(data::read_blob_sequences_response::RunEntry {
                        run_name: run,
                        tags: Vec::new(),
                    });
            }
            result.runs.last_mut().unwrap().tags.push(
                data::read_blob_sequences_response::TagEntry {
                    tag_name: tag,
                    data: Some(series),
                },
            );
        }
        Ok(result)
    }

    pub(crate) fn blob(
        &self,
        key: &crate::blob_key::BlobKey,
    ) -> std::result::Result<Bytes, Status> {
        let c = self.connect().map_err(status)?;
        let bytes: Option<Vec<u8>> = c.query_row("SELECT b.value FROM series s JOIN blobs b ON b.series=s.id WHERE s.run=?1 AND s.tag=?2 AND b.step=?3 AND b.idx=?4",params![key.run.as_ref(),key.tag.as_ref(),key.step.0,key.index as i64],|r|r.get(0)).optional().map_err(status)?;
        bytes
            .map(Bytes::from)
            .ok_or_else(|| Status::not_found("blob not found; it may have been evicted"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::disk_logdir::DiskLogdir;
    use crate::reservoir::StageReservoir;
    use crate::writer::SummaryWriteExt;
    use std::fs::{self, File, OpenOptions};
    use std::io::Write;

    #[test]
    fn metrics_tag_order_compares_numbers_and_path_components() {
        for (a, b) in [
            ("g2/tag", "g10/tag"),
            ("a/a", "a+/a"),
            ("g1.9/tag", "g1.11e1/tag"),
            ("g9/tag", "g1E+1/tag"),
            ("g2", "g!"),
            ("a", "a/a"),
            ("g\u{10000}", "g\u{e000}"),
        ] {
            assert_eq!(compare_tag_names(a, b), Ordering::Less, "{} < {}", a, b);
            assert_eq!(compare_tag_names(b, a), Ordering::Greater);
        }
    }

    #[test]
    fn metrics_tag_order_is_a_strict_total_order() {
        // Exhaustive over short tags from an alphabet of every token class
        // boundary: the previous comparator treated '/' and digits as equal
        // and "1e" as NaN, which is not transitive and corrupted the index.
        let alphabet = [
            '1',
            '2',
            '0',
            '.',
            'e',
            '+',
            '/',
            'a',
            '_',
            '\u{10000}',
            '\u{e000}',
        ];
        let mut tags = vec![String::new()];
        for _ in 0..3 {
            let longer: Vec<String> = tags
                .iter()
                .flat_map(|t| alphabet.iter().map(move |c| format!("{}{}", t, c)))
                .collect();
            tags.extend(longer);
        }
        tags.sort();
        tags.dedup();
        let mut sorted = tags.clone();
        sorted.sort_by(|a, b| compare_tag_names(a, b));
        for (i, a) in sorted.iter().enumerate() {
            assert_eq!(compare_tag_names(a, a), Ordering::Equal);
            for b in &sorted[i + 1..] {
                // A sorted sequence under a total order is pairwise ordered.
                assert_eq!(compare_tag_names(a, b), Ordering::Less, "{:?} < {:?}", a, b);
                assert_eq!(
                    compare_tag_names(b, a),
                    Ordering::Greater,
                    "{:?} > {:?}",
                    b,
                    a
                );
            }
        }
    }

    /// Shared with `compareTagNames` in webapp/metrics/utils_test.ts: the
    /// frontend and the persisted collation must agree on every tag.
    const TAG_ORDER_FIXTURE: &[&str] = &[
        "",
        "/",
        "0..2",
        "0..03",
        "B",
        "a",
        "a/a",
        "a01",
        "a1",
        "a1.5",
        "a1.e5",
        "a+/a",
        "b",
        "g9",
        "g1E+1",
        "layer/w",
        "layer1",
        "lr1e-4",
        "lr1e-3",
        "model/conv/kernel",
        "model/conv1/kernel",
        "res1encoder",
        "res2",
        "res1e5",
        "x\u{10000}",
        "x\u{e000}",
    ];

    #[test]
    fn metrics_tag_order_matches_frontend_fixture() {
        let mut tags = TAG_ORDER_FIXTURE.to_vec();
        tags.reverse();
        tags.sort_by(|a, b| compare_tag_names(a, b));
        assert_eq!(tags, TAG_ORDER_FIXTURE);
    }

    #[test]
    fn metrics_tag_order_examples() {
        for (a, b) in [
            ("a/b", "a1"),
            ("lr1e_x", "lr2"),
            ("x1e5", "x1e6"),
            ("x1.5", "x2"),
            ("x01", "x1"),
        ] {
            assert_eq!(compare_tag_names(a, b), Ordering::Less, "{} < {}", a, b);
            assert_eq!(compare_tag_names(b, a), Ordering::Greater);
        }
    }

    fn write_scalars(path: &Path, steps: std::ops::Range<i64>) -> Result<()> {
        let mut file = OpenOptions::new().create(true).append(true).open(path)?;
        for step in steps {
            file.write_scalar(
                &Tag("loss".into()),
                Step(step),
                WallTime::new(100.0 + step as f64).unwrap(),
                step as f32,
            )?;
        }
        file.flush()?;
        Ok(())
    }

    /// Commits by other connections since the last call on `observer`.
    fn data_version(observer: &Connection) -> i64 {
        observer
            .query_row("PRAGMA data_version", [], |r| r.get(0))
            .unwrap()
    }

    #[test]
    fn unchanged_logdir_reload_commits_nothing() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let logs = dir.path().join("logs");
        for run in ["a", "b", "c"] {
            fs::create_dir_all(logs.join(run))?;
            write_scalars(&logs.join(run).join("tfevents.1"), 0..3)?;
        }
        fs::write(logs.join("c/tfevents.empty"), b"")?;
        let store = DiskStore::at(dir.path().join("index.sqlite"), "idle-test")?;
        let hints = PluginSamplingHint::default();
        let logdir = DiskLogdir::new(logs.clone());
        store.reload(&logdir, &hints, true)?;
        let observer = store.connect()?;
        let version = data_version(&observer);
        store.reload(&logdir, &hints, true)?;
        store.reload(&logdir, &hints, true)?;
        assert_eq!(data_version(&observer), version, "idle reload wrote");

        // Appending to one run writes, and only that run gains points.
        write_scalars(&logs.join("b/tfevents.1"), 3..5)?;
        store.reload(&logdir, &hints, true)?;
        assert_ne!(data_version(&observer), version);
        let version = data_version(&observer);
        let response = store.list_runs(data::ListRunsRequest::default())?;
        assert_eq!(response.runs.len(), 3);
        store.reload(&logdir, &hints, true)?;
        assert_eq!(data_version(&observer), version, "idle reload wrote");

        // A removed file is tombstoned once, then idles again.
        fs::remove_file(logs.join("a/tfevents.1"))?;
        store.reload(&logdir, &hints, true)?;
        let version = data_version(&observer);
        store.reload(&logdir, &hints, true)?;
        assert_eq!(data_version(&observer), version, "tombstone rewritten");
        let runs: Vec<String> = store
            .list_runs(data::ListRunsRequest::default())?
            .runs
            .into_iter()
            .map(|r| r.name)
            .collect();
        assert_eq!(runs.len(), 2, "{:?}", runs);
        Ok(())
    }

    #[test]
    fn reset_run_replays_every_file_and_keeps_tombstones() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let logs = dir.path().join("logs");
        fs::create_dir_all(logs.join("train"))?;
        let first = logs.join("train/tfevents.1");
        let second = logs.join("train/tfevents.2");
        let gone = logs.join("train/tfevents.0");
        write_scalars(&gone, 0..1)?;
        write_scalars(&first, 1..3)?;
        write_scalars(&second, 3..5)?;
        let store = DiskStore::at(dir.path().join("index.sqlite"), "reset-test")?;
        let hints = PluginSamplingHint::default();
        let logdir = DiskLogdir::new(logs.clone());
        store.reload(&logdir, &hints, true)?;
        fs::remove_file(&gone)?;
        store.reload(&logdir, &hints, true)?;
        // Replacing one file replays the run: the untouched sibling is read
        // again, and the tombstoned file stays dead.
        fs::remove_file(&first)?;
        write_scalars(&first, 1..2)?;
        store.reload(&logdir, &hints, true)?;
        let steps: Vec<i64> = values(&store).into_iter().map(|(s, _)| s.0).collect();
        assert_eq!(steps, vec![1, 3, 4]);
        let c = store.connect()?;
        let (dead, present): (bool, bool) = c.query_row(
            "SELECT dead,present FROM files WHERE path=?1",
            params![encode_path(&gone)],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )?;
        assert!(dead && !present);
        Ok(())
    }

    fn integrity_check(store: &DiskStore) -> String {
        store
            .connect()
            .unwrap()
            .query_row("PRAGMA integrity_check(1)", [], |r| r.get(0))
            .unwrap()
    }

    #[test]
    fn collation_migration_and_repair_rebuild_collated_indexes() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let logs = dir.path().join("logs");
        fs::create_dir_all(logs.join("train"))?;
        let mut file = File::create(logs.join("train/tfevents.1"))?;
        for (i, tag) in ["m/b", "m/a", "m/c10", "m/c9", "m/1e"].iter().enumerate() {
            file.write_scalar(
                &Tag((*tag).into()),
                Step(0),
                WallTime::new(1.0 + i as f64).unwrap(),
                1.0,
            )?;
        }
        file.flush()?;
        let path = dir.path().join("index.sqlite");
        let hints = PluginSamplingHint::default();
        let logdir = DiskLogdir::new(logs);
        let store = DiskStore::at(path.clone(), "repair-test")?;
        store.reload(&logdir, &hints, true)?;
        drop(store);

        let old = Connection::open(&path)?;
        old.create_collation("TAG_NAME", compare_legacy_tag_names)?;
        old.execute_batch(
            "
            DROP INDEX metrics_catalog_scope;
            CREATE INDEX metrics_catalog_scope ON metrics_catalog(
                run,plugin,category,tag COLLATE TAG_NAME,cards
            );
            PRAGMA user_version=2;
        ",
        )?;
        let store = DiskStore::at(path.clone(), "repair-test")?;
        store.migrate()?;
        assert_eq!(integrity_check(&store), "ok", "migration did not reindex");
        let error = old
            .execute(
                "INSERT INTO metrics_catalog VALUES(-1,'train','m/new','scalars','m',1)",
                [],
            )
            .unwrap_err();
        assert!(error.to_string().contains("no such collation sequence"));
        drop(old);
        drop(store);

        // Misorder the current index without changing its schema version.
        let wrong = Connection::open(&path)?;
        wrong.create_collation("TAG_NAME_V2", |a: &str, b: &str| b.cmp(a))?;
        wrong.execute_batch("REINDEX TAG_NAME_V2")?;
        drop(wrong);
        let store = DiskStore::at(path.clone(), "repair-test")?;
        store.migrate()?;
        assert_ne!(integrity_check(&store), "ok");
        store.repair()?;
        assert_eq!(integrity_check(&store), "ok");
        assert_eq!(
            store
                .listing("scalars", pb::DataClass::Scalar, None)?
                .rows
                .len(),
            5
        );

        // Startup resets a corrupted database; the next load rebuilds it.
        drop(store);
        fs::write(&path, b"not a database at all, just garbage bytes......")?;
        let store = DiskStore::at(path, "repair-test")?;
        store.reload(&logdir, &hints, true)?;
        assert_eq!(
            store
                .listing("scalars", pb::DataClass::Scalar, None)?
                .rows
                .len(),
            5
        );
        Ok(())
    }

    #[test]
    fn repair_resets_damaged_tables_under_open_connections() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let logs = dir.path().join("logs");
        fs::create_dir_all(logs.join("train"))?;
        let mut file = File::create(logs.join("train/tfevents.1"))?;
        for step in 0..5 {
            file.write_scalar(
                &Tag("loss".into()),
                Step(step),
                WallTime::new(1.0 + step as f64).unwrap(),
                1.0,
            )?;
        }
        file.flush()?;
        let path = dir.path().join("index.sqlite");
        let hints = PluginSamplingHint::default();
        let logdir = DiskLogdir::new(logs);
        let store = DiskStore::at(path.clone(), "repair-test")?;
        store.reload(&logdir, &hints, true)?;

        // Overwrite the samples root page: the header stays valid, the table
        // does not, so REINDEX cannot help and repair must reset the index.
        let c = store.connect()?;
        let (root, page_size): (i64, i64) = c.query_row(
            "SELECT rootpage, (SELECT page_size FROM pragma_page_size)
                FROM sqlite_schema WHERE name='samples'",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )?;
        c.execute_batch("PRAGMA wal_checkpoint(TRUNCATE)")?;
        drop(c);
        {
            use std::io::{Seek, SeekFrom};
            let mut f = fs::OpenOptions::new().write(true).open(&path)?;
            f.seek(SeekFrom::Start(((root - 1) * page_size) as u64))?;
            f.write_all(&vec![0xff; page_size as usize])?;
        }
        assert_ne!(integrity_check(&store), "ok");

        // A connection open across the repair, like a concurrent request.
        let reader = store.connect()?;
        reader.query_row("SELECT count(*) FROM info", [], |r| r.get::<_, i64>(0))?;
        let id = file_id(&path);
        store.repair()?;
        assert_eq!(file_id(&path), id, "repair must reset in place, not delete");
        assert_eq!(integrity_check(&store), "ok");
        let samples: i64 = reader.query_row("SELECT count(*) FROM samples", [], |r| r.get(0))?;
        assert_eq!(samples, 0, "reader must see the reset index");
        store.reload(&logdir, &hints, true)?;
        let samples: i64 = reader.query_row("SELECT count(*) FROM samples", [], |r| r.get(0))?;
        assert_eq!(samples, 5);
        assert_eq!(integrity_check(&store), "ok");
        drop(reader);

        // Damage the schema itself, keeping the file header: `connect()`
        // cannot even apply its pragmas, but the reset still works in place.
        store
            .connect()?
            .execute_batch("PRAGMA wal_checkpoint(TRUNCATE)")?;
        *store.loader.lock().unwrap() = None;
        {
            use std::io::{Seek, SeekFrom};
            let mut f = fs::OpenOptions::new().write(true).open(&path)?;
            f.seek(SeekFrom::Start(100))?;
            f.write_all(&vec![0xff; page_size as usize - 100])?;
        }
        assert!(store.connect().is_err() || integrity_check(&store) != "ok");
        let id = file_id(&path);
        store.repair()?;
        assert_eq!(file_id(&path), id, "repair must reset in place, not delete");
        store.reload(&logdir, &hints, true)?;
        let samples: i64 = store
            .connect()?
            .query_row("SELECT count(*) FROM samples", [], |r| r.get(0))?;
        assert_eq!(samples, 5);
        assert_eq!(integrity_check(&store), "ok");
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn kept_loader_connection_follows_a_replaced_index() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let logs = dir.path().join("logs");
        fs::create_dir_all(logs.join("train"))?;
        let mut file = File::create(logs.join("train/tfevents.1"))?;
        file.write_scalar(
            &Tag("loss".into()),
            Step(0),
            WallTime::new(1.0).unwrap(),
            1.0,
        )?;
        file.flush()?;
        let path = dir.path().join("index.sqlite");
        let hints = PluginSamplingHint::default();
        let logdir = DiskLogdir::new(logs);
        let store = DiskStore::at(path.clone(), "replace-test")?;
        store.reload(&logdir, &hints, true)?;
        // An external process replaces the cache file.
        for suffix in ["", "-wal", "-shm"] {
            let mut p = path.clone().into_os_string();
            p.push(suffix);
            let _ = fs::remove_file(&p);
        }
        DiskStore::at(path.clone(), "replace-test")?;
        store.reload(&logdir, &hints, true)?;
        assert_eq!(
            store
                .listing("scalars", pb::DataClass::Scalar, None)?
                .rows
                .len(),
            1
        );
        Ok(())
    }

    #[test]
    fn kept_loader_connection_holds_no_snapshot_between_reloads() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let logs = dir.path().join("logs");
        fs::create_dir_all(logs.join("train"))?;
        let mut file = File::create(logs.join("train/tfevents.1"))?;
        let write = |file: &mut File, step| {
            file.write_scalar(
                &Tag("loss".into()),
                Step(step),
                WallTime::new(1.0).unwrap(),
                1.0,
            )
        };
        write(&mut file, 0)?;
        file.flush()?;
        let hints = PluginSamplingHint::default();
        let logdir = DiskLogdir::new(logs);
        let store = DiskStore::at(dir.path().join("index.sqlite"), "loader-test")?;
        store.reload(&logdir, &hints, true)?;
        assert!(store.loader.lock().unwrap().is_some());
        // An open read transaction on the kept connection would leave the
        // checkpoint busy and the WAL untruncated.
        let checkpoint = |store: &DiskStore| -> Result<(i64, i64, i64)> {
            Ok(store
                .connect()?
                .query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |r| {
                    Ok((r.get(0)?, r.get(1)?, r.get(2)?))
                })?)
        };
        assert_eq!(checkpoint(&store)?, (0, 0, 0));
        // Writes by other connections are visible through the kept one.
        store.connect()?.execute("DELETE FROM runs", [])?;
        write(&mut file, 1)?;
        file.flush()?;
        store.reload(&logdir, &hints, true)?;
        assert_eq!(
            store
                .listing("scalars", pb::DataClass::Scalar, None)?
                .rows
                .len(),
            1
        );
        assert_eq!(checkpoint(&store)?, (0, 0, 0));
        Ok(())
    }

    /// Rewrites `samples` and `blobs` as the WITHOUT ROWID tables of schema 3.
    fn downgrade_sample_tables(store: &DiskStore) -> Result<()> {
        store.connect()?.execute_batch(
            "PRAGMA foreign_keys=OFF; BEGIN;
            CREATE TABLE samples_old(series INTEGER NOT NULL REFERENCES series(id) ON DELETE CASCADE,
                step INTEGER NOT NULL, wall REAL NOT NULL, value BLOB, length INTEGER, PRIMARY KEY(series,step)) WITHOUT ROWID;
            INSERT INTO samples_old SELECT series,step,wall,value,length FROM samples;
            CREATE TABLE blobs_old(series INTEGER NOT NULL, step INTEGER NOT NULL, idx INTEGER NOT NULL, value BLOB NOT NULL,
                PRIMARY KEY(series,step,idx), FOREIGN KEY(series,step) REFERENCES samples(series,step) ON DELETE CASCADE) WITHOUT ROWID;
            INSERT INTO blobs_old SELECT series,step,idx,value FROM blobs;
            DROP TABLE blobs; DROP TABLE samples;
            ALTER TABLE samples_old RENAME TO samples; ALTER TABLE blobs_old RENAME TO blobs;
            CREATE INDEX samples_valid_step ON samples(series,step) WHERE value IS NOT NULL;
            CREATE INDEX samples_valid_wall ON samples(series,wall) WHERE value IS NOT NULL;
            CREATE INDEX samples_valid_length ON samples(series,length) WHERE value IS NOT NULL;
            PRAGMA user_version=3; COMMIT;",
        )?;
        Ok(())
    }

    fn without_rowid(c: &Connection, table: &str) -> bool {
        c.query_row(
            "SELECT wr FROM pragma_table_list WHERE schema='main' AND name=?1",
            [table],
            |r| r.get(0),
        )
        .unwrap()
    }

    type SampleRow = (i64, i64, f64, Option<Vec<u8>>, Option<i64>);

    fn sample_rows(c: &Connection) -> (Vec<SampleRow>, Vec<(i64, i64, i64, Vec<u8>)>) {
        let samples = c
            .prepare("SELECT series,step,wall,value,length FROM samples ORDER BY series,step")
            .unwrap()
            .query_map([], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?))
            })
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        let blobs = c
            .prepare("SELECT series,step,idx,value FROM blobs ORDER BY series,step,idx")
            .unwrap()
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        (samples, blobs)
    }

    #[test]
    fn without_rowid_sample_tables_migrate_in_place() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let logs = dir.path().join("logs");
        fs::create_dir_all(logs.join("train"))?;
        let events = logs.join("train/tfevents.1");
        write_scalars(&events, 0..5)?;
        let path = dir.path().join("index.sqlite");
        let hints = PluginSamplingHint::default();
        let logdir = DiskLogdir::new(logs);
        let store = DiskStore::at(path.clone(), "rowid-test")?;
        store.reload(&logdir, &hints, true)?;
        let mut c = store.connect()?;
        let tx = c.transaction()?;
        index_event(
            &tx,
            "train",
            pb::Event {
                step: 1,
                wall_time: 1.0,
                what: Some(pb::event::What::GraphDef(Bytes::from_static(b"graph"))),
                ..Default::default()
            },
            &hints,
        )?;
        tx.commit()?;
        assert!(!without_rowid(&c, "samples") && !without_rowid(&c, "blobs"));
        let before = sample_rows(&c);
        assert_eq!((before.0.len(), before.1.len()), (6, 1));
        drop(c);
        downgrade_sample_tables(&store)?;
        assert!(without_rowid(&store.connect()?, "samples"));
        drop(store);

        let store = DiskStore::at(path.clone(), "rowid-test")?;
        let c = store.connect()?;
        // Opening never migrates an existing index; the reloader does.
        assert!(without_rowid(&c, "samples"));
        store.migrate()?;
        assert!(!without_rowid(&c, "samples") && !without_rowid(&c, "blobs"));
        assert_eq!(sample_rows(&c), before);
        let version: i64 = c.query_row("PRAGMA user_version", [], |r| r.get(0))?;
        assert_eq!(version, ROWID_SAMPLES_VERSION);
        assert_eq!(integrity_check(&store), "ok");
        assert_eq!(
            c.query_row("PRAGMA foreign_key_check", [], |r| r.get::<_, String>(0))
                .optional()?,
            None
        );
        let key = crate::blob_key::BlobKey {
            experiment_id: "".into(),
            run: "train".into(),
            tag: GraphDefValue::TAG_NAME.into(),
            step: Step(1),
            index: 0,
        };
        assert_eq!(store.blob(&key)?, Bytes::from_static(b"graph"));

        // Loading and cascading deletes keep working on the migrated tables.
        write_scalars(&events, 5..8)?;
        store.reload(&logdir, &hints, true)?;
        assert_eq!(sample_rows(&c).0.len(), 9);
        c.execute("DELETE FROM runs WHERE name='train'", [])?;
        assert_eq!(sample_rows(&c), (vec![], vec![]));
        Ok(())
    }

    #[test]
    fn recreated_index_does_not_reissue_revisions() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("index.sqlite");
        let mut seen = std::collections::HashSet::new();
        for _ in 0..8 {
            let _ = fs::remove_file(&path);
            let store = DiskStore {
                path: path.clone(),
                identity: "revision-test".into(),
                epoch: 0,
                loader: Default::default(),
            };
            DiskStore::at(path.clone(), "revision-test")?;
            assert!(seen.insert(store.revision()?), "revision reissued");
        }
        Ok(())
    }

    #[test]
    fn request_corruption_is_reported_for_repair() {
        take_corruption_report();
        let _ = status(rusqlite::Error::SqliteFailure(
            rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_CORRUPT),
            None,
        ));
        assert!(take_corruption_report());
        assert!(!take_corruption_report());
        let _ = status(rusqlite::Error::QueryReturnedNoRows);
        assert!(!take_corruption_report());
    }

    fn catalog_series(
        c: &Connection,
        run: &str,
        tag: &str,
        plugin: &str,
        max_length: i64,
        content: &[u8],
    ) -> Result<()> {
        c.execute("INSERT OR IGNORE INTO runs(name) VALUES(?1)", params![run])?;
        let class = match plugin {
            "scalars" => pb::DataClass::Scalar,
            "histograms" => pb::DataClass::Tensor,
            _ => pb::DataClass::BlobSequence,
        };
        let metadata = pb::SummaryMetadata {
            plugin_data: Some(pb::summary_metadata::PluginData {
                plugin_name: plugin.into(),
                content: Bytes::copy_from_slice(content),
            }),
            summary_description: format!("{}:{}", run, tag),
            data_class: class as i32,
            ..Default::default()
        };
        c.execute(
            "INSERT INTO series(run,tag,plugin,class,metadata,capacity,valid,max_length)
             VALUES(?1,?2,?3,?4,?5,-1,1,?6)",
            params![
                run,
                tag,
                plugin,
                class as i32,
                metadata.encode_to_vec(),
                max_length
            ],
        )?;
        Ok(())
    }

    fn catalog_run(experiment: &str, name: &str) -> data::MetricsCatalogRun {
        data::MetricsCatalogRun {
            experiment_id: experiment.into(),
            name: name.into(),
        }
    }

    #[test]
    fn metrics_catalog_unions_scalars_and_qualifies_run_cards_and_pins() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let store = DiskStore::at(dir.path().join("catalog.sqlite"), "catalog-scope")?;
        let c = store.connect()?;
        for run in ["train", "test"] {
            catalog_series(&c, run, "g2/shared", "scalars", 0, b"")?;
            catalog_series(&c, run, "g2/hist", "histograms", 0, b"{}")?;
        }
        catalog_series(&c, "train", "g10/pixels", "images", 5, b"")?;
        catalog_series(&c, "unselected", "g1/hidden", "scalars", 0, b"")?;
        catalog_series(&c, "pinned", "pin/photo", "images", 4, b"")?;
        let req = data::MetricsCatalogRequest {
            runs: vec![
                catalog_run("a", "train"),
                catalog_run("b", "train"),
                catalog_run("a", "test"),
            ],
            group_offset: 1,
            group_limit: 1,
            groups: vec![data::MetricsCatalogGroupRequest {
                name: "g2".into(),
                offset: 3,
                limit: 1,
            }],
            pinned_tags: vec!["pin/photo".into()],
            pinned_runs: vec![catalog_run("z", "pinned")],
            ..Default::default()
        };
        let result = store.metrics_catalog(req.clone())?;
        assert_eq!(
            (result.total_groups, result.total_cards, result.group_offset),
            (2, 10, 1)
        );
        assert_eq!(
            result.groups,
            vec![data::MetricsCatalogGroup {
                name: "g10".into(),
                total_cards: 6
            }]
        );
        assert_eq!(
            result.cards,
            vec![data::MetricCard {
                plugin: "scalars".into(),
                tag: "g2/shared".into(),
                ..Default::default()
            }]
        );
        assert_eq!(
            result
                .series
                .iter()
                .map(|s| (s.tag.as_str(), s.run_id.as_str(), s.max_samples))
                .collect::<Vec<_>>(),
            vec![
                ("g2/shared", "a/test", 0),
                ("g2/shared", "a/train", 0),
                ("g2/shared", "b/train", 0),
                ("pin/photo", "z/pinned", 2)
            ]
        );
        let filtered = store.metrics_catalog(data::MetricsCatalogRequest {
            query: "(?i)HIST".into(),
            filtered_offset: 1,
            filtered_limit: 1,
            plugins: vec!["histograms".into()],
            ..req
        })?;
        assert_eq!((filtered.total_groups, filtered.total_cards), (0, 3));
        assert!(filtered.groups.is_empty());
        assert_eq!(
            filtered.cards,
            vec![data::MetricCard {
                plugin: "histograms".into(),
                tag: "g2/hist".into(),
                run_id: "a/train".into(),
                ..Default::default()
            }]
        );
        assert_eq!(
            filtered
                .series
                .iter()
                .map(|s| s.run_id.as_str())
                .collect::<Vec<_>>(),
            vec!["a/train", "z/pinned"]
        );
        Ok(())
    }

    #[test]
    fn metrics_catalog_pages_image_spans_without_decoding_closed_metadata() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let store = DiskStore::at(dir.path().join("catalog.sqlite"), "catalog-spans")?;
        let c = store.connect()?;
        let samples = 1_000_000_000_000;
        catalog_series(&c, "train", "g2/pixels", "images", samples + 2, b"")?;
        catalog_series(&c, "train", "g10/closed", "scalars", 0, b"")?;
        // Corruption must not be decoded merely to count/list a closed group.
        c.execute(
            "UPDATE series SET metadata=x'ff' WHERE tag='g10/closed'",
            [],
        )?;
        let req = data::MetricsCatalogRequest {
            runs: vec![catalog_run("exp", "train")],
            group_limit: 2,
            groups: vec![data::MetricsCatalogGroupRequest {
                name: "g2".into(),
                offset: samples as u64 - 2,
                limit: 10,
            }],
            ..Default::default()
        };
        let result = store.metrics_catalog(req.clone())?;
        assert_eq!(result.total_cards, samples as u64 + 1);
        assert_eq!(
            result
                .groups
                .iter()
                .map(|g| g.name.as_str())
                .collect::<Vec<_>>(),
            vec!["g2", "g10"]
        );
        assert_eq!(
            result
                .cards
                .iter()
                .map(|c| (c.sample, c.num_sample))
                .collect::<Vec<_>>(),
            vec![
                (samples as u64 - 2, samples as u64),
                (samples as u64 - 1, samples as u64)
            ]
        );
        assert_eq!(
            result.series,
            vec![data::MetricSeries {
                plugin: "images".into(),
                tag: "g2/pixels".into(),
                run_id: "exp/train".into(),
                description: "train:g2/pixels".into(),
                max_samples: samples as u64,
            }]
        );
        let summary = store.metrics_catalog(data::MetricsCatalogRequest {
            groups: vec![],
            ..req
        })?;
        assert!(summary.cards.is_empty());
        assert!(summary.series.is_empty());
        Ok(())
    }

    #[test]
    fn metrics_catalog_counts_follow_validity_image_shape_and_plugin_versions() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("catalog.sqlite");
        let store = DiskStore::at(path.clone(), "catalog-shape")?;
        let c = store.connect()?;
        catalog_series(&c, "train", "g/image", "images", 5, b"")?;
        for plugin in ["scalars", "histograms", "images"] {
            catalog_series(
                &c,
                "train",
                &format!("future/{}", plugin),
                plugin,
                5,
                b"\x08\x01",
            )?;
        }
        catalog_series(&c, "train", "g/legacy", "histograms", 0, b"{}")?;
        let req = data::MetricsCatalogRequest {
            runs: vec![catalog_run("", "train")],
            group_limit: 10,
            ..Default::default()
        };
        assert_eq!(store.metrics_catalog(req.clone())?.total_cards, 4);
        c.execute("UPDATE series SET max_length=3 WHERE tag='g/image'", [])?;
        assert_eq!(store.metrics_catalog(req.clone())?.total_cards, 2);
        c.execute("UPDATE series SET valid=0 WHERE tag='g/legacy'", [])?;
        assert_eq!(store.metrics_catalog(req.clone())?.total_cards, 1);
        c.execute("UPDATE series SET valid=1 WHERE tag='g/legacy'", [])?;
        assert_eq!(store.metrics_catalog(req.clone())?.total_cards, 2);
        c.execute("DELETE FROM series WHERE tag='g/image'", [])?;
        drop(c);
        drop(store);
        let store = DiskStore::at(path, "catalog-shape")?;
        assert_eq!(store.metrics_catalog(req.clone())?.total_cards, 1);
        store
            .connect()?
            .execute("DELETE FROM runs WHERE name='train'", [])?;
        assert_eq!(store.metrics_catalog(req)?.total_groups, 0);
        Ok(())
    }
    fn scalar(step: i64, value: f32) -> pb::Event {
        pb::Event {
            step,
            wall_time: 1000.0 + step as f64,
            what: Some(pb::event::What::Summary(pb::Summary {
                value: vec![pb::summary::Value {
                    tag: "loss".into(),
                    value: Some(pb::summary::value::Value::SimpleValue(value)),
                    ..Default::default()
                }],
            })),
            ..Default::default()
        }
    }

    fn values(store: &DiskStore) -> Vec<(Step, f32)> {
        let filter = data::RunTagFilter {
            runs: Some(data::RunFilter {
                names: vec!["train".into()],
            }),
            tags: Some(data::TagFilter {
                names: vec!["loss".into()],
            }),
            ..Default::default()
        };
        let commit = store
            .read("scalars", pb::DataClass::Scalar, Some(&filter), usize::MAX)
            .unwrap();
        let runs = commit.runs.read().unwrap();
        let run = runs.get("train").unwrap().read().unwrap();
        run.scalars
            .get("loss")
            .unwrap()
            .valid_values()
            .map(|(s, _, v)| (s, v.0))
            .collect()
    }

    #[test]
    fn persisted_rng_matches_reference_across_preemption_and_reopen() -> Result<()> {
        let dir = tempfile::tempdir()?;
        for capacity in [
            Capacity::Bounded(0),
            Capacity::Bounded(1),
            Capacity::Bounded(17),
            Capacity::Unbounded,
        ] {
            let path = dir.path().join(format!("{:?}.sqlite", capacity));
            let hints =
                PluginSamplingHint([("scalars".into(), capacity)].iter().cloned().collect());
            let mut reference = StageReservoir::new(capacity);
            let mut basin = Basin::new();
            let mut store = DiskStore::at(path.clone(), "rng-test")?;
            store
                .connect()?
                .execute("INSERT INTO runs(name) VALUES('train')", [])?;
            // More than a ChaCha block, repeated steps, partial preemption and
            // complete restart. Reopening must preserve the RNG's *word* cursor,
            // including rejection draws performed by Uniform<usize>.
            // Transactions batch a varying number of events, so in-memory
            // reservoir state must match what a reload from the index gives.
            let c = store.connect()?;
            let mut tx = c.unchecked_transaction()?;
            let mut ingest = Ingest::new("train");
            for (i, step) in (0..400)
                .chain(150..700)
                .chain([699, 699, -1])
                .chain(0..500)
                .enumerate()
            {
                let value = i as f32;
                reference.offer(Step(step), value);
                reference.commit(&mut basin);
                ingest.event(&tx, scalar(step, value), &hints)?;
                if i % 7 == 0 || i % 79 == 0 || i % 11 == 3 {
                    ingest.flush(&tx)?;
                    tx.commit()?;
                    tx = c.unchecked_transaction()?;
                    ingest = Ingest::new("train");
                }
                if i % 79 == 0 {
                    drop(store);
                    store = DiskStore::at(path.clone(), "rng-test")?;
                    assert_eq!(values(&store), basin.as_slice().to_vec());
                }
            }
            ingest.flush(&tx)?;
            tx.commit()?;
            assert_eq!(values(&store), basin.as_slice().to_vec());
        }
        Ok(())
    }

    #[test]
    fn max_keeps_an_upper_bound_until_restored() {
        let mut m = Max::exact(None);
        m.insert(3);
        m.insert(1);
        assert_eq!(m, Max::exact(Some(3)));
        m.remove(1);
        assert_eq!(m, Max::exact(Some(3)));
        m.remove(3);
        assert_eq!((m.value, m.exact), (Some(3), false));
        m.insert(2);
        assert!(!m.exact, "2 may not be the greatest remaining value");
        m.insert(3);
        assert_eq!(m, Max::exact(Some(3)));
    }

    /// An event for `tag` whose value is valid unless `elements` is `None`:
    /// a blob sequence of that many elements for "pics", else a scalar.
    fn sample_event(tag: &str, step: i64, wall: f64, elements: Option<usize>) -> pb::Event {
        let pics = tag == "pics";
        let value = match (pics, elements) {
            (false, Some(_)) => pb::summary::value::Value::SimpleValue(step as f32),
            (true, Some(n)) => pb::summary::value::Value::Tensor(pb::TensorProto {
                dtype: pb::DataType::DtString as i32,
                tensor_shape: Some(pb::TensorShapeProto {
                    dim: vec![pb::tensor_shape_proto::Dim {
                        size: n as i64,
                        ..Default::default()
                    }],
                    ..Default::default()
                }),
                string_val: vec![Bytes::from_static(b"px"); n],
                ..Default::default()
            }),
            (_, None) => pb::summary::value::Value::Tensor(pb::TensorProto {
                dtype: pb::DataType::DtString as i32,
                string_val: vec![Bytes::from_static(b"invalid"); 2],
                ..Default::default()
            }),
        };
        let metadata = pics.then(|| pb::SummaryMetadata {
            plugin_data: Some(pb::summary_metadata::PluginData {
                plugin_name: "images".into(),
                ..Default::default()
            }),
            data_class: pb::DataClass::BlobSequence.into(),
            ..Default::default()
        });
        pb::Event {
            step,
            wall_time: wall,
            what: Some(pb::event::What::Summary(pb::Summary {
                value: vec![pb::summary::Value {
                    tag: tag.into(),
                    metadata,
                    value: Some(value),
                    ..Default::default()
                }],
            })),
            ..Default::default()
        }
    }

    #[test]
    fn tracked_extrema_match_stored_samples() -> Result<()> {
        use rand::Rng;
        let dir = tempfile::tempdir()?;
        let store = DiskStore::at(dir.path().join("index.sqlite"), "extrema-test")?;
        let c = store.connect()?;
        c.execute("INSERT INTO runs(name) VALUES('train')", [])?;
        let hints = PluginSamplingHint(
            [
                ("scalars".into(), Capacity::Bounded(6)),
                ("images".into(), Capacity::Bounded(4)),
            ]
            .iter()
            .cloned()
            .collect(),
        );
        let mut rng = ChaCha20Rng::seed_from_u64(7);
        let mut tx = c.unchecked_transaction()?;
        let mut ingest = Ingest::new("train");
        let mut step = 0i64;
        let mut lengths = HashSet::new();
        for _ in 0..4000 {
            // Mostly increasing steps with occasional preempting rewinds, and
            // wall times that are not monotonic.
            step = if rng.gen_ratio(1, 30) {
                step - rng.gen_range(0, 12)
            } else {
                step + 1
            };
            let tag = if rng.gen_bool(0.5) { "pics" } else { "loss" };
            let elements = (!rng.gen_ratio(1, 6)).then(|| rng.gen_range(0, 5));
            let wall = rng.gen_range(0.0, 50.0);
            ingest.event(&tx, sample_event(tag, step, wall, elements), &hints)?;
            if rng.gen_ratio(1, 9) {
                ingest.flush(&tx)?;
                tx.commit()?;
                let mut stored = c.prepare(
                    "SELECT s.tag,s.max_step,s.max_wall,s.max_length,s.valid,s.sample_count,
                        (SELECT max(step) FROM samples WHERE series=s.id AND value IS NOT NULL),
                        (SELECT max(wall) FROM samples WHERE series=s.id AND value IS NOT NULL),
                        (SELECT max(length) FROM samples WHERE series=s.id AND value IS NOT NULL),
                        (SELECT count(*) FROM samples WHERE series=s.id AND value IS NOT NULL),
                        (SELECT count(*) FROM samples WHERE series=s.id)
                    FROM series s",
                )?;
                let mut rows = stored.query([])?;
                while let Some(r) = rows.next()? {
                    let tag: String = r.get(0)?;
                    let tracked: (Option<i64>, Option<f64>, Option<i64>, i64, i64) =
                        (r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?);
                    let actual: (Option<i64>, Option<f64>, Option<i64>, i64, i64) =
                        (r.get(6)?, r.get(7)?, r.get(8)?, r.get(9)?, r.get(10)?);
                    assert_eq!(tracked, actual, "series {}", tag);
                    lengths.extend(tracked.2);
                }
                tx = c.unchecked_transaction()?;
                if rng.gen_bool(0.5) {
                    ingest = Ingest::new("train");
                }
            }
        }
        assert!(
            lengths.len() > 1,
            "blob lengths never varied: {:?}",
            lengths
        );
        Ok(())
    }

    #[test]
    fn reload_resumes_truncated_records_and_keeps_idle_revision() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let logs = dir.path().join("logs");
        fs::create_dir_all(logs.join("train"))?;
        let path = logs.join("train/tfevents.1");
        let mut event = Vec::new();
        event.write_scalar(
            &Tag("loss".into()),
            Step(0),
            WallTime::new(123.0).unwrap(),
            1.0,
        )?;
        let mut file = File::create(&path)?;
        file.write_all(&event[..event.len() - 3])?;
        file.flush()?;
        let index = dir.path().join("index.sqlite");
        let store = DiskStore::at(index.clone(), "append-test")?;
        let hints = PluginSamplingHint::default();
        let logdir = DiskLogdir::new(logs.clone());
        store.reload(&logdir, &hints, true)?;
        assert!(store
            .listing("scalars", pb::DataClass::Scalar, None)?
            .rows
            .is_empty());
        drop(store);
        file.write_all(&event[event.len() - 3..])?;
        file.flush()?;
        let store = DiskStore::at(index, "append-test")?;
        store.reload(&logdir, &hints, true)?;
        assert_eq!(values(&store), vec![(Step(0), 1.0)]);
        let revision = store.revision()?;
        store.reload(&logdir, &hints, true)?;
        assert_eq!(store.revision()?, revision);
        let mut file = OpenOptions::new().append(true).open(&path)?;
        file.write_scalar(
            &Tag("loss".into()),
            Step(1),
            WallTime::new(124.0).unwrap(),
            2.0,
        )?;
        file.flush()?;
        store.reload(&logdir, &hints, true)?;
        assert_eq!(values(&store), vec![(Step(0), 1.0), (Step(1), 2.0)]);
        assert_eq!(store.revision()?, revision);
        let mut replacement = File::create(&path)?;
        replacement.write_scalar(
            &Tag("loss".into()),
            Step(0),
            WallTime::new(125.0).unwrap(),
            9.0,
        )?;
        replacement.flush()?;
        store.reload(&logdir, &hints, true)?;
        assert_eq!(values(&store), vec![(Step(0), 9.0)]);
        fs::remove_file(path)?;
        store.reload(&logdir, &hints, true)?;
        assert!(store
            .list_runs(data::ListRunsRequest::default())?
            .runs
            .is_empty());
        assert_ne!(store.revision()?, revision);
        Ok(())
    }

    #[test]
    fn distinct_tag_pages_intersect_selection_and_do_not_expand_empty_runs() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let store = DiskStore::at(dir.path().join("index.sqlite"), "selection-test")?;
        let mut c = store.connect()?;
        let tx = c.transaction()?;
        for (run, tags) in [
            ("train", vec!["a", "b", "c"]),
            ("eval", vec!["b", "d"]),
            ("unselected", vec!["secret"]),
        ] {
            tx.execute("INSERT INTO runs(name) VALUES(?1)", params![run])?;
            for tag in tags {
                let mut event = scalar(1, 42.0);
                if let Some(pb::event::What::Summary(s)) = &mut event.what {
                    s.value[0].tag = tag.into();
                }
                index_event(&tx, run, event, &PluginSamplingHint::default())?;
            }
        }
        tx.commit()?;
        let mut filter = data::RunTagFilter {
            runs: Some(data::RunFilter {
                names: vec!["train".into(), "eval".into()],
            }),
            tag_offset: 1,
            tag_limit: 1,
            ..Default::default()
        };
        let result = store.listing("scalars", pb::DataClass::Scalar, Some(&filter))?;
        assert_eq!(result.total, 4);
        assert_eq!(
            result
                .rows
                .iter()
                .map(|r| (r.run.as_str(), r.tag.as_str()))
                .collect::<Vec<_>>(),
            vec![("eval", "b"), ("train", "b")]
        );
        filter.tag_query = "^[bc]$".into();
        filter.tag_offset = 0;
        filter.tags = Some(data::TagFilter {
            names: vec!["c".into(), "secret".into()],
        });
        let result = store.listing("scalars", pb::DataClass::Scalar, Some(&filter))?;
        assert_eq!(result.total, 1);
        assert_eq!(result.rows[0].tag, "c");
        filter.runs = Some(data::RunFilter { names: Vec::new() });
        assert_eq!(
            store
                .listing("scalars", pb::DataClass::Scalar, Some(&filter))?
                .total,
            0
        );
        assert!(store
            .read("scalars", pb::DataClass::Scalar, Some(&filter), 1000)?
            .runs
            .read()
            .unwrap()
            .is_empty());
        filter.tag_query = "[".into();
        assert_eq!(
            store
                .listing("scalars", pb::DataClass::Scalar, Some(&filter))
                .err()
                .unwrap()
                .code(),
            tonic::Code::InvalidArgument
        );
        let result = store.list_runs(data::ListRunsRequest {
            query: "^alias/eval$".into(),
            query_prefix: "alias".into(),
            limit: 1,
            ..Default::default()
        })?;
        assert_eq!(result.total, 1);
        assert_eq!(result.runs[0].name, "eval");
        Ok(())
    }

    #[tokio::test]
    async fn run_windows_apply_session_ranks_before_count_and_pagination() -> Result<()> {
        use crate::commit::test_data::CommitBuilder;
        use data::tensor_board_data_provider_server::TensorBoardDataProvider;

        let dir = tempfile::tempdir()?;
        let store = DiskStore::at(dir.path().join("index.sqlite"), "run-ranks")?;
        let c = store.connect()?;
        let mut builder = CommitBuilder::new();
        for (name, start) in [
            ("alpha", Some(20.0)),
            ("alpha/eval", Some(10.0)),
            ("alphabet", Some(30.0)),
            ("alpha/eval/child", Some(40.0)),
            ("alpha/evalish", Some(50.0)),
            ("beta", Some(60.0)),
            ("other", Some(70.0)),
            ("αtrial", Some(80.0)),
            ("alpha/unstarted", None),
        ] {
            c.execute(
                "INSERT INTO runs(name,start) VALUES(?1,?2)",
                params![name, start],
            )?;
            builder = builder.run(name, start);
        }
        let handler = crate::server::DataProviderHandler {
            data_location: String::new(),
            commit: std::sync::Arc::new(builder.build()),
        };
        let entries = |values: &[(&str, i64)]| {
            values
                .iter()
                .map(|(prefix, rank)| data::RunSessionRank {
                    prefix: (*prefix).into(),
                    rank: *rank,
                })
                .collect()
        };
        let ranked = data::ListRunsRequest {
            sort_by: "session_rank".into(),
            session_ranks: entries(&[
                ("alpha", 4),
                ("alpha/eval", -1),
                ("alpha/eval/child", 1),
                ("beta", -1),
                ("beta", 1),
                ("α", 2),
            ]),
            default_rank: -1,
            ..Default::default()
        };
        let cases = vec![
            (
                "compact ranked page",
                data::ListRunsRequest {
                    offset: 1,
                    limit: 3,
                    ..ranked.clone()
                },
                5,
                vec!["beta", "αtrial", "alpha"],
            ),
            (
                "descending rank and name",
                data::ListRunsRequest {
                    descending: true,
                    ..ranked.clone()
                },
                5,
                vec!["alphabet", "alpha", "αtrial", "beta", "alpha/eval/child"],
            ),
            (
                "name sorting still filters",
                data::ListRunsRequest {
                    sort_by: "name".into(),
                    limit: 2,
                    ..ranked.clone()
                },
                5,
                vec!["alpha", "alpha/eval/child"],
            ),
            (
                "start time sorting still filters",
                data::ListRunsRequest {
                    sort_by: String::new(),
                    descending: true,
                    offset: 1,
                    limit: 2,
                    ..ranked.clone()
                },
                5,
                vec!["beta", "alpha/eval/child"],
            ),
            (
                "names and aliased query intersect ranks",
                data::ListRunsRequest {
                    query: "^compare/(alpha|beta)".into(),
                    query_prefix: "compare".into(),
                    names: Some(data::RunFilter {
                        names: vec![
                            "alpha/eval/child".into(),
                            "beta".into(),
                            "alphabet".into(),
                            "other".into(),
                            "beta".into(),
                        ],
                    }),
                    ..ranked.clone()
                },
                3,
                vec!["alpha/eval/child", "beta", "alphabet"],
            ),
            (
                "alias itself matches",
                data::ListRunsRequest {
                    query: "^compare$".into(),
                    query_prefix: "compare".into(),
                    limit: 1,
                    ..ranked.clone()
                },
                5,
                vec!["alpha/eval/child"],
            ),
            (
                "explicit empty names",
                data::ListRunsRequest {
                    names: Some(data::RunFilter { names: vec![] }),
                    ..ranked.clone()
                },
                0,
                vec![],
            ),
            (
                "past final page preserves total",
                data::ListRunsRequest {
                    offset: 5,
                    limit: 2,
                    ..ranked.clone()
                },
                5,
                vec![],
            ),
            (
                "positive default includes unmatched runs",
                data::ListRunsRequest {
                    session_ranks: entries(&[("alpha", -1), ("alpha/eval", 0)]),
                    default_rank: 3,
                    ..ranked.clone()
                },
                6,
                vec![
                    "alpha/eval",
                    "alpha/eval/child",
                    "alpha/evalish",
                    "beta",
                    "other",
                    "αtrial",
                ],
            ),
            (
                "empty prefix overridden by longer prefixes",
                data::ListRunsRequest {
                    session_ranks: entries(&[
                        ("", -1),
                        ("alpha", 2),
                        ("alpha/eval", -1),
                        ("alpha/eval/child", 0),
                    ]),
                    default_rank: 7,
                    ..ranked.clone()
                },
                3,
                vec!["alpha/eval/child", "alpha", "alphabet"],
            ),
            (
                "negative default with no prefixes",
                data::ListRunsRequest {
                    session_ranks: vec![],
                    ..ranked.clone()
                },
                0,
                vec![],
            ),
            (
                "constant rank orders by name",
                data::ListRunsRequest {
                    session_ranks: vec![],
                    default_rank: 0,
                    limit: 2,
                    ..ranked.clone()
                },
                8,
                vec!["alpha", "alpha/eval"],
            ),
        ];
        for (case, request, total, names) in cases {
            let disk = store.list_runs(request.clone())?;
            assert_eq!(disk.total, total, "{}", case);
            assert_eq!(
                disk.runs
                    .iter()
                    .map(|run| run.name.as_str())
                    .collect::<Vec<_>>(),
                names,
                "{}",
                case
            );
            let memory = handler
                .list_runs(tonic::Request::new(request))
                .await?
                .into_inner();
            assert_eq!(memory, disk, "in-memory/disk parity: {}", case);
        }
        for request in [
            data::ListRunsRequest {
                query: "[".into(),
                ..ranked.clone()
            },
            data::ListRunsRequest {
                sort_by: "unknown".into(),
                ..ranked.clone()
            },
            data::ListRunsRequest {
                offset: u64::MAX,
                ..ranked
            },
        ] {
            assert_eq!(
                store.list_runs(request.clone()).unwrap_err().code(),
                tonic::Code::InvalidArgument
            );
            assert_eq!(
                handler
                    .list_runs(tonic::Request::new(request))
                    .await
                    .unwrap_err()
                    .code(),
                tonic::Code::InvalidArgument
            );
        }
        Ok(())
    }

    #[test]
    fn single_run_counts_follow_atomic_validity_transitions() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let store = DiskStore::at(dir.path().join("index.sqlite"), "count-test")?;
        let filter = data::RunTagFilter {
            // Repeated names still denote one run, not multiple series copies.
            runs: Some(data::RunFilter {
                names: vec!["train".into(), "train".into()],
            }),
            tag_limit: 1,
            ..Default::default()
        };
        let listing = || store.listing("scalars", pb::DataClass::Scalar, Some(&filter));
        let mut c = store.connect()?;
        let tx = c.transaction()?;
        tx.execute("INSERT INTO runs(name) VALUES('train')", [])?;
        index_event(&tx, "train", scalar(1, 1.0), &PluginSamplingHint::default())?;
        // A batched source transaction publishes neither a count nor a partial
        // catalog before committing the corresponding event data.
        assert_eq!(listing()?.total, 0);
        tx.commit()?;
        let visible = listing()?;
        assert_eq!(visible.total, 1);
        assert_eq!(visible.rows[0].tag, "loss");

        let mut invalid = scalar(1, 1.0);
        if let Some(pb::event::What::Summary(summary)) = &mut invalid.what {
            summary.value[0].value = Some(pb::summary::value::Value::Tensor(pb::TensorProto {
                dtype: pb::DataType::DtString as i32,
                string_val: vec![Bytes::from_static(b"not a scalar")],
                ..Default::default()
            }));
        }
        let tx = c.transaction()?;
        index_event(&tx, "train", invalid, &PluginSamplingHint::default())?;
        assert_eq!(listing()?.total, 1);
        tx.commit()?;
        let visible = listing()?;
        assert_eq!(visible.total, 0);
        assert!(visible.rows.is_empty());

        let tx = c.transaction()?;
        index_event(&tx, "train", scalar(1, 2.0), &PluginSamplingHint::default())?;
        tx.commit()?;
        assert_eq!(listing()?.total, 1);
        c.execute("DELETE FROM runs WHERE name='train'", [])?;
        assert_eq!(listing()?.total, 0);
        Ok(())
    }

    #[test]
    fn blobs_are_queryable_after_restart_and_preemption_invalidates_shape() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("index.sqlite");
        let store = DiskStore::at(path.clone(), "blob-test")?;
        let mut c = store.connect()?;
        c.execute("INSERT INTO runs(name) VALUES('train')", [])?;
        let event = pb::Event {
            step: 1,
            wall_time: 1.0,
            what: Some(pb::event::What::GraphDef(Bytes::from_static(b"graph-one"))),
            ..Default::default()
        };
        let tx = c.transaction()?;
        index_event(&tx, "train", event, &PluginSamplingHint::default())?;
        tx.commit()?;
        let revision = store.revision()?;
        drop(c);
        drop(store);
        let store = DiskStore::at(path, "blob-test")?;
        assert_ne!(store.revision()?, revision);
        let revision = store.revision()?;
        let key = crate::blob_key::BlobKey {
            experiment_id: "".into(),
            run: "train".into(),
            tag: GraphDefValue::TAG_NAME.into(),
            step: Step(1),
            index: 0,
        };
        assert_eq!(store.blob(&key)?, Bytes::from_static(b"graph-one"));
        let references = store.blob_sequences("graphs", "", None, 1)?;
        let returned_key: crate::blob_key::BlobKey =
            references.runs[0].tags[0].data.as_ref().unwrap().values[0].blob_refs[0]
                .blob_key
                .parse()?;
        assert_eq!(store.blob(&returned_key)?, Bytes::from_static(b"graph-one"));
        let mut c = store.connect()?;
        let tx = c.transaction()?;
        index_event(
            &tx,
            "train",
            pb::Event {
                step: 0,
                wall_time: 2.0,
                what: Some(pb::event::What::GraphDef(Bytes::from_static(
                    b"replacement",
                ))),
                ..Default::default()
            },
            &PluginSamplingHint::default(),
        )?;
        tx.commit()?;
        assert_eq!(
            store.blob(&key).err().unwrap().code(),
            tonic::Code::NotFound
        );
        // Replacing a valid graph with a valid graph preserves metadata shape.
        assert_eq!(store.revision()?, revision);
        assert_eq!(
            store.plugins()?.plugins,
            vec![data::Plugin {
                name: "graphs".into()
            }]
        );
        Ok(())
    }

    #[test]
    fn transient_source_error_resumes_committed_offset_after_restart() -> Result<()> {
        use std::cell::Cell;
        use std::io::{Cursor, Read};
        struct Source {
            bytes: Vec<u8>,
            boundary: u64,
            failed: Cell<bool>,
        }
        struct Reader {
            cursor: Cursor<Vec<u8>>,
            fail_at: Option<u64>,
        }
        impl Read for Reader {
            fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
                if let Some(boundary) = self.fail_at {
                    if self.cursor.position() >= boundary {
                        return Err(io::Error::new(
                            io::ErrorKind::TimedOut,
                            "transient source timeout",
                        ));
                    }
                    let remaining = (boundary - self.cursor.position()) as usize;
                    let n = remaining.min(bytes.len());
                    return self.cursor.read(&mut bytes[..n]);
                }
                self.cursor.read(bytes)
            }
        }
        impl Logdir for Source {
            type File = Reader;
            fn discover(&self) -> io::Result<std::collections::HashMap<Run, Vec<EventFileBuf>>> {
                Ok(
                    [(Run("train".into()), vec![EventFileBuf("tfevents".into())])]
                        .iter()
                        .cloned()
                        .collect(),
                )
            }
            fn open(&self, _: &EventFileBuf) -> io::Result<Reader> {
                Ok(Reader {
                    cursor: Cursor::new(self.bytes.clone()),
                    fail_at: if self.failed.replace(true) {
                        None
                    } else {
                        Some(self.boundary)
                    },
                })
            }
        }
        let mut bytes = Vec::new();
        bytes.write_scalar(
            &Tag("loss".into()),
            Step(0),
            WallTime::new(1.0).unwrap(),
            1.0,
        )?;
        let boundary = bytes.len() as u64;
        bytes.write_scalar(
            &Tag("loss".into()),
            Step(1),
            WallTime::new(2.0).unwrap(),
            2.0,
        )?;
        let source = Source {
            bytes,
            boundary,
            failed: Cell::new(false),
        };
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("index.sqlite");
        let store = DiskStore::at(path.clone(), "transient-test")?;
        assert!(store
            .reload(&source, &PluginSamplingHint::default(), true)
            .is_err());
        assert_eq!(values(&store), vec![(Step(0), 1.0)]);
        drop(store);
        let store = DiskStore::at(path, "transient-test")?;
        store.reload(&source, &PluginSamplingHint::default(), true)?;
        assert_eq!(values(&store), vec![(Step(0), 1.0), (Step(1), 2.0)]);
        Ok(())
    }
}
