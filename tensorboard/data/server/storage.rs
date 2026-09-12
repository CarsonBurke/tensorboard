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
use rusqlite::{params, Connection, OptionalExtension};
use std::borrow::Borrow;
use std::cmp::Ordering;
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

#[derive(Debug)]
pub struct DiskStore {
    path: PathBuf,
    epoch: u128,
}

fn status(e: impl std::fmt::Display) -> Status {
    log::error!("Data index: {}", e);
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
            epoch: rand::random(),
        };
        let c = store.connect()?;
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
            CREATE TABLE IF NOT EXISTS samples(series INTEGER NOT NULL REFERENCES series(id) ON DELETE CASCADE,
                step INTEGER NOT NULL, wall REAL NOT NULL, value BLOB, length INTEGER, PRIMARY KEY(series,step)) WITHOUT ROWID;
            CREATE INDEX IF NOT EXISTS samples_valid_step ON samples(series,step) WHERE value IS NOT NULL;
            CREATE INDEX IF NOT EXISTS samples_valid_wall ON samples(series,wall) WHERE value IS NOT NULL;
            CREATE INDEX IF NOT EXISTS samples_valid_length ON samples(series,length) WHERE value IS NOT NULL;
            CREATE TABLE IF NOT EXISTS blobs(series INTEGER NOT NULL, step INTEGER NOT NULL, idx INTEGER NOT NULL, value BLOB NOT NULL,
                PRIMARY KEY(series,step,idx), FOREIGN KEY(series,step) REFERENCES samples(series,step) ON DELETE CASCADE) WITHOUT ROWID;
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
        c.execute(
            "INSERT INTO info SELECT ?1,0 WHERE NOT EXISTS(SELECT 1 FROM info)",
            params![identity],
        )?;
        let actual: String = c.query_row("SELECT identity FROM info", [], |r| r.get(0))?;
        if actual != identity {
            return Err("data index identity mismatch".into());
        }
        Self::migrate_catalog_counts(&c)?;
        Self::migrate_metrics_catalog(&c)?;
        Ok(store)
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

    fn connect(&self) -> rusqlite::Result<Connection> {
        // Every connection, including loader writers, must know index collations.
        let c = Connection::open(&self.path)?;
        c.create_collation("TAG_NAME", compare_tag_names)?;
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

    pub fn reload(
        &self,
        logdir: &impl Logdir,
        hints: &PluginSamplingHint,
        checksum: bool,
    ) -> Result<()> {
        // A separate SQLite database supplies a crash-released process lock.
        // Holding it across discovery never locks the data database or readers.
        let lock = Connection::open(self.path.with_extension("reload.sqlite"))?;
        lock.busy_handler(Some(|_| {
            std::thread::sleep(std::time::Duration::from_millis(10));
            true
        }))?;
        lock.execute_batch("BEGIN EXCLUSIVE")?;
        let mut c = self.connect()?;
        c.execute_batch("UPDATE files SET present=0; CREATE TEMP TABLE discovered(path BLOB PRIMARY KEY, run TEXT, identity TEXT, modified TEXT, length INTEGER); CREATE INDEX discovered_run ON discovered(run,path);")?;
        logdir.visit(&mut |run, file, fingerprint| {
            let path = encode_path(&file.0);
            let (identity, modified, len) = match fingerprint {
                Some(FileFingerprint {
                    identity,
                    modified,
                    len,
                }) => (Some(identity), Some(modified), Some(len as i64)),
                None => (None, None, None),
            };
            let tx = c.transaction().map_err(io_error)?;
            tx.execute(
                "INSERT OR IGNORE INTO runs(name) VALUES(?1)",
                params![run.0],
            )
            .map_err(io_error)?;
            tx.execute(
                "INSERT INTO discovered VALUES(?1,?2,?3,?4,?5)",
                params![path, run.0, identity, modified, len],
            )
            .map_err(io_error)?;
            tx.commit().map_err(io_error)?;
            Ok(())
        })?;
        // A failed discovery returns above without deleting any data. Replaced
        // or shortened files require replay of their run in canonical file order.
        c.execute_batch("BEGIN IMMEDIATE;
            CREATE TEMP TABLE reset_runs(name TEXT PRIMARY KEY);
            INSERT OR IGNORE INTO reset_runs SELECT f.run FROM files f JOIN discovered d USING(path)
                WHERE (f.identity IS NOT NULL AND d.identity IS NOT f.identity) OR d.length < f.offset
                    OR (d.modified IS NOT f.modified AND d.length<=f.length);
            DELETE FROM series WHERE run IN (SELECT name FROM reset_runs);
            UPDATE runs SET start=NULL WHERE name IN (SELECT name FROM reset_runs);
            UPDATE files SET offset=0,dead=0 WHERE run IN (SELECT name FROM reset_runs);
            INSERT INTO files(path,run,identity,modified,length,present)
                SELECT path,run,identity,modified,length,1 FROM discovered WHERE true
                ON CONFLICT(path) DO UPDATE SET present=1;
            DELETE FROM runs WHERE NOT EXISTS(SELECT 1 FROM discovered d WHERE d.run=runs.name);
            COMMIT;")?;
        // Keyset traversal retains one path, not a Vec of files or run loaders.
        let mut previous = Vec::<u8>::new();
        loop {
            let next = c
                .query_row(
                    "SELECT f.path,f.run,f.offset,f.dead,d.identity,d.modified,d.length,
                    f.identity,f.modified,f.length FROM files f JOIN discovered d USING(path)
                    WHERE f.path>?1 ORDER BY f.path LIMIT 1",
                    params![previous],
                    |r| {
                        Ok((
                            r.get::<_, Vec<u8>>(0)?,
                            r.get::<_, String>(1)?,
                            r.get::<_, i64>(2)?,
                            r.get::<_, bool>(3)?,
                            r.get::<_, Option<String>>(4)?,
                            r.get::<_, Option<String>>(5)?,
                            r.get::<_, Option<i64>>(6)?,
                            r.get::<_, Option<String>>(7)?,
                            r.get::<_, Option<String>>(8)?,
                            r.get::<_, Option<i64>>(9)?,
                        ))
                    },
                )
                .optional()?;
            let (
                path,
                run,
                offset,
                dead,
                identity,
                modified,
                length,
                old_identity,
                old_modified,
                old_length,
            ) = match next {
                Some(x) => x,
                None => break,
            };
            previous = path.clone();
            if dead
                || (offset != 0
                    && identity.is_some()
                    && identity == old_identity
                    && modified == old_modified
                    && length == old_length)
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
                index_event(&tx, &run, event, hints)?;
                tx.prepare_cached("UPDATE files SET offset=?2 WHERE path=?1")?
                    .execute(params![path, offset])?;
                if last_commit.elapsed() >= crate::run::COMMIT_INTERVAL {
                    tx.commit()?;
                    tx = c.unchecked_transaction()?;
                    last_commit = Instant::now();
                }
            }
            tx.commit()?;
            c.execute(
                "UPDATE files SET identity=?2,modified=?3,length=?4 WHERE path=?1",
                params![path, identity, modified, length],
            )?;
        }
        // Keep missing-file tombstones while their run exists, matching RunLoader:
        // removed files must not be re-opened on a subsequent reload.
        c.execute("UPDATE files SET dead=1 WHERE present=0", [])?;
        Ok(())
    }
}

fn io_error(e: rusqlite::Error) -> io::Error {
    io::Error::new(io::ErrorKind::Other, e)
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

fn index_event(
    c: &Connection,
    run: &str,
    event: pb::Event,
    hints: &PluginSamplingHint,
) -> Result<()> {
    if WallTime::new(event.wall_time).is_none() {
        return Ok(());
    }
    c.prepare_cached("UPDATE runs SET start=?2 WHERE name=?1 AND (start IS NULL OR start>?2)")?
        .execute(params![run, event.wall_time])?;
    let step = event.step;
    let wall = event.wall_time;
    match event.what {
        Some(pb::event::What::GraphDef(bytes)) => offer(
            c,
            run,
            GraphDefValue::TAG_NAME,
            step,
            wall,
            EventValue::GraphDef(GraphDefValue(bytes)),
            None,
            hints,
        )?,
        Some(pb::event::What::TaggedRunMetadata(v)) => offer(
            c,
            run,
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
                offer(
                    c,
                    run,
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

fn offer(
    c: &Connection,
    run: &str,
    tag: &str,
    step: i64,
    wall: f64,
    value: EventValue,
    initial: Option<pb::SummaryMetadata>,
    hints: &PluginSamplingHint,
) -> Result<()> {
    let existing = c.prepare_cached("SELECT id,metadata,capacity,seen,rng,sample_count,valid FROM series WHERE run=?1 AND tag=?2")?.query_row(params![run,tag], |r| Ok((r.get::<_,i64>(0)?,r.get::<_,Vec<u8>>(1)?,r.get::<_,i64>(2)?,r.get::<_,i64>(3)?,r.get::<_,String>(4)?,r.get::<_,i64>(5)?,r.get::<_,i64>(6)?))).optional()?;
    let (id, metadata, capacity, mut seen, position, old_len, mut valid) = match existing {
        Some((id, bytes, capacity, seen, position, len, valid)) => (
            id,
            pb::SummaryMetadata::decode(bytes.as_slice())?,
            capacity,
            seen,
            position,
            len,
            valid,
        ),
        None => {
            let metadata = match &value {
                EventValue::GraphDef(_) => GraphDefValue::initial_metadata(),
                EventValue::TaggedRunMetadata(_) => TaggedRunMetadataValue::initial_metadata(),
                EventValue::Summary(v) => v.initial_metadata(initial),
            };
            let capacity = crate::run::StageTimeSeries::capacity(&metadata, hints);
            let capacity = match capacity {
                Capacity::Unbounded => -1,
                Capacity::Bounded(n) => n.try_into()?,
            };
            let plugin = metadata
                .plugin_data
                .as_ref()
                .map(|p| p.plugin_name.as_str())
                .unwrap_or("");
            c.prepare_cached("INSERT INTO series(run,tag,plugin,class,metadata,capacity) VALUES(?1,?2,?3,?4,?5,?6)")?.execute(params![run,tag,plugin,metadata.data_class,metadata.encode_to_vec(),capacity])?;
            (
                c.last_insert_rowid(),
                *metadata,
                capacity,
                0,
                "0".to_owned(),
                0,
                0,
            )
        }
    };
    if capacity == 0 {
        return Ok(());
    }
    let removed_valid: i64 = c
        .prepare_cached(
            "SELECT count(*) FROM samples WHERE series=?1 AND step>=?2 AND value IS NOT NULL",
        )?
        .query_row(params![id, step], |r| r.get(0))?;
    valid -= removed_valid;
    let preempted = c
        .prepare_cached("DELETE FROM samples WHERE series=?1 AND step>=?2")?
        .execute(params![id, step])? as i64;
    let mut len = old_len - preempted;
    if preempted != 0 {
        seen = ((seen as u64 * len as u64) / old_len as u64) as i64;
    }
    seen += 1;
    let mut rng = ChaCha20Rng::seed_from_u64(0);
    rng.set_word_pos(position.parse()?);
    if capacity >= 0 && seen > capacity {
        let destination = rng.destination(seen as usize) as i64;
        let evict = if destination >= capacity {
            Some(len - 1)
        } else if len >= capacity {
            Some(destination)
        } else {
            None
        };
        if let Some(index) = evict {
            if index >= 0 {
                let (evicted_step,evicted_valid): (i64,bool) = c.prepare_cached("SELECT step,value IS NOT NULL FROM samples WHERE series=?1 ORDER BY step LIMIT 1 OFFSET ?2")?.query_row(params![id,index],|r|Ok((r.get(0)?,r.get(1)?)))?;
                c.prepare_cached("DELETE FROM samples WHERE series=?1 AND step=?2")?
                    .execute(params![id, evicted_step])?;
                valid -= i64::from(evicted_valid);
                len -= 1;
            }
        }
    }
    let (payload, blobs, length): (Option<Vec<u8>>, Option<BlobSequenceValue>, Option<i64>) =
        match pb::DataClass::from_i32(metadata.data_class) {
            Some(pb::DataClass::Scalar) => (
                value.into_scalar().ok().map(|v| v.0.to_le_bytes().to_vec()),
                None,
                None,
            ),
            Some(pb::DataClass::Tensor) => (
                value.into_tensor(&metadata).ok().map(|v| v.encode_to_vec()),
                None,
                None,
            ),
            Some(pb::DataClass::BlobSequence) => {
                let blobs = value.into_blob_sequence(&metadata).ok();
                let length = blobs.as_ref().map(|v| v.0.len() as i64);
                (length.map(|_| Vec::new()), blobs, length)
            }
            _ => (None, None, None),
        };
    valid += i64::from(payload.is_some());
    c.prepare_cached("INSERT INTO samples VALUES(?1,?2,?3,?4,?5)")?
        .execute(params![id, step, wall, payload, length])?;
    if let Some(blobs) = blobs {
        for (index, blob) in blobs.0.into_iter().enumerate() {
            c.prepare_cached("INSERT INTO blobs VALUES(?1,?2,?3,?4)")?
                .execute(params![id, step, index as i64, blob.as_ref()])?;
        }
    }
    c.prepare_cached("UPDATE series SET seen=?2,rng=?3,valid=?4,sample_count=?5,
        max_step=(SELECT max(step) FROM samples WHERE series=?1 AND value IS NOT NULL),
        max_wall=(SELECT max(wall) FROM samples WHERE series=?1 AND value IS NOT NULL),
        max_length=(SELECT max(length) FROM samples WHERE series=?1 AND value IS NOT NULL) WHERE id=?1")?
        .execute(params![id,seen,rng.get_word_pos().to_string(),valid,len+1])?;
    Ok(())
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
        CREATE INDEX metrics_catalog_scope ON metrics_catalog(run,plugin,category,tag COLLATE TAG_NAME,cards);
        CREATE UNIQUE INDEX metrics_catalog_tag ON metrics_catalog(run,tag,plugin);
        CREATE TABLE metrics_catalog_groups(run TEXT NOT NULL,plugin TEXT NOT NULL,
            category TEXT NOT NULL,cards INTEGER NOT NULL,
            PRIMARY KEY(run,plugin,category)) WITHOUT ROWID;
    ")
}

/// Frontend compareTagNames: numeric chunks (including decimals/exponents),
/// slash components before ordinary characters, and UTF-16 character order.
/// Lexical ties make equivalent numeric spellings deterministic.
pub(crate) fn compare_tag_names(a: &str, b: &str) -> Ordering {
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
                sum(cards) OVER (ORDER BY tag COLLATE TAG_NAME,rank,run_id ROWS UNBOUNDED PRECEDING) AS end
            FROM spans)
         SELECT plugin,tag,run_id,cards,max(0,?2-(end-cards)),min(cards,?3-(end-cards))
         FROM positions WHERE end>?2 AND end-cards<?3
         ORDER BY tag COLLATE TAG_NAME,
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
            "SELECT category,cards FROM catalog_totals ORDER BY category COLLATE TAG_NAME LIMIT ?1 OFFSET ?2"
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
         ORDER BY m.tag COLLATE TAG_NAME,
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
    c.create_collation("TAG_NAME", compare_tag_names)
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
        let mut runs = result.runs.write().map_err(status)?;
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
            let mut run = run.write().map_err(status)?;
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
            for (i, step) in (0..400)
                .chain(150..700)
                .chain([699, 699, -1])
                .chain(0..500)
                .enumerate()
            {
                let value = i as f32;
                reference.offer(Step(step), value);
                reference.commit(&mut basin);
                let mut c = store.connect()?;
                let tx = c.transaction()?;
                index_event(&tx, "train", scalar(step, value), &hints)?;
                tx.commit()?;
                if i % 79 == 0 {
                    drop(store);
                    store = DiskStore::at(path.clone(), "rng-test")?;
                    assert_eq!(values(&store), basin.as_slice().to_vec());
                }
            }
            assert_eq!(values(&store), basin.as_slice().to_vec());
        }
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
