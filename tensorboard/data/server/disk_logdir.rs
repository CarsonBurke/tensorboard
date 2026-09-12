/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

//! Log directories on local disk.

use log::{info, warn};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

use crate::logdir::{EventFileBuf, FileFingerprint, Logdir, EVENT_FILE_BASENAME_INFIX};
use crate::types::Run;

/// A log directory on local disk.
pub struct DiskLogdir {
    root: PathBuf,
}

impl DiskLogdir {
    /// Creates a `DiskLogdir` with the given root directory.
    pub fn new(root: PathBuf) -> Self {
        DiskLogdir { root }
    }
}

impl Logdir for DiskLogdir {
    type File = BufReader<File>;

    fn discover(&self) -> io::Result<HashMap<Run, Vec<EventFileBuf>>> {
        let mut run_map: HashMap<Run, Vec<EventFileBuf>> = HashMap::new();
        self.visit(&mut |run, file, _| {
            run_map.entry(run).or_default().push(file);
            Ok(())
        })?;
        for files in run_map.values_mut() {
            files.sort();
        }
        Ok(run_map)
    }

    fn visit(
        &self,
        visitor: &mut dyn FnMut(Run, EventFileBuf, Option<FileFingerprint>) -> io::Result<()>,
    ) -> io::Result<()> {
        // Sorting buffers entire directories. A descriptor limit also makes WalkDir buffer
        // unvisited entries when descending, so retain only one open iterator per depth.
        let walker = WalkDir::new(&self.root)
            .max_open(usize::MAX)
            .follow_links(true);
        for walkdir_item in walker {
            let dirent = walkdir_item.map_err(|e| {
                // Missing logdirs may appear later, but this is still an incomplete scan.
                if e.io_error()
                    .map_or(false, |e| e.kind() == io::ErrorKind::NotFound)
                {
                    info!("While walking log directory: {}", e);
                } else {
                    warn!("While walking log directory: {}", e);
                }
                io::Error::from(e)
            })?;
            if !dirent.file_type().is_file()
                || !dirent
                    .file_name()
                    .to_string_lossy()
                    .contains(EVENT_FILE_BASENAME_INFIX)
            {
                continue;
            }
            let run_dir = dirent.path().parent().ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "event file has no parent")
            })?;
            let run_relpath = run_dir
                .strip_prefix(&self.root)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            let run = if run_relpath == Path::new("") {
                Run(".".to_owned())
            } else {
                Run(run_relpath.display().to_string())
            };
            let metadata = dirent.metadata().map_err(io::Error::from)?;
            visitor(
                run,
                EventFileBuf(dirent.into_path()),
                fingerprint(&metadata),
            )?;
        }
        Ok(())
    }

    fn open(&self, path: &EventFileBuf) -> io::Result<Self::File> {
        File::open(&path.0).map(BufReader::new)
    }

    fn open_at(&self, path: &EventFileBuf, offset: u64) -> io::Result<Self::File> {
        let mut file = File::open(&path.0)?;
        file.seek(SeekFrom::Start(offset))?;
        Ok(BufReader::new(file))
    }
}

#[cfg(unix)]
fn fingerprint(metadata: &std::fs::Metadata) -> Option<FileFingerprint> {
    use std::os::unix::fs::MetadataExt;
    Some(FileFingerprint {
        len: metadata.len(),
        identity: format!("{}:{}", metadata.dev(), metadata.ino()),
        modified: format!(
            "{}:{}:{}:{}",
            metadata.mtime(),
            metadata.mtime_nsec(),
            metadata.ctime(),
            metadata.ctime_nsec()
        ),
    })
}

#[cfg(not(unix))]
fn fingerprint(metadata: &std::fs::Metadata) -> Option<FileFingerprint> {
    // Without a creation timestamp there is no safe portable replacement identity.
    Some(FileFingerprint {
        len: metadata.len(),
        identity: format!("{:?}", metadata.created().ok()?),
        modified: format!("{:?}", metadata.modified().ok()?),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};

    fn test_in_logdir(logdir: &Path) -> std::io::Result<()> {
        let run_dir = logdir.join("train");
        std::fs::create_dir(&run_dir)?;
        {
            let mut f = File::create(run_dir.join("foo.tfevents.123"))?;
            f.write_all(b"hello")?;
            f.flush()?;
        }

        let disk_logdir = DiskLogdir::new(logdir.to_path_buf());
        let discoveries = disk_logdir.discover()?;
        assert_eq!(discoveries.len(), 1, "{:?}", discoveries);
        let train_event_files = &discoveries[&Run("train".to_string())];
        assert_eq!(train_event_files.len(), 1, "{:?}", train_event_files);

        let mut event_file = disk_logdir.open(&train_event_files[0])?;
        let mut contents = String::new();
        event_file.read_to_string(&mut contents)?;
        assert_eq!(contents, "hello");

        Ok(())
    }

    #[test]
    fn test_absolute() -> std::io::Result<()> {
        let tmpdir = tempfile::tempdir_in(".")?;
        let logdir = tmpdir.path();
        assert!(logdir.is_absolute(), "expected absolute: {:?}", logdir);
        test_in_logdir(logdir)
    }

    #[test]
    fn test_relative() -> std::io::Result<()> {
        let tmpdir = tempfile::tempdir_in(".")?;
        let logdir = tmpdir
            .path()
            .strip_prefix(std::env::current_dir()?)
            .expect("tmpdir not under $PWD");
        assert!(logdir.is_relative(), "expected relative: {:?}", logdir);
        test_in_logdir(logdir)
    }
}
