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

//! Adapter from GCS to TensorBoard logdirs.

use reqwest::StatusCode;
use std::collections::HashMap;
use std::env;
use std::io::{self, BufReader, Read};
use std::path::{Path, PathBuf};

use super::Client;
use crate::logdir::{EventFileBuf, FileFingerprint, EVENT_FILE_BASENAME_INFIX};
use crate::types::Run;

/// A reference to a GCS object with a read offset.
pub struct File {
    gcs: Client,
    bucket: String,
    object: String,
    pos: u64,
}

impl File {
    fn new(gcs: Client, bucket: String, object: String, pos: u64) -> Self {
        Self {
            gcs,
            bucket,
            object,
            pos,
        }
    }
}

fn reqwest_to_io_error(e: reqwest::Error) -> io::Error {
    let kind = match e.status() {
        Some(StatusCode::NOT_FOUND) => io::ErrorKind::NotFound,
        Some(StatusCode::FORBIDDEN) => io::ErrorKind::PermissionDenied,
        Some(StatusCode::UNAUTHORIZED) => io::ErrorKind::PermissionDenied,
        Some(StatusCode::REQUEST_TIMEOUT) => io::ErrorKind::TimedOut,
        _ if e.is_timeout() => io::ErrorKind::TimedOut,
        _ if e.is_decode() => io::ErrorKind::InvalidData,
        _ => io::ErrorKind::Other,
    };
    io::Error::new(kind, e)
}

#[derive(Debug, thiserror::Error)]
enum VisitError {
    #[error(transparent)]
    Request(#[from] reqwest::Error),
    #[error(transparent)]
    Visitor(#[from] io::Error),
}

impl Read for File {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        let range = self.pos..=self.pos.saturating_add(buf.len() as u64 - 1);
        let result = self
            .gcs
            .read(&self.bucket, &self.object, range)
            .map_err(reqwest_to_io_error)?;
        buf[0..result.len()].copy_from_slice(&result);
        self.pos += result.len() as u64;
        Ok(result.len())
    }
}

pub struct Logdir {
    gcs: Client,
    bucket: String,
    /// Invariant: `prefix` either is empty or ends with `/`, and thus an event file name should be
    /// joined onto `prefix` to form its full object name.
    prefix: String,
    /// Size of the opened file read buffer (in Kb) when reading from GCS.
    /// The `gcs::Logdir::new` will attempt to fetch the `TB_GCS_BUFFER_SIZE_KB` environment
    /// variable that represent the read buffer size (in Kb) for each TF events file.
    /// Note: if reading a large number of TF events files, set an appropriate value for
    /// `buffer_capacity` to prevent running out of memory. This determines the total size of the
    /// allocated memory.
    /// The default value is defined by the `DEFAULT_BUFFER_CAPACITY_KB` constant.
    buffer_capacity: usize,
}

/// Default size of the GCS file read buffer (in Kb).
/// Read large chunks from GCS to reduce network roundtrips.
const DEFAULT_BUFFER_CAPACITY_KB: usize = 1024 * 16;

impl Logdir {
    pub fn new(gcs: Client, bucket: String, mut prefix: String) -> Self {
        if !prefix.is_empty() && !prefix.ends_with('/') {
            prefix.push('/');
        }
        // convert the Kb buffer size to bytes
        let buffer_capacity = match env::var("TB_GCS_BUFFER_SIZE_KB") {
            Ok(val) => {
                val.parse::<usize>()
                    .ok()
                    .unwrap_or(DEFAULT_BUFFER_CAPACITY_KB)
                    * 1024
            }
            Err(_) => DEFAULT_BUFFER_CAPACITY_KB * 1024,
        };

        Self {
            gcs,
            bucket,
            prefix,
            buffer_capacity,
        }
    }
}

impl crate::logdir::Logdir for Logdir {
    type File = BufReader<File>;

    fn discover(&self) -> io::Result<HashMap<Run, Vec<EventFileBuf>>> {
        let mut run_map: HashMap<Run, Vec<EventFileBuf>> = HashMap::new();
        self.visit(&mut |run, file, _| {
            run_map.entry(run).or_default().push(file);
            Ok(())
        })?;
        Ok(run_map)
    }

    fn visit(
        &self,
        visitor: &mut dyn FnMut(Run, EventFileBuf, Option<FileFingerprint>) -> io::Result<()>,
    ) -> io::Result<()> {
        self.gcs
            .visit(&self.bucket, &self.prefix, &mut |object| {
                let name = object.name.strip_prefix(&self.prefix).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "GCS object outside logdir prefix",
                    )
                })?;
                let path = PathBuf::from(name);
                let is_event_file = path.file_name().map_or(false, |n| {
                    n.to_string_lossy().contains(EVENT_FILE_BASENAME_INFIX)
                });
                if !is_event_file {
                    return Ok::<(), VisitError>(());
                }
                let run_relpath = path.parent().unwrap_or_else(|| Path::new(""));
                let run = if run_relpath == Path::new("") {
                    Run(".".to_owned())
                } else {
                    Run(run_relpath.display().to_string())
                };
                let len = object
                    .size
                    .parse::<u64>()
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                let fingerprint = FileFingerprint {
                    len,
                    // GCS objects are immutable: even composing an append creates a new
                    // generation. Replay it rather than mistake a rewrite for an append.
                    identity: object.generation.clone(),
                    modified: format!("{}:{}", object.generation, object.updated),
                };
                visitor(run, EventFileBuf(path), Some(fingerprint))?;
                Ok(())
            })
            .map_err(|e| match e {
                VisitError::Request(e) => reqwest_to_io_error(e),
                VisitError::Visitor(e) => e,
            })
    }

    fn open(&self, path: &EventFileBuf) -> io::Result<Self::File> {
        self.open_at(path, 0)
    }

    fn open_at(&self, path: &EventFileBuf, offset: u64) -> io::Result<Self::File> {
        // Paths as returned by discovery are always valid Unicode.
        let mut object = self.prefix.clone();
        object.push_str(path.0.to_string_lossy().as_ref());
        let file = File::new(self.gcs.clone(), self.bucket.clone(), object, offset);
        Ok(BufReader::with_capacity(self.buffer_capacity, file))
    }
}
