# Performance optimization measurements

Measured September 4, 2026 on the local workstation. These are operation
microbenchmarks, not whole-dashboard speedups. Existing request batching,
metadata-only listing, and run-table changes were present before this work.

## Changes

- Chart hover uses a point accessor with binary search, removing an entire
  coordinate-array copy on each lookup.
- Ordinary chart bounds use one min/max pass. Percentile/outlier bounds retain
  their existing sorting and quantile calculation.
- All six Rust list/read handlers look up explicitly selected runs and tags
  directly when the filter is smaller than the corresponding map.
- Idle and append-only reservoir commits skip scanning previously committed
  samples. Eviction and step-reset commits still reconcile the samples.
- Latest-only reads traverse backwards to the last valid record; empty reads
  do not traverse the series. Other sample counts retain the same algorithm.
- Python returns an independent list directly when every available sample is
  requested, avoiding redundant random sampling and sorting.

## Measurements

For 10,000 points/entries, per operation:

| Operation | Before | After |
| --- | ---: | ---: |
| Chart hover lookup | 0.0490 ms | 0.000480 ms |
| Ordinary chart bounds | 1.421 ms | 0.0788 ms |
| Outlier chart bounds | 1.417 ms | 1.439 ms |
| Rust one-key map filter | 83.2 microseconds | 0.014 microseconds |
| Rust idle reservoir commit | 4.82 microseconds | approximately 0.002 microseconds |
| Rust latest-only sample | 1.02 microseconds | 0.011 microseconds |
| Python requesting every sample | 2.13 ms | 0.0143 ms |

Chart timings use Node 20.19.6, actual transpiled source, the repository's d3,
and medians of five warmed batches. Rust timings use release builds of the
before/after reservoir and downsampling modules and extracted filter helpers,
with opaque benchmark inputs. The Rust filter fixture uses integer keys;
real run/tag strings, locking, serialization, network traffic, and rendering
add costs outside these measurements. Python timings use the actual helper
functions and medians of five batches.

At 100,000 chart points, ordinary bounds improved from 23.1 ms to 0.909 ms.
Outlier bounds remain dominated by sorting and showed no improvement.
A small four-point HTTP request stayed approximately 2 ms before and after;
its returned data matched exactly across the server restart.

## Reproduce chart measurements

From the repository root:

```sh
TB_BENCH_NODE_MODULES="$(npx -y @bazel/bazelisk info output_base)/external/npm/node_modules" \
  node tensorboard/tools/benchmark_chart_utils.js
```

Pass a source-root directory as the final argument to compare against another
version. It needs the two utility files at their normal repository paths;
missing imports fall back to the current source tree. The benchmark verifies
fixture output equality before reporting comparisons. The chart baseline for
this change is available before commit `e04f73ff0`.

## Verification

- Browser suite: 2,116 tests passed.
- Rust core: 118 tests passed.
- Python data provider: 24 tests passed; two framework placeholder tests skipped.
- Independent backend and frontend reviews found no regressions. The frontend
  reviewer additionally ran 294,111 differential checks, including signed zero,
  duplicate coordinates, nonfinite values, and log-scale predicates.
- Rebuilt both `//tensorboard:tensorboard` and
  `//tensorboard/data/server:server`.
- Loaded real multi-run charts in the rebuilt application, then restarted the
  seven managed TensorBoard instances and confirmed that all were serving.

The added regression coverage checks sample equivalence, missing and empty
filters, corruption at the end of a series, training restarts, same-size sample
replacement, duplicate x coordinates, and logarithmic hover access counts.


## Second iteration: metadata, scalar JSON, and chart preprocessing

Measured September 4, 2026, against `d3e401089` on this workstation.

| Workload | Before | After |
|---|---:|---:|
| Live cleanrl unchanged tag request, uncompressed | 2,521–3,791 ms; 2.6 MB | 1.2–2.2 ms; 98 bytes |
| Live cleanrl cached full tag response, uncompressed | full reconstruction | 2.9 ms |
| Scalar response, 10,000 points | 29.1 ms | 15.5 ms |
| Scalar response, 100,000 points | 343.7 ms | 154.7 ms |
| Scalar response, 500,000 points | 1,886.5 ms | 804.4 ms |
| Browser unchanged transforms, 24 charts × 64 runs × 1,000 points | median 152 ms | median 0.3 ms |
| Browser smoothing including restoring point metadata, same fixture | 156.2 ms; one 156 ms main-thread task | 156.4 ms; no main-thread tasks over 50 ms |
| Browser cached smoothing, same fixture | recomputed | 0.5 ms |

These measure different boundaries: live HTTP, an isolated Python response path,
and browser preprocessing without chart rendering. They are not whole-dashboard
speedups. Cold metadata still requires a full listing (3.1 seconds in the live
check). Real dashboard unchanged tag requests were 5.7–52.8 ms including browser
scheduling and a compressed response plus HTTP headers (409 transfer bytes).
One new training run appeared between before/after snapshots; metadata for all
previous runs was identical.

### Implementation and compatibility

- Rust publishes a process-specific metadata revision. Run additions/removals,
  tag creation, valid-data membership changes, and blob maximum-length changes
  invalidate it under mutation locks. Ordinary scalar appends and idle reloads
  preserve it. A before/after revision check prevents caching mixed listings.
- Providers opt into revisions with an explicit authorization-scope contract.
  Unsupported providers and older gRPC servers remain uncached. The plugin
  limits serialized metadata caching to eight entries and 16 MiB, supports
  private ETag revalidation, and offers an additive revision response protocol.
  Frontend metadata reuse preserves namespaced card/index state and still
  resolves newly imported pins.
- Optional scalar columns avoid allocating ScalarDatum objects in the gRPC
  metrics path. Scalar numeric fields are cleansed during response construction,
  avoiding a second recursively copied tree. Legacy providers, nonfinite values,
  signed zero, large integer steps, Unicode, per-request errors, and gzip retain
  their previous wire behavior. Histograms/images keep general JSON cleansing.
- Equal scalar arrays retain identity after refresh. A single normalization and
  partition pass caches points weakly by immutable input and latest settings;
  pinned and original cards share those points. Smoothing caches only the latest
  weight, transfers large jobs to one shared worker, yields via MessageChannel,
  cancels obsolete work, and uses a yielding fallback on worker failure.
  Packing and exceptionally large individual-series reconstruction remain
  synchronous; normal metrics sampling limits each scalar series to 1,000 points.

### Reproduction and evidence

Run the maintained Python benchmark with:

```sh
npx -y @bazel/bazelisk run //tensorboard/tools:benchmark_metrics
```

It uses real protobuf arrays and current provider/plugin code with the RPC
stubbed out, verifies byte-for-byte JSON equality, warms each path, and reports
medians of five samples. The reference path constructs scalar objects and uses
generic HTTP JSON cleansing; the new path uses columns and schema-aware output.

The disposable browser benchmark sources and built assets are at
`~/.cache/tensorboard-perf-iteration2/browser-benchmark/`. It compares the previous
normalization/partition pipeline and the previous actual smoothing source with
current source, checks every point, and records PerformanceObserver long tasks.
The 24-chart fixture retains 1,536,000 scalar points. Rapid synchronous slider
updates published only the latest result. Raw measurements and command logs are
in `/var/tmp/tensorboard-iteration2-*`.

Seven test targets passed: 2,123 frontend tests, 119 Rust tests, protobuf binding
consistency, gRPC/provider tests, the existing TensorFlow metrics suite, and seven
new HTTP/cache compatibility tests. Framework placeholder tests were skipped.
Both runtime binaries rebuilt successfully. The three running tensorwatch
instances (cleanrl, kraggiculture, parameter-golf-postraining) were restarted and
confirmed serving; instances already stopped remained stopped. Live browser
checks rendered multi-run charts and exercised refresh, smoothing, pinning,
and relative-time axes.

Two independent reviewers found no remaining correctness blockers. Additional
read-only differential checks covered 12,000 transforms, 10,000 smoothing cases,
4,374 scalar points, and concurrent smoothing weights with interleaved yields.

## Third iteration: runtime configuration, run filtering, and JSON cleansing

Measured September 5, 2026, against the live cleanrl board (1,760 runs, 87,761
run/tag pairs, 14,527 distinct tags) and a rebuilt instance on the same logdir.

Two configuration defects dominated every backend path and predate all the
code-level work above:

- The launcher exported `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`, so
  every gRPC response was decoded by the pure-Python protobuf backend. Parsing
  the 4.8 MB `ListScalars` response alone took 4.25 s; the native backend takes
  5 ms (`upb`) or 21 ms (`cpp`). The pin was removed; the runfiles' default
  picks the native backend in both build configurations.
- Both binaries were fastbuild (debug) outputs. `bazel-bin` follows the last
  build's configuration, so the launcher had been running a debug Rust data
  server. Rebuilt with `-c opt`; local notes now require it.

Code changes on top of that:

- `matchRunToRegex` compiles the regex once per string instead of once per run.
- `getCurrentRouteRunSelection` keeps its previous map when a regex keystroke
  leaves the effective selection unchanged, so downstream card and run-table
  selectors stay memoized.
- `getScalarTagsForRunSelection` walks the selected runs through an inverted
  run-to-tags index rather than scanning every (tag, run) pair.
- `groupCardIdWithMetdata` skips the re-sort when its input is already ordered,
  which is the case for the card grid; `compareTagNames` compares char codes
  and no longer allocates an enum object per `consumeNumber` call.
- Scalar JSON points are checked for nonfinite values in bulk, avoiding three
  `Cleanse` calls per point. Plugin metadata is parsed once per distinct
  content byte string when filtering by version.

| Workload | Before | After |
|---|---:|---:|
| Rust `ListScalars`, skip statistics, raw bytes | 223 ms | 48 ms |
| Rust `ListScalars`, with statistics, raw bytes | 844 ms | 98 ms |
| Rust full logdir load (3.7 GB, page-cached) | not measured | 2.4 s |
| Live cold `timeseries/tags` (3.3 MB) | 4.9 s | 0.40 s |
| Live `timeSeries` for a 1,649-run tag (30 MB) | 1.50 s | 0.47 s |
| Live `/data/runs` | 17 ms | 3.7 ms |
| `_tags_impl` Python only, native protobuf | 0.44 s | 0.25 s |
| Browser: typing a 22-character run regex, main-thread busy | 6.56 s | 0.71 s |
| Browser: selecting 7 filtered runs, main-thread busy | 1.37 s | 0.22 s |
| Maintained benchmark, columnar path, 500,000 points | 804 ms | 394 ms |

The browser rows come from a CDP `Profiler` sample (200 us interval) over a
headless Chromium driving the real dashboard; the before column used the
previous bundle on the live server, the after column the rebuilt bundle on the
same logdir. Card groups, card counts per group, and rendered charts were
identical between the two builds for the same filter and selection.

Remaining per-keystroke work is the props-based `getRuns` selector and the
run-table sort comparator, each a few milliseconds at 1,760 runs. Cold
`_tags_impl` is now bounded by protobuf field access for 87k entries and by the
RPC itself. `hparams` still lists scalars with statistics (98 ms here); its
tests autospec the provider, so switching it to `list_scalars_metadata` was
left alone.

Verification: 2,124 browser specs, the metrics plugin and gRPC provider Python
suites, and the maintained benchmark's byte-equality check all pass. The three
running boards were restarted on the rebuilt binaries and confirmed serving.

## Fourth iteration: columnar scalar wire format and chart transform

Measured September 5, 2026 against the live cleanrl board (1,826 runs, 95,211
run/tag pairs). Warm HTTP was already 2-6 ms for metadata after the third
iteration, so this round targeted the two costs that remained: the scalar
payload itself and the per-point work the chart does on every frame.

- `timeSeries` scalar responses are columnar: each run carries `steps`,
  `wallTimes`, and `values` arrays instead of one `{wallTime, step, value}`
  object per point. Repeated keys and separators dominated the old payload
  (46% of 27 MB). The provider's columnar read feeds it directly; providers
  without `read_scalar_columns` still get columns built from `ScalarDatum`s.
  `http_api.md` documents the type as `ScalarColumns`, and the frontend data
  source expands it back to `ScalarStepDatum[]` at the same boundary that
  already re-keyed runs, so nothing downstream changed.
- Scalar JSON is emitted with compact separators, and the serve path no longer
  runs a second recursive `Cleanse` copy over every point.
- `Coordinator.transformDataToUiCoords` transforms a whole polyline in one
  call. The per-point path built two arrays and re-read the scale
  configuration per point; the batch path hoists the domain, range, and
  temporal `d3` scale out of the loop. `TemporalScale` memoizes its configured
  scale instead of rebuilding it per `forward`.
- `SeriesLineView` partitions a NaN-free polyline without copying it, and
  renderer cache keys are string concatenations rather than `JSON.stringify`
  calls. The thick-polyline geometry builder no longer allocates per segment.
- The WebGL2 offscreen-canvas probe is memoized; it used to create a real
  context per chart construction.
- `ChartImpl.dispose` was a no-op, so a chart's renderer, scene, and WebGL
  context outlived it and pooled workers kept the chart alive through the
  message port. Dispose now releases geometries and materials, drops the
  context (`dispose` alone keeps it, and a page holds only ~16), detaches the
  context-lost listener so the deliberate loss is not reported as a failure,
  and closes the worker port.

| Workload | Before | After |
|---|---:|---:|
| Live `timeSeries`, 1,722 runs / 423,051 points, identity | 28.55 MB in 0.41-0.76 s | 14.23 MB in 0.26 s |
| Same, gzip | 6.43 MB in 0.76 s | 5.28 MB in 0.52-0.66 s |
| Browser fetch + parse + build data, same request | 1,066-1,388 ms | 708-892 ms |
| ... of which `JSON.parse` | 47-110 ms | 20-26 ms |
| Benchmark serialize, 500,000 points | 879 ms / 34.6 MB | 232 ms / 17.6 MB |
| Benchmark serialize via legacy per-point provider | 879 ms | 570 ms |
| Chart transform, wall-time axis, 100,000 points | 66.8 ms | 1.6 ms |
| Chart transform, step axis, 100,000 points | 5.1 ms | 1.0 ms |

Wire-format equality is checked directly: expanding the live columnar response
reproduces all 1,722 series of the old response point for point, and the
`timeseries/tags` response is unchanged. The maintained
`//tensorboard/tools:benchmark_metrics` harness asserts the same equality for
both provider paths, and `benchmark_chart_utils.js` now measures the transform
against a `git worktree` baseline.

Chart hover, ordinary extent, and outlier extent are unchanged by this round;
outlier extent (25 ms at 100,000 points) is now the most expensive chart
preprocessing step, bounded by its percentile sort.

Verification: 2,127 browser specs (two new renderer-dispose contract tests),
the metrics plugin, metrics performance, and gRPC provider Python suites, and
the two benchmark harnesses. Live checks on the rebuilt binaries rendered
scalar cards with correct axes and legend values, and 14 cycles of tag-filter
churn (~70 chart creations) produced no renderer recovery, no console errors,
and no blank charts.

## Fifth iteration: tag-major metadata listing and a deduplicated listing RPC

Measured September 6, 2026 against the live cleanrl board (1,841 runs, 96,903
run/tag pairs, 19,725 distinct tags). After the fourth iteration the first
paint was dominated by one request: `GET /data/plugin/timeseries/tags` took
about a second and 4.2 MB, because every layer named every (run, tag) pair
separately. A wide experiment has one pair per run per tag, so naming a pair
per entry makes the response quadratic in an experiment's width while the
information in it is not.

- `ListScalarsRequest.dedup_names` asks the data server for indexed tables:
  distinct tag names in `tag_names`, distinct summary metadata in
  `summary_metadata_table`, and per-entry `tag_index` / `summary_metadata_index`
  references. On this logdir 96,903 entries reference 19,725 names and a single
  metadata value. Servers that predate the field answer in the inline form and
  clients handle both, so an older data server still works.
- `DataProvider.list_scalars_tag_index` is the columnar counterpart of
  `list_scalars_metadata`: a `TagIndex` of parallel run/tag/content index
  arrays instead of a `ScalarTimeSeries` object per pair. The gRPC provider
  builds it directly from the response tables; providers that do not implement
  it fall back to `list_scalars_metadata`, which the metrics plugin still uses.
- The `timeseries/tags` JSON is tag-major: `runs` is the index space and
  `tagToRuns` maps each tag to indices into it, replacing `runTagInfo`'s
  repetition of every run name once per tag. `http_api.md` documents
  `TagToRunIndices`.
- The frontend consumes that shape directly. It converts each run name to a run
  id once, keeps `tagToRuns` as the store's own representation instead of
  inverting `runTagInfo` on every metadata load, and derives the run-to-tags
  inverse in a memoized selector for the one consumer that needs it.
  Multi-experiment loads concatenate a tag's run ids rather than overwriting
  them.

| Workload | Before | After |
|---|---:|---:|
| Live `timeseries/tags`, cold HTTP | 1.024 s / 4.22 MB (108,485 pairs) | 0.252 s / 2.09 MB (96,903 pairs) |
| Live listing RPC bytes, same instant | 5.48 MB | 2.14 MB |
| Live listing RPC round trip | 161.9 ms | 104.9 ms |
| Live client protobuf parse | 52.7 ms | 25.7 ms |
| Live provider listing | 373.1 ms | 164.8 ms |
| Live `_tags_impl` end to end | 510.7 ms | 245.6 ms |
| Synthetic 95,836 pairs: wire | 4.83 MB | 1.93 MB |
| Synthetic 95,836 pairs: parse | 26.7 ms | 8.1 ms |
| Synthetic 95,836 pairs: `_tags_impl` | 125.7 ms | 32.3 ms |
| Rust listing build, 96,951 pairs | 31.9 ms | 36.5 ms |
| Rust listing encode, same | 9.9 ms | 5.3 ms |

Before and after are measured in the same process against the same data:
`//tensorboard/tools:benchmark_tags` requests both wire forms and runs both
provider paths, and asserts the two listings are equal. The cold-HTTP row is
the only one whose two sides come from different snapshots of a live logdir
(2,005 and 1,841 runs); the 1,840 runs common to both carry byte-identical
tag sets, and per pair the time falls from 9.4 to 2.6 us. Run against a data
server that predates `dedup_names`, the columnar path still produces the same
listing and still wins (302.5 ms versus 180.4 ms), because it skips building a
Python object per pair. The Rust handler's own cost is a wash: interning names
and comparing metadata values adds about as much as it saves in cloning and
encoding (`//tensorboard/data/server:bench` reports both forms), so the win is
the 3.3 MB that no longer crosses the socket and the client work that no longer
happens.

Verification: 2,128 browser specs, the metrics plugin, metrics performance, and
gRPC provider Python suites, and both benchmark harnesses. All three boards
were restarted on the rebuilt binaries; every payload's tag indices are in
range, and the dashboards render charts, legends, tag autocomplete, and tag and
run filtering with no console errors.
