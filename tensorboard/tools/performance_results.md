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
