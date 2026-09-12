# Metrics plugin HTTP API

This backend exposes summary data related to "metrics". This includes Scalar,
Histogram, Image data.


### Type `TagToRunIndices`
Type: {[tag: string]: number[]}

Map from tag name to the indices, into the enclosing `runs` list, of the runs
that have data for the tag. Indices are ascending.

### Type `TagToDescription`
Type: {[tag: string]: string}

Map from tag name to a description string.

### Type `NonSampledTagMetadata`
Type: Object

Metadata for tags associated with a non-sampled type plugin. Runs are named
once in `runs` and referred to by index afterwards, because an experiment with
many runs has most tags in most runs, and naming a run per (run, tag) pair
dominates the response size.

Properties:
  - runs: string[]
    - Run names, the index space of `tagToRuns`.
  - tagToRuns: TagToRunIndices
  - tagDescriptions: TagToDescription

### Type `SampledTagMetadata`
Type: Object

Metadata for tags associated with a sampled type plugin.

Properties:
  - tagDescriptions: TagToDescription
  - tagRunSampledInfo: TagToRunSampledInfo

### Type `SampledTimeSeriesInfo`
Type: Object

Metadata associated with a time series generated from a sampled plugin.

Properties:
  - maxSamplesPerStep: number
    - The maximum datum count at any step in the time series. Note that the
      actual number of samples may differ at each step.

### Type `TagToRunSampledInfo`
Type: {[tag: string]: {[run: string]: SampledTimeSeriesInfo}}

Map from tag name to a map from run name to sampled time series info.

### Type `PluginType`
Type: string enum
  - SCALARS: 'scalars'
  - HISTOGRAMS: 'histograms'
  - IMAGES: 'images'

### Type `SingleRunPlugin`
Type: PluginType

Plugins of this type require a single run to be specified when requesting
time series data. Non-single-run plugins are not required to specify a run.

### Type `SampledPlugin`
Type: PluginType

Plugins of this type are associated with sampled time series. Sampled time
series may contain multiple samples of data at each step.

### Type `TagMetadata`
Type: Object

Properties:
  - `[PluginType.SCALARS]`: NonSampledTagMetadata
  - `[PluginType.HISTOGRAMS]`: NonSampledTagMetadata
  - `[PluginType.IMAGES]`: SampledTagMetadata

### Type `TimeSeriesRequest`
Type: Object

Request for time series data, which may correspond to at most one
TimeSeriesResponse in a successful case. Backends may handle requests
differently depending on the plugin, or ignore certain plugins completely.
In the future, this may be extended with options for filtering and sampling.

Properties:
  - plugin: PluginType
  - tag: string
  - run: optional string
    - The name of a requested run, required when plugin is a `SingleRunPlugin`.
  - runs: optional string[]
    - When set, only these run names are returned. An empty list returns no
      series. Omit to return every run that has the tag. Ignored by
      `SingleRunPlugin` requests, which use `run` instead.
  - sample: optional number
    - The zero-indexed sample, required when plugin is a `SampledPlugin`.

### Type `RunToSeries`
Type: {[run: string]: ScalarColumns}|
    {[run: string]: HistogramStepDatum[]}|
    {[run: string]: ImageStepDatum[]}

Map from run name to that run's time series data, sorted by step. Scalars use
a columnar representation; the other plugins use a list of step data.

### Type `TimeSeriesSuccessfulResponse`
Type: Object

Response from the backend containing time series data for a TimeSeriesRequest.
The value of `plugin` determines the type of values in the `runToSeries` dict.
For example, if plugin is `scalars`, then each series will be a
`ScalarColumns`.

Properties:
  - plugin: PluginType
  - tag: string
  - run: optional string
    - The name of a requested run, required when plugin is a `SingleRunPlugin`.
  - sample: optional number
    - The zero-indexed sample, required when plugin is a `SampledPlugin`.
  - runToSeries: RunToSeries

### Type `TimeSeriesFailedResponse`
Type: Object

Response from the backend for a TimeSeriesRequest that failed to get data.

Properties:
  - plugin: PluginType
  - tag: string
  - run: optional string
    - The name of a requested run, required when plugin is a `SingleRunPlugin`.
  - sample: optional number
    - The zero-indexed sample, required when plugin is a `SampledPlugin`.
  - error: string
    - The error reason.

### Type `TimeSeriesResponse`
Type: TimeSeriesSuccessfulResponse|TimeSeriesFailedResponse

Response from the backend containing time series data for a TimeSeriesRequest.

### Type `ScalarColumns`
Type: Object

A scalar time series for one run, held as equally sized parallel columns
ordered by step. Point `i` of the series is
`{step: steps[i], wallTime: wallTimes[i], value: values[i]}`. Columns are used
instead of a list of per-point objects because repeated property names
dominate the response size of a tag with many runs.

Properties:
  - steps: number[]
    - The global step of each datum; integers. A step is a unique key among
      data of this time series.
  - wallTimes: number[]
    - The real-world time of each datum, as float seconds since epoch.
  - values: number[]
    - The scalar value of each datum; floats. Nonfinite values are serialized
      as the strings "NaN", "Infinity", and "-Infinity".

### Type `HistogramBin`
Type: Object

Single bin in a histogram, describing the number of items in a value range.

Properties
  - min: number
    - The smaller value of the bin's range.
  - max: number
    - The larger value of the bin's range.
  - count: number
    - The integer number of items in the bin.

### Type `HistogramStepDatum`
Type: Object

Datum for a single step in a histogram time series.

Properties:
  - step: number
    - The global step at which this datum occurred; an integer. This is a unique
      key among data of this time series.
  - wallTime: number
    - The real-world time at which this datum occurred, as float seconds since
      epoch.
  - bins: HistogramBin[]
    - The histogram contents, as a list of HistogramBins. Bins must be sorted
      by increasing 'min' value, and ranges must not overlap.

### Type `ImageStepDatum`
Type: Object

Datum for a single run+tag+sample+step in a image time series. This does not
contain actual image contents. See `ImageData` for contents of a single image.

Properties:
  - step: number
    - The global step at which this datum occurred; an integer. This is a unique
      key among data of this time series.
  - wallTime: number
    - The real-world time at which this datum occurred, as float seconds since
      epoch.
  - imageId: ImageId
    - A unique id for the image data.

### Type `ImageData`
Type: string

A bytestring of raw image bytes.

### Type `ImageId`
Type: string

A unique reference to identify a single image.

### Route `/data/plugin/timeseries/catalog`

Returns a combined catalog for explicitly selected, experiment-qualified run IDs.
Native Time Series clients use this endpoint for automatic scrolling windows,
without retrieving metadata for members of closed categories.

POST a JSON object with all of these fields (GET accepts the same serialized
object in the `request` query parameter):

- `runIds`: selected IDs of the form `experimentId/runName`; an empty array
  selects no runs, not all runs.
- `query`: case-insensitive regular expression applied to tags.
- `plugins`: array containing any of `scalars`, `histograms`, and `images`.
- `groupOffset`, `groupLimit`: window of category summaries.
- `groups`: requested member windows, each `{name, offset, limit}`. Categories
  use the first slash-delimited tag component. An empty list requests no members.
- `filteredOffset`, `filteredLimit`: flattened filtered-card window.
- `pinnedTags`, `pinnedRunIds`: exact pin metadata scope, independent of member
  windows. Pins do not add cards to the returned scrolling window.

Offsets and limits are nonnegative integers; unlike the legacy `/tags` endpoint,
zero limits request no entries. Invalid requests return HTTP 400.

The response contains `groups` (`{name, totalCards}`), `totalGroups`, `groupOffset`,
`cards`, `totalCards`, and `metadata` (`TagMetadata`). Scalar cards are distinct
tags across selected runs; histograms are distinct run/tag pairs; images are
distinct run/tag/sample tuples. Card descriptors include `plugin` and `tag`,
plus `runId` for histograms/images and `sample` and `numSample` for images.
Counts reflect cards before windowing, not materialized image-sample arrays.
Category summaries require no member metadata or histories. Only requested
card and pin metadata is returned; histories are fetched separately for charts
near the viewport. The frontend prepares charts one viewport ahead, retains
mounted charts within a wider exit buffer, and reuses recently viewed histories
within a 64 MiB estimated inactive-data cache. Reload invalidates that cache.

### Route `/data/plugin/timeseries/tags`

Returns tag metadata for a given experiment's logged metrics. Tag descriptions
may be produced by combining several descriptions for the same tag across
multiple runs.

Args:
  - experiment_id: optional string
    - ID of the request's experiment.
  - run, tag: optional repeated strings
    - Exact names to include. Omission means unrestricted; `run_filter=true`
      or `tag_filter=true` makes an omitted list explicitly empty instead.
  - tag_query: optional regular expression
    - Filters tag names within the selected runs.
  - tag_offset, tag_limit: optional nonnegative integers
    - Page distinct tag names separately for each plugin, sorted by name.
      Zero limit explicitly requests all matching tags; it is not a resource cap.
      The returned metadata includes all matching selected runs for those tags.
      Invalid pagination values and patterns return HTTP 400.

Returns:
  - TagMetadata
  - `totalTags` when paging is requested: the maximum matching distinct-tag
    count across plugins, before pagination. Each plugin has its own page.

Clients may opt into conditional metadata retrieval by sending the
`X-TensorBoard-Metadata-Revision` header. An empty value requests the initial
snapshot. The response is then `{ "revision": string | null, "metadata":
TagMetadata | null, "totalTags"?: number }`. On subsequent requests, send the
returned revision in the same header. A matching revision returns `metadata: null`; reuse the
previous snapshot. A null revision means that the provider cannot safely
cache this response, so the next request must fetch metadata again. Older
servers may return the original `TagMetadata` body and should remain supported.

Revisions cover the authorized view of tag metadata, including image sample
counts, but exclude step and wall-time statistics. Requests are authorized
before checking revisions. Responses use private revalidation and vary on the
revision header and accepted encoding. Clients using the original response
format may also revalidate a returned weak ETag with `If-None-Match`.

Revision identity includes the exact run/tag scope, query, and page. A matching
revision omits `totalTags` along with the body; retain the previous page total.
The backend does not retain metadata response bodies between requests.

Example:

    Response:
    {
        "histograms": {
            "runs": ["test_run"],
            "tagToRuns": {"ages": [0]},
            "tagDescriptions": {
                "ages": "<p>a distribution of Walrus ages</p>"
            },
        },
        "images": {
            "tagDescriptions": {
                "images/tagA": "<p>Initial digits</p>",
                "images/tagB": "<p>Reshaped digits</p>",
            },
            "tagRunSampledInfo": {
                "images/tagA": {
                    "run1": {"samples": 1}
                },
                "images/tagB": {
                    "run1": {"samples": 2},
                    "run2": {"samples": 3},
                },
            },
        },
        "scalars": {
            "runs": ["test_run"],
            "tagToRuns": {"eval/population": [0]},
            "tagDescriptions": {
                "eval/population": "<p>the <em>most</em> valuable statistic</p>"
            },
        },
    }

### Route `/data/plugin/timeseries/timeSeries` (POST)

Responds to a list of requests for time series data. A list of requests may
cover multiple tags across multiple runs with different with data produced by
different plugins. Responses may be in any order.
Clients may wish to call this using tag names returned from a calling /tags.

Args:
  - experiment_id: string
    - string ID of the request's experiment.
  - requests: TimeSeriesRequest[]

Returns:
  - TimeSeriesResponse[]

Example:

    Arguments:
    {
      requests: [
        {"plugin": "scalars", "tag": "eval/population"},
        {"plugin": "histograms", "tag": "ages"},
        {"plugin": "images", "tag": "faces", "sample": 2},
      ]
    }

    Response:
    [
      {
        "plugin": "scalars"
        "tag": "eval/population"
        "runToSeries": {
          "run1": {
              "steps": [100, 200],
              "wallTimes": [1550634693, 1550634899],
              "values": [7, 8],
          }
      },
      {
        "plugin": "histograms"
        "tag": "population"
        "runToSeries": {
          "run1": [
              {
                wallTime: 1550634693,
                step: 100,
                value: [[0, 0.5, 9], [1, 0.5, 10], [10, 0.5, 10], ...]},
          ]
      },
      {
        "plugin": "images"
        "tag": "faces"
        "sample": 2,
        "runToSeries": {
          "run1": [
            {wallTime: 1550634693, step: 100, imageId: "..."},
            {wallTime: 1550634899, step: 200, imageId: "..."},
          ],
        }
      },
    ]

### Route `/data/plugin/timeseries/imageData`

Returns an image's data. Instead of reading the raw data, clients may rely
on this endpoint URL as an HTMLImageElement's 'src' attribute.

Args:
  - imageId: ImageId

Returns:
  - Image data
