/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
/**
 * @fileoverview Types produced only by the backend data source.
 *
 * Types defined in `plugins/metrics/http_api.md`.
 */

import {
  HistogramStepDatum,
  ImageStepDatum,
  PluginType,
  TagToDescription,
  TagToRunSampledInfo,
} from './types';

/**
 * Tag-major tag metadata for a non-sampled plugin. `runs` is the index space
 * for the numbers in `tagToRuns`: it holds run names in ascending order, and
 * each `number[]` is an ascending list of indices into `runs`.
 */
export interface BackendNonSampledTagMetadata {
  runs: string[];
  tagToRuns: {[tag: string]: number[]};
  tagDescriptions: TagToDescription;
}

export type BackendSampledTagMetadata = {
  tagDescriptions: TagToDescription;
  tagRunSampledInfo: TagToRunSampledInfo;
};

export type BackendTagMetadata = {
  [PluginType.SCALARS]: BackendNonSampledTagMetadata;
  [PluginType.HISTOGRAMS]: BackendNonSampledTagMetadata;
  [PluginType.IMAGES]: BackendSampledTagMetadata;
};

export interface BackendTimeSeriesRequest {
  plugin: PluginType;
  tag: string;
  run?: string;
  runs?: string[];
  sample?: number;
}

/**
 * A run's scalar series, held as parallel columns; see `ScalarColumns` in
 * http_api.md. Columns keep repeated property names out of responses that
 * can carry hundreds of thousands of points.
 *
 * As with the previous per-point encoding, nonfinite values arrive as the
 * strings "NaN", "Infinity", and "-Infinity" inside `values`.
 */
export interface BackendScalarColumns {
  steps: number[];
  wallTimes: number[];
  values: number[];
}

export type BackendRunToSeries =
  | {[run: string]: BackendScalarColumns}
  | {[run: string]: HistogramStepDatum[]}
  | {[run: string]: ImageStepDatum[]};

export interface BackendTimeSeriesSuccessfulResponse {
  plugin: PluginType;
  tag: string;
  run?: string;
  sample?: number;
  runToSeries: BackendRunToSeries;
  error?: undefined;
}

export interface BackendTimeSeriesFailedResponse {
  plugin: PluginType;
  tag: string;
  run?: string;
  sample?: number;
  error: string;
  runToSeries?: undefined;
}

export type BackendTimeSeriesResponse =
  | BackendTimeSeriesSuccessfulResponse
  | BackendTimeSeriesFailedResponse;
