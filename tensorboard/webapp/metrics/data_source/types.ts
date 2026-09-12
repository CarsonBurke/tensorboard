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
import {Observable} from 'rxjs';
import {CardMetadata, PluginType} from '../internal_types';

export {PluginType} from '../internal_types';

export const METRICS_PLUGIN_ID = 'timeseries';

/** Map from tag name to the run ids that contain the tag. */
export type TagToRuns = {
  [tag: string]: string[];
};

export type TagToDescription = {
  [tag: string]: string;
};

export interface NonSampledTagMetadata {
  tagToRuns: TagToRuns;
  tagDescriptions: TagToDescription;
}

export interface SampledTimeSeriesInfo {
  maxSamplesPerStep: number;
}

export interface RunSampledInfo {
  [runId: string]: SampledTimeSeriesInfo;
}

export interface TagToRunSampledInfo {
  [tag: string]: RunSampledInfo;
}

export type SampledTagMetadata = {
  tagDescriptions: TagToDescription;
  tagRunSampledInfo: TagToRunSampledInfo;
};

export function isPluginType(text: string): text is PluginType {
  return (
    text === PluginType.SCALARS ||
    text === PluginType.HISTOGRAMS ||
    text === PluginType.IMAGES
  );
}

export type SampledPluginType = PluginType.IMAGES;
const sampledPluginTypes = [PluginType.IMAGES];

export type NonSampledPluginType = Exclude<PluginType, SampledPluginType>;

export function isSampledPlugin(
  plugin: PluginType
): plugin is SampledPluginType {
  return sampledPluginTypes.includes(plugin);
}

export type SingleRunPluginType = PluginType.HISTOGRAMS | PluginType.IMAGES;
const singleRunPluginTypes = [PluginType.HISTOGRAMS, PluginType.IMAGES];

export type MultiRunPluginType = Exclude<PluginType, SingleRunPluginType>;

export function isSingleRunPlugin(
  plugin: PluginType
): plugin is SingleRunPluginType {
  return singleRunPluginTypes.includes(plugin);
}

export interface MetricsCatalogViewport {
  groupOffset: number;
  groupLimit: number;
  visibleGroups: string[];
  filteredOffset: number;
  filteredLimit: number;
}

export interface MetricsCatalogGroup {
  name: string;
  totalCards: number;
}

export type MetricsCatalogCard =
  | {plugin: PluginType.SCALARS; tag: string}
  | {plugin: PluginType.HISTOGRAMS; tag: string; runId: string}
  | {
      plugin: PluginType.IMAGES;
      tag: string;
      runId: string;
      sample: number;
      numSample: number;
    };

export function getMetricsCatalogCardMetadata(
  card: MetricsCatalogCard
): CardMetadata {
  const {plugin, tag} = card;
  if (plugin === PluginType.SCALARS) {
    return {plugin, tag, runId: null};
  }
  if (plugin === PluginType.HISTOGRAMS) {
    return {plugin, tag, runId: card.runId};
  }
  return {
    plugin,
    tag,
    runId: card.runId,
    sample: card.sample,
    numSample: card.numSample,
  };
}

export interface MetricsCatalog {
  groups: MetricsCatalogGroup[];
  totalGroups: number;
  groupOffset: number;
  filteredOffset: number;
  cards: MetricsCatalogCard[];
  totalCards: number;
}

export type TagMetadata = {
  [PluginType.SCALARS]: NonSampledTagMetadata;
  [PluginType.HISTOGRAMS]: NonSampledTagMetadata;
  [PluginType.IMAGES]: SampledTagMetadata;
  catalog?: MetricsCatalog;
};

export interface SingleRunTimeSeriesRequest {
  plugin: SingleRunPluginType;
  tag: string;
  sample?: number;
  runId: string;
}

export interface MultiRunTimeSeriesRequest {
  plugin: MultiRunPluginType;
  tag: string;
  sample?: number;
  experimentIds: string[];
  runIds?: string[];
}

export type TimeSeriesRequest =
  | SingleRunTimeSeriesRequest
  | MultiRunTimeSeriesRequest;

export function isSingleRunTimeSeriesRequest(
  request: TimeSeriesRequest
): request is SingleRunTimeSeriesRequest {
  return isSingleRunPlugin(request.plugin);
}

export type RunToSeries =
  | {
      [runId: string]: ScalarStepDatum[];
    }
  | {[runId: string]: HistogramStepDatum[]}
  | {[runId: string]: ImageStepDatum[]};

export interface TimeSeriesSuccessfulResponse {
  plugin: PluginType;
  tag: string;
  runId?: string;
  sample?: number;
  runToSeries: RunToSeries;
  error?: undefined;
}

export interface TimeSeriesFailedResponse {
  plugin: PluginType;
  tag: string;
  runId?: string;
  sample?: number;
  error: string;
  runToSeries?: undefined;
}

export type TimeSeriesResponse =
  | TimeSeriesSuccessfulResponse
  | TimeSeriesFailedResponse;

export interface ScalarStepDatum {
  step: number;
  wallTime: number;
  value: number;
}

export interface HistogramBin {
  min: number;
  max: number;
  count: number;
}

export interface HistogramStepDatum {
  step: number;
  wallTime: number;
  bins: HistogramBin[];
}

export type ImageId = string;

export interface ImageStepDatum {
  step: number;
  wallTime: number;
  imageId: ImageId;
}

export interface TagMetadataRequest {
  runIds: string[];
  query: string;
  plugins?: PluginType[];
  groupOffset: number;
  groupLimit: number;
  groups: Array<{name: string; offset: number; limit: number}>;
  filteredOffset: number;
  filteredLimit: number;
  pinnedTags: string[];
  pinnedRunIds?: string[];
}

export abstract class MetricsDataSource {
  abstract fetchTagMetadata(
    experimentIds: string[],
    request?: TagMetadataRequest
  ): Observable<TagMetadata>;
  abstract fetchTimeSeries(
    requests: TimeSeriesRequest[]
  ): Observable<TimeSeriesResponse[]>;
  abstract imageUrl(imageId: ImageId): string;
  abstract downloadUrl(
    pluginId: PluginType,
    tag: string,
    runId: string,
    downloadType: 'json' | 'csv'
  ): string;
}

export function isFailedTimeSeriesResponse(
  response: TimeSeriesResponse
): response is TimeSeriesFailedResponse {
  return response.hasOwnProperty('error');
}

export type Tag = string;
