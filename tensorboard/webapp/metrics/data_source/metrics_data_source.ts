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
import {Injectable} from '@angular/core';
import {Store} from '@ngrx/store';
import {forkJoin, Observable, of} from 'rxjs';
import {filter, map, switchMap, take, withLatestFrom} from 'rxjs/operators';
import {
  getIsFeatureFlagsLoaded,
  getIsMetricsImageSupportEnabled,
} from '../../feature_flag/store/feature_flag_selectors';
import {State as FeatureFlagAppState} from '../../feature_flag/store/feature_flag_types';
import {TBHttpClient} from '../../webapp_data_source/tb_http_client';
import {hasOwn} from '../../util/lang';
import {
  BackendScalarColumns,
  BackendTimeSeriesRequest,
  BackendTimeSeriesResponse,
} from './metrics_backend_types';
import {
  HistogramStepDatum,
  ImageId,
  ImageStepDatum,
  isFailedTimeSeriesResponse,
  isSingleRunPlugin,
  MetricsDataSource,
  MetricsCatalog,
  MultiRunTimeSeriesRequest,
  PluginType,
  RunToSeries,
  ScalarStepDatum,
  SingleRunTimeSeriesRequest,
  TagMetadata,
  TagMetadataRequest,
  TimeSeriesRequest,
  TimeSeriesResponse,
} from './types';

const HTTP_PATH_PREFIX = 'data/plugin/timeseries';

function parseRunId(runId: string): {run: string; experimentId: string} {
  const slashIndex = runId.indexOf('/');
  return {
    run: runId.substring(slashIndex + 1),
    experimentId: runId.substring(0, slashIndex),
  };
}

function runToRunId(run: string, experimentId: string) {
  return `${experimentId}/${run}`;
}

function buildFrontendTimeSeriesResponse(
  backendResponse: BackendTimeSeriesResponse,
  experimentId: string
): TimeSeriesResponse {
  const {runToSeries, run, ...responseRest} = backendResponse;
  const response = {...responseRest} as TimeSeriesResponse;
  if (runToSeries) {
    if (backendResponse.plugin === PluginType.SCALARS) {
      response.runToSeries = expandScalarColumns(
        runToSeries as {[run: string]: BackendScalarColumns},
        experimentId
      );
    } else {
      response.runToSeries = buildRunIdKeyedObject(
        runToSeries as
          | {[run: string]: HistogramStepDatum[]}
          | {[run: string]: ImageStepDatum[]},
        experimentId
      );
    }
  }
  if (run) {
    response.runId = runToRunId(run, experimentId);
  }
  return response;
}

/**
 * Rebuilds per-point scalar data from the columnar wire format, keyed by run
 * id. See `ScalarColumns` in http_api.md.
 */
function expandScalarColumns(
  runToColumns: {[run: string]: BackendScalarColumns},
  experimentId: string
): RunToSeries {
  // Prototype-free: run names come from the user, so an inherited
  // `constructor` or `hasOwnProperty` must never be visible as a run.
  const runToSeries = Object.create(null) as Record<string, ScalarStepDatum[]>;
  for (const run in runToColumns) {
    if (!hasOwn(runToColumns, run)) {
      continue;
    }
    const {steps, wallTimes, values} = runToColumns[run];
    const series = new Array<ScalarStepDatum>(steps.length);
    for (let i = 0; i < series.length; i++) {
      series[i] = {wallTime: wallTimes[i], step: steps[i], value: values[i]};
    }
    runToSeries[runToRunId(run, experimentId)] = series;
  }
  return runToSeries;
}

function buildRunIdKeyedObject<T extends {}>(
  backendObject: T,
  experimentId: string
): T {
  const frontendObject = Object.create(null) as Record<string, any>;
  for (const run in backendObject) {
    if (hasOwn(backendObject, run)) {
      const runId = runToRunId(run, experimentId);
      frontendObject[runId] = backendObject[run];
    }
  }
  return frontendObject as T;
}

/**
 * An implementation of MetricsDataSource that treats RunIds as identifiers
 * containing run name and experimentId.
 */
@Injectable()
export class TBMetricsDataSource implements MetricsDataSource {
  constructor(
    private readonly http: TBHttpClient,
    private readonly store: Store<FeatureFlagAppState>
  ) {}

  fetchTagMetadata(experimentIds: string[], request?: TagMetadataRequest) {
    const experiments = new Set(experimentIds);
    const inRoute = (id: string) =>
      experiments.has(parseRunId(id).experimentId);
    const scope: TagMetadataRequest = request
      ? {
          ...request,
          runIds: request.runIds.filter(inRoute),
          pinnedRunIds: (request.pinnedRunIds ?? []).filter(inRoute),
        }
      : {
          runIds: [],
          query: '',
          groupOffset: 0,
          groupLimit: 40,
          groups: [],
          filteredOffset: 0,
          filteredLimit: 40,
          pinnedTags: [],
        };
    const empty = (): TagMetadata => ({
      scalars: {tagDescriptions: {}, tagToRuns: {}},
      histograms: {tagDescriptions: {}, tagToRuns: {}},
      images: {tagDescriptions: {}, tagRunSampledInfo: {}},
      catalog: {
        groups: [],
        totalGroups: 0,
        groupOffset: scope.groupOffset,
        filteredOffset: scope.filteredOffset,
        cards: [],
        totalCards: 0,
      },
    });
    if (
      !scope.runIds.length &&
      !(scope.pinnedTags.length && scope.pinnedRunIds?.length)
    ) {
      return of(empty());
    }
    return this.store.select(getIsFeatureFlagsLoaded).pipe(
      filter(Boolean),
      take(1),
      withLatestFrom(this.store.select(getIsMetricsImageSupportEnabled)),
      switchMap(([, imagesSupported]) => {
        const plugins = (
          scope.plugins?.length ? scope.plugins : Object.values(PluginType)
        ).filter((plugin) => imagesSupported || plugin !== PluginType.IMAGES);
        if (!plugins.length && !scope.pinnedTags.length) return of(empty());
        // One server-side union preserves category counts and card page order
        // across experiments. No experiment can force a full tag catalog into
        // the browser merely to merge its page with another experiment.
        return this.http
          .post<
            Omit<MetricsCatalog, 'filteredOffset'> & {metadata: TagMetadata}
          >(
            `/${HTTP_PATH_PREFIX}/catalog`,
            {
              ...scope,
              // A persisted filter may select only a disabled plugin. Pins
              // still resolve, without expanding that empty catalog scope.
              runIds: plugins.length ? scope.runIds : [],
              pinnedRunIds: plugins.length
                ? scope.pinnedRunIds ?? []
                : [
                    ...new Set([
                      ...scope.runIds,
                      ...(scope.pinnedRunIds ?? []),
                    ]),
                  ],
              plugins: plugins.length
                ? plugins
                : [PluginType.SCALARS, PluginType.HISTOGRAMS],
            },
            undefined,
            'request'
          )
          .pipe(
            map(({metadata, ...catalog}) => ({
              ...metadata,
              images: imagesSupported
                ? metadata.images
                : {tagDescriptions: {}, tagRunSampledInfo: {}},
              catalog: {...catalog, filteredOffset: scope.filteredOffset},
            }))
          );
      })
    );
  }

  /**
   * Fetches all requested cards in one HTTP request per experiment.
   *
   * The backend accepts an array of requests, so batching here avoids paying
   * connection, parsing, and scheduling overhead once per visible card.
   */
  fetchTimeSeries(requests: TimeSeriesRequest[]) {
    if (!requests.length) {
      return forkJoin([]);
    }

    type BackendRequestEntry = {
      requestIndex: number;
      experimentId: string;
      backendRequest: BackendTimeSeriesRequest;
    };
    type BackendResponseEntry = BackendRequestEntry & {
      backendResponse: BackendTimeSeriesResponse;
    };

    const backendEntries: BackendRequestEntry[] = [];
    for (const [requestIndex, request] of requests.entries()) {
      if (isSingleRunPlugin(request.plugin)) {
        const {runId, ...requestRest} = request as SingleRunTimeSeriesRequest;
        const {run, experimentId} = parseRunId(runId);
        backendEntries.push({
          requestIndex,
          experimentId,
          backendRequest: {...requestRest, run},
        });
        continue;
      }

      const {experimentIds, runIds, ...requestRest} =
        request as MultiRunTimeSeriesRequest;
      if (!experimentIds.length) {
        return forkJoin([]);
      }

      const experimentIdSet = new Set(experimentIds);
      const runsByExperiment = new Map<string, string[]>();
      if (runIds) {
        for (const runId of runIds) {
          const {run, experimentId} = parseRunId(runId);
          if (!experimentIdSet.has(experimentId)) {
            continue;
          }
          const runs = runsByExperiment.get(experimentId) || [];
          runs.push(run);
          runsByExperiment.set(experimentId, runs);
        }
      }

      for (const experimentId of experimentIds) {
        const backendRequest: BackendTimeSeriesRequest = {...requestRest};
        if (runIds) {
          const runs = runsByExperiment.get(experimentId);
          if (!runs?.length) {
            continue;
          }
          backendRequest.runs = runs;
        }
        backendEntries.push({requestIndex, experimentId, backendRequest});
      }
    }

    const entriesByExperiment = new Map<string, BackendRequestEntry[]>();
    for (const entry of backendEntries) {
      const entries = entriesByExperiment.get(entry.experimentId) || [];
      entries.push(entry);
      entriesByExperiment.set(entry.experimentId, entries);
    }

    const fetches = Array.from(entriesByExperiment.entries()).map(
      ([experimentId, entries]) => {
        return this.fetchTimeSeriesBackendRequests(
          entries.map((entry) => entry.backendRequest),
          experimentId
        ).pipe(
          map((backendResponses) => {
            if (backendResponses.length !== entries.length) {
              throw new Error(
                `Expected ${entries.length} time series responses, got ${backendResponses.length}`
              );
            }
            return entries.map((entry, index) => ({
              ...entry,
              backendResponse: backendResponses[index],
            }));
          })
        );
      }
    );

    const buildResponses = (
      responseBatches: BackendResponseEntry[][]
    ): TimeSeriesResponse[] => {
      const entriesByRequest = requests.map(() => [] as BackendResponseEntry[]);
      for (const batch of responseBatches) {
        for (const entry of batch) {
          entriesByRequest[entry.requestIndex].push(entry);
        }
      }

      return requests.map((request, requestIndex) => {
        const entries = entriesByRequest[requestIndex];
        if (isSingleRunPlugin(request.plugin)) {
          return buildFrontendTimeSeriesResponse(
            entries[0].backendResponse,
            entries[0].experimentId
          );
        }

        const combinedResponse = {
          plugin: request.plugin,
          tag: request.tag,
          ...(request.sample === undefined ? {} : {sample: request.sample}),
        } as TimeSeriesResponse;
        for (const entry of entries) {
          if (combinedResponse.error) {
            continue;
          }
          const frontendResponse = buildFrontendTimeSeriesResponse(
            entry.backendResponse,
            entry.experimentId
          );
          if (isFailedTimeSeriesResponse(frontendResponse)) {
            combinedResponse.error = frontendResponse.error;
            combinedResponse.runToSeries = undefined;
          } else {
            combinedResponse.runToSeries = combinedResponse.runToSeries || {};
            for (const run of Object.keys(frontendResponse.runToSeries)) {
              combinedResponse.runToSeries[run] =
                frontendResponse.runToSeries[run];
            }
          }
        }
        if (!combinedResponse.error) {
          combinedResponse.runToSeries = combinedResponse.runToSeries || {};
        }
        return combinedResponse;
      });
    };

    if (!fetches.length) {
      return of(buildResponses([]));
    }
    return forkJoin(fetches).pipe(map(buildResponses));
  }

  private fetchTimeSeriesBackendRequests(
    backendRequests: BackendTimeSeriesRequest[],
    experimentId: string
  ) {
    const body = new FormData();
    body.append('requests', JSON.stringify(backendRequests));
    return this.http.post<BackendTimeSeriesResponse[]>(
      `/experiment/${experimentId}/${HTTP_PATH_PREFIX}/timeSeries`,
      body
    );
  }

  imageUrl(imageId: ImageId): string {
    return `${HTTP_PATH_PREFIX}/imageData?imageId=${imageId}`;
  }

  downloadUrl(
    pluginId: PluginType,
    tag: string,
    runId: string,
    downloadType: 'json' | 'csv'
  ): string {
    const {run, experimentId} = parseRunId(runId);
    let pluginAndRoute: string;
    switch (pluginId) {
      case PluginType.SCALARS:
        pluginAndRoute = 'scalars/scalars';
        break;
      default:
        throw new Error(
          `Not implemented: downloadUrl for ${pluginId} is not implemented yet`
        );
    }

    if (!experimentId) {
      throw new Error(
        'experimentId is empty; it is required to form downloadUrl.'
      );
    }
    const params = new URLSearchParams({tag, run, format: downloadType});
    return `/experiment/${experimentId}/data/plugin/${pluginAndRoute}?${params}`;
  }
}
