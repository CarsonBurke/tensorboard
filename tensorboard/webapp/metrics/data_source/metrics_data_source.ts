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
import {HttpHeaders} from '@angular/common/http';
import {Store} from '@ngrx/store';
import {forkJoin, Observable, of} from 'rxjs';
import {filter, map, take, withLatestFrom} from 'rxjs/operators';
import {
  getIsFeatureFlagsLoaded,
  getIsMetricsImageSupportEnabled,
} from '../../feature_flag/store/feature_flag_selectors';
import {State as FeatureFlagAppState} from '../../feature_flag/store/feature_flag_types';
import {TBHttpClient} from '../../webapp_data_source/tb_http_client';
import {
  BackendTagMetadata,
  BackendTimeSeriesRequest,
  BackendTimeSeriesResponse,
} from './metrics_backend_types';
import {
  ImageId,
  isSampledPlugin,
  isFailedTimeSeriesResponse,
  isSingleRunPlugin,
  MetricsDataSource,
  MultiRunTimeSeriesRequest,
  PluginType,
  RunSampledInfo,
  RunToSeries,
  RunToTags,
  SingleRunTimeSeriesRequest,
  TagMetadata,
  TagToRunSampledInfo,
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
    response.runToSeries = buildRunIdKeyedObject<RunToSeries>(
      runToSeries,
      experimentId
    );
  }
  if (run) {
    response.runId = runToRunId(run, experimentId);
  }
  return response;
}

function buildRunIdKeyedObject<T extends {}>(
  backendObject: T,
  experimentId: string
): T {
  const frontendObject = {} as Record<string, any>;
  for (const run in backendObject) {
    if (backendObject.hasOwnProperty(run)) {
      const runId = runToRunId(run, experimentId);
      frontendObject[runId] = backendObject[run];
    }
  }
  return frontendObject as T;
}

function buildFrontendTagMetadata(
  backendTagMetadata: BackendTagMetadata,
  experimentId: string
): TagMetadata {
  const tagMetadata = {} as TagMetadata;
  for (const pluginType of Object.keys(backendTagMetadata)) {
    const plugin = pluginType as PluginType;
    if (isSampledPlugin(plugin)) {
      const {tagRunSampledInfo, ...rest} = backendTagMetadata[plugin];
      const frontendTagRunSampledInfo = {} as TagToRunSampledInfo;
      for (const tag in tagRunSampledInfo) {
        if (tagRunSampledInfo.hasOwnProperty(tag)) {
          frontendTagRunSampledInfo[tag] =
            buildRunIdKeyedObject<RunSampledInfo>(
              tagRunSampledInfo[tag],
              experimentId
            );
        }
      }
      tagMetadata[plugin] = {
        ...rest,
        tagRunSampledInfo: frontendTagRunSampledInfo,
      };
    } else {
      const {runTagInfo, ...rest} = backendTagMetadata[plugin];
      tagMetadata[plugin] = {
        ...rest,
        runTagInfo: buildRunIdKeyedObject<RunToTags>(runTagInfo, experimentId),
      };
    }
  }
  return tagMetadata;
}

function buildCombinedTagMetadata(results: TagMetadata[]): TagMetadata {
  // Collate results from different experiments.
  const tagMetadata = {} as TagMetadata;
  for (const experimentTagMetadata of results) {
    for (const plugin of Object.values(PluginType)) {
      if (isSampledPlugin(plugin)) {
        tagMetadata[plugin] = tagMetadata[plugin] || {
          tagDescriptions: {},
          tagRunSampledInfo: {},
        };
        const {tagDescriptions, tagRunSampledInfo} =
          experimentTagMetadata[plugin];
        tagMetadata[plugin].tagDescriptions = {
          ...tagMetadata[plugin].tagDescriptions,
          ...tagDescriptions,
        };
        const combinedTagRunSampledInfo = tagMetadata[plugin].tagRunSampledInfo;
        for (const tag of Object.keys(tagRunSampledInfo)) {
          combinedTagRunSampledInfo[tag] = combinedTagRunSampledInfo[tag] || {};
          for (const runId of Object.keys(tagRunSampledInfo[tag])) {
            combinedTagRunSampledInfo[tag][runId] =
              tagRunSampledInfo[tag][runId];
          }
        }
      } else {
        tagMetadata[plugin] = tagMetadata[plugin] || {
          tagDescriptions: {},
          runTagInfo: {},
        };
        const {tagDescriptions, runTagInfo} = experimentTagMetadata[plugin];
        tagMetadata[plugin].tagDescriptions = {
          ...tagMetadata[plugin].tagDescriptions,
          ...tagDescriptions,
        };
        tagMetadata[plugin].runTagInfo = {
          ...tagMetadata[plugin].runTagInfo,
          ...runTagInfo,
        };
      }
    }
  }
  return tagMetadata;
}

/**
 * An implementation of MetricsDataSource that treats RunIds as identifiers
 * containing run name and experimentId.
 */
@Injectable()
export class TBMetricsDataSource implements MetricsDataSource {
  private readonly tagCache = new Map<
    string,
    {revision: string; metadata: TagMetadata}
  >();
  private combinedTags?: {
    results: TagMetadata[];
    images: boolean;
    metadata: TagMetadata;
  };
  constructor(
    private readonly http: TBHttpClient,
    private readonly store: Store<FeatureFlagAppState>
  ) {}

  fetchTagMetadata(experimentIds: string[]) {
    const fetches = experimentIds.map((experimentId) => {
      const url = `/experiment/${experimentId}/${HTTP_PATH_PREFIX}/tags`;
      const cached = this.tagCache.get(experimentId);
      return this.http
        .get<
          | BackendTagMetadata
          | {revision: string | null; metadata: BackendTagMetadata | null}
        >(url, {
          headers: new HttpHeaders({
            'X-TensorBoard-Metadata-Revision': cached?.revision ?? '',
          }),
        })
        .pipe(
          map((response) => {
            // Older/custom servers can continue returning the original body.
            if (!('metadata' in response)) {
              this.tagCache.delete(experimentId);
              return buildFrontendTagMetadata(response, experimentId);
            }
            if (response.metadata === null) {
              if (!cached || response.revision !== cached.revision) {
                throw new Error(
                  'Metadata revision response has no matching cached data'
                );
              }
              return cached.metadata;
            }
            const metadata = buildFrontendTagMetadata(
              response.metadata,
              experimentId
            );
            this.tagCache.delete(experimentId);
            if (response.revision) {
              this.tagCache.set(experimentId, {
                revision: response.revision,
                metadata,
              });
              if (this.tagCache.size > 8)
                this.tagCache.delete(this.tagCache.keys().next().value!);
            }
            return metadata;
          })
        );
    });
    const isImagesSupported$ = this.store.select(getIsFeatureFlagsLoaded).pipe(
      filter(Boolean),
      take(1),
      withLatestFrom(this.store.select(getIsMetricsImageSupportEnabled)),
      map(([, isImagesSupported]) => {
        return isImagesSupported;
      })
    );
    return forkJoin(fetches).pipe(
      withLatestFrom(isImagesSupported$),
      map(([results, isImagesSupported]) => {
        const previous = this.combinedTags;
        if (
          previous &&
          previous.images === isImagesSupported &&
          previous.results.length === results.length &&
          results.every((result, index) => result === previous.results[index])
        ) {
          return previous.metadata;
        }
        const tagMetadata = buildCombinedTagMetadata(results);
        if (!isImagesSupported) {
          tagMetadata[PluginType.IMAGES] = {
            tagDescriptions: {},
            tagRunSampledInfo: {},
          };
        }
        this.combinedTags = {
          results,
          images: isImagesSupported,
          metadata: tagMetadata,
        };
        return tagMetadata;
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
