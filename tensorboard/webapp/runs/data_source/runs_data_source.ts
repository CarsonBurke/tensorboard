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
import {Observable} from 'rxjs';
import {map} from 'rxjs/operators';
import {TBHttpClient} from '../../webapp_data_source/tb_http_client';
import {
  Run,
  RunPage,
  RunPageRequest,
  RunsDataSource,
} from './runs_data_source_types';

type BackendGetRunsResponse = Array<
  string | {name: string; start_time: number | null}
>;

function runToRunId(run: string, experimentId: string) {
  return `${experimentId}/${run}`;
}

@Injectable()
export class TBRunsDataSource implements RunsDataSource {
  constructor(private readonly http: TBHttpClient) {}

  fetchRunsPage(
    experimentId: string,
    request: RunPageRequest
  ): Observable<RunPage> {
    const params = new URLSearchParams({
      include_start_time: 'true',
      query_prefix: request.queryPrefix ?? '',
      query: request.query,
      offset: String(request.offset),
      limit: String(request.limit),
      sort_by: request.sortBy,
      descending: String(request.descending),
    });
    const url = `/experiment/${experimentId}/data/runs`;
    type Response = {
      runs: Array<{name: string; start_time: number | null}>;
      total: number;
    };
    const response =
      request.sessionRanks || request.names
        ? this.http.post<Response>(url, {
            query: request.query,
            query_prefix: request.queryPrefix ?? '',
            offset: request.offset,
            limit: request.limit,
            sort_by: request.sortBy,
            descending: request.descending,
            ...(request.names ? {name: request.names} : {}),
            session_ranks: request.sessionRanks ?? [],
            default_rank: request.defaultRank ?? 0,
          })
        : this.http.get<Response>(`${url}?${params}`);
    return response.pipe(
      map(({runs, total}) => ({
        total,
        runs: runs.map(({name, start_time}) => ({
          id: runToRunId(name, experimentId),
          name,
          startTime: start_time ?? undefined,
        })),
      }))
    );
  }

  fetchRuns(experimentId: string): Observable<Run[]> {
    return this.http
      .get<BackendGetRunsResponse>(
        `/experiment/${experimentId}/data/runs?include_start_time=true`
      )
      .pipe(
        map((runs) => {
          return runs.map((run) => {
            const name = typeof run === 'string' ? run : run.name;
            return {
              id: runToRunId(name, experimentId),
              name,
              startTime:
                typeof run === 'string'
                  ? undefined
                  : run.start_time ?? undefined,
            };
          });
        })
      );
  }
}
