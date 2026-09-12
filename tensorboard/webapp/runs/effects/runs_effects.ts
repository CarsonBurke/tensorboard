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
import {Actions, createEffect, ofType} from '@ngrx/effects';
import {Store} from '@ngrx/store';
import {EMPTY, forkJoin, merge, Observable, of, throwError} from 'rxjs';
import {
  catchError,
  distinctUntilChanged,
  filter,
  map,
  mergeMap,
  take,
  switchMap,
  tap,
  withLatestFrom,
} from 'rxjs/operators';
import {areSameRouteKindAndExperiments} from '../../app_routing';
import {navigated, stateRehydratedFromUrl} from '../../app_routing/actions';
import {Route, RouteKind} from '../../app_routing/types';
import {State} from '../../app_state';
import * as coreActions from '../../core/actions';
import * as hparamsActions from '../../hparams/_redux/hparams_actions';
import {
  getDashboardSessionGroups,
  getDashboardHparamFilterMap,
  getDashboardMetricsFilterMap,
} from '../../hparams/_redux/hparams_selectors';
import {
  HparamFilter,
  MetricFilter,
} from '../../hparams/_redux/types';
import {SessionGroup} from '../../hparams/_types';
import {ExperimentAlias} from '../../experiments/types';
import {utils as runFilterUtils} from '../../metrics/views/main_view/common_selectors';
import {metricsTagMetadataLoaded} from '../../metrics/actions';
import {
  getActiveRoute,
  getDashboardExperimentNames,
  getExperimentIdsFromRoute,
  getRuns,
  getRunsLoadState,
  getRunSelectionMap,
  getRunSelectorRegexFilter,
  getRunsTableSortingInfo,
  getRunMap,
  getRunCatalog,
  getRunCatalogWindow,
  getExperimentIdToExperimentAliasMap,
  getPinnedCardsWithMetadata,
} from '../../selectors';
import {DataLoadState, LoadState} from '../../types/data';
import {
  ColumnHeaderType,
  SortingInfo,
  SortingOrder,
} from '../../widgets/data_table/types';
import * as actions from '../actions';
import {
  Run,
  RunPageRequest,
  RunsDataSource,
} from '../data_source/runs_data_source_types';
import {ExperimentIdToRuns} from '../types';
import {MAX_NUM_RUNS_TO_ENABLE_BY_DEFAULT} from '../store/runs_types';
import {
  RUN_START_TIME_SORT_KEY,
  sortTableDataItems,
} from '../views/runs_table/sorting_utils';

/**
 * Runs effect for fetching data from the backend.
 */
@Injectable()
export class RunsEffects {
  private requestedRetainedIds = new Set<string>();
  constructor(
    private readonly actions$: Actions,
    private readonly store: Store<State>,
    private readonly runsDataSource: RunsDataSource
  ) {
    this.experimentsWithStaleRunsOnRouteChange$ = this.actions$.pipe(
      ofType(navigated),
      withLatestFrom(this.store.select(getActiveRoute)),
      distinctUntilChanged(([, prevRoute], [, currRoute]) => {
        return areSameRouteKindAndExperiments(prevRoute, currRoute);
      }),
      withLatestFrom(this.store.select(getExperimentIdsFromRoute)),
      filter(([, experimentIds]) => !!experimentIds),
      map(([, experimentIds]) => experimentIds!),
      mergeMap((experimentIds) => {
        return this.getExperimentsWithLoadState(experimentIds, (state) => {
          return (
            state === DataLoadState.FAILED || state === DataLoadState.NOT_LOADED
          );
        }).pipe(
          map((experimentIdsToBeFetched) => {
            return {experimentIds, experimentIdsToBeFetched};
          })
        );
      })
    );
    this.experimentsWithStaleRunsOnReload$ = this.actions$.pipe(
      ofType(coreActions.reload, coreActions.manualReload),
      withLatestFrom(this.store.select(getExperimentIdsFromRoute)),
      filter(([, experimentIds]) => !!experimentIds),
      map(([, experimentIds]) => experimentIds!),
      mergeMap((experimentIds) => {
        return this.getExperimentsWithLoadState(experimentIds, (state) => {
          return state !== DataLoadState.LOADING;
        }).pipe(
          map((experimentIdsToBeFetched) => {
            return {experimentIds, experimentIdsToBeFetched};
          })
        );
      })
    );
    this.loadRunsOnNavigationOrReload$ = createEffect(
      () => {
        return merge(
          this.experimentsWithStaleRunsOnRouteChange$.pipe(
            filter(() => !this.runsDataSource.fetchRunsPage)
          ),
          this.experimentsWithStaleRunsOnReload$,
          this.actions$.pipe(
            ofType(
              actions.runCatalogWindowChanged,
              actions.runSelectorRegexFilterChanged,
              actions.runsTableSortingInfoChanged,
              navigated,
              stateRehydratedFromUrl,
              hparamsActions.hparamsFetchSessionGroupsSucceeded,
              hparamsActions.dashboardHparamFilterAdded,
              hparamsActions.dashboardHparamFilterRemoved,
              hparamsActions.dashboardMetricFilterAdded,
              hparamsActions.dashboardMetricFilterRemoved
            ),
            filter(() => !!this.runsDataSource.fetchRunsPage),
            withLatestFrom(this.store.select(getExperimentIdsFromRoute)),
            map(([, ids]) => ({
              experimentIds: ids ?? [],
              experimentIdsToBeFetched: ids ?? [],
            }))
          ),
          this.actions$.pipe(
            ofType(
              actions.runLocalStorageHydrated,
              actions.runSelectionToggled,
              actions.singleRunSelected,
              actions.runPageSelectionToggled,
              metricsTagMetadataLoaded
            ),
            filter(() => !!this.runsDataSource.fetchRunsPage),
            withLatestFrom(
              this.store.select(getExperimentIdsFromRoute),
              this.store.select(getRunSelectionMap),
              this.store.select(getRunMap),
              this.store.select(getPinnedCardsWithMetadata)
            ),
            filter(([action, ids, selection, runs, pins]) => {
              if (!ids) return false;
              if (action.type === metricsTagMetadataLoaded.type) {
                return pins.some(
                  (pin) =>
                    pin.runId &&
                    !runs.has(pin.runId) &&
                    !this.requestedRetainedIds.has(pin.runId)
                );
              }
              return (
                action.type !== actions.runLocalStorageHydrated.type ||
                [...selection].some(
                  ([id, selected]) =>
                    selected &&
                    !runs.has(id) &&
                    !this.requestedRetainedIds.has(id)
                )
              );
            }),
            map(([, ids]) => ({
              experimentIds: ids!,
              experimentIdsToBeFetched: ids!,
            }))
          )
        ).pipe(
          withLatestFrom(this.store.select(getActiveRoute)),
          // Paged requests belong to the current view. Legacy providers share
          // in-flight full-catalog loads through store state until completion.
          (this.runsDataSource.fetchRunsPage ? switchMap : mergeMap)(
            ([{experimentIds, experimentIdsToBeFetched}, route]) => {
              if (!route || route.routeKind === RouteKind.CARD) return of(null);
              if (this.runsDataSource.fetchRunsPage) {
                return this.fetchRunCatalog(experimentIds);
              }
              return this.fetchAllRunsList(
                experimentIds,
                experimentIdsToBeFetched
              );
            }
          )
        );
      },
      {dispatch: false}
    );
    this.removeHparamFilterWhenColumnIsRemoved$ = createEffect(
      () =>
        this.actions$.pipe(
          ofType(actions.runsTableHeaderRemoved),
          tap(({header}) => {
            if (header.type === ColumnHeaderType.HPARAM) {
              this.store.dispatch(
                hparamsActions.dashboardHparamFilterRemoved({
                  name: header.name,
                })
              );
              return;
            }
            if (header.type === ColumnHeaderType.METRIC) {
              this.store.dispatch(
                hparamsActions.dashboardMetricFilterRemoved({
                  name: header.name,
                })
              );
            }
          })
        ),
      {dispatch: false}
    );

    this.selectAllRuns$ = createEffect(() =>
      this.actions$.pipe(
        ofType(actions.selectAllRuns),
        // Without paged fetching the table already holds every run, so the
        // container toggles the emitted ids directly.
        filter(() => !!this.runsDataSource.fetchRunsPage),
        withLatestFrom(
          this.store.select(getExperimentIdsFromRoute),
          this.store.select(getRunCatalog),
          this.store.select(getRunSelectorRegexFilter),
          this.store.select(getRunsTableSortingInfo),
          this.store.select(getExperimentIdToExperimentAliasMap),
          this.store.select(getActiveRoute),
          this.store.select(getDashboardSessionGroups),
          this.store.select(getDashboardHparamFilterMap),
          this.store.select(getDashboardMetricsFilterMap)
        ),
        filter(([, , catalog]) => !!catalog),
        switchMap(
          ([
            ,
            experimentIds,
            catalog,
            query,
            sorting,
            aliases,
            route,
            sessionGroups,
            hparamFilters,
            metricFilters,
          ]) => {
            try {
              new RegExp(query);
            } catch {
              return of(actions.runPageSelectionToggled({runIds: []}));
            }
            const ids = [...new Set(experimentIds)];
            const totals = catalog!.totals;
            const requests = this.buildRunPageRequests(ids, {
              window: {offset: 0, limit: 0},
              query,
              sorting,
              aliases,
              route,
              sessionGroups,
              hparamFilters,
              metricFilters,
            });
            return forkJoin(
              requests.map((request, index) =>
                this.runsDataSource.fetchRunsPage!(ids[index], {
                  ...request,
                  offset: 0,
                  limit: totals[ids[index]] ?? 0,
                })
              )
            ).pipe(
              map((pages) =>
                actions.runPageSelectionToggled({
                  runIds: pages.flatMap(({runs}) =>
                    runs.map((run) => run.id)
                  ),
                })
              ),
              catchError(() => EMPTY)
            );
          }
        )
      )
    );
  }

  /**
   * Seek a merged sorted catalog without reading the prefix preceding a window.
   * Binary partitions retain only one pivot/probe per experiment; ordinary
   * single-experiment scrolling goes straight to the indexed offset endpoint.
   */
  private fetchMergedRunWindow(
    ids: string[],
    requests: RunPageRequest[],
    window: {offset: number; limit: number}
  ): Observable<{runs: Run[]; totals: Record<string, number>; offset: number}> {
    const fetch = (index: number, offset: number, limit: number) =>
      this.runsDataSource.fetchRunsPage!(ids[index], {
        ...requests[index],
        offset,
        limit,
      });
    if (!ids.length) return of({runs: [], totals: {}, offset: 0});
    if (ids.length === 1) {
      return fetch(0, window.offset, window.limit).pipe(
        switchMap((page) => {
          const offset = Math.min(
            window.offset,
            Math.max(0, page.total - window.limit)
          );
          return offset === window.offset
            ? of({...page, totals: {[ids[0]]: page.total}, offset})
            : fetch(0, offset, window.limit).pipe(
                map((last) => ({
                  runs: last.runs,
                  totals: {[ids[0]]: last.total},
                  offset,
                }))
              );
        })
      );
    }
    // SQLite BINARY and Python use Unicode scalar ordering, not UTF-16 code units.
    const compareNames = (a: string, b: string) => {
      let ai = 0;
      let bi = 0;
      while (ai < a.length && bi < b.length) {
        const ac = a.codePointAt(ai)!;
        const bc = b.codePointAt(bi)!;
        if (ac !== bc) return ac < bc ? -1 : 1;
        ai += ac > 0xffff ? 2 : 1;
        bi += bc > 0xffff ? 2 : 1;
      }
      return ai === a.length ? (bi === b.length ? 0 : -1) : 1;
    };
    const rankMaps = new Map(
      ids.map((id, index) => [
        id,
        {
          ranks: new Map(
            requests[index].sessionRanks?.map(({prefix, rank}) => [
              prefix,
              rank,
            ])
          ),
          defaultRank: requests[index].defaultRank ?? 0,
        },
      ])
    );
    const sessionRank = (run: Run) => {
      const experimentId = run.id.slice(0, run.id.length - run.name.length - 1);
      const scope = rankMaps.get(experimentId)!;
      for (let length = run.name.length; length >= 0; length--) {
        const rank = scope.ranks.get(run.name.slice(0, length));
        if (rank !== undefined) return rank;
      }
      return scope.defaultRank;
    };
    const compare = (a: Run, b: Run) => {
      let order = 0;
      if (requests[0].sortBy === 'start_time') {
        const aTime = a.startTime ?? Infinity;
        const bTime = b.startTime ?? Infinity;
        order = aTime === bTime ? 0 : aTime < bTime ? -1 : 1;
      } else if (requests[0].sortBy === 'session_rank') {
        order = sessionRank(a) - sessionRank(b);
      }
      if (!order) order = compareNames(a.name, b.name);
      if (requests[0].descending) order = -order;
      // Run IDs provide a total order for identical names across experiments.
      return order || compareNames(a.id, b.id);
    };
    return forkJoin(ids.map((_, index) => fetch(index, 0, 1))).pipe(
      switchMap((first) => {
        const totals = Object.fromEntries(
          first.map((page, index) => [ids[index], page.total])
        );
        const total = first.reduce((sum, page) => sum + page.total, 0);
        const offset = Math.min(
          window.offset,
          Math.max(0, total - window.limit)
        );
        const lowerBound = (
          index: number,
          pivot: Run,
          low: number,
          high: number
        ): Observable<number> => {
          if (low >= high) return of(low);
          const mid = Math.floor((low + high) / 2);
          return fetch(index, mid, 1).pipe(
            switchMap((page) =>
              page.runs.length && compare(page.runs[0], pivot) < 0
                ? lowerBound(index, pivot, mid + 1, high)
                : lowerBound(index, pivot, low, mid)
            )
          );
        };
        const seek = (low: number[], high: number[]): Observable<number[]> => {
          if (low.reduce((sum, value) => sum + value, 0) >= offset)
            return of(low);
          let index = 0;
          for (let i = 1; i < ids.length; i++) {
            if (high[i] - low[i] > high[index] - low[index]) index = i;
          }
          if (low[index] >= high[index]) return of(low);
          const mid = Math.floor((low[index] + high[index]) / 2);
          return fetch(index, mid, 1).pipe(
            switchMap((page) => {
              if (!page.runs.length) {
                const nextHigh = [...high];
                nextHigh[index] = mid;
                return seek(low, nextHigh);
              }
              return forkJoin(
                ids.map((_, i) => lowerBound(i, page.runs[0], low[i], high[i]))
              ).pipe(
                switchMap((cuts) => {
                  const rank = cuts.reduce((sum, value) => sum + value, 0);
                  if (rank === offset) return of(cuts);
                  if (rank > offset) return seek(low, cuts);
                  cuts[index]++;
                  return seek(cuts, high);
                })
              );
            })
          );
        };
        return seek(
          ids.map(() => 0),
          first.map((page) => page.total)
        ).pipe(
          switchMap((starts) =>
            forkJoin(ids.map((_, i) => fetch(i, starts[i], window.limit)))
          ),
          map((pages) => ({
            runs: pages
              .flatMap((page) => page.runs)
              .sort(compare)
              .slice(0, window.limit),
            totals,
            offset,
          }))
        );
      })
    );
  }

  /**
   * Builds one paged-run request per experiment for the given table scope.
   *
   * Shared by the windowed catalog fetch and select-all so both resolve the
   * same filtered run list; callers only differ in the window they ask for.
   */
  private buildRunPageRequests(
    experimentIds: string[],
    scope: {
      window: {offset: number; limit: number};
      query: string;
      sorting: SortingInfo;
      aliases: {[id: string]: ExperimentAlias};
      route: Route | null;
      sessionGroups: SessionGroup[];
      hparamFilters: Map<string, HparamFilter>;
      metricFilters: Map<string, MetricFilter>;
    }
  ): RunPageRequest[] {
    const ids = [...new Set(experimentIds)];
    const {
      window,
      query,
      sorting,
      aliases,
      route,
      sessionGroups,
      hparamFilters,
      metricFilters,
    } = scope;
    const sessionSort =
      sorting.name !== 'run' && sorting.name !== RUN_START_TIME_SORT_KEY;
    const aliasSort = sorting.name === 'experimentAlias';
    const ranked =
      sessionSort && !aliasSort
        ? sortTableDataItems(
            [
              ...sessionGroups.map((group, index) => ({
                ...group.hparams,
                id: String(index),
              })),
              {id: 'undefined'},
            ],
            sorting
          )
        : [];
    const ranks = new Map(ranked.map(({id}, rank) => [id, rank]));
    const missingMatches = [
      ...hparamFilters.values(),
      ...metricFilters.values(),
    ].every((filter) => filter.includeUndefined);
    const metricsMatch = [...metricFilters.values()].every(
      (filter) => filter.includeUndefined
    );
    const aliasIds = [...ids].sort(
      (a, b) =>
        ((aliases[a]?.aliasNumber ?? Infinity) -
          (aliases[b]?.aliasNumber ?? Infinity)) *
        (sorting.order === SortingOrder.DESCENDING ? -1 : 1)
    );
    return ids.map(
      (experimentId): RunPageRequest => ({
        ...window,
        query: query ? `(?i)${query}` : '',
        queryPrefix:
          route?.routeKind === RouteKind.COMPARE_EXPERIMENT
            ? aliases[experimentId]?.aliasText ?? ''
            : '',
        sortBy: sessionSort
          ? 'session_rank'
          : sorting.name === 'run'
          ? 'name'
          : 'start_time',
        descending:
          !sessionSort && sorting.order === SortingOrder.DESCENDING,
        ...(sessionSort || hparamFilters.size || metricFilters.size
          ? {
              defaultRank: missingMatches
                ? aliasSort
                  ? aliasIds.indexOf(experimentId)
                  : ranks.get('undefined') ?? 0
                : -1,
              sessionRanks: sessionGroups.flatMap((group, index) => {
                const matches =
                  metricsMatch &&
                  [...hparamFilters].every(([name, filter]) =>
                    runFilterUtils.matchFilter(filter, group.hparams[name])
                  );
                const rank = matches
                  ? aliasSort
                    ? aliasIds.indexOf(experimentId)
                    : ranks.get(String(index)) ?? 0
                  : -1;
                return group.sessions
                  .filter(({name}) => name.startsWith(`${experimentId}/`))
                  .map(({name}) => ({
                    prefix: name.slice(experimentId.length + 1),
                    rank,
                  }));
              }),
            }
          : {}),
      })
    );
  }

  private fetchRunCatalog(experimentIds: string[]) {
    return of(null).pipe(
      withLatestFrom(
        this.store.select(getRunCatalogWindow),
        this.store.select(getRunSelectorRegexFilter),
        this.store.select(getRunsTableSortingInfo),
        this.store.select(getRunSelectionMap),
        this.store.select(getExperimentIdToExperimentAliasMap),
        this.store.select(getActiveRoute),
        this.store.select(getPinnedCardsWithMetadata),
        this.store.select(getDashboardSessionGroups),
        this.store.select(getDashboardHparamFilterMap),
        this.store.select(getDashboardMetricsFilterMap)
      ),
      switchMap(
        ([
          ,
          requestedWindow,
          query,
          sorting,
          selection,
          aliases,
          route,
          pins,
          sessionGroups,
          hparamFilters,
          metricFilters,
        ]) => {
          const window =
            selection.size || requestedWindow.offset
              ? requestedWindow
              : {
                  ...requestedWindow,
                  limit: Math.max(
                    requestedWindow.limit,
                    MAX_NUM_RUNS_TO_ENABLE_BY_DEFAULT
                  ),
                };
          try {
            new RegExp(query);
          } catch {
            return of(null);
          }
          const retainedIds = new Set([
            ...[...selection]
              .filter(([, selected]) => selected)
              .map(([id]) => id),
            ...pins.flatMap((pin) => (pin.runId ? [pin.runId] : [])),
          ]);
          this.requestedRetainedIds = retainedIds;
          this.store.dispatch(
            actions.fetchRunsRequested({
              experimentIds,
              requestedExperimentIds: experimentIds,
            })
          );
          const ids = [...new Set(experimentIds)];
          const requests = this.buildRunPageRequests(ids, {
            window,
            query,
            sorting,
            aliases,
            route,
            sessionGroups,
            hparamFilters,
            metricFilters,
          });
          const retainedRequests = ids.map((experimentId, index) => {
            const names = [...retainedIds]
              .filter((id) => id.startsWith(`${experimentId}/`))
              .map((id) => id.slice(experimentId.length + 1));
            const {sessionRanks, ...retainedRequest} = requests[index];
            return names.length
              ? this.runsDataSource.fetchRunsPage!(experimentId, {
                  ...retainedRequest,
                  query: '',
                  offset: 0,
                  limit: 0,
                  names,
                  defaultRank: 0,
                  sortBy: 'name',
                })
              : of({runs: [], total: 0});
          });
          return forkJoin([
            this.fetchMergedRunWindow(ids, requests, window),
            retainedRequests.length ? forkJoin(retainedRequests) : of([]),
          ]).pipe(
            withLatestFrom(this.store.select(getDashboardExperimentNames)),
            tap(([[catalog, retained], expNameByExpId]) => {
              const newRuns: ExperimentIdToRuns = {};
              const runsForAllExperiments: Run[] = [];
              ids.forEach((experimentId, index) => {
                const runs = [
                  ...new Map(
                    [
                      ...catalog.runs.filter((run) =>
                        run.id.startsWith(`${experimentId}/`)
                      ),
                      ...retained[index].runs,
                    ].map((run) => [run.id, run])
                  ).values(),
                ];
                newRuns[experimentId] = {runs};
                runsForAllExperiments.push(...runs);
              });
              this.store.dispatch(
                actions.fetchRunsSucceeded({
                  experimentIds,
                  newRuns,
                  runsForAllExperiments,
                  expNameByExpId,
                  catalog: {
                    runIds: catalog.runs.map(({id}) => id),
                    totals: catalog.totals,
                    offset: catalog.offset,
                  },
                })
              );
            }),
            catchError(() => {
              this.store.dispatch(
                actions.fetchRunsFailed({
                  experimentIds,
                  requestedExperimentIds: experimentIds,
                })
              );
              return of(null);
            })
          );
        }
      )
    );
  }

  private getRunsListLoadState(experimentId: string): Observable<LoadState> {
    return this.store.select(getRunsLoadState, {experimentId}).pipe(take(1));
  }

  private getExperimentsWithLoadState(
    experimentIds: string[],
    loadStateMatcher: (loadState: DataLoadState) => boolean
  ) {
    return forkJoin(
      experimentIds.map((eid) => {
        return this.getRunsListLoadState(eid);
      })
    ).pipe(
      map((loadStates) => {
        return experimentIds.filter((unused, index) => {
          return loadStateMatcher(loadStates[index].state);
        });
      })
    );
  }

  private readonly experimentsWithStaleRunsOnRouteChange$;

  private readonly experimentsWithStaleRunsOnReload$;

  /**
   * Fetches runs on navigation or in-app reload.
   *
   * @export
   */
  loadRunsOnNavigationOrReload$;

  /**
   * Resolves the full table scope for select-all in paged mode.
   *
   * @export
   */
  selectAllRuns$;

  /**
   * Removes hparam filter when column is removed.
   *
   * @export
   */
  removeHparamFilterWhenColumnIsRemoved$;

  /**
   * IMPORTANT: actions are dispatched even when there are no experiments to
   * fetch.
   *
   * Observable organization:
   * 1. dispatch requested action
   * 2. make requests for experiments that require fetching while waiting for
   *    runs if already loading and return runs
   * 3. combine the result from local + server where server data takaes
   *    precedence.
   * 4. dispatch succeeded if successful. else, dispatch failed.
   */
  private fetchAllRunsList(
    experimentIds: string[],
    experimentIdsToBeFetched: string[]
  ): Observable<null> {
    return of({experimentIds, experimentIdsToBeFetched}).pipe(
      tap(() => {
        this.store.dispatch(
          actions.fetchRunsRequested({
            experimentIds,
            requestedExperimentIds: experimentIdsToBeFetched,
          })
        );
      }),
      mergeMap(() => {
        const eidsToBeFetched = new Set(experimentIdsToBeFetched);

        const fetchOrGetRuns = experimentIds.map((experimentId) => {
          if (eidsToBeFetched.has(experimentId)) {
            return this.fetchRunsForExperiment(experimentId);
          }
          return this.maybeWaitForRunsAndGetRuns(experimentId);
        });
        return forkJoin(fetchOrGetRuns);
      }),
      map((runsAndMedataList) => {
        const newRuns: ExperimentIdToRuns = {};
        const runsForAllExperiments = [];

        for (const runsAndMedata of runsAndMedataList) {
          runsForAllExperiments.push(...runsAndMedata.runs);
          if (runsAndMedata.fromRemote) {
            newRuns[runsAndMedata.experimentId] = {
              runs: runsAndMedata.runs,
            };
          }
        }
        return {newRuns, runsForAllExperiments};
      }),
      withLatestFrom(this.store.select(getDashboardExperimentNames)),
      tap(([runsData, expNameByExpId]) => {
        const {newRuns, runsForAllExperiments} = runsData;
        this.store.dispatch(
          actions.fetchRunsSucceeded({
            experimentIds,
            newRuns,
            runsForAllExperiments,
            expNameByExpId,
          })
        );
      }),
      catchError((error) => {
        this.store.dispatch(
          actions.fetchRunsFailed({
            experimentIds,
            requestedExperimentIds: experimentIdsToBeFetched,
          })
        );
        return of(null);
      }),
      map(() => null)
    );
  }

  private maybeWaitForRunsAndGetRuns(experimentId: string): Observable<{
    fromRemote: false;
    experimentId: string;
    runs: Run[];
  }> {
    return this.store.select(getRunsLoadState, {experimentId}).pipe(
      filter((loadState) => loadState.state !== DataLoadState.LOADING),
      take(1),
      mergeMap((loadState) => {
        if (loadState.state === DataLoadState.FAILED) {
          return throwError(new Error('Pending request failed'));
        }
        return of(loadState);
      }),
      withLatestFrom(this.store.select(getRuns, {experimentId})),
      map(([, runs]) => ({fromRemote: false, experimentId, runs}))
    );
  }

  private fetchRunsForExperiment(experimentId: string): Observable<{
    fromRemote: true;
    experimentId: string;
    runs: Run[];
  }> {
    return this.runsDataSource.fetchRuns(experimentId).pipe(
      map((runs) => {
        return {
          fromRemote: true,
          experimentId,
          runs: runs as Run[],
        };
      })
    );
  }
}
