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
import {Actions, createEffect, ofType, OnInitEffects} from '@ngrx/effects';
import {Action, createAction, createSelector, Store} from '@ngrx/store';
import {
  EMPTY,
  asapScheduler,
  combineLatest,
  forkJoin,
  merge,
  Observable,
  of,
} from 'rxjs';
import {
  catchError,
  debounceTime,
  distinctUntilChanged,
  filter,
  finalize,
  map,
  mergeMap,
  observeOn,
  switchMap,
  take,
  takeUntil,
  tap,
  withLatestFrom,
} from 'rxjs/operators';
import {State} from '../../app_state';
import * as coreActions from '../../core/actions';
import {getActivePlugin} from '../../core/store';
import * as runsActions from '../../runs/actions';
import * as selectors from '../../selectors';
import {DataLoadState} from '../../types/data';
import {selectors as settingsSelectors} from '../../settings';
import * as actions from '../actions';
import {
  isFailedTimeSeriesResponse,
  isSingleRunPlugin,
  isSingleRunTimeSeriesRequest,
  MetricsDataSource,
  METRICS_PLUGIN_ID,
  TagMetadata,
  TimeSeriesRequest,
  TimeSeriesResponse,
  SavedPinsDataSource,
  Tag,
} from '../data_source/index';
import {
  getCardMetadata,
  getCardRunLoadStates,
  getPinnedCardsWithMetadata,
} from '../store';
import {CardId, CardMetadata, PluginType} from '../types';

export type CardFetchInfo = CardMetadata & {
  id: CardId;
  /** Runs that this card's tag is known to have. */
  tagRunIds: string[];
  /** Load state of every run requested for this card so far. */
  runToLoadState: {[runId: string]: DataLoadState};
};

const getCardFetchInfo = createSelector(
  getCardMetadata,
  (state: State, cardId: CardId) => getCardRunLoadStates(cardId)(state),
  (maybeMetadata, runLoadStates, cardId /* props */): CardFetchInfo | null => {
    if (!maybeMetadata) {
      return null;
    }
    return {
      ...maybeMetadata,
      id: cardId,
      tagRunIds: runLoadStates.tagRunIds,
      runToLoadState: runLoadStates.runToLoadState,
    };
  }
);

/**
 * Whether a run's series must be (re)requested. Runs already in flight are
 * never requested again. Runs that are loaded, or that failed, are requested
 * only on an explicit reload, so scrolling cannot retry a failing backend.
 */
function shouldFetchRun(
  runLoadState: DataLoadState | undefined,
  refetchLoaded: boolean
): boolean {
  if (runLoadState === DataLoadState.LOADING) {
    return false;
  }
  return (
    refetchLoaded ||
    runLoadState === undefined ||
    runLoadState === DataLoadState.NOT_LOADED
  );
}

const initAction = createAction('[Metrics Effects] Init');

@Injectable()
export class MetricsEffects implements OnInitEffects {
  /** @export */
  ngrxOnInitEffects(): Action {
    return initAction();
  }

  private readonly reloadRequestedWhileShown$;

  private readonly loadTagMetadata$;

  private getVisibleCardFetchInfos(): Observable<CardFetchInfo[]> {
    const visibleCardIds$ = this.store.select(selectors.getVisibleCardIdSet);
    return visibleCardIds$.pipe(
      switchMap((cardIds) => {
        // Explicitly notify subscribers that there are no visible cards,
        // since `forkJoin` does not emit when passed an empty array.
        if (!cardIds.size) {
          return of([]);
        }
        const observables = [...cardIds].map((cardId) => {
          return this.store.select(getCardFetchInfo, cardId).pipe(take(1));
        });
        return forkJoin(observables);
      }),
      map((fetchInfos) => {
        return fetchInfos.filter(Boolean) as CardFetchInfo[];
      })
    );
  }

  private fetchTimeSeries(requests: TimeSeriesRequest[]) {
    let pending = true;
    return this.metricsDataSource.fetchTimeSeries(requests).pipe(
      tap((responses: TimeSeriesResponse[]) => {
        pending = false;
        const errors = responses.filter(isFailedTimeSeriesResponse);
        if (errors.length) {
          console.error('Time series response contained errors:', errors);
        }

        const requestResponses: Array<{
          request: TimeSeriesRequest;
          response: TimeSeriesResponse;
        }> = [];
        requests.forEach((request, index) => {
          const response = responses[index];
          if (response) {
            requestResponses.push({request, response});
          } else {
            this.store.dispatch(actions.fetchTimeSeriesFailed({request}));
          }
        });
        if (requestResponses.length) {
          this.store.dispatch(
            actions.fetchTimeSeriesLoaded({requestResponses})
          );
        }
      }),
      catchError(() => {
        pending = false;
        for (const request of requests) {
          this.store.dispatch(actions.fetchTimeSeriesFailed({request}));
        }
        return of(null);
      }),
      finalize(() => {
        if (pending) {
          this.store.dispatch(actions.timeSeriesRequestsCancelled({requests}));
        }
      })
    );
  }

  /**
   * Builds and issues the time series requests missing for `fetchInfos`.
   *
   * Multi-run cards request only the selected runs that the card's tag
   * actually has; a card with no such run is not requested at all. Runs absent
   * from `runSelection` are runs whose experiment has not been loaded yet, so
   * they are left to a later `fetchRunsSucceeded`.
   */
  private fetchTimeSeriesForCards(
    fetchInfos: CardFetchInfo[],
    experimentIds: string[],
    runSelection: Map<string, boolean>,
    refetchLoaded: boolean
  ) {
    const requests: TimeSeriesRequest[] = [];
    const requestKeys = new Set<string>();
    for (const fetchInfo of fetchInfos) {
      const {plugin, tag, runId, sample, tagRunIds, runToLoadState} = fetchInfo;
      let partialRequest: TimeSeriesRequest;
      if (isSingleRunPlugin(plugin)) {
        if (!shouldFetchRun(runToLoadState[runId!], refetchLoaded)) {
          continue;
        }
        partialRequest = {plugin, tag, runId: runId!};
      } else {
        const runIds = tagRunIds.filter((tagRunId) => {
          if (!runSelection.get(tagRunId)) {
            return false;
          }
          return shouldFetchRun(runToLoadState[tagRunId], refetchLoaded);
        });
        if (!runIds.length) {
          continue;
        }
        partialRequest = {plugin, tag, experimentIds, runIds};
      }
      if (sample !== undefined) {
        partialRequest.sample = sample;
      }
      const requestKey = JSON.stringify(partialRequest);
      if (!requestKeys.has(requestKey)) {
        requestKeys.add(requestKey);
        requests.push(partialRequest);
      }
    }
    if (!requests.length) {
      return EMPTY;
    }

    // Fetch and handle all visible cards as one batch.
    return of(requests).pipe(
      tap((requests) => {
        this.store.dispatch(actions.multipleTimeSeriesRequested({requests}));
      }),
      mergeMap((requests: TimeSeriesRequest[]) =>
        this.fetchTimeSeries(requests).pipe(
          // Adjacent entries only add requests; they do not tear down work
          // still owned by the buffer. A batch is cancelled if any requested
          // run loses its last owner, then surviving missing runs restart.
          takeUntil(
            combineLatest([
              this.getVisibleCardFetchInfos(),
              this.store.select(selectors.getExperimentIdsFromRoute),
              this.store.select(
                selectors.getRunSelectionMapFilteredToCurrentRoute
              ),
              this.store.select(getActivePlugin),
            ]).pipe(
              filter(
                ([cards, currentExperiments, selection, plugin]) =>
                  plugin !== METRICS_PLUGIN_ID ||
                  JSON.stringify(currentExperiments) !==
                    JSON.stringify(experimentIds) ||
                  requests.some((request) => {
                    const owners = cards.filter(
                      (card) =>
                        card.plugin === request.plugin &&
                        card.tag === request.tag &&
                        card.sample === request.sample
                    );
                    if (isSingleRunTimeSeriesRequest(request)) {
                      return !owners.some(
                        (card) => card.runId === request.runId
                      );
                    }
                    return (
                      !owners.length ||
                      request.runIds!.some((runId) => !selection.get(runId))
                    );
                  })
              )
            )
          ),
          takeUntil(this.reloadRequestedWhileShown$)
        )
      )
    );
  }

  private readonly visibleCardsChanged$;

  private readonly visibleCardsReloaded$;

  private readonly selectedRunsChanged$;

  private readonly tagMetadataLoaded$;

  private readonly purgeUnusedTimeSeries$;

  private readonly loadTimeSeries$;

  private readonly addOrRemovePin$;

  private readonly loadSavedPins$;

  private readonly removeAllPins$;

  private readonly addOrRemovePinsOnToggle$;

  /**
   * In general, this effect dispatch the following actions:
   *
   * On dashboard shown without data loaded:
   * - metricsTagMetadataRequested
   *
   * On changes to the set of cards that are visible:
   * - multipleTimeSeriesRequested
   *
   * On reloads:
   * - metricsTagMetadataRequested
   * - multipleTimeSeriesRequested
   *
   * On data source responses:
   * - metricsTagMetadataLoaded
   * - metricsTagMetadataFailed
   * - fetchTimeSeriesLoaded
   * - fetchTimeSeriesFailed
   */
  /** @export */
  readonly dataEffects$;

  constructor(
    private readonly actions$: Actions,
    private readonly store: Store<State>,
    private readonly metricsDataSource: MetricsDataSource,
    private readonly savedPinsDataSource: SavedPinsDataSource
  ) {
    this.reloadRequestedWhileShown$ = actions$.pipe(
      ofType(coreActions.reload, coreActions.manualReload),
      withLatestFrom(this.store.select(getActivePlugin)),
      filter(([, activePlugin]) => {
        return activePlugin === METRICS_PLUGIN_ID;
      })
    );

    const catalogRequest$ = combineLatest([
      this.store.select(getActivePlugin),
      this.store.select(selectors.getExperimentIdsFromRoute),
      this.store.select(selectors.getRunSelectionMapFilteredToCurrentRoute),
      this.store.select(selectors.getMetricsTagFilter),
      this.store.select(selectors.getMetricsCatalogViewport),
      this.store.select(settingsSelectors.getPageSize),
      this.store.select(getPinnedCardsWithMetadata),
      this.store.select(selectors.getUnresolvedImportedPinnedCards),
      this.store.select(selectors.getMetricsTagGroupExpandedMap),
      this.store.select(selectors.getMetricsTagGroupPageIndexMap),
      this.store.select(selectors.getMetricsFilteredPluginTypes),
    ]).pipe(
      map(
        ([
          plugin,
          experimentIds,
          selection,
          query,
          viewport,
          pageSize,
          pins,
          unresolvedPins,
          expandedGroups,
          groupPages,
          plugins,
        ]) => ({
          plugin,
          experimentIds,
          request: {
            runIds: [...selection]
              .filter(([, selected]) => selected)
              .map(([id]) => id)
              .sort(),
            pinnedRunIds: [
              ...new Set(
                [...pins, ...unresolvedPins].flatMap((pin) =>
                  pin.runId ? [pin.runId] : []
                )
              ),
            ].sort(),
            query,
            plugins: [...plugins].sort(),
            groupOffset: viewport.groupOffset,
            groupLimit: query ? 0 : viewport.groupLimit,
            groups: query
              ? []
              : viewport.visibleGroups
                  .filter((name) => expandedGroups.get(name))
                  .map((name) => ({
                    name,
                    offset: (groupPages.get(name) ?? 0) * pageSize,
                    limit: pageSize,
                  })),
            filteredOffset: viewport.filteredOffset,
            filteredLimit: query ? viewport.filteredLimit : 0,
            pinnedTags: [
              ...new Set([...pins, ...unresolvedPins].map(({tag}) => tag)),
            ].sort(),
          },
        })
      ),
      debounceTime(0, asapScheduler),
      distinctUntilChanged((a, b) => JSON.stringify(a) === JSON.stringify(b))
    );
    this.loadTagMetadata$ = merge(
      catalogRequest$,
      this.reloadRequestedWhileShown$.pipe(
        withLatestFrom(catalogRequest$),
        map(([, request]) => request)
      )
    ).pipe(
      switchMap(({plugin, experimentIds, request}) => {
        if (plugin !== METRICS_PLUGIN_ID || !experimentIds) return EMPTY;
        try {
          new RegExp(request.query);
        } catch {
          return EMPTY;
        }
        this.store.dispatch(actions.metricsTagMetadataRequested());
        return this.metricsDataSource
          .fetchTagMetadata(experimentIds, request)
          .pipe(
            tap((tagMetadata: TagMetadata) => {
              this.store.dispatch(
                actions.metricsTagMetadataLoaded({tagMetadata})
              );
            }),
            catchError(() => {
              this.store.dispatch(actions.metricsTagMetadataFailed());
              return of(null);
            })
          );
      })
    );

    // Each trigger reads the visible cards' fetch state at the time it
    // fires; `withLatestFrom` would reuse the snapshot taken when the
    // visible card set last changed, which goes stale as runs are selected.
    this.visibleCardsChanged$ = this.actions$.pipe(
      ofType(actions.cardVisibilityChanged),
      map(() => ({refetchLoaded: false}))
    );

    this.visibleCardsReloaded$ = this.reloadRequestedWhileShown$.pipe(
      map(() => ({refetchLoaded: true}))
    );

    this.selectedRunsChanged$ = this.actions$.pipe(
      ofType(
        runsActions.runSelectionToggled,
        runsActions.singleRunSelected,
        runsActions.runPageSelectionToggled,
        runsActions.runLocalStorageHydrated,
        runsActions.fetchRunsSucceeded
      ),
      map(() => ({refetchLoaded: false}))
    );

    // New tag metadata can reveal runs that a visible card has never
    // requested; the per-run filter below picks up exactly those.
    this.tagMetadataLoaded$ = this.actions$.pipe(
      ofType(actions.metricsTagMetadataLoaded),
      map(() => ({refetchLoaded: false}))
    );

    this.purgeUnusedTimeSeries$ = this.actions$.pipe(
      ofType(
        runsActions.runSelectionToggled,
        runsActions.singleRunSelected,
        runsActions.runPageSelectionToggled,
        runsActions.runLocalStorageHydrated,
        runsActions.fetchRunsSucceeded,
        actions.fetchTimeSeriesLoaded,
        actions.fetchTimeSeriesFailed,
        actions.cardPinStateToggled,
        actions.metricsClearAllPinnedCards,
        actions.metricsTagMetadataLoaded,
        actions.cardVisibilityChanged,
        actions.timeSeriesRequestsCancelled,
        coreActions.reload,
        coreActions.manualReload,
        coreActions.changePlugin
      ),
      withLatestFrom(
        this.store.select(selectors.getRunSelectionMapFilteredToCurrentRoute)
      ),
      tap(([, runSelection]) => {
        const runIds = [...runSelection]
          .filter(([, selected]) => selected)
          .map(([runId]) => runId);
        this.store.dispatch(actions.unusedTimeSeriesPurged({runIds}));
      })
    );

    this.loadTimeSeries$ = merge(
      this.visibleCardsChanged$,
      this.visibleCardsReloaded$,
      this.selectedRunsChanged$,
      this.tagMetadataLoaded$,
      this.actions$.pipe(
        ofType(coreActions.changePlugin),
        map(() => ({refetchLoaded: false}))
      )
    ).pipe(
      withLatestFrom(
        this.store.select(selectors.getExperimentIdsFromRoute),
        this.store.select(selectors.getRunSelectionMapFilteredToCurrentRoute),
        this.store.select(getActivePlugin)
      ),
      mergeMap(([{refetchLoaded}, experimentIds, runSelection, plugin]) =>
        this.getVisibleCardFetchInfos().pipe(
          take(1),
          map((cards) => ({
            refetchLoaded,
            experimentIds,
            runSelection,
            plugin,
            scope: JSON.stringify([
              plugin,
              experimentIds,
              [
                ...new Set(
                  cards.map(({plugin, tag, runId, sample, tagRunIds}) =>
                    JSON.stringify([
                      plugin,
                      tag,
                      sample,
                      isSingleRunPlugin(plugin)
                        ? [runId]
                        : tagRunIds.filter((id) => runSelection.get(id)).sort(),
                    ])
                  )
                ),
              ].sort(),
            ]),
          }))
        )
      ),
      distinctUntilChanged(
        (previous, next) => !next.refetchLoaded && previous.scope === next.scope
      ),
      mergeMap(({refetchLoaded, experimentIds, plugin}) => {
        if (plugin !== METRICS_PLUGIN_ID || !experimentIds) return EMPTY;
        // NgRx queues cancellation reducers while the triggering action is
        // dispatching. Read statuses after that queue drains, not in finalize.
        return of(null).pipe(
          observeOn(asapScheduler),
          switchMap(() => this.getVisibleCardFetchInfos().pipe(take(1))),
          withLatestFrom(
            this.store.select(
              selectors.getRunSelectionMapFilteredToCurrentRoute
            ),
            this.store.select(selectors.getExperimentIdsFromRoute),
            this.store.select(getActivePlugin)
          ),
          switchMap(
            ([fetchInfos, selection, currentExperiments, activePlugin]) => {
              if (
                activePlugin !== METRICS_PLUGIN_ID ||
                JSON.stringify(currentExperiments) !==
                  JSON.stringify(experimentIds)
              ) {
                return EMPTY;
              }
              return this.fetchTimeSeriesForCards(
                fetchInfos,
                experimentIds,
                selection,
                refetchLoaded
              );
            }
          )
        );
      })
    );

    this.addOrRemovePin$ = this.actions$.pipe(
      ofType(actions.cardPinStateToggled),
      withLatestFrom(
        this.getVisibleCardFetchInfos(),
        this.store.select(selectors.getEnableGlobalPins),
        this.store.select(selectors.getShouldPersistSettings),
        this.store.select(selectors.getMetricsSavingPinsEnabled)
      ),
      filter(
        ([
          ,
          ,
          enableGlobalPinsFeature,
          shouldPersistSettings,
          isMetricsSavingPinsEnabled,
        ]) =>
          enableGlobalPinsFeature &&
          shouldPersistSettings &&
          isMetricsSavingPinsEnabled
      ),
      tap(([{cardId, canCreateNewPins, wasPinned}, fetchInfos]) => {
        const card = fetchInfos.find((value) => value.id === cardId);
        // Saving only scalar pinned cards.
        if (!card || card.plugin !== PluginType.SCALARS) {
          return;
        }
        if (wasPinned) {
          this.savedPinsDataSource.removeScalarPin(card.tag);
        } else if (canCreateNewPins) {
          this.savedPinsDataSource.saveScalarPin(card.tag);
        }
      })
    );

    this.loadSavedPins$ = this.actions$.pipe(
      // Should be dispatch before stateRehydratedFromUrl.
      ofType(initAction),
      withLatestFrom(
        this.store.select(selectors.getEnableGlobalPins),
        this.store.select(selectors.getShouldPersistSettings),
        this.store.select(selectors.getMetricsSavingPinsEnabled)
      ),
      filter(
        ([
          ,
          enableGlobalPinsFeature,
          shouldPersistSettings,
          isMetricsSavingPinsEnabled,
        ]) =>
          enableGlobalPinsFeature &&
          shouldPersistSettings &&
          isMetricsSavingPinsEnabled
      ),
      tap(() => {
        const tags = this.savedPinsDataSource.getSavedScalarPins();
        if (!tags || tags.length === 0) {
          return;
        }
        const unresolvedPinnedCards = tags.map((tag) => ({
          plugin: PluginType.SCALARS,
          tag: tag,
        }));
        this.store.dispatch(
          actions.metricsUnresolvedPinnedCardsFromLocalStorageAdded({
            cards: unresolvedPinnedCards,
          })
        );
      })
    );

    this.removeAllPins$ = this.actions$.pipe(
      ofType(actions.metricsClearAllPinnedCards),
      withLatestFrom(
        this.store.select(selectors.getEnableGlobalPins),
        this.store.select(selectors.getShouldPersistSettings),
        this.store.select(selectors.getMetricsSavingPinsEnabled)
      ),
      filter(
        ([
          ,
          enableGlobalPinsFeature,
          shouldPersistSettings,
          isMetricsSavingPinsEnabled,
        ]) =>
          enableGlobalPinsFeature &&
          shouldPersistSettings &&
          isMetricsSavingPinsEnabled
      ),
      tap(() => {
        this.savedPinsDataSource.removeAllScalarPins();
      })
    );

    this.addOrRemovePinsOnToggle$ = this.actions$.pipe(
      ofType(actions.metricsEnableSavingPinsToggled),
      withLatestFrom(
        this.store.select(selectors.getPinnedCardsWithMetadata),
        this.store.select(selectors.getEnableGlobalPins),
        this.store.select(selectors.getShouldPersistSettings),
        this.store.select(selectors.getMetricsSavingPinsEnabled)
      ),
      filter(
        ([, , enableGlobalPins, getShouldPersistSettings]) =>
          enableGlobalPins && getShouldPersistSettings
      ),
      tap(([, pinnedCards, , , getMetricsSavingPinsEnabled]) => {
        if (getMetricsSavingPinsEnabled) {
          const tags: Tag[] = pinnedCards
            .map((card) => {
              return card.plugin === PluginType.SCALARS ? card.tag : null;
            })
            .filter((v): v is Tag => v !== null);
          this.savedPinsDataSource.saveScalarPins(tags);
        } else {
          this.savedPinsDataSource.removeAllScalarPins();
        }
      })
    );

    this.dataEffects$ = createEffect(
      () => {
        return merge(
          /**
           * Subscribes to: dashboard shown, route navigation, reloads.
           */
          this.loadTagMetadata$,

          /**
           * Subscribes to: card visibility, reloads, run selection changes.
           */
          this.loadTimeSeries$,
          /**
           * Subscribes to: run selection changes, time series responses.
           */
          this.purgeUnusedTimeSeries$,

          /**
           * Subscribes to: cardPinStateToggled.
           */
          this.addOrRemovePin$,
          /**
           * Subscribes to: dashboard shown (initAction).
           */
          this.loadSavedPins$,
          /**
           * Subscribes to: metricsClearAllPinnedCards.
           */
          this.removeAllPins$,
          /**
           * Subscribes to: metricsEnableSavingPinsToggled.
           */
          this.addOrRemovePinsOnToggle$
        );
      },
      {dispatch: false}
    );
  }
}

export const TEST_ONLY = {
  getCardFetchInfo,
  initAction,
};
