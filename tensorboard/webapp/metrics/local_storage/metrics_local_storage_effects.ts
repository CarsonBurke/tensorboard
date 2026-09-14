/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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
import {tap, withLatestFrom} from 'rxjs/operators';
import {State} from '../../app_state';
import * as coreActions from '../../core/actions';
import {
  getCardStateMap,
  getEnvironment,
  getExperimentIdsFromRoute,
  getMetricsTagGroupExpandedMap,
  getMetricsTagGroupPageIndexMap,
  getNonEmptyCardIdsWithMetadata,
} from '../../selectors';
import {DeepReadonly} from '../../util/types';
import * as metricsActions from '../actions';
import {CardStateMap} from '../store/metrics_types';
import {CardId, CardIdWithMetadata} from '../types';
import {groupCardIdWithMetdata} from '../utils';
import {
  MetricsLocalStorageDataSource,
  MetricsLocalStorageState,
  PersistedCardState,
} from './metrics_local_storage_data_source';

function getNamespace(dataLocation: string, experimentIds: string[] | null) {
  if (!dataLocation || !experimentIds) {
    return null;
  }
  return JSON.stringify({
    dataLocation,
    experimentIds,
  });
}

function getTagGroups(cards: DeepReadonly<CardIdWithMetadata[]>): string[] {
  return groupCardIdWithMetdata(cards).map((group) => group.groupName);
}

function mapToRecord<T>(values: Map<string, T>): Record<string, T> {
  return Object.fromEntries(values.entries());
}

/**
 * Narrows the live card state down to the keys we persist, dropping cards that
 * hold none of them.
 */
function projectPersistedCardState(
  cardStateMap: CardStateMap
): Map<CardId, PersistedCardState> {
  const projected = new Map<CardId, PersistedCardState>();
  for (const [cardId, cardState] of Object.entries(cardStateMap)) {
    const persisted: PersistedCardState = {};
    if (cardState.fullWidth !== undefined) {
      persisted.fullWidth = cardState.fullWidth;
    }
    if (cardState.tableExpanded !== undefined) {
      persisted.tableExpanded = cardState.tableExpanded;
    }
    if (cardState.chartHeight !== undefined) {
      persisted.chartHeight = cardState.chartHeight;
    }
    if (cardState.tableHeight !== undefined) {
      persisted.tableHeight = cardState.tableHeight;
    }
    if (Object.keys(persisted).length > 0) {
      projected.set(cardId, persisted);
    }
  }
  return projected;
}

@Injectable()
export class MetricsLocalStorageEffects {
  readonly hydrateFetchedMetadataFromLocalStorage$;
  readonly hydrateExistingMetadataFromLocalStorage$;
  readonly syncMetricsToLocalStorage$;

  constructor(
    private readonly actions$: Actions,
    private readonly store: Store<State>,
    private readonly dataSource: MetricsLocalStorageDataSource
  ) {
    this.hydrateFetchedMetadataFromLocalStorage$ = createEffect(
      () => {
        return this.actions$.pipe(
          ofType(metricsActions.metricsTagMetadataLoaded),
          withLatestFrom(
            this.store.select(getEnvironment),
            this.store.select(getExperimentIdsFromRoute),
            this.store.select(getNonEmptyCardIdsWithMetadata),
            this.store.select(getMetricsTagGroupExpandedMap),
            this.store.select(getMetricsTagGroupPageIndexMap),
            this.store.select(getCardStateMap)
          ),
          tap(
            ([
              ,
              environment,
              experimentIds,
              currentCards,
              currentExpanded,
              currentPageIndex,
              currentCardState,
            ]) => {
              this.hydrateMetricsFromLocalStorage(
                environment.data_location,
                experimentIds,
                getTagGroups(currentCards),
                currentExpanded,
                currentPageIndex,
                currentCardState
              );
            }
          )
        );
      },
      {dispatch: false}
    );

    this.hydrateExistingMetadataFromLocalStorage$ = createEffect(
      () => {
        return this.actions$.pipe(
          ofType(coreActions.environmentLoaded),
          withLatestFrom(
            this.store.select(getExperimentIdsFromRoute),
            this.store.select(getNonEmptyCardIdsWithMetadata),
            this.store.select(getMetricsTagGroupExpandedMap),
            this.store.select(getMetricsTagGroupPageIndexMap),
            this.store.select(getCardStateMap)
          ),
          tap(
            ([
              {environment},
              experimentIds,
              currentCards,
              currentExpanded,
              currentPageIndex,
              currentCardState,
            ]) => {
              this.hydrateMetricsFromLocalStorage(
                environment.data_location,
                experimentIds,
                getTagGroups(currentCards),
                currentExpanded,
                currentPageIndex,
                currentCardState
              );
            }
          )
        );
      },
      {dispatch: false}
    );

    this.syncMetricsToLocalStorage$ = createEffect(
      () => {
        return this.actions$.pipe(
          ofType(
            metricsActions.metricsTagGroupExpansionChanged,
            metricsActions.metricsTagGroupPageIndexChanged,
            metricsActions.metricsCardStateUpdated,
            metricsActions.metricsCardFullSizeToggled
          ),
          withLatestFrom(
            this.store.select(getEnvironment),
            this.store.select(getExperimentIdsFromRoute),
            this.store.select(getNonEmptyCardIdsWithMetadata),
            this.store.select(getMetricsTagGroupExpandedMap),
            this.store.select(getMetricsTagGroupPageIndexMap),
            this.store.select(getCardStateMap)
          ),
          tap(
            ([
              ,
              environment,
              experimentIds,
              currentCards,
              currentExpanded,
              currentPageIndex,
              currentCardState,
            ]) => {
              const tagGroups = getTagGroups(currentCards);
              const namespace = getNamespace(
                environment.data_location,
                experimentIds
              );
              if (!namespace) {
                return;
              }

              this.dataSource.setState(namespace, tagGroups, {
                tagGroupExpanded: currentExpanded,
                tagGroupPageIndex: currentPageIndex,
                cardState: projectPersistedCardState(currentCardState),
              });
            }
          )
        );
      },
      {dispatch: false}
    );
  }

  private hydrateMetricsFromLocalStorage(
    dataLocation: string,
    experimentIds: string[] | null,
    tagGroups: string[],
    currentExpanded: Map<string, boolean>,
    currentPageIndex: Map<string, number>,
    currentCardState: CardStateMap
  ) {
    const namespace = getNamespace(dataLocation, experimentIds);
    if (!namespace) {
      return;
    }
    const storedState = this.dataSource.getState(namespace, tagGroups);
    const tagGroupExpanded = new Map([
      ...currentExpanded,
      ...storedState.tagGroupExpanded,
    ]);
    const tagGroupPageIndex = new Map([
      ...currentPageIndex,
      ...storedState.tagGroupPageIndex,
    ]);
    const cardState = projectPersistedCardState(currentCardState);
    for (const [cardId, stored] of storedState.cardState) {
      cardState.set(cardId, {...cardState.get(cardId), ...stored});
    }
    tagGroups = [
      ...new Set([
        ...tagGroups,
        ...tagGroupExpanded.keys(),
        ...tagGroupPageIndex.keys(),
      ]),
    ];

    this.store.dispatch(
      metricsActions.metricsLocalStorageHydrated({
        tagGroups,
        tagGroupExpanded: mapToRecord(tagGroupExpanded),
        tagGroupPageIndex: mapToRecord(tagGroupPageIndex),
        cardState: mapToRecord(cardState),
      })
    );

    const nextState: MetricsLocalStorageState = {
      tagGroupExpanded,
      tagGroupPageIndex,
      cardState,
    };
    this.dataSource.setState(namespace, tagGroups, nextState);
  }
}

export const TEST_ONLY = {
  getNamespace,
};
