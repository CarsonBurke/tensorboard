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
  getEnvironment,
  getExperimentIdsFromRoute,
  getMetricsTagGroupExpandedMap,
  getMetricsTagGroupPageIndexMap,
  getNonEmptyCardIdsWithMetadata,
} from '../../selectors';
import {DeepReadonly} from '../../util/types';
import * as metricsActions from '../actions';
import {CardIdWithMetadata} from '../types';
import {groupCardIdWithMetdata} from '../utils';
import {
  MetricsLocalStorageDataSource,
  MetricsLocalStorageState,
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

function pickMap<T>(
  values: Map<string, T>,
  currentTagGroups: Set<string>
): Map<string, T> {
  const result = new Map<string, T>();
  for (const [tagGroup, value] of values.entries()) {
    if (currentTagGroups.has(tagGroup)) {
      result.set(tagGroup, value);
    }
  }
  return result;
}

function mapToRecord<T>(values: Map<string, T>): Record<string, T> {
  return Object.fromEntries(values.entries());
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
            this.store.select(getMetricsTagGroupPageIndexMap)
          ),
          tap(
            ([
              ,
              environment,
              experimentIds,
              currentCards,
              currentExpanded,
              currentPageIndex,
            ]) => {
              this.hydrateMetricsFromLocalStorage(
                environment.data_location,
                experimentIds,
                getTagGroups(currentCards),
                currentExpanded,
                currentPageIndex
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
            this.store.select(getMetricsTagGroupPageIndexMap)
          ),
          tap(
            ([
              {environment},
              experimentIds,
              currentCards,
              currentExpanded,
              currentPageIndex,
            ]) => {
              this.hydrateMetricsFromLocalStorage(
                environment.data_location,
                experimentIds,
                getTagGroups(currentCards),
                currentExpanded,
                currentPageIndex
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
            metricsActions.metricsTagGroupPageIndexChanged
          ),
          withLatestFrom(
            this.store.select(getEnvironment),
            this.store.select(getExperimentIdsFromRoute),
            this.store.select(getNonEmptyCardIdsWithMetadata),
            this.store.select(getMetricsTagGroupExpandedMap),
            this.store.select(getMetricsTagGroupPageIndexMap)
          ),
          tap(
            ([
              ,
              environment,
              experimentIds,
              currentCards,
              currentExpanded,
              currentPageIndex,
            ]) => {
              const tagGroups = getTagGroups(currentCards);
              const namespace = getNamespace(
                environment.data_location,
                experimentIds
              );
              if (!namespace || !tagGroups.length) {
                return;
              }

              this.dataSource.setState(namespace, tagGroups, {
                tagGroupExpanded: currentExpanded,
                tagGroupPageIndex: currentPageIndex,
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
    currentPageIndex: Map<string, number>
  ) {
    if (!tagGroups.length) {
      return;
    }
    const namespace = getNamespace(dataLocation, experimentIds);
    if (!namespace) {
      return;
    }

    const currentTagGroups = new Set(tagGroups);
    const storedState = this.dataSource.getState(namespace, tagGroups);
    const tagGroupExpanded = new Map([
      ...pickMap(currentExpanded, currentTagGroups),
      ...storedState.tagGroupExpanded,
    ]);
    const tagGroupPageIndex = new Map([
      ...pickMap(currentPageIndex, currentTagGroups),
      ...storedState.tagGroupPageIndex,
    ]);

    this.store.dispatch(
      metricsActions.metricsLocalStorageHydrated({
        tagGroups,
        tagGroupExpanded: mapToRecord(tagGroupExpanded),
        tagGroupPageIndex: mapToRecord(tagGroupPageIndex),
      })
    );

    const nextState: MetricsLocalStorageState = {
      tagGroupExpanded,
      tagGroupPageIndex,
    };
    this.dataSource.setState(namespace, tagGroups, nextState);
  }
}

export const TEST_ONLY = {
  getNamespace,
};
