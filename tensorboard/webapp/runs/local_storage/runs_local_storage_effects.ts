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
  getDashboardRuns,
  getEnvironment,
  getExperimentIdsFromRoute,
  getRunColorOverride,
  getRunSelectionMap,
  getRunsTableSortingInfo,
} from '../../selectors';
import * as runsActions from '../actions';
import {catalogCoversAllRuns, Run} from '../types';
import {
  RunLocalStorageDataSource,
  RunLocalStorageState,
} from './run_local_storage_data_source';

const NEWEST_RUN_COLOR = '#fff';

function getNamespace(dataLocation: string, experimentIds: string[] | null) {
  if (!dataLocation || !experimentIds) {
    return null;
  }
  return JSON.stringify({
    dataLocation,
    experimentIds,
  });
}

function getNewestRunId(runs: Run[]): string | undefined {
  let newestRun: Run | null = null;
  for (const run of runs) {
    const newestRunStartTime =
      newestRun?.startTime && newestRun.startTime > 0
        ? newestRun.startTime
        : -Infinity;
    const runStartTime =
      run.startTime && run.startTime > 0 ? run.startTime : -Infinity;
    if (
      !newestRun ||
      runStartTime > newestRunStartTime ||
      (runStartTime === newestRunStartTime && run.name > newestRun.name)
    ) {
      newestRun = run;
    }
  }
  return newestRun?.id;
}

function experimentIdsEqual(
  left: string[] | null,
  right: string[] | null
): boolean {
  if (left === right) {
    return true;
  }
  if (!left || !right || left.length !== right.length) {
    return false;
  }
  return left.every((id, index) => id === right[index]);
}

function pickMap<T>(
  values: Map<string, T>,
  currentRunIds: Set<string>
): Map<string, T> {
  const result = new Map<string, T>();
  for (const [runId, value] of values.entries()) {
    if (currentRunIds.has(runId)) {
      result.set(runId, value);
    }
  }
  return result;
}

function mapToRecord<T>(values: Map<string, T>): Record<string, T> {
  return Object.fromEntries(values.entries());
}

function getAutoNewestRunId(
  colorOverrides: Map<string, string>,
  previousAutoNewestRunId?: string
): string | undefined {
  if (
    previousAutoNewestRunId &&
    colorOverrides.get(previousAutoNewestRunId) === NEWEST_RUN_COLOR
  ) {
    return previousAutoNewestRunId;
  }
  return undefined;
}

@Injectable()
export class RunsLocalStorageEffects {
  readonly hydrateFetchedRunsFromLocalStorage$;
  readonly hydrateExistingRunsFromLocalStorage$;
  readonly syncRunsToLocalStorage$;
  private hydratedNamespace?: string;
  private reportedWithoutNamespace = false;

  constructor(
    private readonly actions$: Actions,
    private readonly store: Store<State>,
    private readonly dataSource: RunLocalStorageDataSource
  ) {
    this.hydrateFetchedRunsFromLocalStorage$ = createEffect(
      () => {
        return this.actions$.pipe(
          ofType(runsActions.fetchRunsSucceeded),
          withLatestFrom(
            this.store.select(getEnvironment),
            this.store.select(getExperimentIdsFromRoute),
            this.store.select(getRunSelectionMap),
            this.store.select(getRunColorOverride)
          ),
          tap(
            ([
              {
                experimentIds: fetchedExperimentIds,
                runsForAllExperiments,
                catalog,
              },
              environment,
              experimentIds,
              currentSelection,
              currentColorOverrides,
            ]) => {
              if (!experimentIdsEqual(fetchedExperimentIds, experimentIds)) {
                return;
              }
              this.hydrateRunsFromLocalStorage(
                environment.data_location,
                experimentIds,
                runsForAllExperiments,
                currentSelection,
                currentColorOverrides,
                {
                  removeWhenEmpty: !catalog,
                  paged: !!catalog && !catalogCoversAllRuns(catalog),
                }
              );
            }
          )
        );
      },
      {dispatch: false}
    );

    this.hydrateExistingRunsFromLocalStorage$ = createEffect(
      () => {
        return this.actions$.pipe(
          ofType(coreActions.environmentLoaded),
          withLatestFrom(
            this.store.select(getExperimentIdsFromRoute),
            this.store.select(getDashboardRuns),
            this.store.select(getRunSelectionMap),
            this.store.select(getRunColorOverride)
          ),
          tap(
            ([
              {environment},
              experimentIds,
              currentRuns,
              currentSelection,
              currentColorOverrides,
            ]) => {
              this.hydrateRunsFromLocalStorage(
                environment.data_location,
                experimentIds,
                currentRuns,
                currentSelection,
                currentColorOverrides,
                {environmentLoaded: true}
              );
            }
          )
        );
      },
      {dispatch: false}
    );

    this.syncRunsToLocalStorage$ = createEffect(
      () => {
        return this.actions$.pipe(
          ofType(
            runsActions.runSelectionToggled,
            runsActions.singleRunSelected,
            runsActions.runPageSelectionToggled,
            runsActions.runColorChanged,
            runsActions.runGroupByChanged,
            runsActions.runsTableSortingInfoChanged
          ),
          withLatestFrom(
            this.store.select(getEnvironment),
            this.store.select(getExperimentIdsFromRoute),
            this.store.select(getDashboardRuns),
            this.store.select(getRunSelectionMap),
            this.store.select(getRunColorOverride),
            this.store.select(getRunsTableSortingInfo)
          ),
          tap(
            ([
              ,
              environment,
              experimentIds,
              currentRuns,
              currentSelection,
              currentColorOverrides,
              sortingInfo,
            ]) => {
              const namespace = getNamespace(
                environment.data_location,
                experimentIds
              );
              if (!namespace) {
                return;
              }

              const selection = new Map(currentSelection);
              const colorOverrides = new Map(currentColorOverrides);
              const storedState = this.dataSource.getState(
                namespace,
                currentRuns
              );
              const autoNewestRunId = getAutoNewestRunId(
                colorOverrides,
                storedState.newestRunId
              );

              const nextState: RunLocalStorageState = {
                selection,
                colorOverrides,
                sortingInfo,
              };
              if (autoNewestRunId) {
                nextState.newestRunId = autoNewestRunId;
              }
              this.dataSource.setState(namespace, currentRuns, nextState);
            }
          )
        );
      },
      {dispatch: false}
    );
  }

  private hydrateRunsFromLocalStorage(
    dataLocation: string,
    experimentIds: string[] | null,
    currentRuns: Run[],
    currentSelection: Map<string, boolean>,
    currentColorOverrides: Map<string, string>,
    {
      removeWhenEmpty = false,
      paged = false,
      environmentLoaded = false,
    }: {
      /** Whether an empty run list should clear the stored state. */
      removeWhenEmpty?: boolean;
      /**
       * Whether `currentRuns` is only a window of the catalog. A window cannot
       * name the newest run, so auto coloring stays untouched.
       */
      paged?: boolean;
      /** Whether the environment, and with it the namespace, is known. */
      environmentLoaded?: boolean;
    }
  ) {
    const namespace = getNamespace(dataLocation, experimentIds);
    if (!namespace) {
      // A run response can land before the environment names the data
      // location, and reporting then would hand unseen runs the default
      // selection. Only a loaded environment proves nothing can be persisted,
      // and the store defers its default until some restore is reported.
      if (environmentLoaded && !this.reportedWithoutNamespace) {
        this.reportedWithoutNamespace = true;
        this.store.dispatch(
          runsActions.runLocalStorageHydrated({
            runIds: currentRuns.map(({id}) => id),
            selection: mapToRecord(currentSelection),
            colorOverrides: mapToRecord(currentColorOverrides),
            restoredSelection: false,
          })
        );
      }
      return;
    }
    if (!currentRuns.length && !paged && removeWhenEmpty) {
      this.dataSource.setState(namespace, [], {
        selection: new Map(),
        colorOverrides: new Map(),
      });
    }

    const currentRunIds = new Set(currentRuns.map((run) => run.id));
    const storedState = this.dataSource.getState(namespace, currentRuns);
    const restoreSelection = this.hydratedNamespace !== namespace;
    // Other tabs share storage, not this view's live selection. Restore once
    // on entry; fetching another page or reloading data must not select runs.
    const persistedSelection = restoreSelection
      ? storedState.selection
      : new Map<string, boolean>();
    for (const [id, selected] of currentSelection)
      if (selected) currentRunIds.add(id);
    for (const id of persistedSelection.keys()) currentRunIds.add(id);
    for (const id of storedState.colorOverrides.keys()) currentRunIds.add(id);
    const selection = new Map([
      ...pickMap(currentSelection, currentRunIds),
      ...persistedSelection,
    ]);
    const colorOverrides = new Map([
      ...pickMap(currentColorOverrides, currentRunIds),
      ...storedState.colorOverrides,
    ]);
    const newestRunId = getNewestRunId(currentRuns);
    let autoNewestRunId = getAutoNewestRunId(
      colorOverrides,
      storedState.newestRunId
    );

    if (
      !paged &&
      currentRuns.length &&
      storedState.newestRunId &&
      storedState.newestRunId !== newestRunId &&
      colorOverrides.get(storedState.newestRunId) === NEWEST_RUN_COLOR
    ) {
      colorOverrides.delete(storedState.newestRunId);
      autoNewestRunId = undefined;
    }
    if (!paged && newestRunId && !colorOverrides.get(newestRunId)) {
      colorOverrides.set(newestRunId, NEWEST_RUN_COLOR);
      autoNewestRunId = newestRunId;
    }

    this.store.dispatch(
      runsActions.runLocalStorageHydrated({
        runIds: Array.from(currentRunIds),
        selection: mapToRecord(selection),
        colorOverrides: mapToRecord(colorOverrides),
        restoredSelection: restoreSelection && storedState.selection.size > 0,
      })
    );
    const restoreSorting = this.hydratedNamespace !== namespace;
    this.hydratedNamespace = namespace;
    if (storedState.sortingInfo && restoreSorting) {
      this.store.dispatch(
        runsActions.runsTableSortingInfoChanged({
          sortingInfo: storedState.sortingInfo,
        })
      );
    }
    if (!currentRuns.length) {
      // Runs are not loaded yet; keeping the stored state untouched avoids
      // pruning selections for runs this pass cannot see.
      return;
    }
    const nextState: RunLocalStorageState = {
      selection,
      colorOverrides,
    };
    if (autoNewestRunId) {
      nextState.newestRunId = autoNewestRunId;
    }
    if (storedState.sortingInfo) {
      nextState.sortingInfo = storedState.sortingInfo;
    }
    this.dataSource.setState(namespace, currentRuns, nextState);
  }
}

export const TEST_ONLY = {
  experimentIdsEqual,
  getNewestRunId,
  getNamespace,
  NEWEST_RUN_COLOR,
};
