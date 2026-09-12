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
 * @fileoverview Module that provides convenience selector for UI.
 *
 * NOTE: For accessing a specific feature, please define and use the selectors
 * provided by the feature.
 *
 * This module provides selectors that access states across multiple features
 * for convenience of the UI[1].
 *
 * [1]: In most cases, one should be able to use combination of rxjs primitives
 * like `mergeMap` and `withLatestFrom` to achieve the same thing.
 */

import {
  createSelector,
  createSelectorFactory,
  defaultMemoize,
  MemoizedSelector,
} from '@ngrx/store';
import {
  getExperimentIdsFromRoute,
  getExperimentIdToExperimentAliasMap,
  getRouteKind,
} from '../app_routing/store/app_routing_selectors';
import {RouteKind} from '../app_routing/types';
import {State} from '../app_state';
import {getDarkModeEnabled} from '../feature_flag/store/feature_flag_selectors';
import {getCardRunLoadStates} from '../metrics/store/metrics_selectors';
import {CardId} from '../metrics/types';
import {
  getDefaultRunColorIdMap,
  getRunColorOverride,
  getRunIdToExperimentId,
  getRuns,
  getDashboardRuns,
  getRunSelectionMap,
  getRunSelectorRegexFilter,
} from '../runs/store/runs_selectors';
import {ExperimentId, RunId} from '../runs/store/runs_types';
import {DataLoadState} from '../types/data';
import {selectors} from '../settings';
import {ColorPalette} from './colors';
import {hasOwn} from './lang';
import {matchRunToRegex, RunMatchable} from './matcher';

/**
 * Creates a copy of RunSelectionMap with entries filtered to runs that
 * belong to one of the current experiments in the route.
 *
 * Unlike `getCurrentRouteRunSelection`, the run selector's regex filter is not
 * applied; the result reflects only what the user checked. Data loading keys
 * off of this so that typing in the filter box neither discards nor refetches
 * series.
 */
export const getRunSelectionMapFilteredToCurrentRoute = createSelector<
  State,
  string[] | null,
  Map<string, boolean>,
  Record<RunId, ExperimentId>,
  Map<string, boolean>
>(
  getExperimentIdsFromRoute,
  getRunSelectionMap,
  getRunIdToExperimentId,
  (experimentIds, runSelectionMap, runIds) => {
    if (!experimentIds) {
      // No experiments in the route means there are no runs to select.
      return new Map<string, boolean>();
    }

    const filteredRunSelectionMap = new Map<string, boolean>();
    for (const [runId, value] of runSelectionMap.entries()) {
      const experimentId = runIds[runId];
      if (experimentId && experimentIds.indexOf(experimentId) >= 0) {
        // Run belongs to one of the Route's experiments. Add it to the filtered
        // result.
        filteredRunSelectionMap.set(runId, value);
      }
    }
    return filteredRunSelectionMap;
  }
);

/**
 * Load state for a multi-run card, limited to runs that are still selected.
 *
 * In-flight bookkeeping for a deselected run remains in metrics state so a
 * quick re-selection does not duplicate its request, but it must not keep the
 * visible card's spinner running.
 *
 * Per-card factory: each card owns its memo cell, so the run filter below runs
 * only when that card's runs or the run selection actually change.
 */
export const getMultiRunCardLoadState = (cardId: CardId) =>
  createSelector(
    getCardRunLoadStates(cardId),
    getRunSelectionMapFilteredToCurrentRoute,
    ({tagRunIds, runToLoadState}, runSelection): DataLoadState => {
      const trackedRunIds = tagRunIds.filter(
        (runId) => runSelection.get(runId) && hasOwn(runToLoadState, runId)
      );
      if (!trackedRunIds.length) {
        return DataLoadState.NOT_LOADED;
      }
      if (
        trackedRunIds.every(
          (runId) => runToLoadState[runId] === DataLoadState.LOADED
        )
      ) {
        return DataLoadState.LOADED;
      }
      return trackedRunIds.some(
        (runId) => runToLoadState[runId] === DataLoadState.LOADING
      )
        ? DataLoadState.LOADING
        : DataLoadState.NOT_LOADED;
    }
  );

const getRunMatchableMap = createSelector(
  getExperimentIdToExperimentAliasMap,
  getDashboardRuns,
  (aliasMap, runs) => {
    const runMatchableMap = new Map<string, RunMatchable>();
    for (const run of runs) {
      runMatchableMap.set(run.id, {
        runName: run.name,
        experimentAlias: aliasMap[run.experimentId],
      });
    }
    return runMatchableMap;
  }
);

function areRunSelectionsEqual(
  a: Map<string, boolean> | null,
  b: Map<string, boolean> | null
): boolean {
  if (a === b) return true;
  if (!a || !b || a.size !== b.size) return false;
  for (const [runId, selected] of a) {
    // Values are booleans, so a missing key (`undefined`) also compares unequal.
    if (b.get(runId) !== selected) return false;
  }
  return true;
}

/**
 * Selects the run selection (runId to boolean) of current set of experiments.
 *
 * Note that emits null when current route is not about an experiment.
 *
 * The regex filter is folded into the selection, so it recomputes on every
 * keystroke. Returning the previous map when the effective selection is
 * unchanged keeps every downstream selector (card lists, run tables, chart
 * requests) memoized.
 */
export const getCurrentRouteRunSelection: MemoizedSelector<
  State,
  Map<string, boolean> | null
> = createSelectorFactory<State, Map<string, boolean> | null>((projector) =>
  defaultMemoize(projector, undefined, areRunSelectionsEqual)
)(
  getExperimentIdsFromRoute,
  getRunSelectionMapFilteredToCurrentRoute,
  getRunSelectorRegexFilter,
  getRunMatchableMap,
  getRouteKind,
  (
    experimentIds: string[] | null,
    runSelection: Map<string, boolean>,
    regexFilter: string,
    runMatchableMap: Map<string, RunMatchable>,
    routeKind: RouteKind
  ) => {
    if (!experimentIds) {
      // There are no experiments in the route. Return null.
      return null;
    }
    if (!regexFilter) {
      // Nothing to match against: the selection is already scoped to the
      // route, and no run has to be resolved to a name or alias.
      return new Map(runSelection);
    }

    const includeExperimentInfo = routeKind === RouteKind.COMPARE_EXPERIMENT;
    const filteredSelection = new Map<string, boolean>();

    for (const [runId, value] of runSelection.entries()) {
      const runMatchable = runMatchableMap.get(runId);
      // A refresh can drop runs while their selection entry lingers in the
      // runs store. Such a run has no name or alias left to match against,
      // so a nonempty filter cannot keep it.
      if (!runMatchable) {
        filteredSelection.set(runId, false);
        continue;
      }
      filteredSelection.set(
        runId,
        matchRunToRegex(runMatchable, regexFilter, includeExperimentInfo) &&
          value
      );
    }
    return filteredSelection;
  }
);

/**
 * Returns Observable that emits map of run id to run color (hex) from
 * current color palettes.
 */
export const getRunColorMap = createSelector<
  State,
  ColorPalette,
  Map<string, number>,
  Map<string, string>,
  boolean,
  {[runId: string]: string}
>(
  selectors.getColorPalette,
  getDefaultRunColorIdMap,
  getRunColorOverride,
  getDarkModeEnabled,
  (
    colorPalette,
    defaultRunColorId,
    colorOverride,
    useDarkMode
  ): Record<string, string> => {
    const colorObject: Record<string, string> = {};
    defaultRunColorId.forEach((colorId, runId) => {
      let colorHexValue = useDarkMode
        ? colorPalette.inactive.darkHex
        : colorPalette.inactive.lightHex;
      if (colorOverride.has(runId)) {
        colorHexValue = colorOverride.get(runId)!;
      } else if (colorId >= 0) {
        const color = colorPalette.colors[colorId % colorPalette.colors.length];
        colorHexValue = useDarkMode ? color.darkHex : color.lightHex;
      }
      colorObject[runId] = colorHexValue;
    });
    return colorObject;
  }
);
