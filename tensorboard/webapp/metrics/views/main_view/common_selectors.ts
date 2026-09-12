/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
import {
  createSelector,
  createSelectorFactory,
  defaultMemoize,
  MemoizedSelector,
} from '@ngrx/store';
import {State} from '../../../app_state';
import {
  getCurrentRouteRunSelection,
  getMetricsHideEmptyCards,
  getMetricsTagMetadata,
  getExperimentIdsFromRoute,
  getExperimentIdToExperimentAliasMap,
  getRunColorMap,
  getRunSelectorRegexFilter,
  getRouteKind,
  getDashboardRuns,
  getColumnHeadersForCard,
  getDashboardExperimentNames,
} from '../../../selectors';
import {DeepReadonly} from '../../../util/types';
import {
  getDashboardDefaultHparamFilters,
  getDashboardDisplayedHparamColumns,
  getDashboardHparamFilterMap,
  getDashboardHparamSpecs,
  getDashboardMetricsFilterMap,
} from '../../../hparams/_redux/hparams_selectors';
import {
  DiscreteFilter,
  DiscreteHparamValue,
  DomainType,
  IntervalFilter,
} from '../../../hparams/types';
import {
  RunTableItem,
  RunTableExperimentItem,
} from '../../../runs/views/runs_table/types';
import {matchRunToRegex} from '../../../util/matcher';
import {isSingleRunPlugin, PluginType} from '../../data_source';
import {
  getNonEmptyCardIdsWithMetadata,
  getMetricsCatalogCards,
  getMetricsFilteredPluginTypes,
  getMetricsTagFilter,
  TagMetadata,
} from '../../store';
import {compareTagNames} from '../../utils';
import {CardIdWithMetadata} from '../metrics_view_types';
import {RouteKind} from '../../../app_routing/types';
import {memoize} from '../../../util/memoize';
import {
  ColumnHeader,
  ColumnHeaderType,
} from '../card_renderer/scalar_card_types';

// Inverse of `tagToRuns`, recomputed only when tag metadata changes.
const getScalarRunToTags = createSelector(
  getMetricsTagMetadata,
  (tagMetadata: DeepReadonly<TagMetadata>): Map<string, string[]> => {
    const runToTags = new Map<string, string[]>();
    for (const [tag, runs] of Object.entries(tagMetadata.scalars.tagToRuns)) {
      for (const run of runs) {
        const tags = runToTags.get(run);
        if (tags) {
          tags.push(tag);
        } else {
          runToTags.set(run, [tag]);
        }
      }
    }
    return runToTags;
  }
);

// Returned while empty cards are shown, when no consumer reads the tags.
// Sharing one instance also keeps dependent selectors memoized.
const EMPTY_SCALAR_TAGS = new Set<string>();

// Module-private: the result is only meaningful while empty cards are hidden,
// so it must not gain a consumer that reads it unconditionally.
const getScalarTagsForRunSelection = createSelector(
  getMetricsHideEmptyCards,
  getMetricsTagMetadata,
  getScalarRunToTags,
  getCurrentRouteRunSelection,
  (
    hideEmptyCards: boolean,
    tagMetadata: DeepReadonly<TagMetadata>,
    runToTags: Map<string, string[]>,
    runSelection: Map<string, boolean> | null
  ): Set<string> => {
    // `getRenderableCardIdsWithMetadata` is the only consumer and only reads
    // these tags while empty cards are hidden, so skip the walk over the
    // selected runs' tags, which would otherwise run on every selection change.
    if (!hideEmptyCards) {
      return EMPTY_SCALAR_TAGS;
    }
    if (!runSelection || !runSelection.size) {
      return new Set(Object.keys(tagMetadata.scalars.tagToRuns));
    }
    // Walk the selected runs rather than every (tag, run) pair: selection
    // changes are frequent and usually involve few runs, while the pair count
    // grows with runs × tags.
    const tags = new Set<string>();
    for (const [runId, selected] of runSelection) {
      if (!selected) continue;
      const runTags = runToTags.get(runId);
      if (!runTags) continue;
      for (const tag of runTags) {
        tags.add(tag);
      }
    }
    return tags;
  }
);

const getSortedNonEmptyCardIdsWithMetadata = createSelector(
  getNonEmptyCardIdsWithMetadata,
  (cardList) => {
    return [...cardList].sort((cardA, cardB) => {
      return compareTagNames(cardA.tag, cardB.tag);
    });
  }
);

function areCardListsEqual(
  previous: DeepReadonly<CardIdWithMetadata>[],
  next: DeepReadonly<CardIdWithMetadata>[]
): boolean {
  if (previous === next) {
    return true;
  }
  if (previous.length !== next.length) {
    return false;
  }
  // Elements are the objects built by `getNonEmptyCardIdsWithMetadata`, which
  // rebuilds them whenever the card list or any card's metadata changes.
  // Comparing identities therefore catches both a changed card id sequence and
  // changed metadata, which consumers read.
  for (let i = 0; i < previous.length; i++) {
    if (previous[i] !== next[i]) {
      return false;
    }
  }
  return true;
}

// A run selection change usually leaves the rendered cards untouched, since
// only `hideEmptyCards` can drop a card. Keeping the previous array in that
// case spares the card groups, each grid's pagination and every `*ngFor` diff.
const getRenderableCardIdsWithMetadata: MemoizedSelector<
  State,
  DeepReadonly<CardIdWithMetadata>[]
> = createSelectorFactory<State, DeepReadonly<CardIdWithMetadata>[]>(
  (projector) => defaultMemoize(projector, undefined, areCardListsEqual)
)(
  getSortedNonEmptyCardIdsWithMetadata,
  getCurrentRouteRunSelection,
  getMetricsHideEmptyCards,
  getScalarTagsForRunSelection,
  (
    cardList: DeepReadonly<CardIdWithMetadata>[],
    runSelectionMap: Map<string, boolean> | null,
    hideEmptyScalarCards: boolean,
    scalarTagsForRunSelection: Set<string>
  ): DeepReadonly<CardIdWithMetadata>[] => {
    let areAnyRunsSelected = false;
    for (const selected of runSelectionMap?.values() || []) {
      if (selected) {
        areAnyRunsSelected = true;
        break;
      }
    }
    return cardList.filter((card) => {
      if (!isSingleRunPlugin(card.plugin)) {
        if (
          hideEmptyScalarCards &&
          areAnyRunsSelected &&
          card.plugin === PluginType.SCALARS
        ) {
          return scalarTagsForRunSelection.has(card.tag);
        }
        return true;
      }
      return Boolean(runSelectionMap && runSelectionMap.get(card.runId!));
    });
  }
);

export const getSortedRenderableCardIdsWithMetadata =
  getRenderableCardIdsWithMetadata;

/** Catalog order and membership are authoritative, including image samples. */
export const getCatalogCardIdsWithMetadata = createSelector(
  getMetricsCatalogCards,
  getNonEmptyCardIdsWithMetadata,
  (descriptors, cards): CardIdWithMetadata[] => {
    const key = (card: {
      plugin: string;
      tag: string;
      runId?: string | null;
      sample?: number;
      numSample?: number;
    }) =>
      JSON.stringify([
        card.plugin,
        card.tag,
        card.runId ?? null,
        card.sample ?? null,
        card.numSample ?? null,
      ]);
    const byDescriptor = new Map(cards.map((card) => [key(card), card]));
    return descriptors.flatMap((descriptor) => {
      const card = byDescriptor.get(key(descriptor));
      return card ? [card] : [];
    });
  }
);

/** Only navigation and filters reset scroll; run selection updates in place. */
export const getCatalogViewScope = createSelector(
  getExperimentIdsFromRoute,
  getMetricsTagFilter,
  getMetricsFilteredPluginTypes,
  (experiments, query, plugins) =>
    JSON.stringify([experiments, query, [...plugins].sort()])
);

export const utils = {
  filterRunItemsByRegex(
    runItems: RunTableItem[],
    regexString: string,
    shouldIncludeExperimentName: boolean
  ): RunTableItem[] {
    if (!regexString) {
      return runItems;
    }

    return runItems.filter((item) => {
      return matchRunToRegex(
        {
          runName: item.run.name,
          experimentAlias: item.experimentAlias,
        },
        regexString,
        shouldIncludeExperimentName
      );
    });
  },

  matchFilter(
    filter: DiscreteFilter | IntervalFilter,
    value: number | DiscreteHparamValue | undefined
  ): boolean {
    if (value === undefined) {
      return filter.includeUndefined;
    }
    if (filter.type === DomainType.DISCRETE) {
      // (upcast to work around bad TypeScript libdefs)
      const values: Readonly<Array<(typeof filter.filterValues)[number]>> =
        filter.filterValues;
      return values.includes(value);
    } else if (filter.type === DomainType.INTERVAL) {
      return (
        typeof value === 'number' &&
        filter.filterLowerValue <= value &&
        value <= filter.filterUpperValue
      );
    }
    return false;
  },

  filterRunItemsByHparamAndMetricFilter(
    runItems: RunTableItem[],
    hparamFilters: Map<string, IntervalFilter | DiscreteFilter>,
    metricFilters: Map<string, IntervalFilter>
  ) {
    return runItems.filter(({hparams, metrics}) => {
      const hparamMatches = [...hparamFilters.entries()].every(
        ([hparamName, filter]) => {
          const value = hparams.get(hparamName);
          return utils.matchFilter(filter, value);
        }
      );

      const metricMatches = [...metricFilters.entries()].every(
        ([metricTag, filter]) => {
          const value = metrics.get(metricTag);
          return utils.matchFilter(filter, value);
        }
      );

      return hparamMatches && metricMatches;
    });
  },
};

export const getCurrentColumnFilters = createSelector(
  getDashboardDefaultHparamFilters,
  getDashboardHparamFilterMap,
  getDashboardMetricsFilterMap,
  (defaultHparamsFilters, hparamFilters, metricFilters) => {
    return new Map([
      ...defaultHparamsFilters,
      ...hparamFilters,
      ...metricFilters,
    ]);
  }
);

export const getRenderableRuns = createSelector(
  getDashboardRuns,
  getDashboardExperimentNames,
  getCurrentRouteRunSelection,
  getRunColorMap,
  getExperimentIdToExperimentAliasMap,
  (
    runs,
    experimentNames,
    selectionMap,
    colorMap,
    experimentIdToAlias
  ): Array<RunTableExperimentItem> => {
    return runs.map((run) => {
      const hparamMap: RunTableItem['hparams'] = new Map();
      (run.hparams || []).forEach((hparam) => {
        hparamMap.set(hparam.name, hparam.value);
      });
      const metricMap: RunTableItem['metrics'] = new Map();
      (run.metrics || []).forEach((metric) => {
        metricMap.set(metric.tag, metric.value);
      });
      return {
        run,
        experimentName: experimentNames[run.experimentId] || '',
        experimentAlias: experimentIdToAlias[run.experimentId],
        selected: Boolean(selectionMap && selectionMap.get(run.id)),
        runColor: colorMap[run.id],
        hparams: hparamMap,
        metrics: metricMap,
      };
    });
  }
);

export const getFilteredRenderableRuns = createSelector(
  getRunSelectorRegexFilter,
  getRenderableRuns,
  getDashboardHparamFilterMap,
  getDashboardMetricsFilterMap,
  getRouteKind,
  (
    regexFilter,
    runItems,
    hparamFilters,
    metricFilters,
    routeKind
  ): RunTableItem[] => {
    const regexFilteredItems = utils.filterRunItemsByRegex(
      runItems,
      regexFilter,
      routeKind === RouteKind.COMPARE_EXPERIMENT
    );

    return utils.filterRunItemsByHparamAndMetricFilter(
      regexFilteredItems,
      hparamFilters,
      metricFilters
    );
  }
);

export const getFilteredRenderableRunsIds = createSelector(
  getFilteredRenderableRuns,
  (filteredRenderableRuns) => {
    return new Set(filteredRenderableRuns.map(({run: {id}}) => id));
  }
);

export const getPotentialHparamColumns = createSelector(
  getDashboardHparamSpecs,
  getExperimentIdsFromRoute,
  (hparamSpecs, experimentIds): ColumnHeader[] => {
    if (!experimentIds) {
      return [];
    }

    return hparamSpecs.map((hparamSpec) => ({
      type: ColumnHeaderType.HPARAM,
      name: hparamSpec.name,
      // According to the api spec when the displayName is empty, the name should
      // be displayed tensorboard/plugins/hparams/api.proto
      displayName: hparamSpec.displayName || hparamSpec.name,
      enabled: false,
      tags: hparamSpec.differs ? ['differs'] : [],
      removable: true,
      sortable: true,
      movable: true,
      filterable: true,
    }));
  }
);

export const getSelectableColumns = createSelector(
  getPotentialHparamColumns,
  getDashboardDisplayedHparamColumns,
  (potentialColumns, currentColumns) => {
    const currentColumnNames = new Set(currentColumns.map(({name}) => name));
    return potentialColumns.filter((columnHeader) => {
      return !currentColumnNames.has(columnHeader.name);
    });
  }
);

export const getAllPotentialColumnsForCard = memoize((cardId: string) => {
  return createSelector(
    getColumnHeadersForCard(cardId),
    getPotentialHparamColumns,
    (staticColumnHeaders, potentialHparamColumns) => {
      return [...staticColumnHeaders, ...potentialHparamColumns];
    }
  );
});

export const TEST_ONLY = {
  getRenderableRuns,
  getRenderableCardIdsWithMetadata,
  getScalarTagsForRunSelection,
  getCurrentColumnFilters,
  utils,
};
