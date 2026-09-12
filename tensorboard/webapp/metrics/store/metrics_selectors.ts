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
import {createFeatureSelector, createSelector} from '@ngrx/store';
import {DataLoadState, LoadState} from '../../types/data';
import {ElementId} from '../../util/dom';
import {DeepReadonly} from '../../util/types';
import {MetricsCatalogCard, MetricsCatalogGroup} from '../data_source';
import {
  CardId,
  CardIdWithMetadata,
  CardMetadata,
  CardUniqueInfo,
  HistogramMode,
  NonPinnedCardId,
  PinnedCardId,
  PluginType,
  TimeSelection,
  TooltipSort,
  XAxisType,
} from '../types';
import {MinMaxStep} from '../views/card_renderer/scalar_card_types';
import {formatTimeSelection} from '../views/card_renderer/utils';
import * as storeUtils from './metrics_store_internal_utils';
import {
  cardRangeSelectionEnabled,
  getCardSelectionStateToBoolean,
  getMinMaxStepFromCardState,
} from './metrics_store_internal_utils';
import {
  CardMetadataMap,
  CardState,
  CardStateMap,
  CardStepIndexMetaData,
  MetricsSettings,
  DEFAULT_METRICS_CATALOG_VIEWPORT,
  MetricsState,
  METRICS_FEATURE_KEY,
  RunToSeries,
  TagMetadata,
  TimeSeriesData,
} from './metrics_types';
import {ColumnHeader, DataTableMode} from '../../widgets/data_table/types';
import {Extent} from '../../widgets/line_chart_v2/lib/public_types';
import {hasOwn} from '../../util/lang';
import {getDashboardDisplayedHparamColumns} from '../../hparams/_redux/hparams_selectors';
import {dataTableUtils} from '../../widgets/data_table/utils';

const selectMetricsState =
  createFeatureSelector<MetricsState>(METRICS_FEATURE_KEY);

export const getMetricsTagMetadataLoadState = createSelector(
  selectMetricsState,
  (state: MetricsState): LoadState => state.tagMetadataLoadState
);

export const getMetricsCatalogViewport = createSelector(
  selectMetricsState,
  (state) => state.catalogViewport ?? DEFAULT_METRICS_CATALOG_VIEWPORT
);

const getMetricsCatalog = createSelector(
  selectMetricsState,
  (state) => state.tagMetadataSource?.catalog
);
const EMPTY_CATALOG_GROUPS: MetricsCatalogGroup[] = [];
const EMPTY_CATALOG_CARDS: MetricsCatalogCard[] = [];

export const getMetricsCatalogEnabled = createSelector(
  getMetricsCatalog,
  (catalog) => !!catalog
);
export const getMetricsCatalogGroups = createSelector(
  getMetricsCatalog,
  (catalog) => catalog?.groups ?? EMPTY_CATALOG_GROUPS
);
export const getMetricsCatalogGroupOffset = createSelector(
  getMetricsCatalog,
  (catalog) => catalog?.groupOffset ?? 0
);
export const getMetricsCatalogFilteredOffset = createSelector(
  getMetricsCatalog,
  (catalog) => catalog?.filteredOffset ?? 0
);
export const getMetricsCatalogTotalGroups = createSelector(
  getMetricsCatalog,
  (catalog) => catalog?.totalGroups ?? 0
);
export const getMetricsCatalogCards = createSelector(
  getMetricsCatalog,
  (catalog) => catalog?.cards ?? EMPTY_CATALOG_CARDS
);
export const getMetricsCatalogTotalCards = createSelector(
  getMetricsCatalog,
  (catalog) => catalog?.totalCards ?? 0
);

export const getMetricsTagMetadata = createSelector(
  selectMetricsState,
  (state: MetricsState): DeepReadonly<TagMetadata> => {
    return state.tagMetadata;
  }
);

/**
 * Cards
 */
const getCardIds = createSelector(selectMetricsState, (state): CardId[] => {
  return state.cardList;
});

export const getCardLoadState = createSelector(
  selectMetricsState,
  (state: MetricsState, cardId: CardId): DataLoadState => {
    if (!hasOwn(state.cardMetadataMap, cardId)) {
      return DataLoadState.NOT_LOADED;
    }
    const {plugin, tag, runId, sample} = state.cardMetadataMap[cardId];
    const loadable = storeUtils.getTimeSeriesLoadable(
      state.timeSeriesData,
      plugin,
      tag,
      sample
    );
    if (!loadable) {
      return DataLoadState.NOT_LOADED;
    }
    const runToLoadState = loadable.runToLoadState;
    if (runId) {
      return hasOwn(runToLoadState, runId)
        ? runToLoadState[runId]
        : DataLoadState.NOT_LOADED;
    }

    const runIds = storeUtils.getRunIds(state.tagMetadata, plugin, tag, sample);
    // Only the runs that have actually been requested are tracked. The
    // dashboard requests the selected subset of a tag's runs, so requiring
    // every run of the tag to be loaded would keep cards loading forever.
    const trackedRunIds = runIds.filter((id) => hasOwn(runToLoadState, id));
    if (!trackedRunIds.length) {
      return DataLoadState.NOT_LOADED;
    }
    if (
      trackedRunIds.every((id) => runToLoadState[id] === DataLoadState.LOADED)
    ) {
      return DataLoadState.LOADED;
    }
    return trackedRunIds.some(
      (id) => runToLoadState[id] === DataLoadState.LOADING
    )
      ? DataLoadState.LOADING
      : DataLoadState.NOT_LOADED;
  }
);

const getCardMetadataMap = createSelector(
  selectMetricsState,
  (state: MetricsState): CardMetadataMap => {
    return state.cardMetadataMap;
  }
);

const selectTagMetadata = createSelector(
  selectMetricsState,
  (state: MetricsState): TagMetadata => state.tagMetadata
);

const selectTimeSeriesData = createSelector(
  selectMetricsState,
  (state: MetricsState): TimeSeriesData => state.timeSeriesData
);

/**
 * Per-run fetch bookkeeping for a card: the runs the card's tag is known to
 * have, and the load state of every run requested for the card so far.
 */
export interface CardRunLoadStates {
  tagRunIds: string[];
  runToLoadState: {[runId: string]: DataLoadState};
}

const EMPTY_TAG_RUN_IDS: string[] = [];
const EMPTY_RUN_TO_LOAD_STATE: {[runId: string]: DataLoadState} = {};

// Factories are intentionally not cached globally: the subscribing component
// or request owns the selector and its last input state.
const getCardTagRunIds = (cardId: CardId) =>
  createSelector(
    getCardMetadataMap,
    selectTagMetadata,
    (cardMetadataMap, tagMetadata): string[] => {
      if (!hasOwn(cardMetadataMap, cardId)) {
        return EMPTY_TAG_RUN_IDS;
      }
      const {plugin, tag, sample} = cardMetadataMap[cardId];
      return storeUtils.getRunIds(tagMetadata, plugin, tag, sample);
    }
  );

const getCardRunToLoadState = (cardId: CardId) =>
  createSelector(
    getCardMetadataMap,
    selectTimeSeriesData,
    (cardMetadataMap, timeSeriesData): {[runId: string]: DataLoadState} => {
      if (!hasOwn(cardMetadataMap, cardId)) {
        return EMPTY_RUN_TO_LOAD_STATE;
      }
      const {plugin, tag, sample} = cardMetadataMap[cardId];
      const loadable = storeUtils.getTimeSeriesLoadable(
        timeSeriesData,
        plugin,
        tag,
        sample
      );
      return loadable ? loadable.runToLoadState : EMPTY_RUN_TO_LOAD_STATE;
    }
  );

/**
 * Per-card selector factory: each card owns its memo cell, so subscribing N
 * cards no longer makes every store emission recompute all N. Both fields are
 * references the store already holds, so the result only changes when the
 * card's tag runs or its own runs' load states change.
 */
export const getCardRunLoadStates = (cardId: CardId) =>
  createSelector(
    getCardTagRunIds(cardId),
    getCardRunToLoadState(cardId),
    (tagRunIds, runToLoadState): CardRunLoadStates => ({
      tagRunIds,
      runToLoadState,
    })
  );

export const getCardTimeSeries = createSelector(
  selectMetricsState,
  (state: MetricsState, cardId: CardId): DeepReadonly<RunToSeries> | null => {
    if (!hasOwn(state.cardMetadataMap, cardId)) {
      return null;
    }

    const {plugin, tag, sample} = state.cardMetadataMap[cardId];
    return (
      storeUtils.getTimeSeriesLoadable(
        state.timeSeriesData,
        plugin,
        tag,
        sample
      )?.runToSeries ?? null
    );
  }
);

export const getCardMetadata = createSelector(
  getCardMetadataMap,
  (
    metadataMap: CardMetadataMap,
    cardId: CardId
  ): DeepReadonly<CardMetadata> | null => {
    if (!hasOwn(metadataMap, cardId)) {
      return null;
    }
    return metadataMap[cardId];
  }
);

export const getCardStateMap = createSelector(
  selectMetricsState,
  (state: MetricsState): CardStateMap => {
    return state.cardStateMap;
  }
);

// A cheap identity selector to skip recomputing selectors when `state` changes.
const selectVisibleCardMap = createSelector(
  selectMetricsState,
  (state): Map<ElementId, CardId> => {
    return state.visibleCardMap;
  }
);

export const getVisibleCardIdSet = createSelector(
  selectVisibleCardMap,
  (visibleCardMap): Set<CardId> => {
    return new Set<CardId>(visibleCardMap.values());
  }
);

/**
 * Returns current list of card data whose metadata is loaded.
 */
export const getNonEmptyCardIdsWithMetadata = createSelector(
  getCardIds,
  getCardMetadataMap,
  (
    cardIds: CardId[],
    metadataMap: CardMetadataMap
  ): DeepReadonly<CardIdWithMetadata[]> => {
    return cardIds
      .filter((cardId) => {
        return hasOwn(metadataMap, cardId);
      })
      .map((cardId) => {
        return {cardId, ...metadataMap[cardId]};
      });
  }
);

/**
 * The index metadata into the step values array for a card's UI. This may be greater
 * than the number of step values available, if time series data is not loaded.
 */
export const getCardStepIndexMetaData = createSelector(
  selectMetricsState,
  (state: MetricsState, cardId: CardId): CardStepIndexMetaData | null => {
    if (!hasOwn(state.cardStepIndex, cardId)) {
      return null;
    }
    return state.cardStepIndex[cardId];
  }
);

/**
 * Returns step values of an image card.
 */
export const getMetricsImageCardSteps = createSelector(
  selectMetricsState,
  (state: MetricsState, cardId: CardId): number[] => {
    return storeUtils.getImageCardSteps(
      cardId,
      state.cardMetadataMap,
      state.timeSeriesData
    );
  }
);

const getCardToPinnedCopy = createSelector(
  selectMetricsState,
  (state): Map<NonPinnedCardId, PinnedCardId> => {
    return state.cardToPinnedCopy;
  }
);

const getPinnedCardToOriginal = createSelector(
  selectMetricsState,
  (state): Map<PinnedCardId, NonPinnedCardId> => {
    return state.pinnedCardToOriginal;
  }
);

/**
 * Returns an ordered list of the cards in the pinned location.
 */
export const getPinnedCardsWithMetadata = createSelector(
  getCardToPinnedCopy,
  getCardMetadataMap,
  (
    cardToPinnedCopy: Map<NonPinnedCardId, PinnedCardId>,
    metadataMap: CardMetadataMap
  ): DeepReadonly<CardIdWithMetadata[]> => {
    return [...cardToPinnedCopy.values()]
      .filter((cardId) => {
        return hasOwn(metadataMap, cardId);
      })
      .map((cardId) => {
        return {cardId, ...metadataMap[cardId]};
      });
  }
);

/**
 * Returns true if a card is pinned or a separate card exists that is a pinned
 * copy of this card. Defaults to false if the card is unknown.
 */
export const getCardPinnedState = createSelector(
  getCardToPinnedCopy,
  getPinnedCardToOriginal,
  (
    cardToPinnedCopy: Map<NonPinnedCardId, PinnedCardId>,
    pinnedCardToOriginal: Map<PinnedCardId, NonPinnedCardId>,
    cardId: NonPinnedCardId | PinnedCardId
  ): boolean => {
    return cardToPinnedCopy.has(cardId) || pinnedCardToOriginal.has(cardId);
  }
);

export const getUnresolvedImportedPinnedCards = createSelector(
  selectMetricsState,
  (state: MetricsState): CardUniqueInfo[] => {
    return state.unresolvedImportedPinnedCards;
  }
);

/**
 * Whether the UI is allowed to pin more cards. This may be limited if the URL
 * contains too many pins already.
 */
export const getCanCreateNewPins = createSelector(
  selectMetricsState,
  (state: MetricsState): boolean => {
    return storeUtils.canCreateNewPins(state);
  }
);

export const getLastPinnedCardTime = createSelector(
  selectMetricsState,
  (state: MetricsState): number => {
    return state.lastPinnedCardTime;
  }
);

const selectSettings = createSelector(
  selectMetricsState,
  (state): MetricsSettings => {
    return {
      ...state.settings,
      ...state.settingOverrides,
    };
  }
);

/**
 * Settings.
 */
export const getMetricsSettingOverrides = createSelector(
  selectMetricsState,
  (state): Partial<MetricsSettings> => {
    return state.settingOverrides;
  }
);

export const getMetricsCardMinWidth = createSelector(
  selectSettings,
  (settings): number | null => settings.cardMinWidth
);

export const getMetricsTooltipSort = createSelector(
  selectSettings,
  (settings): TooltipSort => settings.tooltipSort
);

export const getMetricsIgnoreOutliers = createSelector(
  selectSettings,
  (settings): boolean => settings.ignoreOutliers
);

export const getMetricsXAxisType = createSelector(
  selectSettings,
  (settings): XAxisType => settings.xAxisType
);

export const getMetricsHistogramMode = createSelector(
  selectSettings,
  (settings): HistogramMode => settings.histogramMode
);

export const getMetricsHideEmptyCards = createSelector(
  selectSettings,
  (settings): boolean => settings.hideEmptyCards
);

export const getMetricsScalarSmoothing = createSelector(
  selectSettings,
  (settings): number => settings.scalarSmoothing
);

export const getMetricsScalarPartitionNonMonotonicX = createSelector(
  selectSettings,
  (settings): boolean => settings.scalarPartitionNonMonotonicX
);

export const getMetricsIsTooltipRowsLimitEnabled = createSelector(
  selectSettings,
  (settings): boolean => settings.isTooltipRowsLimitEnabled
);

export const getMetricsTooltipRowsLimit = createSelector(
  selectSettings,
  (settings): number => settings.tooltipRowsLimit
);

export const getMetricsImageBrightnessInMilli = createSelector(
  selectSettings,
  (settings): number => settings.imageBrightnessInMilli
);

export const getMetricsImageContrastInMilli = createSelector(
  selectSettings,
  (settings): number => settings.imageContrastInMilli
);

export const getMetricsImageShowActualSize = createSelector(
  selectSettings,
  (settings): boolean => settings.imageShowActualSize
);

export const getMetricsSavingPinsEnabled = createSelector(
  selectSettings,
  (settings): boolean => settings.savingPinsEnabled
);

export const getMetricsTagFilter = createSelector(
  selectMetricsState,
  (state): string => state.tagFilter
);

export const getMetricsTagGroupExpansionState = createSelector(
  selectMetricsState,
  (state: MetricsState, tagGroup: string): boolean => {
    return Boolean(state.tagGroupExpanded.get(tagGroup));
  }
);

export const getMetricsTagGroupExpandedMap = createSelector(
  selectMetricsState,
  (state: MetricsState): Map<string, boolean> => {
    return state.tagGroupExpanded;
  }
);

export const getMetricsTagGroupPageIndexMap = createSelector(
  selectMetricsState,
  (state: MetricsState): Map<string, number> => {
    return state.tagGroupPageIndex;
  }
);

export const getMetricsTagGroupPageIndex = createSelector(
  selectMetricsState,
  (state: MetricsState, tagGroup: string): number => {
    return state.tagGroupPageIndex.get(tagGroup) ?? 0;
  }
);

export const getMetricsLinkedTimeEnabled = createSelector(
  selectMetricsState,
  (state: MetricsState): boolean => {
    return state.linkedTimeEnabled;
  }
);

export const getMetricsStepSelectorEnabled = createSelector(
  selectMetricsState,
  (state: MetricsState): boolean => {
    return state.stepSelectorEnabled;
  }
);

export const getMetricsRangeSelectionEnabled = createSelector(
  selectMetricsState,
  (state: MetricsState): boolean => {
    return state.rangeSelectionEnabled;
  }
);

export const getMetricsStepMinMax = createSelector(
  selectMetricsState,
  (state: MetricsState): {min: number; max: number} => {
    const {min, max} = state.stepMinMax;
    return {
      min: min === Infinity ? 0 : min,
      max: max === -Infinity ? 1000 : max,
    };
  }
);

/**
 * Returns value of the linked time set by user. When linked time selection is never
 * set, it returns the default value which is derived from the timeseries data
 * loaded thus far.
 *
 * This selector is intended to used by settings panel only. Other views should
 * use `getMetricsLinkedTimeSelection` that returns `TimeSelection` value according to
 * the setting.
 *
 * @see getMetricsLinkedTimeSelection For most views.
 */
export const getMetricsLinkedTimeSelectionSetting = createSelector(
  selectMetricsState,
  getMetricsStepMinMax,
  (state, stepMinMax): TimeSelection => {
    if (!state.linkedTimeSelection) {
      return {
        start: {
          step: stepMinMax.max,
        },
        end: null,
      };
    }

    return state.linkedTimeSelection;
  }
);

/**
 * Returns linked time selection set by user. If linkedTime is disabled, it returns
 * `null`. Also, when range selection mode is disabled, it returns `end=null`
 * even if it has value set.
 *
 * Virtually all views should use this selector.
 */
export const getMetricsLinkedTimeSelection = createSelector(
  selectMetricsState,
  getMetricsLinkedTimeSelectionSetting,
  (
    state: MetricsState,
    linkedTimeSelection: TimeSelection
  ): TimeSelection | null => {
    if (!state.linkedTimeEnabled) return null;
    return linkedTimeSelection;
  }
);

export const getMetricsFilteredPluginTypes = createSelector(
  selectMetricsState,
  (state: MetricsState): Set<PluginType> => {
    return state.filteredPluginTypes;
  }
);

export const isMetricsSettingsPaneOpen = createSelector(
  selectMetricsState,
  (state): boolean => state.isSettingsPaneOpen
);

export const isMetricsSlideoutMenuOpen = createSelector(
  selectMetricsState,
  (state): boolean => state.isSlideoutMenuOpen
);

export const getTableEditorSelectedTab = createSelector(
  selectMetricsState,
  (state): DataTableMode => state.tableEditorSelectedTab
);

export const getMetricsCardRangeSelectionEnabled = (cardId: CardId) =>
  createSelector(
    getCardStateMap,
    getMetricsRangeSelectionEnabled,
    getMetricsLinkedTimeEnabled,
    (
      cardStateMap: CardStateMap,
      globalRangeSelectionEnabled: boolean,
      linkedTimeEnabled: boolean
    ) =>
      cardRangeSelectionEnabled(
        cardStateMap,
        globalRangeSelectionEnabled,
        linkedTimeEnabled,
        cardId
      )
  );

/**
 * A single card's slice of the card state map. Card states are replaced
 * individually by the reducers, so this lets a card's derived selectors ignore
 * every action that only touches other cards.
 */
const getCardState = (cardId: CardId) =>
  createSelector(
    getCardStateMap,
    (cardStateMap: CardStateMap): Partial<CardState> | undefined =>
      cardStateMap[cardId]
  );

/**
 * Gets the min and max step visible in a metrics card.
 * This value can either be the data min max or be overridden
 * by min max within userViewBox.
 *
 * Note: min max within userViewBox is not necessarily a subset of dataMinMax.
 */
export const getMetricsCardMinMax = (cardId: CardId) =>
  createSelector(getCardState(cardId), (cardState): MinMaxStep | undefined => {
    if (!cardState) return;
    return getMinMaxStepFromCardState(cardState);
  });

/**
 * Returns the min and max step found in the cards data.
 */
export const getMetricsCardDataMinMax = createSelector(
  getCardStateMap,
  (cardStateMap: CardStateMap, cardId: CardId): MinMaxStep | undefined => {
    return cardStateMap[cardId]?.dataMinMax;
  }
);

/**
 * Returns user defined view extent. Null means no zoom in, user box is the same as data extent.
 */
export const getMetricsCardUserViewBox = createSelector(
  getCardStateMap,
  (cardStateMap: CardStateMap, cardId: CardId): Extent | null => {
    return cardStateMap[cardId]?.userViewBox ?? null;
  }
);

/**
 * Gets the time selection of a metrics card.
 */
export const getMetricsCardTimeSelection = createSelector(
  getCardStateMap,
  getMetricsStepSelectorEnabled,
  getMetricsRangeSelectionEnabled,
  getMetricsLinkedTimeEnabled,
  getMetricsLinkedTimeSelection,
  (
    cardStateMap: CardStateMap,
    globalStepSelectionEnabled: boolean,
    globalRangeSelectionEnabled: boolean,
    linkedTimeEnabled: boolean,
    linkedTimeSelection: TimeSelection | null,
    cardId: CardId
  ): TimeSelection | undefined => {
    const cardState = cardStateMap[cardId];
    if (!cardState) {
      return;
    }
    const minMaxStep = getMinMaxStepFromCardState(cardState);
    if (!minMaxStep) {
      return;
    }

    // Handling Linked Time
    if (linkedTimeEnabled && linkedTimeSelection) {
      return formatTimeSelection(
        linkedTimeSelection,
        minMaxStep,
        // Note that globalRangeSelection should always be used with linked time.
        globalRangeSelectionEnabled
      );
    }

    // If the user has disabled step selection, nothing should be returned.
    if (
      !getCardSelectionStateToBoolean(
        cardState.stepSelectionOverride,
        globalStepSelectionEnabled
      )
    ) {
      return;
    }

    const rangeSelectionEnabled = getCardSelectionStateToBoolean(
      cardState.rangeSelectionOverride,
      globalRangeSelectionEnabled
    );

    let timeSelection = cardState.timeSelection;
    if (!timeSelection) {
      timeSelection = {
        start: {step: minMaxStep.minStep},
        end: {step: minMaxStep.maxStep},
      };
    }
    if (rangeSelectionEnabled) {
      if (!timeSelection.end) {
        // Enabling range selection from single selection selects the first
        // step as the start of the range. The previous start step from single
        // selection is now the end step.
        timeSelection = {
          start: {step: minMaxStep.minStep},
          end: timeSelection.start,
        };
      }
    } else {
      // Disabling range selection keeps the largest step from the range.
      timeSelection = {
        start: timeSelection.end ?? timeSelection.start,
        end: null,
      };
    }

    return formatTimeSelection(
      timeSelection,
      minMaxStep,
      rangeSelectionEnabled
    );
  }
);

export const getSingleSelectionHeaders = createSelector(
  selectMetricsState,
  (state: MetricsState): ColumnHeader[] => {
    return state.singleSelectionHeaders;
  }
);

export const getRangeSelectionHeaders = createSelector(
  selectMetricsState,
  (state: MetricsState): ColumnHeader[] => {
    return state.rangeSelectionHeaders;
  }
);

export const getColumnHeadersForCard = (cardId: string) => {
  return createSelector(
    getMetricsCardRangeSelectionEnabled(cardId),
    getSingleSelectionHeaders,
    getRangeSelectionHeaders,
    (
      cardRangeSelectionEnabled,
      singleSelectionHeaders,
      rangeSelectionHeaders
    ) => {
      return cardRangeSelectionEnabled
        ? rangeSelectionHeaders
        : singleSelectionHeaders;
    }
  );
};

export const getGroupedHeadersForCard = (cardId: string) =>
  createSelector(
    getColumnHeadersForCard(cardId),
    getDashboardDisplayedHparamColumns,
    (standardColumns, hparamColumns) =>
      dataTableUtils.groupColumns([...standardColumns, ...hparamColumns])
  );
