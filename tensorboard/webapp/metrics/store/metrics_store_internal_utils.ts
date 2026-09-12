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
 * @fileoverview Utilities used internally by the Metrics feature's NgRx store.
 */

import {DataLoadState} from '../../types/data';
import {hasOwn} from '../../util/lang';
import {isSampledPlugin, PluginType, SampledPluginType} from '../data_source';
import {
  CardId,
  CardMetadata,
  CardUniqueInfo,
  MinMaxStep,
  NonPinnedCardId,
  TimeSelection,
} from '../types';
import {
  CardFeatureOverride,
  CardMetadataMap,
  CardState,
  CardStateMap,
  CardStepIndexMap,
  CardStepIndexMetaData,
  CardToPinnedCard,
  MetricsState,
  PinnedCardToCard,
  RunToLoadState,
  RunToSeries,
  TagMetadata,
  TimeSeriesData,
  TimeSeriesLoadables,
} from './metrics_types';

type ResolvedPinPartialState = Pick<
  MetricsState,
  | 'cardMetadataMap'
  | 'cardToPinnedCopy'
  | 'cardToPinnedCopyCache'
  | 'pinnedCardToOriginal'
  | 'cardStepIndex'
  | 'cardStateMap'
>;

const DISTANCE_RATIO = 0.1;

/**
 * Returns the loadable information for a specific tag, containing its series
 * data and load state. Returns `null` when the requested tag has no initial
 * loadable in `timeSeriesData`.
 */
export function getTimeSeriesLoadable(
  timeSeriesData: TimeSeriesData,
  plugin: PluginType,
  tag: string,
  sample?: number
): TimeSeriesLoadables[typeof plugin] | null {
  const pluginData = timeSeriesData[plugin];
  if (!hasOwn(pluginData, tag)) {
    return null;
  }
  if (isSampledPlugin(plugin)) {
    if (!hasOwn(timeSeriesData[plugin][tag], sample!)) {
      return null;
    }
    return timeSeriesData[plugin][tag][sample!];
  }
  return timeSeriesData[plugin][tag];
}

/**
 * Create a new plugin data with new references to a new time series loadable.
 * The return object is a shallow clone, so consumers must clone fields as
 * needed.
 */
export function createPluginDataWithLoadable(
  timeSeriesData: TimeSeriesData,
  plugin: PluginType,
  tag: string,
  sample?: number
): TimeSeriesData[typeof plugin] {
  if (isSampledPlugin(plugin)) {
    const pluginData = {...timeSeriesData[plugin]};
    const tagData = createSampledTagDataWithLoadable<typeof plugin>(
      pluginData,
      tag,
      sample!
    );
    pluginData[tag] = tagData;
    return pluginData;
  }

  const pluginData = {...timeSeriesData[plugin]};
  const hasTag = hasOwn(pluginData, tag);
  pluginData[tag] = hasTag
    ? {...pluginData[tag]}
    : buildTimeSeriesLoadable<typeof plugin>();
  return pluginData;
}

function createSampledTagDataWithLoadable<P extends SampledPluginType>(
  pluginData: TimeSeriesData[SampledPluginType],
  tag: string,
  sample: number
) {
  const hasTag = hasOwn(pluginData, tag);
  const tagData = hasTag ? {...pluginData[tag]} : {};

  const hasSample = hasOwn(tagData, sample);
  tagData[sample] = hasSample
    ? {...tagData[sample]}
    : buildTimeSeriesLoadable<P>();
  return tagData;
}

function buildTimeSeriesLoadable<
  P extends PluginType
>(): TimeSeriesLoadables[P] {
  return {
    runToSeries: {},
    runToLoadState: {},
  };
}

/**
 * Note: do not rely on the implementation details of these ID generators below.
 * Clients should operate on `CardId`s, whose type may be open to change.
 */

export function getCardId(cardMetadata: CardMetadata) {
  return JSON.stringify(cardMetadata);
}

export function getPinnedCardId(baseCardId: CardId) {
  return JSON.stringify({baseCardId});
}

/**
 * Identifies the loadable that holds a (plugin, tag, sample)'s series. Unlike
 * a `CardId`, several cards can share one key: every run of a single-run
 * plugin's tag reads the same loadable.
 *
 * The tag goes last because it is arbitrary user text, so any separator can
 * appear inside it and only a trailing field keeps the key unambiguous.
 */
export function getLoadableKey(
  plugin: PluginType,
  tag: string,
  sample?: number
): string {
  return `${plugin}\u0000${sample ?? ''}\u0000${tag}`;
}

/**
 * Creates a RunToLoadState with a specific load state for all specified runs.
 */
export function createRunToLoadState(
  loadState: DataLoadState,
  runs: string[],
  prevRunToLoadState?: RunToLoadState
): RunToLoadState {
  const runToLoadState = {...prevRunToLoadState} as RunToLoadState;
  for (const run of runs) {
    runToLoadState[run] = loadState;
  }
  return runToLoadState;
}

export function getRunIds(
  tagMetadata: TagMetadata,
  plugin: PluginType,
  tag: string,
  sample?: number
) {
  if (isSampledPlugin(plugin)) {
    const tagRunSampledInfo = tagMetadata[plugin].tagRunSampledInfo;
    if (!hasOwn(tagRunSampledInfo, tag)) {
      return [];
    }
    const runIds = Object.keys(tagRunSampledInfo[tag]);
    return runIds.filter((runId) => {
      return sample! < tagRunSampledInfo[tag][runId].maxSamplesPerStep;
    });
  }
  const tagToRunIds = tagMetadata[plugin].tagToRuns;
  return hasOwn(tagToRunIds, tag) ? tagToRunIds[tag] : [];
}

export interface RetainedTimeSeries {
  timeSeriesData: TimeSeriesData;
  inactiveTimeSeries: Map<string, number>;
  /**
   * `getLoadableKey` keys of the loadables that dropped runs. Empty whenever
   * `timeSeriesData` is the unchanged input.
   */
  changedLoadableKeys: Set<string>;
}

export const INACTIVE_TIME_SERIES_BYTES = 64 * 1024 * 1024;

// Series arrays are immutable. Weak keys avoid retaining evicted payloads and
// avoid walking every histogram bin / image identifier on each scroll.
const seriesBytes = new WeakMap<object, number>();

function estimatePayloadBytes(value: unknown): number {
  if (typeof value === 'string') return 32 + 2 * value.length;
  if (typeof value !== 'object' || value === null) return 8;
  let bytes = 64;
  if (Array.isArray(value)) {
    bytes += 8 * value.length;
    for (const item of value) bytes += estimatePayloadBytes(item);
  } else {
    for (const [key, item] of Object.entries(value)) {
      bytes += 16 + 2 * key.length + estimatePayloadBytes(item);
    }
  }
  return bytes;
}

function estimateSeriesBytes(series: RunToSeries[string] | undefined): number {
  if (!series) return 0;
  const cached = seriesBytes.get(series);
  if (cached !== undefined) return cached;
  const bytes = estimatePayloadBytes(series);
  seriesBytes.set(series, bytes);
  return bytes;
}

/**
 * Protects render-buffer owners and keeps selected, completed inactive histories
 * in least-recently-used order. An active history leaves the LRU; when its last
 * owner exits it becomes newest. Deselected and invalidated histories are never
 * cached. Failed states are retained too, so scrolling does not retry errors.
 *
 * Accounting includes nested histogram bins and UTF-16 image identifiers (the
 * store contains no decoded image pixels), plus conservative object/map costs.
 * The budget bounds estimated inactive bytes only, not the active working set.
 */
export function retainTimeSeriesRuns(
  timeSeriesData: TimeSeriesData,
  keepRunIds: ReadonlySet<string>,
  visibleCards: readonly CardMetadata[],
  previousInactive: ReadonlyMap<string, number> = new Map(),
  budget = INACTIVE_TIME_SERIES_BYTES
): RetainedTimeSeries {
  const ownership = new Map<string, ReadonlySet<string>>();
  for (const card of visibleCards) {
    const key = getLoadableKey(card.plugin, card.tag, card.sample);
    if (card.runId) {
      const runs = new Set(ownership.get(key));
      runs.add(card.runId);
      ownership.set(key, runs);
    } else {
      ownership.set(key, keepRunIds);
    }
  }
  const candidates = new Map<string, number>();
  const candidateOwners = new Map<string, {key: string; runId: string}>();
  const retain = (
    loadable: TimeSeriesLoadables[PluginType],
    plugin: PluginType,
    tag: string,
    sample?: number
  ) => {
    const key = getLoadableKey(plugin, tag, sample);
    const active = ownership.get(key) ?? new Set<string>();
    const keep = new Set(active);
    ownership.set(key, keep);
    for (const runId of new Set([
      ...Object.keys(loadable.runToSeries),
      ...Object.keys(loadable.runToLoadState),
    ])) {
      if (active.has(runId) || !keepRunIds.has(runId)) continue;
      const status = loadable.runToLoadState[runId];
      if (status !== DataLoadState.LOADED && status !== DataLoadState.FAILED) {
        continue;
      }
      const cacheKey = JSON.stringify([key, runId]);
      const bytes =
        256 +
        2 * cacheKey.length +
        estimateSeriesBytes(loadable.runToSeries[runId]);
      candidates.set(cacheKey, bytes);
      candidateOwners.set(cacheKey, {key, runId});
    }
  };
  for (const [tag, loadable] of Object.entries(timeSeriesData.scalars)) {
    retain(loadable, PluginType.SCALARS, tag);
  }
  for (const [tag, loadable] of Object.entries(timeSeriesData.histograms)) {
    retain(loadable, PluginType.HISTOGRAMS, tag);
  }
  for (const [tag, samples] of Object.entries(timeSeriesData.images)) {
    for (const [sample, loadable] of Object.entries(samples)) {
      retain(loadable, PluginType.IMAGES, tag, Number(sample));
    }
  }
  const inactiveTimeSeries = new Map<string, number>();
  let inactiveBytes = 0;
  for (const key of previousInactive.keys()) {
    const bytes = candidates.get(key);
    if (bytes === undefined) continue;
    inactiveTimeSeries.set(key, bytes);
    inactiveBytes += bytes;
    candidates.delete(key);
  }
  for (const [key, bytes] of candidates) {
    inactiveTimeSeries.set(key, bytes);
    inactiveBytes += bytes;
  }
  for (const [key, bytes] of inactiveTimeSeries) {
    if (inactiveBytes <= budget) break;
    inactiveTimeSeries.delete(key);
    inactiveBytes -= bytes;
  }
  for (const cacheKey of inactiveTimeSeries.keys()) {
    const {key, runId} = candidateOwners.get(cacheKey)!;
    (ownership.get(key) as Set<string>).add(runId);
  }
  const changedLoadableKeys = new Set<string>();
  const scalars = retainNonSampledPluginData(
    timeSeriesData[PluginType.SCALARS],
    PluginType.SCALARS,
    ownership,
    changedLoadableKeys
  );
  const histograms = retainNonSampledPluginData(
    timeSeriesData[PluginType.HISTOGRAMS],
    PluginType.HISTOGRAMS,
    ownership,
    changedLoadableKeys
  );
  const images = retainSampledPluginData(
    timeSeriesData[PluginType.IMAGES],
    PluginType.IMAGES,
    ownership,
    changedLoadableKeys
  );
  if (
    scalars === timeSeriesData[PluginType.SCALARS] &&
    histograms === timeSeriesData[PluginType.HISTOGRAMS] &&
    images === timeSeriesData[PluginType.IMAGES]
  ) {
    return {timeSeriesData, inactiveTimeSeries, changedLoadableKeys};
  }
  return {
    inactiveTimeSeries,
    timeSeriesData: {
      [PluginType.SCALARS]: scalars,
      [PluginType.HISTOGRAMS]: histograms,
      [PluginType.IMAGES]: images,
    },
    changedLoadableKeys,
  };
}

function retainNonSampledPluginData<
  T extends {runToSeries: {}; runToLoadState: {}}
>(
  pluginData: Record<string, T>,
  plugin: PluginType,
  ownership: ReadonlyMap<string, ReadonlySet<string>>,
  changedLoadableKeys: Set<string>
) {
  let changed = false;
  const nextPluginData: Record<string, T> = {};
  for (const [tag, loadable] of Object.entries(pluginData)) {
    const key = getLoadableKey(plugin, tag);
    const keepRunIds = ownership.get(key);
    if (!keepRunIds?.size) {
      changed = true;
      changedLoadableKeys.add(key);
      continue;
    }
    const nextLoadable = retainLoadableRuns(loadable, keepRunIds);
    if (nextLoadable !== loadable) {
      changed = true;
      changedLoadableKeys.add(getLoadableKey(plugin, tag));
    }
    nextPluginData[tag] = nextLoadable;
  }
  return changed ? nextPluginData : pluginData;
}

function retainSampledPluginData<
  T extends {runToSeries: {}; runToLoadState: {}}
>(
  pluginData: Record<string, Record<number, T>>,
  plugin: PluginType,
  ownership: ReadonlyMap<string, ReadonlySet<string>>,
  changedLoadableKeys: Set<string>
) {
  let changed = false;
  const nextPluginData: Record<string, Record<number, T>> = {};
  for (const [tag, sampleData] of Object.entries(pluginData)) {
    let sampleChanged = false;
    const nextSampleData: Record<number, T> = {};
    for (const [sample, loadable] of Object.entries(sampleData)) {
      const key = getLoadableKey(plugin, tag, Number(sample));
      const keepRunIds = ownership.get(key);
      if (!keepRunIds?.size) {
        sampleChanged = true;
        changedLoadableKeys.add(key);
        continue;
      }
      const nextLoadable = retainLoadableRuns(loadable, keepRunIds);
      if (nextLoadable !== loadable) {
        sampleChanged = true;
        changedLoadableKeys.add(getLoadableKey(plugin, tag, Number(sample)));
      }
      nextSampleData[Number(sample)] = nextLoadable;
    }
    changed = changed || sampleChanged;
    if (Object.keys(nextSampleData).length) {
      nextPluginData[tag] = sampleChanged ? nextSampleData : sampleData;
    }
  }
  return changed ? nextPluginData : pluginData;
}

function retainLoadableRuns<
  T extends {runToSeries: {}; runToLoadState: {[runId: string]: DataLoadState}}
>(loadable: T, keepRunIds: ReadonlySet<string>): T {
  const seriesRunIds = Object.keys(loadable.runToSeries);
  const loadStateRunIds = Object.keys(loadable.runToLoadState);
  if (
    seriesRunIds.every((runId) => keepRunIds.has(runId)) &&
    loadStateRunIds.every((runId) => keepRunIds.has(runId))
  ) {
    return loadable;
  }
  const runToSeries = {} as Record<string, unknown>;
  for (const runId of seriesRunIds) {
    if (keepRunIds.has(runId)) {
      runToSeries[runId] =
        loadable.runToSeries[runId as keyof typeof loadable.runToSeries];
    }
  }
  const runToLoadState = {} as Record<string, DataLoadState>;
  for (const runId of loadStateRunIds) {
    if (keepRunIds.has(runId)) {
      runToLoadState[runId] = loadable.runToLoadState[runId];
    }
  }
  return {
    ...loadable,
    runToSeries: runToSeries as T['runToSeries'],
    runToLoadState: runToLoadState as T['runToLoadState'],
  };
}

/**
 * Returns whether the CardMetadata exactly matches the pinned card from
 * storage.
 */
function cardMatchesCardUniqueInfo(
  cardMetadata: CardMetadata,
  cardUniqueInfo: CardUniqueInfo
): boolean {
  const noRunId = !cardMetadata.runId && !cardUniqueInfo.runId;
  return (
    cardMetadata.plugin === cardUniqueInfo.plugin &&
    cardMetadata.tag === cardUniqueInfo.tag &&
    cardMetadata.sample === cardUniqueInfo.sample &&
    (cardMetadata.runId === cardUniqueInfo.runId || noRunId)
  );
}

/**
 * Attempts to resolve the imported pins against the list of non-pinned cards
 * provided. Returns the resulting state.
 *
 * Note: this assumes input has already been sanitized and validated. Untrusted
 * data from URLs must be cleaned before being passed to the store.
 */
export function buildOrReturnStateWithUnresolvedImportedPins(
  unresolvedImportedPinnedCards: CardUniqueInfo[],
  nonPinnedCards: NonPinnedCardId[],
  cardMetadataMap: CardMetadataMap,
  cardToPinnedCopy: CardToPinnedCard,
  cardToPinnedCopyCache: CardToPinnedCard,
  pinnedCardToOriginal: PinnedCardToCard,
  cardStepIndexMap: CardStepIndexMap,
  cardStateMap: CardStateMap
): ResolvedPinPartialState & {unresolvedImportedPinnedCards: CardUniqueInfo[]} {
  const unresolvedPinSet = new Set(unresolvedImportedPinnedCards);
  const nonPinnedCardsWithMatch: string[] = [];
  for (const unresolvedPin of unresolvedImportedPinnedCards) {
    for (const nonPinnedCardId of nonPinnedCards) {
      const cardMetadata = cardMetadataMap[nonPinnedCardId];
      if (cardMatchesCardUniqueInfo(cardMetadata, unresolvedPin)) {
        nonPinnedCardsWithMatch.push(nonPinnedCardId);
        unresolvedPinSet.delete(unresolvedPin);
        break;
      }
    }
  }

  if (!nonPinnedCardsWithMatch.length) {
    return {
      unresolvedImportedPinnedCards,
      cardMetadataMap,
      cardToPinnedCopy,
      cardToPinnedCopyCache,
      pinnedCardToOriginal,
      cardStepIndex: cardStepIndexMap,
      cardStateMap,
    };
  }

  let stateWithResolvedPins = {
    cardToPinnedCopy,
    cardToPinnedCopyCache,
    pinnedCardToOriginal,
    cardStepIndex: cardStepIndexMap,
    cardMetadataMap,
    cardStateMap,
  };
  for (const cardToPin of nonPinnedCardsWithMatch) {
    stateWithResolvedPins = buildOrReturnStateWithPinnedCopy(
      cardToPin,
      stateWithResolvedPins.cardToPinnedCopy,
      stateWithResolvedPins.cardToPinnedCopyCache,
      stateWithResolvedPins.pinnedCardToOriginal,
      stateWithResolvedPins.cardStepIndex,
      stateWithResolvedPins.cardMetadataMap,
      cardStateMap
    );
  }

  return {
    ...stateWithResolvedPins,
    unresolvedImportedPinnedCards: [...unresolvedPinSet],
  };
}

/**
 * Return the state produced by creating a new pinned copy of the provided card.
 * May throw if the card provided has no metadata.
 */
export function buildOrReturnStateWithPinnedCopy(
  cardId: NonPinnedCardId,
  cardToPinnedCopy: CardToPinnedCard,
  cardToPinnedCopyCache: CardToPinnedCard,
  pinnedCardToOriginal: PinnedCardToCard,
  cardStepIndexMap: CardStepIndexMap,
  cardMetadataMap: CardMetadataMap,
  cardStateMap: CardStateMap
): ResolvedPinPartialState {
  // No-op if the card already has a pinned copy.
  if (cardToPinnedCopy.has(cardId)) {
    return {
      cardToPinnedCopy,
      cardToPinnedCopyCache,
      pinnedCardToOriginal,
      cardStepIndex: cardStepIndexMap,
      cardMetadataMap,
      cardStateMap,
    };
  }

  const nextCardToPinnedCopy = new Map(cardToPinnedCopy);
  const nextCardToPinnedCopyCache = new Map(cardToPinnedCopyCache);
  const nextPinnedCardToOriginal = new Map(pinnedCardToOriginal);
  const nextCardStepIndexMap = {...cardStepIndexMap};
  const nextCardMetadataMap = {...cardMetadataMap};
  const nextCardStateMap = {...cardStateMap};

  const pinnedCardId = getPinnedCardId(cardId);
  nextCardToPinnedCopy.set(cardId, pinnedCardId);
  nextCardToPinnedCopyCache.set(cardId, pinnedCardId);
  nextPinnedCardToOriginal.set(pinnedCardId, cardId);

  if (hasOwn(cardStepIndexMap, cardId)) {
    nextCardStepIndexMap[pinnedCardId] = cardStepIndexMap[cardId];
  }

  const metadata = cardMetadataMap[cardId];
  if (!metadata) {
    throw new Error('Cannot pin a card without metadata');
  }
  nextCardMetadataMap[pinnedCardId] = metadata;

  if (nextCardStateMap[cardId]) {
    // This shared reference is okay because the reducer will force the referenced
    // object to be updated when any changes are made to it.
    // https://github.com/tensorflow/tensorboard/pull/6172#discussion_r1115007044
    nextCardStateMap[pinnedCardId] = nextCardStateMap[cardId];
  }

  return {
    cardToPinnedCopy: nextCardToPinnedCopy,
    cardToPinnedCopyCache: nextCardToPinnedCopyCache,
    pinnedCardToOriginal: nextPinnedCardToOriginal,
    cardStepIndex: nextCardStepIndexMap,
    cardMetadataMap: nextCardMetadataMap,
    cardStateMap: nextCardStateMap,
  };
}

/**
 * Determines which pinned cards should continue to be present in the metrics
 * state after an update to the set of experiments/tags/cards. Pinned cards are
 * removed if their corresponding card is no longer present in the state (perhaps
 * because the corresponding experiment and tag have been removed from the
 * comparison). Pinned cards are added if they are previously pinned by the user.
 * It generates new cardToPinnedCopy, pinnedCardToOriginal maps to reflect the new
 * set of pinned cards. It generates a cardMetadataMap object that is a combination
 * of the input nextCardMetadataMap as well as metadata for the new set of pinned cards.
 * @param previousCardToPinnedCopyCache The set of pinned cards that were previously
 *  pinned by the user. This is a superset of pinned cards in cardToPinnedCopy and the
 *  set remains the same after the update.
 * @param nextCardMetadataMap Metadata for all cards that will be present after
 *  the update (we assume it does not yet include metadata for pinned copies of cards).
 * @param nextCardList The set of base cards that will continue to be present in the
 *  dashboard after the update.
 */
export function generateNextPinnedCardMappings(
  previousCardToPinnedCopyCache: CardToPinnedCard,
  nextCardMetadataMap: CardMetadataMap,
  nextCardList: CardId[]
) {
  const nextCardToPinnedCopy = new Map() as CardToPinnedCard;
  const nextPinnedCardToOriginal = new Map() as PinnedCardToCard;
  const pinnedCardMetadataMap = {} as CardMetadataMap;
  previousCardToPinnedCopyCache.forEach((pinnedCardId, cardId) => {
    if (nextCardList.indexOf(cardId) !== -1) {
      nextCardToPinnedCopy.set(cardId, pinnedCardId);
      nextPinnedCardToOriginal.set(pinnedCardId, cardId);
      pinnedCardMetadataMap[pinnedCardId] = nextCardMetadataMap[cardId];
    }
  });

  return {
    nextCardToPinnedCopy,
    nextPinnedCardToOriginal,
    pinnedCardMetadataMap,
  };
}

/**
 * Removes cards not in cardMetadataMap from cardStepIndex mapping.
 */
export function generateNextCardStepIndex(
  previousCardStepIndex: CardStepIndexMap,
  nextCardMetadataMap: CardMetadataMap
): CardStepIndexMap {
  const nextCardStepIndexMap = {} as CardStepIndexMap;
  Object.entries(previousCardStepIndex).forEach(([cardId, step]) => {
    if (nextCardMetadataMap[cardId]) {
      nextCardStepIndexMap[cardId] = step;
    }
  });
  return nextCardStepIndexMap;
}

export function generateScalarCardMinMaxStep(
  runsToSeries: RunToSeries<PluginType.SCALARS>
): MinMaxStep {
  let minStep = Infinity;
  let maxStep = -Infinity;

  for (const series of Object.values(runsToSeries)) {
    for (const stepDatum of series) {
      minStep = Math.min(minStep, stepDatum.step);
      maxStep = Math.max(maxStep, stepDatum.step);
    }
  }

  return {
    minStep,
    maxStep,
  };
}

/**
 * The maximum number of pins we allow the user to create. This is intentionally
 * finite at the moment to mitigate super long URL lengths, until there is more
 * durable value storage for pins.
 * https://github.com/tensorflow/tensorboard/issues/4242
 */
const util = {
  MAX_PIN_COUNT: 10,
};

export function canCreateNewPins(state: MetricsState) {
  const pinCountInURL =
    state.pinnedCardToOriginal.size +
    state.unresolvedImportedPinnedCards.length;
  return pinCountInURL < util.MAX_PIN_COUNT;
}
/**
 * Sets cardStepIndex for image card based on linked time selection.
 */
export function generateNextCardStepIndexFromLinkedTimeSelection(
  previousCardStepIndex: CardStepIndexMap,
  cardMetadataMap: CardMetadataMap,
  timeSeriesData: TimeSeriesData,
  timeSelection: TimeSelection
): CardStepIndexMap {
  let nextCardStepIndex = {...previousCardStepIndex};

  Object.keys(previousCardStepIndex).forEach((cardId) => {
    if (!cardId.includes('"plugin":"images"')) return;

    const steps = getImageCardSteps(cardId, cardMetadataMap, timeSeriesData);

    let nextStepIndexMetaData: CardStepIndexMetaData | null = null;
    if (timeSelection.end === null) {
      // Single Selection
      nextStepIndexMetaData = getNextImageCardStepIndexFromSingleSelection(
        timeSelection.start.step,
        steps
      );
    } else {
      // Range Selection
      const currentStepIndex = previousCardStepIndex[cardId]!.index!;
      const step = steps[currentStepIndex];
      const selectedSteps = getSelectedSteps(timeSelection, steps);

      nextStepIndexMetaData = getNextImageCardStepIndexFromRangeSelection(
        selectedSteps,
        steps,
        step
      );
    }

    if (nextStepIndexMetaData !== null) {
      nextCardStepIndex[cardId] = nextStepIndexMetaData;
    }
  });

  return nextCardStepIndex;
}

/**
 * Returns steps of an image card.
 */
export function getImageCardSteps(
  cardId: string,
  cardMetadataMap: CardMetadataMap,
  timeSeriesData: TimeSeriesData
): number[] {
  if (!hasOwn(cardMetadataMap, cardId)) {
    return [];
  }

  const {plugin, tag, sample, runId} = cardMetadataMap[cardId];
  if (runId === null) {
    return [];
  }

  const loadable = getTimeSeriesLoadable(timeSeriesData, plugin, tag, sample);
  if (loadable === null || !hasOwn(loadable.runToSeries, runId)) {
    return [];
  }

  return loadable.runToSeries[runId].map((stepDatum) => stepDatum.step);
}

/**
 * Returns the subset of steps that are within linked time selection given a list of steps
 */
function getSelectedSteps(
  timeSelection: TimeSelection | null,
  steps: number[]
) {
  if (!timeSelection) return [];

  // Single selection: returns start step if matching any step in the list, otherwise returns nothing.
  if (timeSelection.end === null) {
    if (steps.indexOf(timeSelection.start.step) !== -1)
      return [timeSelection.start.step];
    return [];
  }

  // Range selection.
  const selectedStepsInRange: number[] = [];
  for (const step of steps) {
    if (step >= timeSelection.start.step && step <= timeSelection.end.step) {
      selectedStepsInRange.push(step);
    }
  }
  return selectedStepsInRange;
}

/**
 * Gets next stepIndex for an image card based on single selection. Returns null if nothing should change.
 * @param selectedStep The selected step from linked time selection. It is equivalent to start step here
 *  since there is no `end` in linked time selection when it is single seleciton.
 */
function getNextImageCardStepIndexFromSingleSelection(
  selectedStep: number,
  steps: number[]
): CardStepIndexMetaData | null {
  // Checks exact match.
  const maybeMatchedStepIndex = steps.indexOf(selectedStep);
  if (maybeMatchedStepIndex !== -1) {
    return {index: maybeMatchedStepIndex, isClosest: false};
  }

  // Checks if start step is "close" enough to a step value and move it
  for (let i = 0; i < steps.length - 1; i++) {
    const currentStep = steps[i];
    const nextStep = steps[i + 1];
    const distance = (nextStep - currentStep) * DISTANCE_RATIO;

    if (selectedStep < currentStep) return null;
    if (selectedStep > nextStep) continue;

    if (selectedStep - currentStep <= distance) {
      return {index: i, isClosest: true};
    }
    if (nextStep - selectedStep <= distance) {
      return {index: i + 1, isClosest: true};
    }
  }

  return null;
}

/**
 * Gets next stepIndex for an image card based on range selection. Returns null if nothing should change.
 * @param selectedSteps The selected steps from linked time selection. Empty array means no steps within range.
 * @param steps The steps in an image card.
 * @param step The step where the current step index locates.
 */
function getNextImageCardStepIndexFromRangeSelection(
  selectedSteps: number[],
  steps: number[],
  step: number
): CardStepIndexMetaData | null {
  if (selectedSteps.length === 0) return null;

  const firstSelectedStep = selectedSteps[0];
  const lastSelectedStep = selectedSteps[selectedSteps.length - 1];

  // Updates step index to the closest index if it is outside the range.
  if (step > lastSelectedStep) {
    return {index: steps.indexOf(lastSelectedStep), isClosest: false};
  }
  if (step < firstSelectedStep) {
    return {index: steps.indexOf(firstSelectedStep), isClosest: false};
  }

  // Does not update index when it is in selected range.
  return null;
}

/**
 * Determines what a cards realized min max should be by examining the min and
 * max steps in the data as well as any user defined min and max
 * @param cardState
 */
export function getMinMaxStepFromCardState(cardState: Partial<CardState>) {
  const {dataMinMax, userViewBox} = cardState;
  const x = userViewBox?.x;
  if (!x) return dataMinMax;

  const minStep = x[0] < x[1] ? x[0] : x[1];
  const maxStep = minStep === x[0] ? x[1] : x[0];
  return {minStep: Math.ceil(minStep), maxStep: Math.floor(maxStep)};
}

export function getCardSelectionStateToBoolean(
  cardOverrideState: CardFeatureOverride | undefined,
  globalValue: boolean
) {
  switch (cardOverrideState) {
    case CardFeatureOverride.OVERRIDE_AS_ENABLED:
      return true;
    case CardFeatureOverride.OVERRIDE_AS_DISABLED:
      return false;
    default:
      return globalValue;
  }
}

export function cardRangeSelectionEnabled(
  cardStateMap: CardStateMap,
  globalRangeSelectionEnabled: boolean,
  linkedTimeEnabled: boolean,
  cardId: CardId
): boolean {
  if (linkedTimeEnabled) {
    return globalRangeSelectionEnabled;
  }

  const cardState = cardStateMap[cardId];
  return getCardSelectionStateToBoolean(
    cardState?.rangeSelectionOverride,
    globalRangeSelectionEnabled
  );
}

export const TEST_ONLY = {
  getImageCardSteps,
  getSelectedSteps,
  getNextImageCardStepIndexFromRangeSelection,
  getNextImageCardStepIndexFromSingleSelection,
  generateNextCardStepIndexFromLinkedTimeSelection,
  util,
};
