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
import {CardId} from '../types';

const METRICS_LOCAL_STORAGE_KEY = '_tb_metrics_state.v1';
const VERSION = 1;

/**
 * Upper bound on the number of cards whose view state we keep in local
 * storage. Card ids are serialized card metadata, so an experiment with many
 * tags could otherwise grow the payload without bound.
 */
const MAX_PERSISTED_CARDS = 256;

/** Bounds on drag-resized heights, in px. */
const MIN_CARD_DIMENSION_PX = 40;
const MAX_CARD_DIMENSION_PX = 5000;

declare interface StoredMetricsStateV1 {
  version: 1;
  namespaces: Record<string, StoredMetricsNamespaceV1>;
}

declare interface StoredMetricsNamespaceV1 {
  updatedAtMs: number;
  tagGroups: string[];
  tagGroupExpanded: Record<string, boolean>;
  tagGroupPageIndex: Record<string, number>;
  // Added after VERSION 1 shipped; optional so older payloads still parse.
  cardState?: Record<string, PersistedCardState>;
}

/**
 * The subset of `CardState` that survives a page load.
 */
export interface PersistedCardState {
  fullWidth?: boolean;
  tableExpanded?: boolean;
  chartHeight?: number;
  tableHeight?: number;
}

export interface MetricsLocalStorageState {
  tagGroupExpanded: Map<string, boolean>;
  tagGroupPageIndex: Map<string, number>;
  cardState: Map<CardId, PersistedCardState>;
}

function safeParse(serialized: string | null): StoredMetricsStateV1 {
  if (!serialized) {
    return {version: VERSION, namespaces: {}};
  }

  try {
    const parsed = JSON.parse(serialized) as Partial<StoredMetricsStateV1>;
    if (
      parsed.version !== VERSION ||
      !parsed.namespaces ||
      typeof parsed.namespaces !== 'object'
    ) {
      return {version: VERSION, namespaces: {}};
    }
    return parsed as StoredMetricsStateV1;
  } catch {
    return {version: VERSION, namespaces: {}};
  }
}

function sanitizeNamespace(
  storedNamespace: StoredMetricsNamespaceV1 | undefined,
  currentTagGroups: Set<string>
): MetricsLocalStorageState {
  const tagGroupExpanded = new Map<string, boolean>();
  const tagGroupPageIndex = new Map<string, number>();
  const cardState = new Map<CardId, PersistedCardState>();
  if (!storedNamespace || typeof storedNamespace !== 'object') {
    return {tagGroupExpanded, tagGroupPageIndex, cardState};
  }

  if (
    storedNamespace.tagGroupExpanded &&
    typeof storedNamespace.tagGroupExpanded === 'object'
  ) {
    for (const [tagGroup, value] of Object.entries(
      storedNamespace.tagGroupExpanded
    )) {
      if (currentTagGroups.has(tagGroup) && typeof value === 'boolean') {
        tagGroupExpanded.set(tagGroup, value);
      }
    }
  }

  if (
    storedNamespace.tagGroupPageIndex &&
    typeof storedNamespace.tagGroupPageIndex === 'object'
  ) {
    for (const [tagGroup, value] of Object.entries(
      storedNamespace.tagGroupPageIndex
    )) {
      if (
        currentTagGroups.has(tagGroup) &&
        Number.isInteger(value) &&
        value >= 0
      ) {
        tagGroupPageIndex.set(tagGroup, value);
      }
    }
  }

  if (
    storedNamespace.cardState &&
    typeof storedNamespace.cardState === 'object'
  ) {
    for (const [cardId, value] of Object.entries(storedNamespace.cardState)) {
      if (cardState.size >= MAX_PERSISTED_CARDS) {
        break;
      }
      const sanitized = sanitizeCardState(value);
      if (sanitized) {
        cardState.set(cardId, sanitized);
      }
    }
  }

  return {tagGroupExpanded, tagGroupPageIndex, cardState};
}

function isPersistedDimension(value: unknown): value is number {
  return (
    Number.isInteger(value) &&
    (value as number) >= MIN_CARD_DIMENSION_PX &&
    (value as number) <= MAX_CARD_DIMENSION_PX
  );
}

/**
 * Keeps only the persisted keys, and only when they hold a usable value. Cards
 * left without any usable value are dropped by returning `null`.
 */
function sanitizeCardState(value: unknown): PersistedCardState | null {
  if (!value || typeof value !== 'object') {
    return null;
  }
  const stored = value as Record<keyof PersistedCardState, unknown>;
  const sanitized: PersistedCardState = {};
  if (typeof stored.fullWidth === 'boolean') {
    sanitized.fullWidth = stored.fullWidth;
  }
  if (typeof stored.tableExpanded === 'boolean') {
    sanitized.tableExpanded = stored.tableExpanded;
  }
  if (isPersistedDimension(stored.chartHeight)) {
    sanitized.chartHeight = stored.chartHeight;
  }
  if (isPersistedDimension(stored.tableHeight)) {
    sanitized.tableHeight = stored.tableHeight;
  }
  return Object.keys(sanitized).length === 0 ? null : sanitized;
}

/**
 * Merges the card state handed to `setState` over what is already stored.
 * Given entries win key by key; entries that were not given are kept, because
 * the store prunes card state for cards outside the current catalog window and
 * a sync must not erase the state of a card that is merely off-window. Only
 * the cap drops entries, and it keeps the given ones.
 */
function mergeCardStates(
  storedCardState: Record<string, PersistedCardState> | undefined,
  nextCardState: Map<CardId, PersistedCardState>
): Record<string, PersistedCardState> {
  const merged = new Map<string, PersistedCardState>();
  for (const [cardId, value] of nextCardState) {
    const sanitized = sanitizeCardState(value);
    if (sanitized) {
      merged.set(cardId, sanitized);
    }
  }
  if (storedCardState && typeof storedCardState === 'object') {
    for (const [cardId, value] of Object.entries(storedCardState)) {
      const sanitized = sanitizeCardState(value);
      if (!sanitized) {
        continue;
      }
      const given = merged.get(cardId);
      merged.set(cardId, given ? {...sanitized, ...given} : sanitized);
    }
  }
  if (merged.size <= MAX_PERSISTED_CARDS) {
    return Object.fromEntries(merged);
  }
  return Object.fromEntries(
    Array.from(merged.entries()).slice(0, MAX_PERSISTED_CARDS)
  );
}

/**
 * Order-independent identity of a card state record, so that reordering card
 * ids or object keys alone does not count as a change.
 */
function cardStateFingerprint(
  cardState: Record<string, PersistedCardState> | undefined
): string {
  const entries = Object.entries(cardState ?? {}).sort(([a], [b]) =>
    a < b ? -1 : a > b ? 1 : 0
  );
  return JSON.stringify(
    entries.map(([cardId, value]) => [
      cardId,
      value?.fullWidth,
      value?.tableExpanded,
      value?.chartHeight,
      value?.tableHeight,
    ])
  );
}

function mapToObject<T>(
  values: Map<string, T>,
  currentTagGroups: Set<string>
): Record<string, T> {
  return Object.fromEntries(
    Array.from(values.entries()).filter(([tagGroup]) =>
      currentTagGroups.has(tagGroup)
    )
  );
}

function pageIndexMapToObject(
  values: Map<string, number>,
  currentTagGroups: Set<string>
): Record<string, number> {
  return Object.fromEntries(
    Array.from(values.entries()).filter(([tagGroup, value]) => {
      return (
        currentTagGroups.has(tagGroup) && Number.isInteger(value) && value >= 0
      );
    })
  );
}

function namespacesAreEquivalent(
  storedNamespace: StoredMetricsNamespaceV1 | undefined,
  nextNamespace: StoredMetricsNamespaceV1
): boolean {
  if (!storedNamespace) {
    return false;
  }
  return (
    JSON.stringify({
      tagGroups: storedNamespace.tagGroups,
      tagGroupExpanded: storedNamespace.tagGroupExpanded,
      tagGroupPageIndex: storedNamespace.tagGroupPageIndex,
    }) ===
      JSON.stringify({
        tagGroups: nextNamespace.tagGroups,
        tagGroupExpanded: nextNamespace.tagGroupExpanded,
        tagGroupPageIndex: nextNamespace.tagGroupPageIndex,
      }) &&
    cardStateFingerprint(storedNamespace.cardState) ===
      cardStateFingerprint(nextNamespace.cardState)
  );
}

@Injectable({providedIn: 'root'})
export class MetricsLocalStorageDataSource {
  private inMemoryValue: string | null | undefined;
  private hasPendingWrite = false;

  getState(
    namespaceId: string,
    currentTagGroups: string[]
  ): MetricsLocalStorageState {
    const currentTagGroupSet = new Set(currentTagGroups);
    const storedState = safeParse(this.getItem());
    const stored = storedState.namespaces[namespaceId];
    for (const group of Object.keys(stored?.tagGroupExpanded ?? {})) {
      currentTagGroupSet.add(group);
    }
    for (const group of Object.keys(stored?.tagGroupPageIndex ?? {})) {
      currentTagGroupSet.add(group);
    }
    return sanitizeNamespace(
      storedState.namespaces[namespaceId],
      currentTagGroupSet
    );
  }

  setState(
    namespaceId: string,
    currentTagGroups: string[],
    state: MetricsLocalStorageState
  ) {
    const currentTagGroupSet = new Set(currentTagGroups);
    for (const group of state.tagGroupExpanded.keys())
      currentTagGroupSet.add(group);
    for (const group of state.tagGroupPageIndex.keys())
      currentTagGroupSet.add(group);

    const storedState = safeParse(this.getItem());
    const cardState = mergeCardStates(
      storedState.namespaces[namespaceId]?.cardState,
      state.cardState
    );
    const hasCardState = Object.keys(cardState).length > 0;
    if (currentTagGroupSet.size === 0 && !hasCardState) {
      this.removeItem();
      return;
    }

    const nextNamespace: StoredMetricsNamespaceV1 = {
      updatedAtMs: 0,
      tagGroups: Array.from(currentTagGroupSet),
      tagGroupExpanded: mapToObject(state.tagGroupExpanded, currentTagGroupSet),
      tagGroupPageIndex: pageIndexMapToObject(
        state.tagGroupPageIndex,
        currentTagGroupSet
      ),
    };
    if (hasCardState) {
      nextNamespace.cardState = cardState;
    }

    if (
      !this.hasPendingWrite &&
      Object.keys(storedState.namespaces).length === 1 &&
      namespacesAreEquivalent(
        storedState.namespaces[namespaceId],
        nextNamespace
      )
    ) {
      return;
    }

    nextNamespace.updatedAtMs = Date.now();
    const nextState: StoredMetricsStateV1 = {
      version: VERSION,
      namespaces: {
        [namespaceId]: nextNamespace,
      },
    };
    this.setItem(JSON.stringify(nextState));
  }

  private getItem(): string | null {
    if (this.inMemoryValue !== undefined) {
      return this.inMemoryValue;
    }
    try {
      return window.localStorage.getItem(METRICS_LOCAL_STORAGE_KEY);
    } catch {
      return null;
    }
  }

  private setItem(value: string) {
    try {
      window.localStorage.setItem(METRICS_LOCAL_STORAGE_KEY, value);
      this.inMemoryValue = undefined;
      this.hasPendingWrite = false;
    } catch {
      // Keep the newest value for this session so a stale stored value cannot
      // overwrite live state after quota or availability failures.
      this.inMemoryValue = value;
      this.hasPendingWrite = true;
    }
  }

  private removeItem() {
    try {
      window.localStorage.removeItem(METRICS_LOCAL_STORAGE_KEY);
      this.inMemoryValue = undefined;
      this.hasPendingWrite = false;
    } catch {
      this.inMemoryValue = null;
      this.hasPendingWrite = true;
    }
  }
}

export const TEST_ONLY = {
  METRICS_LOCAL_STORAGE_KEY,
  MAX_PERSISTED_CARDS,
};
