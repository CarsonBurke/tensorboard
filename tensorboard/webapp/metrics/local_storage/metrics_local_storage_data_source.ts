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

const METRICS_LOCAL_STORAGE_KEY = '_tb_metrics_state.v1';
const VERSION = 1;

declare interface StoredMetricsStateV1 {
  version: 1;
  namespaces: Record<string, StoredMetricsNamespaceV1>;
}

declare interface StoredMetricsNamespaceV1 {
  updatedAtMs: number;
  tagGroups: string[];
  tagGroupExpanded: Record<string, boolean>;
  tagGroupPageIndex: Record<string, number>;
}

export interface MetricsLocalStorageState {
  tagGroupExpanded: Map<string, boolean>;
  tagGroupPageIndex: Map<string, number>;
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
  if (!storedNamespace || typeof storedNamespace !== 'object') {
    return {tagGroupExpanded, tagGroupPageIndex};
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

  return {tagGroupExpanded, tagGroupPageIndex};
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
    })
  );
}

@Injectable({providedIn: 'root'})
export class MetricsLocalStorageDataSource {
  getState(
    namespaceId: string,
    currentTagGroups: string[]
  ): MetricsLocalStorageState {
    const currentTagGroupSet = new Set(currentTagGroups);
    const storedState = safeParse(this.getItem());
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
    if (currentTagGroupSet.size === 0) {
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

    const storedState = safeParse(this.getItem());
    if (
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
    try {
      return window.localStorage.getItem(METRICS_LOCAL_STORAGE_KEY);
    } catch {
      return null;
    }
  }

  private setItem(value: string) {
    try {
      window.localStorage.setItem(METRICS_LOCAL_STORAGE_KEY, value);
    } catch {
      // localStorage can be unavailable or full. Metrics UI persistence should
      // never break TensorBoard itself.
    }
  }

  private removeItem() {
    try {
      window.localStorage.removeItem(METRICS_LOCAL_STORAGE_KEY);
    } catch {
      // Ignore localStorage failures.
    }
  }
}

export const TEST_ONLY = {
  METRICS_LOCAL_STORAGE_KEY,
};
