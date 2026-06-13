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
import {Run} from '../types';

const RUN_LOCAL_STORAGE_KEY = '_tb_run_state.v1';
const VERSION = 1;
const HEX_COLOR_RE =
  /^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$/;

declare interface StoredRunStateV1 {
  version: 1;
  namespaces: Record<string, StoredRunNamespaceV1>;
}

declare interface StoredRunNamespaceV1 {
  updatedAtMs: number;
  runIds: string[];
  selection: Record<string, boolean>;
  colorOverrides: Record<string, string>;
  newestRunId?: string;
}

export interface RunLocalStorageState {
  selection: Map<string, boolean>;
  colorOverrides: Map<string, string>;
  newestRunId?: string;
}

function safeParse(serialized: string | null): StoredRunStateV1 {
  if (!serialized) {
    return {version: VERSION, namespaces: {}};
  }

  try {
    const parsed = JSON.parse(serialized) as Partial<StoredRunStateV1>;
    if (
      parsed.version !== VERSION ||
      !parsed.namespaces ||
      typeof parsed.namespaces !== 'object'
    ) {
      return {version: VERSION, namespaces: {}};
    }
    return parsed as StoredRunStateV1;
  } catch {
    return {version: VERSION, namespaces: {}};
  }
}

function sanitizeNamespace(
  storedNamespace: StoredRunNamespaceV1 | undefined,
  currentRunIds: Set<string>
): RunLocalStorageState {
  const selection = new Map<string, boolean>();
  const colorOverrides = new Map<string, string>();
  if (!storedNamespace || typeof storedNamespace !== 'object') {
    return {selection, colorOverrides};
  }

  if (
    storedNamespace.selection &&
    typeof storedNamespace.selection === 'object'
  ) {
    for (const [runId, value] of Object.entries(storedNamespace.selection)) {
      if (currentRunIds.has(runId) && typeof value === 'boolean') {
        selection.set(runId, value);
      }
    }
  }

  if (
    storedNamespace.colorOverrides &&
    typeof storedNamespace.colorOverrides === 'object'
  ) {
    for (const [runId, value] of Object.entries(
      storedNamespace.colorOverrides
    )) {
      if (
        currentRunIds.has(runId) &&
        typeof value === 'string' &&
        HEX_COLOR_RE.test(value)
      ) {
        colorOverrides.set(runId, value);
      }
    }
  }

  const state: RunLocalStorageState = {
    selection,
    colorOverrides,
  };
  if (
    storedNamespace.newestRunId &&
    currentRunIds.has(storedNamespace.newestRunId)
  ) {
    state.newestRunId = storedNamespace.newestRunId;
  }
  return state;
}

function mapToObject<T>(
  values: Map<string, T>,
  currentRunIds: Set<string>
): Record<string, T> {
  return Object.fromEntries(
    Array.from(values.entries()).filter(([runId]) => currentRunIds.has(runId))
  );
}

function namespacesAreEquivalent(
  storedNamespace: StoredRunNamespaceV1 | undefined,
  nextNamespace: StoredRunNamespaceV1
): boolean {
  if (!storedNamespace) {
    return false;
  }
  return (
    JSON.stringify({
      runIds: storedNamespace.runIds,
      selection: storedNamespace.selection,
      colorOverrides: storedNamespace.colorOverrides,
      newestRunId: storedNamespace.newestRunId,
    }) ===
    JSON.stringify({
      runIds: nextNamespace.runIds,
      selection: nextNamespace.selection,
      colorOverrides: nextNamespace.colorOverrides,
      newestRunId: nextNamespace.newestRunId,
    })
  );
}

@Injectable({providedIn: 'root'})
export class RunLocalStorageDataSource {
  getState(namespaceId: string, currentRuns: Run[]): RunLocalStorageState {
    const currentRunIds = new Set(currentRuns.map((run) => run.id));
    const storedState = safeParse(this.getItem());
    return sanitizeNamespace(
      storedState.namespaces[namespaceId],
      currentRunIds
    );
  }

  setState(
    namespaceId: string,
    currentRuns: Run[],
    state: RunLocalStorageState
  ) {
    const currentRunIds = new Set(currentRuns.map((run) => run.id));
    if (currentRunIds.size === 0) {
      this.removeItem();
      return;
    }

    const nextNamespace: StoredRunNamespaceV1 = {
      updatedAtMs: 0,
      runIds: Array.from(currentRunIds),
      selection: mapToObject(state.selection, currentRunIds),
      colorOverrides: mapToObject(state.colorOverrides, currentRunIds),
    };
    if (state.newestRunId && currentRunIds.has(state.newestRunId)) {
      nextNamespace.newestRunId = state.newestRunId;
    }

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
    const nextState: StoredRunStateV1 = {
      version: VERSION,
      namespaces: {
        [namespaceId]: nextNamespace,
      },
    };
    this.setItem(JSON.stringify(nextState));
  }

  private getItem(): string | null {
    try {
      return window.localStorage.getItem(RUN_LOCAL_STORAGE_KEY);
    } catch {
      return null;
    }
  }

  private setItem(value: string) {
    try {
      window.localStorage.setItem(RUN_LOCAL_STORAGE_KEY, value);
    } catch {
      // localStorage can be unavailable or full. Run state persistence should
      // never break TensorBoard itself.
    }
  }

  private removeItem() {
    try {
      window.localStorage.removeItem(RUN_LOCAL_STORAGE_KEY);
    } catch {
      // Ignore localStorage failures.
    }
  }
}

export const TEST_ONLY = {
  RUN_LOCAL_STORAGE_KEY,
};
