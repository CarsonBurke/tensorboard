# Run Local Storage State

**Status:** Proposed.

## Background

The Time Series dashboard already keeps run selection and run color state in the
NgRx `runs` feature:

- `RunsUiNamespacedState.selectionState` stores whether each run is selected.
- `RunsDataNamespacedState.runColorOverrideForGroupBy` stores user-selected run
  color overrides.
- `getRunColorMap` combines default color ids, user overrides, and the current
  color palette for rendering.

These states are currently in-memory. Reloading TensorBoard loses run
selection and color choices. For local experiment iteration, especially in
repositories with many generated runs, preserving these choices is valuable.

## Goals

- Persist run selection and user-set run colors in browser `localStorage`.
- Persist Time Series tag group expansion state and per-group pagination in
  browser `localStorage`.
- Persist only runs that are currently present in the active run directory or
  experiment set.
- Prune old runs every time state is synchronized.
- Color the most recent run white by default when the user has not explicitly
  chosen a color for that run.
- Keep the implementation isolated so syncing this fork with upstream
  TensorBoard is low-conflict.
- Keep synchronization cheap for large run directories.
- Validate the behavior on the local TensorBoard workflows for:
  - `/home/marvin/Documents/repositories/cleanrl`
  - `/home/marvin/Documents/repositories/orbit-wars`

## Non-Goals

- Do not create a TensorBoard plugin. This changes existing run selector state,
  not a new dashboard.
- Do not persist scalar data, card data, hparams, or backend run metadata.
- Do not change default color assignment, grouping, or chart rendering logic
  except for applying persisted overrides and the newest-run default.
- Do not introduce backend storage. This is intentionally browser-local.
- Do not attempt to share state across different TensorBoard origins, browsers,
  or machines.

## Current State

Relevant files:

- `tensorboard/webapp/runs/store/runs_types.ts`
- `tensorboard/webapp/runs/actions/runs_actions.ts`
- `tensorboard/webapp/runs/store/runs_reducers.ts`
- `tensorboard/webapp/runs/store/runs_selectors.ts`
- `tensorboard/webapp/runs/effects/runs_effects.ts`
- `tensorboard/webapp/util/ui_selectors.ts`

The existing `persistent_settings` module persists global UI settings in
`localStorage`. Run selection and colors are different: they are scoped to the
active run directory or experiment set and must be pruned when runs disappear.
This state should live near the `runs` feature rather than being added to
global settings.

## Design

Add a small local-storage adapter under the `runs` feature:

```text
tensorboard/webapp/runs/local_storage/
  run_local_storage_data_source.ts
  run_local_storage_data_source_test.ts
  runs_local_storage_effects.ts
  runs_local_storage_effects_test.ts
  BUILD
```

The data source owns serialization, schema migration, validation, and pruning.
Reducers remain pure and do not call `localStorage` directly. A dedicated
`RunsLocalStorageEffects` class calls the data source after fetches and user
edits.

Add a second small adapter under the `metrics` feature for Time Series
dashboard UI state:

```text
tensorboard/webapp/metrics/local_storage/
  metrics_local_storage_data_source.ts
  metrics_local_storage_data_source_test.ts
  metrics_local_storage_effects.ts
  metrics_local_storage_effects_test.ts
  BUILD
```

This keeps metrics card UI persistence out of the runs feature and avoids
changing the main metrics data-loading effects.

### Storage Key

Use one TensorBoard-owned key:

```ts
const RUN_LOCAL_STORAGE_KEY = '_tb_run_state.v1';
```

Use a separate key for metrics dashboard group state:

```ts
const METRICS_LOCAL_STORAGE_KEY = '_tb_metrics_state.v1';
```

Inside that key, store the active route/directory namespace:

```ts
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
```

Metrics state stores only tag groups present in the current dashboard:

```ts
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
```

The namespace should be derived from `Environment.data_location` plus active
route experiment ids. Do not use `NamespaceContextedState`'s active namespace
id; that value is browser-history state and is not a stable run-directory
identity.

### Pruning Rule

Every sync receives the current run ids for the active route:

```ts
currentRunIds: Set<RunId>
```

Before writing, the data source must filter all persisted maps:

```ts
selection = pick(selection, currentRunIds);
colorOverrides = pick(colorOverrides, currentRunIds);
runIds = Array.from(currentRunIds);
newestRunId = currentRunIds.has(newestRunId) ? newestRunId : undefined;
```

Metrics sync receives the current tag groups derived from loaded card metadata:

```ts
currentTagGroups: Set<string>
```

Before writing, the metrics data source filters persisted maps:

```ts
tagGroupExpanded = pick(tagGroupExpanded, currentTagGroups);
tagGroupPageIndex = pick(nonNegativeIntegerPageIndex, currentTagGroups);
tagGroups = Array.from(currentTagGroups);
```

This means localStorage never accumulates stale runs for the current directory.
If a run is deleted from disk and TensorBoard reloads, its persisted selection
and color are removed on that same sync.

On write, replace each stored `namespaces` object with only the active
namespace. This intentionally discards inactive directory state so localStorage
contains only the currently synchronized directory/route.

If the active namespace is already the only stored namespace and the pruned
payload is unchanged, skip the write. This avoids synchronous localStorage
work on hydration or repeated UI events that do not change persisted state.
Only stamp `updatedAtMs` when a write is actually needed.

### Hydration Flow

Add actions:

```ts
export const runLocalStorageHydrated = createAction(
  '[Runs] Run Local Storage Hydrated',
  props<{
    runIds: string[];
    selection: Record<string, boolean>;
    colorOverrides: Record<string, string>;
  }>()
);
```

`runLocalStorageHydrated` should merge persisted state only for run ids that
exist in the current fetch result. It should not create entries for unknown
runs.

Use serializable action props. Reducers can convert records to `Map`
internally.

Hydration happens after `fetchRunsSucceeded`, because that is when the frontend
knows the current run set. The effect should:

1. Build the active namespace.
2. Read localStorage for the namespace.
3. Prune the stored record to `runsForAllExperiments`.
4. Dispatch `runLocalStorageHydrated`.
5. If no persisted user color exists for the newest run, include
   `colorOverrides[runId] = '#fff'` in `runLocalStorageHydrated`.
6. Write the pruned and hydrated state back to localStorage.

### Sync Flow

Add a runs effect that listens to:

- `runSelectionToggled`
- `singleRunSelected`
- `runPageSelectionToggled`
- `runColorChanged`

For user-edit actions, read the latest selector values with `withLatestFrom`:

- current `Environment.data_location`
- current route experiment ids
- current dashboard runs
- current run selection
- run color overrides

Then call:

```ts
sync(namespace, currentRuns, selection, colorOverrides);
```

`sync` must always prune before writing.

Do not add a second sync listener on `fetchRunsSucceeded`; hydration owns that
path so it can read, prune, apply newest-run coloring, dispatch hydration, and
write final state in one ordered sequence.

### Metrics Group Hydration and Sync

Add actions:

```ts
export const metricsTagGroupPageIndexChanged = createAction(
  '[Metrics] Metrics Tag Group Page Index Changed',
  props<{tagGroup: string; pageIndex: number}>()
);

export const metricsLocalStorageHydrated = createAction(
  '[Metrics] Metrics Local Storage Hydrated',
  props<{
    tagGroups: string[];
    tagGroupExpanded: Record<string, boolean>;
    tagGroupPageIndex: Record<string, number>;
  }>()
);
```

Keep `tagGroupExpanded` in metrics NgRx state and add
`tagGroupPageIndex: Map<string, number>`. `CardGridContainer` should read page
index from the store only when it has a real tag group name. Pinned and filtered
grids can keep their local page state because they are not collapsible tag
groups.

Hydrate metrics group state after `metricsTagMetadataLoaded` and
`environmentLoaded`. Both are needed because tag metadata and environment
metadata load independently. Hydration should:

1. Build the active namespace from `Environment.data_location` and route
   experiment ids.
2. Derive current tag groups from current card metadata.
3. Read localStorage for that namespace.
4. Merge stored group expansion and page indices over current in-memory
   defaults.
5. Dispatch `metricsLocalStorageHydrated`.
6. Write the pruned hydrated state back to localStorage.

Sync metrics group state after:

- `metricsTagGroupExpansionChanged`
- `metricsTagGroupPageIndexChanged`

The effect should read current group names and current group maps from
selectors, then write the pruned active namespace.

### Newest Run Detection

Use `Run.startTime` as the primary ordering key. If `startTime` is missing,
zero, or tied, use `Run.name` as a deterministic fallback.

The newest-run default color applies only when:

- the run exists in the current route,
- there is no persisted color override for that run,
- there is no in-memory user color override for that run, and
- this run has not already been assigned a newest-run default in the stored
  namespace.

This prevents repeatedly overwriting a user color. If the user changes the
newest run away from white, the explicit `runColorChanged` value wins and is
persisted.

### Performance

The effects should keep synchronization O(number of runs or groups in the
current route).
Avoid scanning all `localStorage` keys. Use one key and one JSON parse/stringify
per sync.

For large run lists:

- Convert current run ids to a `Set` once.
- Prune maps in a single pass.
- Avoid deep equality checks over full run metadata.
- Page-level selection already dispatches one action; if a future UI dispatches
  many row actions for a batch edit, add `auditTime(0)` or a small debounce.
- Do not write if the serialized namespace is byte-for-byte unchanged from the
  last write in this browser session.

The stored payload is intentionally compact: booleans and hex strings keyed by
run id. It should scale to thousands of runs without materially affecting UI
responsiveness.

### Upstream Sync Strategy

Keep upstream merge conflicts low by isolating custom code:

- Add new files under `tensorboard/webapp/runs/local_storage/`.
- Keep changes to existing files minimal:
  - add action definitions in `runs_actions.ts`,
  - add reducer cases in `runs_reducers.ts`,
  - add selectors only if existing selectors are insufficient,
  - register `RunsLocalStorageEffects` in `runs_module.ts`,
  - add Bazel deps.
- Do not modify chart components, runs table components, color palette code, or
  global `persistent_settings`.
- Do not modify `runs/effects/runs_effects.ts`.
- Prefer helper functions in the new data-source file over spreading storage
  logic through reducers/effects.

If upstream later adds native run-state persistence, this local feature should
be removable by deleting `runs/local_storage` and reverting the small action,
reducer, effect, and BUILD wiring changes.

## Validation Plan

Unit tests:

- Data source:
  - reads empty/malformed localStorage as empty state,
  - prunes missing run ids on every sync,
  - removes unknown selection and color entries,
  - preserves only current-directory run ids,
  - handles storage quota or unavailable localStorage without breaking the app.
- Reducers:
  - hydration merges only known run ids,
  - persisted selection overrides default selection,
  - persisted colors override default color ids,
  - newest-run default writes white only when no user override exists.
- Effects:
  - hydrate happens after `fetchRunsSucceeded`,
  - user selection/color edits trigger one pruned write,
  - deleted runs are removed after reload,
  - newest run is selected for white by `startTime`.

Manual validation:

1. From `/home/marvin/Documents/repositories/cleanrl`, run TensorBoard against
   the local runs directory.
2. Select a subset of runs and change at least one color.
3. Reload the browser; selection and colors should be restored.
4. Delete or move a run directory, reload TensorBoard, and confirm the deleted
   run is removed from localStorage.
5. Start a newer run; after reload, that run should default to white unless a
   user color exists.
6. Repeat the same workflow from
   `/home/marvin/Documents/repositories/orbit-wars`.
7. Confirm the two repositories do not leak run selections or colors into each
   other when served from the same TensorBoard origin.

## Open Questions

- White is ideal for dark mode but low contrast in light mode. The requested
  behavior is white; a later refinement could make this theme-aware.
- Existing color grouping resets user color overrides on group-by changes. This
  spec keeps that behavior unless the persisted override is rehydrated after
  the next run fetch.
