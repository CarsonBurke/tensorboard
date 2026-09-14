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
import {TestBed} from '@angular/core/testing';
import {Actions} from '@ngrx/effects';
import {provideMockActions} from '@ngrx/effects/testing';
import {Action, Store} from '@ngrx/store';
import {MockStore} from '@ngrx/store/testing';
import {ReplaySubject} from 'rxjs';
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
import {SortingOrder} from '../../widgets/data_table/types';
import * as runsActions from '../actions';
import {GroupByKey, Run} from '../types';
import {provideMockTbStore} from '../../testing/utils';
import {
  RunLocalStorageDataSource,
  TEST_ONLY as STORAGE_TEST_ONLY,
} from './run_local_storage_data_source';
import {RunsLocalStorageEffects, TEST_ONLY} from './runs_local_storage_effects';

function createRun(id: string, startTime: number): Run {
  return {id, name: id, startTime};
}

describe('RunsLocalStorageEffects', () => {
  let actions: ReplaySubject<Action>;
  let effects: RunsLocalStorageEffects;
  let store: MockStore<State>;
  let dataSource: RunLocalStorageDataSource;
  let dispatchedActions: Action[];

  beforeEach(async () => {
    actions = new ReplaySubject<Action>(1);

    await TestBed.configureTestingModule({
      providers: [
        provideMockActions(actions),
        provideMockTbStore(),
        RunsLocalStorageEffects,
        RunLocalStorageDataSource,
      ],
    }).compileComponents();

    store = TestBed.inject<Store<State>>(Store) as MockStore<State>;
    dataSource = TestBed.inject(RunLocalStorageDataSource);
    effects = TestBed.inject(RunsLocalStorageEffects);
    dispatchedActions = [];
    (spyOn(store, 'dispatch') as jasmine.Spy).and.callFake((action: Action) => {
      dispatchedActions.push(action);
    });

    store.overrideSelector(getEnvironment, {
      data_location: '/tmp/tensorboard/runs',
      window_title: '',
    });
    store.overrideSelector(getExperimentIdsFromRoute, ['exp1']);
    store.overrideSelector(getRunSelectionMap, new Map<string, boolean>());
    store.overrideSelector(getRunColorOverride, new Map<string, string>());
    store.overrideSelector(getDashboardRuns, []);
    store.overrideSelector(getRunsTableSortingInfo, {
      name: 'run',
      order: SortingOrder.ASCENDING,
    });
  });

  afterEach(() => {
    store?.resetSelectors();
    window.localStorage.removeItem(STORAGE_TEST_ONLY.RUN_LOCAL_STORAGE_KEY);
  });

  it('chooses by name when start times are zero or missing', () => {
    expect(
      TEST_ONLY.getNewestRunId([
        {id: 'a', name: 'a', startTime: 0},
        {id: 'z', name: 'z', startTime: undefined},
      ])
    ).toBe('z');
  });

  it('clears persisted state when the fetched route has no runs', () => {
    const setStateSpy = spyOn(dataSource, 'setState').and.stub();
    effects.hydrateFetchedRunsFromLocalStorage$.subscribe();

    actions.next(
      runsActions.fetchRunsSucceeded({
        experimentIds: ['exp1'],
        runsForAllExperiments: [],
        newRuns: {exp1: {runs: []}},
      })
    );

    expect(setStateSpy).toHaveBeenCalledOnceWith(
      jasmine.stringMatching('/tmp/tensorboard/runs'),
      [],
      {
        selection: new Map(),
        colorOverrides: new Map(),
      }
    );
    expect(dispatchedActions).toEqual([
      runsActions.runLocalStorageHydrated({
        runIds: [],
        selection: {},
        colorOverrides: {},
        restoredSelection: false,
      }),
    ]);
  });

  it('reports an empty restore before runs have loaded without clearing', () => {
    const setStateSpy = spyOn(dataSource, 'setState').and.stub();
    effects.hydrateExistingRunsFromLocalStorage$.subscribe();

    actions.next(
      coreActions.environmentLoaded({
        environment: {
          data_location: '/tmp/tensorboard/runs',
          window_title: '',
        },
      })
    );

    expect(setStateSpy).not.toHaveBeenCalled();
    expect(dispatchedActions).toEqual([
      runsActions.runLocalStorageHydrated({
        runIds: [],
        selection: {},
        colorOverrides: {},
        restoredSelection: false,
      }),
    ]);
  });

  it('restores a persisted selection before runs have loaded', () => {
    const namespace = TEST_ONLY.getNamespace('/tmp/tensorboard/runs', [
      'exp1',
    ])!;
    dataSource.setState(namespace, [createRun('paged', 1)], {
      selection: new Map([
        ['paged', false],
        ['chosen', true],
      ]),
      colorOverrides: new Map(),
    });
    const setStateSpy = spyOn(dataSource, 'setState').and.callThrough();
    effects.hydrateExistingRunsFromLocalStorage$.subscribe();

    actions.next(
      coreActions.environmentLoaded({
        environment: {
          data_location: '/tmp/tensorboard/runs',
          window_title: '',
        },
      })
    );

    expect(dispatchedActions).toEqual([
      runsActions.runLocalStorageHydrated({
        runIds: ['paged', 'chosen'],
        selection: {paged: false, chosen: true},
        colorOverrides: {},
        restoredSelection: true,
      }),
    ]);
    // Nothing may be pruned while the run list is still unknown.
    expect(setStateSpy).not.toHaveBeenCalled();
  });

  it('hydrates from storage, moves auto white to the newest run, and writes once', () => {
    const oldRun = createRun('old', 1);
    const newRun = createRun('new', 2);
    spyOn(dataSource, 'getState').and.returnValue({
      selection: new Map([['old', false]]),
      colorOverrides: new Map([['old', '#fff']]),
      newestRunId: 'old',
    });
    const setStateSpy = spyOn(dataSource, 'setState').and.stub();
    store.overrideSelector(
      getRunSelectionMap,
      new Map([
        ['old', true],
        ['new', true],
      ])
    );
    store.refreshState();

    effects.hydrateFetchedRunsFromLocalStorage$.subscribe();
    actions.next(
      runsActions.fetchRunsSucceeded({
        experimentIds: ['exp1'],
        runsForAllExperiments: [oldRun, newRun],
        newRuns: {exp1: {runs: [oldRun, newRun]}},
      })
    );

    expect(dispatchedActions).toEqual([
      runsActions.runLocalStorageHydrated({
        runIds: ['old', 'new'],
        selection: {old: false, new: true},
        colorOverrides: {new: '#fff'},
        restoredSelection: true,
      }),
    ]);
    expect(setStateSpy).toHaveBeenCalledOnceWith(
      jasmine.stringMatching('/tmp/tensorboard/runs'),
      [oldRun, newRun],
      {
        selection: new Map([
          ['old', false],
          ['new', true],
        ]),
        colorOverrides: new Map([['new', '#fff']]),
        newestRunId: 'new',
      }
    );
  });

  it('colors the newest run white on a window holding the whole catalog', () => {
    const oldRun = createRun('old', 1);
    const newRun = createRun('new', 2);
    const setStateSpy = spyOn(dataSource, 'setState').and.stub();
    effects.hydrateFetchedRunsFromLocalStorage$.subscribe();

    actions.next(
      runsActions.fetchRunsSucceeded({
        experimentIds: ['exp1'],
        runsForAllExperiments: [oldRun, newRun],
        newRuns: {exp1: {runs: [oldRun, newRun]}},
        catalog: {runIds: ['old', 'new'], totals: {exp1: 2}, offset: 0},
      })
    );

    expect(dispatchedActions).toEqual([
      runsActions.runLocalStorageHydrated({
        runIds: ['old', 'new'],
        selection: {},
        colorOverrides: {new: '#fff'},
        restoredSelection: false,
      }),
    ]);
    expect(setStateSpy).toHaveBeenCalledOnceWith(
      jasmine.stringMatching('/tmp/tensorboard/runs'),
      [oldRun, newRun],
      {
        selection: new Map(),
        colorOverrides: new Map([['new', '#fff']]),
        newestRunId: 'new',
      }
    );
  });

  it('leaves auto coloring alone for a partial window', () => {
    const oldRun = createRun('old', 1);
    const newRun = createRun('new', 2);
    const setStateSpy = spyOn(dataSource, 'setState').and.stub();
    effects.hydrateFetchedRunsFromLocalStorage$.subscribe();

    actions.next(
      runsActions.fetchRunsSucceeded({
        experimentIds: ['exp1'],
        runsForAllExperiments: [oldRun, newRun],
        newRuns: {exp1: {runs: [oldRun, newRun]}},
        catalog: {runIds: ['old', 'new'], totals: {exp1: 1000}, offset: 0},
      })
    );

    expect(dispatchedActions).toEqual([
      runsActions.runLocalStorageHydrated({
        runIds: ['old', 'new'],
        selection: {},
        colorOverrides: {},
        restoredSelection: false,
      }),
    ]);
    expect(setStateSpy).toHaveBeenCalledOnceWith(
      jasmine.stringMatching('/tmp/tensorboard/runs'),
      [oldRun, newRun],
      {selection: new Map(), colorOverrides: new Map()}
    );
  });

  it('hydrates existing runs when environment loads after runs', () => {
    const run = createRun('run1', 1);
    spyOn(dataSource, 'getState').and.returnValue({
      selection: new Map([['run1', true]]),
      colorOverrides: new Map([['run1', '#123456']]),
    });
    const setStateSpy = spyOn(dataSource, 'setState').and.stub();
    store.overrideSelector(getDashboardRuns, [
      {...run, hparams: null, metrics: null, experimentId: 'exp1'},
    ]);
    store.overrideSelector(getRunSelectionMap, new Map([['run1', false]]));
    store.refreshState();

    effects.hydrateExistingRunsFromLocalStorage$.subscribe();
    actions.next(
      coreActions.environmentLoaded({
        environment: {
          data_location: '/tmp/tensorboard/runs',
          window_title: '',
        },
      })
    );

    expect(dispatchedActions).toEqual([
      runsActions.runLocalStorageHydrated({
        runIds: ['run1'],
        selection: {run1: true},
        colorOverrides: {run1: '#123456'},
        restoredSelection: true,
      }),
    ]);
    expect(setStateSpy).toHaveBeenCalledOnceWith(
      jasmine.stringMatching('/tmp/tensorboard/runs'),
      [jasmine.objectContaining({id: 'run1', experimentId: 'exp1'})],
      {
        selection: new Map([['run1', true]]),
        colorOverrides: new Map([['run1', '#123456']]),
      }
    );
  });

  it('keeps live off-page selections when another view changes saved selection', () => {
    const chosen = createRun('chosen', 2);
    const old = createRun('old', 1);
    const namespace = TEST_ONLY.getNamespace('/tmp/tensorboard/runs', [
      'exp1',
    ])!;
    dataSource.setState(namespace, [chosen, old], {
      selection: new Map([
        ['chosen', true],
        ['old', false],
      ]),
      colorOverrides: new Map(),
    });
    effects.hydrateFetchedRunsFromLocalStorage$.subscribe();
    const fetchPage = (runs: Run[]) =>
      actions.next(
        runsActions.fetchRunsSucceeded({
          experimentIds: ['exp1'],
          runsForAllExperiments: runs,
          newRuns: {exp1: {runs}},
          catalog: {
            runIds: runs.map(({id}) => id),
            totals: {exp1: 1000},
            offset: 0,
          },
        })
      );
    fetchPage([chosen]);
    store.overrideSelector(
      getRunSelectionMap,
      new Map([
        ['chosen', true],
        ['old', false],
      ])
    );
    store.refreshState();

    // A second tab saves a different selection before this view pages/reloads.
    dataSource.setState(namespace, [chosen, old], {
      selection: new Map([
        ['chosen', false],
        ['old', true],
      ]),
      colorOverrides: new Map(),
    });
    dispatchedActions = [];
    fetchPage([old]);

    expect(dispatchedActions).toContain(
      runsActions.runLocalStorageHydrated({
        runIds: ['old', 'chosen'],
        selection: {chosen: true, old: false},
        colorOverrides: {},
        restoredSelection: false,
      })
    );
    expect(dataSource.getState(namespace, [old]).selection).toEqual(
      new Map([
        ['chosen', true],
        ['old', false],
      ])
    );
  });

  it('ignores stale fetches from a previous route', () => {
    const run = createRun('run1', 1);
    const getStateSpy = spyOn(dataSource, 'getState').and.returnValue({
      selection: new Map([['run1', true]]),
      colorOverrides: new Map([['run1', '#123456']]),
    });
    const setStateSpy = spyOn(dataSource, 'setState').and.stub();
    store.overrideSelector(getExperimentIdsFromRoute, ['current']);
    store.refreshState();

    effects.hydrateFetchedRunsFromLocalStorage$.subscribe();
    actions.next(
      runsActions.fetchRunsSucceeded({
        experimentIds: ['stale'],
        runsForAllExperiments: [run],
        newRuns: {stale: {runs: [run]}},
      })
    );

    expect(dispatchedActions).toEqual([]);
    expect(getStateSpy).not.toHaveBeenCalled();
    expect(setStateSpy).not.toHaveBeenCalled();
  });

  it('syncs current run selection and color overrides on user edits', () => {
    const run = createRun('run1', 1);
    const currentRun = {
      ...run,
      hparams: null,
      metrics: null,
      experimentId: 'exp1',
    };
    spyOn(dataSource, 'getState').and.returnValue({
      selection: new Map(),
      colorOverrides: new Map(),
    });
    const setStateSpy = spyOn(dataSource, 'setState').and.stub();
    store.overrideSelector(getDashboardRuns, [currentRun]);
    store.overrideSelector(getRunSelectionMap, new Map([['run1', false]]));
    store.overrideSelector(getRunColorOverride, new Map([['run1', '#abc']]));
    store.refreshState();

    effects.syncRunsToLocalStorage$.subscribe();
    actions.next(runsActions.runSelectionToggled({runId: 'run1'}));

    expect(setStateSpy).toHaveBeenCalledOnceWith(
      jasmine.stringMatching('/tmp/tensorboard/runs'),
      [currentRun],
      {
        selection: new Map([['run1', false]]),
        colorOverrides: new Map([['run1', '#abc']]),
        sortingInfo: {name: 'run', order: SortingOrder.ASCENDING},
      }
    );
  });

  it('persists a group-by color reset before later hydration', () => {
    const run = createRun('run1', 1);
    const currentRun = {
      ...run,
      hparams: null,
      metrics: null,
      experimentId: 'exp1',
    };
    const namespace = TEST_ONLY.getNamespace('/tmp/tensorboard/runs', [
      'exp1',
    ])!;
    dataSource.setState(namespace, [currentRun], {
      selection: new Map(),
      colorOverrides: new Map([['run1', '#123456']]),
    });
    store.overrideSelector(getDashboardRuns, [currentRun]);
    store.overrideSelector(getRunColorOverride, new Map<string, string>());
    store.refreshState();

    effects.syncRunsToLocalStorage$.subscribe();
    effects.hydrateExistingRunsFromLocalStorage$.subscribe();
    actions.next(
      runsActions.runGroupByChanged({
        experimentIds: ['exp1'],
        groupBy: {key: GroupByKey.RUN},
      })
    );

    expect(dataSource.getState(namespace, [currentRun]).colorOverrides).toEqual(
      new Map()
    );

    dispatchedActions = [];
    actions.next(
      coreActions.environmentLoaded({
        environment: {
          data_location: '/tmp/tensorboard/runs',
          window_title: '',
        },
      })
    );

    expect(dispatchedActions).toContain(
      runsActions.runLocalStorageHydrated({
        runIds: ['run1'],
        selection: {},
        colorOverrides: {run1: '#fff'},
        restoredSelection: false,
      })
    );
  });

  it('persists the run sorting selection when it changes', () => {
    const run = createRun('run1', 1);
    const currentRun = {
      ...run,
      hparams: null,
      metrics: null,
      experimentId: 'exp1',
    };
    spyOn(dataSource, 'getState').and.returnValue({
      selection: new Map(),
      colorOverrides: new Map(),
    });
    const setStateSpy = spyOn(dataSource, 'setState').and.stub();
    store.overrideSelector(getDashboardRuns, [currentRun]);
    store.overrideSelector(getRunsTableSortingInfo, {
      name: '\0runStartTime',
      order: SortingOrder.DESCENDING,
    });
    store.refreshState();

    effects.syncRunsToLocalStorage$.subscribe();
    actions.next(
      runsActions.runsTableSortingInfoChanged({
        sortingInfo: {name: '\0runStartTime', order: SortingOrder.DESCENDING},
      })
    );

    expect(setStateSpy).toHaveBeenCalledOnceWith(
      jasmine.stringMatching('/tmp/tensorboard/runs'),
      [currentRun],
      {
        selection: new Map(),
        colorOverrides: new Map(),
        sortingInfo: {name: '\0runStartTime', order: SortingOrder.DESCENDING},
      }
    );
  });

  it('restores the persisted run sorting selection on hydration', () => {
    const run = createRun('run1', 1);
    spyOn(dataSource, 'getState').and.returnValue({
      selection: new Map([['run1', true]]),
      colorOverrides: new Map([['run1', '#123456']]),
      sortingInfo: {name: '\0runStartTime', order: SortingOrder.DESCENDING},
    });
    const setStateSpy = spyOn(dataSource, 'setState').and.stub();
    store.overrideSelector(getDashboardRuns, [
      {...run, hparams: null, metrics: null, experimentId: 'exp1'},
    ]);
    store.overrideSelector(getRunSelectionMap, new Map([['run1', false]]));
    store.refreshState();

    effects.hydrateExistingRunsFromLocalStorage$.subscribe();
    actions.next(
      coreActions.environmentLoaded({
        environment: {
          data_location: '/tmp/tensorboard/runs',
          window_title: '',
        },
      })
    );

    expect(dispatchedActions).toContain(
      runsActions.runsTableSortingInfoChanged({
        sortingInfo: {name: '\0runStartTime', order: SortingOrder.DESCENDING},
      })
    );
    expect(setStateSpy).toHaveBeenCalledOnceWith(
      jasmine.stringMatching('/tmp/tensorboard/runs'),
      [jasmine.objectContaining({id: 'run1'})],
      {
        selection: new Map([['run1', true]]),
        colorOverrides: new Map([['run1', '#123456']]),
        sortingInfo: {name: '\0runStartTime', order: SortingOrder.DESCENDING},
      }
    );
  });
});
