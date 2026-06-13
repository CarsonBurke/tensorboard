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
} from '../../selectors';
import * as runsActions from '../actions';
import {Run} from '../types';
import {provideMockTbStore} from '../../testing/utils';
import {RunLocalStorageDataSource} from './run_local_storage_data_source';
import {RunsLocalStorageEffects} from './runs_local_storage_effects';

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
  });

  afterEach(() => {
    store?.resetSelectors();
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
      }
    );
  });
});
