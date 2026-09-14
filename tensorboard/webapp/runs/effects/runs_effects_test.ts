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
import {TestBed} from '@angular/core/testing';
import {Actions} from '@ngrx/effects';
import {provideMockActions} from '@ngrx/effects/testing';
import {Action, Store} from '@ngrx/store';
import {MockStore} from '@ngrx/store/testing';
import {of, ReplaySubject} from 'rxjs';
import {
  buildCompareRoute,
  buildExperimentRouteFromId,
  buildNavigatedAction,
  buildRoute,
} from '../../app_routing/testing';
import {RouteKind} from '../../app_routing/types';
import {State} from '../../app_state';
import * as coreActions from '../../core/actions';
import * as hparamsActions from '../../hparams/_redux/hparams_actions';
import {stateRehydratedFromUrl} from '../../app_routing/actions';
import {
  getDashboardHparamFilterMap,
  getDashboardSessionGroups,
} from '../../hparams/_redux/hparams_selectors';
import {buildSessionGroup} from '../../hparams/_redux/testing';
import {
  getActiveRoute,
  getDashboardExperimentNames,
  getExperimentIdsFromRoute,
  getExperimentIdToExperimentAliasMap,
  getRuns,
  getRunsLoadState,
  getRunCatalog,
  getRunCatalogWindow,
  getRunSelectorRegexFilter,
  getRunsTableSortingInfo,
} from '../../selectors';
import {provideMockTbStore} from '../../testing/utils';
import {DataLoadState} from '../../types/data';
import * as actions from '../actions';
import {Run, RunPageRequest} from '../data_source/runs_data_source_types';
import {
  provideTestingRunsDataSource,
  TestingRunsDataSource,
} from '../data_source/testing';
import {RunsEffects} from './index';
import {ColumnHeaderType, SortingOrder} from '../../widgets/data_table/types';
import {RUN_START_TIME_SORT_KEY} from '../views/runs_table/sorting_utils';
import {DomainType} from '../../widgets/data_table/types';

function createRun(override: Partial<Run> = {}) {
  return {
    id: '123',
    name: 'foo',
    startTime: 0,
    ...override,
  };
}

describe('runs_effects', () => {
  let runsDataSource: TestingRunsDataSource;
  let effects: RunsEffects;
  let store: MockStore<State>;
  let action: ReplaySubject<Action>;
  let fetchRunsSubjects: Array<ReplaySubject<Run[]>>;
  let actualActions: Action[];
  let selectSpy: jasmine.Spy;

  function flushFetchRuns(requestIndex: number, runs: Run[]) {
    expect(fetchRunsSubjects.length).toBeGreaterThan(requestIndex);
    fetchRunsSubjects[requestIndex].next(runs);
    fetchRunsSubjects[requestIndex].complete();
  }

  function flushRunsError(requestIndex: number) {
    expect(fetchRunsSubjects.length).toBeGreaterThan(requestIndex);
    fetchRunsSubjects[requestIndex].error(new ErrorEvent('error'));
    fetchRunsSubjects[requestIndex].complete();
  }

  beforeEach(async () => {
    action = new ReplaySubject<Action>(1);

    await TestBed.configureTestingModule({
      providers: [
        provideMockActions(action),
        RunsEffects,
        provideMockTbStore(),
        provideTestingRunsDataSource(),
      ],
    }).compileComponents();

    store = TestBed.inject<Store<State>>(Store) as MockStore<State>;
    selectSpy = spyOn(store, 'select').and.callThrough();

    actualActions = [];
    // Cast to jasmine.Spy for compatibility between NgRx dispatch signature overloads.
    (spyOn(store, 'dispatch') as jasmine.Spy).and.callFake((action: Action) => {
      actualActions.push(action);
    });
    effects = TestBed.inject(RunsEffects);
    runsDataSource = TestBed.inject(TestingRunsDataSource);
    fetchRunsSubjects = [];
    spyOn(runsDataSource, 'fetchRuns').and.callFake(() => {
      const subject = new ReplaySubject<Run[]>(1);
      fetchRunsSubjects.push(subject);
      return subject;
    });

    store.overrideSelector(getRunsLoadState, {
      state: DataLoadState.NOT_LOADED,
      lastLoadedTimeInMs: 0,
    });
    store.overrideSelector(getExperimentIdsFromRoute, null);
    store.overrideSelector(getActiveRoute, buildRoute());
  });

  afterEach(() => {
    store?.resetSelectors();
  });

  describe('loadRunsOnNavigationOrReload', () => {
    beforeEach(() => {
      effects.loadRunsOnNavigationOrReload$.subscribe(() => {});
    });

    [
      {specAction: buildNavigatedAction, specName: 'navigation'},
      {specAction: coreActions.manualReload, specName: 'manual reload'},
      {specAction: coreActions.reload, specName: 'auto reload'},
    ].forEach(({specAction, specName}) => {
      describe(`on ${specName}`, () => {
        it(`fetches runs based on expIds in the route`, () => {
          store.overrideSelector(
            getActiveRoute,
            buildCompareRoute(['exp1:123', 'exp2:456'])
          );
          store.overrideSelector(getExperimentIdsFromRoute, ['123', '456']);
          store.overrideSelector(getDashboardExperimentNames, {
            456: 'exp2',
            123: 'exp1',
          });
          const createFooRuns = () => [
            createRun({
              id: 'foo/runA',
              name: 'runA',
            }),
          ];
          const createBarRuns = () => [
            createRun({
              id: 'bar/runB',
              name: 'runB',
            }),
          ];
          store.refreshState();

          action.next(specAction());
          // Flush second request first to spice things up.
          flushFetchRuns(1, createBarRuns());
          flushFetchRuns(0, createFooRuns());

          expect(actualActions).toEqual([
            actions.fetchRunsRequested({
              experimentIds: ['123', '456'],
              requestedExperimentIds: ['123', '456'],
            }),
            actions.fetchRunsSucceeded({
              experimentIds: ['123', '456'],
              runsForAllExperiments: [...createFooRuns(), ...createBarRuns()],
              newRuns: {
                456: {
                  runs: createBarRuns(),
                },
                123: {
                  runs: createFooRuns(),
                },
              },
              expNameByExpId: {
                456: 'exp2',
                123: 'exp1',
              },
            }),
          ]);
        });

        it('fetches only runs that are not loading', () => {
          const createFooRuns = () => [
            createRun({
              id: 'foo/runA',
              name: 'runA',
            }),
          ];
          const createBarRuns = () => [
            createRun({
              id: 'bar/runB',
              name: 'runB',
            }),
          ];

          const get123LoadState = new ReplaySubject(1);
          get123LoadState.next({
            state: DataLoadState.LOADING,
            lastLoadedTimeInMs: 0,
          });
          selectSpy
            .withArgs(getRuns, {experimentId: '123'})
            .and.returnValue(of(createFooRuns()));
          selectSpy
            .withArgs(getRuns, {experimentId: '456'})
            .and.returnValue(of(createBarRuns()));
          selectSpy
            .withArgs(getRunsLoadState, {experimentId: '123'})
            .and.returnValue(get123LoadState);
          selectSpy
            .withArgs(getRunsLoadState, {experimentId: '456'})
            .and.returnValue(
              of({
                state: DataLoadState.NOT_LOADED,
                lastLoadedTimeInMs: null,
              })
            );
          store.overrideSelector(
            getActiveRoute,
            buildCompareRoute(['exp1:123', ' exp2:456'])
          );
          store.overrideSelector(getExperimentIdsFromRoute, ['123', '456']);
          store.overrideSelector(getDashboardExperimentNames, {
            456: 'exp2',
            123: 'exp1',
          });
          store.refreshState();

          action.next(specAction());
          flushFetchRuns(0, createBarRuns());

          expect(actualActions).toEqual([
            actions.fetchRunsRequested({
              experimentIds: ['123', '456'],
              requestedExperimentIds: ['456'],
            }),
          ]);

          // Since the stream is waiting until the loading runs are
          // resolved, we need to change the load state in order to get the
          // `fetchRunsSucceeded`.
          get123LoadState.next({
            state: DataLoadState.LOADED,
            lastLoadedTimeInMs: 123,
          });

          expect(actualActions).toEqual([
            actions.fetchRunsRequested({
              experimentIds: ['123', '456'],
              requestedExperimentIds: ['456'],
            }),
            actions.fetchRunsSucceeded({
              experimentIds: ['123', '456'],
              runsForAllExperiments: [...createFooRuns(), ...createBarRuns()],
              newRuns: {
                456: {
                  runs: createBarRuns(),
                },
              },
              expNameByExpId: {
                456: 'exp2',
                123: 'exp1',
              },
            }),
          ]);
        });
      });
    });

    [
      {specAction: buildNavigatedAction, specName: 'navigation'},
      {specAction: coreActions.manualReload, specName: 'manual reload'},
      {specAction: coreActions.reload, specName: 'auto reload'},
    ].forEach(({specAction, specName}) => {
      it(`does not fetch runs on card route when action is ${specName}`, () => {
        store.overrideSelector(getActiveRoute, {
          routeKind: RouteKind.CARD,
          params: {},
        });
        store.refreshState();

        action.next(specAction());

        expect(actualActions).toEqual([]);
      });
    });

    describe('on navigation', () => {
      it('fetches for runs if not loaded before', () => {
        const createFooRuns = () => [
          createRun({
            id: 'foo/runA',
            name: 'runA',
          }),
        ];
        const createBarRuns = () => [
          createRun({
            id: 'bar/runB',
            name: 'runB',
          }),
        ];

        selectSpy
          .withArgs(getRuns, {experimentId: '123'})
          .and.returnValue(of(createFooRuns()));
        selectSpy
          .withArgs(getRuns, {experimentId: '456'})
          .and.returnValue(of(createBarRuns()));
        selectSpy
          .withArgs(getRunsLoadState, {experimentId: '123'})
          .and.returnValue(
            of({
              state: DataLoadState.LOADED,
              lastLoadedTimeInMs: 0,
            })
          );
        selectSpy
          .withArgs(getRunsLoadState, {experimentId: '456'})
          .and.returnValue(
            of({
              state: DataLoadState.NOT_LOADED,
              lastLoadedTimeInMs: null,
            })
          );
        store.overrideSelector(
          getActiveRoute,
          buildCompareRoute(['exp1:123', ' exp2:456'])
        );
        store.overrideSelector(getExperimentIdsFromRoute, ['123', '456']);
        store.overrideSelector(getDashboardExperimentNames, {
          456: 'exp1',
          123: 'exp2',
        });
        store.refreshState();

        action.next(buildNavigatedAction());
        flushFetchRuns(0, createBarRuns());

        expect(actualActions).toEqual([
          actions.fetchRunsRequested({
            experimentIds: ['123', '456'],
            requestedExperimentIds: ['456'],
          }),
          actions.fetchRunsSucceeded({
            experimentIds: ['123', '456'],
            runsForAllExperiments: [...createFooRuns(), ...createBarRuns()],
            newRuns: {
              456: {
                runs: createBarRuns(),
              },
            },
            expNameByExpId: {
              456: 'exp1',
              123: 'exp2',
            },
          }),
        ]);
      });

      it('ignores a navigation to same route and experiments (hash changes)', () => {
        store.overrideSelector(getActiveRoute, buildRoute());
        store.overrideSelector(getExperimentIdsFromRoute, ['123']);
        store.overrideSelector(getDashboardExperimentNames, {123: 'exp1'});
        const createFooRuns = () => [
          createRun({
            id: 'foo/runA',
            name: 'runA',
          }),
        ];

        selectSpy
          .withArgs(getRuns, {experimentId: '123'})
          .and.returnValue(of(createFooRuns()));
        store.overrideSelector(getRunsLoadState, {
          state: DataLoadState.LOADED,
          lastLoadedTimeInMs: 0,
        });
        store.refreshState();

        // Only the first one goes through.
        action.next(buildNavigatedAction());

        expect(actualActions).toEqual([
          actions.fetchRunsRequested({
            experimentIds: ['123'],
            requestedExperimentIds: [],
          }),
          actions.fetchRunsSucceeded({
            experimentIds: ['123'],
            runsForAllExperiments: [...createFooRuns()],
            newRuns: {},
            expNameByExpId: {123: 'exp1'},
          }),
        ]);

        action.next(buildNavigatedAction());
        action.next(buildNavigatedAction());
        expect(actualActions.length).toBe(2);
      });

      it('dispatches fetchRunsSucceeded even if data is already loaded', () => {
        const createFooRuns = () => [
          createRun({
            id: 'foo/runA',
            name: 'runA',
          }),
        ];

        selectSpy
          .withArgs(getRuns, {experimentId: 'foo'})
          .and.returnValue(of(createFooRuns()));
        selectSpy
          .withArgs(getRunsLoadState, {experimentId: 'foo'})
          .and.returnValue(
            of({
              state: DataLoadState.LOADED,
              lastLoadedTimeInMs: 0,
            })
          );
        store.overrideSelector(getExperimentIdsFromRoute, ['foo']);
        store.overrideSelector(getDashboardExperimentNames, {foo: 'exp1'});
        store.refreshState();

        action.next(buildNavigatedAction());

        expect(actualActions).toEqual([
          actions.fetchRunsRequested({
            experimentIds: ['foo'],
            requestedExperimentIds: [],
          }),
          actions.fetchRunsSucceeded({
            experimentIds: ['foo'],
            runsForAllExperiments: [...createFooRuns()],
            newRuns: {},
            expNameByExpId: {foo: 'exp1'},
          }),
        ]);
      });
    });

    it('does not hang because one run failed to fetch', () => {
      store.overrideSelector(
        getActiveRoute,
        buildCompareRoute(['exp1:123', 'exp2:456'])
      );
      store.overrideSelector(getExperimentIdsFromRoute, ['123', '456']);
      store.refreshState();

      action.next(buildNavigatedAction());

      flushRunsError(0);
      flushFetchRuns(1, [createRun({id: 'bar/runB', name: 'runB'})]);

      expect(actualActions).toEqual([
        actions.fetchRunsRequested({
          experimentIds: ['123', '456'],
          requestedExperimentIds: ['123', '456'],
        }),
        actions.fetchRunsFailed({
          experimentIds: ['123', '456'],
          requestedExperimentIds: ['123', '456'],
        }),
      ]);
    });

    it('does not cancel request even if user navigates away', () => {
      store.overrideSelector(getActiveRoute, buildExperimentRouteFromId('123'));
      store.overrideSelector(getExperimentIdsFromRoute, ['123']);

      const createFooRuns = () => [
        createRun({
          id: 'foo/runA',
          name: 'runA',
        }),
      ];
      const createBarRuns = () => [
        createRun({
          id: 'bar/runB',
          name: 'runB',
        }),
      ];
      store.refreshState();

      action.next(buildNavigatedAction());

      // Emulate navigation to a new experiment route.
      store.overrideSelector(getActiveRoute, buildExperimentRouteFromId('456'));
      store.overrideSelector(getExperimentIdsFromRoute, ['456']);
      store.overrideSelector(getDashboardExperimentNames, {
        456: 'exp1',
        123: 'exp2',
      });
      // Force selectors to re-evaluate with a change in store.
      store.refreshState();

      action.next(buildNavigatedAction());

      flushFetchRuns(1, createBarRuns());
      flushFetchRuns(0, createFooRuns());

      expect(actualActions).toEqual([
        actions.fetchRunsRequested({
          experimentIds: ['123'],
          requestedExperimentIds: ['123'],
        }),
        actions.fetchRunsRequested({
          experimentIds: ['456'],
          requestedExperimentIds: ['456'],
        }),
        actions.fetchRunsSucceeded({
          experimentIds: ['456'],
          runsForAllExperiments: createBarRuns(),
          newRuns: {
            456: {runs: createBarRuns()},
          },
          expNameByExpId: {456: 'exp1', 123: 'exp2'},
        }),
        actions.fetchRunsSucceeded({
          experimentIds: ['123'],
          runsForAllExperiments: createFooRuns(),
          newRuns: {
            123: {runs: createFooRuns()},
          },
          expNameByExpId: {456: 'exp1', 123: 'exp2'},
        }),
      ]);
    });

    it('fires FAILED action when at least one runs fetch failed', () => {
      store.overrideSelector(
        getActiveRoute,
        buildCompareRoute(['exp1:123', 'exp2:456'])
      );
      store.overrideSelector(getExperimentIdsFromRoute, ['123', '456']);
      store.refreshState();

      action.next(buildNavigatedAction());

      flushRunsError(0);
      flushFetchRuns(1, []);

      expect(actualActions).toEqual([
        actions.fetchRunsRequested({
          experimentIds: ['123', '456'],
          requestedExperimentIds: ['123', '456'],
        }),
        actions.fetchRunsFailed({
          experimentIds: ['123', '456'],
          requestedExperimentIds: ['123', '456'],
        }),
      ]);
    });

    describe('multiple actions', () => {
      it('waits for already loading runs so actions do not fire out of order', () => {
        // When actions are fired out of order, then the list of runs can be
        // stale and lead to incorrect run selection.
        const createFooBeforeRuns = () => [
          createRun({
            id: 'foo/runA',
            name: 'runA',
          }),
        ];
        const createFooAfterRuns = () => [
          createRun({
            id: 'foo/runA',
            name: 'runA',
          }),
          createRun({
            id: 'foo/runB',
            name: 'runB',
          }),
        ];
        const createBarRuns = () => [
          createRun({
            id: 'bar/runB',
            name: 'runB',
          }),
        ];

        const runsSubject = new ReplaySubject(1);
        runsSubject.next(createFooBeforeRuns());
        const runsLoadStateSubject = new ReplaySubject(1);
        runsLoadStateSubject.next({
          state: DataLoadState.NOT_LOADED,
          lastLoadedTimeInMs: 0,
        });

        store.overrideSelector(getExperimentIdsFromRoute, ['foo']);
        store.overrideSelector(getDashboardExperimentNames, {
          foo: 'exp1',
          bar: 'exp2',
        });
        selectSpy
          .withArgs(getRuns, {experimentId: 'foo'})
          .and.returnValue(runsSubject);
        selectSpy
          .withArgs(getRunsLoadState, {experimentId: 'foo'})
          .and.returnValue(runsLoadStateSubject);
        selectSpy
          .withArgs(getRuns, {experimentId: 'bar'})
          .and.returnValue(of(null));
        selectSpy
          .withArgs(getRunsLoadState, {experimentId: 'bar'})
          .and.returnValue(
            of({
              state: DataLoadState.NOT_LOADED,
              lastLoadedTimeInMs: 0,
            })
          );
        store.refreshState();

        // User triggered reload on `/experiment/foo/`
        action.next(coreActions.manualReload());

        // User navigates to `/compare/a:foo,b:bar/`
        store.overrideSelector(getExperimentIdsFromRoute, ['foo', 'bar']);
        runsLoadStateSubject.next({
          state: DataLoadState.LOADING,
          lastLoadedTimeInMs: 0,
        });
        store.refreshState();
        action.next(buildNavigatedAction());

        // Flush the request for `bar`'s runs.
        flushFetchRuns(1, createBarRuns());

        // Flush the request for `foo`'s runs.
        flushFetchRuns(0, createFooAfterRuns());
        runsSubject.next(createFooAfterRuns());
        runsLoadStateSubject.next({
          state: DataLoadState.LOADED,
          lastLoadedTimeInMs: 123,
        });

        expect(actualActions).toEqual([
          actions.fetchRunsRequested({
            experimentIds: ['foo'],
            requestedExperimentIds: ['foo'],
          }),
          actions.fetchRunsRequested({
            experimentIds: ['foo', 'bar'],
            requestedExperimentIds: ['bar'],
          }),
          actions.fetchRunsSucceeded({
            experimentIds: ['foo'],
            runsForAllExperiments: [...createFooAfterRuns()],
            newRuns: {
              foo: {
                runs: createFooAfterRuns(),
              },
            },
            expNameByExpId: {foo: 'exp1', bar: 'exp2'},
          }),
          actions.fetchRunsSucceeded({
            experimentIds: ['foo', 'bar'],
            runsForAllExperiments: [
              ...createFooAfterRuns(),
              ...createBarRuns(),
            ],
            newRuns: {
              bar: {
                runs: createBarRuns(),
              },
            },
            expNameByExpId: {foo: 'exp1', bar: 'exp2'},
          }),
        ]);
      });

      it('dispatches action when an already loading run fails to load', () => {
        const createFooRuns = () => [];

        const runsSubject = new ReplaySubject(1);
        runsSubject.next(createFooRuns());
        const runsLoadStateSubject = new ReplaySubject(1);
        runsLoadStateSubject.next({
          state: DataLoadState.NOT_LOADED,
          lastLoadedTimeInMs: 0,
        });

        store.overrideSelector(getExperimentIdsFromRoute, ['foo']);
        selectSpy
          .withArgs(getRuns, {experimentId: 'foo'})
          .and.returnValue(runsSubject);
        selectSpy
          .withArgs(getRunsLoadState, {experimentId: 'foo'})
          .and.returnValue(runsLoadStateSubject);
        store.refreshState();

        action.next(coreActions.reload());

        runsLoadStateSubject.next({
          state: DataLoadState.LOADING,
          lastLoadedTimeInMs: 0,
        });
        store.refreshState();
        action.next(coreActions.manualReload());

        flushRunsError(0);
        runsLoadStateSubject.next({
          state: DataLoadState.FAILED,
          lastLoadedTimeInMs: 0,
        });

        expect(actualActions).toEqual([
          actions.fetchRunsRequested({
            experimentIds: ['foo'],
            requestedExperimentIds: ['foo'],
          }),
          actions.fetchRunsRequested({
            experimentIds: ['foo'],
            requestedExperimentIds: [],
          }),
          actions.fetchRunsFailed({
            experimentIds: ['foo'],
            requestedExperimentIds: ['foo'],
          }),
          actions.fetchRunsFailed({
            experimentIds: ['foo'],
            requestedExperimentIds: [],
          }),
        ]);
      });
    });
  });

  it('discards obsolete native catalog responses after navigation', () => {
    const previous = new ReplaySubject<{runs: Run[]; total: number}>(1);
    const current = new ReplaySubject<{runs: Run[]; total: number}>(1);
    Object.assign(runsDataSource, {
      fetchRunsPage: jasmine
        .createSpy()
        .and.callFake((experimentId: string) =>
          experimentId === '123' ? previous : current
        ),
    });
    effects = new RunsEffects(TestBed.inject(Actions), store, runsDataSource);
    const subscription = effects.loadRunsOnNavigationOrReload$.subscribe();
    store.overrideSelector(getActiveRoute, buildExperimentRouteFromId('123'));
    store.overrideSelector(getExperimentIdsFromRoute, ['123']);
    store.overrideSelector(getDashboardExperimentNames, {});
    store.refreshState();
    action.next(buildNavigatedAction());

    store.overrideSelector(getActiveRoute, buildExperimentRouteFromId('456'));
    store.overrideSelector(getExperimentIdsFromRoute, ['456']);
    store.refreshState();
    action.next(buildNavigatedAction());

    const run = createRun({id: '456/current', name: 'current'});
    current.next({runs: [run], total: 1_000_000});
    current.complete();
    previous.next({runs: [createRun({id: '123/obsolete'})], total: 1});
    previous.complete();

    expect(
      actualActions.filter(
        (action) => action.type === actions.fetchRunsSucceeded.type
      )
    ).toEqual([
      actions.fetchRunsSucceeded({
        experimentIds: ['456'],
        runsForAllExperiments: [run],
        newRuns: {456: {runs: [run]}},
        expNameByExpId: {},
        catalog: {runIds: [run.id], totals: {456: 1_000_000}, offset: 0},
      }),
    ]);
    subscription.unsubscribe();
  });

  it('cancels native requests when query, sorting, window, or restored URL changes', () => {
    const responses = Array.from(
      {length: 5},
      () => new ReplaySubject<{runs: Run[]; total: number}>(1)
    );
    let next = 0;
    Object.assign(runsDataSource, {
      fetchRunsPage: () => responses[next++],
    });
    effects = new RunsEffects(TestBed.inject(Actions), store, runsDataSource);
    const subscription = effects.loadRunsOnNavigationOrReload$.subscribe();
    store.overrideSelector(getActiveRoute, buildExperimentRouteFromId('123'));
    store.overrideSelector(getExperimentIdsFromRoute, ['123']);
    store.overrideSelector(getDashboardExperimentNames, {});
    store.refreshState();
    action.next(buildNavigatedAction());
    action.next(actions.runSelectorRegexFilterChanged({regexString: 'loss'}));
    action.next(
      actions.runsTableSortingInfoChanged({
        sortingInfo: {name: 'run', order: SortingOrder.ASCENDING},
      })
    );
    action.next(actions.runCatalogWindowChanged({offset: 100, limit: 100}));
    store.overrideSelector(getRunSelectorRegexFilter, 'restored');
    store.refreshState();
    action.next({type: stateRehydratedFromUrl.type});
    for (const response of responses.slice(0, 4)) {
      expect(response.observers.length).toBe(0);
      response.next({runs: [createRun({id: '123/obsolete'})], total: 1000});
      response.complete();
    }
    const current = createRun({id: '123/current'});
    responses[4].next({runs: [current], total: 1000});
    responses[4].complete();
    const succeeded = actualActions.filter(
      (value) => value.type === actions.fetchRunsSucceeded.type
    );
    expect(succeeded).toEqual([
      jasmine.objectContaining({
        catalog: jasmine.objectContaining({runIds: [current.id]}),
      }),
    ]);
    subscription.unsubscribe();
  });

  for (const sortBy of ['name', 'start_time'] as const) {
    it(`seeks a globally merged ${sortBy} window across experiments`, () => {
      const runs = {
        a: [
          createRun({id: 'a/alpha', name: 'alpha', startTime: 50}),
          createRun({id: 'a/charlie', name: 'charlie', startTime: 20}),
          createRun({id: 'a/echo', name: 'echo', startTime: 10}),
        ],
        b: [
          createRun({id: 'b/bravo', name: 'bravo', startTime: 40}),
          createRun({id: 'b/charlie', name: 'charlie', startTime: 20}),
          createRun({id: 'b/delta', name: 'delta', startTime: 30}),
        ],
      };
      Object.assign(runsDataSource, {
        fetchRunsPage: (id: keyof typeof runs, request: RunPageRequest) => {
          expect(request.limit).toBeGreaterThan(0);
          expect(request.limit).toBeLessThanOrEqual(2);
          const sorted = [...runs[id]].sort(
            (a, b) =>
              (sortBy === 'start_time' ? a.startTime! - b.startTime! : 0) ||
              (a.name < b.name ? -1 : a.name === b.name ? 0 : 1)
          );
          return of({
            runs: sorted.slice(request.offset, request.offset + request.limit),
            total: sorted.length,
          });
        },
      });
      effects = new RunsEffects(TestBed.inject(Actions), store, runsDataSource);
      const subscription = effects.loadRunsOnNavigationOrReload$.subscribe();
      store.overrideSelector(getActiveRoute, buildExperimentRouteFromId('a'));
      store.overrideSelector(getExperimentIdsFromRoute, ['a', 'b']);
      store.overrideSelector(getDashboardExperimentNames, {});
      store.overrideSelector(getRunSelectorRegexFilter, '');
      store.overrideSelector(getRunCatalogWindow, {offset: 2, limit: 2});
      store.overrideSelector(getRunsTableSortingInfo, {
        name: sortBy === 'name' ? 'run' : RUN_START_TIME_SORT_KEY,
        order: SortingOrder.ASCENDING,
      });
      store.refreshState();
      action.next(actions.runCatalogWindowChanged({offset: 2, limit: 2}));
      const succeeded = actualActions.find(
        (value) => value.type === actions.fetchRunsSucceeded.type
      );
      expect(succeeded).toEqual(
        jasmine.objectContaining({
          catalog: jasmine.objectContaining({
            runIds:
              sortBy === 'name'
                ? ['a/charlie', 'b/charlie']
                : ['b/charlie', 'b/delta'],
          }),
        })
      );
      subscription.unsubscribe();
    });
  }
  it('resolves every run in scope for select-all in paged mode', () => {
    const runs = {
      a: [
        createRun({id: 'a/alpha', name: 'alpha'}),
        createRun({id: 'a/beta', name: 'beta'}),
        createRun({id: 'a/gamma', name: 'gamma'}),
      ],
      b: [
        createRun({id: 'b/delta', name: 'delta'}),
        createRun({id: 'b/epsilon', name: 'epsilon'}),
      ],
    };
    const seenRequests: Array<{id: string; request: RunPageRequest}> = [];
    Object.assign(runsDataSource, {
      fetchRunsPage: (id: keyof typeof runs, request: RunPageRequest) => {
        seenRequests.push({id, request});
        const all = [...runs[id]].sort((x, y) => (x.name < y.name ? -1 : 1));
        return of({
          runs: all.slice(request.offset, request.offset + request.limit),
          total: all.length,
        });
      },
    });
    effects = new RunsEffects(TestBed.inject(Actions), store, runsDataSource);
    const emitted: Action[] = [];
    const subscription = effects.selectAllRuns$.subscribe((action) =>
      emitted.push(action)
    );
    store.overrideSelector(getActiveRoute, buildExperimentRouteFromId('a'));
    store.overrideSelector(getExperimentIdsFromRoute, ['a', 'b']);
    store.overrideSelector(getRunCatalog, {
      runIds: ['a/alpha'],
      totals: {a: 3, b: 2},
      offset: 0,
    });
    store.overrideSelector(getRunSelectorRegexFilter, '');
    store.overrideSelector(getRunsTableSortingInfo, {
      name: 'run',
      order: SortingOrder.ASCENDING,
    });
    store.refreshState();
    action.next(actions.selectAllRuns());

    expect(emitted).toEqual([
      actions.runPageSelectionToggled({
        runIds: ['a/alpha', 'a/beta', 'a/gamma', 'b/delta', 'b/epsilon'],
      }),
    ]);
    // Full-scope fetch per experiment, not the current window.
    expect(
      seenRequests.map(({id, request}) => [id, request.offset, request.limit])
    ).toEqual([
      ['a', 0, 3],
      ['b', 0, 2],
    ]);
    subscription.unsubscribe();
  });

  for (const newerIntent of [
    actions.singleRunSelected({runId: 'a/alpha'}),
    actions.runSelectorRegexFilterChanged({regexString: 'alpha'}),
  ]) {
    it(`discards a pending select-all after ${newerIntent.type}`, () => {
      const pending = new ReplaySubject<{runs: Run[]; total: number}>(1);
      const runs = [
        createRun({id: 'a/alpha', name: 'alpha'}),
        createRun({id: 'a/beta', name: 'beta'}),
      ];
      Object.assign(runsDataSource, {fetchRunsPage: () => pending});
      effects = new RunsEffects(TestBed.inject(Actions), store, runsDataSource);
      store.overrideSelector(getActiveRoute, buildExperimentRouteFromId('a'));
      store.overrideSelector(getExperimentIdsFromRoute, ['a']);
      store.overrideSelector(getRunCatalog, {
        runIds: ['a/alpha'],
        totals: {a: 2},
        offset: 0,
      });
      store.refreshState();
      const emitted: Action[] = [];
      const subscription = effects.selectAllRuns$.subscribe((value) =>
        emitted.push(value)
      );

      action.next(actions.selectAllRuns());
      action.next(newerIntent);
      pending.next({runs, total: 2});
      pending.complete();
      expect(emitted).toEqual([]);

      // Cancellation must not disable a subsequent deliberate select-all.
      action.next(actions.selectAllRuns());
      expect(emitted).toEqual([
        actions.runPageSelectionToggled({runIds: ['a/alpha', 'a/beta']}),
      ]);
      subscription.unsubscribe();
    });
  }

  it('does nothing for select-all without a catalog', () => {
    Object.assign(runsDataSource, {
      fetchRunsPage: () => {
        throw new Error('should not fetch without a catalog');
      },
    });
    effects = new RunsEffects(TestBed.inject(Actions), store, runsDataSource);
    const emitted: Action[] = [];
    const subscription = effects.selectAllRuns$.subscribe((action) =>
      emitted.push(action)
    );
    store.refreshState();
    action.next(actions.selectAllRuns());

    expect(emitted).toEqual([]);
    subscription.unsubscribe();
  });

  it('filters and ranks hparam subruns before a compact catalog window', () => {
    const names = ['p/train', 'p/child/eval', 'q/eval', 'r/test', 'unknown'];
    Object.assign(runsDataSource, {
      fetchRunsPage: (_id: string, request: RunPageRequest) => {
        const prefixes = [...(request.sessionRanks ?? [])].sort(
          (a, b) => b.prefix.length - a.prefix.length
        );
        const rows = names
          .map((name) => ({
            run: createRun({id: `exp/${name}`, name}),
            rank:
              prefixes.find(({prefix}) => name.startsWith(prefix))?.rank ??
              request.defaultRank ??
              0,
          }))
          .filter(({rank}) => rank >= 0)
          .sort((a, b) => a.rank - b.rank);
        return of({
          runs: rows
            .slice(request.offset, request.offset + request.limit)
            .map(({run}) => run),
          total: rows.length,
        });
      },
    });
    effects = new RunsEffects(TestBed.inject(Actions), store, runsDataSource);
    const subscription = effects.loadRunsOnNavigationOrReload$.subscribe();
    store.overrideSelector(getActiveRoute, buildExperimentRouteFromId('exp'));
    store.overrideSelector(getExperimentIdsFromRoute, ['exp']);
    store.overrideSelector(getDashboardExperimentNames, {});
    store.overrideSelector(getRunCatalogWindow, {offset: 1, limit: 1});
    store.overrideSelector(getRunsTableSortingInfo, {
      name: 'score',
      order: SortingOrder.ASCENDING,
    });
    store.overrideSelector(getDashboardSessionGroups, [
      buildSessionGroup({
        hparams: {score: 30, enabled: true},
        sessions: [{name: 'exp/p'}],
      }),
      buildSessionGroup({
        hparams: {score: 10, enabled: false},
        sessions: [{name: 'exp/p/child'}],
      }),
      buildSessionGroup({
        hparams: {score: 20, enabled: true},
        sessions: [{name: 'exp/q'}],
      }),
      buildSessionGroup({
        hparams: {score: 10, enabled: true},
        sessions: [{name: 'exp/r'}],
      }),
    ]);
    store.overrideSelector(
      getDashboardHparamFilterMap,
      new Map([
        [
          'enabled',
          {
            type: DomainType.DISCRETE,
            includeUndefined: false,
            possibleValues: [true, false],
            filterValues: [true],
          },
        ],
      ])
    );
    store.refreshState();
    action.next(actions.runCatalogWindowChanged({offset: 1, limit: 1}));
    expect(actualActions).toContain(
      jasmine.objectContaining({
        type: actions.fetchRunsSucceeded.type,
        catalog: {runIds: ['exp/q/eval'], totals: {exp: 3}, offset: 1},
      })
    );
    subscription.unsubscribe();
  });

  it('merges non-BMP run names in backend Unicode scalar order', () => {
    const names: Record<string, string[]> = {
      a: ['\ue000', '\u{10000}'],
      b: ['\ue001'],
    };
    Object.assign(runsDataSource, {
      fetchRunsPage: (id: string, request: RunPageRequest) =>
        of({
          runs: names[id]
            .slice(request.offset, request.offset + request.limit)
            .map((name) => createRun({id: `${id}/${name}`, name})),
          total: names[id].length,
        }),
    });
    effects = new RunsEffects(TestBed.inject(Actions), store, runsDataSource);
    const subscription = effects.loadRunsOnNavigationOrReload$.subscribe();
    store.overrideSelector(getActiveRoute, buildExperimentRouteFromId('a'));
    store.overrideSelector(getExperimentIdsFromRoute, ['a', 'b']);
    store.overrideSelector(getDashboardExperimentNames, {});
    store.overrideSelector(getRunCatalogWindow, {offset: 1, limit: 1});
    store.overrideSelector(getRunsTableSortingInfo, {
      name: 'run',
      order: SortingOrder.ASCENDING,
    });
    store.refreshState();
    action.next(actions.runCatalogWindowChanged({offset: 1, limit: 1}));
    expect(actualActions).toContain(
      jasmine.objectContaining({
        type: actions.fetchRunsSucceeded.type,
        catalog: {runIds: ['b/\ue001'], totals: {a: 2, b: 1}, offset: 1},
      })
    );
    subscription.unsubscribe();
  });

  it('orders comparison windows by experiment alias rather than start time', () => {
    const names: Record<string, string[]> = {
      a: ['alpha', 'bravo'],
      b: ['yankee', 'zulu'],
    };
    Object.assign(runsDataSource, {
      fetchRunsPage: (id: string, request: RunPageRequest) =>
        of({
          runs: names[id]
            .slice(request.offset, request.offset + request.limit)
            .map((name) => createRun({id: `${id}/${name}`, name})),
          total: names[id].length,
        }),
    });
    effects = new RunsEffects(TestBed.inject(Actions), store, runsDataSource);
    const subscription = effects.loadRunsOnNavigationOrReload$.subscribe();
    store.overrideSelector(getActiveRoute, buildExperimentRouteFromId('a'));
    store.overrideSelector(getExperimentIdsFromRoute, ['a', 'b']);
    store.overrideSelector(getDashboardExperimentNames, {});
    store.overrideSelector(getExperimentIdToExperimentAliasMap, {
      a: {aliasText: 'second', aliasNumber: 2},
      b: {aliasText: 'first', aliasNumber: 1},
    });
    store.overrideSelector(getRunCatalogWindow, {offset: 1, limit: 2});
    store.overrideSelector(getRunsTableSortingInfo, {
      name: 'experimentAlias',
      order: SortingOrder.ASCENDING,
    });
    store.refreshState();
    action.next(actions.runCatalogWindowChanged({offset: 1, limit: 2}));
    expect(actualActions).toContain(
      jasmine.objectContaining({
        type: actions.fetchRunsSucceeded.type,
        catalog: {
          runIds: ['b/zulu', 'a/alpha'],
          totals: {a: 2, b: 2},
          offset: 1,
        },
      })
    );
    subscription.unsubscribe();
  });

  describe('removeHparamFilterWhenColumnIsRemoved$', () => {
    beforeEach(() => {
      effects.removeHparamFilterWhenColumnIsRemoved$.subscribe(() => {});
    });

    it('dispatches dashboardHparamFilterRemoved when column type is hparam', () => {
      action.next(
        actions.runsTableHeaderRemoved({
          header: {
            type: ColumnHeaderType.HPARAM,
            name: 'some_hparam',
            enabled: true,
            displayName: 'Some Hparam',
          },
        })
      );
      store.refreshState();

      expect(actualActions).toEqual([
        hparamsActions.dashboardHparamFilterRemoved({
          name: 'some_hparam',
        }),
      ]);
    });

    it('dispatches dashboardMetricFilterRemoved when column type is metric', () => {
      action.next(
        actions.runsTableHeaderRemoved({
          header: {
            type: ColumnHeaderType.METRIC,
            name: 'some_metric',
            enabled: true,
            displayName: 'Some Metric',
          },
        })
      );
      store.refreshState();

      expect(actualActions).toEqual([
        hparamsActions.dashboardMetricFilterRemoved({
          name: 'some_metric',
        }),
      ]);
    });
  });
});
