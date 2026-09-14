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
  getCardStateMap,
  getEnvironment,
  getExperimentIdsFromRoute,
  getMetricsTagGroupExpandedMap,
  getMetricsTagGroupPageIndexMap,
  getNonEmptyCardIdsWithMetadata,
} from '../../selectors';
import {provideMockTbStore} from '../../testing/utils';
import * as metricsActions from '../actions';
import {PluginType} from '../data_source';
import {
  MetricsLocalStorageDataSource,
  TEST_ONLY as STORAGE_TEST_ONLY,
} from './metrics_local_storage_data_source';
import {
  MetricsLocalStorageEffects,
  TEST_ONLY,
} from './metrics_local_storage_effects';

describe('MetricsLocalStorageEffects', () => {
  let actions: ReplaySubject<Action>;
  let effects: MetricsLocalStorageEffects;
  let store: MockStore<State>;
  let dataSource: MetricsLocalStorageDataSource;
  let dispatchedActions: Action[];

  beforeEach(async () => {
    window.localStorage.removeItem(STORAGE_TEST_ONLY.METRICS_LOCAL_STORAGE_KEY);
    actions = new ReplaySubject<Action>(1);

    await TestBed.configureTestingModule({
      providers: [
        provideMockActions(actions),
        provideMockTbStore(),
        MetricsLocalStorageEffects,
        MetricsLocalStorageDataSource,
      ],
    }).compileComponents();

    store = TestBed.inject<Store<State>>(Store) as MockStore<State>;
    dataSource = TestBed.inject(MetricsLocalStorageDataSource);
    effects = TestBed.inject(MetricsLocalStorageEffects);
    dispatchedActions = [];
    (spyOn(store, 'dispatch') as jasmine.Spy).and.callFake((action: Action) => {
      dispatchedActions.push(action);
    });

    store.overrideSelector(getEnvironment, {
      data_location: '/tmp/tensorboard/runs',
      window_title: '',
    });
    store.overrideSelector(getExperimentIdsFromRoute, ['exp1']);
    store.overrideSelector(getNonEmptyCardIdsWithMetadata, [
      {
        cardId: 'card1',
        plugin: PluginType.SCALARS,
        tag: 'foo/accuracy',
        runId: null,
      },
      {
        cardId: 'card2',
        plugin: PluginType.SCALARS,
        tag: 'bar/loss',
        runId: null,
      },
    ]);
    store.overrideSelector(
      getMetricsTagGroupExpandedMap,
      new Map([['foo', true]])
    );
    store.overrideSelector(
      getMetricsTagGroupPageIndexMap,
      new Map([['foo', 0]])
    );
    store.overrideSelector(getCardStateMap, {});
  });

  afterEach(() => {
    actions.complete();
    window.localStorage.removeItem(STORAGE_TEST_ONLY.METRICS_LOCAL_STORAGE_KEY);
    store?.resetSelectors();
  });

  it('restores category choices even when no member metadata is loaded', () => {
    const namespace = TEST_ONLY.getNamespace('/tmp/tensorboard/runs', [
      'exp1',
    ])!;
    dataSource.setState(namespace, ['offscreen'], {
      tagGroupExpanded: new Map([['offscreen', false]]),
      tagGroupPageIndex: new Map([['offscreen', 7]]),
      cardState: new Map(),
    });
    store.overrideSelector(getNonEmptyCardIdsWithMetadata, []);
    store.refreshState();
    const subscription =
      effects.hydrateFetchedMetadataFromLocalStorage$.subscribe();
    actions.next(
      metricsActions.metricsTagMetadataLoaded({
        tagMetadata: {
          scalars: {tagDescriptions: {}, tagToRuns: {}},
          histograms: {tagDescriptions: {}, tagToRuns: {}},
          images: {tagDescriptions: {}, tagRunSampledInfo: {}},
        },
      })
    );
    expect(new MetricsLocalStorageDataSource().getState(namespace, [])).toEqual(
      {
        tagGroupExpanded: new Map([
          ['foo', true],
          ['offscreen', false],
        ]),
        tagGroupPageIndex: new Map([
          ['foo', 0],
          ['offscreen', 7],
        ]),
        cardState: new Map(),
      }
    );
    subscription.unsubscribe();
  });

  it('hydrates group expansion and page index after metadata loads', () => {
    spyOn(dataSource, 'getState').and.returnValue({
      tagGroupExpanded: new Map([
        ['foo', false],
        ['bar', true],
      ]),
      tagGroupPageIndex: new Map([
        ['foo', 2],
        ['bar', 1],
      ]),
      cardState: new Map(),
    });
    const setStateSpy = spyOn(dataSource, 'setState').and.stub();
    store.refreshState();

    effects.hydrateFetchedMetadataFromLocalStorage$.subscribe();
    actions.next(
      metricsActions.metricsTagMetadataLoaded({
        tagMetadata: {
          scalars: {tagDescriptions: {}, tagToRuns: {}},
          histograms: {tagDescriptions: {}, tagToRuns: {}},
          images: {tagDescriptions: {}, tagRunSampledInfo: {}},
        },
      })
    );

    expect(dispatchedActions).toEqual([
      metricsActions.metricsLocalStorageHydrated({
        tagGroups: ['bar', 'foo'],
        tagGroupExpanded: {foo: false, bar: true},
        tagGroupPageIndex: {foo: 2, bar: 1},
        cardState: {},
      }),
    ]);
    expect(setStateSpy).toHaveBeenCalledOnceWith(
      jasmine.stringMatching('/tmp/tensorboard/runs'),
      ['bar', 'foo'],
      {
        tagGroupExpanded: new Map([
          ['foo', false],
          ['bar', true],
        ]),
        tagGroupPageIndex: new Map([
          ['foo', 2],
          ['bar', 1],
        ]),
        cardState: new Map(),
      }
    );
  });

  it('hydrates existing metadata when environment loads after metadata', () => {
    spyOn(dataSource, 'getState').and.returnValue({
      tagGroupExpanded: new Map([['foo', false]]),
      tagGroupPageIndex: new Map([['foo', 3]]),
      cardState: new Map(),
    });
    const setStateSpy = spyOn(dataSource, 'setState').and.stub();
    store.refreshState();

    effects.hydrateExistingMetadataFromLocalStorage$.subscribe();
    actions.next(
      coreActions.environmentLoaded({
        environment: {
          data_location: '/tmp/tensorboard/runs',
          window_title: '',
        },
      })
    );

    expect(dispatchedActions).toEqual([
      metricsActions.metricsLocalStorageHydrated({
        tagGroups: ['bar', 'foo'],
        tagGroupExpanded: {foo: false},
        tagGroupPageIndex: {foo: 3},
        cardState: {},
      }),
    ]);
    expect(setStateSpy).toHaveBeenCalledOnceWith(
      jasmine.stringMatching('/tmp/tensorboard/runs'),
      ['bar', 'foo'],
      {
        tagGroupExpanded: new Map([['foo', false]]),
        tagGroupPageIndex: new Map([['foo', 3]]),
        cardState: new Map(),
      }
    );
  });

  it('syncs current group state on group UI changes', () => {
    const setStateSpy = spyOn(dataSource, 'setState').and.stub();
    store.overrideSelector(
      getMetricsTagGroupExpandedMap,
      new Map([
        ['foo', false],
        ['bar', true],
      ])
    );
    store.overrideSelector(
      getMetricsTagGroupPageIndexMap,
      new Map([
        ['foo', 4],
        ['bar', 0],
      ])
    );
    store.refreshState();

    effects.syncMetricsToLocalStorage$.subscribe();
    actions.next(
      metricsActions.metricsTagGroupExpansionChanged({tagGroup: 'foo'})
    );

    expect(setStateSpy).toHaveBeenCalledOnceWith(
      jasmine.stringMatching('/tmp/tensorboard/runs'),
      ['bar', 'foo'],
      {
        tagGroupExpanded: new Map([
          ['foo', false],
          ['bar', true],
        ]),
        tagGroupPageIndex: new Map([
          ['foo', 4],
          ['bar', 0],
        ]),
        cardState: new Map(),
      }
    );
  });

  it('lets stored card state win over live card state on hydration', () => {
    spyOn(dataSource, 'getState').and.returnValue({
      tagGroupExpanded: new Map(),
      tagGroupPageIndex: new Map(),
      cardState: new Map([
        ['card1', {chartHeight: 480}],
        ['card2', {fullWidth: true, tableHeight: 260}],
      ]),
    });
    const setStateSpy = spyOn(dataSource, 'setState').and.stub();
    store.overrideSelector(getCardStateMap, {
      card1: {
        fullWidth: false,
        tableExpanded: true,
        chartHeight: 300,
        tableHeight: 120,
        logScale: true,
      },
    });
    store.refreshState();

    effects.hydrateFetchedMetadataFromLocalStorage$.subscribe();
    actions.next(
      metricsActions.metricsTagMetadataLoaded({
        tagMetadata: {
          scalars: {tagDescriptions: {}, tagToRuns: {}},
          histograms: {tagDescriptions: {}, tagToRuns: {}},
          images: {tagDescriptions: {}, tagRunSampledInfo: {}},
        },
      })
    );

    const expectedCardState = {
      card1: {
        fullWidth: false,
        tableExpanded: true,
        // Stored value wins; `logScale` is not persisted.
        chartHeight: 480,
        tableHeight: 120,
      },
      card2: {fullWidth: true, tableHeight: 260},
    };
    expect(dispatchedActions).toEqual([
      metricsActions.metricsLocalStorageHydrated({
        tagGroups: ['bar', 'foo'],
        tagGroupExpanded: {foo: true},
        tagGroupPageIndex: {foo: 0},
        cardState: expectedCardState,
      }),
    ]);
    expect(setStateSpy.calls.mostRecent().args[2].cardState).toEqual(
      new Map(Object.entries(expectedCardState))
    );
  });

  it('writes the persisted card state keys when a card state changes', () => {
    const setStateSpy = spyOn(dataSource, 'setState').and.stub();
    store.overrideSelector(getCardStateMap, {
      card1: {chartHeight: 333, logScale: true},
      card2: {userViewBox: null},
    });
    store.refreshState();

    effects.syncMetricsToLocalStorage$.subscribe();
    actions.next(
      metricsActions.metricsCardStateUpdated({
        cardId: 'card1',
        settings: {chartHeight: 333},
      })
    );

    expect(setStateSpy.calls.mostRecent().args[2].cardState).toEqual(
      // `card2` holds no persisted key, so it is not written at all.
      new Map([['card1', {chartHeight: 333}]])
    );
  });

  it('writes the card state when a card is toggled to full size', () => {
    const setStateSpy = spyOn(dataSource, 'setState').and.stub();
    store.overrideSelector(getCardStateMap, {card1: {fullWidth: true}});
    store.refreshState();

    effects.syncMetricsToLocalStorage$.subscribe();
    actions.next(metricsActions.metricsCardFullSizeToggled({cardId: 'card1'}));

    expect(setStateSpy.calls.mostRecent().args[2].cardState).toEqual(
      new Map([['card1', {fullWidth: true}]])
    );
  });
});
