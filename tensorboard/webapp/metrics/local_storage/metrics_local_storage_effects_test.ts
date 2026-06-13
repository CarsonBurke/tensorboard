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
  getEnvironment,
  getExperimentIdsFromRoute,
  getMetricsTagGroupExpandedMap,
  getMetricsTagGroupPageIndexMap,
  getNonEmptyCardIdsWithMetadata,
} from '../../selectors';
import {provideMockTbStore} from '../../testing/utils';
import * as metricsActions from '../actions';
import {PluginType} from '../data_source';
import {MetricsLocalStorageDataSource} from './metrics_local_storage_data_source';
import {MetricsLocalStorageEffects} from './metrics_local_storage_effects';

describe('MetricsLocalStorageEffects', () => {
  let actions: ReplaySubject<Action>;
  let effects: MetricsLocalStorageEffects;
  let store: MockStore<State>;
  let dataSource: MetricsLocalStorageDataSource;
  let dispatchedActions: Action[];

  beforeEach(async () => {
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
  });

  afterEach(() => {
    store?.resetSelectors();
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
    });
    const setStateSpy = spyOn(dataSource, 'setState').and.stub();
    store.refreshState();

    effects.hydrateFetchedMetadataFromLocalStorage$.subscribe();
    actions.next(
      metricsActions.metricsTagMetadataLoaded({
        tagMetadata: {
          scalars: {tagDescriptions: {}, runTagInfo: {}},
          histograms: {tagDescriptions: {}, runTagInfo: {}},
          images: {tagDescriptions: {}, tagRunSampledInfo: {}},
        },
      })
    );

    expect(dispatchedActions).toEqual([
      metricsActions.metricsLocalStorageHydrated({
        tagGroups: ['bar', 'foo'],
        tagGroupExpanded: {foo: false, bar: true},
        tagGroupPageIndex: {foo: 2, bar: 1},
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
      }
    );
  });

  it('hydrates existing metadata when environment loads after metadata', () => {
    spyOn(dataSource, 'getState').and.returnValue({
      tagGroupExpanded: new Map([['foo', false]]),
      tagGroupPageIndex: new Map([['foo', 3]]),
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
      }),
    ]);
    expect(setStateSpy).toHaveBeenCalledOnceWith(
      jasmine.stringMatching('/tmp/tensorboard/runs'),
      ['bar', 'foo'],
      {
        tagGroupExpanded: new Map([['foo', false]]),
        tagGroupPageIndex: new Map([['foo', 3]]),
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
      }
    );
  });
});
