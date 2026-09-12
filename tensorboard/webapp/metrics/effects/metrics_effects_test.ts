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
import {provideMockActions} from '@ngrx/effects/testing';
import {Action, Store} from '@ngrx/store';
import {MockStore, provideMockStore} from '@ngrx/store/testing';
import {
  asapScheduler,
  of,
  queueScheduler,
  Subject,
  Subscription,
  VirtualTimeScheduler,
} from 'rxjs';
import {buildNavigatedAction, buildRoute} from '../../app_routing/testing';
import {RouteKind} from '../../app_routing/types';
import {State} from '../../app_state';
import * as coreActions from '../../core/actions';
import {getActivePlugin} from '../../core/store';
import * as coreTesting from '../../core/testing';
import * as runsActions from '../../runs/actions';
import * as selectors from '../../selectors';
import {DataLoadState} from '../../types/data';
import {nextElementId} from '../../util/dom';
import {TBHttpClientTestingModule} from '../../webapp_data_source/tb_http_client_testing';
import * as actions from '../actions';
import {
  MetricsDataSource,
  METRICS_PLUGIN_ID,
  MultiRunPluginType,
  PluginType,
  SingleRunPluginType,
  TagMetadata,
  TimeSeriesRequest,
  TimeSeriesResponse,
  SavedPinsDataSource,
} from '../data_source';
import {getMetricsTagMetadataLoadState, MetricsState} from '../store';
import {reducers} from '../store/metrics_reducers';
import {
  appStateFromMetricsState,
  buildDataSourceTagMetadata,
  buildMetricsState,
  createCardMetadata,
  createScalarStepData,
  provideTestingMetricsDataSource,
  provideTestingSavedPinsDataSource,
} from '../testing';
import {CardId, TooltipSort} from '../types';
import {CardFetchInfo, MetricsEffects, TEST_ONLY} from './index';
import {buildMockState} from '../../testing/utils';

describe('metrics effects', () => {
  let metricsDataSource: MetricsDataSource;
  let savedPinsDataSource: SavedPinsDataSource;
  let effects: MetricsEffects;
  let store: MockStore<State>;
  let actions$: Subject<Action>;
  let actualActions: Action[] = [];
  let dataEffectsSubscription: Subscription | undefined;
  let scheduler: VirtualTimeScheduler;

  beforeEach(async () => {
    actions$ = new Subject<Action>();
    actualActions = [];
    // The effect subscribes in beforeEach, before any fakeAsync test zone.
    // Its initial ASAP task can otherwise hold later tasks outside that zone.
    // Use RxJS's cancellable deferred actions, advancing them explicitly only
    // after synchronous store/queueScheduler work has finished.
    scheduler = new VirtualTimeScheduler();
    spyOn(asapScheduler, 'schedule').and.callFake(
      scheduler.schedule.bind(scheduler)
    );
    spyOn(asapScheduler, 'now').and.callFake(scheduler.now.bind(scheduler));

    await TestBed.configureTestingModule({
      imports: [TBHttpClientTestingModule],
      providers: [
        provideMockActions(actions$),
        provideTestingMetricsDataSource(),
        provideTestingSavedPinsDataSource(),
        MetricsEffects,
        provideMockStore({
          initialState: {
            ...buildMockState({
              ...appStateFromMetricsState(buildMetricsState()),
              ...coreTesting.createState(coreTesting.createCoreState()),
            }),
          },
        }),
      ],
    }).compileComponents();

    store = TestBed.inject<Store<State>>(Store) as MockStore<State>;
    // Cast to jasmine.Spy for compatibility between NgRx dispatch signature overloads.
    (spyOn(store, 'dispatch') as jasmine.Spy).and.callFake((action: Action) => {
      actualActions.push(action);
    });
    effects = TestBed.inject(MetricsEffects);
    metricsDataSource = TestBed.inject(MetricsDataSource);
    savedPinsDataSource = TestBed.inject(SavedPinsDataSource);
    store.overrideSelector(selectors.getExperimentIdsFromRoute, null);
    store.overrideSelector(
      selectors.getRunSelectionMapFilteredToCurrentRoute,
      new Map([
        ['run1', true],
        ['run2', true],
      ])
    );
    store.overrideSelector(selectors.getPinnedCardsWithMetadata, []);
    store.overrideSelector(selectors.getMetricsIgnoreOutliers, false);
    store.overrideSelector(selectors.getMetricsScalarSmoothing, 0.3);
    store.overrideSelector(
      selectors.getMetricsTooltipSort,
      TooltipSort.ALPHABETICAL
    );
  });

  afterEach(() => {
    dataEffectsSubscription?.unsubscribe();
    dataEffectsSubscription = undefined;
    actions$.complete();
    store?.resetSelectors();
  });

  function dispatchAction(action: Action) {
    actions$.next(action);
    scheduler.flush();
  }

  describe('#dataEffects', () => {
    beforeEach(() => {
      dataEffectsSubscription = effects.dataEffects$.subscribe();
    });

    describe('loadTagMetadata', () => {
      let fetchTagMetadataSpy: jasmine.Spy;
      beforeEach(() => {
        fetchTagMetadataSpy = spyOn(
          metricsDataSource,
          'fetchTagMetadata'
        ).and.returnValue(new Subject<TagMetadata>());
      });

      it('requests members only for visible expanded categories and their current page', () => {
        store.overrideSelector(selectors.getExperimentIdsFromRoute, ['exp']);
        store.overrideSelector(getActivePlugin, METRICS_PLUGIN_ID);
        store.overrideSelector(selectors.getMetricsCatalogViewport, {
          groupOffset: 40,
          groupLimit: 40,
          visibleGroups: ['closed', 'visible'],
          filteredOffset: 0,
          filteredLimit: 40,
        });
        store.overrideSelector(
          selectors.getMetricsTagGroupExpandedMap,
          new Map([
            ['closed', false],
            ['visible', true],
            ['offscreen', true],
          ])
        );
        store.overrideSelector(
          selectors.getMetricsTagGroupPageIndexMap,
          new Map([['visible', 2]])
        );
        store.refreshState();
        expect(fetchTagMetadataSpy).not.toHaveBeenCalled();
        scheduler.flush();
        expect(fetchTagMetadataSpy).toHaveBeenCalledTimes(1);
        const request = fetchTagMetadataSpy.calls.mostRecent().args[1];
        expect(request.groups).toEqual([
          {
            name: 'visible',
            offset: 2 * request.groups[0].limit,
            limit: request.groups[0].limit,
          },
        ]);
        expect(request.filteredLimit).toBe(0);

        store.overrideSelector(selectors.getMetricsTagFilter, 'loss');
        store.refreshState();
        scheduler.flush();
        const filtered = fetchTagMetadataSpy.calls.mostRecent().args[1];
        expect(filtered.groups).toEqual([]);
        expect(filtered.groupLimit).toBe(0);
        expect(filtered.filteredLimit).toBe(40);
      });

      it('cancels obsolete selections and accepts only the current catalog response', () => {
        const oldResponse = new Subject<TagMetadata>();
        const newResponse = new Subject<TagMetadata>();
        fetchTagMetadataSpy.and.callFake((_experimentIds, request) =>
          request.runIds.includes('exp/second') ? newResponse : oldResponse
        );
        store.overrideSelector(selectors.getExperimentIdsFromRoute, ['exp']);
        store.overrideSelector(getActivePlugin, METRICS_PLUGIN_ID);
        store.overrideSelector(
          selectors.getRunSelectionMapFilteredToCurrentRoute,
          new Map([['exp/first', true]])
        );
        store.refreshState();
        scheduler.flush();
        expect(oldResponse.observed).toBeTrue();
        store.overrideSelector(
          selectors.getRunSelectionMapFilteredToCurrentRoute,
          new Map([['exp/second', true]])
        );
        store.refreshState();
        scheduler.flush();
        expect(oldResponse.observed).toBeFalse();
        actualActions = [];
        oldResponse.next(buildDataSourceTagMetadata());
        expect(actualActions).toEqual([]);
        const current = buildDataSourceTagMetadata();
        current.scalars.tagToRuns = {loss: ['exp/second']};
        newResponse.next(current);
        expect(actualActions).toContain(
          actions.metricsTagMetadataLoaded({tagMetadata: current})
        );
      });

      it('does not request catalog data while another dashboard is active', () => {
        store.overrideSelector(selectors.getExperimentIdsFromRoute, ['exp']);
        store.overrideSelector(getActivePlugin, 'scalars');
        store.refreshState();
        scheduler.flush();
        expect(fetchTagMetadataSpy).not.toHaveBeenCalled();
        store.overrideSelector(getActivePlugin, METRICS_PLUGIN_ID);
        store.refreshState();
        scheduler.flush();
        expect(fetchTagMetadataSpy).toHaveBeenCalledTimes(1);
      });
    });

    describe('reloading', () => {
      let fetchTagMetadataSpy: jasmine.Spy;
      let fetchTimeSeriesSpy: jasmine.Spy;
      let selectSpy: jasmine.Spy;

      beforeEach(() => {
        fetchTagMetadataSpy = spyOn(
          metricsDataSource,
          'fetchTagMetadata'
        ).and.returnValue(of(buildDataSourceTagMetadata()));
        fetchTimeSeriesSpy = spyOn(metricsDataSource, 'fetchTimeSeries');
        selectSpy = spyOn(store, 'select').and.callThrough();
      });

      function provideCardFetchInfo(
        specs: Array<Partial<CardFetchInfo> & {id: CardId}>
      ) {
        for (const {id, ...rest} of specs) {
          selectSpy.withArgs(TEST_ONLY.getCardFetchInfo, id).and.returnValue(
            of({
              id,
              plugin: PluginType.SCALARS,
              tag: 'tagA',
              runId: null,
              sample: undefined,
              tagRunIds: ['run1'],
              runToLoadState: {},
              ...rest,
            })
          );
        }
      }

      function buildTimeSeriesResponse(): TimeSeriesResponse {
        return {
          plugin: PluginType.SCALARS,
          tag: 'tagA',
          runToSeries: {
            run1: createScalarStepData(),
          },
        };
      }

      const reloadSpecs = [
        {reloadAction: coreActions.manualReload, reloadName: 'manual reload'},
        {reloadAction: coreActions.reload, reloadName: 'auto reload'},
      ];
      for (const {reloadAction, reloadName} of reloadSpecs) {
        it(`re-fetches data on ${reloadName}, while dashboard is open`, () => {
          store.overrideSelector(selectors.getExperimentIdsFromRoute, ['exp1']);
          store.overrideSelector(getMetricsTagMetadataLoadState, {
            state: DataLoadState.LOADED,
            lastLoadedTimeInMs: 1,
          });
          store.overrideSelector(getActivePlugin, METRICS_PLUGIN_ID);
          store.overrideSelector(
            selectors.getVisibleCardIdSet,
            new Set(['card1', 'card2'])
          );
          provideCardFetchInfo([{id: 'card1'}, {id: 'card2'}]);
          store.refreshState();
          scheduler.flush();
          fetchTagMetadataSpy.calls.reset();
          actualActions = [];
          fetchTimeSeriesSpy.and.returnValue(of([buildTimeSeriesResponse()]));

          dispatchAction(reloadAction());

          expect(fetchTagMetadataSpy).toHaveBeenCalledTimes(1);
          expect(fetchTimeSeriesSpy).toHaveBeenCalledTimes(1);
          expect(fetchTimeSeriesSpy).toHaveBeenCalledWith([
            {
              plugin: PluginType.SCALARS,
              tag: 'tagA',
              experimentIds: ['exp1'],
              runIds: ['run1'],
            },
          ]);
        });

        it(`re-fetches data on ${reloadName}, only for non-loading cards`, () => {
          store.overrideSelector(selectors.getExperimentIdsFromRoute, ['exp1']);
          store.overrideSelector(getMetricsTagMetadataLoadState, {
            state: DataLoadState.LOADING,
            lastLoadedTimeInMs: null,
          });
          store.overrideSelector(getActivePlugin, METRICS_PLUGIN_ID);
          store.overrideSelector(
            selectors.getVisibleCardIdSet,
            new Set(['card1', 'card2'])
          );
          provideCardFetchInfo([
            {id: 'card1', runToLoadState: {run1: DataLoadState.LOADED}},
            {id: 'card2', runToLoadState: {run1: DataLoadState.LOADING}},
          ]);
          store.refreshState();
          scheduler.flush();
          fetchTagMetadataSpy.calls.reset();
          actualActions = [];
          fetchTimeSeriesSpy.and.returnValue(of([buildTimeSeriesResponse()]));

          dispatchAction(reloadAction());

          expect(fetchTagMetadataSpy).toHaveBeenCalledTimes(1);
          expect(fetchTimeSeriesSpy).toHaveBeenCalledTimes(1);
          expect(fetchTimeSeriesSpy).toHaveBeenCalledWith([
            {
              plugin: PluginType.SCALARS,
              tag: 'tagA',
              experimentIds: ['exp1'],
              runIds: ['run1'],
            },
          ]);
        });
      }

      it('does not re-fetch data on reload, if open and already loading', () => {
        store.overrideSelector(getMetricsTagMetadataLoadState, {
          state: DataLoadState.LOADING,
          lastLoadedTimeInMs: null,
        });
        store.overrideSelector(getActivePlugin, METRICS_PLUGIN_ID);
        store.overrideSelector(
          selectors.getVisibleCardIdSet,
          new Set(['card1', 'card2'])
        );
        provideCardFetchInfo([
          {id: 'card1', runToLoadState: {run1: DataLoadState.LOADING}},
          {id: 'card2', runToLoadState: {run1: DataLoadState.LOADING}},
        ]);
        store.refreshState();
        fetchTimeSeriesSpy.and.returnValue(of([buildTimeSeriesResponse()]));

        dispatchAction(coreActions.manualReload());
        dispatchAction(coreActions.reload());

        expect(fetchTagMetadataSpy).not.toHaveBeenCalled();
        expect(fetchTimeSeriesSpy).not.toHaveBeenCalled();
      });

      it('does not re-fetch tag metadata if dashboard is inactive', () => {
        store.overrideSelector(getActivePlugin, null);
        store.refreshState();

        dispatchAction(coreActions.manualReload());
        dispatchAction(coreActions.reload());

        expect(fetchTagMetadataSpy).not.toHaveBeenCalled();
      });

      it('does not re-fetch time series, if no cards are visible', () => {
        store.overrideSelector(getActivePlugin, METRICS_PLUGIN_ID);
        store.overrideSelector(selectors.getVisibleCardIdSet, new Set([]));
        store.refreshState();
        fetchTimeSeriesSpy.and.returnValue(of([buildTimeSeriesResponse()]));

        dispatchAction(coreActions.manualReload());
        dispatchAction(coreActions.reload());

        expect(fetchTimeSeriesSpy).not.toHaveBeenCalled();
      });

      it('does not re-fetch time series, until a valid experiment id', () => {
        // Reset any `getExperimentIdsFromRoute` overrides above.
        store.resetSelectors();
        store.overrideSelector(getActivePlugin, METRICS_PLUGIN_ID);
        // `resetSelectors` also dropped the run selection default.
        store.overrideSelector(
          selectors.getRunSelectionMapFilteredToCurrentRoute,
          new Map([['run1', true]])
        );
        store.overrideSelector(selectors.getPinnedCardsWithMetadata, []);
        store.overrideSelector(
          selectors.getVisibleCardIdSet,
          new Set(['card1'])
        );
        provideCardFetchInfo([
          {id: 'card1', runToLoadState: {run1: DataLoadState.LOADED}},
        ]);
        store.overrideSelector(selectors.getExperimentIdsFromRoute, null);
        store.refreshState();
        fetchTimeSeriesSpy.and.returnValue(of([buildTimeSeriesResponse()]));

        dispatchAction(coreActions.manualReload());
        dispatchAction(coreActions.reload());

        expect(fetchTimeSeriesSpy).not.toHaveBeenCalled();

        store.overrideSelector(selectors.getExperimentIdsFromRoute, ['exp1']);
        store.refreshState();

        dispatchAction(coreActions.manualReload());
        dispatchAction(coreActions.reload());

        expect(fetchTimeSeriesSpy).toHaveBeenCalledTimes(2);
      });
    });

    describe('loadTimeSeriesForVisibleCardsWithoutData', () => {
      let fetchTimeSeriesSpy: jasmine.Spy;
      const runToSeries = {run1: createScalarStepData()};
      const sampleBackendResponses: TimeSeriesResponse[] = [
        {
          plugin: PluginType.SCALARS,
          tag: 'scalarTag',
          runToSeries: runToSeries,
        },
        {
          plugin: PluginType.SCALARS,
          tag: 'scalarTag2',
          runToSeries: runToSeries,
        },
      ];

      beforeEach(() => {
        store.overrideSelector(getActivePlugin, METRICS_PLUGIN_ID);
        store.overrideSelector(selectors.getExperimentIdsFromRoute, ['exp1']);
      });

      it('does not fetch when nothing is visible', () => {
        fetchTimeSeriesSpy = spyOn(
          metricsDataSource,
          'fetchTimeSeries'
        ).and.returnValue(of(sampleBackendResponses));
        store.overrideSelector(selectors.getExperimentIdsFromRoute, ['exp1']);
        store.overrideSelector(TEST_ONLY.getCardFetchInfo, {
          id: 'card1',
          plugin: PluginType.SCALARS,
          tag: 'tagA',
          runId: null,
          tagRunIds: ['run1'],
          runToLoadState: {},
        });
        store.refreshState();

        dispatchAction(
          actions.cardVisibilityChanged({enteredCards: [], exitedCards: []})
        );

        expect(fetchTimeSeriesSpy).not.toHaveBeenCalled();
      });

      it('fetches when a previously offscreen card becomes visible', () => {
        fetchTimeSeriesSpy = spyOn(
          metricsDataSource,
          'fetchTimeSeries'
        ).and.returnValue(of(sampleBackendResponses));
        store.overrideSelector(selectors.getExperimentIdsFromRoute, ['exp1']);
        store.overrideSelector(TEST_ONLY.getCardFetchInfo, {
          id: 'card1',
          plugin: PluginType.SCALARS,
          tag: 'tagA',
          runId: null,
          tagRunIds: ['run1'],
          runToLoadState: {},
        });

        const card1ElementId = nextElementId();
        store.overrideSelector(
          selectors.getVisibleCardIdSet,
          new Set<string>([])
        );
        store.refreshState();
        dispatchAction(
          actions.cardVisibilityChanged({
            enteredCards: [],
            exitedCards: [{elementId: card1ElementId, cardId: 'card1'}],
          })
        );

        expect(fetchTimeSeriesSpy).not.toHaveBeenCalled();

        store.overrideSelector(
          selectors.getVisibleCardIdSet,
          new Set(['card1'])
        );
        store.refreshState();
        dispatchAction(
          actions.cardVisibilityChanged({
            enteredCards: [{elementId: card1ElementId, cardId: 'card1'}],
            exitedCards: [],
          })
        );

        const expectedRequest: TimeSeriesRequest = {
          plugin: PluginType.SCALARS as MultiRunPluginType,
          tag: 'tagA',
          experimentIds: ['exp1'],
          runIds: ['run1'],
        };
        expect(fetchTimeSeriesSpy.calls.count()).toBe(1);
        expect(fetchTimeSeriesSpy).toHaveBeenCalledWith([expectedRequest]);
      });

      it('fetches only the selected runs that the tag has', () => {
        fetchTimeSeriesSpy = spyOn(
          metricsDataSource,
          'fetchTimeSeries'
        ).and.returnValue(of(sampleBackendResponses));
        store.overrideSelector(selectors.getExperimentIdsFromRoute, ['exp1']);
        store.overrideSelector(
          selectors.getRunSelectionMapFilteredToCurrentRoute,
          new Map([
            ['exp1/run1', true],
            ['exp1/run2', false],
          ])
        );
        store.overrideSelector(TEST_ONLY.getCardFetchInfo, {
          id: 'card1',
          plugin: PluginType.SCALARS,
          tag: 'tagA',
          runId: null,
          // 'exp1/run3' is selected nowhere; 'exp1/run4' has no data for the
          // tag.
          tagRunIds: ['exp1/run1', 'exp1/run2'],
          runToLoadState: {},
        });
        store.overrideSelector(
          selectors.getVisibleCardIdSet,
          new Set(['card1'])
        );
        store.refreshState();
        dispatchAction(
          actions.cardVisibilityChanged({
            enteredCards: [{elementId: nextElementId(), cardId: 'card1'}],
            exitedCards: [],
          })
        );

        const expectedRequest: TimeSeriesRequest = {
          plugin: PluginType.SCALARS as MultiRunPluginType,
          tag: 'tagA',
          experimentIds: ['exp1'],
          runIds: ['exp1/run1'],
        };
        expect(fetchTimeSeriesSpy).toHaveBeenCalledWith([expectedRequest]);
      });

      it('fetches newly selected runs without re-fetching loaded runs', () => {
        fetchTimeSeriesSpy = spyOn(
          metricsDataSource,
          'fetchTimeSeries'
        ).and.returnValue(of(sampleBackendResponses));
        store.overrideSelector(selectors.getExperimentIdsFromRoute, ['exp1']);
        store.overrideSelector(
          selectors.getRunSelectionMapFilteredToCurrentRoute,
          new Map([
            ['exp1/run1', true],
            ['exp1/run2', true],
          ])
        );
        store.overrideSelector(TEST_ONLY.getCardFetchInfo, {
          id: 'card1',
          plugin: PluginType.SCALARS,
          tag: 'tagA',
          runId: null,
          tagRunIds: ['exp1/run1', 'exp1/run2'],
          runToLoadState: {'exp1/run1': DataLoadState.LOADED},
        });
        store.overrideSelector(
          selectors.getVisibleCardIdSet,
          new Set(['card1'])
        );
        store.refreshState();
        dispatchAction(runsActions.runSelectionToggled({runId: 'exp1/run2'}));

        // Only the run that is missing is requested.
        const expectedRequest: TimeSeriesRequest = {
          plugin: PluginType.SCALARS as MultiRunPluginType,
          tag: 'tagA',
          experimentIds: ['exp1'],
          runIds: ['exp1/run2'],
        };
        expect(fetchTimeSeriesSpy).toHaveBeenCalledWith([expectedRequest]);
      });

      it('does not fetch when the regex filter changes', () => {
        fetchTimeSeriesSpy = spyOn(
          metricsDataSource,
          'fetchTimeSeries'
        ).and.returnValue(of(sampleBackendResponses));
        store.overrideSelector(selectors.getExperimentIdsFromRoute, ['exp1']);
        store.overrideSelector(
          selectors.getRunSelectionMapFilteredToCurrentRoute,
          new Map([
            ['exp1/run1', true],
            ['exp1/run2', true],
          ])
        );
        store.overrideSelector(TEST_ONLY.getCardFetchInfo, {
          id: 'card1',
          plugin: PluginType.SCALARS,
          tag: 'tagA',
          runId: null,
          tagRunIds: ['exp1/run1', 'exp1/run2'],
          runToLoadState: {
            'exp1/run1': DataLoadState.LOADED,
            'exp1/run2': DataLoadState.LOADED,
          },
        });
        store.overrideSelector(
          selectors.getVisibleCardIdSet,
          new Set(['card1'])
        );
        store.refreshState();
        dispatchAction(
          runsActions.runSelectorRegexFilterChanged({regexString: '('})
        );

        // The filter is a view concern; discarding data on a keystroke would
        // refetch every run once the filter widens again.
        expect(fetchTimeSeriesSpy).not.toHaveBeenCalled();
      });

      it('does not fetch when every selected run is loaded', () => {
        fetchTimeSeriesSpy = spyOn(
          metricsDataSource,
          'fetchTimeSeries'
        ).and.returnValue(of(sampleBackendResponses));
        store.overrideSelector(selectors.getExperimentIdsFromRoute, ['exp1']);
        store.overrideSelector(
          selectors.getRunSelectionMapFilteredToCurrentRoute,
          new Map([
            ['exp1/run1', true],
            ['exp1/run2', false],
          ])
        );
        store.overrideSelector(TEST_ONLY.getCardFetchInfo, {
          id: 'card1',
          plugin: PluginType.SCALARS,
          tag: 'tagA',
          runId: null,
          tagRunIds: ['exp1/run1', 'exp1/run2'],
          runToLoadState: {'exp1/run1': DataLoadState.LOADED},
        });
        store.overrideSelector(
          selectors.getVisibleCardIdSet,
          new Set(['card1'])
        );
        store.refreshState();
        dispatchAction(runsActions.runSelectionToggled({runId: 'exp1/run2'}));

        expect(fetchTimeSeriesSpy).not.toHaveBeenCalled();
      });

      it('does not fetch when no run is selected', () => {
        fetchTimeSeriesSpy = spyOn(
          metricsDataSource,
          'fetchTimeSeries'
        ).and.returnValue(of(sampleBackendResponses));
        store.overrideSelector(selectors.getExperimentIdsFromRoute, ['exp1']);
        store.overrideSelector(
          selectors.getRunSelectionMapFilteredToCurrentRoute,
          new Map([
            ['exp1/run1', false],
            ['exp1/run2', false],
          ])
        );
        store.overrideSelector(TEST_ONLY.getCardFetchInfo, {
          id: 'card1',
          plugin: PluginType.SCALARS,
          tag: 'tagA',
          runId: null,
          tagRunIds: ['exp1/run1', 'exp1/run2'],
          runToLoadState: {},
        });
        store.overrideSelector(
          selectors.getVisibleCardIdSet,
          new Set(['card1'])
        );
        store.refreshState();
        dispatchAction(
          actions.cardVisibilityChanged({
            enteredCards: [{elementId: nextElementId(), cardId: 'card1'}],
            exitedCards: [],
          })
        );

        expect(fetchTimeSeriesSpy).not.toHaveBeenCalled();
      });

      it('fetches multiple card data', () => {
        store.overrideSelector(selectors.getExperimentIdsFromRoute, ['exp1']);
        const selectSpy = spyOn(store, 'select').and.callThrough();
        selectSpy.withArgs(TEST_ONLY.getCardFetchInfo, 'card1').and.returnValue(
          of({
            id: 'card1',
            plugin: PluginType.SCALARS,
            tag: 'tagA',
            runId: null,
            sample: undefined,
            tagRunIds: ['run1'],
            runToLoadState: {},
          })
        );
        selectSpy.withArgs(TEST_ONLY.getCardFetchInfo, 'card2').and.returnValue(
          of({
            id: 'card2',
            plugin: PluginType.IMAGES,
            tag: 'tagB',
            runId: 'run1',
            sample: 5,
            tagRunIds: ['run1'],
            runToLoadState: {},
          })
        );

        const expectedRequests: TimeSeriesRequest[] = [
          {
            plugin: PluginType.SCALARS as MultiRunPluginType,
            tag: 'tagA',
            experimentIds: ['exp1'],
            runIds: ['run1'],
          },
          {
            plugin: PluginType.IMAGES as SingleRunPluginType,
            tag: 'tagB',
            runId: 'run1',
            sample: 5,
          },
        ];
        fetchTimeSeriesSpy = spyOn(
          metricsDataSource,
          'fetchTimeSeries'
        ).and.returnValue(of(sampleBackendResponses));

        store.overrideSelector(
          selectors.getVisibleCardIdSet,
          new Set(['card1', 'card2'])
        );
        store.refreshState();
        dispatchAction(
          actions.cardVisibilityChanged({
            enteredCards: [
              {elementId: nextElementId(), cardId: 'card1'},
              {elementId: nextElementId(), cardId: 'card2'},
            ],
            exitedCards: [],
          })
        );

        expect(fetchTimeSeriesSpy.calls.allArgs()).toEqual([
          [expectedRequests],
        ]);
      });

      const metaSpec = [
        {runLoadState: DataLoadState.FAILED, tag: 'failed'},
        {runLoadState: DataLoadState.LOADED, tag: 'loaded'},
        {runLoadState: DataLoadState.LOADING, tag: 'loading'},
      ];
      for (const spec of metaSpec) {
        const {runLoadState, tag} = spec;
        const title = `should not fetch when load state is ${tag}`;
        it(title, () => {
          const selectSpy = spyOn(store, 'select').and.callThrough();
          selectSpy
            .withArgs(TEST_ONLY.getCardFetchInfo, 'card1')
            .and.returnValue(
              of({
                id: 'card1',
                plugin: PluginType.SCALARS,
                tag: 'tagA',
                tagRunIds: ['run1'],
                runToLoadState: {run1: runLoadState},
              })
            );
          fetchTimeSeriesSpy = spyOn(metricsDataSource, 'fetchTimeSeries');

          store.overrideSelector(
            selectors.getVisibleCardIdSet,
            new Set(['card1'])
          );
          store.refreshState();
          dispatchAction(
            actions.cardVisibilityChanged({
              enteredCards: [{elementId: nextElementId(), cardId: 'card1'}],
              exitedCards: [],
            })
          );

          expect(fetchTimeSeriesSpy).not.toHaveBeenCalled();
        });
      }
    });

    describe('visible history lifecycle', () => {
      let metricsState: MetricsState;
      let responses: Subject<TimeSeriesResponse[]>[];
      let fetchTimeSeriesSpy: jasmine.Spy;
      const original = {elementId: nextElementId(), cardId: 'card1'};
      const pinned = {elementId: nextElementId(), cardId: 'pinned'};
      const second = {elementId: nextElementId(), cardId: 'card2'};

      function publishState() {
        store.setState(
          buildMockState({
            ...appStateFromMetricsState(metricsState),
            ...coreTesting.createState(coreTesting.createCoreState()),
          })
        );
      }

      function dispatchThroughReducer(action: Action) {
        // NgRx emits scanned actions from inside queueScheduler. Dispatches
        // from finalize must wait until the triggering action finishes.
        queueScheduler.schedule(() => {
          metricsState = reducers(metricsState, action);
          publishState();
          actions$.next(action);
        });
      }

      function changeVisibility(
        enteredCards: Array<typeof original>,
        exitedCards: Array<typeof original>
      ) {
        dispatchThroughReducer(
          actions.cardVisibilityChanged({
            enteredCards,
            exitedCards,
          })
        );
        scheduler.flush();
      }

      beforeEach(() => {
        responses = [];
        metricsState = buildMetricsState({
          cardMetadataMap: {
            card1: {plugin: PluginType.SCALARS, tag: 'tagA', runId: null},
            pinned: {plugin: PluginType.SCALARS, tag: 'tagA', runId: null},
            card2: {plugin: PluginType.SCALARS, tag: 'tagB', runId: null},
          },
          cardToPinnedCopy: new Map([['card1', 'pinned']]),
          pinnedCardToOriginal: new Map([['pinned', 'card1']]),
          tagMetadata: {
            ...buildMetricsState().tagMetadata,
            scalars: {
              tagDescriptions: {},
              tagToRuns: {tagA: ['run1', 'run2'], tagB: ['run1', 'run2']},
            },
          },
        });
        store.overrideSelector(getActivePlugin, METRICS_PLUGIN_ID);
        store.overrideSelector(selectors.getExperimentIdsFromRoute, ['exp1']);
        store.overrideSelector(
          selectors.getRunSelectionMapFilteredToCurrentRoute,
          new Map([
            ['run1', true],
            ['run2', false],
          ])
        );
        (store.dispatch as jasmine.Spy).and.callFake(dispatchThroughReducer);
        // Keep the catalog pending so it cannot replace the already-loaded
        // metadata fixture while this suite exercises history requests.
        spyOn(metricsDataSource, 'fetchTagMetadata').and.returnValue(
          new Subject<TagMetadata>()
        );
        fetchTimeSeriesSpy = spyOn(
          metricsDataSource,
          'fetchTimeSeries'
        ).and.callFake(() => {
          const response = new Subject<TimeSeriesResponse[]>();
          responses.push(response);
          return response;
        });
        publishState();
      });

      afterEach(() => {
        dataEffectsSubscription?.unsubscribe();
        dataEffectsSubscription = undefined;
        for (const response of responses) response.complete();
      });

      it('shares in-flight ownership and reuses history after every copy exits', () => {
        changeVisibility([original], []);
        const pending = responses[0];
        expect(pending.observers.length).toBe(1);
        const duplicate = {elementId: nextElementId(), cardId: 'card1'};
        changeVisibility([duplicate, pinned], []);
        changeVisibility([], [original, duplicate]);
        expect(fetchTimeSeriesSpy).toHaveBeenCalledTimes(1);
        expect(pending.observers.length).toBe(1);

        const series = createScalarStepData();
        pending.next([
          {
            plugin: PluginType.SCALARS,
            tag: 'tagA',
            runToSeries: {run1: series},
          },
        ]);
        pending.complete();
        expect(
          metricsState.timeSeriesData.scalars['tagA'].runToSeries['run1']
        ).toBe(series);
        changeVisibility([], [pinned]);
        expect(
          metricsState.timeSeriesData.scalars['tagA'].runToSeries['run1']
        ).toBe(series);
        changeVisibility([original], []);
        expect(fetchTimeSeriesSpy).toHaveBeenCalledTimes(1);
      });

      it('cancels the last visible request, ignores late data, and fetches again on re-entry', () => {
        changeVisibility([original], []);
        const cancelled = responses[0];
        expect(
          metricsState.timeSeriesData.scalars['tagA'].runToLoadState['run1']
        ).toBe(DataLoadState.LOADING);
        changeVisibility([], [original]);
        expect(cancelled.observers.length).toBe(0);
        expect(metricsState.timeSeriesData.scalars).toEqual({});
        cancelled.next([
          {
            plugin: PluginType.SCALARS,
            tag: 'tagA',
            runToSeries: {run1: createScalarStepData()},
          },
        ]);
        expect(metricsState.timeSeriesData.scalars).toEqual({});

        changeVisibility([original], []);
        expect(fetchTimeSeriesSpy).toHaveBeenCalledTimes(2);
        expect(responses[1].observers.length).toBe(1);
        expect(
          metricsState.timeSeriesData.scalars['tagA'].runToLoadState['run1']
        ).toBe(DataLoadState.LOADING);
      });

      it('cancels and releases visible histories when leaving the dashboard', () => {
        changeVisibility([original], []);
        const cancelled = responses[0];
        store.overrideSelector(getActivePlugin, 'images');
        store.refreshState();
        dispatchThroughReducer(coreActions.changePlugin({plugin: 'images'}));
        scheduler.flush();
        expect(cancelled.observers.length).toBe(0);
        expect(metricsState.visibleCardMap.size).toBe(0);
        expect(metricsState.timeSeriesData.scalars).toEqual({});
      });

      it('keeps ongoing requests when an adjacent chart enters', () => {
        changeVisibility([original], []);
        const pending = responses[0];
        dispatchThroughReducer(
          actions.cardVisibilityChanged({
            enteredCards: [second],
            exitedCards: [],
          })
        );
        expect(pending.observers.length).toBe(1);
        expect(fetchTimeSeriesSpy).toHaveBeenCalledTimes(1);
        scheduler.flush();
        expect(fetchTimeSeriesSpy.calls.allArgs()).toEqual([
          [
            [
              {
                plugin: PluginType.SCALARS,
                tag: 'tagA',
                experimentIds: ['exp1'],
                runIds: ['run1'],
              },
            ],
          ],
          [
            [
              {
                plugin: PluginType.SCALARS,
                tag: 'tagB',
                experimentIds: ['exp1'],
                runIds: ['run1'],
              },
            ],
          ],
        ]);
        expect(
          metricsState.timeSeriesData.scalars['tagA'].runToLoadState['run1']
        ).toBe(DataLoadState.LOADING);
        expect(pending.observers.length).toBe(1);
        expect(responses[1].observers.length).toBe(1);
      });

      it('restarts the still-owned part of a batch after another owner exits', () => {
        changeVisibility([original, second], []);
        const cancelled = responses[0];
        changeVisibility([], [second]);
        expect(cancelled.observers.length).toBe(0);
        expect(fetchTimeSeriesSpy.calls.mostRecent().args).toEqual([
          [
            {
              plugin: PluginType.SCALARS,
              tag: 'tagA',
              experimentIds: ['exp1'],
              runIds: ['run1'],
            },
          ],
        ]);
        expect(metricsState.timeSeriesData.scalars['tagB']).toBeUndefined();
        expect(responses[1].observers.length).toBe(1);
      });

      it('drops deselected late responses even when selected runs remain', () => {
        changeVisibility([original], []);
        responses[0].next([
          {
            plugin: PluginType.SCALARS,
            tag: 'tagA',
            runToSeries: {run1: createScalarStepData()},
          },
        ]);
        responses[0].complete();
        const series =
          metricsState.timeSeriesData.scalars['tagA'].runToSeries['run1'];
        for (const [tag, runId] of [['tagA', 'run2']]) {
          dispatchThroughReducer(
            actions.fetchTimeSeriesLoaded({
              requestResponses: [
                {
                  request: {
                    plugin: PluginType.SCALARS,
                    tag,
                    experimentIds: ['exp1'],
                    runIds: [runId],
                  },
                  response: {
                    plugin: PluginType.SCALARS,
                    tag,
                    runToSeries: {[runId]: createScalarStepData()},
                  },
                },
              ],
            })
          );
        }
        expect(metricsState.timeSeriesData.scalars).toEqual({
          tagA: {
            runToSeries: {run1: series},
            runToLoadState: {run1: DataLoadState.LOADED},
          },
        });
        expect(fetchTimeSeriesSpy).toHaveBeenCalledTimes(1);
      });

      it('invalidates cached offscreen histories on reload before re-entry', () => {
        changeVisibility([original], []);
        responses[0].next([
          {
            plugin: PluginType.SCALARS,
            tag: 'tagA',
            runToSeries: {run1: createScalarStepData()},
          },
        ]);
        responses[0].complete();
        changeVisibility([], [original]);
        dispatchThroughReducer(coreActions.manualReload());
        scheduler.flush();
        expect(metricsState.timeSeriesData.scalars).toEqual({});
        changeVisibility([original], []);
        expect(fetchTimeSeriesSpy).toHaveBeenCalledTimes(2);
        expect(
          metricsState.timeSeriesData.scalars['tagA'].runToLoadState['run1']
        ).toBe(DataLoadState.LOADING);
      });

      it('does not turn revisiting a backend failure into a retry', () => {
        changeVisibility([original], []);
        responses[0].error(new Error('backend unavailable'));
        expect(
          metricsState.timeSeriesData.scalars['tagA'].runToLoadState['run1']
        ).toBe(DataLoadState.FAILED);
        changeVisibility([], [original]);
        changeVisibility([original], []);
        expect(fetchTimeSeriesSpy).toHaveBeenCalledTimes(1);
        dispatchThroughReducer(coreActions.manualReload());
        scheduler.flush();
        expect(fetchTimeSeriesSpy).toHaveBeenCalledTimes(2);
      });
    });

    describe('addOrRemovePin', () => {
      let saveScalarPinSpy: jasmine.Spy;
      let removeScalarPinSpy: jasmine.Spy;

      beforeEach(() => {
        saveScalarPinSpy = spyOn(savedPinsDataSource, 'saveScalarPin');
        removeScalarPinSpy = spyOn(savedPinsDataSource, 'removeScalarPin');

        store.overrideSelector(selectors.getEnableGlobalPins, true);
        store.overrideSelector(TEST_ONLY.getCardFetchInfo, {
          id: 'card1',
          plugin: PluginType.SCALARS,
          tag: 'tagA',
          runId: null,
          tagRunIds: ['run1'],
          runToLoadState: {run1: DataLoadState.LOADED},
        });
        store.overrideSelector(
          selectors.getVisibleCardIdSet,
          new Set(['card1'])
        );
        store.refreshState();
      });

      it('removes scalar pin if the given card was pinned', () => {
        actions$.next(
          actions.cardPinStateToggled({
            cardId: 'card1',
            wasPinned: true,
            canCreateNewPins: true,
          })
        );

        expect(removeScalarPinSpy).toHaveBeenCalledWith('tagA');
        expect(saveScalarPinSpy).not.toHaveBeenCalled();
      });

      it('pins the card if the given card was not pinned and canCreateNewPins is true', () => {
        actions$.next(
          actions.cardPinStateToggled({
            cardId: 'card1',
            wasPinned: false,
            canCreateNewPins: true,
          })
        );

        expect(saveScalarPinSpy).toHaveBeenCalledWith('tagA');
        expect(removeScalarPinSpy).not.toHaveBeenCalled();
      });

      it('does not pin the card if the given card was not pinned and canCreateNewPins is false', () => {
        actions$.next(
          actions.cardPinStateToggled({
            cardId: 'card1',
            wasPinned: false,
            canCreateNewPins: false,
          })
        );

        expect(saveScalarPinSpy).not.toHaveBeenCalled();
        expect(removeScalarPinSpy).not.toHaveBeenCalled();
      });

      it('does not pin the card if the plugin type is not a scalar', () => {
        store.overrideSelector(TEST_ONLY.getCardFetchInfo, {
          id: 'card2',
          plugin: PluginType.HISTOGRAMS,
          tag: 'tagA',
          runId: null,
          tagRunIds: ['run1'],
          runToLoadState: {run1: DataLoadState.LOADED},
        });
        store.overrideSelector(
          selectors.getVisibleCardIdSet,
          new Set(['card2'])
        );
        store.refreshState();

        actions$.next(
          actions.cardPinStateToggled({
            cardId: 'card2',
            wasPinned: false,
            canCreateNewPins: true,
          })
        );

        expect(saveScalarPinSpy).not.toHaveBeenCalled();
        expect(removeScalarPinSpy).not.toHaveBeenCalled();
      });

      it('does not pin the card if there is no matching card', () => {
        actions$.next(
          actions.cardPinStateToggled({
            cardId: 'card3',
            wasPinned: false,
            canCreateNewPins: true,
          })
        );

        expect(saveScalarPinSpy).not.toHaveBeenCalled();
        expect(removeScalarPinSpy).not.toHaveBeenCalled();
      });

      it('does not pin the card if getEnableGlobalPins is false', () => {
        store.overrideSelector(selectors.getEnableGlobalPins, false);
        store.refreshState();

        actions$.next(
          actions.cardPinStateToggled({
            cardId: 'card1',
            wasPinned: false,
            canCreateNewPins: true,
          })
        );

        expect(saveScalarPinSpy).not.toHaveBeenCalled();
        expect(removeScalarPinSpy).not.toHaveBeenCalled();
      });

      it('does not pin the card if getShouldPersistSettings is false', () => {
        store.overrideSelector(selectors.getShouldPersistSettings, false);
        store.refreshState();

        actions$.next(
          actions.cardPinStateToggled({
            cardId: 'card1',
            wasPinned: false,
            canCreateNewPins: true,
          })
        );

        expect(saveScalarPinSpy).not.toHaveBeenCalled();
        expect(removeScalarPinSpy).not.toHaveBeenCalled();
      });

      it('does not pin the card if getMetricsSavingPinsEnabled is false', () => {
        store.overrideSelector(selectors.getMetricsSavingPinsEnabled, false);
        store.refreshState();

        actions$.next(
          actions.cardPinStateToggled({
            cardId: 'card1',
            wasPinned: false,
            canCreateNewPins: true,
          })
        );

        expect(saveScalarPinSpy).not.toHaveBeenCalled();
        expect(removeScalarPinSpy).not.toHaveBeenCalled();
      });
    });

    describe('loadSavedPins', () => {
      const fakeTagList = ['tagA', 'tagB'];
      const fakeUniqueCardInfos = fakeTagList.map((tag) => ({
        plugin: PluginType.SCALARS,
        tag: tag,
      }));
      let getSavedScalarPinsSpy: jasmine.Spy;

      beforeEach(() => {
        store.overrideSelector(selectors.getEnableGlobalPins, true);
        store.refreshState();
      });

      it('dispatches unresolvedPinnedCards action if tag is given', () => {
        getSavedScalarPinsSpy = spyOn(
          savedPinsDataSource,
          'getSavedScalarPins'
        ).and.returnValue(['tagA', 'tagB']);

        actions$.next(TEST_ONLY.initAction());

        expect(actualActions).toEqual([
          actions.metricsUnresolvedPinnedCardsFromLocalStorageAdded({
            cards: fakeUniqueCardInfos,
          }),
        ]);
      });

      it('does not dispatch unresolvedPinnedCards action if tag is an empty list', () => {
        getSavedScalarPinsSpy = spyOn(
          savedPinsDataSource,
          'getSavedScalarPins'
        ).and.returnValue([]);

        actions$.next(TEST_ONLY.initAction());

        expect(actualActions).toEqual([]);
      });

      it('does not load saved pins if getEnableGlobalPins is false', () => {
        getSavedScalarPinsSpy = spyOn(
          savedPinsDataSource,
          'getSavedScalarPins'
        ).and.returnValue(['tagA', 'tagB']);
        store.overrideSelector(selectors.getEnableGlobalPins, false);
        store.refreshState();

        actions$.next(TEST_ONLY.initAction());

        expect(actualActions).toEqual([]);
      });

      it('does not load saved pins if getShouldPersistSettings is false', () => {
        getSavedScalarPinsSpy = spyOn(
          savedPinsDataSource,
          'getSavedScalarPins'
        ).and.returnValue(['tagA', 'tagB']);
        store.overrideSelector(selectors.getShouldPersistSettings, false);
        store.refreshState();

        actions$.next(TEST_ONLY.initAction());

        expect(actualActions).toEqual([]);
      });

      it('does not load saved pins if getMetricsSavingPinsEnabled is false', () => {
        getSavedScalarPinsSpy = spyOn(
          savedPinsDataSource,
          'getSavedScalarPins'
        ).and.returnValue(['tagA', 'tagB']);
        store.overrideSelector(selectors.getMetricsSavingPinsEnabled, false);
        store.refreshState();

        actions$.next(TEST_ONLY.initAction());

        expect(actualActions).toEqual([]);
      });
    });

    describe('removeAllPins', () => {
      let removeAllScalarPinsSpy: jasmine.Spy;

      beforeEach(() => {
        removeAllScalarPinsSpy = spyOn(
          savedPinsDataSource,
          'removeAllScalarPins'
        );
        store.overrideSelector(selectors.getEnableGlobalPins, true);
        store.refreshState();
      });

      it('removes all pins by calling removeAllScalarPins method', () => {
        actions$.next(actions.metricsClearAllPinnedCards());

        expect(removeAllScalarPinsSpy).toHaveBeenCalled();
      });

      it('does not remove pins if getEnableGlobalPins is false', () => {
        store.overrideSelector(selectors.getEnableGlobalPins, false);
        store.refreshState();

        actions$.next(actions.metricsClearAllPinnedCards());

        expect(removeAllScalarPinsSpy).not.toHaveBeenCalled();
      });

      it('does not remove pins if getShouldPersistSettings is false', () => {
        store.overrideSelector(selectors.getShouldPersistSettings, false);
        store.refreshState();

        actions$.next(actions.metricsClearAllPinnedCards());

        expect(removeAllScalarPinsSpy).not.toHaveBeenCalled();
      });

      it('does not remove pins if getMetricsSavingPinsEnabled is false', () => {
        store.overrideSelector(selectors.getMetricsSavingPinsEnabled, false);
        store.refreshState();

        actions$.next(actions.metricsClearAllPinnedCards());

        expect(removeAllScalarPinsSpy).not.toHaveBeenCalled();
      });
    });

    describe('addOrRemovePinsOnToggle', () => {
      let removeAllScalarPinsSpy: jasmine.Spy;
      let saveScalarPinsSpy: jasmine.Spy;

      beforeEach(() => {
        removeAllScalarPinsSpy = spyOn(
          savedPinsDataSource,
          'removeAllScalarPins'
        );
        saveScalarPinsSpy = spyOn(savedPinsDataSource, 'saveScalarPins');
        store.overrideSelector(selectors.getPinnedCardsWithMetadata, [
          {
            cardId: 'card1',
            ...createCardMetadata(PluginType.SCALARS),
            tag: 'tag1',
          },
          {
            cardId: 'card2',
            ...createCardMetadata(PluginType.IMAGES),
            tag: 'tag2',
          },
          {
            cardId: 'card3',
            ...createCardMetadata(PluginType.SCALARS),
            tag: 'tag3',
          },
        ]);
        store.overrideSelector(selectors.getEnableGlobalPins, true);
        store.overrideSelector(selectors.getMetricsSavingPinsEnabled, false);
        store.refreshState();
      });

      it('removes all pins if getMetricsSavingPinsEnabled is false', () => {
        actions$.next(actions.metricsEnableSavingPinsToggled());

        expect(removeAllScalarPinsSpy).toHaveBeenCalled();
        expect(saveScalarPinsSpy).not.toHaveBeenCalled();
      });

      it('add existing pins if getMetricsSavingPinsEnabled is true', () => {
        store.overrideSelector(selectors.getMetricsSavingPinsEnabled, true);
        store.refreshState();

        actions$.next(actions.metricsEnableSavingPinsToggled());

        expect(saveScalarPinsSpy).toHaveBeenCalledWith(['tag1', 'tag3']);
        expect(removeAllScalarPinsSpy).not.toHaveBeenCalled();
      });

      it('does not add or remove pins if getEnableGlobalPins is false', () => {
        store.overrideSelector(selectors.getEnableGlobalPins, false);
        store.refreshState();

        actions$.next(actions.metricsEnableSavingPinsToggled());

        expect(removeAllScalarPinsSpy).not.toHaveBeenCalled();
        expect(saveScalarPinsSpy).not.toHaveBeenCalled();
      });

      it('does not add or remove pins if getShouldPersistSettings is false', () => {
        store.overrideSelector(selectors.getShouldPersistSettings, false);
        store.refreshState();

        actions$.next(actions.metricsEnableSavingPinsToggled());

        expect(removeAllScalarPinsSpy).not.toHaveBeenCalled();
        expect(saveScalarPinsSpy).not.toHaveBeenCalled();
      });
    });
  });
});
