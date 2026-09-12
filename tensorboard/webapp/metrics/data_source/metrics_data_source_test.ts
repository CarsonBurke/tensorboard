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
import {Store} from '@ngrx/store';
import {MockStore} from '@ngrx/store/testing';
import {State} from '../../app_state';
import {buildFeatureFlag} from '../../feature_flag/testing';
import * as selectors from '../../selectors';
import {provideMockTbStore} from '../../testing/utils';
import {
  HttpTestingController,
  TBHttpClientTestingModule,
} from '../../webapp_data_source/tb_http_client_testing';
import {
  BackendScalarColumns,
  BackendTimeSeriesRequest,
  BackendTimeSeriesResponse,
} from './metrics_backend_types';
import {TBMetricsDataSource} from './metrics_data_source';
import {MetricsDataSource, PluginType, TagMetadataRequest} from './types';

const EMPTY_COLUMNS: BackendScalarColumns = {
  steps: [],
  wallTimes: [],
  values: [],
};

describe('TBMetricsDataSource test', () => {
  let httpMock: HttpTestingController;
  let dataSource: MetricsDataSource;
  let store: MockStore<State>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [TBHttpClientTestingModule],
      providers: [
        {provide: MetricsDataSource, useClass: TBMetricsDataSource},
        provideMockTbStore(),
      ],
    }).compileComponents();

    httpMock = TestBed.inject(HttpTestingController);
    dataSource = TestBed.inject(MetricsDataSource);
    store = TestBed.inject<Store<State>>(Store) as MockStore<State>;
    store.overrideSelector(selectors.getFeatureFlags, buildFeatureFlag());
    store.overrideSelector(selectors.getIsFeatureFlagsLoaded, true);
    store.overrideSelector(selectors.getIsMetricsImageSupportEnabled, true);
  });

  afterEach(() => {
    httpMock.verify();
    store?.resetSelectors();
  });

  describe('fetchTagMetadata', () => {
    const scope: TagMetadataRequest = {
      runIds: ['exp1/run/a', 'exp2/run/b'],
      query: '',
      groupOffset: 0,
      groupLimit: 40,
      groups: [{name: 'loss', offset: 20, limit: 10}],
      filteredOffset: 0,
      filteredLimit: 40,
      pinnedTags: [],
    };

    it('does not expand an absent selection into a catalog request', () => {
      const result = jasmine.createSpy();
      dataSource.fetchTagMetadata(['exp1']).subscribe(result);
      expect(result).toHaveBeenCalledWith({
        scalars: {tagDescriptions: {}, tagToRuns: {}},
        histograms: {tagDescriptions: {}, tagToRuns: {}},
        images: {tagDescriptions: {}, tagRunSampledInfo: {}},
        catalog: {
          groups: [],
          totalGroups: 0,
          groupOffset: 0,
          filteredOffset: 0,
          cards: [],
          totalCards: 0,
        },
      });
      httpMock.expectNone(() => true);
    });

    it('excludes runs outside the route without broadening an empty selection for pins', () => {
      const subscription = dataSource
        .fetchTagMetadata(['exp1'], {
          ...scope,
          runIds: ['exp2/run/b'],
          pinnedRunIds: ['exp1/pin', 'exp2/pin'],
          pinnedTags: ['image'],
        })
        .subscribe();
      const request = httpMock.expectOne('/data/plugin/timeseries/catalog');
      expect(request.request.body.runIds).toEqual([]);
      expect(request.request.body.pinnedRunIds).toEqual(['exp1/pin']);
      expect(request.request.body.pinnedTags).toEqual(['image']);
      subscription.unsubscribe();
    });

    it('excludes unsupported images from server-side card counts and page offsets', () => {
      store.overrideSelector(selectors.getIsMetricsImageSupportEnabled, false);
      const subscription = dataSource
        .fetchTagMetadata(['exp1'], scope)
        .subscribe();
      const request = httpMock.expectOne('/data/plugin/timeseries/catalog');
      expect(request.request.body.plugins).toEqual([
        PluginType.SCALARS,
        PluginType.HISTOGRAMS,
      ]);
      subscription.unsubscribe();
    });

    it('keeps pins when a persisted filter selects only a disabled plugin', () => {
      store.overrideSelector(selectors.getIsMetricsImageSupportEnabled, false);
      const subscription = dataSource
        .fetchTagMetadata(['exp1'], {
          ...scope,
          plugins: [PluginType.IMAGES],
          pinnedTags: ['loss'],
        })
        .subscribe();
      const request = httpMock.expectOne('/data/plugin/timeseries/catalog');
      expect(request.request.body.runIds).toEqual([]);
      expect(request.request.body.pinnedRunIds).toEqual(['exp1/run/a']);
      expect(request.request.body.pinnedTags).toEqual(['loss']);
      subscription.unsubscribe();
    });

    it('cancels an obsolete catalog request when its consumer unsubscribes', () => {
      const subscription = dataSource
        .fetchTagMetadata(['exp1'], scope)
        .subscribe();
      const request = httpMock.expectOne('/data/plugin/timeseries/catalog');
      subscription.unsubscribe();
      expect(request.cancelled).toBeTrue();
    });
  });

  describe('fetchTimeSeries', () => {
    it('does not fetch when no experiment is passed', () => {
      const resultSpy = jasmine.createSpy();
      dataSource.fetchTimeSeries([]).subscribe(resultSpy);

      expect(resultSpy).not.toHaveBeenCalled();
    });

    it('does not fetch when request has empty experiment ids', () => {
      const resultSpy = jasmine.createSpy();
      dataSource
        .fetchTimeSeries([
          {
            plugin: PluginType.SCALARS,
            tag: 'tag1',
            experimentIds: [],
          },
        ])
        .subscribe(resultSpy);

      expect(resultSpy).not.toHaveBeenCalled();
    });

    it('expands scalar columns into per-point series', () => {
      const resultSpy = jasmine.createSpy();
      dataSource
        .fetchTimeSeries([
          {
            plugin: PluginType.SCALARS,
            tag: 'tag1',
            experimentIds: ['exp1'],
          },
        ])
        .subscribe(resultSpy);

      httpMock
        .expectOne('/experiment/exp1/data/plugin/timeseries/timeSeries')
        .flush([
          {
            plugin: PluginType.SCALARS,
            tag: 'tag1',
            runToSeries: {
              run1: {
                steps: [0, 10, 20],
                wallTimes: [1234, 1235, 1236],
                // Nonfinite values arrive as strings; see http_api.md.
                values: [0.5, 'NaN', '-Infinity'],
              },
              run2: EMPTY_COLUMNS,
            },
          },
        ] as unknown as BackendTimeSeriesResponse[]);

      expect(resultSpy).toHaveBeenCalledWith([
        {
          plugin: PluginType.SCALARS,
          tag: 'tag1',
          runToSeries: {
            'exp1/run1': [
              {wallTime: 1234, step: 0, value: 0.5},
              {wallTime: 1235, step: 10, value: 'NaN'},
              {wallTime: 1236, step: 20, value: '-Infinity'},
            ],
            'exp1/run2': [],
          },
        },
      ]);
    });

    it('batches multiple card requests for the same experiment', () => {
      const resultSpy = jasmine.createSpy();
      dataSource
        .fetchTimeSeries([
          {
            plugin: PluginType.SCALARS,
            tag: 'tag1',
            experimentIds: ['exp1'],
          },
          {
            plugin: PluginType.SCALARS,
            tag: 'tag2',
            experimentIds: ['exp1'],
          },
        ])
        .subscribe(resultSpy);

      const request = httpMock.expectOne(
        '/experiment/exp1/data/plugin/timeseries/timeSeries'
      );
      const body = request.request.body as FormData;
      const parsedRequests = JSON.parse(
        body.get('requests') as string
      ) as BackendTimeSeriesRequest[];
      expect(parsedRequests).toEqual([
        {plugin: PluginType.SCALARS, tag: 'tag1'},
        {plugin: PluginType.SCALARS, tag: 'tag2'},
      ]);
      request.flush([
        {
          plugin: PluginType.SCALARS,
          tag: 'tag1',
          runToSeries: {run1: EMPTY_COLUMNS},
        },
        {
          plugin: PluginType.SCALARS,
          tag: 'tag2',
          runToSeries: {run2: EMPTY_COLUMNS},
        },
      ] as BackendTimeSeriesResponse[]);

      expect(resultSpy).toHaveBeenCalledWith([
        {
          plugin: PluginType.SCALARS,
          tag: 'tag1',
          runToSeries: {'exp1/run1': []},
        },
        {
          plugin: PluginType.SCALARS,
          tag: 'tag2',
          runToSeries: {'exp1/run2': []},
        },
      ]);
    });

    it('makes requests per experiment id', () => {
      const resultSpy = jasmine.createSpy();
      dataSource
        .fetchTimeSeries([
          {
            plugin: PluginType.SCALARS,
            tag: 'tag1',
            experimentIds: ['exp1', 'exp2'],
          },
        ])
        .subscribe(resultSpy);

      const req1 = httpMock.expectOne(
        '/experiment/exp1/data/plugin/timeseries/timeSeries'
      );
      req1.flush([
        {
          plugin: PluginType.SCALARS,
          tag: 'tag1',
          runToSeries: {run1: EMPTY_COLUMNS},
        },
      ] as BackendTimeSeriesResponse[]);

      const req2 = httpMock.expectOne(
        '/experiment/exp2/data/plugin/timeseries/timeSeries'
      );
      req2.flush([
        {
          plugin: PluginType.SCALARS,
          tag: 'tag1',
          runToSeries: {run1: EMPTY_COLUMNS},
        },
      ] as BackendTimeSeriesResponse[]);

      expect(resultSpy).toHaveBeenCalledWith([
        {
          plugin: PluginType.SCALARS,
          tag: 'tag1',
          runToSeries: {'exp1/run1': [], 'exp2/run1': []},
        },
      ]);
    });

    it('sends selected run names and returns only runs with data', () => {
      const resultSpy = jasmine.createSpy();
      dataSource
        .fetchTimeSeries([
          {
            plugin: PluginType.SCALARS,
            tag: 'tag1',
            experimentIds: ['exp1'],
            runIds: ['exp1/run1', 'exp1/run2'],
          },
        ])
        .subscribe(resultSpy);

      const req = httpMock.expectOne(
        '/experiment/exp1/data/plugin/timeseries/timeSeries'
      );
      const body = req.request.body as FormData;
      const parsedRequests = JSON.parse(
        body.get('requests') as string
      ) as BackendTimeSeriesRequest[];
      expect(parsedRequests).toEqual([
        {
          plugin: PluginType.SCALARS,
          tag: 'tag1',
          runs: ['run1', 'run2'],
        },
      ]);
      req.flush([
        {
          plugin: PluginType.SCALARS,
          tag: 'tag1',
          runToSeries: {run1: EMPTY_COLUMNS},
        },
      ] as BackendTimeSeriesResponse[]);

      expect(resultSpy).toHaveBeenCalledWith([
        {
          plugin: PluginType.SCALARS,
          tag: 'tag1',
          runToSeries: {'exp1/run1': []},
        },
      ]);
    });

    it('requests each experiment with only its own selected runs', () => {
      const resultSpy = jasmine.createSpy();
      dataSource
        .fetchTimeSeries([
          {
            plugin: PluginType.SCALARS,
            tag: 'tag1',
            experimentIds: ['exp1', 'exp2'],
            runIds: ['exp2/run1'],
          },
        ])
        .subscribe(resultSpy);

      httpMock.expectNone('/experiment/exp1/data/plugin/timeseries/timeSeries');
      const req = httpMock.expectOne(
        '/experiment/exp2/data/plugin/timeseries/timeSeries'
      );
      const body = req.request.body as FormData;
      const parsedRequests = JSON.parse(
        body.get('requests') as string
      ) as BackendTimeSeriesRequest[];
      expect(parsedRequests).toEqual([
        {
          plugin: PluginType.SCALARS,
          tag: 'tag1',
          runs: ['run1'],
        },
      ]);
      req.flush([
        {
          plugin: PluginType.SCALARS,
          tag: 'tag1',
          runToSeries: {run1: EMPTY_COLUMNS},
        },
      ] as BackendTimeSeriesResponse[]);

      expect(resultSpy).toHaveBeenCalledWith([
        {
          plugin: PluginType.SCALARS,
          tag: 'tag1',
          runToSeries: {'exp2/run1': []},
        },
      ]);
    });

    it('answers without fetching when selected run ids are empty', () => {
      const resultSpy = jasmine.createSpy();
      dataSource
        .fetchTimeSeries([
          {
            plugin: PluginType.SCALARS,
            tag: 'tag1',
            experimentIds: ['exp1'],
            runIds: [],
          },
        ])
        .subscribe(resultSpy);

      httpMock.expectNone('/experiment/exp1/data/plugin/timeseries/timeSeries');
      // Every request answers exactly once; otherwise its runs would stay in
      // the loading state forever.
      expect(resultSpy).toHaveBeenCalledWith([
        {
          plugin: PluginType.SCALARS,
          tag: 'tag1',
          runToSeries: {},
        },
      ]);
    });

    it('drops series data if one experiment had an error', () => {
      const resultSpy = jasmine.createSpy();
      dataSource
        .fetchTimeSeries([
          {
            plugin: PluginType.SCALARS,
            tag: 'tag1',
            experimentIds: ['exp1', 'exp2'],
          },
        ])
        .subscribe(resultSpy);

      const req1 = httpMock.expectOne(
        '/experiment/exp1/data/plugin/timeseries/timeSeries'
      );
      req1.flush([
        {
          plugin: PluginType.SCALARS,
          tag: 'tag1',
          error: 'Something bad happened',
        },
      ] as BackendTimeSeriesResponse[]);

      const req2 = httpMock.expectOne(
        '/experiment/exp2/data/plugin/timeseries/timeSeries'
      );
      req2.flush([
        {
          plugin: PluginType.SCALARS,
          tag: 'tag1',
          runToSeries: {run1: EMPTY_COLUMNS},
        },
      ] as BackendTimeSeriesResponse[]);

      expect(resultSpy).toHaveBeenCalledWith([
        {
          plugin: PluginType.SCALARS,
          tag: 'tag1',
          error: 'Something bad happened',
          runToSeries: undefined,
        },
      ]);
    });

    it('makes single-run requests', () => {
      const resultSpy = jasmine.createSpy();
      dataSource
        .fetchTimeSeries([
          {
            plugin: PluginType.HISTOGRAMS,
            tag: 'tag1',
            runId: 'exp1/run1',
          },
        ])
        .subscribe(resultSpy);

      const req1 = httpMock.expectOne(
        '/experiment/exp1/data/plugin/timeseries/timeSeries'
      );
      req1.flush([
        {
          plugin: PluginType.HISTOGRAMS,
          tag: 'tag1',
          runToSeries: {run1: []},
        },
      ] as BackendTimeSeriesResponse[]);

      expect(resultSpy).toHaveBeenCalledWith([
        {
          plugin: PluginType.HISTOGRAMS,
          tag: 'tag1',
          runToSeries: {'exp1/run1': []},
        },
      ]);
    });
  });

  describe('#downloadUrl', () => {
    it('forms correct Url', () => {
      expect(
        dataSource.downloadUrl(PluginType.SCALARS, 'tag1', 'exp1/run1', 'json')
      ).toBe(
        '/experiment/exp1/data/plugin/scalars/scalars?tag=tag1&run=run1&format=json'
      );
      expect(
        dataSource.downloadUrl(PluginType.SCALARS, 'tag1', 'exp1/run1', 'csv')
      ).toBe(
        '/experiment/exp1/data/plugin/scalars/scalars?tag=tag1&run=run1&format=csv'
      );
      expect(
        dataSource.downloadUrl(PluginType.SCALARS, 'tag1', 'e2/run1', 'json')
      ).toBe(
        '/experiment/e2/data/plugin/scalars/scalars?tag=tag1&run=run1&format=json'
      );
      expect(
        dataSource.downloadUrl(PluginType.SCALARS, 'tag1', 'e2/run1', 'csv')
      ).toBe(
        '/experiment/e2/data/plugin/scalars/scalars?tag=tag1&run=run1&format=csv'
      );
    });

    it('throws for histogram data type', () => {
      expect(() =>
        dataSource.downloadUrl(PluginType.HISTOGRAMS, 'tag1', 'e/r', 'json')
      ).toThrowError(/Not implemented/);
    });

    it('throws when experiment id is missing', () => {
      expect(() =>
        dataSource.downloadUrl(PluginType.SCALARS, 'tag1', 'run1', 'json')
      ).toThrowError(/experimentId is empty/);
    });
  });
});
