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
import {
  MetricsLocalStorageDataSource,
  PersistedCardState,
  TEST_ONLY,
} from './metrics_local_storage_data_source';

describe('metrics_local_storage_data_source', () => {
  let dataSource: MetricsLocalStorageDataSource;

  beforeEach(() => {
    window.localStorage.clear();
    dataSource = new MetricsLocalStorageDataSource();
  });

  afterEach(() => {
    window.localStorage.clear();
  });

  it('persists and restores group expansion and page state', () => {
    dataSource.setState('namespace1', ['foo', 'bar'], {
      tagGroupExpanded: new Map([
        ['foo', true],
        ['bar', false],
      ]),
      tagGroupPageIndex: new Map([
        ['foo', 2],
        ['bar', 0],
      ]),
      cardState: new Map(),
    });

    expect(dataSource.getState('namespace1', ['foo', 'bar'])).toEqual({
      tagGroupExpanded: new Map([
        ['foo', true],
        ['bar', false],
      ]),
      tagGroupPageIndex: new Map([
        ['foo', 2],
        ['bar', 0],
      ]),
      cardState: new Map(),
    });
  });

  it('persists groups with object prototype names', () => {
    dataSource.setState('namespace1', ['__proto__'], {
      tagGroupExpanded: new Map([['__proto__', true]]),
      tagGroupPageIndex: new Map([['__proto__', 2]]),
      cardState: new Map(),
    });

    const serialized = window.localStorage.getItem(
      TEST_ONLY.METRICS_LOCAL_STORAGE_KEY
    );
    expect(serialized).toContain('"__proto__"');
    expect(dataSource.getState('namespace1', ['__proto__'])).toEqual({
      tagGroupExpanded: new Map([['__proto__', true]]),
      tagGroupPageIndex: new Map([['__proto__', 2]]),
      cardState: new Map(),
    });
  });

  it('preserves configured categories outside the current catalog window', () => {
    const configured = {
      tagGroupExpanded: new Map([
        ['foo', false],
        ['offscreen', true],
      ]),
      tagGroupPageIndex: new Map([
        ['foo', 3],
        ['offscreen', 9],
      ]),
      cardState: new Map<string, PersistedCardState>(),
    };
    dataSource.setState('namespace1', ['foo', 'offscreen'], configured);
    const offscreenState = new MetricsLocalStorageDataSource().getState(
      'namespace1',
      ['foo']
    );
    expect(offscreenState).toEqual(configured);
    dataSource.setState('namespace1', [], offscreenState);
    expect(
      new MetricsLocalStorageDataSource().getState('namespace1', [])
    ).toEqual(configured);
  });

  it('does not rewrite when active namespace state is unchanged', () => {
    window.localStorage.setItem(
      TEST_ONLY.METRICS_LOCAL_STORAGE_KEY,
      JSON.stringify({
        version: 1,
        namespaces: {
          namespace1: {
            updatedAtMs: 1,
            tagGroups: ['foo'],
            tagGroupExpanded: {foo: true},
            tagGroupPageIndex: {foo: 2},
            cardState: {card1: {fullWidth: true, chartHeight: 300}},
          },
        },
      })
    );
    const setItemSpy = spyOn(window.localStorage, 'setItem').and.callThrough();

    dataSource.setState('namespace1', ['foo'], {
      tagGroupExpanded: new Map([['foo', true]]),
      tagGroupPageIndex: new Map([['foo', 2]]),
      cardState: new Map([['card1', {fullWidth: true, chartHeight: 300}]]),
    });

    expect(setItemSpy).not.toHaveBeenCalled();
  });

  it('ignores malformed stored values', () => {
    window.localStorage.setItem(
      TEST_ONLY.METRICS_LOCAL_STORAGE_KEY,
      JSON.stringify({
        version: 1,
        namespaces: {
          namespace1: {
            updatedAtMs: 1,
            tagGroups: ['foo'],
            tagGroupExpanded: {foo: 'yes'},
            tagGroupPageIndex: {foo: -1, bar: 1.5},
          },
        },
      })
    );

    expect(dataSource.getState('namespace1', ['foo'])).toEqual({
      tagGroupExpanded: new Map(),
      tagGroupPageIndex: new Map(),
      cardState: new Map(),
    });
  });

  it('does not write invalid page indices', () => {
    dataSource.setState('namespace1', ['foo', 'bar'], {
      tagGroupExpanded: new Map(),
      tagGroupPageIndex: new Map([
        ['foo', 1.5],
        ['bar', -1],
      ]),
      cardState: new Map(),
    });

    expect(dataSource.getState('namespace1', ['foo', 'bar'])).toEqual({
      tagGroupExpanded: new Map(),
      tagGroupPageIndex: new Map(),
      cardState: new Map(),
    });
  });

  it('removes storage when there are no current tag groups', () => {
    window.localStorage.setItem(TEST_ONLY.METRICS_LOCAL_STORAGE_KEY, '{}');

    dataSource.setState('namespace1', [], {
      tagGroupExpanded: new Map(),
      tagGroupPageIndex: new Map(),
      cardState: new Map(),
    });

    expect(
      window.localStorage.getItem(TEST_ONLY.METRICS_LOCAL_STORAGE_KEY)
    ).toBeNull();
  });

  it('prefers the latest session state after a failed write', () => {
    window.localStorage.setItem(
      TEST_ONLY.METRICS_LOCAL_STORAGE_KEY,
      JSON.stringify({
        version: 1,
        namespaces: {
          namespace1: {
            updatedAtMs: 1,
            tagGroups: ['foo'],
            tagGroupExpanded: {foo: true},
            tagGroupPageIndex: {foo: 1},
          },
        },
      })
    );
    const setItemSpy = spyOn(window.localStorage, 'setItem').and.throwError(
      'quota exceeded'
    );

    dataSource.setState('namespace1', ['foo'], {
      tagGroupExpanded: new Map([['foo', false]]),
      tagGroupPageIndex: new Map([['foo', 2]]),
      cardState: new Map(),
    });

    expect(dataSource.getState('namespace1', ['foo'])).toEqual({
      tagGroupExpanded: new Map([['foo', false]]),
      tagGroupPageIndex: new Map([['foo', 2]]),
      cardState: new Map(),
    });

    setItemSpy.and.callThrough();
    dataSource.setState('namespace1', ['foo'], {
      tagGroupExpanded: new Map([['foo', false]]),
      tagGroupPageIndex: new Map([['foo', 2]]),
      cardState: new Map(),
    });

    expect(setItemSpy).toHaveBeenCalledTimes(2);
    expect(
      window.localStorage.getItem(TEST_ONLY.METRICS_LOCAL_STORAGE_KEY)
    ).toContain('"foo":false');
  });

  it('does not read stale state after a failed removal', () => {
    window.localStorage.setItem(
      TEST_ONLY.METRICS_LOCAL_STORAGE_KEY,
      JSON.stringify({
        version: 1,
        namespaces: {
          namespace1: {
            updatedAtMs: 1,
            tagGroups: ['foo'],
            tagGroupExpanded: {foo: true},
            tagGroupPageIndex: {foo: 1},
          },
        },
      })
    );
    spyOn(window.localStorage, 'removeItem').and.throwError('unavailable');

    dataSource.setState('namespace1', [], {
      tagGroupExpanded: new Map(),
      tagGroupPageIndex: new Map(),
      cardState: new Map(),
    });

    expect(dataSource.getState('namespace1', ['foo'])).toEqual({
      tagGroupExpanded: new Map(),
      tagGroupPageIndex: new Map(),
      cardState: new Map(),
    });
  });

  it('persists and restores card view state', () => {
    dataSource.setState('namespace1', ['foo'], {
      tagGroupExpanded: new Map(),
      tagGroupPageIndex: new Map(),
      cardState: new Map([
        ['card1', {fullWidth: true, chartHeight: 420}],
        ['card2', {tableExpanded: true, tableHeight: 240}],
      ]),
    });

    expect(
      new MetricsLocalStorageDataSource().getState('namespace1', ['foo'])
        .cardState
    ).toEqual(
      new Map([
        ['card1', {fullWidth: true, chartHeight: 420}],
        ['card2', {tableExpanded: true, tableHeight: 240}],
      ])
    );
  });

  it('keeps stored card state for cards the caller did not pass', () => {
    dataSource.setState('namespace1', ['foo'], {
      tagGroupExpanded: new Map(),
      tagGroupPageIndex: new Map(),
      cardState: new Map([
        ['card1', {fullWidth: true, chartHeight: 300}],
        ['offscreen', {tableExpanded: true, tableHeight: 250}],
      ]),
    });

    // The store prunes card state for cards outside the catalog window, so a
    // later sync only carries the cards that are currently live.
    dataSource.setState('namespace1', ['foo'], {
      tagGroupExpanded: new Map(),
      tagGroupPageIndex: new Map(),
      cardState: new Map([['card1', {chartHeight: 500}]]),
    });

    expect(
      new MetricsLocalStorageDataSource().getState('namespace1', ['foo'])
        .cardState
    ).toEqual(
      new Map([
        // Given keys win; keys that were not given survive.
        ['card1', {fullWidth: true, chartHeight: 500}],
        ['offscreen', {tableExpanded: true, tableHeight: 250}],
      ])
    );
  });

  it('writes when only the card state changed', () => {
    dataSource.setState('namespace1', ['foo'], {
      tagGroupExpanded: new Map([['foo', true]]),
      tagGroupPageIndex: new Map([['foo', 0]]),
      cardState: new Map([['card1', {fullWidth: true}]]),
    });

    dataSource.setState('namespace1', ['foo'], {
      tagGroupExpanded: new Map([['foo', true]]),
      tagGroupPageIndex: new Map([['foo', 0]]),
      cardState: new Map([['card1', {fullWidth: false}]]),
    });

    expect(
      new MetricsLocalStorageDataSource().getState('namespace1', ['foo'])
        .cardState
    ).toEqual(new Map([['card1', {fullWidth: false}]]));
  });

  it('drops card state values that are out of bounds or mistyped', () => {
    dataSource.setState('namespace1', ['foo'], {
      tagGroupExpanded: new Map(),
      tagGroupPageIndex: new Map(),
      cardState: new Map<string, PersistedCardState>([
        ['tooShort', {chartHeight: 39, tableHeight: 300}],
        ['tooTall', {chartHeight: 5001}],
        ['fractional', {chartHeight: 300.5}],
        ['notANumber', {tableHeight: Number.NaN}],
        // Values coming from storage are not type checked at runtime.
        ['notABoolean', {fullWidth: 'yes'} as unknown as PersistedCardState],
      ]),
    });

    expect(
      new MetricsLocalStorageDataSource().getState('namespace1', ['foo'])
        .cardState
    ).toEqual(new Map([['tooShort', {tableHeight: 300}]]));
  });

  it('drops mistyped card state values found in storage', () => {
    window.localStorage.setItem(
      TEST_ONLY.METRICS_LOCAL_STORAGE_KEY,
      JSON.stringify({
        version: 1,
        namespaces: {
          namespace1: {
            updatedAtMs: 1,
            tagGroups: ['foo'],
            tagGroupExpanded: {},
            tagGroupPageIndex: {},
            cardState: {
              card1: {
                fullWidth: 'yes',
                tableExpanded: true,
                chartHeight: '300',
                tableHeight: 1e9,
              },
              card2: {chartHeight: null},
              card3: 'nonsense',
            },
          },
        },
      })
    );

    expect(dataSource.getState('namespace1', ['foo']).cardState).toEqual(
      new Map([['card1', {tableExpanded: true}]])
    );
  });

  it('caps persisted cards, keeping the ones just passed in', () => {
    const cap = TEST_ONLY.MAX_PERSISTED_CARDS;
    const overflowing = new Map<string, PersistedCardState>();
    for (let i = 0; i < cap + 50; i++) {
      overflowing.set(`card${i}`, {chartHeight: 100 + i});
    }

    dataSource.setState('namespace1', ['foo'], {
      tagGroupExpanded: new Map(),
      tagGroupPageIndex: new Map(),
      cardState: overflowing,
    });

    const stored = new MetricsLocalStorageDataSource().getState('namespace1', [
      'foo',
    ]).cardState;
    expect(stored.size).toBe(cap);
    expect(stored.has('card0')).toBeTrue();
    expect(stored.has(`card${cap}`)).toBeFalse();

    dataSource.setState('namespace1', ['foo'], {
      tagGroupExpanded: new Map(),
      tagGroupPageIndex: new Map(),
      cardState: new Map([['fresh', {tableHeight: 200}]]),
    });

    const afterFresh = new MetricsLocalStorageDataSource().getState(
      'namespace1',
      ['foo']
    ).cardState;
    expect(afterFresh.size).toBe(cap);
    expect(afterFresh.get('fresh')).toEqual({tableHeight: 200});
    expect(afterFresh.has(`card${cap - 1}`)).toBeFalse();
  });
});
