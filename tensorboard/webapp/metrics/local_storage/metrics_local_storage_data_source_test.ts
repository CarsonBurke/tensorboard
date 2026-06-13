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
    });
  });

  it('persists groups with object prototype names', () => {
    dataSource.setState('namespace1', ['__proto__'], {
      tagGroupExpanded: new Map([['__proto__', true]]),
      tagGroupPageIndex: new Map([['__proto__', 2]]),
    });

    const serialized = window.localStorage.getItem(
      TEST_ONLY.METRICS_LOCAL_STORAGE_KEY
    );
    expect(serialized).toContain('"__proto__"');
    expect(dataSource.getState('namespace1', ['__proto__'])).toEqual({
      tagGroupExpanded: new Map([['__proto__', true]]),
      tagGroupPageIndex: new Map([['__proto__', 2]]),
    });
  });

  it('prunes stale groups and inactive namespaces on sync', () => {
    window.localStorage.setItem(
      TEST_ONLY.METRICS_LOCAL_STORAGE_KEY,
      JSON.stringify({
        version: 1,
        namespaces: {
          namespace1: {
            updatedAtMs: 1,
            tagGroups: ['foo', 'old'],
            tagGroupExpanded: {foo: true, old: true},
            tagGroupPageIndex: {foo: 3, old: 9},
          },
          namespace2: {
            updatedAtMs: 1,
            tagGroups: ['other'],
            tagGroupExpanded: {other: true},
            tagGroupPageIndex: {other: 4},
          },
        },
      })
    );

    const state = dataSource.getState('namespace1', ['foo']);
    expect(state).toEqual({
      tagGroupExpanded: new Map([['foo', true]]),
      tagGroupPageIndex: new Map([['foo', 3]]),
    });

    dataSource.setState('namespace1', ['foo'], state);

    const stored = JSON.parse(
      window.localStorage.getItem(TEST_ONLY.METRICS_LOCAL_STORAGE_KEY)!
    ) as {
      namespaces: Record<
        string,
        {
          tagGroupExpanded: Record<string, boolean>;
          tagGroupPageIndex: Record<string, number>;
        }
      >;
    };
    expect(Object.keys(stored.namespaces)).toEqual(['namespace1']);
    expect(stored.namespaces.namespace1.tagGroupExpanded).toEqual({foo: true});
    expect(stored.namespaces.namespace1.tagGroupPageIndex).toEqual({foo: 3});
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
          },
        },
      })
    );
    const setItemSpy = spyOn(window.localStorage, 'setItem').and.callThrough();

    dataSource.setState('namespace1', ['foo'], {
      tagGroupExpanded: new Map([['foo', true]]),
      tagGroupPageIndex: new Map([['foo', 2]]),
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
    });
  });

  it('does not write invalid page indices', () => {
    dataSource.setState('namespace1', ['foo', 'bar'], {
      tagGroupExpanded: new Map(),
      tagGroupPageIndex: new Map([
        ['foo', 1.5],
        ['bar', -1],
      ]),
    });

    expect(dataSource.getState('namespace1', ['foo', 'bar'])).toEqual({
      tagGroupExpanded: new Map(),
      tagGroupPageIndex: new Map(),
    });
  });
});
