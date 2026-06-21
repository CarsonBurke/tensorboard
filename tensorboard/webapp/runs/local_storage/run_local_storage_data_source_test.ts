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
import {SortingOrder} from '../../widgets/data_table/types';
import {Run} from '../types';
import {
  RunLocalStorageDataSource,
  TEST_ONLY,
} from './run_local_storage_data_source';

function createRun(id: string): Run {
  return {id, name: id, startTime: 0};
}

describe('RunLocalStorageDataSource', () => {
  let dataSource: RunLocalStorageDataSource;

  beforeEach(() => {
    dataSource = new RunLocalStorageDataSource();
    window.localStorage.removeItem(TEST_ONLY.RUN_LOCAL_STORAGE_KEY);
  });

  afterEach(() => {
    window.localStorage.removeItem(TEST_ONLY.RUN_LOCAL_STORAGE_KEY);
  });

  it('returns empty state for empty storage', () => {
    const state = dataSource.getState('namespace', [createRun('run1')]);

    expect(state.selection).toEqual(new Map());
    expect(state.colorOverrides).toEqual(new Map());
    expect(state.newestRunId).toBeUndefined();
  });

  it('returns empty state for malformed storage', () => {
    window.localStorage.setItem(TEST_ONLY.RUN_LOCAL_STORAGE_KEY, '{');

    const state = dataSource.getState('namespace', [createRun('run1')]);

    expect(state.selection).toEqual(new Map());
    expect(state.colorOverrides).toEqual(new Map());
  });

  it('prunes stored runs that are not in the current run directory', () => {
    window.localStorage.setItem(
      TEST_ONLY.RUN_LOCAL_STORAGE_KEY,
      JSON.stringify({
        version: 1,
        namespaces: {
          namespace: {
            updatedAtMs: 1,
            runIds: ['run1', 'deleted'],
            selection: {run1: true, deleted: false},
            colorOverrides: {run1: '#fff', deleted: '#000'},
            newestRunId: 'deleted',
          },
        },
      })
    );

    const state = dataSource.getState('namespace', [createRun('run1')]);

    expect(state.selection).toEqual(new Map([['run1', true]]));
    expect(state.colorOverrides).toEqual(new Map([['run1', '#fff']]));
    expect(state.newestRunId).toBeUndefined();
  });

  it('writes only the active namespace and current runs', () => {
    window.localStorage.setItem(
      TEST_ONLY.RUN_LOCAL_STORAGE_KEY,
      JSON.stringify({
        version: 1,
        namespaces: {
          oldNamespace: {
            updatedAtMs: 1,
            runIds: ['oldRun'],
            selection: {oldRun: true},
            colorOverrides: {oldRun: '#abc'},
          },
        },
      })
    );

    dataSource.setState('namespace', [createRun('run1')], {
      selection: new Map([
        ['run1', false],
        ['deleted', true],
      ]),
      colorOverrides: new Map([
        ['run1', '#fff'],
        ['deleted', '#000'],
      ]),
      newestRunId: 'run1',
    });

    const serialized = window.localStorage.getItem(
      TEST_ONLY.RUN_LOCAL_STORAGE_KEY
    );
    expect(JSON.parse(serialized!) as object).toEqual({
      version: 1,
      namespaces: {
        namespace: {
          updatedAtMs: jasmine.any(Number),
          runIds: ['run1'],
          selection: {run1: false},
          colorOverrides: {run1: '#fff'},
          newestRunId: 'run1',
        },
      },
    });
  });

  it('persists runs with object prototype names', () => {
    dataSource.setState('namespace', [createRun('__proto__')], {
      selection: new Map([['__proto__', true]]),
      colorOverrides: new Map([['__proto__', '#fff']]),
    });

    const serialized = window.localStorage.getItem(
      TEST_ONLY.RUN_LOCAL_STORAGE_KEY
    );
    expect(serialized).toContain('"__proto__"');
    expect(dataSource.getState('namespace', [createRun('__proto__')])).toEqual({
      selection: new Map([['__proto__', true]]),
      colorOverrides: new Map([['__proto__', '#fff']]),
    });
  });

  it('does not rewrite when active namespace state is unchanged', () => {
    window.localStorage.setItem(
      TEST_ONLY.RUN_LOCAL_STORAGE_KEY,
      JSON.stringify({
        version: 1,
        namespaces: {
          namespace: {
            updatedAtMs: 1,
            runIds: ['run1'],
            selection: {run1: true},
            colorOverrides: {run1: '#fff'},
            newestRunId: 'run1',
          },
        },
      })
    );
    const setItemSpy = spyOn(window.localStorage, 'setItem').and.callThrough();

    dataSource.setState('namespace', [createRun('run1')], {
      selection: new Map([['run1', true]]),
      colorOverrides: new Map([['run1', '#fff']]),
      newestRunId: 'run1',
    });

    expect(setItemSpy).not.toHaveBeenCalled();
  });

  it('removes storage when there are no current runs', () => {
    window.localStorage.setItem(TEST_ONLY.RUN_LOCAL_STORAGE_KEY, '{}');

    dataSource.setState('namespace', [], {
      selection: new Map([['run1', true]]),
      colorOverrides: new Map([['run1', '#fff']]),
    });

    expect(window.localStorage.getItem(TEST_ONLY.RUN_LOCAL_STORAGE_KEY)).toBe(
      null
    );
  });

  it('persists and restores the run sorting selection', () => {
    dataSource.setState('namespace', [createRun('run1')], {
      selection: new Map([['run1', true]]),
      colorOverrides: new Map(),
      sortingInfo: {name: '\0runStartTime', order: SortingOrder.DESCENDING},
    });

    const state = dataSource.getState('namespace', [createRun('run1')]);

    expect(state.sortingInfo).toEqual({
      name: '\0runStartTime',
      order: SortingOrder.DESCENDING,
    });
  });

  it('ignores malformed stored sorting selection', () => {
    window.localStorage.setItem(
      TEST_ONLY.RUN_LOCAL_STORAGE_KEY,
      JSON.stringify({
        version: 1,
        namespaces: {
          namespace: {
            updatedAtMs: 1,
            runIds: ['run1'],
            selection: {run1: true},
            colorOverrides: {},
            sortingInfo: {name: 'run', order: 5},
          },
        },
      })
    );

    const state = dataSource.getState('namespace', [createRun('run1')]);

    expect(state.sortingInfo).toBeUndefined();
  });

  it('ignores invalid hex colors', () => {
    window.localStorage.setItem(
      TEST_ONLY.RUN_LOCAL_STORAGE_KEY,
      JSON.stringify({
        version: 1,
        namespaces: {
          namespace: {
            updatedAtMs: 1,
            runIds: ['short', 'bad5', 'bad7', 'long'],
            selection: {},
            colorOverrides: {
              short: '#abc',
              bad5: '#12345',
              bad7: '#1234567',
              long: '#12345678',
            },
          },
        },
      })
    );

    const state = dataSource.getState('namespace', [
      createRun('short'),
      createRun('bad5'),
      createRun('bad7'),
      createRun('long'),
    ]);

    expect(state.colorOverrides).toEqual(
      new Map([
        ['short', '#abc'],
        ['long', '#12345678'],
      ])
    );
  });
});
