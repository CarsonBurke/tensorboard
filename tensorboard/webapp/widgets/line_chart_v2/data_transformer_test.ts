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

import {classicSmoothing} from './data_transformer';
import {buildSeries} from './lib/testing';

import {smoothPackedValues} from './smoothing_kernel';
import {SmoothingWorkerClient} from './smoothing_worker_client';

describe('line_chart_v2/data_transformer test', () => {
  it('reuses immutable smoothing results and preserves point metadata', async () => {
    const points = [
      {x: 1, y: 2, wallTime: 99},
      {x: 2, y: 5, wallTime: 100},
    ];
    const first = await classicSmoothing([{id: 'run', points}], 0.6);
    const pinned = await classicSmoothing([{id: 'pinned', points}], 0.6);
    expect(pinned[0].points).toBe(first[0].points);
    expect(pinned[0].points[0].wallTime).toBe(99);
    const [low, high] = await Promise.all([
      classicSmoothing([{id: 'run', points}], 0.2),
      classicSmoothing([{id: 'run', points}], 0.9),
    ]);
    expect(low[0].points[1].y).not.toBe(high[0].points[1].y);
  });

  it('transfers worker buffers, cancels stale jobs, and recovers from worker failure', async () => {
    const listeners = new Map<string, (event: any) => void>();
    const messages: any[] = [];
    const fake = {
      addEventListener: (name: string, listener: (event: any) => void) =>
        listeners.set(name, listener),
      postMessage: (message: any, transfer: Transferable[]) => {
        if (!message.cancel) expect(transfer).toEqual([message.buffer]);
        messages.push(message);
      },
      terminate: jasmine.createSpy('terminate'),
    } as unknown as Worker;
    const client = new SmoothingWorkerClient(() => fake);
    const makeValues = () =>
      Float64Array.from({length: 8192}, (_, i) => i % 13);
    const first = client.smooth(makeValues, [8192], 0.6);
    const request = messages[0];
    await smoothPackedValues(
      new Float64Array(request.buffer),
      request.lengths,
      request.weight
    );
    listeners.get('message')!({data: {id: request.id, buffer: request.buffer}});
    const expected = await first;
    const controller = new AbortController();
    const stale = client.smooth(makeValues, [8192], 0.2, controller.signal);
    controller.abort();
    await expectAsync(stale).toBeRejectedWithError('Smoothing cancelled');
    expect(messages[messages.length - 1].cancel).toBeTrue();
    const failed = client.smooth(makeValues, [8192], 0.6);
    listeners.get('error')!({});
    expect(await failed).toEqual(expected);
    expect(fake.terminate).toHaveBeenCalled();
  });

  it('stops the kernel at a cancellation checkpoint', async () => {
    let cancelled = false;
    const values = Float64Array.from({length: 20000}, (_, i) => i % 7);
    const originalTail = values.slice(8192);
    await expectAsync(
      smoothPackedValues(
        values,
        [values.length],
        0.6,
        () => cancelled,
        async () => {
          cancelled = true;
        }
      )
    ).toBeRejectedWithError('Smoothing cancelled');
    expect(values.slice(8192)).toEqual(originalTail);
  });

  describe('#classicSmoothing', () => {
    it('smoothes data series', async () => {
      const dataSeries = [
        buildSeries({
          id: 's1',
          points: [
            {x: 0, y: 1},
            {x: 1, y: 0.5},
            {x: 2, y: 0},
          ],
        }),
        buildSeries({
          id: 's2',
          points: [
            {x: 0, y: 0},
            {x: 1, y: 0.5},
            {x: 2, y: 1},
          ],
        }),
        buildSeries({
          id: 's3',
          points: [],
        }),
      ];
      const actual = await classicSmoothing(dataSeries, 0.6);
      expect(actual).toEqual([
        {
          id: 's1',
          points: [
            // 0.4 * 1 / (1 - 0.6^1) = 1
            {x: 0, y: 1},
            // (0.5 * 0.4 + 0.4 * 1 * 0.6) / (1 - 0.6^2) = 0.6875
            {x: 1, y: 0.6875},
            // ~ 0.33673
            {
              x: 2,
              y:
                ((0.5 * 0.4 + 0.4 * 1 * 0.6) * 0.6 + 0) /
                (1 - Math.pow(0.6, 3)),
            },
          ],
        },
        {
          id: 's2',
          points: [
            {x: 0, y: 0},
            // (0.5 * 0.4) / (1 - 0.6^2) = 0.3125
            {x: 1, y: 0.3125},
            // ~0.6633
            {x: 2, y: (1 * 0.4 + 0.5 * 0.4 * 0.6) / (1 - Math.pow(0.6, 3))},
          ],
        },
        {
          id: 's3',
          points: [],
        },
      ]);
    });

    it('does not smooth at all when weight is 0', async () => {
      const dataSeries = [
        buildSeries({
          id: 's1',
          points: [
            {x: 0, y: 1},
            {x: 1, y: 0.5},
            {x: 2, y: 0},
          ],
        }),
        buildSeries({
          id: 's2',
          points: [
            {x: 0, y: 0},
            {x: 1, y: 0.5},
            {x: 2, y: 1},
          ],
        }),
      ];
      const actual = await classicSmoothing(dataSeries, 0.0);
      expect(actual).toEqual([
        {
          id: 's1',
          points: [
            {x: 0, y: 1},
            {x: 1, y: 0.5},
            {x: 2, y: 0},
          ],
        },
        {
          id: 's2',
          points: [
            {x: 0, y: 0},
            {x: 1, y: 0.5},
            {x: 2, y: 1},
          ],
        },
      ]);
    });

    it('omits not finite values in smoothing', async () => {
      const actual = await classicSmoothing(
        [
          buildSeries({
            id: 's1',
            points: [
              {x: 0, y: -Infinity},
              {x: 0, y: 1},
              {x: 0.5, y: NaN},
              {x: 0.75, y: Infinity},
              {x: 1, y: 0.5},
            ],
          }),
        ],
        0.6
      );
      expect(actual).toEqual([
        {
          id: 's1',
          points: [
            {x: 0, y: -Infinity},
            {x: 0, y: 1},
            {x: 0.5, y: NaN},
            {x: 0.75, y: Infinity},
            // Please refer to the "smoothes data series" spec for details of this value.
            {x: 1, y: 0.6875},
          ],
        },
      ]);
    });

    it('returns 0 when smoothing weight is 1', async () => {
      const actual = await classicSmoothing(
        [
          buildSeries({
            id: 's1',
            points: [
              {x: 0, y: 1},
              {x: 1, y: 0.5},
              {x: 2, y: 0},
            ],
          }),
        ],
        1
      );
      expect(actual).toEqual([
        {
          id: 's1',
          points: [
            {x: 0, y: 0},
            {x: 1, y: 0},
            {x: 2, y: 0},
          ],
        },
      ]);
    });

    it('does not inject floating point noise when numbers are constant', async () => {
      const actual = await classicSmoothing(
        [
          buildSeries({
            id: 's1',
            points: [
              {x: 0, y: 0.3},
              {x: 1, y: 0.3},
              {x: 2, y: 0.3},
            ],
          }),
        ],
        0.1
      );
      expect(actual).toEqual([
        {
          id: 's1',
          points: [
            {x: 0, y: 0.3},
            {x: 1, y: 0.3},
            {x: 2, y: 0.3},
          ],
        },
      ]);
    });

    describe('smoothing weight clipping', () => {
      for (const smoothingWeight of [NaN, -1, -Infinity, Infinity]) {
        it(`clips smoothing weight=${smoothingWeight} to 0`, async () => {
          const actual = await classicSmoothing(
            [
              buildSeries({
                id: 's1',
                points: [
                  {x: 0, y: 1},
                  {x: 1, y: 0.5},
                ],
              }),
            ],
            smoothingWeight
          );
          expect(actual).toEqual([
            {
              id: 's1',
              points: [
                {x: 0, y: 1},
                {x: 1, y: 0.5},
              ],
            },
          ]);
        });
      }

      it('clips smoothing weight larger than 1 to 1', async () => {
        const actual = await classicSmoothing(
          [
            buildSeries({
              id: 's1',
              points: [
                {x: 0, y: 1},
                {x: 1, y: 0.5},
              ],
            }),
          ],
          2
        );
        expect(actual).toEqual([
          {
            id: 's1',
            points: [
              {x: 0, y: 0},
              {x: 1, y: 0},
            ],
          },
        ]);
      });
    });
  });
});
