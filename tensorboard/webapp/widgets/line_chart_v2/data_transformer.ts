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

import {DataSeries} from './lib/public_types';
import {yieldToEventLoop} from './smoothing_kernel';
import {smoothOffThread} from './smoothing_worker_client';

// Cache only the most recent weight. Weak keys do not retain removed series.
const smoothedPoints = new WeakMap<
  DataSeries['points'],
  {
    weight: number;
    points: DataSeries['points'];
  }
>();

/** Classical TensorBoard EMA, preserving point metadata and immutable inputs. */
export async function classicSmoothing<T extends DataSeries>(
  data: T[],
  smoothingWeight: number,
  signal?: AbortSignal
): Promise<T[]> {
  // Coalesce synchronous setting emissions before packing/transferring points.
  await Promise.resolve();
  if (signal?.aborted) throw new Error('Smoothing cancelled');
  let weight = Number.isFinite(smoothingWeight) ? smoothingWeight : 0;
  weight = Math.max(0, Math.min(weight, 1));
  const results = new Map<DataSeries['points'], DataSeries['points']>();
  const missing = data.filter(({points}) => {
    const cached = smoothedPoints.get(points);
    if (cached?.weight !== weight) return true;
    results.set(points, cached.points);
    return false;
  });
  if (missing.length) {
    const lengths = missing.map(({points}) => points.length);
    const makeValues = () => {
      const values = new Float64Array(
        lengths.reduce((sum, length) => sum + length, 0)
      );
      let index = 0;
      for (const {points} of missing)
        for (const point of points) values[index++] = point.y;
      return values;
    };
    const values = await smoothOffThread(makeValues, lengths, weight, signal);
    if (signal?.aborted) throw new Error('Smoothing cancelled');
    let offset = 0;
    let sinceYield = 0;
    for (const {points} of missing) {
      const original = points;
      const initial = points[0]?.y;
      const constant = points.every((point) => point.y === initial);
      const result = constant
        ? original
        : points.map((point, i) => ({...point, y: values[offset + i]}));
      smoothedPoints.set(original, {weight, points: result});
      results.set(original, result);
      offset += points.length;
      sinceYield += points.length;
      if (sinceYield >= 8192 && offset < values.length) {
        sinceYield = 0;
        await yieldToEventLoop();
        if (signal?.aborted) throw new Error('Smoothing cancelled');
      }
    }
  }
  // Another chart can finish a different weight concurrently; retain results
  // belonging to this invocation, not a later cache writer's weight.
  return data.map((series) => {
    const points = results.get(series.points)!;
    return points === series.points ? series : {...series, points};
  });
}
