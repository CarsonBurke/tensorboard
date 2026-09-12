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
import {
  DataSeries,
  DataSeriesMetadataMap,
  RendererType,
} from './lib/public_types';
import {ChartUtils} from './lib/utils';

/**
 * Scratch space for the outlier percentiles, reused across calls: this runs
 * for every chart whenever a series is toggled, and a growable boxed array
 * cost more than the selection itself.
 */
let yValues = new Float64Array(0);

/**
 * Capacity above which the scratch space is released instead of retained, so
 * one enormous card does not hold the buffer for the rest of the session.
 */
const MAX_RETAINED_Y_VALUES = 1 << 20;

/**
 * Reorders `values[from..to]` so that index `nth` holds the value a full
 * ascending sort would put there, and returns it.
 *
 * Hoare partitioning around the middle element. Only two order statistics
 * are needed, so sorting would pay O(n log n) for two values; this is O(n)
 * expected. On return the range is partitioned about `nth`, which lets the
 * caller narrow the range for a subsequent, larger `nth`.
 */
function selectNth(
  values: Float64Array,
  nth: number,
  from: number,
  to: number
): number {
  let left = from;
  let right = to;
  while (left < right) {
    const pivot = values[(left + right) >> 1];
    let low = left;
    let high = right;
    while (low <= high) {
      while (values[low] < pivot) low++;
      while (values[high] > pivot) high--;
      if (low <= high) {
        const value = values[low];
        values[low] = values[high];
        values[high] = value;
        low++;
        high--;
      }
    }
    if (nth <= high) {
      right = high;
    } else if (nth >= low) {
      left = low;
    } else {
      // `nth` sits in the run of values equal to the pivot.
      break;
    }
  }
  return values[nth];
}

/**
 * Returns extent, min and max values of each dimensions, of all data series points.
 *
 * When ignoreYOutliers is true, it will calculate extent using values within 5th and 95th
 * quantiles.
 *
 * Note that it excludes auxillary data points and invisible data series.
 */
export function computeDataSeriesExtent(
  data: DataSeries[],
  metadataMap: DataSeriesMetadataMap,
  ignoreYOutliers: boolean,
  isXSafeNumber: (x: number) => boolean,
  isYSafeNumber: (x: number) => boolean
): {x: [number, number] | undefined; y: [number, number] | undefined} {
  let xMin: number | null = null;
  let xMax: number | null = null;
  let yMin: number | undefined;
  let yMax: number | undefined;
  let yCount = 0;

  for (const {id, points} of data) {
    const meta = metadataMap[id];
    if (!meta || meta.aux || !meta.visible) continue;

    for (let index = 0; index < points.length; index++) {
      const {x, y} = points[index];
      if (isXSafeNumber(x)) {
        xMin = xMin === null || x < xMin ? x : xMin;
        xMax = xMax === null || x > xMax ? x : xMax;
      }
      if (isYSafeNumber(y)) {
        if (ignoreYOutliers) {
          if (yCount === yValues.length) {
            const grown = new Float64Array(Math.max(1024, yCount * 2));
            grown.set(yValues);
            yValues = grown;
          }
          yValues[yCount++] = y;
        }
        yMin = yMin === undefined || y < yMin ? y : yMin;
        // Keep the last equal maximum, as the stable sort does (including -0).
        yMax = yMax === undefined || y >= yMax ? y : yMax;
      }
    }
  }

  if (ignoreYOutliers && yCount > 2) {
    const lowNth = Math.ceil((yCount - 1) * 0.05);
    const highNth = Math.floor((yCount - 1) * 0.95);
    yMin = selectNth(yValues, lowNth, 0, yCount - 1);
    // Selecting `lowNth` partitioned the array about it, so the larger
    // percentile cannot lie below it.
    yMax = selectNth(yValues, highNth, lowNth, yCount - 1);
  }
  if (yValues.length > MAX_RETAINED_Y_VALUES) {
    yValues = new Float64Array(0);
  }

  return {
    x: xMin !== null && xMax !== null ? [xMin, xMax] : undefined,
    y: yMin !== undefined && yMax !== undefined ? [yMin, yMax] : undefined,
  };
}

export function getRendererType(
  preferredRendererType: RendererType
): RendererType {
  switch (preferredRendererType) {
    case RendererType.SVG:
      return RendererType.SVG;
    case RendererType.WEBGL:
      return ChartUtils.isWebGl2Supported()
        ? RendererType.WEBGL
        : RendererType.SVG;
    default:
      const _ = preferredRendererType as never;
      throw new Error(`Unknown rendererType: ${preferredRendererType}`);
  }
}
