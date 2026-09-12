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

import {Rect, Scale, ScaleType} from './internal_types';
import {createScale} from './scale';

/**
 * A stateful convenient utility around scale for converting coordinate systems.
 *
 * Definitions.
 *
 * Example for better illustration: we are viewing a diagonal line that goes from
 * <0, 0> -> <1, 2> onto a canvas with size <100, 200>.
 *
 * - data coordinate: coordinate in raw data space. For example above, you would have a
 *     line by connecting two points at <0, 0> and <1, 2>.
 * - ui coordinate: coordinate of a data in pixel/view-space. For example above, a data at
 *     <0.5, 0.5> will be on <50, 100> in UI coordinates.
 * - internal coordinate: in case like webgl, you can use an internal static
 *     coordinate system separate from the ui coordinate.
 * - view box: a rect in data coordinate that describes what should be visible
 *     on the screen.
 */
export class Coordinator {
  protected xScale: Scale = createScale(ScaleType.LINEAR);
  protected yScale: Scale = createScale(ScaleType.LINEAR);

  protected domContainerRect: Rect = {
    x: 0,
    width: 1,
    y: 0,
    height: 1,
  };

  private lastUpdated: number = 0;
  private currentViewBoxRect: Rect = {
    x: 0,
    width: 1,
    y: 0,
    height: 1,
  };

  getUpdateIdentifier() {
    return this.lastUpdated;
  }

  private updateIdentifier() {
    this.lastUpdated++;
  }

  /**
   * Returns whether y axis is pointing down in the output space.
   *
   * ↑
   * | isYAxisPointedDown = false (e.g., cartesian coordinates, 3d scene)
   * |
   * |-------------→
   * |
   * | isYAxisPointedDown = true (e.g., DOM)
   * ↓
   */
  isYAxisPointedDown(): boolean {
    return true;
  }

  setXScale(scale: Scale) {
    this.xScale = scale;
    this.updateIdentifier();
  }

  setYScale(scale: Scale) {
    this.yScale = scale;
    this.updateIdentifier();
  }

  getCurrentViewBoxRect(): Rect {
    return this.currentViewBoxRect;
  }

  setViewBoxRect(rectInDataCoordinate: Rect) {
    this.currentViewBoxRect = rectInDataCoordinate;
    this.updateIdentifier();
  }

  setDomContainerRect(rect: Rect) {
    this.domContainerRect = rect;
    this.updateIdentifier();
  }

  /**
   * Reused by the transform methods. A chart re-transforms every point of
   * every series on each pan, zoom, and resize frame, so the domain and range
   * tuples are hoisted out of the per-point work.
   */
  private readonly xDomain: [number, number] = [0, 0];
  private readonly yDomain: [number, number] = [0, 0];
  private readonly xRange: [number, number] = [0, 0];
  private readonly yRange: [number, number] = [0, 0];

  private prepareTransform(rectInUiCoordinate: Rect) {
    const rect = rectInUiCoordinate;
    const viewBox = this.currentViewBoxRect;
    this.xDomain[0] = viewBox.x;
    this.xDomain[1] = viewBox.x + viewBox.width;
    this.yDomain[0] = viewBox.y;
    this.yDomain[1] = viewBox.y + viewBox.height;
    this.xRange[0] = rect.x;
    this.xRange[1] = rect.x + rect.width;
    if (this.isYAxisPointedDown()) {
      this.yRange[0] = rect.y + rect.height;
      this.yRange[1] = rect.y;
    } else {
      this.yRange[0] = rect.y;
      this.yRange[1] = rect.y + rect.height;
    }
  }

  /**
   * Converts data coordinate into ui coordinates where the ui coordinate bounds are
   * specified in `rectInUiCoordinate`.
   */
  transformDataToUiCoord(
    rectInUiCoordinate: Rect,
    dataCoordinate: [number, number]
  ): [number, number] {
    this.prepareTransform(rectInUiCoordinate);
    return [
      this.xScale.forward(this.xDomain, this.xRange, dataCoordinate[0]),
      this.yScale.forward(this.yDomain, this.yRange, dataCoordinate[1]),
    ];
  }

  /**
   * Writes an entire series of data coordinates into `polyline` as
   * interleaved x and y ui coordinates. Equivalent to
   * `transformDataToUiCoord` per point, without its per-point allocations.
   *
   * `polyline` must hold two entries per point.
   *
   * Returns whether any written coordinate is NaN, which callers would
   * otherwise have to rediscover with a second pass over `polyline`.
   */
  transformDataToUiCoords(
    rectInUiCoordinate: Rect,
    dataCoordinates: ReadonlyArray<{x: number; y: number}>,
    polyline: Float32Array
  ): boolean {
    this.prepareTransform(rectInUiCoordinate);
    const {xScale, yScale, xDomain, yDomain, xRange, yRange} = this;
    let hasNaN = false;
    for (let index = 0; index < dataCoordinates.length; index++) {
      const dataCoordinate = dataCoordinates[index];
      const x = xScale.forward(xDomain, xRange, dataCoordinate.x);
      const y = yScale.forward(yDomain, yRange, dataCoordinate.y);
      hasNaN = hasNaN || isNaN(x) || isNaN(y);
      polyline[index * 2] = x;
      polyline[index * 2 + 1] = y;
    }
    return hasNaN;
  }
}
