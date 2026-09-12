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

import {Coordinator} from './coordinator';
import {
  DataInternalSeries,
  DataSeries,
  DataSeriesMetadataMap,
  Rect,
} from './internal_types';
import {PaintBrush} from './paint_brush';
import {ObjectRenderer} from './renderer/renderer_types';

type Cacheable = {};

/**
 * Coordinate identifier of a series that has never been transformed. The
 * coordinator's identifiers start at 0 and only increase.
 */
const NEVER_TRANSFORMED = -1;

export interface RenderCache {
  getFromPreviousFrame(key: string): Cacheable | null;
  setToCurrentFrame(key: string, value: Cacheable): void;
}

class RenderCacheContainer implements RenderCache {
  private prevFrameCache = new Map<string, Cacheable>();
  private currFrameCache = new Map<string, Cacheable>();

  getFromPreviousFrame(key: string): Cacheable | null {
    const value = this.prevFrameCache.get(key);
    return value ?? null;
  }

  setToCurrentFrame(key: string, value: Cacheable) {
    this.currFrameCache.set(key, value);
  }

  /**
   * Flush the current frame cache into previous frame cache. At this point, you should
   * not update the current frame cache with `set` calls.
   *
   * It returns cached objects that got removed from the new frame.
   */
  finalizeFrameAndGetRemoved(): ReadonlyArray<Cacheable> {
    const removed = [];

    for (const [key, value] of this.prevFrameCache.entries()) {
      if (!this.currFrameCache.has(key)) {
        removed.push(value);
      }
    }

    this.prevFrameCache = this.currFrameCache;
    this.currFrameCache = new Map();

    return removed;
  }
}

export interface DrawableConfig {
  coordinator: Coordinator;
  getMetadataMap: () => DataSeriesMetadataMap;
  renderer: ObjectRenderer;
}

/**
 * A view that renders data in a rectangular region. A client of DataDrawable is expected
 * to subclass DataDrawable and implement `redraw` method.
 *
 * The base class maintains cache of coordinate mapped data, `series` and rendered scene.
 *
 * Example:
 *
 * class LineView extends DataDrawable {
 *   redraw() {
 *     for (const line of this.series) {
 *       this.paintBrush.setLine('uniqId', line, ...);
 *     }
 *   }
 * }
 */
export abstract class DataDrawable {
  private rawSeriesData: DataSeries[] = [];
  // UI coordinate mapped data.
  protected series: DataInternalSeries[] = [];
  protected readonly coordinator: Coordinator;
  protected readonly paintBrush: PaintBrush;

  private paintDirty = true;
  private readonly getMetadataMapImpl: () => DataSeriesMetadataMap;
  private readonly renderer: ObjectRenderer;
  private readonly renderCache = new RenderCacheContainer();
  /**
   * Coordinate identifier each entry of `series` was transformed with,
   * parallel to `series`. Hidden series are skipped by the transform, so
   * entries may lag the coordinator's current identifier.
   */
  private seriesIdentifiers: number[] = [];
  private layout: Rect = {x: 0, width: 1, y: 0, height: 1};

  constructor(config: DrawableConfig) {
    this.getMetadataMapImpl = config.getMetadataMap;
    this.coordinator = config.coordinator;
    this.renderer = config.renderer;
    this.paintBrush = new PaintBrush(this.renderCache, this.renderer);
  }

  setLayoutRect(layout: Rect) {
    if (
      this.layout.x !== layout.x ||
      this.layout.width !== layout.width ||
      this.layout.y !== layout.y ||
      this.layout.height !== layout.height
    ) {
      this.paintDirty = true;
      // Ui coordinates are relative to the layout, so they are all stale.
      this.seriesIdentifiers.fill(NEVER_TRANSFORMED);
    }
    this.layout = layout;
  }

  protected getLayoutRect(): Rect {
    return this.layout;
  }

  protected getMetadataMap(): DataSeriesMetadataMap {
    return this.getMetadataMapImpl();
  }

  /**
   * Manually marks paint as dirty. Drawable automatically marks paint as dirty when data
   * or layout changes. If there are other conditions in which redraw must happen, invoke
   * this method.
   */
  markAsPaintDirty() {
    this.paintDirty = true;
  }

  /**
   * Renders a rectangular region if paint is dirty.
   *
   * @final Do not override.
   */
  render() {
    this.transformCoordinatesIfStale();

    if (!this.paintDirty) return;

    this.redraw();

    for (const removedObj of this.renderCache.finalizeFrameAndGetRemoved()) {
      this.renderer.destroyObject(removedObj);
    }

    this.paintDirty = false;
  }

  setData(data: DataSeries[]) {
    this.rawSeriesData = data;
    this.seriesIdentifiers.fill(NEVER_TRANSFORMED);
  }

  /**
   * Maps data coordinates of stale series into ui coordinates.
   *
   * Every pan, zoom, and resize frame invalidates the transform, making this
   * the hottest loop in the chart, so a series is transformed only when its
   * own coordinates are stale and something paints it: a hidden series is
   * left untransformed until the frame it becomes visible again.
   */
  private transformCoordinatesIfStale(): void {
    const identifier = this.coordinator.getUpdateIdentifier();
    const layoutRect = this.getLayoutRect();
    const metadataMap = this.getMetadataMap();
    let transformed = false;

    if (this.series.length !== this.rawSeriesData.length) {
      this.series.length = this.rawSeriesData.length;
      this.seriesIdentifiers.length = this.rawSeriesData.length;
    }

    for (let index = 0; index < this.rawSeriesData.length; index++) {
      const datum = this.rawSeriesData[index];
      const internalSeries = this.series[index];
      if (
        !internalSeries ||
        internalSeries.id !== datum.id ||
        internalSeries.polyline.length !== datum.points.length * 2
      ) {
        // Keeps `series` parallel to the data even for a series that is
        // never painted. Zeroed coordinates are never drawn: only a visible
        // series is transformed, and only a visible series is painted.
        this.series[index] = {
          id: datum.id,
          polyline: new Float32Array(datum.points.length * 2),
          hasNaN: false,
        };
        this.seriesIdentifiers[index] = NEVER_TRANSFORMED;
      }
      if (this.seriesIdentifiers[index] === identifier) {
        continue;
      }
      const metadata = metadataMap[datum.id];
      if (!metadata?.visible) {
        // Nothing paints this series, so its stale coordinates cannot be
        // seen. Leaving its identifier behind transforms it on the frame it
        // is painted again.
        continue;
      }
      // Renderers cache the polyline they last drew and compare its contents
      // to decide whether to rewrite the DOM or the GPU geometry, so each
      // transform must produce a new buffer instead of writing in place.
      // Skipping the transform above leaves the previous buffer in place,
      // which those comparisons short-circuit on identity.
      const polyline = new Float32Array(datum.points.length * 2);
      const hasNaN = this.coordinator.transformDataToUiCoords(
        layoutRect,
        datum.points,
        polyline
      );
      this.series[index] = {id: datum.id, polyline, hasNaN};
      this.seriesIdentifiers[index] = identifier;
      transformed = true;
    }

    if (transformed) {
      this.markAsPaintDirty();
    }
  }

  /**
   * Draws a rectangular region with coordinate system transformed `this.series` and
   * `this.paintBrush`.
   */
  protected abstract redraw(): void;
}
