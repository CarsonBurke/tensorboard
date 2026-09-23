/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
==============================================================================*/

import * as uPlotModule from 'uplot';
import {
  DataSeries,
  DataSeriesMetadataMap,
  Dimension,
  Extent,
} from './internal_types';
import {Chart, ChartCallbacks, UPlotChartOptions} from './chart_types';
import {ScaleType} from './scale_types';

const MAX_POINTS_PER_PIXEL = 4;

type UPlotConstructor = {
  new (
    options: uPlotModule.Options,
    data: unknown,
    target: HTMLElement
  ): uPlotModule;
};

// uPlot is published as CommonJS (`export =`). Bazel's ESM bundler exposes
// that value as either the namespace itself or its `default` property.
const UPlot =
  (uPlotModule as unknown as {default?: UPlotConstructor}).default ??
  (uPlotModule as unknown as UPlotConstructor);

// Mode 2 keeps each series' x coordinates independent. This is important for
// TensorBoard: runs can have different sampling steps and timestamps, so
// constructing one union x-axis would multiply memory by the number of runs.
type PlotData = [null, ...Array<[number[], Array<number | null>]>];

function scaleDistribution(type: ScaleType): 1 | 3 {
  return type === ScaleType.LOG10 ? 3 : 1;
}

/**
 * Canvas time-series chart backed by uPlot.
 *
 * Axes and interaction remain TensorBoard-owned. uPlot is used only for the
 * high-volume line rasterization, which keeps its DOM surface small and makes
 * visibility/style changes cheap.
 */
export class UPlotChart implements Chart {
  private readonly callbacks: ChartCallbacks;
  private readonly container: HTMLElement;
  private dimensions: Dimension;
  private data: DataSeries[] = [];
  private metadata: DataSeriesMetadataMap = {};
  private viewBox: Extent = {x: [0, 1], y: [0, 1]};
  private xScaleType = ScaleType.LINEAR;
  private yScaleType = ScaleType.LINEAR;
  private plot: uPlotModule | null = null;
  private disposed = false;

  constructor(options: UPlotChartOptions) {
    this.callbacks = options.callbacks;
    this.container = options.container;
    this.dimensions = options.domDimension;
    this.recreate();
  }

  resize(dim: Dimension): void {
    if (this.disposed) return;
    const widthChanged = this.dimensions.width !== dim.width;
    this.dimensions = dim;
    if (!this.plot) return;
    this.plot.batch(() => {
      if (widthChanged) {
        this.plot!.setData(
          this.makePlotData() as unknown as uPlotModule.AlignedData,
          false
        );
      }
      this.plot!.setSize({width: dim.width, height: dim.height});
    });
  }

  setMetadata(metadataMap: DataSeriesMetadataMap): void {
    if (this.disposed) return;
    this.metadata = metadataMap;
    if (!this.plot) return;
    for (let index = 0; index < this.data.length; index++) {
      const series = this.plot.series[index + 1];
      const metadata = this.metadata[this.data[index].id];
      series.show = metadata?.visible ?? false;
      series.alpha = metadata?.opacity ?? 1;
    }
    // Style changes do not invalidate geometry. Stroke callbacks read the
    // current metadata for both lines and points on the next draw.
    this.plot.redraw(false);
  }

  setViewBox(extent: Extent): void {
    if (this.disposed) return;
    this.viewBox = extent;
    if (!this.plot) return;
    this.plot.batch(() => {
      this.plot!.setScale('x', {min: extent.x[0], max: extent.x[1]});
      this.plot!.setScale('y', {min: extent.y[0], max: extent.y[1]});
    });
  }

  setData(data: DataSeries[]): void {
    if (this.disposed) return;
    const previousData = this.data;
    this.data = data;
    if (!this.plot || data.length === 0) {
      this.recreate();
      return;
    }

    // Keep the canvas and unchanged prefix. uPlot supports changing its
    // series list in place; rebuilding the entire plot causes visible churn
    // whenever another run finishes loading.
    let firstChanged = 0;
    while (
      firstChanged < previousData.length &&
      firstChanged < data.length &&
      previousData[firstChanged].id === data[firstChanged].id
    ) {
      firstChanged++;
    }
    for (let index = previousData.length; index > firstChanged; index--) {
      this.plot.delSeries(index);
    }
    for (let index = firstChanged; index < data.length; index++) {
      this.plot.addSeries(this.makeSeriesOptions(data[index]));
    }
    this.plot.setData(
      this.makePlotData() as unknown as uPlotModule.AlignedData,
      false
    );
    // setData(..., false) invalidates paths but deliberately does not paint.
    // An unchanged domain (e.g. while zoomed) must still show new samples.
    this.plot.redraw(false);
  }

  setXScaleType(type: ScaleType): void {
    if (this.xScaleType === type) return;
    this.xScaleType = type;
    this.recreate();
  }

  setYScaleType(type: ScaleType): void {
    if (this.yScaleType === type) return;
    this.yScaleType = type;
    this.recreate();
  }

  setUseDarkMode(_useDarkMode: boolean): void {
    // Line colors are owned by metadata. The chart surface is transparent and
    // inherits the card background, so a theme change needs no canvas clear.
  }

  dispose(): void {
    this.disposed = true;
    this.plot?.destroy();
    this.plot = null;
    this.container.replaceChildren();
  }

  private recreate(): void {
    if (this.disposed) return;
    this.plot?.destroy();
    this.plot = null;
    this.container.replaceChildren();

    // uPlot's mode-2 initialization requires at least one data series
    // (`series[1]`). With no data there is nothing to draw, so leave the
    // container empty instead of constructing an invalid plot. The plot is
    // (re)created when data arrives via setData().
    if (this.data.length === 0) return;

    const options: uPlotModule.Options = {
      width: this.dimensions.width,
      height: this.dimensions.height,
      pxAlign: 1,
      scales: {
        x: {
          time: this.xScaleType === ScaleType.TIME,
          distr: scaleDistribution(this.xScaleType),
          auto: false,
          range: () => this.viewBox.x,
        },
        y: {
          distr: scaleDistribution(this.yScaleType),
          auto: false,
          range: () => this.viewBox.y,
        },
      },
      mode: 2,
      series: [
        null as unknown as uPlotModule.Series,
        ...this.data.map((series) => this.makeSeriesOptions(series)),
      ],
      hooks: {draw: [() => this.callbacks.onDrawEnd()]},
      axes: [{show: false}, {show: false}],
      legend: {show: false},
      cursor: {show: false},
      select: {show: false, left: 0, top: 0, width: 0, height: 0},
      // uPlot expects seconds for time scales by default; TensorBoard stores
      // wall time in milliseconds in the chart data.
      ms: this.xScaleType === ScaleType.TIME ? 1e-3 : 1,
    };

    this.plot = new UPlot(options, this.makePlotData(), this.container);
  }

  private makeSeriesOptions(series: DataSeries): uPlotModule.Series {
    const metadata = this.metadata[series.id];
    const stroke = () => this.metadata[series.id]?.color ?? null;
    return {
      label: series.id,
      // Data and metadata arrive independently. Do not display a new run
      // until its visibility and color are known.
      show: metadata?.visible ?? false,
      stroke,
      width: 2,
      alpha: metadata?.opacity ?? 1,
      spanGaps: false,
      points: {
        // In mode 2 uPlot does not supply aligned point indices. Explicitly
        // select the lone sample after duplicate-x normalization.
        show: false,
        filter: (plot, index) => {
          const data = plot.data as unknown as PlotData;
          return data[index]![0].length === 1 ? [0] : null;
        },
        stroke,
        fill: stroke,
      },
      facets: [
        {scale: 'x', auto: false},
        {scale: 'y', auto: false},
      ],
    };
  }

  private makePlotData(): PlotData {
    const maxPoints = Math.max(
      4,
      Math.floor(this.dimensions.width * MAX_POINTS_PER_PIXEL)
    );
    const result: PlotData = [null];
    for (const series of this.data) {
      const points = this.decimate(series.points, maxPoints);
      const x: number[] = [];
      const y: Array<number | null> = [];
      for (const point of points) {
        if (!Number.isFinite(point.x)) continue;
        // uPlot uses null to represent a gap. Preserve repeated timestamps by
        // keeping the last sample, which is the same policy as TensorBoard's
        // existing line-series preparation.
        if (x.length > 0 && x[x.length - 1] === point.x) {
          y[y.length - 1] = Number.isFinite(point.y) ? point.y : null;
        } else {
          x.push(point.x);
          y.push(Number.isFinite(point.y) ? point.y : null);
        }
      }
      result.push([x, y]);
    }
    return result;
  }

  private decimate(
    points: ReadonlyArray<{x: number; y: number}>,
    maxPoints: number
  ): ReadonlyArray<{x: number; y: number}> {
    if (points.length <= maxPoints) return points;
    // Keep each bucket's endpoints and extrema, plus discontinuities. Gaps
    // may exceed the point budget: dropping them would invent connecting lines.
    const bucketCount = Math.max(1, Math.floor(maxPoints / 4));
    const stride = points.length / bucketCount;
    const output: Array<{x: number; y: number}> = [];
    for (let bucket = 0; bucket < bucketCount; bucket++) {
      const start = Math.floor(bucket * stride);
      const end = Math.min(points.length, Math.floor((bucket + 1) * stride));
      if (end <= start) continue;
      let min = -1;
      let max = -1;
      const indices = new Set([start, end - 1]);
      for (let index = start; index < end; index++) {
        if (!Number.isFinite(points[index].y)) {
          indices.add(index);
          if (index > start) indices.add(index - 1);
          if (index + 1 < end) indices.add(index + 1);
          continue;
        }
        if (min === -1 || points[index].y < points[min].y) min = index;
        if (max === -1 || points[index].y > points[max].y) max = index;
      }
      if (min !== -1) indices.add(min);
      if (max !== -1) indices.add(max);
      for (const index of [...indices].sort((a, b) => a - b)) {
        const point = points[index];
        if (output.length === 0 || output[output.length - 1] !== point) {
          output.push(point);
        }
      }
    }
    return output;
  }
}
