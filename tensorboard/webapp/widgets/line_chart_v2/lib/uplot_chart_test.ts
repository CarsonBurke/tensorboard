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

import {DataSeries, RendererType, UPlotChartOptions} from './public_types';
import {UPlotChart} from './uplot_chart';

describe('line_chart_v2/lib/uplot_chart test', () => {
  let container: HTMLElement;
  let options: UPlotChartOptions;

  function makeSeries(id: string, nPoints: number): DataSeries {
    const points = [];
    for (let i = 0; i < nPoints; i++) {
      points.push({x: i, y: i});
    }
    return {id, points};
  }

  beforeEach(() => {
    container = document.createElement('div');
    document.body.appendChild(container);
    options = {
      type: RendererType.UPLOT,
      container,
      devicePixelRatio: 1,
      callbacks: {onDrawEnd: () => {}, onContextLost: () => {}},
      domDimension: {width: 300, height: 200},
      useDarkMode: false,
    };
  });

  afterEach(() => {
    container.remove();
  });

  it('constructs with no data without throwing and renders nothing', () => {
    let chart: UPlotChart | null = null;
    expect(() => {
      chart = new UPlotChart(options);
    }).not.toThrow();
    expect(container.querySelector('.uplot')).toBeNull();
    chart!.dispose();
  });

  it('creates the plot once data arrives after construction', () => {
    const chart = new UPlotChart(options);
    chart.setMetadata({
      run1: {
        id: 'run1',
        displayName: 'run1',
        visible: true,
        color: '#f00',
      },
    });
    chart.setData([makeSeries('run1', 10)]);
    chart.setViewBox({x: [0, 9], y: [0, 9]});
    expect(container.querySelector('.uplot')).not.toBeNull();
    expect(container.querySelector('canvas')).not.toBeNull();
    chart.dispose();
  });

  it('draws visible pixels for in-range data', () => {
    const chart = new UPlotChart(options);
    chart.setMetadata({
      run1: {
        id: 'run1',
        displayName: 'run1',
        visible: true,
        color: '#f00',
      },
    });
    chart.setData([makeSeries('run1', 100)]);
    chart.setViewBox({x: [0, 99], y: [0, 99]});
    const canvas = container.querySelector('canvas') as HTMLCanvasElement;
    expect(canvas).not.toBeNull();
    const ctx = canvas.getContext('2d')!;
    const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    let nonBlank = 0;
    for (let i = 3; i < pixels.length; i += 4) {
      if (pixels[i] > 0) nonBlank++;
    }
    expect(nonBlank).toBeGreaterThan(0);
    chart.dispose();
  });

  function metadata(color: string, visible = true) {
    return {run1: {id: 'run1', displayName: 'run1', visible, color}};
  }

  async function paintedPixels() {
    // uPlot commits redraws in a microtask.
    await Promise.resolve();
    const canvas = container.querySelector('canvas')!;
    const pixels = canvas
      .getContext('2d')!
      .getImageData(0, 0, canvas.width, canvas.height).data;
    let red = 0;
    let blue = 0;
    let opaque = 0;
    for (let i = 0; i < pixels.length; i += 4) {
      if (!pixels[i + 3]) continue;
      opaque++;
      if (pixels[i] > pixels[i + 2]) red++;
      if (pixels[i + 2] > pixels[i]) blue++;
    }
    return {red, blue, opaque};
  }

  it('repaints colors without a resize or view box change', async () => {
    const chart = new UPlotChart(options);
    try {
      chart.setMetadata(metadata('#f00'));
      chart.setData([makeSeries('run1', 100)]);
      chart.setViewBox({x: [0, 99], y: [0, 99]});
      expect((await paintedPixels()).red).toBeGreaterThan(0);
      chart.setMetadata(metadata('#00f'));
      const pixels = await paintedPixels();
      expect(pixels.blue).toBeGreaterThan(0);
      expect(pixels.red).toBe(0);
    } finally {
      chart.dispose();
    }
  });

  it('repaints data updates within an unchanged view box', async () => {
    const chart = new UPlotChart(options);
    try {
      chart.setMetadata(metadata('#f00'));
      chart.setData([makeSeries('run1', 10)]);
      chart.setViewBox({x: [0, 9], y: [0, 9]});
      expect((await paintedPixels()).opaque).toBeGreaterThan(0);
      chart.setData([
        {
          id: 'run1',
          points: [
            {x: 0, y: 20},
            {x: 9, y: 20},
          ],
        },
      ]);
      expect((await paintedPixels()).opaque).toBe(0);
    } finally {
      chart.dispose();
    }
  });

  it('hides unstyled series until metadata arrives', async () => {
    const chart = new UPlotChart(options);
    try {
      chart.setData([makeSeries('run1', 10)]);
      chart.setViewBox({x: [0, 9], y: [0, 9]});
      expect((await paintedPixels()).opaque).toBe(0);
      chart.setMetadata(metadata('#00f'));
      expect((await paintedPixels()).blue).toBeGreaterThan(0);
    } finally {
      chart.dispose();
    }
  });

  it('repaints visibility changes and removed metadata', async () => {
    const chart = new UPlotChart(options);
    try {
      chart.setMetadata(metadata('#f00'));
      chart.setData([makeSeries('run1', 10)]);
      chart.setViewBox({x: [0, 9], y: [0, 9]});
      expect((await paintedPixels()).red).toBeGreaterThan(0);
      chart.setMetadata(metadata('#f00', false));
      expect((await paintedPixels()).opaque).toBe(0);
      chart.setMetadata(metadata('#00f'));
      expect((await paintedPixels()).blue).toBeGreaterThan(0);
      chart.setMetadata({});
      expect((await paintedPixels()).opaque).toBe(0);
    } finally {
      chart.dispose();
    }
  });

  it('keeps the canvas when runs are added or removed', async () => {
    const chart = new UPlotChart(options);
    try {
      chart.setMetadata(metadata('#f00'));
      chart.setData([makeSeries('run1', 10)]);
      chart.setViewBox({x: [0, 9], y: [0, 9]});
      const canvas = container.querySelector('canvas');
      chart.setData([makeSeries('run2', 10), makeSeries('run1', 10)]);
      expect(container.querySelector('canvas')).toBe(canvas);
      expect((await paintedPixels()).red).toBeGreaterThan(0);
      chart.setData([makeSeries('run1', 10)]);
      expect(container.querySelector('canvas')).toBe(canvas);
      expect((await paintedPixels()).red).toBeGreaterThan(0);
    } finally {
      chart.dispose();
    }
  });

  it('draws overlapping lines in the full supplied order after reordering', async () => {
    const chart = new UPlotChart(options);
    try {
      chart.setMetadata(
        Object.fromEntries(
          [
            ['run1', '#f00'],
            ['run2', '#0f0'],
            ['run3', '#00f'],
          ].map(([id, color]) => [
            id,
            {id, displayName: id, visible: true, color},
          ])
        )
      );
      const runs = ['run1', 'run2', 'run3'].map((id) => makeSeries(id, 10));
      chart.setData(runs);
      chart.setViewBox({x: [0, 9], y: [0, 9]});
      expect((await paintedPixels()).blue).toBeGreaterThan(0);
      expect((await paintedPixels()).red).toBe(0);
      chart.setData([runs[1], runs[2], runs[0]]);
      expect((await paintedPixels()).red).toBeGreaterThan(0);
      expect((await paintedPixels()).blue).toBe(0);
      chart.setData([runs[1], runs[2]]);
      expect((await paintedPixels()).blue).toBeGreaterThan(0);
    } finally {
      chart.dispose();
    }
  });

  it('renders a single sample and recolors its point', async () => {
    const chart = new UPlotChart(options);
    try {
      chart.setMetadata(metadata('#f00'));
      chart.setData([{id: 'run1', points: [{x: 5, y: 5}]}]);
      chart.setViewBox({x: [0, 10], y: [0, 10]});
      expect((await paintedPixels()).red).toBeGreaterThan(0);
      chart.setMetadata(metadata('#00f'));
      const pixels = await paintedPixels();
      expect(pixels.blue).toBeGreaterThan(0);
      expect(pixels.red).toBe(0);
    } finally {
      chart.dispose();
    }
  });

  it('renders the last sample when repeated x coordinates collapse to one point', async () => {
    const chart = new UPlotChart(options);
    try {
      chart.setMetadata(metadata('#f00'));
      chart.setData([
        {
          id: 'run1',
          points: [
            {x: 5, y: 50},
            {x: 5, y: 5},
          ],
        },
      ]);
      chart.setViewBox({x: [0, 10], y: [0, 10]});
      expect((await paintedPixels()).red).toBeGreaterThan(0);
    } finally {
      chart.dispose();
    }
  });

  it('preserves missing-data gaps while downsampling', async () => {
    const chart = new UPlotChart({
      ...options,
      domDimension: {width: 50, height: 100},
    });
    try {
      chart.setMetadata(metadata('#f00'));
      const points = Array.from({length: 1000}, (_, x) => ({
        x,
        y: x === 10 ? NaN : 0,
      }));
      chart.setData([{id: 'run1', points}]);
      chart.setViewBox({x: [0, 20], y: [-1, 1]});
      expect((await paintedPixels()).red).toBeGreaterThan(0);
      const canvas = container.querySelector('canvas')!;
      const pixels = canvas
        .getContext('2d')!
        .getImageData(Math.floor(canvas.width / 2), 0, 1, canvas.height).data;
      expect(
        Array.from(pixels)
          .filter((_, index) => index % 4 === 3)
          .every((alpha) => alpha === 0)
      ).toBe(true);
    } finally {
      chart.dispose();
    }
  });

  it('updates the series when the series count changes', () => {
    const chart = new UPlotChart(options);
    chart.setData([makeSeries('run1', 10)]);
    const firstCanvas = container.querySelector('canvas');
    expect(firstCanvas).not.toBeNull();

    chart.setData([makeSeries('run1', 10), makeSeries('run2', 10)]);
    const canvases = container.querySelectorAll('canvas');
    // Changing runs must not accumulate canvases.
    expect(canvases.length).toBe(1);
    expect(container.querySelector('.uplot')).not.toBeNull();
    chart.dispose();
  });

  it('updates the series when series identity changes', () => {
    const chart = new UPlotChart(options);
    chart.setData([makeSeries('run1', 10)]);
    chart.setData([makeSeries('run2', 10)]);
    expect(container.querySelectorAll('canvas').length).toBe(1);
    expect(container.querySelector('.uplot')).not.toBeNull();
    chart.dispose();
  });

  it('clears the plot when data becomes empty', () => {
    const chart = new UPlotChart(options);
    chart.setData([makeSeries('run1', 10)]);
    expect(container.querySelector('.uplot')).not.toBeNull();

    chart.setData([]);
    expect(container.querySelector('.uplot')).toBeNull();
    chart.dispose();
  });
});
