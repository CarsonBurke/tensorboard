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

  it('keeps drawing after metadata updates on an existing plot', async () => {
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

    // A metadata update once the plot exists mutates the live series. A
    // subsequent draw must still paint the line.
    chart.setMetadata({
      run1: {
        id: 'run1',
        displayName: 'run1',
        visible: true,
        color: '#00f',
      },
    });
    chart.resize({width: 300, height: 200});
    await new Promise((resolve) => setTimeout(resolve, 100));

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

  it('rebuilds the series when the series count changes', () => {
    const chart = new UPlotChart(options);
    chart.setData([makeSeries('run1', 10)]);
    const firstCanvas = container.querySelector('canvas');
    expect(firstCanvas).not.toBeNull();

    chart.setData([makeSeries('run1', 10), makeSeries('run2', 10)]);
    const canvases = container.querySelectorAll('canvas');
    // A rebuilt plot replaces the old canvas instead of accumulating one.
    expect(canvases.length).toBe(1);
    expect(container.querySelector('.uplot')).not.toBeNull();
    chart.dispose();
  });

  it('rebuilds the series when series identity changes', () => {
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
