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
/**
 * Source-level chart microbenchmarks, independent of browser rendering.
 *
 * Run with Node and the repository's npm dependencies:
 *   TB_BENCH_NODE_MODULES="$(bazel info output_base)/external/npm/node_modules" \
 *     node tensorboard/tools/benchmark_chart_utils.js [baseline-source-root]
 *
 * An optional baseline root supplies the two chart utility source files from
 * before a change. Other imports use the current tree. No source files are
 * modified. The benchmark transpiles actual functions and verifies identical
 * results before timing them. Timings are medians of five warmed batches.
 */
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const dependency = (name) =>
  require(process.env.TB_BENCH_NODE_MODULES
    ? path.join(process.env.TB_BENCH_NODE_MODULES, name)
    : name);
const ts = dependency('typescript');
const {performance} = require('perf_hooks');
const vm = require('vm');

const root = path.resolve(__dirname, '../..');
const chart = 'tensorboard/webapp/widgets/line_chart_v2/';

function loadUtils(sourceRoot) {
  const cache = new Map();
  const context = vm.createContext({self: {}});
  function load(relativePath) {
    if (cache.has(relativePath)) return cache.get(relativePath);
    const candidate = path.join(sourceRoot, relativePath);
    const filename = fs.existsSync(candidate)
      ? candidate
      : path.join(root, relativePath);
    const source = ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
      compilerOptions: {
        module: ts.ModuleKind.CommonJS,
        target: ts.ScriptTarget.ES2020,
      },
    }).outputText;
    const exports = {};
    cache.set(relativePath, exports);
    const localRequire = (name) =>
      name.startsWith('.')
        ? load(
            path.normalize(path.join(path.dirname(relativePath), name + '.ts'))
          )
        : dependency(name);
    const evaluate = vm.runInContext(
      '(function(exports, require) {' + source + '\n})',
      context
    );
    evaluate(exports, localRequire);
    return exports;
  }
  return {
    ...load(chart + 'line_chart_internal_utils.ts'),
    ...load(chart + 'sub_view/line_chart_interactive_utils.ts'),
    coordinator: load(chart + 'lib/coordinator.ts'),
    scale: load(chart + 'lib/scale.ts'),
    scaleTypes: load(chart + 'lib/scale_types.ts'),
  };
}

/**
 * Transforms every point of a series into ui coordinates, the way a chart
 * does on each pan, zoom, and resize frame. Uses the batch coordinator API
 * when the source tree has one, and the per-point API otherwise.
 */
function transformSeries(utils, coordinator, layoutRect, points, polyline) {
  if (coordinator.transformDataToUiCoords) {
    coordinator.transformDataToUiCoords(layoutRect, points, polyline);
    return polyline;
  }
  for (let index = 0; index < points.length; index++) {
    const [x, y] = coordinator.transformDataToUiCoord(layoutRect, [
      points[index].x,
      points[index].y,
    ]);
    polyline[index * 2] = x;
    polyline[index * 2 + 1] = y;
  }
  return polyline;
}

function buildCoordinator(utils, scaleType) {
  const coordinator = new utils.coordinator.Coordinator();
  const scale = utils.scale.createScale(scaleType);
  coordinator.setXScale(scale);
  coordinator.setYScale(utils.scale.createScale(utils.scaleTypes.ScaleType.LINEAR));
  coordinator.setDomContainerRect({x: 0, y: 0, width: 800, height: 400});
  coordinator.setViewBoxRect({x: 0, y: 0, width: 1000, height: 1});
  return coordinator;
}

function measure(fn, repetitions) {
  for (let i = 0; i < 10; i++) fn();
  const times = [];
  for (let batch = 0; batch < 5; batch++) {
    const start = performance.now();
    for (let i = 0; i < repetitions; i++) fn();
    times.push((performance.now() - start) / repetitions);
  }
  return times.sort((a, b) => a - b)[2];
}

const current = loadUtils(root);
const baseline = process.argv[2]
  ? loadUtils(path.resolve(process.argv[2]))
  : null;
const layoutRect = {x: 0, y: 0, width: 800, height: 400};
const results = [];
for (const count of [1000, 10000, 100000]) {
  let seed = 1;
  const points = Array.from({length: count}, (_, x) => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return {x, y: seed / 0x100000000};
  });
  const data = [{id: 'series', points}];
  const metadata = {series: {visible: true, aux: false}};
  const buildState = (utils) => ({
    linear: buildCoordinator(utils, utils.scaleTypes.ScaleType.LINEAR),
    temporal: buildCoordinator(utils, utils.scaleTypes.ScaleType.TIME),
    polyline: new Float32Array(count * 2),
  });
  const currentState = buildState(current);
  const baselineState = baseline ? buildState(baseline) : null;
  for (const [name, call, repetitions] of [
    ['hover', (utils) => utils.findClosestIndex(points, count * 0.53), 1000],
    [
      'extent',
      (utils) =>
        utils.computeDataSeriesExtent(
          data,
          metadata,
          false,
          Number.isFinite,
          Number.isFinite
        ),
      20,
    ],
    [
      'outlier extent',
      (utils) =>
        utils.computeDataSeriesExtent(
          data,
          metadata,
          true,
          Number.isFinite,
          Number.isFinite
        ),
      20,
    ],
    [
      'step axis transform',
      (utils, state) =>
        transformSeries(
          utils,
          state.linear,
          layoutRect,
          points,
          state.polyline
        ),
      20,
    ],
    [
      'wall time axis transform',
      (utils, state) =>
        transformSeries(
          utils,
          state.temporal,
          layoutRect,
          points,
          state.polyline
        ),
      20,
    ],
  ]) {
    if (baseline) {
      // Sources are evaluated in separate contexts, so compare values rather
      // than prototypes.
      const serialize = (value) =>
        JSON.stringify(ArrayBuffer.isView(value) ? Array.from(value) : value);
      assert.strictEqual(
        serialize(call(current, currentState)),
        serialize(call(baseline, baselineState))
      );
    }
    const before = baseline
      ? measure(() => call(baseline, baselineState), repetitions)
      : null;
    const after = measure(() => call(current, currentState), repetitions);
    results.push({
      operation: name,
      points: count,
      baseline_ms: before,
      current_ms: after,
      speedup: before === null ? null : before / after,
    });
  }
}
console.log(JSON.stringify({node: process.version, results}, null, 2));
