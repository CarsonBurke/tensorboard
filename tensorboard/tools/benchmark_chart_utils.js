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
  };
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
const results = [];
for (const count of [1000, 10000, 100000]) {
  let seed = 1;
  const points = Array.from({length: count}, (_, x) => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return {x, y: seed / 0x100000000};
  });
  const data = [{id: 'series', points}];
  const metadata = {series: {visible: true, aux: false}};
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
  ]) {
    if (baseline)
      assert.strictEqual(
        JSON.stringify(call(current)),
        JSON.stringify(call(baseline))
      );
    const before = baseline ? measure(() => call(baseline), repetitions) : null;
    const after = measure(() => call(current), repetitions);
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
