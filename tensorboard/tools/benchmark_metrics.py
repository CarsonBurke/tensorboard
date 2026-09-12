# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Compare per-point object and columnar scalar JSON response paths.

Run: bazel run //tensorboard/tools:benchmark_metrics
Uses real protobuf columns and provider/plugin serializers, with the RPC stubbed
out. Reports warmed median CPU-path latency and response size for the columnar
format the plugin serves today against the per-point object format it replaced.
These are not HTTP timings; the gzip column measures the compression that
`http_util.Respond` performs for clients that accept it.
"""
import gzip
import json
import statistics
import time
from unittest import mock

from werkzeug import wrappers

from tensorboard import context
from tensorboard.backend import http_util, json_util
from tensorboard.data import grpc_provider
from tensorboard.data.proto import data_provider_pb2
from tensorboard.plugins import base_plugin
from tensorboard.plugins.metrics import metrics_plugin


def _reference_objects_json(plugin, ctx, queries):
    """Serializes scalars the way TensorBoard did before columns.

    One dict per point, cleansed field by field, with `json.dumps` defaults.
    """
    responses = []
    for query in queries:
        mapping = plugin._data_provider.read_scalars(
            ctx,
            experiment_id="",
            plugin_name="scalars",
            downsample=plugin._plugin_downsampling["scalars"],
            run_tag_filter=None,
        )
        run_to_series = {}
        for run, tags in mapping.items():
            if query["tag"] not in tags:
                continue
            run_to_series[json_util.Cleanse(run)] = [
                {
                    "wallTime": json_util.Cleanse(datum.wall_time),
                    "step": json_util.Cleanse(datum.step),
                    "value": json_util.Cleanse(datum.value),
                }
                for datum in tags[query["tag"]]
            ]
        responses.append(
            {
                "plugin": query["plugin"],
                "tag": query["tag"],
                "runToSeries": run_to_series,
            }
        )
    return json.dumps(responses, allow_nan=False)


def _expand_columns(responses):
    """Rebuilds per-point dicts from a columnar response, as the webapp does."""
    expanded = []
    for response in responses:
        run_to_series = {}
        for run, columns in response["runToSeries"].items():
            run_to_series[run] = [
                {"wallTime": wall_time, "step": step, "value": value}
                for step, wall_time, value in zip(
                    columns["steps"], columns["wallTimes"], columns["values"]
                )
            ]
        expanded.append({**response, "runToSeries": run_to_series})
    return expanded


def benchmark(points):
    response = data_provider_pb2.ReadScalarsResponse()
    for run in range((points + 999) // 1000):
        data = (
            response.runs.add(run_name=str(run)).tags.add(tag_name="loss").data
        )
        count = min(1000, points - 1000 * run)
        data.step.extend(range(count))
        data.wall_time.extend(1700000000 + i for i in range(count))
        data.value.extend((i % 17) / 17 for i in range(count))
    stub = mock.Mock()
    stub.ReadScalars.return_value = response
    columnar_provider = grpc_provider.GrpcDataProvider("benchmark", stub)

    class LegacyProvider(grpc_provider.GrpcDataProvider):
        """A provider without columnar reads, exercising the fallback."""

        def read_scalar_columns(self, *args, **kwargs):
            return None

        def read_scalars(self, *args, **kwargs):
            return columnar_provider.read_scalars(*args, **kwargs)

    legacy = metrics_plugin.MetricsPlugin(
        base_plugin.TBContext(data_provider=LegacyProvider("benchmark", stub))
    )
    current = metrics_plugin.MetricsPlugin(
        base_plugin.TBContext(data_provider=columnar_provider)
    )
    ctx = context.RequestContext()
    request = wrappers.Request.from_values()
    queries = [{"plugin": "scalars", "tag": "loss"}]

    def objects():
        return http_util.Respond(
            request,
            _reference_objects_json(legacy, ctx, queries),
            "application/json",
        ).data

    def columns():
        data = current._time_series_impl(ctx, "", queries)
        return http_util.Respond(
            request,
            json.dumps(data, allow_nan=False, separators=(",", ":")),
            "application/json",
        ).data

    def columns_from_read_scalars():
        data = legacy._time_series_impl(ctx, "", queries)
        return json.dumps(data, allow_nan=False, separators=(",", ":"))

    # The columnar response must carry exactly the points the previous format
    # did, from either provider.
    reference = json.loads(objects())
    assert _expand_columns(json.loads(columns())) == reference
    assert _expand_columns(json.loads(columns_from_read_scalars())) == reference

    result = {"points": points}
    for name, fn in [
        ("objects_ms", objects),
        ("columns_ms", columns),
        ("columns_from_read_scalars_ms", columns_from_read_scalars),
    ]:
        body = fn()
        samples = []
        for _ in range(5):
            start = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - start) * 1000)
        result[name] = round(statistics.median(samples), 1)
        if name != "columns_from_read_scalars_ms":
            key = name[: -len("_ms")]
            result[key + "_bytes"] = len(body)
            result[key + "_gzip_bytes"] = len(
                gzip.compress(
                    body if isinstance(body, bytes) else body.encode(), 3
                )
            )
    return result


if __name__ == "__main__":
    print(json.dumps([benchmark(n) for n in [10000, 100000, 500000]], indent=2))
