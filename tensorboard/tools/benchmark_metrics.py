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
"""Compare legacy scalar object/cleansing and columnar JSON response paths.

Run: bazel run //tensorboard/tools:benchmark_metrics
Uses real protobuf columns and provider/plugin serializers, with the RPC stubbed
out. Reports warmed median CPU-path latency; these are not HTTP timings.
"""
import json
import statistics
import time
from unittest import mock

from werkzeug import wrappers

from tensorboard import context
from tensorboard.backend import http_util
from tensorboard.data import grpc_provider
from tensorboard.data.proto import data_provider_pb2
from tensorboard.plugins import base_plugin
from tensorboard.plugins.metrics import metrics_plugin


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
    columnar = grpc_provider.GrpcDataProvider("benchmark", stub)

    class LegacyProvider(grpc_provider.GrpcDataProvider):
        def read_scalar_columns(self, *args, **kwargs):
            return None

        def read_scalars(self, *args, **kwargs):
            return columnar.read_scalars(*args, **kwargs)

    legacy = metrics_plugin.MetricsPlugin(
        base_plugin.TBContext(data_provider=LegacyProvider("benchmark", stub))
    )
    current = metrics_plugin.MetricsPlugin(
        base_plugin.TBContext(data_provider=columnar)
    )
    ctx = context.RequestContext()
    request = wrappers.Request.from_values()
    queries = [{"plugin": "scalars", "tag": "loss"}]

    def before():
        data = legacy._time_series_impl(ctx, "", queries)
        return http_util.Respond(request, data, "application/json").data

    def after():
        data = current._time_series_impl(ctx, "", queries, for_json=True)
        return http_util.Respond(
            request, json.dumps(data, allow_nan=False), "application/json"
        ).data

    assert before() == after()
    result = {"points": points}
    for name, fn in [
        ("objects_and_cleanse_ms", before),
        ("columns_and_schema_ms", after),
    ]:
        fn()
        samples = []
        for _ in range(5):
            start = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - start) * 1000)
        result[name] = statistics.median(samples)
    return result


if __name__ == "__main__":
    print(json.dumps([benchmark(n) for n in [10000, 100000, 500000]], indent=2))
