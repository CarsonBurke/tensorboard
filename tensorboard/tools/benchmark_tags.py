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
"""Break down the cost of the metrics tag listing.

Run: bazel run //tensorboard/tools:benchmark_tags

`GET /data/plugin/timeseries/tags` is on the dashboard's critical path: nothing
renders until the webapp knows which runs carry which tags. With tens of
thousands of run/tag pairs the response takes hundreds of milliseconds to
build, and it holds the GIL while it does, so every other request on the load
path waits for it.

This stubs the data server RPC with a synthetic `ListScalarsResponse` of the
requested shape and times each stage the Python process actually performs:
parsing the protobuf response, translating it into provider values, formatting
the JSON structure, serializing, and compressing. Both wire forms (names
repeated per entry, and names deduplicated into response tables) and both
provider paths (per-pair objects, and the columnar tag index) are measured.
Reported numbers are warmed medians in milliseconds; the RPC itself is excluded
because it is the data server's cost, not Python's.
"""
import gzip
import json
import statistics
import sys
import time
from unittest import mock

import grpc

from tensorboard import context
from tensorboard.data import grpc_provider
from tensorboard.data.proto import data_provider_pb2
from tensorboard.plugins import base_plugin
from tensorboard.plugins.metrics import metrics_plugin
from tensorboard.plugins.scalar import metadata as scalar_metadata


def _synthetic_response(runs, tags_per_run, dedup):
    """Builds a listing response shaped like a real training logdir's.

    Run names are long and unique, tag names are shared across runs, and the
    summary metadata is the handful of distinct byte strings a scalar writer
    emits. `dedup` selects the response encoding, as `dedup_names` does on a
    real server.
    """
    content = scalar_metadata.create_summary_metadata(
        "", None
    ).plugin_data.content
    tag_names = [
        "%s/%s" % (group, leaf)
        for group in ("charts", "losses", "eval", "accounting")
        for leaf in ("SPS", "value_loss", "policy_loss", "entropy", "steps")
    ]
    response = data_provider_pb2.ListScalarsResponse()
    if dedup:
        response.tag_names.extend(
            "%s_%d" % (tag_names[i % len(tag_names)], i)
            for i in range(tags_per_run)
        )
        table_entry = response.summary_metadata_table.add()
        table_entry.plugin_data.plugin_name = scalar_metadata.PLUGIN_NAME
        table_entry.plugin_data.content = content
    for run_index in range(runs):
        run_entry = response.runs.add(
            run_name="HalfCheetah-v4__experiment_variant_%05d__1__178%07d"
            % (run_index, run_index)
        )
        for tag_index in range(tags_per_run):
            tag_entry = run_entry.tags.add()
            tag_entry.metadata.max_step = 100000
            tag_entry.metadata.max_wall_time = 1788492351.0
            if dedup:
                tag_entry.tag_index = tag_index
                continue
            tag_entry.tag_name = "%s_%d" % (
                tag_names[tag_index % len(tag_names)],
                tag_index,
            )
            tag_entry.metadata.summary_metadata.plugin_data.plugin_name = (
                scalar_metadata.PLUGIN_NAME
            )
            tag_entry.metadata.summary_metadata.plugin_data.content = content
    return response


class _PairwiseProvider(grpc_provider.GrpcDataProvider):
    """A provider without the columnar listing, exercising the fallback."""

    def list_scalars_tag_index(self, *args, **kwargs):
        return None


def benchmark(runs, tags_per_run):
    inline = _synthetic_response(runs, tags_per_run, dedup=False)
    deduped = _synthetic_response(runs, tags_per_run, dedup=True)
    stub = mock.Mock()
    # Answer as a real server does: indexed tables only when asked for them.
    stub.ListScalars.side_effect = lambda req: (
        deduped if req.dedup_names else inline
    )
    stub.ListTensors.return_value = data_provider_pb2.ListTensorsResponse()
    stub.ListBlobSequences.return_value = (
        data_provider_pb2.ListBlobSequencesResponse()
    )
    columnar_provider = grpc_provider.GrpcDataProvider("benchmark", stub)
    pairwise_provider = _PairwiseProvider("benchmark", stub)
    plugin = metrics_plugin.MetricsPlugin(
        base_plugin.TBContext(data_provider=columnar_provider)
    )
    pairwise_plugin = metrics_plugin.MetricsPlugin(
        base_plugin.TBContext(data_provider=pairwise_provider)
    )
    ctx = context.RequestContext()
    inline_wire = inline.SerializeToString()
    deduped_wire = deduped.SerializeToString()

    def parse_inline():
        # The gRPC stub pays this on every listing: the data server's bytes
        # become a Python message before the provider sees a single field.
        parsed = data_provider_pb2.ListScalarsResponse()
        parsed.ParseFromString(inline_wire)
        return parsed

    def parse_deduped():
        parsed = data_provider_pb2.ListScalarsResponse()
        parsed.ParseFromString(deduped_wire)
        return parsed

    def list_metadata():
        return pairwise_provider.list_scalars_metadata(
            ctx, experiment_id="", plugin_name=scalar_metadata.PLUGIN_NAME
        )

    def tag_index():
        return columnar_provider.list_scalars_tag_index(
            ctx, experiment_id="", plugin_name=scalar_metadata.PLUGIN_NAME
        )

    def pairwise_tags_impl():
        return pairwise_plugin._tags_impl(ctx, experiment="")

    def tags_impl():
        return plugin._tags_impl(ctx, experiment="")

    def serialize():
        return json.dumps(payload, allow_nan=False, separators=(",", ":"))

    def compress():
        return gzip.compress(body, 3)

    payload = tags_impl()
    body = serialize().encode()
    if pairwise_tags_impl() != payload:
        raise AssertionError("columnar and pairwise listings disagree")

    result = {
        "runs": runs,
        "tags_per_run": tags_per_run,
        "pairs": runs * tags_per_run,
        "inline_rpc_bytes": len(inline_wire),
        "deduped_rpc_bytes": len(deduped_wire),
        "bytes": len(body),
        "gzip_bytes": len(compress()),
    }
    for name, fn in [
        ("parse_inline_ms", parse_inline),
        ("parse_deduped_ms", parse_deduped),
        ("list_metadata_ms", list_metadata),
        ("tag_index_ms", tag_index),
        ("pairwise_tags_impl_ms", pairwise_tags_impl),
        ("tags_impl_ms", tags_impl),
        ("serialize_ms", serialize),
        ("gzip_ms", compress),
    ]:
        samples = []
        for _ in range(5):
            start = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - start) * 1000)
        result[name] = round(statistics.median(samples), 1)
    return result


def benchmark_live(port):
    """Measures the same stages against a real data server.

    The synthetic mode above cannot see what the gRPC hop costs: the data
    server's encoding, the transfer, and the client's protobuf parse. Point
    this at the port in a running server's `--port-file` to measure them.
    """
    address = "localhost:%d" % port
    channel = grpc.insecure_channel(
        address,
        options=[("grpc.max_receive_message_length", 1024 * 1024 * 256)],
    )
    stub = grpc_provider.make_stub(channel)
    columnar_provider = grpc_provider.GrpcDataProvider(address, stub)
    pairwise_provider = _PairwiseProvider(address, stub)
    plugin = metrics_plugin.MetricsPlugin(
        base_plugin.TBContext(data_provider=columnar_provider)
    )
    pairwise_plugin = metrics_plugin.MetricsPlugin(
        base_plugin.TBContext(data_provider=pairwise_provider)
    )
    ctx = context.RequestContext()
    inline_req = data_provider_pb2.ListScalarsRequest(
        plugin_filter=data_provider_pb2.PluginFilter(
            plugin_name=scalar_metadata.PLUGIN_NAME
        ),
        skip_statistics=True,
    )
    deduped_req = data_provider_pb2.ListScalarsRequest()
    deduped_req.CopyFrom(inline_req)
    deduped_req.dedup_names = True

    def inline_rpc():
        return stub.ListScalars(inline_req)

    def deduped_rpc():
        return stub.ListScalars(deduped_req)

    def parse_inline():
        parsed = data_provider_pb2.ListScalarsResponse()
        parsed.ParseFromString(inline_wire)
        return parsed

    def parse_deduped():
        parsed = data_provider_pb2.ListScalarsResponse()
        parsed.ParseFromString(deduped_wire)
        return parsed

    def list_metadata():
        return pairwise_provider.list_scalars_metadata(
            ctx, experiment_id="", plugin_name=scalar_metadata.PLUGIN_NAME
        )

    def tag_index():
        return columnar_provider.list_scalars_tag_index(
            ctx, experiment_id="", plugin_name=scalar_metadata.PLUGIN_NAME
        )

    def pairwise_tags_impl():
        return pairwise_plugin._tags_impl(ctx, experiment="")

    def tags_impl():
        return plugin._tags_impl(ctx, experiment="")

    def serialize():
        return json.dumps(payload, allow_nan=False, separators=(",", ":"))

    inline_response = inline_rpc()
    inline_wire = inline_response.SerializeToString()
    deduped_wire = deduped_rpc().SerializeToString()
    payload = tags_impl()
    if pairwise_tags_impl() != payload:
        raise AssertionError("columnar and pairwise listings disagree")
    result = {
        "runs": len(inline_response.runs),
        "pairs": sum(len(run.tags) for run in inline_response.runs),
        "inline_rpc_bytes": len(inline_wire),
        "deduped_rpc_bytes": len(deduped_wire),
        "bytes": len(serialize().encode()),
    }
    for name, fn in [
        ("inline_rpc_ms", inline_rpc),
        ("deduped_rpc_ms", deduped_rpc),
        ("parse_inline_ms", parse_inline),
        ("parse_deduped_ms", parse_deduped),
        ("list_metadata_ms", list_metadata),
        ("tag_index_ms", tag_index),
        ("pairwise_tags_impl_ms", pairwise_tags_impl),
        ("tags_impl_ms", tags_impl),
        ("serialize_ms", serialize),
    ]:
        samples = []
        for _ in range(5):
            start = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - start) * 1000)
        result[name] = round(statistics.median(samples), 1)
    return result


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else None
    if port:
        print(json.dumps(benchmark_live(port), indent=2))
    else:
        print(
            json.dumps(
                [benchmark(runs, 52) for runs in (100, 1000, 1843)],
                indent=2,
            )
        )
