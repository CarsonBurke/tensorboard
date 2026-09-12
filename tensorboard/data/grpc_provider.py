# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A data provider that talks to a gRPC server."""

import collections
import contextlib

import grpc

from tensorboard.util import tensor_util
from tensorboard.util import timing
from tensorboard import errors
from tensorboard.data import provider
from tensorboard.data.proto import data_provider_pb2
from tensorboard.data.proto import data_provider_pb2_grpc


def make_stub(channel):
    """Wraps a gRPC channel with a service stub."""
    return data_provider_pb2_grpc.TensorBoardDataProviderStub(channel)


class GrpcDataProvider(provider.DataProvider):
    """Data provider that talks over gRPC."""

    def __init__(self, addr, stub):
        """Initializes a GrpcDataProvider.

        Args:
          addr: String address of the remote peer. Used cosmetically for
            data location.
          stub: `data_provider_pb2_grpc.TensorBoardDataProviderStub`
            value. See `make_stub` to construct one from a channel.
        """
        self._addr = addr
        self._stub = stub

    def __str__(self):
        return "GrpcDataProvider(addr=%r)" % self._addr

    def experiment_metadata(self, ctx, *, experiment_id):
        req = data_provider_pb2.GetExperimentRequest()
        req.experiment_id = experiment_id
        with _translate_grpc_error():
            res = self._stub.GetExperiment(req)
        res = provider.ExperimentMetadata(
            data_location=res.data_location,
            experiment_name=res.name,
            experiment_description=res.description,
            creation_time=_timestamp_proto_to_float(res.creation_time),
        )
        return res

    def metadata_revision(self, ctx, *, experiment_id):
        # Like the other RPCs, this provider exposes the local server's entire
        # view. Older servers omit this optional field and remain uncached.
        req = data_provider_pb2.GetExperimentRequest(
            experiment_id=experiment_id
        )
        with _translate_grpc_error():
            res = self._stub.GetExperiment(req)
        return res.metadata_revision or None

    def list_plugins(self, ctx, *, experiment_id):
        req = data_provider_pb2.ListPluginsRequest()
        req.experiment_id = experiment_id
        with _translate_grpc_error():
            res = self._stub.ListPlugins(req)
        return [p.name for p in res.plugins]

    def list_runs(self, ctx, *, experiment_id):
        req = data_provider_pb2.ListRunsRequest()
        req.experiment_id = experiment_id
        with _translate_grpc_error():
            res = self._stub.ListRuns(req)
        return [
            provider.Run(
                run_id=run.name,
                run_name=run.name,
                start_time=run.start_time,
            )
            for run in res.runs
        ]

    def list_runs_page(
        self,
        ctx,
        *,
        experiment_id,
        query="",
        offset=0,
        limit=0,
        sort_by="start_time",
        descending=False,
        names=None,
        query_prefix="",
        session_ranks=None,
        default_rank=0,
    ):
        req = data_provider_pb2.ListRunsRequest(
            experiment_id=experiment_id,
            query=query,
            offset=offset,
            limit=limit,
            sort_by=sort_by,
            descending=descending,
            query_prefix=query_prefix,
            session_ranks=session_ranks or (),
            default_rank=default_rank,
        )
        if names is not None:
            req.names.names[:] = sorted(set(names))
        with _translate_grpc_error():
            res = self._stub.ListRuns(req)
        # Older servers ignore window fields and omit the newly added total.
        # Keep their full-list protocol behind the provider's existing paging
        # implementation instead of reporting a populated catalog as empty.
        if res.runs and not res.total:
            return super().list_runs_page(
                ctx,
                experiment_id=experiment_id,
                query=query,
                offset=offset,
                limit=limit,
                sort_by=sort_by,
                descending=descending,
                names=names,
                query_prefix=query_prefix,
                session_ranks=session_ranks,
                default_rank=default_rank,
            )
        return provider.RunPage(
            [
                provider.Run(
                    run_id=run.name,
                    run_name=run.name,
                    start_time=run.start_time,
                )
                for run in res.runs
            ],
            res.total,
        )

    def list_tags_page(
        self,
        ctx,
        *,
        experiment_id,
        plugin_name,
        data_class,
        run_tag_filter=None,
        query="",
        offset=0,
        limit=0,
    ):
        if data_class == "scalars":
            req = data_provider_pb2.ListScalarsRequest(
                skip_statistics=True, dedup_names=True
            )
            rpc = self._stub.ListScalars
            convert = lambda res: self._scalar_mapping(res, True)
        elif data_class == "tensors":
            req = data_provider_pb2.ListTensorsRequest(skip_statistics=True)
            rpc = self._stub.ListTensors
            convert = lambda res: self._tensor_mapping(res, True)
        elif data_class == "blob_sequences":
            req = data_provider_pb2.ListBlobSequencesRequest()
            rpc = self._stub.ListBlobSequences
            convert = self._blob_mapping
        else:
            raise ValueError("Unknown data class: %r" % data_class)
        req.experiment_id = experiment_id
        req.plugin_filter.plugin_name = plugin_name
        _populate_rtf(run_tag_filter, req.run_tag_filter)
        req.run_tag_filter.tag_query = query
        req.run_tag_filter.tag_offset = offset
        req.run_tag_filter.tag_limit = limit
        with _translate_grpc_error():
            res = rpc(req)
        return provider.TagPage(convert(res), res.total_tags)

    def list_metrics_catalog(self, ctx, *, request):
        req = data_provider_pb2.MetricsCatalogRequest(
            query="(?i)" + request["query"] if request["query"] else "",
            plugins=request["plugins"],
            group_offset=request["groupOffset"],
            group_limit=request["groupLimit"],
            filtered_offset=request["filteredOffset"],
            filtered_limit=request["filteredLimit"],
            pinned_tags=sorted(set(request["pinnedTags"])),
        )
        for field, run_ids in (
            (req.runs, request["runIds"]),
            (req.pinned_runs, request["pinnedRunIds"]),
        ):
            for run_id in sorted(set(run_ids)):
                experiment, name = run_id.split("/", 1)
                field.add(experiment_id=experiment, name=name)
        for group in request["groups"]:
            req.groups.add(
                name=group["name"],
                offset=group["offset"],
                limit=group["limit"],
            )
        with _translate_grpc_error():
            try:
                res = self._stub.ListMetricsCatalog(req)
            except grpc.RpcError as error:
                if error.code() != grpc.StatusCode.UNIMPLEMENTED:
                    raise
                return super().list_metrics_catalog(ctx, request=request)
        cards = []
        for item in res.cards:
            card = {"plugin": item.plugin, "tag": item.tag}
            if item.plugin != "scalars":
                card["runId"] = item.run_id
            if item.plugin == "images":
                card.update(sample=item.sample, numSample=item.num_sample)
            cards.append(card)
        return {
            "groups": [
                {"name": group.name, "totalCards": group.total_cards}
                for group in res.groups
            ],
            "totalGroups": res.total_groups,
            "groupOffset": res.group_offset,
            "cards": cards,
            "totalCards": res.total_cards,
            "metadata": provider.metrics_catalog_metadata(
                (
                    series.plugin,
                    series.tag,
                    series.run_id,
                    series.description,
                    series.max_samples,
                )
                for series in res.series
            ),
        }

    def list_scalars(
        self, ctx, *, experiment_id, plugin_name, run_tag_filter=None
    ):
        return self._list_scalars(
            experiment_id, plugin_name, run_tag_filter, skip_statistics=False
        )

    def list_scalars_metadata(
        self, ctx, *, experiment_id, plugin_name, run_tag_filter=None
    ):
        return self._list_scalars(
            experiment_id, plugin_name, run_tag_filter, skip_statistics=True
        )

    def list_scalars_tag_index(self, ctx, *, experiment_id, plugin_name):
        res = self._list_scalars_response(
            experiment_id,
            plugin_name,
            run_tag_filter=None,
            skip_statistics=True,
        )
        return _build_tag_index(res)

    @timing.log_latency
    def _list_scalars_response(
        self, experiment_id, plugin_name, run_tag_filter, skip_statistics
    ):
        with timing.log_latency("build request"):
            req = data_provider_pb2.ListScalarsRequest()
            req.experiment_id = experiment_id
            req.plugin_filter.plugin_name = plugin_name
            req.skip_statistics = skip_statistics
            # Servers that predate this field answer with names inline; both
            # forms are handled below.
            req.dedup_names = True
            _populate_rtf(run_tag_filter, req.run_tag_filter)
        with timing.log_latency("_stub.ListScalars"):
            with _translate_grpc_error():
                return self._stub.ListScalars(req)

    @timing.log_latency
    def _list_scalars(
        self, experiment_id, plugin_name, run_tag_filter, skip_statistics
    ):
        res = self._list_scalars_response(
            experiment_id, plugin_name, run_tag_filter, skip_statistics
        )
        return self._scalar_mapping(res, skip_statistics)

    @staticmethod
    def _scalar_mapping(res, skip_statistics):
        with timing.log_latency("build result"):
            tag_names = res.tag_names
            metadata_table = res.summary_metadata_table
            result = {}
            for run_entry in res.runs:
                tags = {}
                result[run_entry.run_name] = tags
                for tag_entry in run_entry.tags:
                    time_series = tag_entry.metadata
                    # Each submessage access allocates a wrapper; with tens of
                    # thousands of tags, fetch it once per entry.
                    summary_metadata = (
                        metadata_table[time_series.summary_metadata_index]
                        if metadata_table
                        else time_series.summary_metadata
                    )
                    tag_name = (
                        tag_names[tag_entry.tag_index]
                        if tag_names
                        else tag_entry.tag_name
                    )
                    tags[tag_name] = provider.ScalarTimeSeries(
                        max_step=(
                            None if skip_statistics else time_series.max_step
                        ),
                        max_wall_time=(
                            None
                            if skip_statistics
                            else time_series.max_wall_time
                        ),
                        plugin_content=summary_metadata.plugin_data.content,
                        description=summary_metadata.summary_description,
                        display_name=summary_metadata.display_name,
                    )
            return result

    @timing.log_latency
    def read_scalars(
        self,
        ctx,
        *,
        experiment_id,
        plugin_name,
        downsample=None,
        run_tag_filter=None,
    ):
        columns = self.read_scalar_columns(
            ctx,
            experiment_id=experiment_id,
            plugin_name=plugin_name,
            downsample=downsample,
            run_tag_filter=run_tag_filter,
        )
        return {
            run: {
                tag: [
                    provider.ScalarDatum(step=step, wall_time=wt, value=value)
                    for step, wt, value in zip(d.steps, d.wall_times, d.values)
                ]
                for tag, d in tags.items()
            }
            for run, tags in columns.items()
        }

    @timing.log_latency
    def read_scalar_columns(
        self,
        ctx,
        *,
        experiment_id,
        plugin_name,
        downsample=None,
        run_tag_filter=None,
    ):
        with timing.log_latency("build request"):
            req = data_provider_pb2.ReadScalarsRequest()
            req.experiment_id = experiment_id
            req.plugin_filter.plugin_name = plugin_name
            _populate_rtf(run_tag_filter, req.run_tag_filter)
            req.downsample.num_points = downsample
        with timing.log_latency("_stub.ReadScalars"):
            with _translate_grpc_error():
                res = self._stub.ReadScalars(req)
        with timing.log_latency("build result"):
            result = {}
            for run_entry in res.runs:
                tags = {}
                result[run_entry.run_name] = tags
                for tag_entry in run_entry.tags:
                    d = tag_entry.data
                    tags[tag_entry.tag_name] = provider.ScalarColumnData(
                        d.step, d.wall_time, d.value
                    )
            return result

    @timing.log_latency
    def read_last_scalars(
        self,
        ctx,
        *,
        experiment_id,
        plugin_name,
        run_tag_filter=None,
    ):
        with timing.log_latency("build request"):
            req = data_provider_pb2.ReadScalarsRequest()
            req.experiment_id = experiment_id
            req.plugin_filter.plugin_name = plugin_name
            _populate_rtf(run_tag_filter, req.run_tag_filter)
            # `ReadScalars` always includes the most recent datum, therefore
            # downsampling to one means fetching the latest value.
            req.downsample.num_points = 1
        with timing.log_latency("_stub.ReadScalars"):
            with _translate_grpc_error():
                res = self._stub.ReadScalars(req)
        with timing.log_latency("build result"):
            result = collections.defaultdict(dict)
            for run_entry in res.runs:
                run_name = run_entry.run_name
                for tag_entry in run_entry.tags:
                    d = tag_entry.data
                    # There should be no more than one datum in
                    # `tag_entry.data` since downsample was set to 1.
                    for step, wt, value in zip(d.step, d.wall_time, d.value):
                        result[run_name][
                            tag_entry.tag_name
                        ] = provider.ScalarDatum(
                            step=step,
                            wall_time=wt,
                            value=value,
                        )
            return result

    def list_tensors(
        self, ctx, *, experiment_id, plugin_name, run_tag_filter=None
    ):
        return self._list_tensors(
            experiment_id, plugin_name, run_tag_filter, skip_statistics=False
        )

    def list_tensors_metadata(
        self, ctx, *, experiment_id, plugin_name, run_tag_filter=None
    ):
        return self._list_tensors(
            experiment_id, plugin_name, run_tag_filter, skip_statistics=True
        )

    @timing.log_latency
    def _list_tensors(
        self, experiment_id, plugin_name, run_tag_filter, skip_statistics
    ):
        with timing.log_latency("build request"):
            req = data_provider_pb2.ListTensorsRequest()
            req.experiment_id = experiment_id
            req.plugin_filter.plugin_name = plugin_name
            req.skip_statistics = skip_statistics
            _populate_rtf(run_tag_filter, req.run_tag_filter)
        with timing.log_latency("_stub.ListTensors"):
            with _translate_grpc_error():
                res = self._stub.ListTensors(req)
        return self._tensor_mapping(res, skip_statistics)

    @staticmethod
    def _tensor_mapping(res, skip_statistics):
        with timing.log_latency("build result"):
            result = {}
            for run_entry in res.runs:
                tags = {}
                result[run_entry.run_name] = tags
                for tag_entry in run_entry.tags:
                    time_series = tag_entry.metadata
                    summary_metadata = time_series.summary_metadata
                    tags[tag_entry.tag_name] = provider.TensorTimeSeries(
                        max_step=(
                            None if skip_statistics else time_series.max_step
                        ),
                        max_wall_time=(
                            None
                            if skip_statistics
                            else time_series.max_wall_time
                        ),
                        plugin_content=summary_metadata.plugin_data.content,
                        description=summary_metadata.summary_description,
                        display_name=summary_metadata.display_name,
                    )
            return result

    @timing.log_latency
    def read_tensors(
        self,
        ctx,
        *,
        experiment_id,
        plugin_name,
        downsample=None,
        run_tag_filter=None,
    ):
        with timing.log_latency("build request"):
            req = data_provider_pb2.ReadTensorsRequest()
            req.experiment_id = experiment_id
            req.plugin_filter.plugin_name = plugin_name
            _populate_rtf(run_tag_filter, req.run_tag_filter)
            req.downsample.num_points = downsample
        with timing.log_latency("_stub.ReadTensors"):
            with _translate_grpc_error():
                res = self._stub.ReadTensors(req)
        with timing.log_latency("build result"):
            result = {}
            for run_entry in res.runs:
                tags = {}
                result[run_entry.run_name] = tags
                for tag_entry in run_entry.tags:
                    series = []
                    tags[tag_entry.tag_name] = series
                    d = tag_entry.data
                    for step, wt, value in zip(d.step, d.wall_time, d.value):
                        point = provider.TensorDatum(
                            step=step,
                            wall_time=wt,
                            numpy=tensor_util.make_ndarray(value),
                        )
                        series.append(point)
            return result

    @timing.log_latency
    def list_blob_sequences(
        self, ctx, experiment_id, plugin_name, run_tag_filter=None
    ):
        with timing.log_latency("build request"):
            req = data_provider_pb2.ListBlobSequencesRequest()
            req.experiment_id = experiment_id
            req.plugin_filter.plugin_name = plugin_name
            _populate_rtf(run_tag_filter, req.run_tag_filter)
        with timing.log_latency("_stub.ListBlobSequences"):
            with _translate_grpc_error():
                res = self._stub.ListBlobSequences(req)
        return self._blob_mapping(res)

    @staticmethod
    def _blob_mapping(res):
        with timing.log_latency("build result"):
            result = {}
            for run_entry in res.runs:
                tags = {}
                result[run_entry.run_name] = tags
                for tag_entry in run_entry.tags:
                    time_series = tag_entry.metadata
                    tags[tag_entry.tag_name] = provider.BlobSequenceTimeSeries(
                        max_step=time_series.max_step,
                        max_wall_time=time_series.max_wall_time,
                        max_length=time_series.max_length,
                        plugin_content=time_series.summary_metadata.plugin_data.content,
                        description=time_series.summary_metadata.summary_description,
                        display_name=time_series.summary_metadata.display_name,
                    )
            return result

    @timing.log_latency
    def read_blob_sequences(
        self,
        ctx,
        experiment_id,
        plugin_name,
        downsample=None,
        run_tag_filter=None,
    ):
        with timing.log_latency("build request"):
            req = data_provider_pb2.ReadBlobSequencesRequest()
            req.experiment_id = experiment_id
            req.plugin_filter.plugin_name = plugin_name
            _populate_rtf(run_tag_filter, req.run_tag_filter)
            req.downsample.num_points = downsample
        with timing.log_latency("_stub.ReadBlobSequences"):
            with _translate_grpc_error():
                res = self._stub.ReadBlobSequences(req)
        with timing.log_latency("build result"):
            result = {}
            for run_entry in res.runs:
                tags = {}
                result[run_entry.run_name] = tags
                for tag_entry in run_entry.tags:
                    series = []
                    tags[tag_entry.tag_name] = series
                    d = tag_entry.data
                    for step, wt, blob_sequence in zip(
                        d.step, d.wall_time, d.values
                    ):
                        values = []
                        for ref in blob_sequence.blob_refs:
                            values.append(
                                provider.BlobReference(
                                    blob_key=ref.blob_key, url=ref.url or None
                                )
                            )
                        point = provider.BlobSequenceDatum(
                            step=step, wall_time=wt, values=tuple(values)
                        )
                        series.append(point)
            return result

    @timing.log_latency
    def read_blob(self, ctx, blob_key):
        with timing.log_latency("build request"):
            req = data_provider_pb2.ReadBlobRequest()
            req.blob_key = blob_key
        with timing.log_latency("list(_stub.ReadBlob)"):
            with _translate_grpc_error():
                responses = list(self._stub.ReadBlob(req))
        with timing.log_latency("build result"):
            return b"".join(res.data for res in responses)


def _build_tag_index(res):
    """Builds a `provider.TagIndex` from a `ListScalarsResponse`.

    Servers honoring `dedup_names` supply the name and metadata tables, which
    become the index spaces directly; older servers repeat names per entry, so
    the tables are built here instead.
    """
    deduped = bool(res.tag_names)
    if deduped:
        tags = list(res.tag_names)
        metadata = list(res.summary_metadata_table)
    else:
        tags = []
        metadata = []
    tag_indices = {}
    metadata_indices = {}
    contents = [md.plugin_data.content for md in metadata]
    entry_descriptions = [md.summary_description for md in metadata]
    runs = []
    run_tags = []
    run_contents = []
    descriptions = {}
    for run_entry in res.runs:
        run_index = len(runs)
        runs.append(run_entry.run_name)
        entry_tags = []
        entry_contents = []
        for tag_entry in run_entry.tags:
            if deduped:
                tag_index = tag_entry.tag_index
                content_index = tag_entry.metadata.summary_metadata_index
            else:
                tag_name = tag_entry.tag_name
                tag_index = tag_indices.get(tag_name)
                if tag_index is None:
                    tag_index = len(tags)
                    tag_indices[tag_name] = tag_index
                    tags.append(tag_name)
                summary_metadata = tag_entry.metadata.summary_metadata
                content = summary_metadata.plugin_data.content
                description = summary_metadata.summary_description
                content_index = metadata_indices.get((content, description))
                if content_index is None:
                    content_index = len(contents)
                    metadata_indices[(content, description)] = content_index
                    contents.append(content)
                    entry_descriptions.append(description)
            entry_tags.append(tag_index)
            entry_contents.append(content_index)
            if entry_descriptions[content_index]:
                descriptions[(run_index, tag_index)] = entry_descriptions[
                    content_index
                ]
        run_tags.append(entry_tags)
        run_contents.append(entry_contents)
    return provider.TagIndex(
        runs=runs,
        tags=tags,
        contents=contents,
        run_tags=run_tags,
        run_contents=run_contents,
        descriptions=descriptions,
    )


@contextlib.contextmanager
def _translate_grpc_error():
    try:
        yield
    except grpc.RpcError as e:
        if e.code() == grpc.StatusCode.INVALID_ARGUMENT:
            raise errors.InvalidArgumentError(e.details())
        if e.code() == grpc.StatusCode.NOT_FOUND:
            raise errors.NotFoundError(e.details())
        if e.code() == grpc.StatusCode.PERMISSION_DENIED:
            raise errors.PermissionDeniedError(e.details())
        raise


def _populate_rtf(run_tag_filter, rtf_proto):
    """Copies `run_tag_filter` into `rtf_proto`."""
    if run_tag_filter is None:
        return
    if run_tag_filter.runs is not None:
        rtf_proto.runs.names[:] = sorted(run_tag_filter.runs)
    if run_tag_filter.tags is not None:
        rtf_proto.tags.names[:] = sorted(run_tag_filter.tags)


def _timestamp_proto_to_float(ts):
    """Converts `timestamp_pb2.Timestamp` to float seconds since epoch."""
    return ts.ToNanoseconds() / 1e9
