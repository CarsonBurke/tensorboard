# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Integration tests for the Metrics Plugin."""

import argparse
import collections.abc
import json
import os.path
from unittest import mock

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
from werkzeug import test, wrappers

from tensorboard import context
from tensorboard.backend.event_processing import data_provider
from tensorboard.backend.event_processing import (
    plugin_event_multiplexer as event_multiplexer,
)
from tensorboard.data import provider
from tensorboard.plugins import base_plugin
from tensorboard.plugins.image import metadata as image_metadata
from tensorboard.plugins.metrics import metrics_plugin
from tensorboard.plugins.scalar import metadata as scalar_metadata
from tensorboard.plugins.scalar import plugin_data_pb2

tf1.enable_eager_execution()


class MetricsPluginTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        self._logdir = self.get_temp_dir()
        self._multiplexer = event_multiplexer.EventMultiplexer()

        flags = argparse.Namespace(generic_data="true")
        provider = data_provider.MultiplexerDataProvider(
            self._multiplexer, self._logdir
        )
        ctx = base_plugin.TBContext(
            flags=flags,
            logdir=self._logdir,
            multiplexer=self._multiplexer,
            data_provider=provider,
        )
        self._plugin = metrics_plugin.MetricsPlugin(ctx)

    ### Writing utilities.

    def _write_scalar(self, run, tag, description=None):
        subdir = os.path.join(self._logdir, run)
        writer = tf.summary.create_file_writer(subdir)

        with writer.as_default():
            tf.summary.scalar(tag, 42, step=0, description=description)
            writer.flush()
        self._multiplexer.AddRunsFromDirectory(self._logdir)

    def _write_scalar_data(self, run, tag, data=[]):
        """Writes scalar data, starting at step 0.

        Args:
          run: string run name.
          tag: string tag name.
          data: list of scalar values to write at each step.
        """
        subdir = os.path.join(self._logdir, run)
        writer = tf.summary.create_file_writer(subdir)

        with writer.as_default():
            step = 0
            for datum in data:
                tf.summary.scalar(tag, datum, step=step)
                step += 1
            writer.flush()
        self._multiplexer.AddRunsFromDirectory(self._logdir)

    def _write_histogram(self, run, tag, description=None):
        subdir = os.path.join(self._logdir, run)
        writer = tf.summary.create_file_writer(subdir)

        with writer.as_default():
            data = tf.random.normal(shape=[3])
            tf.summary.histogram(tag, data, step=0, description=description)
            writer.flush()
        self._multiplexer.AddRunsFromDirectory(self._logdir)

    def _write_histogram_data(self, run, tag, data=[]):
        """Writes histogram data, starting at step 0.

        Args:
          run: string run name.
          tag: string tag name.
          data: list of histogram values to write at each step.
        """
        subdir = os.path.join(self._logdir, run)
        writer = tf.summary.create_file_writer(subdir)

        with writer.as_default():
            step = 0
            for datum in data:
                tf.summary.histogram(tag, datum, step=step)
                step += 1
            writer.flush()
        self._multiplexer.AddRunsFromDirectory(self._logdir)

    def _write_image(self, run, tag, samples=2, description=None):
        subdir = os.path.join(self._logdir, run)
        writer = tf.summary.create_file_writer(subdir)

        with writer.as_default():
            data = tf.random.normal(shape=[samples, 8, 8, 1])
            tf.summary.image(
                tag, data, step=0, max_outputs=samples, description=description
            )
            writer.flush()
        self._multiplexer.AddRunsFromDirectory(self._logdir)

    ### Misc utilities.

    def _clean_time_series_responses(self, responses):
        """Cleans non-deterministic data from a TimeSeriesResponse, in
        place."""
        for response in responses:
            for series in response.get("runToSeries", {}).values():
                if isinstance(series, dict):
                    # Columnar scalars; see `ScalarColumns` in http_api.md.
                    series["wallTimes"] = ["<wall_time>"] * len(
                        series["wallTimes"]
                    )
                    continue
                for datum in series:
                    if "wallTime" in datum:
                        datum["wallTime"] = "<wall_time>"
                    if "imageId" in datum:
                        datum["imageId"] = "<image_id>"

        return responses

    def _scalar_columns(self, steps, values):
        """Builds an expected `ScalarColumns` dict with cleaned wall times."""
        return {
            "steps": list(steps),
            "wallTimes": ["<wall_time>"] * len(steps),
            "values": list(values),
        }

    def _get_image_blob_key(self, run, tag, step=0, sample=0):
        """Returns a single image's blob_key after it has been written."""
        mapping = self._plugin._data_provider.read_blob_sequences(
            context.RequestContext(),
            experiment_id="expid",
            plugin_name=image_metadata.PLUGIN_NAME,
            downsample=10,
            run_tag_filter=provider.RunTagFilter(tags=[tag]),
        )
        blob_sequence_datum = mapping[run][tag][step]
        # For images, the first 2 datum values are ignored.
        return blob_sequence_datum.values[2 + sample].blob_key

    ### Actual tests.

    def test_routes_provided(self):
        """Tests that the plugin offers the correct routes."""
        routes = self._plugin.get_plugin_apps()
        self.assertIsInstance(routes["/tags"], collections.abc.Callable)

    def test_tags_empty(self):
        response = self._plugin._tags_impl(context.RequestContext(), "eid")

        expected_tags = {
            "runs": [],
            "tagToRuns": {},
            "tagDescriptions": {},
        }
        self.assertEqual(expected_tags, response["scalars"])
        self.assertEqual(expected_tags, response["histograms"])
        self.assertEqual(
            {
                "tagDescriptions": {},
                "tagRunSampledInfo": {},
            },
            response["images"],
        )

    def test_tag_pages_filter_runs_before_pagination(self):
        for run, tags in (("selected", ["a", "b"]), ("other", ["b", "c"])):
            for tag in tags:
                self._write_scalar(run, tag)
        self._multiplexer.Reload()
        client = test.Client(self._plugin._serve_tags, wrappers.Response)
        page = client.get(
            "/?run=selected&tag_offset=1&tag_limit=1",
            headers={"X-TensorBoard-Metadata-Revision": ""},
        ).json
        self.assertEqual(page["totalTags"], 2)
        self.assertEqual(page["metadata"]["scalars"]["runs"], ["selected"])
        self.assertEqual(page["metadata"]["scalars"]["tagToRuns"], {"b": [0]})
        exact = client.get("/?run=selected&run=other&tag=b&tag_limit=0").json
        self.assertEqual(exact["totalTags"], 1)
        self.assertEqual(exact["scalars"]["runs"], ["other", "selected"])
        self.assertEqual(exact["scalars"]["tagToRuns"], {"b": [0, 1]})
        empty = client.get("/?run_filter=true&tag_limit=1").json
        self.assertEqual(empty["totalTags"], 0)
        self.assertEqual(empty["scalars"]["tagToRuns"], {})
        self.assertEqual(
            client.get("/?run=selected&tag_offset=-1&tag_limit=1").status_code,
            400,
        )

    def _catalog_request(self, **overrides):
        body = dict(
            runIds=["experiment/selected"],
            query="",
            plugins=[],
            groupOffset=0,
            groupLimit=40,
            groups=[dict(name="group", offset=0, limit=1)],
            filteredOffset=0,
            filteredLimit=40,
            pinnedTags=[],
            pinnedRunIds=[],
        )
        body.update(overrides)
        return body

    def test_catalog_json_and_colab_get_keep_qualified_scope(self):
        self._write_scalar("selected", "group/tag2", "**safe**")
        self._write_scalar("selected", "group/tag10")
        self._write_scalar("other", "private/tag")
        self._multiplexer.Reload()
        client = test.Client(self._plugin._serve_catalog, wrappers.Response)
        body = self._catalog_request()
        response = client.post("/", data=json.dumps(body))
        self.assertEqual(response.status_code, 200)
        catalog = response.json
        self.assertEqual(catalog["totalCards"], 2)
        self.assertEqual(
            catalog["cards"], [dict(plugin="scalars", tag="group/tag2")]
        )
        self.assertEqual(
            catalog["metadata"]["scalars"],
            {
                "tagToRuns": {"group/tag2": ["experiment/selected"]},
                "tagDescriptions": {
                    "group/tag2": "<p><strong>safe</strong></p>"
                },
            },
        )
        self.assertEqual(
            client.get("/", query_string={"request": json.dumps(body)}).json,
            catalog,
        )
        empty = client.post(
            "/", data=json.dumps(self._catalog_request(runIds=[]))
        ).json
        self.assertEqual(empty["totalCards"], 0)
        self.assertEqual(empty["metadata"]["scalars"]["tagToRuns"], {})

    def test_catalog_rejects_malformed_requests_before_provider_access(self):
        client = test.Client(self._plugin._serve_catalog, wrappers.Response)
        with mock.patch.object(
            self._plugin._data_provider,
            "list_metrics_catalog",
            side_effect=AssertionError("invalid request reached provider"),
        ):
            for body in (
                [],
                self._catalog_request(groupOffset=-1),
                self._catalog_request(groupLimit=True),
                self._catalog_request(filteredLimit=2**64),
                self._catalog_request(filteredOffset=1.5),
                self._catalog_request(query="["),
                self._catalog_request(runIds=["unqualified"]),
                self._catalog_request(plugins=["audio"]),
                self._catalog_request(groups=[dict(name="group", offset=0)]),
                self._catalog_request(
                    groups=[dict(name="group", offset=0, limit=-1)]
                ),
            ):
                with self.subTest(body=body):
                    self.assertEqual(
                        client.post("/", data=json.dumps(body)).status_code,
                        400,
                    )
            self.assertEqual(client.post("/", data="{").status_code, 400)
            self.assertEqual(client.delete("/").status_code, 405)

    def test_tags(self):
        self._write_scalar("run1", "scalars/tagA", None)
        self._write_scalar("run1", "scalars/tagA", None)
        self._write_scalar("run1", "scalars/tagB", None)
        self._write_scalar("run2", "scalars/tagB", None)
        self._write_histogram("run1", "histograms/tagA", None)
        self._write_histogram("run1", "histograms/tagA", None)
        self._write_histogram("run1", "histograms/tagB", None)
        self._write_histogram("run2", "histograms/tagB", None)
        self._write_image("run1", "images/tagA", 1, None)
        self._write_image("run1", "images/tagA", 2, None)
        self._write_image("run1", "images/tagB", 3, None)
        self._write_image("run2", "images/tagB", 4, None)

        self._multiplexer.Reload()

        response = self._plugin._tags_impl(context.RequestContext(), "eid")

        self.assertEqual(
            {
                "runs": ["run1", "run2"],
                "tagToRuns": {
                    "scalars/tagA": [0],
                    "scalars/tagB": [0, 1],
                },
                "tagDescriptions": {},
            },
            response["scalars"],
        )
        self.assertEqual(
            {
                "runs": ["run1", "run2"],
                "tagToRuns": {
                    "histograms/tagA": [0],
                    "histograms/tagB": [0, 1],
                },
                "tagDescriptions": {},
            },
            response["histograms"],
        )
        self.assertEqual(
            {
                "tagDescriptions": {},
                "tagRunSampledInfo": {
                    "images/tagA": {"run1": {"maxSamplesPerStep": 2}},
                    "images/tagB": {
                        "run1": {"maxSamplesPerStep": 3},
                        "run2": {"maxSamplesPerStep": 4},
                    },
                },
            },
            response["images"],
        )

    def test_tags_with_descriptions(self):
        self._write_scalar("run1", "scalars/tagA", "Describing tagA")
        self._write_scalar("run1", "scalars/tagB", "Describing tagB")
        self._write_scalar("run2", "scalars/tagB", "Describing tagB")
        self._write_histogram("run1", "histograms/tagA", "Describing tagA")
        self._write_histogram("run1", "histograms/tagB", "Describing tagB")
        self._write_histogram("run2", "histograms/tagB", "Describing tagB")
        self._write_image("run1", "images/tagA", 1, "Describing tagA")
        self._write_image("run1", "images/tagB", 2, "Describing tagB")
        self._write_image("run2", "images/tagB", 3, "Describing tagB")
        self._multiplexer.Reload()

        response = self._plugin._tags_impl(context.RequestContext(), "eid")

        self.assertEqual(
            {
                "runs": ["run1", "run2"],
                "tagToRuns": {
                    "scalars/tagA": [0],
                    "scalars/tagB": [0, 1],
                },
                "tagDescriptions": {
                    "scalars/tagA": "<p>Describing tagA</p>",
                    "scalars/tagB": "<p>Describing tagB</p>",
                },
            },
            response["scalars"],
        )
        self.assertEqual(
            {
                "runs": ["run1", "run2"],
                "tagToRuns": {
                    "histograms/tagA": [0],
                    "histograms/tagB": [0, 1],
                },
                "tagDescriptions": {
                    "histograms/tagA": "<p>Describing tagA</p>",
                    "histograms/tagB": "<p>Describing tagB</p>",
                },
            },
            response["histograms"],
        )
        self.assertEqual(
            {
                "tagDescriptions": {
                    "images/tagA": "<p>Describing tagA</p>",
                    "images/tagB": "<p>Describing tagB</p>",
                },
                "tagRunSampledInfo": {
                    "images/tagA": {"run1": {"maxSamplesPerStep": 1}},
                    "images/tagB": {
                        "run1": {"maxSamplesPerStep": 2},
                        "run2": {"maxSamplesPerStep": 3},
                    },
                },
            },
            response["images"],
        )

    def test_tags_conflicting_description(self):
        self._write_scalar("run1", "scalars/tagA", None)
        self._write_scalar("run2", "scalars/tagA", "tagA is hot")
        self._write_scalar("run3", "scalars/tagA", "tagA is cold")
        self._write_scalar("run4", "scalars/tagA", "tagA is cold")
        self._write_histogram("run1", "histograms/tagA", None)
        self._write_histogram("run2", "histograms/tagA", "tagA is hot")
        self._write_histogram("run3", "histograms/tagA", "tagA is cold")
        self._write_histogram("run4", "histograms/tagA", "tagA is cold")
        self._multiplexer.Reload()

        response = self._plugin._tags_impl(context.RequestContext(), "eid")

        expected_composite_description = (
            "<h1>Multiple descriptions</h1>\n"
            "<h2>For runs: run3, run4</h2>\n"
            "<p>tagA is cold</p>\n"
            "<h2>For run: run2</h2>\n"
            "<p>tagA is hot</p>"
        )
        self.assertEqual(
            {"scalars/tagA": expected_composite_description},
            response["scalars"]["tagDescriptions"],
        )
        self.assertEqual(
            {"histograms/tagA": expected_composite_description},
            response["histograms"]["tagDescriptions"],
        )

    def test_tags_from_columnar_provider(self):
        current = plugin_data_pb2.ScalarPluginData(
            version=scalar_metadata.PROTO_VERSION
        ).SerializeToString()
        too_new = plugin_data_pb2.ScalarPluginData(
            version=scalar_metadata.PROTO_VERSION + 1
        ).SerializeToString()
        data_provider = mock.Mock(spec=provider.DataProvider)
        data_provider.list_scalars_tag_index.return_value = provider.TagIndex(
            # Providers list runs in storage order, so the response must
            # renumber them without disturbing which runs hold which tags.
            runs=["zeta", "alpha"],
            tags=["scalars/tagA", "scalars/tagB", "scalars/tagFuture"],
            contents=[current, too_new],
            run_tags=[[0, 2], [0, 1]],
            run_contents=[[0, 1], [0, 0]],
            descriptions={(0, 0): "tagA is hot", (1, 0): "tagA is cold"},
        )
        data_provider.list_tensors_metadata.return_value = {}
        data_provider.list_blob_sequences.return_value = {}
        plugin = metrics_plugin.MetricsPlugin(
            base_plugin.TBContext(data_provider=data_provider)
        )

        response = plugin._tags_impl(context.RequestContext(), "eid")

        self.assertEqual(
            {
                "runs": ["alpha", "zeta"],
                "tagToRuns": {
                    "scalars/tagA": [0, 1],
                    "scalars/tagB": [0],
                },
                "tagDescriptions": {
                    "scalars/tagA": (
                        "<h1>Multiple descriptions</h1>\n"
                        "<h2>For run: alpha</h2>\n"
                        "<p>tagA is cold</p>\n"
                        "<h2>For run: zeta</h2>\n"
                        "<p>tagA is hot</p>"
                    )
                },
            },
            response["scalars"],
        )
        data_provider.list_scalars_metadata.assert_not_called()

    def test_tags_unsafe_description(self):
        self._write_scalar("<&#run>", "scalars/<&#tag>", "<&#description>")
        self._write_histogram(
            "<&#run>", "histograms/<&#tag>", "<&#description>"
        )
        self._multiplexer.Reload()

        response = self._plugin._tags_impl(context.RequestContext(), "eid")

        self.assertEqual(
            {"scalars/<&#tag>": "<p>&lt;&amp;#description&gt;</p>"},
            response["scalars"]["tagDescriptions"],
        )
        self.assertEqual(
            {"histograms/<&#tag>": "<p>&lt;&amp;#description&gt;</p>"},
            response["histograms"]["tagDescriptions"],
        )

    def test_tags_unsafe_conflicting_description(self):
        self._write_scalar("<&#run1>", "scalars/<&#tag>", None)
        self._write_scalar("<&#run2>", "scalars/<&#tag>", "<&# is hot>")
        self._write_scalar("<&#run3>", "scalars/<&#tag>", "<&# is cold>")
        self._write_scalar("<&#run4>", "scalars/<&#tag>", "<&# is cold>")
        self._write_histogram("<&#run1>", "histograms/<&#tag>", None)
        self._write_histogram("<&#run2>", "histograms/<&#tag>", "<&# is hot>")
        self._write_histogram("<&#run3>", "histograms/<&#tag>", "<&# is cold>")
        self._write_histogram("<&#run4>", "histograms/<&#tag>", "<&# is cold>")
        self._multiplexer.Reload()

        response = self._plugin._tags_impl(context.RequestContext(), "eid")

        expected_composite_description = (
            "<h1>Multiple descriptions</h1>\n"
            "<h2>For runs: &lt;&amp;#run3&gt;, &lt;&amp;#run4&gt;</h2>\n"
            "<p>&lt;&amp;# is cold&gt;</p>\n"
            "<h2>For run: &lt;&amp;#run2&gt;</h2>\n"
            "<p>&lt;&amp;# is hot&gt;</p>"
        )
        self.assertEqual(
            {"scalars/<&#tag>": expected_composite_description},
            response["scalars"]["tagDescriptions"],
        )
        self.assertEqual(
            {"histograms/<&#tag>": expected_composite_description},
            response["histograms"]["tagDescriptions"],
        )

    def test_time_series_scalar(self):
        self._write_scalar_data("run1", "scalars/tagA", [0, 100, -200])
        self._multiplexer.Reload()

        requests = [{"plugin": "scalars", "tag": "scalars/tagA"}]
        response = self._plugin._time_series_impl(
            context.RequestContext(), "", requests
        )
        clean_response = self._clean_time_series_responses(response)

        self.assertEqual(
            [
                {
                    "plugin": "scalars",
                    "tag": "scalars/tagA",
                    "runToSeries": {
                        "run1": self._scalar_columns(
                            [0, 1, 2], [0.0, 100.0, -200.0]
                        )
                    },
                }
            ],
            clean_response,
        )

    def test_time_series_histogram(self):
        self._write_histogram_data("run1", "histograms/tagA", [0, 10])
        self._multiplexer.Reload()

        requests = [
            {"plugin": "histograms", "tag": "histograms/tagA", "run": "run1"}
        ]
        response = self._plugin._time_series_impl(
            context.RequestContext(), "", requests
        )
        clean_response = self._clean_time_series_responses(response)

        # By default 30 bins will be generated.
        bins_zero = [{"min": 0, "max": 0, "count": 0}] * 29 + [
            {"min": 0, "max": 0, "count": 1.0}
        ]
        bins_ten = [{"min": 10, "max": 10, "count": 0}] * 29 + [
            {"min": 10, "max": 10, "count": 1.0}
        ]

        self.assertEqual(
            [
                {
                    "plugin": "histograms",
                    "tag": "histograms/tagA",
                    "run": "run1",
                    "runToSeries": {
                        "run1": [
                            {
                                "wallTime": "<wall_time>",
                                "step": 0,
                                "bins": bins_zero,
                            },
                            {
                                "wallTime": "<wall_time>",
                                "step": 1,
                                "bins": bins_ten,
                            },
                        ]
                    },
                }
            ],
            clean_response,
        )

    def test_time_series_unmatching_request(self):
        self._write_scalar_data("run1", "scalars/tagA", [0, 100, -200])

        self._multiplexer.Reload()

        requests = [{"plugin": "scalars", "tag": "nothing-matches"}]
        response = self._plugin._time_series_impl(
            context.RequestContext(), "", requests
        )
        clean_response = self._clean_time_series_responses(response)

        self.assertEqual(
            [
                {
                    "plugin": "scalars",
                    "runToSeries": {},
                    "tag": "nothing-matches",
                }
            ],
            clean_response,
        )

    def test_time_series_multiple_runs(self):
        self._write_scalar_data("run1", "scalars/tagA", [0])
        self._write_scalar_data("run2", "scalars/tagA", [1])
        self._write_scalar_data("run2", "scalars/tagB", [2])

        self._multiplexer.Reload()

        requests = [{"plugin": "scalars", "tag": "scalars/tagA"}]
        response = self._plugin._time_series_impl(
            context.RequestContext(), "", requests
        )
        clean_response = self._clean_time_series_responses(response)

        self.assertEqual(
            [
                {
                    "plugin": "scalars",
                    "runToSeries": {
                        "run1": self._scalar_columns([0], [0.0]),
                        "run2": self._scalar_columns([0], [1.0]),
                    },
                    "tag": "scalars/tagA",
                }
            ],
            clean_response,
        )

    def test_time_series_multiple_requests(self):
        self._write_scalar_data("run1", "scalars/tagA", [0])
        self._write_scalar_data("run2", "scalars/tagB", [1])

        self._multiplexer.Reload()

        requests = [
            {"plugin": "scalars", "tag": "scalars/tagA"},
            {"plugin": "scalars", "tag": "scalars/tagB"},
            {"plugin": "scalars", "tag": "scalars/tagB"},
        ]
        response = self._plugin._time_series_impl(
            context.RequestContext(), "", requests
        )
        clean_response = self._clean_time_series_responses(response)

        self.assertEqual(
            [
                {
                    "plugin": "scalars",
                    "runToSeries": {
                        "run1": self._scalar_columns([0], [0.0]),
                    },
                    "tag": "scalars/tagA",
                },
                {
                    "plugin": "scalars",
                    "runToSeries": {
                        "run2": self._scalar_columns([0], [1.0]),
                    },
                    "tag": "scalars/tagB",
                },
                {
                    "plugin": "scalars",
                    "runToSeries": {
                        "run2": self._scalar_columns([0], [1.0]),
                    },
                    "tag": "scalars/tagB",
                },
            ],
            clean_response,
        )

    def test_time_series_single_request_specific_run(self):
        self._write_scalar_data("run1", "scalars/tagA", [0])
        self._write_scalar_data("run2", "scalars/tagA", [1])

        self._multiplexer.Reload()

        requests = [{"plugin": "scalars", "tag": "scalars/tagA", "run": "run2"}]
        response = self._plugin._time_series_impl(
            context.RequestContext(), "", requests
        )
        clean_response = self._clean_time_series_responses(response)

        self.assertEqual(
            [
                {
                    "plugin": "scalars",
                    "runToSeries": {
                        "run2": self._scalar_columns([0], [1.0]),
                    },
                    "tag": "scalars/tagA",
                    "run": "run2",
                }
            ],
            clean_response,
        )

    def test_time_series_empty_legacy_run_admits_all_runs(self):
        self._write_scalar_data("run1", "scalars/tagA", [0])
        self._write_scalar_data("run2", "scalars/tagA", [1])
        self._multiplexer.Reload()

        requests = [{"plugin": "scalars", "tag": "scalars/tagA", "run": ""}]
        response = self._plugin._time_series_impl(
            context.RequestContext(), "", requests
        )

        self.assertEqual(
            {"run1", "run2"}, set(response[0]["runToSeries"].keys())
        )

    def test_time_series_request_filters_runs_list(self):
        self._write_scalar_data("run1", "scalars/tagA", [0])
        self._write_scalar_data("run2", "scalars/tagA", [1])
        self._write_scalar_data("run3", "scalars/tagA", [2])

        self._multiplexer.Reload()

        requests = [
            {
                "plugin": "scalars",
                "tag": "scalars/tagA",
                "runs": ["run1", "run3"],
            }
        ]
        response = self._plugin._time_series_impl(
            context.RequestContext(), "", requests
        )
        clean_response = self._clean_time_series_responses(response)

        self.assertEqual(
            [
                {
                    "plugin": "scalars",
                    "runToSeries": {
                        "run1": self._scalar_columns([0], [0.0]),
                        "run3": self._scalar_columns([0], [2.0]),
                    },
                    "tag": "scalars/tagA",
                }
            ],
            clean_response,
        )

    def test_time_series_distinct_run_filters_stay_separate(self):
        self._write_scalar_data("run1", "scalars/tagA", [0])
        self._write_scalar_data("run2", "scalars/tagA", [1])

        self._multiplexer.Reload()

        # Requests for one plugin and tag are read in a single provider call,
        # so each response must still see only the runs its own filter admits.
        requests = [
            {"plugin": "scalars", "tag": "scalars/tagA", "runs": ["run1"]},
            {"plugin": "scalars", "tag": "scalars/tagA", "run": "run2"},
            {"plugin": "scalars", "tag": "scalars/tagA"},
        ]
        response = self._plugin._time_series_impl(
            context.RequestContext(), "", requests
        )

        self.assertEqual(
            [{"run1"}, {"run2"}, {"run1", "run2"}],
            [set(series["runToSeries"]) for series in response],
        )

    def test_time_series_empty_runs_list_returns_no_series(self):
        self._write_scalar_data("run1", "scalars/tagA", [0])
        self._multiplexer.Reload()

        requests = [{"plugin": "scalars", "tag": "scalars/tagA", "runs": []}]
        response = self._plugin._time_series_impl(
            context.RequestContext(), "", requests
        )
        clean_response = self._clean_time_series_responses(response)

        self.assertEqual(
            [
                {
                    "plugin": "scalars",
                    "runToSeries": {},
                    "tag": "scalars/tagA",
                }
            ],
            clean_response,
        )

    def test_image_data(self):
        self._write_image("run1", "images/tagA", 1, None)
        self._multiplexer.Reload()

        # Get the blob_key manually.
        image_id = self._get_image_blob_key(
            "run1", "images/tagA", step=0, sample=0
        )
        data, content_type = self._plugin._image_data_impl(
            context.RequestContext(), image_id
        )

        self.assertIsInstance(data, bytes)
        self.assertEqual(content_type, "image/png")
        self.assertGreater(len(data), 0)

    def test_time_series_bad_arguments(self):
        requests = [
            {"plugin": "images"},
            {"plugin": "unknown_plugin", "tag": "tagA"},
            {"plugin": "scalars", "tag": "tagA", "run": 123},
            {"plugin": "scalars", "tag": "tagA", "runs": "run1"},
        ]
        response = self._plugin._time_series_impl(
            context.RequestContext(), "expid", requests
        )
        errors = [
            series_response.get("error", "") for series_response in response
        ]

        self.assertEqual(
            ["Missing tag", "Invalid plugin", "Invalid run", "Invalid runs"],
            errors,
        )

    def test_image_data_from_time_series_query(self):
        self._write_image("run1", "images/tagA", samples=3)
        self._multiplexer.Reload()

        requests = [
            {
                "plugin": "images",
                "tag": "images/tagA",
                "run": "run1",
                "sample": 2,
            }
        ]
        original_response = self._plugin._time_series_impl(
            context.RequestContext(), "expid", requests
        )
        response = self._plugin._time_series_impl(
            context.RequestContext(), "expid", requests
        )
        clean_response = self._clean_time_series_responses(response)

        self.assertEqual(
            [
                {
                    "plugin": "images",
                    "tag": "images/tagA",
                    "run": "run1",
                    "sample": 2,
                    "runToSeries": {
                        "run1": [
                            {
                                "wallTime": "<wall_time>",
                                "step": 0,
                                "imageId": "<image_id>",
                            }
                        ]
                    },
                }
            ],
            clean_response,
        )

        image_id = original_response[0]["runToSeries"]["run1"][0]["imageId"]
        data, content_type = self._plugin._image_data_impl(
            context.RequestContext(), image_id
        )

        self.assertIsInstance(data, bytes)
        self.assertGreater(len(data), 0)

    def test_image_bad_request(self):
        self._write_image("run1", "images/tagA", 1, None)
        self._multiplexer.Reload()

        invalid_sample = 999
        requests = [
            {
                "plugin": "images",
                "tag": "images/tagA",
                "sample": invalid_sample,
                "run": "run1",
            },
            {"plugin": "images", "tag": "images/tagA", "run": "run1"},
            {
                "plugin": "images",
                "tag": "images/tagA",
            },
        ]
        response = self._plugin._time_series_impl(
            context.RequestContext(), "expid", requests
        )
        errors = [
            series_response.get("error", "") for series_response in response
        ]

        self.assertEqual(errors, ["", "Missing sample", "Missing run"])


if __name__ == "__main__":
    tf.test.main()
