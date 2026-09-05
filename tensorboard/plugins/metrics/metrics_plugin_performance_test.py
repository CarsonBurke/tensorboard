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
"""Compatibility and invalidation tests without a TensorFlow dependency."""
import gzip
import json
from unittest import mock

from werkzeug import test, wrappers

from tensorboard import context, test as tb_test
from tensorboard.backend import json_util
from tensorboard.data import provider
from tensorboard.plugins import base_plugin
from tensorboard.plugins.metrics import metrics_plugin


class MetricsPerformanceTest(tb_test.TestCase):
    def setUp(self):
        super().setUp()
        self.provider = mock.Mock(spec=provider.DataProvider)
        self.provider.metadata_revision.return_value = None
        self.provider.read_scalar_columns.return_value = None
        self.plugin = metrics_plugin.MetricsPlugin(
            base_plugin.TBContext(data_provider=self.provider)
        )
        self.tags = mock.patch.object(
            self.plugin,
            "_tags_impl",
            return_value={"scalars": {"run": ["tag"]}},
        ).start()
        self.addCleanup(mock.patch.stopall)
        self.tags_client = test.Client(
            self.plugin._serve_tags, wrappers.Response
        )
        self.series_client = test.Client(
            self.plugin._serve_time_series, wrappers.Response
        )

    def test_unsupported_provider_always_lists(self):
        for _ in range(2):
            response = self.tags_client.get("/")
            self.assertEqual(response.json, self.tags.return_value)
            self.assertNotIn("ETag", response.headers)
        self.assertEqual(self.tags.call_count, 2)

    def test_revision_reuses_body_and_conditional_responses(self):
        self.provider.metadata_revision.return_value = "epoch:1"
        first = self.tags_client.get("/")
        second = self.tags_client.get(
            "/", headers={"If-None-Match": first.headers["ETag"]}
        )
        self.assertEqual(second.status_code, 304)
        self.assertEqual(second.data, b"")
        third = self.tags_client.get("/")
        self.assertEqual(first.data, third.data)
        self.assertEqual(self.tags.call_count, 1)
        self.assertEqual(self.provider.metadata_revision.call_count, 4)
        self.assertIn("private", third.headers["Cache-Control"])

    def test_versioned_protocol_and_invalidation(self):
        self.provider.metadata_revision.return_value = "epoch:1"
        headers = {"X-TensorBoard-Metadata-Revision": ""}
        first = self.tags_client.get("/", headers=headers).json
        self.assertEqual(first["metadata"], self.tags.return_value)
        headers["X-TensorBoard-Metadata-Revision"] = first["revision"]
        self.assertEqual(
            self.tags_client.get("/", headers=headers).json,
            {"revision": first["revision"], "metadata": None},
        )
        self.provider.metadata_revision.return_value = "epoch:2"
        self.tags.return_value = {"scalars": {"run": ["new tag"]}}
        changed = self.tags_client.get("/", headers=headers).json
        self.assertNotEqual(first["revision"], changed["revision"])
        self.assertEqual(changed["metadata"], self.tags.return_value)
        self.assertEqual(self.tags.call_count, 2)
        # Reconnection to a new server with the same numeric revision.
        self.provider.metadata_revision.return_value = "new-epoch:2"
        self.assertNotEqual(
            changed["revision"],
            self.tags_client.get("/", headers=headers).json["revision"],
        )

    def test_reload_during_listing_never_labels_mixed_response(self):
        self.provider.metadata_revision.side_effect = ["a", "b", "b", "b"]
        response = self.tags_client.get(
            "/", headers={"X-TensorBoard-Metadata-Revision": ""}
        )
        self.assertIsNone(response.json["revision"])
        self.assertNotIn("ETag", response.headers)
        self.tags_client.get("/")
        self.assertEqual(self.tags.call_count, 2)

    def test_revision_check_reauthorizes_and_scopes_cache(self):
        self.provider.metadata_revision.side_effect = (
            lambda ctx, **_: ctx.client_feature_flags["view"]
        )
        for view in ["alice", "bob", "alice"]:
            env = {}
            context.set_in_environ(
                env, context.RequestContext(client_feature_flags={"view": view})
            )
            self.tags.return_value = {"view": view}
            response = self.tags_client.get("/", environ_overrides=env)
            self.assertEqual(response.json, {"view": view})
        self.assertEqual(self.tags.call_count, 2)
        self.provider.metadata_revision.side_effect = RuntimeError(
            "access revoked"
        )
        with self.assertRaisesRegex(RuntimeError, "access revoked"):
            self.tags_client.get("/")

    def test_cache_is_bounded(self):
        for revision in range(12):
            self.provider.metadata_revision.return_value = str(revision)
            self.tags_client.get("/")
        self.assertEqual(len(self.plugin._tag_cache), 8)
        self.assertEqual(
            self.plugin._tag_cache_bytes,
            sum(map(len, self.plugin._tag_cache.values())),
        )

    def test_scalar_json_matches_legacy_cleansing_with_both_providers(self):
        points = [
            provider.ScalarDatum(step=2**60 + i, wall_time=wt, value=value)
            for i, (wt, value) in enumerate(
                [
                    (1.25, 3.5),
                    (float("nan"), float("inf")),
                    (float("-inf"), float("nan")),
                    (-0.0, -0.0),
                    (2.0, float("-inf")),
                ]
            )
        ]
        self.provider.read_scalars.return_value = {
            "run/é": {"loss": points},
            "empty": {"loss": []},
            "missing": {},
        }
        requests = [
            {"plugin": "scalars", "tag": "loss", "runs": ["run/é", "empty"]},
            {"plugin": {"invalid": float("inf")}, "tag": "loss"},
        ]
        expected = json_util.Cleanse(
            self.plugin._time_series_impl(
                context.RequestContext(), "", requests
            )
        )
        for columnar in (False, True):
            if columnar:
                self.provider.read_scalar_columns.return_value = {
                    "run/é": {
                        "loss": provider.ScalarColumnData(
                            [p.step for p in points],
                            [p.wall_time for p in points],
                            [p.value for p in points],
                        )
                    },
                    "empty": {"loss": provider.ScalarColumnData([], [], [])},
                    "missing": {},
                }
            for compressed in (False, True):
                response = self.series_client.post(
                    "/",
                    data={"requests": json.dumps(requests)},
                    headers={
                        "Accept-Encoding": "gzip" if compressed else "identity"
                    },
                )
                body = (
                    gzip.decompress(response.data)
                    if compressed
                    else response.data
                )
                self.assertEqual(json.loads(body), expected)
                self.assertIn(b'"step": 1152921504606846976', body)
                self.assertIn(b'"value": -0.0', body)


if __name__ == "__main__":
    tb_test.main()
