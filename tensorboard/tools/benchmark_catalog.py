# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
"""Generate real event catalogs and measure selected HTTP queries.

Examples (always build/run with -c opt):
  bazel run -c opt //tensorboard/tools:benchmark_catalog -- generate \
      /tmp/catalog-wide --runs 1 --tags 1000000
  tensorboard --logdir /tmp/catalog-wide --port 6006
  bazel run -c opt //tensorboard/tools:benchmark_catalog -- measure \
      http://127.0.0.1:6006 --run run00000000 --tag metric00999999 --pid PID

Generation refuses to overwrite an existing directory. Measurements use the
actual run catalog, scoped tag catalog, and chart HTTP endpoints, not mocks.
Run after indexing completes; restart the server to measure warm-index startup.
The optional Linux PID adds RSS/PSS/open-file observations for the process tree.
"""

import argparse
import json
from pathlib import Path
import statistics
import time
import urllib.parse
import urllib.request

from tensorboard.compat.proto import event_pb2, summary_pb2
from tensorboard.summary.writer.record_writer import RecordWriter


def generate(path, runs, tags, steps):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    for run in range(runs):
        directory = path / ("run%08d" % run)
        directory.mkdir()
        with (directory / "events.out.tfevents.catalog-benchmark").open(
            "wb"
        ) as file:
            writer = RecordWriter(file)
            for step in range(steps):
                for tag in range(tags):
                    event = event_pb2.Event(
                        wall_time=1700000000.0 + step,
                        step=step,
                        summary=summary_pb2.Summary(
                            value=[
                                summary_pb2.Summary.Value(
                                    tag="metric%08d" % tag,
                                    simple_value=float(step),
                                )
                            ]
                        ),
                    )
                    writer.write(event.SerializeToString())
            writer.flush()
    return {
        "logdir": str(path),
        "runs": runs,
        "tags_per_run": tags,
        "series": runs * tags,
        "points": runs * tags * steps,
        "seconds": time.perf_counter() - start,
    }


def process_tree(pid):
    result = []
    pending = [pid]
    while pending:
        current = pending.pop()
        root = Path("/proc") / str(current)
        try:
            memory = {}
            for line in (root / "smaps_rollup").read_text().splitlines()[1:]:
                name, value = line.split(":", 1)
                if name in ("Rss", "Pss", "Anonymous"):
                    memory[name + "_KiB"] = int(value.split()[0])
            pending.extend(
                map(
                    int,
                    (root / "task" / str(current) / "children")
                    .read_text()
                    .split(),
                )
            )
            result.append(
                {
                    "pid": current,
                    **memory,
                    "open_files": sum(1 for _ in (root / "fd").iterdir()),
                }
            )
        except FileNotFoundError:
            continue
    return result


def request(base, route, params, post=False):
    encoded = urllib.parse.urlencode(params).encode()
    url = base.rstrip("/") + route
    if post:
        req = urllib.request.Request(url, data=encoded)
    else:
        req = urllib.request.Request(url + "?" + encoded.decode())
    with urllib.request.urlopen(req, timeout=120) as response:
        data = response.read()
        return json.loads(data), len(data)


def measure(base, run, tag, repetitions, pid):
    routes = {
        "browse_runs": ("/data/runs", {"limit": 50, "sort_by": "name"}, False),
        "find_run": ("/data/runs", {"name": run, "limit": 1}, False),
        "browse_selected_tags": (
            "/data/plugin/timeseries/tags",
            {"run": run, "tag_limit": 50},
            False,
        ),
        "find_selected_tag": (
            "/data/plugin/timeseries/tags",
            {"run": run, "tag": tag, "tag_limit": 1},
            False,
        ),
        "selected_chart": (
            "/data/plugin/timeseries/timeSeries",
            {
                "requests": json.dumps(
                    [{"plugin": "scalars", "tag": tag, "runs": [run]}]
                )
            },
            True,
        ),
    }
    results = {"url": base, "run": run, "tag": tag}
    if pid:
        results["processes_before"] = process_tree(pid)
    for name, (route, params, post) in routes.items():
        start = time.perf_counter()
        body, size = request(base, route, params, post)
        samples = [1000 * (time.perf_counter() - start)]
        for _ in range(1, repetitions):
            start = time.perf_counter()
            body, size = request(base, route, params, post)
            samples.append(1000 * (time.perf_counter() - start))
        if name == "find_run" and [item["name"] for item in body["runs"]] != [
            run
        ]:
            raise RuntimeError(
                "Selected run missing; wait for indexing before measuring"
            )
        if name == "find_selected_tag":
            metadata = body["scalars"]
            if metadata["runs"] != [run] or metadata["tagToRuns"] != {tag: [0]}:
                raise RuntimeError(
                    "Selected scalar metadata missing or unselected data returned"
                )
        if name == "selected_chart":
            if len(body) != 1 or set(body[0].get("runToSeries", {})) != {run}:
                raise RuntimeError(
                    "Selected chart missing or unselected series returned"
                )
        results[name] = {
            "first_ms": samples[0],
            "median_ms": statistics.median(samples),
            "response_bytes": size,
        }
        if name == "browse_runs":
            results["catalog_runs"] = body["total"]
        elif name == "browse_selected_tags":
            results["catalog_tags_in_selected_run"] = body["totalTags"]
    if pid:
        results["processes_after"] = process_tree(pid)
    return results


def positive(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate_parser = commands.add_parser("generate")
    generate_parser.add_argument("path")
    generate_parser.add_argument("--runs", type=positive, default=1)
    generate_parser.add_argument("--tags", type=positive, default=1000)
    generate_parser.add_argument("--steps", type=positive, default=2)
    measure_parser = commands.add_parser("measure")
    measure_parser.add_argument("url")
    measure_parser.add_argument("--run", required=True)
    measure_parser.add_argument("--tag", required=True)
    measure_parser.add_argument("--repetitions", type=positive, default=5)
    measure_parser.add_argument("--pid", type=positive)
    args = parser.parse_args()
    if args.command == "generate":
        result = generate(args.path, args.runs, args.tags, args.steps)
    else:
        result = measure(
            args.url, args.run, args.tag, args.repetitions, args.pid
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
