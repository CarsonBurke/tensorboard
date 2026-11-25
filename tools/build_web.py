#!/usr/bin/env python3
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
"""Builds TensorBoard's web assets without Bazel.

Produces the same runtime artifacts as `//tensorboard:webfiles` and
`//tensorboard/plugins/projector/tf_projector_plugin:projector_assets`:

    tensorboard/webfiles.zip
    tensorboard/plugins/projector/tf_projector_plugin/projector_binary.{html,js}

Pipeline, in dependency order:

 1. Fetch the non-npm assets (Roboto, Material icons) into out/downloads.
 2. Write the sources Bazel generates with genrules (`genrules`).
 3. Compile every non-partial .scss into out/genfiles: components name the
    compiled stylesheet in `styleUrls`, which the Angular compiler resolves
    through the tsconfig `rootDirs`, like Bazel's output tree.
 4. Bundle the Polymer app, the two chart web workers and the projector with
    esbuild (the `tf_js_binary` equivalents), then write the worker URLs, with
    content digests, into the TypeScript that loads them.
 5. Build the Angular app with `ng build` (angular.json) and convert its ES
    module output into a classic-script IIFE, since index.js concatenates it
    with other classic scripts.
 6. Inline HTML imports, stylesheets and scripts like Vulcanize.java and zip
    the result.

Usage:
    build_web.py build [--dev]
    build_web.py watch [-- TENSORBOARD_ARGS...]

`watch` rebuilds incrementally on source changes and, given arguments, runs
TensorBoard with them and restarts it after each rebuild (TensorBoard reads
webfiles.zip once, at startup).
"""

import argparse
import base64
import contextlib
import hashlib
import html.parser
import io
import json
import os
import posixpath
import re
import signal
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
WEB_OUT = OUT / "web"
NG_OUT = OUT / "webapp"
# Generated sources that only Sass and the Angular compiler read; both search
# here as well as in the source tree (`--load-path`, tsconfig `rootDirs`).
GENFILES = OUT / "genfiles"
DOWNLOADS = OUT / "downloads"
NODE_BIN = ROOT / "node_modules" / ".bin"
NODE_MODULES = ROOT / "node_modules"
THIRD_PARTY_WEB = ROOT / "tools" / "third_party_web"
# Not .bin/esbuild, which may link to Angular's esbuild-wasm.
ESBUILD = str(NODE_MODULES / "esbuild" / "bin" / "esbuild")

WEBFILES_ZIP = ROOT / "tensorboard" / "webfiles.zip"
PROJECTOR_DIR = ROOT / "tensorboard/plugins/projector/tf_projector_plugin"

LINE_CHART = "tensorboard/webapp/widgets/line_chart_v2"
DEBUGGER = "tensorboard/plugins/debugger_v2/tf_debugger_v2_plugin"

# esbuild entry points, mirroring the `tf_js_binary` targets. Keys name the
# output files in out/web.
ESBUILD_ENTRY_POINTS = {
    "chart_worker": f"{LINE_CHART}/lib/worker/worker_chart_bridge.ts",
    "smoothing_worker": f"{LINE_CHART}/smoothing_worker_bridge.ts",
    "polymer3_lib_binary_no_shim": "tensorboard/components/polymer3_lib.ts",
    "standalone_bundle": "tensorboard/plugins/projector/vz_projector/bundle.ts",
}

# Worker bundles, and the generated module that holds each one's URL with a
# digest query parameter (`tf_resource_digest_suffixer`).
WORKER_RESOURCES = {
    "chart_worker": (f"{LINE_CHART}/lib/worker/worker_resource", "%CHART_WORKER.js%"),
    "smoothing_worker": (f"{LINE_CHART}/smoothing_resource", "%SMOOTHING_WORKER.js%"),
}

# Python helpers that the Bazel genrules run; reused as-is.
INLINE_FILE_CONTENT = "tensorboard/tools/inline_file_content.py"
INLINE_IMAGES = "tensorboard/defs/inline_images.py"
DIGEST_SUFFIXER = "tensorboard/defs/internal/resource_digest_suffixer.py"
MAT_BUNDLE_ICON_SVG = "tensorboard/tools/mat_bundle_icon_svg.py"


def genrules():
    """Returns (output, inputs, command) for the sources Bazel generates.

    Outputs are relative to the source tree, or absolute in GENFILES when only
    Sass and the Angular compiler read them. (Generated TypeScript has to be in
    the source tree for esbuild to resolve imports of it.) Commands write the
    output to stdout.
    """
    theme = "tensorboard/webapp/theme"
    plottable = ["tensorboard/components/polymer/plottable-style.uninlined.ts",
                 "node_modules/plottable/plottable.css"]
    inactive = f"{DEBUGGER}/views/inactive"
    inactive_inputs = [
        f"{inactive}/inactive_component.uninlined.ng.html",
        *sorted(str(p.relative_to(ROOT)) for p in (ROOT / inactive / "images").glob("*.png")),
    ]
    return [
        (GENFILES / theme / "_tb_theme.scss",
         [f"{theme}/_variable.scss", f"{theme}/_tb_theme.template.scss"],
         ["cat", f"{theme}/_variable.scss", f"{theme}/_tb_theme.template.scss"]),
        ("tensorboard/components/polymer/plottable-style.ts",
         [INLINE_FILE_CONTENT, *plottable],
         [sys.executable, INLINE_FILE_CONTENT, *plottable]),
        (f"{DEBUGGER}/tf_dtypes.ts",
         [f"{DEBUGGER}/extract_dtypes_from_python.py",
          "tensorboard/compat/tensorflow_stub/dtypes.py"],
         [sys.executable, "-m",
          "tensorboard.plugins.debugger_v2.tf_debugger_v2_plugin.extract_dtypes_from_python",
          "/dev/stdout"]),
        (GENFILES / inactive / "inactive_component.ng.html",
         [INLINE_IMAGES, *inactive_inputs],
         [sys.executable, INLINE_IMAGES, *inactive_inputs]),
    ]


def log(msg):
    print(f"[build_web] {msg}", file=sys.stderr, flush=True)


def run(cmd, **kwargs):
    subprocess.run(cmd, cwd=ROOT, check=True, **kwargs)


def write_if_changed(path, content):
    """Writes `content`, leaving the file (and its mtime) alone if unchanged.

    Keeps watchers (ng, sass) from rebuilding for no-op regenerations.
    """
    path = Path(path)
    data = content.encode() if isinstance(content, str) else content
    if path.exists() and path.read_bytes() == data:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    return True


# --- 1. Third-party assets -------------------------------------------------


def third_party_manifest():
    with open(THIRD_PARTY_WEB / "files.json") as f:
        manifest = json.load(f)
    manifest.pop("//")
    return manifest


def fetch_third_party():
    for group, files in third_party_manifest().items():
        for name, spec in files.items():
            dest = DOWNLOADS / group / name
            if dest.exists() and _sha256(dest.read_bytes()) == spec["sha256"]:
                continue
            log(f"fetching {spec['url']}")
            request = urllib.request.Request(
                spec["url"], headers={"User-Agent": "tensorboard-build"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
            if _sha256(data) != spec["sha256"]:
                raise RuntimeError(f"sha256 mismatch for {spec['url']}")
            write_if_changed(dest, data)


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


# --- 2. Generated sources --------------------------------------------------


def generate_sources():
    for output, _, cmd in genrules():
        result = subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.PIPE)
        write_if_changed(ROOT / output, result.stdout)


def write_worker_resources():
    """Points the worker loaders at the current worker bundles."""
    changed = False
    for bundle, (module, handlebar) in WORKER_RESOURCES.items():
        result = subprocess.run(
            [sys.executable, DIGEST_SUFFIXER, f"{module}.template.ts",
             handlebar, str(WEB_OUT / f"{bundle}.js")],
            cwd=ROOT, check=True, stdout=subprocess.PIPE,
        )
        changed |= write_if_changed(ROOT / f"{module}.ts", result.stdout)
    return changed


# --- 3. Sass ---------------------------------------------------------------


def sass_args(watch=False):
    pairs = [
        f"{p.relative_to(ROOT)}:{(GENFILES / p.relative_to(ROOT)).with_suffix('.css')}"
        for p in sorted((ROOT / "tensorboard").rglob("*.scss"))
        if not p.name.startswith("_")
    ]
    return [
        # The package.json pin, sass 1.69.5 like rules_sass. (.bin/sass may
        # link to a newer copy that Angular's tooling nests.)
        "node",
        str(NODE_MODULES / "sass/sass.js"),
        "--no-source-map",
        "--style=compressed",  # rules_sass's default output style
        # Bazel's include paths: the workspace root (for
        # `@use 'tensorboard/webapp/theme/tb_theme'`), generated files and
        # node_modules.
        "--load-path=.",
        f"--load-path={GENFILES}",
        "--load-path=node_modules",
        "--quiet-deps",
        *(["--watch"] if watch else []),
        *pairs,
    ]


# --- 4. esbuild bundles ------------------------------------------------------


def esbuild_args(dev, watch=False):
    return [
        ESBUILD,
        *[f"{name}={src}" for name, src in ESBUILD_ENTRY_POINTS.items()],
        f"--outdir={WEB_OUT}",
        "--bundle",
        # "iife" rather than "esm": an ES module run as a classic script
        # declares its top-level symbols as globals (see `tf_js_binary`).
        "--format=iife",
        # The @bazel/esbuild default. (The Angular app needs ES2022 anyway, but
        # this keeps the bundles closest to Bazel's.)
        "--target=es2015",
        # From `tf_js_binary`; the esbuild defaults pull in Node-only builds of
        # some libraries.
        "--main-fields=browser,es2015,module,jsnext:main,main",
        # Polymer property accessors must not be shadowed by class fields;
        # Bazel's tsc emits ES2020 with assignment semantics.
        '--tsconfig-raw={"compilerOptions": {"experimentalDecorators": true,'
        ' "useDefineForClassFields": false}}',
        "--log-level=warning",
        *([] if dev else ["--minify"]),
        # Plain --watch exits when stdin closes, i.e. when this script dies.
        *(["--watch"] if watch else []),
    ]


def concat_polymer_shim():
    """`//tensorboard/components:polymer3_lib_binary`."""
    parts = [
        NODE_MODULES / "web-animations-js/web-animations-next-lite.min.js",
        WEB_OUT / "polymer3_lib_binary_no_shim.js",
    ]
    write_if_changed(
        WEB_OUT / "polymer3_lib_binary.js",
        b"".join(p.read_bytes() + b"\n" for p in parts),
    )


# --- 5. Angular app ----------------------------------------------------------


def ng_args(dev, watch=False):
    return [
        str(NODE_BIN / "ng"),
        "build",
        *(["--configuration=development"] if dev else []),
        *(["--watch"] if watch else []),
    ]


def convert_ng_output(dev):
    """Rewraps the Angular ES module bundle as an IIFE classic script.

    index.js concatenates the Angular app with the Polymer bundle and
    require.js. As a classic script, the module's top-level declarations would
    become globals and could clash with require.js's `require` and `define`.
    """
    run([
        ESBUILD,
        str(NG_OUT / "main.js"),
        f"--outfile={WEB_OUT / 'tb_webapp_binary.js'}",
        "--bundle",
        "--format=iife",
        "--log-level=warning",
        *([] if dev else ["--minify"]),
    ])


# --- 6. Vulcanize and zip ----------------------------------------------------


class _Vulcanizer:
    """Inlines an HTML document's imports the way Vulcanize.java does.

    Vulcanize.java parses each document with jsoup's HTML5 parser, inlines it,
    and then flattens the nested documents into one `<head>` and one `<body>`.
    This port models the HTML5 insertion modes that TensorBoard's entry
    documents use, and fails on markup outside that model:

    - `<link rel="import">` is replaced by the imported document, once.
    - `<link rel="stylesheet">` becomes an inline `<style>`.
    - Every `<script>`'s code (inline, or read from `src`) is appended to one
      JavaScript output in document order and the script removed. The first
      script inlined (not `jscomp-ignore`) is preceded by
      `var CLOSURE_NO_DEPS = true;`. `<script src=js_path>` ends the body.
    - Text in the head is kept. Whitespace-only text directly in the body is
      dropped.
    - The first top-level `@license` comment of the root document is kept;
      other comments are dropped. (Vulcanize.java means to merge every
      distinct license into it, but its jsoup `attr("comment", ...)` call
      doesn't change a comment's text.)
    """

    def __init__(self, webfiles):
        self.webfiles = webfiles
        self.inlined = set()
        self.license = None  # the root document's first top-level @license
        self.head = []
        self.body = []
        self.scripts = []
        self.inlined_script = False

    def vulcanize(self, input_path, js_path):
        _Document(self, input_path, root=True).parse()
        html = "".join([
            "<!doctype html>",
            f"<!--{self.license}-->" if self.license is not None else "",
            "<html><head>", *self.head, "</head><body>", *self.body,
            f'<script src="{js_path.lstrip("/")}"></script>', "</body></html>",
        ])
        js = "".join(code + "\n" for code in self.scripts)
        return html, js

    def read(self, webpath, referrer):
        if webpath not in self.webfiles:
            raise ValueError(f"{referrer} references unknown webfile {webpath}")
        return Path(self.webfiles[webpath]).read_bytes().decode("utf-8")


# Elements that jsoup's "in head" insertion mode keeps in the head.
_HEAD_TAGS = frozenset({
    "base", "basefont", "bgsound", "command", "link", "meta", "noframes",
    "noscript", "script", "style", "template", "title",
})
_VOID_TAGS = frozenset({
    "area", "base", "basefont", "bgsound", "br", "col", "command", "embed",
    "hr", "img", "input", "keygen", "link", "meta", "param", "source",
    "track", "wbr",
})
# Vulcanize.java's shouldIgnoreUri.
_IGNORED_URI = re.compile(r"^(#|data:|javascript:|mailto:)|//|/$|\[\[.*\]\]|\{\{.*\}\}")
_SOURCE_MAP = re.compile(r'[^"]//# sourceMappingURL=.*[^"]')


class _Document(html.parser.HTMLParser):
    """Parses one document into its `_Vulcanizer`'s output."""

    def __init__(self, vulcanizer, webpath, root=False):
        super().__init__(convert_charrefs=False)
        self.v = vulcanizer
        self.webpath = webpath
        self.root = root
        self.mode = "before head"  # or "in head", "after head", "in body"
        self.depth = 0  # open elements below <head> or <body>
        self.script = None  # (attrs, chunks) of the open <script>

    def parse(self):
        self.feed(self.v.read(self.webpath, "input"))
        self.close()
        if self.script is not None:
            raise ValueError(f"unclosed <script> in {self.webpath}")

    def _fail(self, what):
        raise ValueError(f"{self.webpath}:{self.getpos()[0]}: unsupported {what}")

    def _resolve(self, uri):
        if uri.startswith("/"):
            return posixpath.normpath(uri)
        return posixpath.normpath(posixpath.join(posixpath.dirname(self.webpath), uri))

    def _out(self):
        return self.v.body if self.mode == "in body" else self.v.head

    def _enter(self, tag):
        """Applies the insertion mode transitions a start tag causes."""
        if self.depth:
            return
        if self.mode in ("before head", "in head", "after head"):
            if tag in _HEAD_TAGS and self.mode != "after head":
                self.mode = "in head"
            elif tag in _HEAD_TAGS:
                self._fail(f"<{tag}> after </head>")
            else:
                self.mode = "in body"

    def handle_starttag(self, tag, attrs):
        if tag == "html":
            return
        if tag == "head":
            if self.mode != "before head":
                self._fail("<head> here")
            self.mode = "in head"
            return
        if tag == "body":
            if self.mode == "in body" and self.depth:
                self._fail("nested <body>")
            self.mode = "in body"
            return
        self._enter(tag)
        a = dict(attrs)
        if tag == "script":
            self.script = (a, [])
            return
        if tag == "link" and a.get("rel") == "import" and a.get("type") not in ("css", "text/css"):
            if self.mode == "in body":
                self._fail("<link rel=import> in <body>")  # Vulcanize.java too
            href = self._resolve(a["href"])
            if href not in self.v.inlined:
                self.v.inlined.add(href)
                _Document(self.v, href).parse()
            return
        if tag == "link" and a.get("rel") in ("stylesheet", "import") and a.get("href") \
                and not _IGNORED_URI.search(a["href"]):
            css = self.v.read(self._resolve(a["href"]), self.webpath)
            rest = [(k, v) for k, v in attrs if k not in ("rel", "href")]
            self._out().append(f"{_start_tag('style', rest)}{css}</style>")
            return
        self._out().append(_start_tag(tag, attrs))
        if tag not in _VOID_TAGS:
            self.depth += 1

    def handle_startendtag(self, tag, attrs):
        # HTML ignores the self-closing flag of non-void elements.
        self.handle_starttag(tag, attrs)
        if tag == "script":
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if tag in ("html", "body"):
            return
        if tag == "head":
            if self.mode != "in head" or self.depth:
                self._fail("</head> here")
            self.mode = "after head"
            return
        if tag == "script":
            self._end_script()
            return
        if tag in _VOID_TAGS:
            return
        if not self.depth:
            self._fail(f"</{tag}> without a matching start tag")
        self.depth -= 1
        self._out().append(f"</{tag}>")

    def _end_script(self):
        if self.script is None:
            self._fail("</script> without <script>")
        attrs, chunks = self.script
        self.script = None
        src = attrs.get("src")
        if src and _IGNORED_URI.search(src):
            self._fail(f"remote script {src}")
        ignored = "jscomp-ignore" in attrs
        code = self.v.read(self._resolve(src), self.webpath) if src else "".join(chunks)
        if not ignored:
            if src:
                code = _SOURCE_MAP.sub("", code.replace("</script>", "</JAVA_SCRIIIIPT/>"))
            if not self.v.inlined_script:
                self.v.inlined_script = True
                self.v.scripts.append("var CLOSURE_NO_DEPS = true;")
        self.v.scripts.append(code)

    def handle_data(self, data):
        if self.script is not None:
            self.script[1].append(data)
            return
        if self.depth:
            self._out().append(data)
            return
        blank = not data.strip()
        if self.mode == "before head" or self.mode == "after head":
            if not blank:
                self._fail("text outside <head> and <body>")
        elif self.mode == "in head":
            if blank:
                self.v.head.append(data)
            else:
                self._fail("text in <head>")
        elif not blank:
            self.v.body.append(data)

    def handle_comment(self, data):
        # Only top-level comments of the root document survive flattening.
        if "@license" in data and self.v.license is None and self.root \
                and self.mode == "before head":
            self.v.license = data

    def handle_entityref(self, name):
        self.handle_data(f"&{name};")

    def handle_charref(self, name):
        self.handle_data(f"&#{name};")

    def handle_decl(self, decl):
        if not decl.lower().startswith("doctype"):
            self._fail(f"<!{decl}>")

    def handle_pi(self, data):
        self._fail("processing instruction")

    def unknown_decl(self, data):
        self._fail(f"<![{data}]>")


def _start_tag(tag, attrs):
    """Serializes a start tag the way jsoup does."""
    parts = [tag]
    for k, v in attrs:
        if v is None:
            parts.append(k)
        else:
            v = v.replace("&", "&amp;").replace("\xa0", "&nbsp;").replace('"', "&quot;")
            parts.append(f'{k}="{v}"')
    return "<%s>" % " ".join(parts)


class StaleError(Exception):
    """Built parts are out of sync, e.g. while watchers are rebuilding."""


def assemble():
    """Writes webfiles.zip and the projector binary from the built parts.

    Returns whether webfiles.zip changed.
    """
    favicon = base64.b64encode((ROOT / "tensorboard/logo/favicon.png").read_bytes()).decode()
    index_template = (ROOT / "tensorboard/webapp/index_polymer3.uninlined.html").read_text()
    write_if_changed(
        WEB_OUT / "index_polymer3.inlined.html",
        index_template.replace("%TENSORBOARD_FAVICON_URI%", f"data:image/png;base64,{favicon}"),
    )
    shared_webfiles = {
        "/tf-imports/roboto.html": ROOT / "tensorboard/components/tf_imports/roboto.html",
        "/font-roboto/roboto.html": THIRD_PARTY_WEB / "roboto.html",
    }

    # //tensorboard/webapp:index (+ :index_js)
    index_html, index_js = _Vulcanizer({
        **shared_webfiles,
        "/index_polymer3.inlined.html": WEB_OUT / "index_polymer3.inlined.html",
        "/styles.css": GENFILES / "tensorboard/webapp/styles.css",
        "/polymer3_lib_binary.js": WEB_OUT / "polymer3_lib_binary.js",
        "/tb-webapp/tb_webapp_binary.js": WEB_OUT / "tb_webapp_binary.js",
        "/tf-imports/require.js": NODE_MODULES / "requirejs/require.js",
    }).vulcanize("/index_polymer3.inlined.html", "/%INDEX.js%")
    for bundle in WORKER_RESOURCES:
        url = f"{bundle}.js?_file_hash={_sha256((WEB_OUT / f'{bundle}.js').read_bytes())[:8]}"
        if url not in index_js:
            raise StaleError(f"the Angular bundle doesn't load the current {url}")
    js_hash = _sha256(index_js.encode())[:8]
    index_html = index_html.replace("%INDEX.js%", f"index.js?_file_hash={js_hash}")

    # //tensorboard/plugins/projector/tf_projector_plugin:projector_binary
    projector_html, projector_js = _Vulcanizer({
        **shared_webfiles,
        "/tf-projector/tf-projector-plugin.html": PROJECTOR_DIR / "tf-projector-plugin.html",
        "/standalone_bundle.js": WEB_OUT / "standalone_bundle.js",
    }).vulcanize("/tf-projector/tf-projector-plugin.html", "/projector_binary.js")
    write_if_changed(PROJECTOR_DIR / "projector_binary.html", projector_html)
    write_if_changed(PROJECTOR_DIR / "projector_binary.js", projector_js)

    # //tensorboard/webapp:svg_bundle
    icons = DOWNLOADS / "material-icons"
    icon_svg = subprocess.run(
        [sys.executable, MAT_BUNDLE_ICON_SVG,
         *[str(icons / name) for name in sorted(third_party_manifest()["material-icons"])]],
        cwd=ROOT, check=True, stdout=subprocess.PIPE,
    ).stdout

    monaco = NODE_MODULES / "monaco-editor-core/dev/vs"
    monaco_languages = NODE_MODULES / "monaco-languages/release/dev"
    entries = {
        "index.html": index_html.encode(),
        "index.js": index_js.encode(),
        "icon_bundle.svg": icon_svg,
        "chart_worker.js": WEB_OUT / "chart_worker.js",
        "smoothing_worker.js": WEB_OUT / "smoothing_worker.js",
        "font-roboto/roboto.html": THIRD_PARTY_WEB / "roboto.html",
        **{
            f"font-roboto/{name}": DOWNLOADS / "font-roboto" / name
            for name in third_party_manifest()["font-roboto"]
        },
        **{
            f"tf-imports/vs/{path}": monaco / path
            for path in [
                "base/browser/ui/codicons/codicon/codicon.ttf",
                "base/worker/workerMain.js",
                "editor/editor.main.css",
                "editor/editor.main.js",
                "editor/editor.main.nls.js",
            ]
        },
        "tf-imports/vs/monaco.contribution.js": monaco_languages / "monaco.contribution.js",
        "tf-imports/vs/python/python.js": monaco_languages / "python/python.js",
    }
    archive = io.BytesIO()
    # Fixed timestamps keep the archive reproducible. TensorBoard re-gzips
    # every entry when it starts, so favor speed (like Bazel's Zipper).
    with zipfile.ZipFile(archive, "w") as zf:
        for name, content in entries.items():
            data = content if isinstance(content, bytes) else Path(content).read_bytes()
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            zf.writestr(info, data, zipfile.ZIP_DEFLATED, compresslevel=1)
    return write_if_changed(WEBFILES_ZIP, archive.getvalue())


# --- Drivers -------------------------------------------------------------------


def build(dev):
    t0 = time.monotonic()
    steps = [
        ("fetch third-party assets", fetch_third_party),
        ("generate sources", generate_sources),
        ("sass", lambda: run(sass_args())),
        ("esbuild bundles", lambda: run(esbuild_args(dev))),
        ("worker resources", write_worker_resources),
        ("ng build", lambda: run(ng_args(dev), stdin=subprocess.DEVNULL)),
        ("convert Angular bundle", lambda: convert_ng_output(dev)),
        ("Polymer shim", concat_polymer_shim),
        ("assemble", assemble),
    ]
    for name, step in steps:
        t = time.monotonic()
        step()
        log(f"{name}: {time.monotonic() - t:.1f}s")
    log(f"built {WEBFILES_ZIP.relative_to(ROOT)} in {time.monotonic() - t0:.1f}s")


class _Server:
    """A TensorBoard subprocess that is restarted after each rebuild."""

    def __init__(self, args):
        self._args = args
        self._proc = None

    def restart(self):
        self.stop()
        log("starting TensorBoard")
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "tensorboard.main", *self._args], cwd=ROOT,
            preexec_fn=_die_with_parent,
        )

    def stop(self):
        if self._proc and self._proc.poll() is None:
            self._proc.send_signal(signal.SIGINT)
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()


def _die_with_parent():
    """A `preexec_fn` that has Linux stop the child if this script is killed."""
    if sys.platform == "linux":
        import ctypes

        pr_set_pdeathsig = 1
        ctypes.CDLL(None).prctl(pr_set_pdeathsig, int(signal.SIGTERM))


def _start_watcher(cmd):
    """Starts a watcher that won't outlive this script.

    Its own process group lets cleanup stop everything it spawns (e.g. the
    native binary behind esbuild's Node shim).
    """
    return subprocess.Popen(
        cmd, cwd=ROOT, stdin=subprocess.PIPE, start_new_session=True,
        preexec_fn=_die_with_parent,
    )


def watch(tensorboard_args):
    """Rebuilds on change.

    Sass, esbuild and ng watch their own inputs. This loop reruns the genrules
    when their inputs change and reassembles when the tools' outputs change.
    (New .scss files and images need a restart: they're listed at startup.)
    """
    build(dev=True)
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *_: sys.exit(1))  # Runs the `finally` below.
    watchers = []
    server = _Server(tensorboard_args) if tensorboard_args else None
    try:
        for cmd in (sass_args(watch=True), esbuild_args(dev=True, watch=True),
                    ng_args(dev=True, watch=True)):
            watchers.append(_start_watcher(cmd))
        if server:
            server.restart()
        _watch_loop(watchers, server)
    except KeyboardInterrupt:
        pass
    finally:
        # Ctrl-C and `mise` may deliver more signals; finish cleaning up.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if server:
            server.stop()
        for w in watchers:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(w.pid, signal.SIGTERM)
        for w in watchers:
            w.wait()
        log(f"note: {WEBFILES_ZIP.relative_to(ROOT)} is a development build; "
            "run `mise run build:web` before packaging")


def _watch_loop(watchers, server):
    tracked = [
        NG_OUT / "main.js",
        GENFILES / "tensorboard/webapp/styles.css",
        ROOT / "tensorboard/webapp/index_polymer3.uninlined.html",
        ROOT / "tensorboard/components/tf_imports/roboto.html",
        ROOT / "tensorboard/logo/favicon.png",
        PROJECTOR_DIR / "tf-projector-plugin.html",
        THIRD_PARTY_WEB / "roboto.html",
        *[WEB_OUT / f"{name}.js" for name in ESBUILD_ENTRY_POINTS],
        ROOT / DIGEST_SUFFIXER,
        *[ROOT / f"{module}.template.ts" for module, _ in WORKER_RESOURCES.values()],
    ]
    genrule_inputs = sorted({ROOT / i for _, inputs, _ in genrules() for i in inputs})

    def snapshot(paths):
        return {p: p.stat().st_mtime_ns if p.exists() else None for p in paths}

    last, last_inputs = snapshot(tracked), snapshot(genrule_inputs)
    while True:
        dead = [w for w in watchers if w.poll() is not None]
        if dead:
            raise SystemExit(f"watcher exited ({dead[0].returncode}): {' '.join(dead[0].args[:3])}")
        time.sleep(0.3)
        if snapshot(genrule_inputs) != last_inputs:
            last_inputs = snapshot(genrule_inputs)
            try:
                generate_sources()  # The watchers pick up the outputs.
            except subprocess.CalledProcessError as e:
                log(f"generating sources failed: {e}")
        current = snapshot(tracked)
        if current == last:
            continue
        time.sleep(0.3)  # Let the writers finish.
        current = snapshot(tracked)
        changed = {p for p in tracked if current[p] != last[p]}
        last = current
        try:
            if NG_OUT / "main.js" in changed:
                convert_ng_output(dev=True)
            if write_worker_resources():
                continue  # ng rebuilds with the new worker URLs; assemble then.
            concat_polymer_shim()
            if not assemble():
                continue  # Nothing changed, e.g. the watchers' first builds.
        except StaleError:
            continue  # An ng rebuild is still on its way.
        except (subprocess.CalledProcessError, OSError, ValueError) as e:
            log(f"assemble failed: {e}")
            continue
        log("rebuilt webfiles.zip")
        if server:
            server.restart()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="one-off build")
    b.add_argument("--dev", action="store_true", help="unminified, faster build")
    w = sub.add_parser("watch", help="rebuild on change, optionally running TensorBoard")
    w.add_argument("tensorboard_args", nargs=argparse.REMAINDER,
                   help="arguments for tensorboard.main, after `--`")
    args = parser.parse_args()
    WEB_OUT.mkdir(parents=True, exist_ok=True)
    if args.command == "build":
        build(args.dev)
    else:
        tb_args = args.tensorboard_args
        watch(tb_args[1:] if tb_args[:1] == ["--"] else tb_args)


if __name__ == "__main__":
    main()
