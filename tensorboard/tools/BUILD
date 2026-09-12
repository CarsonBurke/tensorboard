load("@rules_python//python:py_binary.bzl", "py_binary")
load("@rules_python//python:py_test.bzl", "py_test")

package(default_visibility = ["//tensorboard:internal"])

licenses(["notice"])

py_binary(
    name = "import_google_fonts",
    srcs = ["import_google_fonts.py"],
    deps = ["//tensorboard:expect_tensorflow_installed"],
)

py_binary(
    name = "mat_bundle_icon_svg",
    srcs = ["mat_bundle_icon_svg.py"],
)

py_test(
    name = "mat_bundle_icon_svg_test",
    srcs = ["mat_bundle_icon_svg_test.py"],
    deps = [
        ":mat_bundle_icon_svg",
        "//tensorboard:test",
    ],
)

py_binary(
    name = "inline_file_content",
    srcs = ["inline_file_content.py"],
)

py_binary(
    name = "benchmark_metrics",
    srcs = ["benchmark_metrics.py"],
    deps = [
        "//tensorboard:context",
        "//tensorboard/backend:http_util",
        "//tensorboard/data:grpc_provider",
        "//tensorboard/data/proto:protos_all_py_pb2",
        "//tensorboard/plugins:base_plugin",
        "//tensorboard/plugins/metrics:metrics_plugin",
        "@tensorboard_pip_deps//werkzeug",
    ],
)

py_binary(
    name = "benchmark_tags",
    srcs = ["benchmark_tags.py"],
    deps = [
        "//tensorboard:context",
        "//tensorboard:expect_grpc_installed",
        "//tensorboard/data:grpc_provider",
        "//tensorboard/data/proto:protos_all_py_pb2",
        "//tensorboard/plugins:base_plugin",
        "//tensorboard/plugins/metrics:metrics_plugin",
        "//tensorboard/plugins/scalar:metadata",
    ],
)

py_binary(
    name = "benchmark_catalog",
    srcs = ["benchmark_catalog.py"],
    deps = [
        "//tensorboard/compat/proto:protos_all_py_pb2",
        "//tensorboard/summary/writer",
    ],
)
