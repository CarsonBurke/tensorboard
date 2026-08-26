# Description:
#   TensorBoard core plugin.
load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python:py_test.bzl", "py_test")

package(default_visibility = ["//tensorboard:internal"])

licenses(["notice"])

py_library(
    name = "core_plugin",
    srcs = ["core_plugin.py"],
    deps = [
        "//tensorboard:plugin_util",
        "//tensorboard:version",
        "//tensorboard/backend:http_util",
        "//tensorboard/data:ingester",
        "//tensorboard/plugins:base_plugin",
        "//tensorboard/util:grpc_util",
        "//tensorboard/util:tb_logging",
        "@tensorboard_pip_deps//werkzeug",
    ],
)

py_test(
    name = "core_plugin_test",
    size = "small",
    srcs = ["core_plugin_test.py"],
    deps = [
        ":core_plugin",
        "//tensorboard:expect_tensorflow_installed",
        "//tensorboard/backend:application",
        "//tensorboard/backend/event_processing:data_provider",
        "//tensorboard/backend/event_processing:event_multiplexer",
        "//tensorboard/data:ingester",
        "//tensorboard/data:provider",
        "//tensorboard/plugins:base_plugin",
        "//tensorboard/util:test_util",
        "@tensorboard_pip_deps//werkzeug",
    ],
)
