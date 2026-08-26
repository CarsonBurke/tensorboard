load("@rules_web_testing_python//web:py.bzl", "py_web_test_suite")

package(default_visibility = ["//tensorboard:internal"])

licenses(["notice"])

py_web_test_suite(
    name = "core_test",
    srcs = ["core_test.py"],
    browsers = ["//tensorboard/functionaltests/browsers:chromium"],
    data = [
        "//tensorboard",
    ],
    # TODO(@jart): Make this fast and not flaky.
    flaky = True,
    tags = [
        "manual",
        "webtest",
    ],
    deps = [
        "//tensorboard/plugins/audio:audio_demo",
        "//tensorboard/plugins/scalar:scalars_demo",
        "@rules_web_testing_python//testing/web",
    ],
)
