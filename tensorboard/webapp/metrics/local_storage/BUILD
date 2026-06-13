load("//tensorboard/defs:defs.bzl", "tf_ng_module", "tf_ts_library")

package(default_visibility = ["//tensorboard:internal"])

licenses(["notice"])

tf_ng_module(
    name = "local_storage",
    srcs = [
        "metrics_local_storage_data_source.ts",
        "metrics_local_storage_effects.ts",
    ],
    deps = [
        "//tensorboard/webapp:app_state",
        "//tensorboard/webapp:selectors",
        "//tensorboard/webapp/core/actions",
        "//tensorboard/webapp/metrics:types",
        "//tensorboard/webapp/metrics:utils",
        "//tensorboard/webapp/metrics/actions",
        "//tensorboard/webapp/util:types",
        "@npm//@angular/core",
        "@npm//@ngrx/effects",
        "@npm//@ngrx/store",
        "@npm//rxjs",
    ],
)

tf_ts_library(
    name = "local_storage_test",
    testonly = True,
    srcs = [
        "metrics_local_storage_data_source_test.ts",
        "metrics_local_storage_effects_test.ts",
    ],
    deps = [
        ":local_storage",
        "//tensorboard/webapp:app_state",
        "//tensorboard/webapp:selectors",
        "//tensorboard/webapp/angular:expect_angular_core_testing",
        "//tensorboard/webapp/angular:expect_ngrx_store_testing",
        "//tensorboard/webapp/core/actions",
        "//tensorboard/webapp/metrics:test_lib",
        "//tensorboard/webapp/metrics/actions",
        "//tensorboard/webapp/metrics/data_source",
        "//tensorboard/webapp/testing:utils",
        "@npm//@ngrx/effects",
        "@npm//@ngrx/store",
        "@npm//@types/jasmine",
        "@npm//rxjs",
    ],
)
