load("//tensorboard/defs:defs.bzl", "tf_ng_module", "tf_ts_library")

package(default_visibility = ["//tensorboard:internal"])

licenses(["notice"])

tf_ng_module(
    name = "local_storage",
    srcs = [
        "run_local_storage_data_source.ts",
        "runs_local_storage_effects.ts",
    ],
    deps = [
        "//tensorboard/webapp:app_state",
        "//tensorboard/webapp:selectors",
        "//tensorboard/webapp/core/actions",
        "//tensorboard/webapp/runs:types",
        "//tensorboard/webapp/runs/actions",
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
        "run_local_storage_data_source_test.ts",
        "runs_local_storage_effects_test.ts",
    ],
    deps = [
        ":local_storage",
        "//tensorboard/webapp:app_state",
        "//tensorboard/webapp:selectors",
        "//tensorboard/webapp/angular:expect_angular_core_testing",
        "//tensorboard/webapp/angular:expect_ngrx_store_testing",
        "//tensorboard/webapp/core/actions",
        "//tensorboard/webapp/runs/actions",
        "//tensorboard/webapp/runs:types",
        "//tensorboard/webapp/testing:utils",
        "@npm//@ngrx/effects",
        "@npm//@ngrx/store",
        "@npm//@types/jasmine",
        "@npm//rxjs",
    ],
)
