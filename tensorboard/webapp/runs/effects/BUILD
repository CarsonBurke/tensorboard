load("//tensorboard/defs:defs.bzl", "tf_ng_module", "tf_ts_library")

package(default_visibility = ["//tensorboard:internal"])

licenses(["notice"])

tf_ng_module(
    name = "effects",
    srcs = [
        "index.ts",
        "runs_effects.ts",
    ],
    deps = [
        "//tensorboard/webapp:app_state",
        "//tensorboard/webapp:selectors",
        "//tensorboard/webapp/app_routing",
        "//tensorboard/webapp/app_routing:types",
        "//tensorboard/webapp/app_routing/actions",
        "//tensorboard/webapp/core/actions",
        "//tensorboard/webapp/experiments:types",
        "//tensorboard/webapp/hparams:_types",
        "//tensorboard/webapp/hparams/_redux:hparams_actions",
        "//tensorboard/webapp/hparams/_redux:hparams_selectors",
        "//tensorboard/webapp/hparams/_redux:types",
        "//tensorboard/webapp/metrics/actions",
        "//tensorboard/webapp/metrics/views/main_view:common_selectors",
        "//tensorboard/webapp/runs:types",
        "//tensorboard/webapp/runs/actions",
        "//tensorboard/webapp/runs/data_source",
        "//tensorboard/webapp/runs/store:types",
        "//tensorboard/webapp/runs/views/runs_table:sorting_utils",
        "//tensorboard/webapp/types",
        "//tensorboard/webapp/widgets/data_table:types",
        "@npm//@angular/core",
        "@npm//@ngrx/effects",
        "@npm//@ngrx/store",
        "@npm//rxjs",
    ],
)

tf_ts_library(
    name = "effects_test",
    testonly = True,
    srcs = [
        "runs_effects_test.ts",
    ],
    deps = [
        ":effects",
        "//tensorboard/webapp:app_state",
        "//tensorboard/webapp:selectors",
        "//tensorboard/webapp/angular:expect_angular_core_testing",
        "//tensorboard/webapp/angular:expect_ngrx_store_testing",
        "//tensorboard/webapp/app_routing:testing",
        "//tensorboard/webapp/app_routing:types",
        "//tensorboard/webapp/app_routing/actions",
        "//tensorboard/webapp/core/actions",
        "//tensorboard/webapp/hparams/_redux:hparams_actions",
        "//tensorboard/webapp/hparams/_redux:hparams_selectors",
        "//tensorboard/webapp/hparams/_redux:testing",
        "//tensorboard/webapp/runs/actions",
        "//tensorboard/webapp/runs/data_source",
        "//tensorboard/webapp/runs/data_source:testing",
        "//tensorboard/webapp/runs/views/runs_table:sorting_utils",
        "//tensorboard/webapp/testing:utils",
        "//tensorboard/webapp/types",
        "//tensorboard/webapp/widgets/data_table:types",
        "@npm//@ngrx/effects",
        "@npm//@ngrx/store",
        "@npm//@types/jasmine",
        "@npm//rxjs",
    ],
)
