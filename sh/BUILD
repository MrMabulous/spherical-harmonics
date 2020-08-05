config_setting(
    name = "windows",
    constraint_values = [
        "@bazel_tools//platforms:windows",
    ],
)

cc_library(
    name = "image",
    hdrs = ["image.h"],
    deps = [
        "@eigen3//:eigen3",
    ],
    visibility = ["//visibility:public"],
    alwayslink = 1
)

cc_library(
    name = "spherical_harmonics",
    srcs = ["spherical_harmonics.cc"],
    hdrs = [
        "spherical_harmonics.h",
        "efficient_sh_evaluation.h"
    ],
    deps = [
        ":image",
        "@eigen3//:eigen3",
    ],
    #copts = [
    #    "/arch:AVX2",
    #    "/D__SSE__",
    #    "/D__SSE2__",
    #    "/D__SSE3__",
    #    "/D__SSSE3__",
    #    "/D__SSE4_1__",
    #    "/D__SSE4_2__"
    #],
    defines = select({
        ":windows": ["_USE_MATH_DEFINES",],
    }),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "default_image",
    srcs = ["default_image.cc"],
    hdrs = ["default_image.h"],
    deps = [
        ":image",
    ]
)

cc_test(
    name = "spherical_harmonics_test",
    size = "small",
    srcs = [
        "spherical_harmonics_test.cc",
    ],
    deps = [
        ":default_image",
        ":spherical_harmonics",
        "@gtest//:main",
    ],
    #copts = [
    #    "/arch:AVX2",
    #    "/D__SSE__",
    #    "/D__SSE2__",
    #    "/D__SSE3__",
    #    "/D__SSSE3__",
    #    "/D__SSE4_1__",
    #    "/D__SSE4_2__"
    #],
    linkopts = ["-lm"],
)

cc_test(
    name = "spherical_harmonics_benchmark",
    size = "small",
    srcs = [
        "spherical_harmonics_benchmark.cc",
    ],
    deps = [
        ":spherical_harmonics",
        "@gtest//:main",
    ],
    #copts = [
    #    "/arch:AVX2",
    #    "/D__SSE__",
    #    "/D__SSE2__",
    #    "/D__SSE3__",
    #    "/D__SSSE3__",
    #    "/D__SSE4_1__",
    #    "/D__SSE4_2__"
    #],
    linkopts = ["-lm"],
)
