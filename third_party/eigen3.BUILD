licenses(["restricted"])  # MPL2, portions GPL v3, LGPL v3, BSD-like

exports_files(["LICENSE"])

cc_library(
    name = "eigen3",
    visibility = ["//visibility:public"],
    hdrs = glob(
        include = ["Eigen/**"],
        exclude = ["Eigen/**/CMakeLists.txt"],
    ),
    defines = ["EIGEN_MPL2_ONLY",
		       "EIGEN_DONT_PARALLELIZE",
		       "EIGEN_UNALIGNED_VECTORIZE=0",
		       "EIGEN_MAX_ALIGN_BYTES=32",
		       "EIGEN_MAX_STATIC_ALIGN_BYTES=32",
		       "EIGEN_NO_AUTOMATIC_RESIZING",
    ],
)