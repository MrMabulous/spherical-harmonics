workspace(name = "spherical_harmonics")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "eigen3",
    strip_prefix = "eigen-3.3.7",
    urls = ["https://github.com/MrMabulous/eigen/archive/3.3.7.zip"],
	build_file = "@//third_party:eigen3.BUILD",
)

http_archive(
  name = "gtest",
  url = "https://github.com/google/googletest/archive/release-1.8.0.zip",
  strip_prefix = "googletest-release-1.8.0",
  sha256 = "f3ed3b58511efd272eb074a3a6d6fb79d7c2e6a0e374323d1e6bcbcc1ef141bf",
  build_file = "@//third_party:gtest.BUILD",
)

http_archive(
    name = "mabu_trace",
    strip_prefix = "MabuTrace-windows/MabuTrace",
    urls = ["https://github.com/MrMabulous/MabuTrace/archive/windows.zip"],
)