
package(default_visibility = ["//visibility:public"])

load("//tensorflow:tensorflow.bzl", "tf_copts")

cc_binary(
    name = "libtensorflow_rnn.so",
    srcs = glob([
        "jni/**/*.cc",
        "jni/**/*.h",
    ]) + [":libpthread.so"],
    copts = tf_copts(),
    linkopts = [
        "-landroid",
        "-llog",
        "-lm",
        "-z defs",
        "-s",
        "-std=c++11",
        "-Wl,--icf=all",  # Identical Code Folding
        "-Wl,--exclude-libs,ALL",  # Exclude syms in all libs from auto export
    ],
    linkshared = 1,
    linkstatic = 1,
    tags = [
        "manual",
        "notap",
    ],
    deps = ["//tensorflow/core:android_tensorflow_lib",
            "//tensorflow/contrib/quantization/kernels:quantized_ops",
            ":main_lib"],
)

cc_library(
    name = "main_lib",
    srcs = ["main/str_util.cc",
            "main/dict.cc"],
    hdrs = ["main/str_util.h",
            "main/dict.h"],
)

# This library only exists as a workaround to satisfy dependencies
# that declare -lpthread in their linkopts. Although Android supports
# pthreads, it does not provide it as a separate library.
cc_binary(
    name = "libpthread.so",
    srcs = [],
    linkopts = ["-shared"],
    tags = [
        "manual",
        "notap",
    ],
)

cc_library(
    name = "tensorflow_rnn_native_libs",
    srcs = [
        ":libpthread.so",
        ":libtensorflow_rnn.so",
    ],
    tags = [
        "manual",
        "notap",
    ],
)

android_binary(
    name = "androidrnn",
    srcs = glob([
        "src/**/*.java",
    ]),
    assets = glob(["assets/**"]),
    assets_dir = "assets",
    custom_package = "com.singulariti.androidrnn",
    inline_constants = 1,
    manifest = "AndroidManifest.xml",
    resource_files = glob(["res/**"]),
    tags = [
        "manual",
        "notap",
    ],
    deps=["//external:android/appcompat_v7",
          "//external:android/appcompat_v4",
          ":tensorflow_rnn_native_libs"],
)

filegroup(
    name = "java_files",
    srcs = glob(["src/**/*.java"]),
)

filegroup(
    name = "jni_files",
    srcs = glob([
        "jni/**/*.cc",
        "jni/**/*.h",
    ]),
)

filegroup(
    name = "resource_files",
    srcs = glob(["res/**"]),
)

exports_files(["AndroidManifest.xml"])
