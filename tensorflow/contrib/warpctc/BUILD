# Description:
#   Contains ops to train linear models on top of TensorFlow.
#   APIs here are meant to evolve over time.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//visibility:public"])

load("//tensorflow:tensorflow.bzl", "cuda_py_tests")
load(
    "//tensorflow:tensorflow.bzl",
    "tf_custom_op_library",
    "tf_cc_test",
    "tf_cc_tests",
    "tf_kernel_libraries",
    "tf_kernel_library",
    "cc_header_only_library",
)
load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_proto_library",
    "tf_kernel_tests_linkstatic",
)

py_library(
    name = "warpctc_py",
    srcs = ["__init__.py"] + glob(["python/ops/*.py"]),
    data = [
        ":python/ops/_warpctc_ops.so",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
)

config_setting(
    name = "is_ci",
    values = {"define": "CI=true"}
)

tf_custom_op_library(
    name = "python/ops/_warpctc_ops.so",
    srcs = [
        "kernels/warpctc_ops.cc",
        "kernels/warpctc_ops.h",
        "ops/warpctc_ops.cc",
	"kernels/warp-ctc/tests/test.h",
    ],
    gpu_srcs = [
        "kernels/warpctc_ops.cu.cc",
        "kernels/warpctc_ops.h",
#	"kernels/warp-ctc/tests/test_gpu.h",
	"kernels/warp-ctc/tests/test.h",
    ],
    deps =select({
        ":is_ci": ["//tensorflow/contrib/warpctc/kernels/warp-ctc/src:src_cpu"],
        "//conditions:default": ["//tensorflow/contrib/warpctc/kernels/warp-ctc/src:src_gpu"],}) +
    ["//tensorflow/contrib/warpctc/kernels/warp-ctc/include:libcpu",
#	"//tensorflow/core/kernels:bounds_check",
#	"//tensorflow/core/kernels:ops_util",
#        "//tensorflow/core:framework",
#        "//tensorflow/core:lib",
#        "//tensorflow/core:user_ops_op_lib",
#        "//tensorflow/contrib/warpctc/kernels/warp-ctc/tests:warpctc",
        "//tensorflow/core/kernels:eigen_helpers",
    ],
)

cuda_py_tests(
    name = "warpctc_ops_test",
    size = "small",
    srcs = ["python/kernel_tests/warpctc_ops_test.py"],
    additional_deps = [
        ":warpctc_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)
