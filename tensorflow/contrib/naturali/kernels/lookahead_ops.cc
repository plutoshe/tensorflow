// Copyright by Naturali. 2016
// Author Xibai, Pluto, Sean
// All rights reserved.

#include <memory>
#include <string>
#include <utility>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
using namespace tensorflow;

template<typename T>
class LookaheadCpuOp : public OpKernel {
 public:
  explicit LookaheadCpuOp(OpKernelConstruction* context) : OpKernel(context) {
    const DataType dt = DataTypeToEnum<T>::v();
    OP_REQUIRES_OK(context, context->MatchSignature({dt, dt}, {dt}));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.tensor<T, 3>();

    const Tensor& filter_tensor = context->input(1);
    auto filter = filter_tensor.matrix<T>();

    // Check that dimension is equal
    OP_REQUIRES(
        context, input_tensor.dim_size(2) == filter_tensor.dim_size(1),
        errors::InvalidArgument("f is not equal in filter and input"));

    auto TS = input_tensor.dim_size(0);
    auto B = input_tensor.dim_size(1);
    auto F = input_tensor.dim_size(2);
    auto W = filter_tensor.dim_size(0);

    // Create output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->template tensor<T, 3>();

    for (int t = 0; t < TS; t++) {
      for (int b = 0; b < B; b++) {
        for (int f = 0; f < F; f++) {
          output(t, b, f) = 0;
        }
        for(int tau = 0; tau < W && t + tau < TS; tau++) {
          for (int f = 0; f < F; f++) {
            output(t, b, f) += input(t + tau, b, f) * filter(tau, f);
          }
        }
      }
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("Lookahead").Device(DEVICE_CPU), LookaheadCpuOp<float>);

template<typename T>
class ColConvCpuOp : public OpKernel {
 public:
  explicit ColConvCpuOp(OpKernelConstruction* context) : OpKernel(context) {
    const DataType dt = DataTypeToEnum<T>::v();
    OP_REQUIRES_OK(context, context->MatchSignature({dt, dt}, {dt}));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.tensor<T, 3>();

    const Tensor& filter_tensor = context->input(1);
    auto filter = filter_tensor.matrix<T>();

    // Check that dimension is equal
    OP_REQUIRES(
        context, input_tensor.dim_size(2) == filter_tensor.dim_size(1),
        errors::InvalidArgument("f is not equal in filter and input"));

    OP_REQUIRES(
        context, filter_tensor.dim_size(0)%2 == 1,
        errors::InvalidArgument("filter window need to be odd nubmer"));

    auto TS = input_tensor.dim_size(0);
    auto B = input_tensor.dim_size(1);
    auto F = input_tensor.dim_size(2);
    auto H = filter_tensor.dim_size(0)/2;

    // Create output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->template tensor<T, 3>();

    for (int t = 0; t < TS; t++) {
      for (int b = 0; b < B; b++) {
        for (int f = 0; f < F; f++) {
          output(t, b, f) = 0;
        }
        for(int tau = -H; tau <= H && t + tau >= 0 && t + tau < TS; tau++) {
          for (int f = 0; f < F; f++) {
            output(t, b, f) += input(t + tau, b, f) * filter(tau, f);
          }
        }
      }
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("Colconv").Device(DEVICE_CPU), ColConvCpuOp<float>);