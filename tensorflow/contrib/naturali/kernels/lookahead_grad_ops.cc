#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

using namespace tensorflow;

template<typename T>
class LookaheadGradCpuOp : public OpKernel {
 public:
  explicit LookaheadGradCpuOp(OpKernelConstruction* context) : OpKernel(context) {
    const DataType dt = DataTypeToEnum<T>::v();
    OP_REQUIRES_OK(context, context->MatchSignature({dt, dt, dt}, {dt, dt}));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.tensor<T, 3>();

    const Tensor& filter_tensor = context->input(1);
    auto filter = filter_tensor.matrix<T>();

    const Tensor& grad_tensor = context->input(2);
    auto grad = grad_tensor.tensor<T, 3>();

     // Check that dimension is equal
    OP_REQUIRES(
        context, input_tensor.dim_size(2) == filter_tensor.dim_size(1),
        errors::InvalidArgument("f is not equal in filter and input"));
    OP_REQUIRES(
        context, (input_tensor.dim_size(0) == grad_tensor.dim_size(0)) &&
                 (input_tensor.dim_size(1) == grad_tensor.dim_size(1)) &&
                 (input_tensor.dim_size(2) == grad_tensor.dim_size(2)),
        errors::InvalidArgument("input's dimensions and grad's dimensions are not equal"));


    // Create input grad output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->template tensor<T, 3>();
    int FW = filter_tensor.dim_size(0);
    for (int t = 0; t < input_tensor.dim_size(0); t++) {
      for (int b = 0; b < input_tensor.dim_size(1); b++) {
        for (int f = 0; f < input_tensor.dim_size(2); f++) {
          output(t, b, f) = 0;
          for (int tau = -FW + 1; tau <= 0; tau++) {
            int gidx = tau + t;
            int fidx = -tau;
            if (gidx >= 0) {
              output(t, b, f) += grad(gidx, b, f) * filter(fidx, f);
            }
          }
        }
      }
    }
    // Create filter grad output tensor
    OP_REQUIRES_OK(context, context->allocate_output(1, filter_tensor.shape(),
                                                     &output_tensor));
    auto output2 = output_tensor->template matrix<T>();
    int TS = input_tensor.dim_size(0);
    for (int tau = 0; tau < filter_tensor.dim_size(0); tau++) {
      for (int f = 0; f < filter_tensor.dim_size(1); f++) {
        output2(tau, f) = 0;
      }
    }
    for (int b = 0; b < input_tensor.dim_size(1); b++) {
      for (int f = 0; f < filter_tensor.dim_size(1); f++) {
        for (int tau = 0; tau < filter_tensor.dim_size(0); tau++) {
          for (int t = 0; t < TS - tau; t++) {
            output2(tau, f) += grad(t, b, f) * input(t + tau, b, f);
          }
        }
      }
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("Lookaheadgrad").Device(DEVICE_CPU), LookaheadGradCpuOp<float>);

template<typename T>
class ColConvGradCpuOp : public OpKernel {
 public:
  explicit ColConvGradCpuOp(OpKernelConstruction* context) : OpKernel(context) {
    const DataType dt = DataTypeToEnum<T>::v();
    OP_REQUIRES_OK(context, context->MatchSignature({dt, dt, dt}, {dt, dt}));
  }
  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.tensor<T, 3>();

    const Tensor& filter_tensor = context->input(1);
    auto filter = filter_tensor.matrix<T>();

    const Tensor& grad_tensor = context->input(2);
    auto grad = grad_tensor.tensor<T, 3>();

     // Check that dimension is equal
    OP_REQUIRES(
        context, input_tensor.dim_size(2) == filter_tensor.dim_size(1),
        errors::InvalidArgument("f is not equal in filter and input"));
    OP_REQUIRES(
        context, (input_tensor.dim_size(0) == grad_tensor.dim_size(0)) &&
                 (input_tensor.dim_size(1) == grad_tensor.dim_size(1)) &&
                 (input_tensor.dim_size(2) == grad_tensor.dim_size(2)),
        errors::InvalidArgument("input's dimensions and grad's dimensions are not equal"));


    // Create input grad output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->template tensor<T, 3>();
    int FW = filter_tensor.dim_size(0);
    int FH = filter_tensor.dim_size(0)/2;
    for (int t = 0; t < input_tensor.dim_size(0); t++) {
      for (int b = 0; b < input_tensor.dim_size(1); b++) {
        for (int f = 0; f < input_tensor.dim_size(2); f++) {
          output(t, b, f) = 0;
          for (int tau = -FH; tau <= FH; tau++) {
            int gidx = tau + t;
            int fidx = FH - tau;
            if (gidx >= 0 && gidx < input_tensor.dim_size(0)) {
              output(t, b, f) += grad(gidx, b, f) * filter(fidx, f);
            }
          }
        }
      }
    }
    // Create filter grad output tensor
    OP_REQUIRES_OK(context, context->allocate_output(1, filter_tensor.shape(),
                                                     &output_tensor));
    auto output2 = output_tensor->template matrix<T>();
    int TS = input_tensor.dim_size(0);
    for (int tau = 0; tau < filter_tensor.dim_size(0); tau++) {
      for (int f = 0; f < filter_tensor.dim_size(1); f++) {
        output2(tau, f) = 0;
      }
    }
    for (int b = 0; b < input_tensor.dim_size(1); b++) {
      for (int f = 0; f < filter_tensor.dim_size(1); f++) {
        for (int tau = -FH; tau <= FH; tau++) {
          for (int t = 0; t < TS - tau && t + tau >= 0; t++) {
            output2(tau + FH, f) += grad(t, b, f) * input(t + tau, b, f);
          }
        }
      }
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("Colconvgrad").Device(DEVICE_CPU), ColConvGradCpuOp<float>);