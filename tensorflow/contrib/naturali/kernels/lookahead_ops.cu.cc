#define EIGEN_USE_THREADS
#define EIGEN_USE_GPU

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
#include <cuda_runtime.h>

using namespace tensorflow;

template<typename T>
__global__ void lakernel(int dim_tau, const T* input, const T* filter, T* output) {
  int TS = gridDim.x;
  int B = gridDim.y;
  int F = blockDim.x;
  int t = blockIdx.x;
  int b = blockIdx.y;
  int f = threadIdx.x;
  output[(t * B + b) * F + f] = 0;
  for(int tau = 0; tau < dim_tau && t + tau < TS; tau++) {
    output[(t * B + b) * F + f] += input[((t + tau) * B + b) * F + f] * filter[tau * F + f];
  }
}
template<typename T>
class LookaheadGpuOp : public OpKernel {
 public:
  explicit LookaheadGpuOp(OpKernelConstruction* context) : OpKernel(context) {
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

    // Create output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->template tensor<T, 3>();
    int b_size = input_tensor.dim_size(1);
    int TS = input_tensor.dim_size(0);
    int F = input_tensor.dim_size(2);
    int W = filter_tensor.dim_size(0);
    dim3 grid(TS, b_size);
    auto device = context->template eigen_device<Eigen::GpuDevice>();
    auto stream = device.stream();
    lakernel<T><<<grid, F, 0, stream>>>(W, &input(0, 0, 0), &filter(0, 0), &output(0, 0, 0));
  }
};

REGISTER_KERNEL_BUILDER(Name("Lookahead").Device(DEVICE_GPU), LookaheadGpuOp<float>);

template<typename T>
__global__ void cckernel(int W, const T* input, const T* filter, T* output) {
  int TS = gridDim.x;
  int B = gridDim.y;
  int F = blockDim.x;
  int H = W/2;
  int t = blockIdx.x;
  int b = blockIdx.y;
  int f = threadIdx.x;
  output[(t * B + b) * F + f] = 0;
  for(int tau = -H; tau <= H && t + tau >= 0 && t + tau < TS; tau++) {
    output[(t * B + b) * F + f] += input[((t + tau) * B + b) * F + f] * filter[tau * F + f];
  }
}

template<typename T>
class ColConvGpuOp : public OpKernel {
 public:
  explicit ColConvGpuOp(OpKernelConstruction* context) : OpKernel(context) {
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
        errors::InvalidArgument("filter width need to be"));

    // Create output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->template tensor<T, 3>();
    int b_size = input_tensor.dim_size(1);
    int TS = input_tensor.dim_size(0);
    int F = input_tensor.dim_size(2);
    int W = filter_tensor.dim_size(0);
    dim3 grid(TS, b_size);
    auto device = context->template eigen_device<Eigen::GpuDevice>();
    auto stream = device.stream();
    cckernel<T><<<grid, F, 0, stream>>>(W, &input(0, 0, 0), &filter(0, 0), &output(0, 0, 0));
  }
};

REGISTER_KERNEL_BUILDER(Name("ColConv").Device(DEVICE_GPU), ColConvGpuOp<float>);