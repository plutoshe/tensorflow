#define EIGEN_USE_THREADS
#define EIGEN_USE_GPU



#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include <cuda_runtime.h>

using namespace tensorflow;

template<typename T>
__global__ void lakernel_input(int FW, const T* filter, const T* grad, T* output) {
  int TS = gridDim.x;
  int B = gridDim.y;
  int F = blockDim.x;
  int t = blockIdx.x;
  int b = blockIdx.y;
  int f = threadIdx.x;
  output[(t * B + b) * F + f] = 0;
  for(int tau = -FW + 1; tau <= 0; tau++) {
    int gidx = tau + t;
    int fidx = -tau;
    if (gidx >= 0) {
      output[(t * B + b) * F + f] += grad[(gidx * B + b) * F + f] * filter[fidx * F + f];
    }
  }
}

template<typename T>
__global__ void lakernel_filter(int B, int TS, const T* input, const T* grad, T* output) {
  int F = blockDim.x;
  int FW = gridDim.x;
  int f = threadIdx.x;
  int tau = blockIdx.x;
  output[tau * F + f] = 0;
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < TS - tau; t++) {
      output[tau * F + f] += grad[(t * B + b) * F + f] * input[((t + tau) * B + b) * F + f];
    }
  }
}

template<typename T>
class LookaheadGradGpuOp : public OpKernel {
 public:
  explicit LookaheadGradGpuOp(OpKernelConstruction* context) : OpKernel(context) {
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

    int TS = input_tensor.dim_size(0);
    int B = input_tensor.dim_size(1);
    int F = input_tensor.dim_size(2);
    int FW = filter_tensor.dim_size(0);

    dim3 grid(TS, B);
    auto device = context->template eigen_device<Eigen::GpuDevice>();
    auto stream = device.stream();
    lakernel_input<T><<<grid, F, 0, stream>>>(FW, &filter(0, 0), &grad(0, 0, 0), &output(0, 0, 0));
    // Create filter grad output tensor
    OP_REQUIRES_OK(context, context->allocate_output(1, filter_tensor.shape(),
                                                     &output_tensor));
    auto output2 = output_tensor->template matrix<T>();

    lakernel_filter<T><<<FW, F, 0, stream>>>(B, TS, &input(0, 0, 0), &grad(0, 0, 0), &output2(0, 0));
  }
};

REGISTER_KERNEL_BUILDER(Name("Lookaheadgrad").Device(DEVICE_GPU), LookaheadGradGpuOp<float>);


template<typename T>
__global__ void cckernel_input(int FW, const T* filter, const T* grad, T* output) {
  int TS = gridDim.x;
  int B = gridDim.y;
  int F = blockDim.x;
  int t = blockIdx.x;
  int b = blockIdx.y;
  int f = threadIdx.x;
  output[(t * B + b) * F + f] = 0;
  int FH = FW%2;
  for(int tau = -FH; tau <= FH; tau++) {
    int gidx = tau + t;
    int fidx = FH - tau;
    if (gidx >= 0) {
      output[(t * B + b) * F + f] += grad[(gidx * B + b) * F + f] * filter[fidx * F + f];
    }
  }
}

template<typename T>
__global__ void cckernel_filter(int B, int TS, const T* input, const T* grad, T* output) {
  int F = blockDim.x;
  int FW = gridDim.x;
  int FH = FW%2;
  int f = threadIdx.x;
  int tau = blockIdx.x - FH;
  output[tau * F + f] = 0;
  for (int b = 0; b < B; b++) {
    for (int t = 0; t < TS - tau && t + tau >= 0; t++) {
      output[(tau + FH)* F + f] += grad[(t * B + b) * F + f] * input[((t + tau) * B + b) * F + f];
    }
  }
}


template<typename T>
class ColConvGradGpuOp : public OpKernel {
 public:
  explicit ColConvGradGpuOp(OpKernelConstruction* context) : OpKernel(context) {
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

    int TS = input_tensor.dim_size(0);
    int B = input_tensor.dim_size(1);
    int F = input_tensor.dim_size(2);
    int FW = filter_tensor.dim_size(0);

    dim3 grid(TS, B);
    auto device = context->template eigen_device<Eigen::GpuDevice>();
    auto stream = device.stream();
    cckernel_input<T><<<grid, F, 0, stream>>>(FW, &filter(0, 0), &grad(0, 0, 0), &output(0, 0, 0));
    // Create filter grad output tensor
    OP_REQUIRES_OK(context, context->allocate_output(1, filter_tensor.shape(),
                                                     &output_tensor));
    auto output2 = output_tensor->template matrix<T>();

    cckernel_filter<T><<<FW, F, 0, stream>>>(B, TS, &input(0, 0, 0), &grad(0, 0, 0), &output2(0, 0));
  }
};

REGISTER_KERNEL_BUILDER(Name("Colconvgrad").Device(DEVICE_GPU), ColConvGradGpuOp<float>);

