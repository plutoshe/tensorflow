/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// See docs in ../ops/ctc_ops.cc.
#include <cuda_runtime.h>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"
#define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/contrib/warpctc/kernels/gpu_ctc_kernels.h"

namespace tensorflow {

using namespace ctc_helper;

template<typename T>
class Gpu {
 public:
  static T* Clone(const T* src, int size, cudaStream_t stream) {
    T* res = nullptr;
    cudaError_t err = cudaMalloc(&res, sizeof(T)*size);
    if (err != cudaSuccess) {
      size_t available, total;
      cudaMemGetInfo(&available, &total);
      CHECK(false) << "CUDA error: " << cudaGetErrorString(err) << ". Device has "
          << available << " of " << total << " need " << sizeof(T)*size  << std::endl;
    }

    err = cudaMemcpyAsync(res, src, sizeof(T)*size, cudaMemcpyHostToDevice, stream);
    CHECK(err == cudaSuccess) << cudaGetErrorString(err) << std::endl;
    return res;
  }

  // Allocate the memory for size number of T's
  static T* allocate(int size) {
    T* res;
    cudaError_t err = cudaMalloc(&res, sizeof(T)*size);
    if (err != cudaSuccess) {
      LOG(INFO) << "CUDA error: " << cudaGetErrorString(err) << std::endl;
      size_t available, total;
      cudaMemGetInfo(&available, &total);
      LOG(INFO) << "Device has " << available << " out of " << total << " but need " << sizeof(T)*size;
      CHECK(false) << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
    return res;
  }

  static T* allocate(int size, T default_value, cudaStream_t stream) {
    T* res;
    cudaError_t err = cudaMalloc(&res, sizeof(T)*size);
    if (err != cudaSuccess) {
      LOG(INFO) << "CUDA error: " << cudaGetErrorString(err) << std::endl;
      size_t available, total;
      cudaMemGetInfo(&available, &total);
      LOG(INFO) << "Device has " << available << " out of " << total << " but need " << sizeof(T)*size;
      CHECK(false) << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
    cudaMemsetAsync(res, default_value, sizeof(T)*size, stream);
    return res;
  }

  // delete the memory needed for this.
  static void deallocate(T* ptr, int size = 0) {
    cudaFree(ptr);
  }
};

// Instead of going for the log(f), we will stay  with f in computing softmax on the T*B*F.
// While this is not the most efficient solution, it should have plenty of speedup and code is
// still readable.
template<typename T>
__global__ void softmax(const T* activations, T* probs, int B, int F) {
  int tidx = blockIdx.x;
  int bidx = threadIdx.x;

  int start = tidx * B * F  + bidx * F;
  T lmax = activations[start];

  // first we get the max out of activations.
  for (int i = start + 1; i < start + F; ++i) {
    if (lmax < activations[i]) lmax = activations[i];
  }

  T sum = 0;
  for (int i = start; i < start + F; ++i) {
    probs[i] = std::exp(activations[i] - lmax);
    sum += probs[i];
  }

  for (int i = start; i < start + F; ++i) {
    probs[i] /= sum;
  }
}

template <typename ProbT>
class GpuCTC {
 public:
  ~GpuCTC() {
    Gpu<ProbT>::deallocate(nll_forward_);
    Gpu<ProbT>::deallocate(nll_backward_);
    Gpu<int>::deallocate(repeats_);
    Gpu<int>::deallocate(label_offsets_);

    Gpu<int>::deallocate(utt_length_);
    Gpu<int>::deallocate(label_sizes_); // L

    Gpu<int>::deallocate(labels_without_blanks_);
    Gpu<int>::deallocate(labels_with_blanks_);
    Gpu<ProbT>::deallocate(alphas_);
    Gpu<ProbT>::deallocate(denoms_);
    Gpu<ProbT>::deallocate(probs_);
  }

  GpuCTC(int alphabet_size, int minibatch, cudaStream_t stream) :
    out_dim_(alphabet_size), minibatch_(minibatch), stream_(stream) {
    nll_forward_ = Gpu<ProbT>::allocate(minibatch_);
    nll_backward_ = Gpu<ProbT>::allocate(minibatch_);
    repeats_ = Gpu<int>::allocate(minibatch_);
    label_offsets_ = Gpu<int>::allocate(minibatch_);
  };

  // Noncopyable
  GpuCTC(const GpuCTC&) = delete;
  GpuCTC& operator=(const GpuCTC&) = delete;

  ctcStatus_t
  cost_and_grad(const ProbT* const activations,
                ProbT* grads,
                ProbT* costs,
                const int* const flat_labels,
                const int* const label_lengths,
                const int* const input_lengths) {
    size_t best_config;
    ctcStatus_t status = choose_config(flat_labels, label_lengths,
                                       input_lengths, best_config);
    if (status != CTC_STATUS_SUCCESS)
      return status;

    softmax<float><<<T_, minibatch_, 0, stream_>>>(activations, probs_, minibatch_, out_dim_);

    launch_gpu_kernels(probs_, grads, best_config);

    cudaError_t status_mem, status_sync;
    status_mem = cudaMemcpyAsync(costs, nll_forward_,
                                 sizeof(ProbT) * minibatch_,
                                 cudaMemcpyDeviceToHost, stream_);
    status_sync = cudaStreamSynchronize(stream_);
    if (status_mem != cudaSuccess || status_sync != cudaSuccess)
      return CTC_STATUS_MEMOPS_FAILED;

    return CTC_STATUS_SUCCESS;
  }

 private:

  template<int NT, int VT>
  ctcStatus_t launch_alpha_beta_kernels(const ProbT* const probs,
                                        ProbT* grads) {
    compute_alpha_kernel<ProbT, NT, VT><<<minibatch_, NT, 0, stream_>>>
        (probs, label_sizes_, utt_length_,
            repeats_, labels_without_blanks_, label_offsets_,
            labels_with_blanks_, alphas_, nll_forward_,
            minibatch_, out_dim_, S_, T_);

    compute_betas_and_grad_kernel<ProbT, NT, VT><<<minibatch_, NT, 0, stream_>>>
        (probs, label_sizes_, utt_length_, repeats_,
            labels_with_blanks_, alphas_, nll_forward_, nll_backward_,
            grads, minibatch_, out_dim_, S_, T_);

    cudaStreamSynchronize(stream_);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
      return CTC_STATUS_EXECUTION_FAILED;

    return CTC_STATUS_SUCCESS;
  }


  ctcStatus_t
  launch_gpu_kernels(const ProbT* const probs,
                     ProbT* grads, size_t config) {
    switch(config) {
      case 0:   {return launch_alpha_beta_kernels<32,   1>(probs, grads);}
      case 1:   {return launch_alpha_beta_kernels<64,   1>(probs, grads);}
      case 2:   {return launch_alpha_beta_kernels<128,  1>(probs, grads);}
      case 3:   {return launch_alpha_beta_kernels<64,   3>(probs, grads);}
      case 4:   {return launch_alpha_beta_kernels<128,  2>(probs, grads);}
      case 5:   {return launch_alpha_beta_kernels<32,   9>(probs, grads);}
      case 6:   {return launch_alpha_beta_kernels<64,   6>(probs, grads);}
      case 7:   {return launch_alpha_beta_kernels<128,  4>(probs, grads);}
      case 8:   {return launch_alpha_beta_kernels<64,   9>(probs, grads);}
      case 9:   {return launch_alpha_beta_kernels<128,  6>(probs, grads);}
      case 10:  {return launch_alpha_beta_kernels<128,  9>(probs, grads);}
      case 11:  {return launch_alpha_beta_kernels<128, 10>(probs, grads);}
    }

    return CTC_STATUS_EXECUTION_FAILED;
  }

  void setup(const int* const flat_labels,
             const int* const label_lengths,
             const int* const input_lengths) {
    // This is the max of all S and T for all valid examples in the minibatch.
    // A valid example is one for which L + repeats <= T
    S_ = 0;
    T_ = 0;

    // This is the max of all timesteps, valid or not. Needed to compute offsets
    int Tmax = 0;

    // This is the max of all labels, valid or not. Needed to compute offsets
    int Lmax = 0;
    int total_label_length = 0;


    int repeats[minibatch_];
    int label_offsets[minibatch_];

    for (int j = 0; j < minibatch_; ++j) {
      const int L = label_lengths[j];
      const int local_T = input_lengths[j];
      const int *label_ptr = &(flat_labels[total_label_length]);

      label_offsets[j] = total_label_length;
      total_label_length += L;

      int repeat_counter = 0;

      for (int i = 1; i < L; ++i)
        repeat_counter += (label_ptr[i] == label_ptr[i-1]);

      repeats[j] = repeat_counter;
      const bool valid_label = ((L + repeat_counter) <= local_T);

      // Only update S and T if label is valid
      S_ = (valid_label) ? std::max(S_, L) : S_;
      T_ = (valid_label) ? std::max(T_, local_T) : T_;

      Tmax = std::max(Tmax, local_T);
      Lmax = std::max(Lmax, L);
    }

    repeats_ = Gpu<int>::Clone(repeats, minibatch_, stream_);
    label_offsets_ = Gpu<int>::Clone(label_offsets, minibatch_, stream_);

    S_ = 2 * S_ + 1;
    cols_ = minibatch_ * Tmax;

    // Allocate memory for T
    utt_length_ = Gpu<int>::Clone(input_lengths, minibatch_, stream_);
    label_sizes_ = Gpu<int>::Clone(label_lengths, minibatch_, stream_);
    labels_without_blanks_ = Gpu<int>::Clone(flat_labels, total_label_length, stream_);
    labels_with_blanks_ = Gpu<int>::allocate(S_*minibatch_, ctc_helper::BLANK, stream_);
    alphas_ = Gpu<ProbT>::allocate((S_ * T_) * minibatch_);
    denoms_ = Gpu<ProbT>::allocate(cols_);
    probs_ = Gpu<ProbT>::allocate(out_dim_ * cols_ * T_);
  }

  ctcStatus_t
  choose_config(const int* const flat_labels,
                const int* const label_lengths,
                const int* const input_lengths,
                size_t& best_config) {
    // Setup the metadata for GPU
    setup(flat_labels, label_lengths, input_lengths);

    constexpr int num_configs = 12;

    int config_NT[num_configs] =
    {32, 64, 128, 64, 128, 32, 64, 128, 64, 128, 128, 128};
    int config_VT[num_configs] =
    { 1,  1,   1,  3,   2,  9,  6,   4,  9,   6,   9,  10};

    best_config = 0;

    for (int i = 0; i < num_configs; ++i) {
      if ((config_NT[i]* config_VT[i]) >= S_)
        break;
      else
        best_config++;
    }

    if (best_config >= num_configs)
      return CTC_STATUS_UNKNOWN_ERROR;

    return CTC_STATUS_SUCCESS;
  }

  int out_dim_; // Number of characters plus blank
  int minibatch_;

  int S_ = -1;
  int T_ = -1;

  int cols_ = -1; // Number of columns in activations

  cudaStream_t stream_;

  int *utt_length_ = nullptr; // T
  int *label_sizes_ = nullptr; // L
  int *repeats_ = nullptr; // repeats_
  int *label_offsets_ = nullptr;
  int *labels_without_blanks_ = nullptr;
  int *labels_with_blanks_ = nullptr;
  ProbT *alphas_ = nullptr;
  ProbT *nll_forward_ = nullptr;
  ProbT *nll_backward_ = nullptr;
  ProbT *denoms_ = nullptr; // Temporary storage for denoms for softmax
  ProbT *probs_ = nullptr; // Temporary storage for probabilities (softmax output)
};

class GpuWarpCTCLossOp : public OpKernel {
 public:
  explicit GpuWarpCTCLossOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    bool p, c;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("preprocess_collapse_repeated", &p));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("ctc_merge_repeated", &c));
  }

  void Compute(OpKernelContext* ctx) override {

    // Calculate the score analytically
    cudaStream_t stream = ctx->template eigen_device<Eigen::GpuDevice>().stream();

    const Tensor& input_tensor = ctx->input(0);
    const Tensor& labels_indices_tensor = ctx->input(1);
    const Tensor& labels_values_tensor = ctx->input(2);
    const Tensor& seq_len_tensor = ctx->input(3);

    auto input = input_tensor.tensor<float, 3>();
    float *activations = (float*)input.data();
    const TensorShape& inputs_shape = input_tensor.shape();
    const int64 max_time = inputs_shape.dim_size(0);
    const int64 batch_size = inputs_shape.dim_size(1);
    const int64 num_classes = inputs_shape.dim_size(2);

    const int alphabet_size = num_classes;
    TensorShape labels_shape({batch_size, max_time});
    std::vector<int64> order{0, 1};
    auto labels_indices = labels_indices_tensor.tensor<int64, 2>();
    auto labels_values = labels_values_tensor.tensor<int, 1>();
    auto seq_len = seq_len_tensor.tensor<int, 1>();

    int64 last_first = 0, sum = 0;
    std::vector<int> labels;
    std::vector<int> label_lengths;
    for(int i = 0; i < labels_indices_tensor.dim_size(0); i++) {
      int64 first;
      int label_of_first;
      cudaMemcpyAsync(&first, &labels_indices(i,0), sizeof(int64), cudaMemcpyDeviceToHost, stream);
      cudaMemcpyAsync(&label_of_first, &labels_values(i), sizeof(int), cudaMemcpyDeviceToHost, stream);
      labels.push_back(label_of_first);
      if (first == last_first) {
        sum++;
      } else {
        label_lengths.push_back(sum);
        last_first = first;
        sum = 1;
      }
    }
    if (sum != 0) {
      label_lengths.push_back(sum);
    }
    int size_of_lengths = seq_len_tensor.dim_size(0);
    int lengths[size_of_lengths];
    cudaMemcpyAsync(lengths, seq_len.data(), size_of_lengths * sizeof(int), cudaMemcpyDeviceToHost, stream);

    Tensor* loss = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("loss", seq_len_tensor.shape(), &loss));
    auto loss_t = loss->vec<float>();

    Tensor* gradient;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("gradient", inputs_shape, &gradient));
    auto gradients = gradient->tensor<float, 3>();
    float loss_cpu[size_of_lengths];

    cudaMemset(gradients.data(), 0, gradient->NumElements() * sizeof(float));

    GpuCTC<float> ctc(alphabet_size, size_of_lengths, stream);

    ctc.cost_and_grad(activations, gradients.data(), loss_cpu,
                      labels.data(), label_lengths.data(), lengths);

    cudaMemcpyAsync(loss_t.data(), loss_cpu, size_of_lengths * sizeof(float), cudaMemcpyHostToDevice, stream);
  }
};

REGISTER_KERNEL_BUILDER(Name("WarpCtcLoss").Device(DEVICE_GPU), GpuWarpCTCLossOp);
}  // end namespace tensorflow
