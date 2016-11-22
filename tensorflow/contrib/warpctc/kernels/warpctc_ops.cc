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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

#include "tensorflow/contrib/warpctc/kernels/ctc_helper.h"

namespace tensorflow {
// This a helper class that use for compute for an single sequence, it will be reused for
// each sequence in a minibatch.
template<typename ProbT>
class CpuCTCWorkspace {

 private:
  bool cmp(const int* const labels, const int* const s_label, const int* const e_label, int a, int b) {
    for (int j = s_label[a]; j < e_label[a]; j++)
      for (int k = s_label[b]; k < e_label[b]; k++)
        if (labels[j] == labels[k])
          return true;
    return false;
  }
  int setup_labels(const int* const labels, int L, int S, int BLANK) {
    int e_counter = 0;
    int s_counter = 0;

    s_inc[s_counter++] = 1;

    int repeats = 0;
    int l = 1;
    s_label[0] = 0;
    e_label[0] = 1;
    l_label[0] = 0;
    int last_i = 1;
    for (int i = 1; i < S; ++i) {
      if (labels[i] == BLANK) {
        s_label[l] = last_i;
        e_label[l] = i;
        ++l;
        s_label[l] = i;
        e_label[l] = i + 1;
        // blank setting
        l_label[i] = l;
        ++l;
        last_i = i + 1;
      } else {
        l_label[i] = l;
        // non-blank  setting
      }
    }
    for (int i = 1; i < L; ++i) {
      if (cmp(labels, s_label, e_label, i - 1, i)) {
        s_inc[s_counter++] = e_label[i * 2 - 1] - s_label[i * 2 - 1];
        s_inc[s_counter++] = 1;
        e_inc[e_counter++] = 1;
        e_inc[e_counter++] = e_label[i * 2 + 1] - s_label[i * 2 + 1];
        ++repeats;
      }
      else {
        s_inc[s_counter++] = e_label[i * 2 - 1] - s_label[i * 2 - 1] + 1;
        e_inc[e_counter++] = e_label[i * 2 + 1] - s_label[i * 2 + 1] + 1;
      }
    }
    e_inc[e_counter++] = 1;

    for (int i = 0; i < S; ++i) {
      labels_w_blanks[i] = labels[i];
    }
    // std::cout << "repeats:" << repeats << std::endl;
    // std::cout << "L:" << L << std::endl;
    // std::cout << "S:" << S << std::endl;
    // std::cout << "BLANK:" << BLANK << std::endl;
    // for (int i = 0; i < S; i++) {
    //   std::cout << "labels(" << i << "): " << labels[i] << std::endl;
    // }

    // for (int i = 0; i < L + repeats; i++) {
    //   std::cout << "s_inc(" << i << "): " << s_inc[i] << std::endl;
    // }
    // for (int i = 0; i < L + repeats; i++) {
    //   std::cout << "e_inc(" << i << "): " << e_inc[i] << std::endl;
    // }
    // for (int i = 0; i < S; i++) {
    //   std::cout << "l_label(" << i << "): " << l_label[i] << std::endl;
    // }
    // for (int i = 0; i < l; i++) {
    //   std::cout << "s_label(" << i << "): " << s_label[i] << std::endl;
    // }
    // for (int i = 0; i < l; i++) {
    //   std::cout << "e_label(" << i << "): " << e_label[i] << std::endl;
    // }

    return repeats;
  }

 public:
  ~CpuCTCWorkspace() {
    delete[] alphas;
    delete[] betas;
    delete[] labels_w_blanks;
    delete[] e_inc;
    delete[] s_inc;
    delete[] l_label;
    delete[] e_label;
    delete[] s_label;
    delete[] output;
  }

  CpuCTCWorkspace(int L, int S, int T, int a, int blank_index) : repeats(0), alphabet_size(a){
    alphas = new ProbT[S*T];
    betas = new ProbT[S];

    labels_w_blanks = new int[S];
    e_inc = new int[S];
    s_inc = new int[S];

    l_label = new int[S];

    e_label = new int[S];
    s_label = new int[S];


    output = new ProbT[alphabet_size];
    blank_index_ = blank_index;
  }

  void reset(int L, int S, int T, const int* const labels) {
    std::fill(alphas, alphas + S * T, ctc_helper::neg_inf<ProbT>());
    std::fill(betas, betas + S, ctc_helper::neg_inf<ProbT>());
    repeats = setup_labels(labels, L, S, blank_index_);
  }

  ProbT* alphas;
  ProbT* betas;
  int* labels_w_blanks;

  int* l_label;
  int* e_label;
  int* s_label;
  int* e_inc;
  int* s_inc;

  ProbT* output;
  int repeats;
  int blank_index_;
  const int alphabet_size;
};

template<typename ProbT, bool allow_skip_blank = true>
class CpuCTC {
 public:
  ~CpuCTC() {
    delete workspace_;
    delete[] probs;
  }

  // Noncopyable
  CpuCTC(int alphabet_size, int minibatch,
         const ProbT* const pactivations,
         const int* const pflat_labels,
         const int* const plabel_lengths,
         const int* const plabel_real_lengths,
         const int* const pinput_lengths,
         ProbT* pcosts, ProbT* pgradients,
         const int blank_index = 0) :
    alphabet_size_(alphabet_size), blank_index_(blank_index),
    minibatch_(minibatch), workspace_(nullptr),
    activations(pactivations), flat_labels(pflat_labels), label_lengths(plabel_lengths), label_real_lengths(plabel_real_lengths),
    input_lengths(pinput_lengths), costs(pcosts), grads(pgradients),
    probs(nullptr) {
    int maxT = *std::max_element(input_lengths, input_lengths + minibatch_);
    int maxL = *std::max_element(label_real_lengths, label_real_lengths + minibatch_);
    int maxS = *std::max_element(label_lengths, label_lengths + minibatch_);
    // int maxS = maxL * 2 + 1;
    workspace_ = new CpuCTCWorkspace<ProbT>(maxL, maxS, maxT, alphabet_size_, blank_index_);
    // compute softmax, need to make sure the order is right: currently it is t x b x f
    probs = new ProbT[maxT * minibatch_ * alphabet_size_];
  };

  CpuCTC(const CpuCTC&) = delete;
  CpuCTC& operator=(const CpuCTC&) = delete;

  void cost_and_grad() {
    // compute softmax, need to make sure the order is right: currently it is t x b x f
    for (int mb = 0; mb < minibatch_; ++mb) {
      for(int c = 0; c < input_lengths[mb]; ++c) {
        int col_offset = (mb + minibatch_ * c) * alphabet_size_;
        ProbT max_activation = -std::numeric_limits<ProbT>::infinity();
        for(int r = 0; r < alphabet_size_; ++r)
          max_activation = std::max(max_activation, activations[r + col_offset]);

        ProbT denom = ProbT(0.);
        for(int r = 0; r < alphabet_size_; ++r) {
          probs[r + col_offset] = std::exp(activations[r + col_offset] - max_activation);
          denom += probs[r + col_offset];
        }

        for(int r = 0; r < alphabet_size_; ++r) {
          probs[r + col_offset] /= denom;
        }
      }
    }

    for (int mb = 0; mb < minibatch_; ++mb) {
      const int T = input_lengths[mb]; // Length of utterance (time)
      const int S = label_lengths[mb]; // Length of utterance (time)
      const int L = label_real_lengths[mb]; // Number of labels in transcription

      int label_count = std::accumulate(label_lengths, label_lengths + mb, 0);
      costs[mb] = cost_and_grad_kernel(flat_labels + label_count, T, S, L, mb);
    }
  }

 private:
  ctc_helper::log_plus<ProbT> log_add;

  int alphabet_size_; // Number of characters plus blank
  int blank_index_;
  int minibatch_;
  CpuCTCWorkspace<ProbT>* workspace_;

  const ProbT* const activations;
  const int* const flat_labels;
  const int* const label_lengths;
  const int* const label_real_lengths;
  const int* const input_lengths;

  ProbT* costs;
  ProbT* grads;

  ProbT* probs;


  ProbT cost_and_grad_kernel(const int* const labels, int T, int S, int L, int mb) {
    workspace_->reset(L, S, T, labels);
    CpuCTCWorkspace<ProbT>& ctcm(*workspace_);

    bool over_threshold = false;

    // T should be at least greater than L + number of repeats.
    CHECK(L + ctcm.repeats <= T);

    ProbT llForward = compute_alphas(L, S, T, mb, ctcm);
    // std::cout << "llForward: " << llForward << std::endl;
    ProbT llBackward = compute_betas_and_grad(llForward, L, S, T, mb, ctcm);

    ProbT diff = std::abs(llForward - llBackward);
    if (diff > ctc_helper::threshold) {
      over_threshold = true;
    }

    // std::cout << "llBackward: " << llBackward << std::endl;

    return -llForward;
    // return -llForward;
  }


  ProbT compute_alphas(int L, int S, int T, int mb, CpuCTCWorkspace<ProbT>& ctcm) {
    int repeats = ctcm.repeats;
    const int* const e_inc = ctcm.e_inc;
    const int* const s_inc = ctcm.s_inc;
    const int* const labels = ctcm.labels_w_blanks;
    const int* const e_label = ctcm.e_label;
    const int* const s_label = ctcm.s_label;
    const int* const l_label = ctcm.l_label;

    Accessor2D<ProbT> alphas(alphabet_size_, ctcm.alphas);
    Accessor3D<ProbT> yprobs(minibatch_, alphabet_size_, probs);
    int start = ((L + repeats - T) < 0) ? 0 : 1;
    int end = L > 0 ? e_label[1] : 1;
    // std::cout << "start:" << start << std::endl;
    // std::cout << "end:" << end << std::endl;
    // // Setup the boundary condition.
    // std::cout << "einc:" << std::endl;
    // for(int t = 0; t < S; ++t) {
    //   std::cout << e_inc[t] << std::endl;
    // }
    // std::cout << "sinc:" << std::endl;
    // for(int t = 0; t < S; ++t) {
    //   std::cout << s_inc[t] << std::endl;
    // }

    for (int i = start; i < end; ++i) {
        alphas(0, i) = std::log(yprobs(0, mb, labels[i]));
    }
    for(int t = 1; t < T; ++t) {
      // Still a bit murky here.
      int remain = L + repeats - (T - t);
      // std::cout << "start, end, remain, t -1:" << start << "," << end << "," << remain <<  "," << t - 1 << std::endl;
      if(remain >= 0) start += s_inc[remain];
      if(t <= L + repeats) end += e_inc[t - 1];
      int startloop = start;

      if (start == 0) {
        alphas(t, 0) = alphas(t-1, 0) + std::log(yprobs(t, mb, blank_index_));
        startloop += 1;
      }

      for(int i = startloop; i < end; ++i) {
        int l = l_label[i];
        ProbT prev_sum = alphas(t - 1, i);
        for (int j = s_label[l - 1]; j < e_label[l - 1]; j++) {
           prev_sum = log_add(prev_sum, alphas(t - 1, j));
        }
        // Skip two if not on blank and not on repeat.
        if (allow_skip_blank && labels[i] != blank_index_ && l >= 2) {
          // std::cout <<s_label[l - 2] <<  "l" << e_label[l - 2] << std::endl;

          for (int j = s_label[l - 2]; j < e_label[l - 2]; j++)
            if (labels[i] != labels[j]) {
              prev_sum = log_add(prev_sum, alphas(t - 1, j));
            }
        }
        alphas(t, i) = prev_sum + std::log(yprobs(t, mb, labels[i]));
      }
    }

    ProbT loglike = ctc_helper::neg_inf<ProbT>();
    for(int i = start; i < end; ++i) {
      loglike = log_add(loglike, alphas(T- 1, i));
    }

    return loglike;
  }


  // Starting from T, we sweep backward over the alpha array computing one column
  // of betas as we go.  At each position we can update product alpha * beta and then
  // sum into the gradient associated with each label.
  // NOTE computes gradient w.r.t UNNORMALIZED final layer activations.
  // Assumed passed in grads are already zeroed!
  ProbT compute_betas_and_grad(ProbT log_partition, int L, int S, int T, int mb,
                               CpuCTCWorkspace<ProbT>& ctcm) {
    int repeats = ctcm.repeats;
    const int* const e_inc = ctcm.e_inc;
    const int* const s_inc = ctcm.s_inc;
    const int* const labels = ctcm.labels_w_blanks;
    const int* const e_label = ctcm.e_label;
    const int* const s_label = ctcm.s_label;
    const int* const l_label = ctcm.l_label;

    Accessor2D<ProbT> alphas(alphabet_size_, ctcm.alphas);
    Accessor3D<ProbT> yprobs(minibatch_, alphabet_size_, probs);
    Accessor3D<ProbT> ygrads(minibatch_, alphabet_size_, grads);

    ProbT* betas = ctcm.betas;
    ProbT* output = ctcm.output;

    int start = S > 1 ? s_label[l_label[S - 2]] : 0;
    int end = (T > L + repeats) ? S : S-1;

    std::fill(output, output + alphabet_size_, ctc_helper::neg_inf<ProbT>());

    //set the starting values in the beta column at the very right edge
    for (int i = start; i < end; ++i) {
      betas[i] = std::log(yprobs(T -1, mb, labels[i]));

      //compute alpha * beta in log space at this position in (S, T) space
      alphas(T -1, i) += betas[i];

      //update the gradient associated with this label
      //essentially performing a reduce-by-key in a sequential manner
      output[labels[i]] = log_add(alphas(T -1, i), output[labels[i]]);
    }

    //update the gradient wrt to each unique label
    for (int i = 0; i < alphabet_size_; ++i) {
      if (output[i] == 0.0 || output[i] == ctc_helper::neg_inf<ProbT>() ||
          yprobs(T - 1, mb, i) == 0.0) {
        ygrads(T - 1, mb, i) = yprobs(T - 1, mb, i);
      } else {
        ygrads(T - 1, mb, i) = yprobs(T - 1, mb, i)
                  - std::exp(output[i] - std::log(yprobs(T - 1, mb, i)) - log_partition);
      }
    }
    //loop from the second to last column all the way to the left
    for(int t = T - 2; t >= 0; --t) {
      int remain = L + repeats - (T - t);
      if(remain >= -1) start -= s_inc[remain + 1];
      if(t < L + repeats) end -= e_inc[t];
      int endloop = end == S ? end - 1 : end;

      std::fill(output, output + alphabet_size_, ctc_helper::neg_inf<ProbT>());
      // std::cout << "t:" << t << std::endl;
      // std::cout << "start:" << start << std::endl;
      // std::cout << "endloop:" << endloop << std::endl;
      for(int i = start; i < endloop; ++i) {
        int l = l_label[i];
        ProbT next_sum = betas[i];
        for (int j = s_label[l + 1]; j < e_label[l + 1]; j++)
          next_sum = log_add(next_sum, betas[j]);
        // Skip two if not on blank and not on repeat.
        if (labels[i] != blank_index_){
          for (int j = s_label[l + 2]; j < e_label[l + 2]; j++)
            if (labels[i] != labels[j])
              next_sum = log_add(next_sum, betas[j]);
        }
        betas[i] = next_sum + std::log(yprobs(t, mb, labels[i]));

        //compute alpha * beta in log space
        alphas(t, i) += betas[i];

        //update the gradient associated with this label
        output[labels[i]] = log_add(alphas(t, i), output[labels[i]]);
      }

      if (end == S) {
        int l = l_label[S - 1];
        for (int j = s_label[l]; j < e_label[l]; j++) {
          betas[j] = betas[j] + std::log(yprobs(t, mb, blank_index_));
          alphas(t, j) += betas[j];

          output[labels[j]] = log_add(alphas(t, j), output[labels[j]]);
        }
      }

      // go over the unique labels and compute the final grad
      // wrt to each one at this time step
      for (int i = 0; i < alphabet_size_; ++i) {
        if (output[i] == 0.0 || output[i] == ctc_helper::neg_inf<ProbT>() ||
            yprobs(t, mb, i) == 0.0) {
          ygrads(t, mb, i) = yprobs(t, mb, i);
        } else {
          ygrads(t, mb, i) =
              yprobs(t, mb, i) - std::exp(output[i] - std::log(yprobs(t, mb, i)) - log_partition);
        }
      }
    }

    ProbT loglike = ctc_helper::neg_inf<ProbT>();
    for(int i = start; i < end; ++i) {
      loglike = log_add(loglike, betas[i]);
    }

    return loglike;
  }
};

class CpuWarpCTCLossOp : public OpKernel {
 public:
  int blank_index = -1;
  explicit CpuWarpCTCLossOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    bool p, c;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("preprocess_collapse_repeated", &p));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("blank_index", &blank_index));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("ctc_merge_repeated", &c));
  }

  void Compute(OpKernelContext* ctx) override {
    /* activations pointer to the activations in either CPU or GPU
    *  addressable memory, depending on info.  We assume a fixed
    *  memory layout for this 3 dimensional tensor, which has dimension
    *  (t, n, p), where t is the time index, n is the minibatch index,
    *  and p indexes over probabilities of each symbol in the alphabet.
    *  More precisely, element (t, n, p), for a problem with mini_batch examples
    *  in the mini batch, and alphabet_size symbols in the alphabet, is located at:
    *  activations[(t * mini_batch + n) * alphabet_size + p]
    */
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

    // get the label and label_length ready.
    int64 last_first = 0, sum = 0, label_sum = 0, last_second = 0;
    //  A concatenation of all the labels for the minibatch.
    std::vector<int> labels;
    // The length of each label for each example in the minibatch.
    // and add a blank between adjacent labels
    // add blank at the begin and the end of minibatch
    std::vector<int> label_lengths;
    std::vector<int> label_real_lengths;
    if (labels_indices_tensor.dim_size(0) > 0) {
      // initialization
      labels.push_back(blank_index);
      last_first = labels_indices(0, 0);
      last_second = labels_indices(0, 1);
      labels.push_back(labels_values(0));
      sum = 1; label_sum = 2;

      for(int i = 1; i < labels_indices_tensor.dim_size(0); i++) {
        int64 first, second;
        int label_of_first;
        first = labels_indices(i, 0);
        second = labels_indices(i, 1);
        label_of_first = labels_values(i);
        if (first == last_first) {
          if (second != last_second) {
              labels.push_back(blank_index);
              label_sum++;
              sum++;
          }
          labels.push_back(label_of_first);
          label_sum++;
        } else {
          labels.push_back(blank_index);
          label_lengths.push_back(label_sum+1);
          label_real_lengths.push_back(sum);
          labels.push_back(blank_index);
          labels.push_back(label_of_first);
          last_first = first;
          sum = 1;
          label_sum = 2;
        }
        last_second = second;
      }
      if (sum != 0) {
        labels.push_back(blank_index);
        label_lengths.push_back(label_sum+1);
        label_real_lengths.push_back(sum);
      }
    }

    Tensor* loss = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("loss", seq_len_tensor.shape(), &loss));
    auto loss_t = loss->vec<float>();

    Tensor* gradient;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("gradient", inputs_shape, &gradient));
    auto gradient_t = gradient->tensor<float, 3>();
    CpuCTC<float> ctc(alphabet_size, seq_len_tensor.dim_size(0), activations,
                      labels.data(), label_lengths.data(), label_real_lengths.data(),
                      seq_len.data(), loss_t.data(), gradient_t.data(), blank_index);

    ctc.cost_and_grad();
  }
};

REGISTER_KERNEL_BUILDER(Name("WarpCtcLoss").Device(DEVICE_CPU), CpuWarpCTCLossOp);
}  // end namespace tensorflow
