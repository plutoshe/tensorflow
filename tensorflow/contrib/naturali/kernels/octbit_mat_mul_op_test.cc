/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include <functional>
#include <memory>
#include <vector>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

class OctbitMatMulTest : public OpsTestBase {
 protected:
};

// Runs two small matrices through the operator, and leaves all the parameters
// at their default values.
TEST_F(OctbitMatMulTest, Small_NoParams) {
  Tensor a1 = Tensor(DT_FLOAT, {2});
  auto b1 = a1.vec<float>();
  for (int i = 0; i < 2; i++) {
    b1(i) = 127 * 2016;
  }

  TF_ASSERT_OK(NodeDefBuilder("octbit_mat_mul_op", "OctbitMatMul")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_QINT8))
                   .Attr("scale", 3.0)
                   .Attr("bias", a1)
                   .Attr("T", DataTypeToEnum<float>::v())
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  std::vector<float> v2;
  int B = 1;
  for (int i = 0; i < 64 * B; i++) {
      v2.push_back(-1.);
  }
  AddInputFromArray<float>(TensorShape({B, 64}), v2);

  std::vector<qint8> v;
  int A = 1;
  for (int i = 0; i < 64 * A; i++) {
      v.push_back(qint8(i));
  }

  AddInputFromArray<qint8>(TensorShape({A, 64}), v);

  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_FLOAT, TensorShape({B, A}));
  test::FillValues<float>(&expected, {-6048.});
  auto bb = *GetOutput(0);
  auto bmat = bb.matrix<float>();
  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < A; ++j) {
      std::cout << bmat(i, j) << std::endl;
    }
  }
  std::cout << bb.matrix<float>()(0, 0);
  test::ExpectTensorEqual<float>(expected, bb);
}

TEST_F(OctbitMatMulTest, NoParams) {
  Tensor a1 = Tensor(DT_FLOAT, {4});
  auto b1 = a1.vec<float>();
  for (int i = 0; i < 4; i++) {
    b1(i) = 2016 * 127;
  }
  b1(0) = 64 * 127;
  TF_ASSERT_OK(NodeDefBuilder("octbit_mat_mul_op", "OctbitMatMul")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_QINT8))
                   .Attr("scale", 2.0)
                   .Attr("bias", a1)
                   .Attr("T", DataTypeToEnum<float>::v())
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  std::vector<float> v2;
  int B = 2;
  for (int i = 0; i < 64 * B; i++) {
      v2.push_back(-1.);
  }
  AddInputFromArray<float>(TensorShape({B, 64}), v2);

  std::vector<qint8> v;
  int A = 4;
  for (int i = 0; i < 64 * A; i++) {
      if (i < 64) {
        v.push_back(qint8(1));
      } else {
        v.push_back(qint8(i%64));
      }
  }

  AddInputFromArray<qint8>(TensorShape({A, 64}), v);

  TF_ASSERT_OK(RunOpKernel());
  Tensor expected(allocator(), DT_FLOAT, TensorShape({B, A}));
  test::FillValues<float>(&expected, {-128., -4032, -4032, -4032, -128, -4032, -4032, -4032});
  auto bb = *GetOutput(0);
  auto bmat = bb.matrix<float>();
  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < A; ++j) {
      std::cout << bmat(i, j) << std::endl;
    }
  }
  std::cout << bb.matrix<float>()(0, 0);
  test::ExpectTensorEqual<float>(expected, bb);
}
}  // namespace tensorflow

