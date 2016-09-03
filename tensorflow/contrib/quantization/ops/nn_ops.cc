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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/padding.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("QuantizedAvgPool")
    .Input("input: T")
    .Input("min_input: float")
    .Input("max_input: float")
    .Output("output: T")
    .Output("min_output: float")
    .Output("max_output: float")
    .Attr("T: quantizedtype")
    .Attr("ksize: list(int)")
    .Attr("strides: list(int)")
    .Attr(GetPaddingAttrString())
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::AvgPoolShape(c));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return Status::OK();
    })

REGISTER_OP("QuantizedBiasAdd")
    .Input("input: T1")
    .Input("bias: T2")
    .Input("min_input: float")
    .Input("max_input: float")
    .Input("min_bias: float")
    .Input("max_bias: float")
    .Output("output: out_type")
    .Output("min_out: float")
    .Output("max_out: float")
    .Attr("T1: quantizedtype")
    .Attr("T2: quantizedtype")
    .Attr("out_type: quantizedtype")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::BiasAddShape(c));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return Status::OK();
    })

REGISTER_OP("QuantizedConv2D")
    .Input("input: Tinput")
    .Input("filter: Tfilter")
    .Input("min_input: float")
    .Input("max_input: float")
    .Input("min_filter: float")
    .Input("max_filter: float")
    .Output("output: out_type")
    .Output("min_output: float")
    .Output("max_output: float")
    .Attr("Tinput: quantizedtype")
    .Attr("Tfilter: quantizedtype")
    .Attr("out_type: quantizedtype = DT_QINT32")
    .Attr("strides: list(int)")
    .Attr(GetPaddingAttrString())
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::Conv2DShape(c));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return Status::OK();
    })

REGISTER_OP("QuantizedMaxPool")
    .Input("input: T")
    .Input("min_input: float")
    .Input("max_input: float")
    .Output("output: T")
    .Output("min_output: float")
    .Output("max_output: float")
    .Attr("T: quantizedtype")
    .Attr("ksize: list(int)")
    .Attr("strides: list(int)")
    .Attr(GetPaddingAttrString())
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::MaxPoolShape(c));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return Status::OK();
    })

REGISTER_OP("QuantizedRelu")
    .Input("features: Tinput")
    .Input("min_features: float")
    .Input("max_features: float")
    .Output("activations: out_type")
    .Output("min_activations: float")
    .Output("max_activations: float")
    .Attr("Tinput: quantizedtype")
    .Attr("out_type: quantizedtype = DT_QUINT8")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::UnchangedShape(c));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return Status::OK();
    })

REGISTER_OP("QuantizedRelu6")
    .Input("features: Tinput")
    .Input("min_features: float")
    .Input("max_features: float")
    .Output("activations: out_type")
    .Output("min_activations: float")
    .Output("max_activations: float")
    .Attr("Tinput: quantizedtype")
    .Attr("out_type: quantizedtype = DT_QUINT8")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::UnchangedShape(c));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return Status::OK();
    })

REGISTER_OP("QuantizedReluX")
    .Input("features: Tinput")
    .Input("max_value: float")
    .Input("min_features: float")
    .Input("max_features: float")
    .Output("activations: out_type")
    .Output("min_activations: float")
    .Output("max_activations: float")
    .Attr("Tinput: quantizedtype")
    .Attr("out_type: quantizedtype = DT_QUINT8")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::UnchangedShape(c));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return Status::OK();
    })

REGISTER_OP("QuantizedBatchNormWithGlobalNormalization")
    .Input("t: Tinput")
    .Input("t_min: float")
    .Input("t_max: float")
    .Input("m: Tinput")
    .Input("m_min: float")
    .Input("m_max: float")
    .Input("v: Tinput")
    .Input("v_min: float")
    .Input("v_max: float")
    .Input("beta: Tinput")
    .Input("beta_min: float")
    .Input("beta_max: float")
    .Input("gamma: Tinput")
    .Input("gamma_min: float")
    .Input("gamma_max: float")
    .Output("result: out_type")
    .Output("result_min: float")
    .Output("result_max: float")
    .Attr("Tinput: quantizedtype")
    .Attr("out_type: quantizedtype")
    .Attr("variance_epsilon: float")
    .Attr("scale_after_normalization: bool")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input));

      DimensionHandle last_dim = c->Dim(input, 3);
      for (int i = 1; i < 5; ++i) {  // covers m, v, beta, gamma
        ShapeHandle vec;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i * 3), 1, &vec));
        TF_RETURN_IF_ERROR(c->Merge(last_dim, c->Dim(vec, 0), &last_dim));
      }

      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->ReplaceDim(input, 3, last_dim, &out));
      c->set_output(0, out);
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());

      return Status::OK();
    })

}  // namespace tensorflow
