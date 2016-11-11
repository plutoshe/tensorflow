#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("OctbitMatMul")
    .Input("input_a: T")
    .Input("input_b: qint8")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = true")
    .Attr("scale: float = 0.0")
    .Attr("bias: tensor")
    .Output("output: T")
    .Attr("T: type = DT_FLOAT")
    .SetShapeFn([](InferenceContext* c) {
        return Status::OK();
     });
}
