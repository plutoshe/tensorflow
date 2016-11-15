from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import load_library
from tensorflow.python.framework import ops
from tensorflow.python.framework import common_shapes
from tensorflow.python.platform import resource_loader
# from tensorflow.contrib.naturali.python.ops import lookahead_grad_ops

_octbit_ops_so = load_library.load_op_library(
    resource_loader.get_path_to_datafile("_octbit_ops.so"))
assert _octbit_ops_so, "Could not load _octbit_ops.so."

def octbit_mat_mul(x1, x2):
    return _octbit_ops_so.octbit_mat_mul(x1, x2)


