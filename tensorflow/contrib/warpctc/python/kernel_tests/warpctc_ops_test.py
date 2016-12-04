# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tests for tensorflow.ctc_ops.ctc_decoder_ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib.warpctc.python.ops import warpctc_ops


def MultiLabelSimpleSparseTensorFrom(x, blank_index):
  """Create a very simple SparseTensor with dimensions (batch, time).
  Args:
    x: a list of lists of type int
  Returns:
    x_ix and x_val, the indices and values of the SparseTensor<2>.
  """
  x_ix = []
  x_val = []
  for batch_i, batch in enumerate(x):
    time_slot = 0
    for i in range(len(batch)):
      if batch[i] == blank_index:
        time_slot += 1
        x_ix.append([batch_i, time_slot])
        x_val.append(batch[i])
        time_slot += 1
      else:
        x_ix.append([batch_i, time_slot])
        x_val.append(batch[i])

  x_shape = [len(x), np.asarray(x_ix).max(0)[1]+1]
  x_ix = tf.constant(x_ix, tf.int64)
  x_val = tf.constant(x_val, tf.int32)
  x_shape = tf.constant(x_shape, tf.int64)

  return tf.SparseTensor(x_ix, x_val, x_shape)


def SimpleSparseTensorFrom(x, blank_index):
  """Create a very simple SparseTensor with dimensions (batch, time).
  Args:
    x: a list of lists of type int
  Returns:
    x_ix and x_val, the indices and values of the SparseTensor<2>.
  """
  x_ix = []
  x_val = []
  for batch_i, batch in enumerate(x):
    time_slot = 0
    for i in range(len(batch)):
      if batch[i] == blank_index:
        time_slot += 1
        x_ix.append([batch_i, time_slot])
        x_val.append(batch[i])
        time_slot += 1
      else:
        x_ix.append([batch_i, time_slot])
        x_val.append(batch[i])

  x_shape = [len(x), np.asarray(x_ix).max(0)[1]+1]
  x_ix = tf.constant(x_ix, tf.int64)
  x_val = tf.constant(x_val, tf.int32)
  x_shape = tf.constant(x_shape, tf.int64)

  return tf.SparseTensor(x_ix, x_val, x_shape)


class CTCLossTest(tf.test.TestCase):

  def _testCTCLoss(self, inputs, seq_lens, labels,
                   loss_truth, grad_truth, blank_index=0, expected_err_re=None):
    self.assertEquals(len(inputs), len(grad_truth))

    inputs_t = tf.constant(inputs)

    with self.test_session(use_gpu=True) as sess:
      loss = warpctc_ops.warp_ctc_loss(inputs=inputs_t,
                                      labels=labels,
                                      sequence_length=seq_lens,
                                      blank_index=blank_index)
      grad = tf.gradients(loss, [inputs_t])[0]
      self.assertShapeEqual(loss_truth, loss)
      self.assertShapeEqual(grad_truth, grad)

      if expected_err_re is None:
        (tf_loss, tf_grad) = sess.run([loss, grad])
        self.assertAllClose(tf_loss, loss_truth, atol=1e-6)
        self.assertAllClose(tf_grad, grad_truth, atol=1e-6)
      else:
        with self.assertRaisesOpError(expected_err_re):
          sess.run([loss, grad])

  def testBasic(self):
    """Test two batch entries."""
    # Input and ground truth from Alex Graves' implementation.
    #
    #### Batch entry 0 #####
    # targets: 5 0 5 1 5 2 5 1 5
    # outputs:
    # 0 [0.633766, 0.221185, 0.0917319, 0.0129757, 0.0142857, 0.0260553],
    # 1 [0.111121, 0.588392, 0.278779, 0.0055756, 0.00569609, 0.010436],
    # 2 [0.0357786, 0.633813, 0.321418, 0.00249248, 0.00272882, 0.0037688],
    # 3 [0.0663296, 0.643849, 0.280111, 0.00283995, 0.0035545, 0.00331533],
    # 4 [0.458235, 0.396634, 0.123377, 0.00648837, 0.00903441, 0.00623107],
    # 5 [0.0357786, 0.633813, 0.321418, 0.00249248, 0.00272882, 0.0037688],
    # 6 [0.0357786, 0.633813, 0.321418, 0.00249248, 0.00272882, 0.0037688],
    # alpha:
    # 0 -inf -0.45607 -inf -inf -inf -inf -inf -inf -inf 
    # 1 -inf -inf -5.01856 -inf -inf -inf -inf -inf -inf 
    # 2 -inf -inf -inf -5.47457 -inf -inf -inf -inf -inf 
    # 3 -inf -inf -inf -inf -11.18376 -inf -inf -inf -inf
    # 4 -inf -inf -inf -inf -inf -13.276 -inf -inf -inf 
    # 5 -inf -inf -inf -inf -inf -inf -18.856 -inf -inf 
    # 6 -inf -inf -inf -inf -inf -inf -inf -19.313 -inf 
    # beta:
    # 0 -inf -2.88604 -inf -inf -inf -inf -inf -inf -inf -inf -inf
    # 1 -inf -inf -inf -2.35568 -inf -inf -inf -inf -inf -inf -inf
    # 2 -inf -inf -inf -inf -inf -1.22066 -inf -inf -inf -inf -inf
    # 3 -inf -inf -inf -inf -inf -inf -inf -0.780373 -inf -inf -inf
    # 4 -inf -inf -inf -inf -inf -inf -inf -inf -inf 0 0
    # prob: -3.34211
    # outputDerivs:
    # 0 -0.366234 0.221185 0.0917319 0.0129757 0.0142857 0.0260553
    # 1 0.111121 -0.411608 0.278779 0.0055756 0.00569609 0.010436
    # 2 0.0357786 0.633813 -0.678582 0.00249248 0.00272882 0.0037688
    # 3 0.0663296 -0.356151 0.280111 0.00283995 0.0035545 0.00331533
    # 4 -0.541765 0.396634 0.123377 0.00648837 0.00903441 0.00623107
    #
    
    # max_time_steps == 7
    depth = 6
    blank_index_ = 5
    # seq_len_0 == 5
    targets_0 = [5, 0, 5, 1, 5, 2, 5, 1, 5]
    loss_log_prob_0 = -19.313276
    # dimensions are time x depth
    input_prob_matrix_0 = np.asarray(
        [[0.633766, 0.221185, 0.0917319, 0.0129757, 0.0142857, 0.0260553],
         [0.111121, 0.588392, 0.278779, 0.0055756, 0.00569609, 0.010436],
         [0.0357786, 0.633813, 0.321418, 0.00249248, 0.00272882, 0.0037688],
         [0.0663296, 0.643849, 0.280111, 0.00283995, 0.0035545, 0.00331533],
         [0.458235, 0.396634, 0.123377, 0.00648837, 0.00903441, 0.00623107],
         [0.0357786, 0.633813, 0.321418, 0.00249248, 0.00272882, 0.0037688],
         [0.0357786, 0.633813, 0.321418, 0.00249248, 0.00272882, 0.0037688]],
        dtype=np.float32)
    input_log_prob_matrix_0 = np.log(input_prob_matrix_0)
    gradient_log_prob_0 = np.asarray(
        [[-0.366232,0.221185,0.091732,0.0129757,0.0142857,0.0260553],
         [0.111121,0.588392,0.278779,0.0055756,0.00569609,-0.989564],
         [0.0357786,-0.366187,0.321418,0.00249248,0.00272882,0.0037688],
         [0.0663296,0.643849,0.280111,0.00283995,0.0035545,-0.996685],
         [0.458235,0.396634,-0.876623,0.00648837,0.00903441,0.00623107],
         [0.0357786,0.633813,0.321418,0.00249248,0.00272882,-0.996231],
         [0.0357786,-0.366187,0.321418,0.00249248,0.00272882,0.0037688]],
      
        dtype=np.float32)


    # len max_time_steps array of 2 x depth matrices
    inputs = [np.vstack([input_log_prob_matrix_0[t, :]])
              for t in range(7)]

    # convert inputs into [max_time x batch_size x depth tensor] Tensor
    inputs = np.asarray(inputs, dtype=np.float32)

    # len batch_size array of label vectors
    labels = SimpleSparseTensorFrom([targets_0], blank_index_)

    # batch_size length vector of sequence_lengths
    seq_lens = np.array([7], dtype=np.int32)

    # output: batch_size length vector of negative log probabilities
    loss_truth = np.array([-loss_log_prob_0], np.float32)

    # output: len max_time_steps array of 2 x depth matrices
    grad_truth = [np.vstack([gradient_log_prob_0[t, :]])
                  for t in range(7)]

    # convert grad_truth into [max_time x batch_size x depth] Tensor
    grad_truth = np.asarray(grad_truth, dtype=np.float32)

    self._testCTCLoss(inputs, seq_lens, labels, loss_truth, grad_truth, blank_index=blank_index_)

  def testMultiLabel(self):
    """Test two batch entries."""
    # Input and ground truth from Alex Graves' implementation.
    #
    #### Batch entry 0 #####
    # targets: b 0 b 1 2 b 3 4 b 1 b
    # outputs:
    # 0 0.633766 0.221185 0.0917319 0.0129757 0.0142857 0.0260553
    # 1 0.111121 0.588392 0.278779 0.0055756 0.00569609 0.010436
    # 2 0.0357786 0.633813 0.321418 0.00249248 0.00272882 0.0037688
    # 3 0.0663296 0.643849 0.280111 0.00283995 0.0035545 0.00331533
    # 4 0.458235 0.396634 0.123377 0.00648837 0.00903441 0.00623107
    # 5 0.1      0.1      0.1      0.1        0.1        0.5
    # 6 0.3      0.2      0.2      0.1        0.1        0.1
    # alpha:
    # 0 -inf -0.4560 -inf -inf -inf -inf -inf -inf -inf -inf -inf
    # 1 -inf -inf -5.0180 -inf -inf -inf -inf -inf -inf -inf -inf
    # 2 -inf -inf -inf -5.474 -6.153 -inf -inf -inf -inf -inf -inf
    # 3 -inf -inf -inf -inf -inf -10.7736 -inf -inf -inf -inf -inf
    # 4 -inf -inf -inf -inf -inf -inf -15.8113 -15.4803 -inf -inf -inf
    # 5 -inf -inf -inf -inf -inf -inf -inf -inf -15.6322 -inf -inf
    # 6 -inf -inf -inf -inf -inf -inf -inf -inf -inf -17.2416 -inf
    # prob: -17.2416
    #
    #### Batch entry 1 #####
    #
    # targets: (0 3) (1 2) (2 3)
    #### Batch entry 0 #####
    # targets: b 0 3 b 1 2 b 2 3 b
    # outputs:
    # 0 0.30176, 0.28562, 0.0831517, 0.0862751, 0.0816851, 0.161508
    # 1 0.24082, 0.397533, 0.0557226, 0.0546814, 0.0557528, 0.19549
    # 2 0.230246, 0.450868, 0.0389607, 0.038309, 0.0391602, 0.202456
    # 3 0.280884, 0.429522, 0.0326593, 0.0339046, 0.0326856, 0.190345
    # 4 0.423286, 0.315517, 0.0338439, 0.0393744, 0.0339315, 0.154046
    # 5 0.423286, 0.315517, 0.0338439, 0.0393744, 0.0339315, 0.154046
    # 6 0.423286, 0.315517, 0.0338439, 0.0393744, 0.0339315, 0.154046

    # prob: -7.970009

    # max_time_steps == 7
    depth = 6

    # seq_len_0 == 5
    blank_index_ = 5
    m_targets_0 = [5, 0, 5, 1, 2, 5, 3, 4, 5, 1, 5]
    loss_log_prob_0 = -17.2416
    # dimensions are time x depth
    input_prob_matrix_0 = np.asarray(
        [[0.633766, 0.221185, 0.0917319, 0.0129757, 0.0142857, 0.0260553],
         [0.111121, 0.588392, 0.278779, 0.0055756, 0.00569609, 0.010436],
         [0.0357786, 0.633813, 0.321418, 0.00249248, 0.00272882, 0.0037688],
         [0.0663296, 0.643849, 0.280111, 0.00283995, 0.0035545, 0.00331533],
         [0.458235, 0.396634, 0.123377, 0.00648837, 0.00903441, 0.00623107],
         [0.1, 0.1, 0.1, 0.1, 0.1, 0.5],
         [0.3, 0.2, 0.2, 0.1, 0.1, 0.1]],
        dtype=np.float32)
    input_log_prob_matrix_0 = np.log(input_prob_matrix_0)
    gradient_log_prob_0 = np.asarray(
        [[-0.366234,0.221185,0.091732,0.0129757,0.0142857,0.0260553],
         [0.111121,0.588392,0.278779,0.0055756,0.00569609,-0.989566],
         [0.0357786,-0.0297055,-0.015064,0.00249248,0.00272882,0.0037688],
         [0.0663296,0.643849,0.280111,0.00283995,0.0035545,-0.996685],
         [0.458235,0.396634,0.123377,-0.411502,-0.572976,0.00623107],
         [0.1,0.1,0.1,0.1,0.1,-0.5],
         [0.3,-0.8,0.2,0.1,0.1,0.1]],
        dtype=np.float32)

    # seq_len_1 = 5

    m_targets_1 = [5, 0, 3, 5, 1, 2, 5, 2, 3, 5]

    loss_log_prob_1 = -7.970009
    # dimensions are time x depth

    input_prob_matrix_1 = np.asarray(
        [[0.30176, 0.28562, 0.0831517, 0.0862751, 0.0816851, 0.161508],
         [0.24082, 0.397533, 0.0557226, 0.0546814, 0.0557528, 0.19549],
         [0.230246, 0.450868, 0.0389607, 0.038309, 0.0391602, 0.202456],
         [0.280884, 0.429522, 0.0326593, 0.0339046, 0.0326856, 0.190345],
         [0.423286, 0.315517, 0.0338439, 0.0393744, 0.0339315, 0.154046],
         [0.423286, 0.315517, 0.0338439, 0.0393744, 0.0339315, 0.154046],[0.423286, 0.315517, 0.0338439, 0.0393744, 0.0339315, 0.154046]], dtype=np.float32)

    input_log_prob_matrix_1 = np.log(input_prob_matrix_1)
    gradient_log_prob_1 = np.asarray(
        [[-0.394703,0.28562,0.0831517,-0.0644007,0.0816851,0.00864685],
         [-0.0820998,0.397533,0.0557226,0.0211066,0.0557528,-0.448016],
         [0.158737,0.01862,0.027771,0.0345583,0.0391602,-0.278846],
         [0.280884,-0.263634,0.0150443,0.0339046,0.0326856,-0.0988829],
         [0.423287,-0.163337,-0.00440865,0.0141216,0.0339315,-0.303594],
         [0.423287,0.315517,-0.0857938,-0.104772,0.0339315,-0.582169],
         [0.423287,0.315517,-0.311577,-0.366764,0.0339315,-0.0943942]],
        dtype=np.float32)

    inputs = [np.vstack([input_log_prob_matrix_0[t, :],
                         input_log_prob_matrix_1[t, :]])
              for t in range(7)]
    # convert inputs into [max_time x batch_size x depth tensor] Tensor
    inputs = np.asarray(inputs, dtype=np.float32)

    # len batch_size array of label vectors
    labels = MultiLabelSimpleSparseTensorFrom([m_targets_0, m_targets_1], blank_index_)

    # batch_size length vector of sequence_lengths
    seq_lens = np.array([7, 7], dtype=np.int32)

    # output: batch_size length vector of negative log probabilities
    loss_truth = np.array([-loss_log_prob_0, -loss_log_prob_1], np.float32)

    # output: len max_time_steps array of 2 x depth matrices
    grad_truth = [np.vstack([gradient_log_prob_0[t, :],
                             gradient_log_prob_1[t, :]])
                  for t in range(7)]

    # convert grad_truth into [max_time x batch_size x depth] Tensor
    grad_truth = np.asarray(grad_truth, dtype=np.float32)

    self._testCTCLoss(inputs, seq_lens, labels, loss_truth, grad_truth, blank_index=blank_index_)

if __name__ == "__main__":
  tf.test.main()
