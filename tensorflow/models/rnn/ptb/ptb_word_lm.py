# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Example / benchmark for building a PTB LSTM model.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

To run:

$ python ptb_word_lm.py --data_path=simple-examples/data/

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf
import os

from tensorflow.models.rnn.ptb import reader

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None, "data_path")

FLAGS = flags.FLAGS

def _build_vocab_chinese(pinyinfile, markingfile):
  # read raw input data from filename
  # only contain the basic word
  # with tf.gfile.GFile(pinyinfile, "r") as f:
  #   data = f.read().replace("\n", "<eos>").split()

  with open(pinyinfile, "r") as f:
    data = f.read().replace("\n", " ").split()

  pinyin_to_id = dict(zip(data, range(len(data))))
  chinese_to_pinyin = dict()
  with open(markingfile, "r") as f:
    for line in f:
        line = unicode(line.replace("\n", ""), 'utf-8')
        tokens = line.split(" ")
        for i in range(0, int(len(tokens) / 3)):
          chinese_to_pinyin[tokens[3 * i]] = tokens[3 * i + 2]
  # print(chinese_to_pinyin)
  chinese_to_id = dict(zip([key for key, _ in chinese_to_pinyin.items()], range(len(chinese_to_pinyin))))
  chinese_id_to_pinyin_id = dict((chinese_to_id[k], pinyin_to_id[v]) for k, v in chinese_to_pinyin.items())
  return pinyin_to_id, chinese_to_id, chinese_id_to_pinyin_id, len(pinyin_to_id), len(chinese_to_id)

pinyin_to_id, chinese_to_id, chinese_id_to_pinyin_id, piyin_size, chinese_size = _build_vocab_chinese('/home/plutoshe/work/tensorflow/tensorflow/models/rnn/ptb/pinyin.dat', '/home/plutoshe/work/tensorflow/tensorflow/models/rnn/ptb/marking.dat')

def _read_words1(filename):
  with tf.gfile.GFile(filename, "r") as f:
    return f.read().replace("\n", " ").split()

def _file_to_chinese_id(filename, word_to_id):
  data = _read_words1(filename)
  ids = []
  for sent in data:
    tokens = unicode(sent,'utf-8')
    for i in range(len(tokens)):
      word = tokens[i]
      it = word_to_id.get(word)
      if it is None:
        it = word_to_id.get('unk')
      ids.append(it)
  return ids
  # return [word_to_id[word] for word in data]




def ptc_raw_data(data_path=None):
  train_path = os.path.join(data_path, "ptc.train.txt")
  valid_path = os.path.join(data_path, "ptc.valid.txt")
  test_path = os.path.join(data_path, "ptc.test.txt")


  # transfer data words to id
  train_data = _file_to_chinese_id(train_path, chinese_to_id)
  valid_data = _file_to_chinese_id(valid_path, chinese_to_id)
  test_data = _file_to_chinese_id(test_path, chinese_to_id)

  return train_data, valid_data, test_data, 0


def ptc_iterator(raw_data, batch_size, num_steps):
  """Iterate on the raw PTB data.

  This generates batch_size pointers into the raw PTB data, and allows
  minibatch iteration along these pointers.

  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.

  Yields:
    Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
    The second element of the tuple is the same data time-shifted to the
    right by one.

  Raises:
    ValueError: if batch_size or num_steps are too high.
  """
  raw_data = np.array(raw_data, dtype=np.int32)
  data_len = len(raw_data)
  batch_len = data_len // batch_size
  data = np.zeros([batch_size, batch_len], dtype=np.int32)
  for i in range(batch_size):
    data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

  epoch_size = (batch_len - 1) // num_steps
  if epoch_size == 0:
    raise ValueError("epoch_size == 0, decrease batch_size or num_steps")
  for i in range(epoch_size):
    x = []
    for row in data:
      x.append([chinese_id_to_pinyin_id[j] for j in row[i*num_steps:(i+1)*num_steps]])
    y = data[:, i*num_steps+1:(i+1)*num_steps+1]
    yield (x, y)

def pinyin_convert():
  return


class PTBModel(object):
  """The PTB model."""

  def __init__(self, is_training, config):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    size = config.hidden_size
    # print("size:",size)

    # TODO:
    # add chinese/pinyin character vocabular size
    # for targe and input
    # piyin_size
    # chinese_size

    self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
    print(self._input_data.get_shape())
    self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])
    print(self._targets.get_shape())
    # raw_input("")
    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0)
    if is_training and config.keep_prob < 1:
      lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
          lstm_cell, output_keep_prob=config.keep_prob)
    # print(config.num_layers)
    # print(lstm_cell)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)
    # print(cell)


    self._initial_state = cell.zero_state(batch_size, tf.float32)
    with tf.device("/cpu:0"):
      embedding = tf.get_variable("embedding", [piyin_size, size])
      print("embeding:", embedding.get_shape())
      print("_input_data:", self._input_data.get_shape())
      inputs = tf.nn.embedding_lookup(embedding, self._input_data)
      # print("inputs shape:", inputs.get_shape())
      # print("------self._input_data: ", self._input_data)
      # print("------embedding: ", embedding)
      # print("------inputs: ", inputs)

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)
    # print("-----inputs:" ,inputs)

    # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use the rnn() or state_saving_rnn() from rnn.py.
    #
    # The alternative version of the code below is:
    #
    # from tensorflow.models.rnn import rnn
    # inputs = [tf.squeeze(input_, [1])
    #           for input_ in tf.split(1, num_steps, inputs)]
    # outputs, state = rnn.rnn(cell, inputs, initial_state=self._initial_state)

    ## stimulate unrolled rnn operation
    outputs = []
    state = self._initial_state
    # print('state:', state)
    with tf.variable_scope("RNN"):
      # for every step
      for time_step in range(num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        # print('cell_output:', cell_output)
        # print('state:', state)
        outputs.append(cell_output)
        # print('outputs:', outputs)
        # raw_input("")
    # raw_input("")
    # print("----outputs: ", outputs)
    # print("--concat", tf.concat(1, outputs))
    output = tf.reshape(tf.concat(1, outputs), [-1, size])
    # print("----outputs: ", outputs)
    # print("----output: ", output)
    # from embedding feature to activate wanted word
    softmax_w = tf.get_variable("softmax_w", [size, chinese_size])
    softmax_b = tf.get_variable("softmax_b", [chinese_size])
    logits = tf.matmul(output, softmax_w) + softmax_b
    loss = tf.nn.seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(self._targets, [-1])],
        [tf.ones([batch_size * num_steps])])
    self._cost = cost = tf.reduce_sum(loss) / batch_size
    self._final_state = state
    # print("loss:", loss)
    # raw_input("")
    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self.lr)
    self._train_op = optimizer.apply_gradients(zip(grads, tvars))

  def assign_lr(self, session, lr_value):
    session.run(tf.assign(self.lr, lr_value))

  @property
  def input_data(self):
    return self._input_data

  @property
  def targets(self):
    return self._targets

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op


class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 300
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000

class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 6500
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
  vocab_size = 10000


class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 15000
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 20
  vocab_size = 10000


class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 2
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000


def run_epoch(session, m, data, eval_op, verbose=False):
  """Runs the model on the given data."""
  epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = m.initial_state.eval()
  for step, (x, y) in enumerate(ptc_iterator(data, m.batch_size,
                                                    m.num_steps)):
    print("step:", step)
    print("final state:", m.final_state)
    cost, state, _ = session.run([m.cost, m.final_state, eval_op],
                                 {m.input_data: x,
                                  m.targets: y,
                                  m.initial_state: state})
    costs += cost

    iters += m.num_steps
    if verbose and step % (epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / epoch_size, np.exp(costs / iters),
             iters * m.batch_size / (time.time() - start_time)))

  return np.exp(costs / iters)


def get_config():
  if FLAGS.model == "small":
    return SmallConfig()
  elif FLAGS.model == "medium":
    return MediumConfig()
  elif FLAGS.model == "large":
    return LargeConfig()
  elif FLAGS.model == "test":
    return TestConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)


def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")

  raw_data = ptc_raw_data(FLAGS.data_path)
  train_data, valid_data, test_data, _ = raw_data

  config = get_config()
  eval_config = get_config()
  eval_config.batch_size = 1
  eval_config.num_steps = 1
  # tf.Graph() defines current default graph
  # following ops is consisted in this graph
  # example:
  # g = tf.Graph()
  # with g.as_default():
  #   # Define operations and tensors in `g`.
  #   c = tf.constant(30.0)
  #   assert c.graph is g
  with tf.Graph().as_default(), tf.Session() as session:
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
    with tf.variable_scope("model", reuse=None, initializer=initializer):
      m = PTBModel(is_training=True, config=config)
    with tf.variable_scope("model", reuse=True, initializer=initializer):
      mvalid = PTBModel(is_training=False, config=config)
      mtest = PTBModel(is_training=False, config=eval_config)

    tf.initialize_all_variables().run()

    for i in range(config.max_max_epoch):
      lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
      print(lr_decay)
      # raw_input("")
      m.assign_lr(session, config.learning_rate * lr_decay)

      print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
      train_perplexity = run_epoch(session, m, train_data, m.train_op,
                                   verbose=True)
      print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
      valid_perplexity = run_epoch(session, mvalid, valid_data, tf.no_op())
      print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

    test_perplexity = run_epoch(session, mtest, test_data, tf.no_op())
    print("Test Perplexity: %.3f" % test_perplexity)


if __name__ == "__main__":
  tf.app.run()
