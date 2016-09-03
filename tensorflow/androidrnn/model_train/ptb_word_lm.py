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

import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.client import graph_util

import reader

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None, "data_path")
flags.DEFINE_string("mode", "test", "train or test")
flags.DEFINE_string("graph_path", "./graph_train.pb", "graph_path")

FLAGS = flags.FLAGS


class TrainedPTBModel(object):
  def __init__(self, initial_state, cost, input_data, targets, final_state, prob, config):
    self._initial_state = initial_state
    self._cost = cost
    self._input_data = input_data
    self._targets = targets
    self._final_state = final_state
    self._prob = prob
    self.batch_size = config.batch_size
    self.num_steps = config.num_steps

  @property
  def input_data(self):
    return self._input_data

  @property
  def cost(self):
    return self._cost

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def targets(self):
    return self._targets

  @property
  def final_state(self):
    return self._final_state

  @property
  def prob(self):
    return self._prob


class PTBModel(object):
  """The PTB model."""

  def __init__(self, is_training, config, name_prefix):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size

    self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps], name_prefix+'input_data')
    self._targets = tf.placeholder(tf.int32, [batch_size, num_steps], name_prefix+'target')
    print(self.input_data)
    print(self.targets)
    
    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0)
    if is_training and config.keep_prob < 1:
      lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
          lstm_cell, output_keep_prob=config.keep_prob)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)

    self._initial_state = cell.zero_state(batch_size, tf.float32)
    print(self.initial_state)

    with tf.device("/cpu:0"):
      embedding = tf.get_variable("embedding", [vocab_size, size])
      inputs = tf.nn.embedding_lookup(embedding, self._input_data)

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

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
    outputs = []
    state = self._initial_state
    with tf.variable_scope("RNN"):
      for time_step in range(num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)

    output = tf.reshape(tf.concat(1, outputs), [-1, size])

    softmax_w = tf.get_variable("softmax_w", [size, vocab_size])
    softmax_b = tf.get_variable("softmax_b", [vocab_size])
    logits = tf.matmul(output, softmax_w) + softmax_b
    self._proba = tf.nn.softmax(logits)
    print(self.proba)

    loss = tf.nn.seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(self._targets, [-1])],
        [tf.ones([batch_size * num_steps])])
    self._cost = cost = tf.reduce_sum(loss) / batch_size
    self._final_state = state

    print(self.cost)
    print(self.final_state)

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

  @property
  def proba(self):
    return self._proba


class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 200
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
  hidden_size = 650
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
  hidden_size = 1500
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
  for step, (x, y) in enumerate(reader.ptb_iterator(data, m.batch_size,
                                                    m.num_steps)):
    if eval_op.name == 'NoOp':
      cost, state = session.run([m.cost, m.final_state],
                                   {m.input_data: x,
                                    m.targets: y,
                                    m.initial_state: state})
    else:
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


################################################################################
##                                  train                                     ##
################################################################################

def train_job():
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")

  raw_data = reader.ptb_raw_data(FLAGS.data_path)
  train_data, valid_data, test_data, vocabulary = raw_data

  reader.dump_vocabulary(vocabulary, 'dict.txt')

  config = get_config()
  eval_config = get_config()
  eval_config.batch_size = 1
  eval_config.num_steps = 1

  with tf.Graph().as_default(), tf.Session() as session:
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
    with tf.variable_scope("model", reuse=None, initializer=initializer):
      m = PTBModel(is_training=True, config=config, name_prefix='train_')
    with tf.variable_scope("model", reuse=True, initializer=initializer):
      mvalid = PTBModel(is_training=False, config=config, name_prefix='valid_')
      mtest = PTBModel(is_training=False, config=eval_config, name_prefix='test_')

    tf.initialize_all_variables().run()

    tf.train.write_graph(session.graph_def, './', 'graph_raw.pb', as_text=False)

    saver = tf.train.Saver()

    for i in range(config.max_max_epoch):
      lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
      m.assign_lr(session, config.learning_rate * lr_decay)

      print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
      train_perplexity = run_epoch(session, m, train_data, m.train_op,
                                   verbose=True)
      print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
      valid_perplexity = run_epoch(session, mvalid, valid_data, tf.no_op(),
                                   verbose=False)
      print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
      test_perplexity = run_epoch(session, mtest, test_data, tf.no_op(),
                                  verbose=False)
      print("Test Perplexity: %.3f" % test_perplexity)

      constant_graph = graph_util.convert_variables_to_constants(
        session, session.graph_def,
        ['model_1/valid_target', 'model_1/truediv', 'model_1/RNN/concat_19', 'model_1/Softmax',
         'model_1/test_target', 'model_1/truediv_1', 'model_1/RNN_1/concat', 'model_1/Softmax_1'])
      tf.train.write_graph(constant_graph, './', 'graph_trained' + str(i) + '.pb', as_text=False)
      # tf.train.write_graph(constant_graph, './', 'graph_trained' + str(i) + '.pb.text', as_text=True)

      save_path = saver.save(session, 'model' + str(i) + '.ckpt')
      print("Model saved in file: %s" % save_path)


################################################################################
##                                  test                                      ##
################################################################################

def test_job():
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")

  raw_data = reader.ptb_raw_data(FLAGS.data_path)
  train_data, valid_data, test_data, _ = raw_data

  config = get_config()
  eval_config = get_config()
  eval_config.batch_size = 1
  eval_config.num_steps = 1

  with tf.Graph().as_default(), tf.Session() as session:
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
    with tf.variable_scope("model", reuse=None, initializer=initializer):
      m = PTBModel(is_training=True, config=config, name_prefix='train_')
    with tf.variable_scope("model", reuse=True, initializer=initializer):
      mvalid = PTBModel(is_training=False, config=config, name_prefix='valid_')
      mtest = PTBModel(is_training=False, config=eval_config, name_prefix='test_')

    saver = tf.train.Saver()
    saver.restore(session, 'model0.ckpt')
    print('Restored!')

    valid_perplexity = run_epoch(session, mvalid, valid_data, tf.no_op(),
                                 verbose=True)
    print("Valid Perplexity: %.3f" % valid_perplexity)

    test_perplexity = run_epoch(session, mtest, test_data, tf.no_op(),
                                verbose=True)
    print("Test Perplexity: %.3f" % test_perplexity)


def test_perplexity_black_magic():
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")

  raw_data = reader.ptb_raw_data(FLAGS.data_path)
  train_data, valid_data, test_data, _ = raw_data

  config = get_config()
  eval_config = get_config()
  eval_config.batch_size = 1
  eval_config.num_steps = 1

  with tf.Session() as session:
    graph_def = tf.GraphDef()
    with open(FLAGS.graph_path, 'rb') as fin:
        proto_b = fin.read()
        graph_def.ParseFromString(proto_b)
        session.graph.as_default()
        tf.import_graph_def(graph_def, name="")
    mvalid = TrainedPTBModel(initial_state= session.graph.get_tensor_by_name('model_1/zeros:0'),
                             cost=session.graph.get_tensor_by_name('model_1/truediv:0'),
                             input_data='model_1/valid_input_data:0',
                             targets='model_1/valid_target:0',
                             final_state=session.graph.get_tensor_by_name('model_1/RNN/concat_19:0'),
                             config=config)
    valid_perplexity = run_epoch(session, mvalid, valid_data, tf.no_op(),
                                 verbose=True)
    print("Valid Perplexity: %.3f" % valid_perplexity)
    mtest = TrainedPTBModel(initial_state= session.graph.get_tensor_by_name('model_1/zeros_1:0'),
                            cost=session.graph.get_tensor_by_name('model_1/truediv_1:0'),
                            input_data='model_1/test_input_data:0',
                            targets='model_1/test_target:0',
                            final_state=session.graph.get_tensor_by_name('model_1/RNN_1/concat:0'),
                            config=eval_config)
    test_perplexity = run_epoch(session, mtest, test_data, tf.no_op(),
                                verbose=True)
    print("Test Perplexity: %.3f" % test_perplexity)


################################################################################
##                                predict                                     ##
################################################################################

def predict(session, m, state, input_word_id):
  prob, new_state = session.run([m.prob, m.final_state],
                                {m.input_data: [[input_word_id]],
                                 m.initial_state: state})
  return prob, new_state


def predict_black_magic():
  UNK_WORD = '<unk>'
  EOF_WORD = '<eos>'

  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")

  vocabulary = reader.load_vocabulary('dict.txt')

  config = get_config()
  eval_config = get_config()
  eval_config.batch_size = 1
  eval_config.num_steps = 1

  id_to_word = {}
  for key, value in vocabulary.items():
    id_to_word[value] = key

  with tf.Session() as session:
    graph_def = tf.GraphDef()
    with open(FLAGS.graph_path, 'rb') as fin:
        proto_b = fin.read()
        graph_def.ParseFromString(proto_b)
        session.graph.as_default()
        tf.import_graph_def(graph_def, name="")
    m = TrainedPTBModel(initial_state= session.graph.get_tensor_by_name('model_1/zeros_1:0'),
                          cost=session.graph.get_tensor_by_name('model_1/truediv_1:0'),
                          input_data='model_1/test_input_data:0',
                          targets='model_1/test_target:0',
                          final_state=session.graph.get_tensor_by_name('model_1/RNN_1/concat:0'),
                          prob=session.graph.get_tensor_by_name('model_1/Softmax_1:0'),
                          config=eval_config)
    while True:
      st = input('>').strip()
      word_id_list = [vocabulary[x] if x in vocabulary else vocabulary[UNK_WORD] for x in st.split(' ')]

      answer = st
      state = m.initial_state.eval()
      for x in word_id_list:
        prob, state = predict(session, m, state, x)
      while True:
        max_id_list = sorted(range(prob.shape[1]), key=lambda k: prob[0][k], reverse=True)
        next_word_id = max_id_list[0]
        for i in range(len(vocabulary)):
          if max_id_list[i] != vocabulary[UNK_WORD]:
            next_word_id = max_id_list[i]
            break
        answer += ' ' + id_to_word[next_word_id]
        if next_word_id == vocabulary[EOF_WORD]:
          break
        prob, state = predict(session, m, state, next_word_id)
      print(answer)


if __name__ == "__main__":
  if FLAGS.mode == "train":
    train_job()
  elif FLAGS.mode == "test":
    # test_job()
    test_perplexity_black_magic()
  elif FLAGS.mode == "predict":
    predict_black_magic()
