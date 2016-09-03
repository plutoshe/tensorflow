// Copyright © 2016 Singulariti, Inc.
// All rights reserved.
// Created by Hao Chen (hao.chen@misingularity.io) on 6/5/2016

#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>
#include <queue>

#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <jni.h>
#include <pthread.h>
#include <sys/stat.h>

#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/stat_summarizer.h"
#include "tensorflow/androidrnn/jni/jni_utils.h"
#include "tensorflow/androidrnn/main/str_util.h"
#include "tensorflow/androidrnn/main/dict.h"

using namespace tensorflow;

const int kNumResults = 5;
const float kThreshold = 5 / 10000.0f;
const int kLimitLen = 30;

// Global variables that holds the Tensorflow classifier.
static std::unique_ptr<tensorflow::Session> session;
static std::unique_ptr<StatSummarizer> g_stats;

static bool g_compute_graph_initialized_ = false;
static androidrnn::WordsDictionary* ch_dict_ = nullptr;
static androidrnn::WordsDictionary* py_dict_ = nullptr;

inline static int64 CurrentThreadTimeUs() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec * 1000000 + tv.tv_usec;
}

static std::vector<int> inference(const std::vector<int>& pinyin_str) {
  std::vector<int> ret;


  tensorflow::Status s;
  std::vector<tensorflow::Tensor> output_tensors;
  std::vector<std::string> output_tensor_names;
  std::vector<std::pair<std::string, tensorflow::Tensor> > inputs;

  // initial state
  output_tensors.clear();
  output_tensor_names.clear();
  inputs.clear();


  // feed data into queue
  // {"model/input_data:0" : [[[0] * T], [indexs]],
  //         "model/targets:0" : [indexs],
  //         "model/weights:0" : np.ones((1, len(indexs)))})
  LOG(INFO) << "phase1";
  const int T = 25;
  tensorflow::Tensor input_tensor(
        tensorflow::DT_INT32,
        tensorflow::TensorShape({2, 1, 25}));
  // tensorflow::Tensor input_tensor_1(
  //       tensorflow::DT_FLOAT,
  //       tensorflow::TensorShape({1, 25}));
  // tensorflow::Tensor input_tensor_2(
  //       tensorflow::DT_FLOAT,
  //       tensorflow::TensorShape({1, 25}));
  LOG(INFO) << "phase11";
  auto input_tensor_data = input_tensor.tensor<int, 3>();
  // auto input_tensor_targets = input_tensor_1.tensor<int, 2>();
  // auto input_tensor_weights = input_tensor_2.tensor<float, 2>();
  LOG(INFO) << "phase2";
  for (int i = 0; i < T; i++) {
    input_tensor_data(0,0,i) = 0;
    input_tensor_data(1,0,i) = pinyin_str[i];
    // input_tensor_targets(0,i) = pinyin_str[i];
    // input_tensor_weights(0,i) = 1;
  }
  LOG(INFO) << "phase3";
  inputs.push_back(std::make_pair("model/input_data:0", input_tensor));
  // inputs.push_back(std::make_pair("model/targets:0", input_tensor_1));
  // inputs.push_back(std::make_pair("model/weights:0", input_tensor_2));
  LOG(INFO) << "phase4";
  // output_tensor_names.push_back("model/input_queue");
  // s = session->Run(inputs, output_tensor_names, {}, &output_tensors);
  // LOG(INFO) << "phase5";
  // // get outputs
  // output_tensors.clear();
  // output_tensor_names.clear();
  // inputs.clear();
  // LOG(INFO) << "phase6";
  output_tensor_names.push_back("model/word_outputs:0");
  s = session->Run(inputs, output_tensor_names, {}, &output_tensors);
  LOG(INFO) << s;
  tensorflow::Tensor* ans = &output_tensors[0];
  LOG(INFO) << "phase7";
  LOG(INFO) << ans;
  // const Eigen::TensorMap<Eigen::Tensor<int32, 1>,
  //                          Eigen::Aligned> ch = ans->flat<int>(T);
  auto ch = ans->flat<int64>();
  LOG(INFO) << "phase71";
  for (int i = 0; i < T; i++) {
    ret.push_back(ch(i));
  }
  LOG(INFO) << "phase8";
  return ret;
}

extern "C" {

JNIEXPORT void JNICALL
Java_com_singulariti_androidrnn_TensorflowRNNJni_initFromJNI(
    JNIEnv *env, jobject,
    jobject java_asset_manager, jstring model, jstring chinese_labels, jstring pinyin_labels) {
  if (g_compute_graph_initialized_) {
    LOG(INFO) << "Compute graph already loaded. skipping.";
    return;
  }

  const int64 start_time = CurrentThreadTimeUs();

  const char* const model_cstr = env->GetStringUTFChars(model, NULL);
  const char* const ch_labels_cstr = env->GetStringUTFChars(chinese_labels, NULL);
  const char* const py_labels_cstr = env->GetStringUTFChars(pinyin_labels, NULL);

  LOG(INFO) << "Loading Tensorflow.";

  LOG(INFO) << "Making new SessionOptions.";
  tensorflow::SessionOptions options;
  tensorflow::ConfigProto& config = options.config;
  LOG(INFO) << "Got config, " << config.device_count_size() << " devices";

  session.reset(tensorflow::NewSession(options));
  LOG(INFO) << "Session created.";

  tensorflow::GraphDef tensorflow_graph;
  LOG(INFO) << "Graph created.";

  AAssetManager* const asset_manager =
    AAssetManager_fromJava(env, java_asset_manager);

  LOG(INFO) << "Reading file to proto: " << model_cstr;
  ReadFileToProto(asset_manager, model_cstr, &tensorflow_graph);

  g_stats.reset(new StatSummarizer(tensorflow_graph));

  LOG(INFO) << "Creating session.";
  tensorflow::Status s = session->Create(tensorflow_graph);
  if (!s.ok()) {
    LOG(ERROR) << "Could not create Tensorflow Graph: " << s;
    return;
  }

  // Clear the proto to save memory space.
  tensorflow_graph.Clear();
  LOG(INFO) << "Tensorflow graph loaded from: " << model_cstr;

  // Read the label list
  std::vector<std::string> g_label_strings;
  ReadFileToVector(asset_manager, ch_labels_cstr, &g_label_strings);
  LOG(INFO) << g_label_strings.size() << " label strings loaded from: "
            << ch_labels_cstr;
  ch_dict_ = new androidrnn::WordsDictionary(g_label_strings);

  ReadFileToVector(asset_manager, py_labels_cstr, &g_label_strings);
  LOG(INFO) << g_label_strings.size() << " label strings loaded from: "
            << py_labels_cstr;
  py_dict_ = new androidrnn::WordsDictionary(g_label_strings);

  g_compute_graph_initialized_ = true;

  const int64 end_time = CurrentThreadTimeUs();
  LOG(INFO) << "Initialization done in " << (end_time - start_time) / 1000
            << "ms";
}

JNIEXPORT jstring JNICALL
Java_com_singulariti_androidrnn_TensorflowRNNJni_decode(
    JNIEnv *env, jobject, jstring jtext) {
  const char *text = env->GetStringUTFChars(jtext, NULL);
  std::string res = std::string(text);
  androidrnn::strip(&res);
  LOG(INFO) << "Decode \'" << res << "\'";
  if (res == "" || !g_compute_graph_initialized_) {
    return env->NewStringUTF("");
  } else {
    std::vector<std::string> words = androidrnn::split(res, " ");
    std::vector<int> pinyin_str;
    int T = 25;
    for (const std::string& w : words) {
      pinyin_str.push_back(py_dict_->getIdByWord(w));
      if (pinyin_str.size() > T) {
        break;
      }
    }
    if (pinyin_str.size() < T) {
        int remains = T - pinyin_str.size();
        for (int i = 0; i < remains; i++) {
          pinyin_str.push_back(py_dict_->getIdByWord("<null>"));
        }
    }
    LOG(INFO) << "changdu" << pinyin_str.size();
    std::vector<int> ids = inference(pinyin_str);
    LOG(INFO) << "phase9";
    std::string ret = "";
    for (int id : ids) {
      ret += ch_dict_->getWordById(id) + " ";
    }
    LOG(INFO) << "phase10" << ret.size();
    LOG(INFO) << ret;
    LOG(INFO) << ret.c_str();
    LOG(INFO) << env->NewStringUTF(ret.c_str());
    LOG(INFO) << "finished";
    return env->NewStringUTF(ret.c_str());

  }
}

}