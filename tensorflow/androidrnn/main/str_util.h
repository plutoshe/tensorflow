// Copyright © 2016 Singulariti, Inc.
// All rights reserved.
// Created by Hao Chen (hao.chen@misingularity.io) on 6/5/2016

#ifndef ANDROIDRNN_MAIN_JNI_STR_UTIL_H_
#define ANDROIDRNN_MAIN_JNI_STR_UTIL_H_

#include <algorithm>
#include <string>
#include <vector>

namespace androidrnn {

// trim from start
std::string* ltrim(std::string* s);

// trim from end
std::string* rtrim(std::string* s);

// trim from both ends
void strip(std::string* s);

// split strings
std::vector<std::string> split(std::string text, const std::string &delim);

// convert int to string
std::string itoa(int x);

}  // namespace androidrnn

#endif  // ANDROIDRNN_MAIN_JNI_STR_UTIL_H_