// Copyright © 2016 Singulariti, Inc.
// All rights reserved.
// Created by Hao Chen (hao.chen@misingularity.io) on 6/5/2016

#include "tensorflow/androidrnn/main/str_util.h"

namespace androidrnn {

// trim from start
std::string* ltrim(std::string* s) {
  s->erase(s->begin(),
           std::find_if(s->begin(),
                        s->end(),
                        std::not1(std::ptr_fun<int, int>(std::isspace))));
  return s;
}

// trim from end
std::string* rtrim(std::string* s) {
  s->erase(std::find_if(s->rbegin(),
                        s->rend(),
                        std::not1(std::ptr_fun<int, int>(std::isspace))).base(),
           s->end());
  return s;
}

// trim from both ends
void strip(std::string* s) {
  ltrim(rtrim(s));
}

// This method used to split text into pieces based on delimiter
std::vector<std::string> split(std::string text, const std::string &delim) {
  std::vector<std::string> result;
  text += delim;
  int token_start = 0;
  std::size_t pos = text.find(delim, token_start);
  while (pos != std::string::npos) {
    if (pos > token_start) {
      result.push_back(text.substr(token_start, pos - token_start));
    }
    token_start = pos + delim.length();
    pos = text.find(delim, token_start);
  }
  return result;
}

std::string itoa(int x) {
  char str[15];
  sprintf(str, "%d", x);
  return std::string(str);
}

}  // namespace androidrnn