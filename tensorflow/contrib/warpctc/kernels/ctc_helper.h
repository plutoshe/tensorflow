#pragma once

#include <limits>
#include <algorithm>
#include <cmath>

#ifdef __CUDACC__
#define HOSTDEVICE __host__ __device__
#else
#define HOSTDEVICE
#endif

// Used for easy access to 3d tensor.
template<typename T>
class Accessor3D {
 public:
  HOSTDEVICE Accessor3D(int pb, int pf, T* pv) : b(pb), f(pf), values(pv) {}

  T HOSTDEVICE operator()(int ti, int bi, int fi) const {
    return values[(ti*b + bi)*f + fi];
  }

  T& HOSTDEVICE operator()(int ti, int bi, int fi) {
    return values[(ti*b + bi)*f + fi];
  }

 private:
  int b, f;
  // not owner.
  T* values;
};

template<typename T>
class Accessor2D {
 public:
  HOSTDEVICE Accessor2D(int pf, T* pv) : f(pf), values(pv) {}

  T HOSTDEVICE operator()(int bi, int fi) const {
    return values[bi*f + fi];
  }

  T& HOSTDEVICE operator()(int bi, int fi) {
    return values[bi*f + fi];
  }

 private:
  int f;
  // not owner.
  T* values;
};


typedef enum {
  CTC_STATUS_SUCCESS = 0,
  CTC_STATUS_MEMOPS_FAILED = 1,
  CTC_STATUS_INVALID_VALUE = 2,
  CTC_STATUS_EXECUTION_FAILED = 3,
  CTC_STATUS_UNKNOWN_ERROR = 4
} ctcStatus_t;

namespace ctc_helper {

static const int BLANK = 0;
static const float threshold = 1e-1;

template<typename T>
HOSTDEVICE
T neg_inf() { return -T(INFINITY); }

inline int div_up(int x, int y) {
  return (x + y - 1) / y;
}

template <typename Arg, typename Res = Arg> struct maximum {
  HOSTDEVICE
  Res operator()(const Arg& x, const Arg& y) const {
    return x < y ? y : x;
  }
};

template <typename Arg, typename Res = Arg> struct add {
  HOSTDEVICE
  Res operator()(const Arg& x, const Arg& y) const {
    return x + y;
  }
};

template <typename Arg, typename Res = Arg> struct identity {
  HOSTDEVICE Res operator()(const Arg& x) const {return Res(x);}
};

template <typename Arg, typename Res = Arg> struct negate {
  HOSTDEVICE Res operator()(const Arg& x) const {return Res(-x);}
};

template <typename Arg, typename Res = Arg> struct exponential {
  HOSTDEVICE Res operator()(const Arg& x) const {return std::exp(x);}
};

template<typename Arg1, typename Arg2 = Arg1, typename Res=Arg1>
struct log_plus {
  typedef Res result_type;
  HOSTDEVICE
  Res operator()(const Arg1& p1, const Arg2& p2) {
    if (p1 == neg_inf<Arg1>())
      return p2;
    if (p2 == neg_inf<Arg2>())
      return p1;
    Res result = log1p(exp(-fabs(p1 - p2))) + maximum<Res>()(p1, p2);
    return result;
  }
};

}
