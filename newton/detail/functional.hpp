#pragma once

#include <thrust/functional.h>
#include <cmath>
#include <math.h>

namespace newton
{
namespace detail
{


inline __host__ __device__
float abs(float x)
{
  return ::fabs(x);
}

template<typename T>
inline __host__ __device__
T abs(T x)
{
  return x < 0 ? -x : x;
}

template<typename T>
  struct absolute_value
    : thrust::unary_function<T,T>
{
  inline __host__ __device__
  T operator()(const T &x) const
  {
    return abs(x);
  }
};


template<typename T>
  struct arc_cosine
    : thrust::unary_function<T,T>
{
  inline __host__ __device__
  T operator()(const T &x) const
  {
    return acos(x);
  }
};


template<typename T>
  struct arc_sine
    : thrust::unary_function<T,T>
{
  inline __host__ __device__
  T operator()(const T &x) const
  {
    return asin(x);
  }
};


template<typename T>
  struct arc_tangent
    : thrust::unary_function<T,T>
{
  inline __host__ __device__
  T operator()(const T &x) const
  {
    return atan(x);
  }
};


} // end detail
} // end newton

