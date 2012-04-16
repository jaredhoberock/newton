#pragma once

#include <newton/detail/range/range_traits.hpp>
#include <newton/detail/functional.hpp>
#include <newton/detail/range/constant_range.hpp>

namespace newton
{


template<typename NumericRange>
inline __host__ __device__
  typename detail::lazy_enable_if_range<
    NumericRange,
    detail::transform1_result<
      const NumericRange,
      detail::absolute_value<
        typename detail::range_value<NumericRange>::type
      >
    >
  >::type
    abs(const NumericRange &rng)
{
  typedef typename detail::range_value<NumericRange>::type value_type;
  return detail::transform(rng, detail::absolute_value<value_type>());
} // end abs


template<typename NumericRange>
inline __host__ __device__
  typename detail::lazy_enable_if_range<
    NumericRange,
    detail::transform1_result<
      const NumericRange,
      detail::arc_cosine<
        typename detail::range_value<NumericRange>::type
      >
    >
  >::type
    acos(const NumericRange &rng)
{
  typedef typename detail::range_value<NumericRange>::type value_type;
  return detail::transform(rng, detail::arc_cosine<value_type>());
} // end acos


template<typename NumericRange>
inline __host__ __device__
  typename detail::lazy_enable_if_range<
    NumericRange,
    detail::transform1_result<
      const NumericRange,
      detail::arc_sine<
        typename detail::range_value<NumericRange>::type
      >
    >
  >::type
    asin(const NumericRange &rng)
{
  typedef typename detail::range_value<NumericRange>::type value_type;
  return detail::transform(rng, detail::arc_sine<value_type>());
} // end asin


template<typename NumericRange>
inline __host__ __device__
  typename detail::lazy_enable_if_range<
    NumericRange,
    detail::transform1_result<
      const NumericRange,
      detail::arc_tangent<
        typename detail::range_value<NumericRange>::type
      >
    >
  >::type
    atan(const NumericRange &rng)
{
  typedef typename detail::range_value<NumericRange>::type value_type;
  return detail::transform(rng, detail::arc_tangent<value_type>());
} // end atan


template<typename NumericRange1, typename NumericRange2>
inline __host__ __device__
  typename detail::lazy_enable_if_ranges<
    NumericRange1, NumericRange2,
    detail::transform2_result<
      const NumericRange1,
      const NumericRange2,
      detail::arc_tangent2<
        typename detail::range_value<NumericRange1>::type
      >
    >
  >::type
    atan2(const NumericRange1 &rng1, const NumericRange2 &rng2)
{
  typedef typename detail::range_value<NumericRange1>::type value_type1;
  return detail::transform(rng1, rng2, detail::arc_tangent2<value_type1>());
} // end atan2


template<typename NumericRange, typename T>
inline __host__ __device__
  typename detail::lazy_enable_if_range_and_scalar<
    NumericRange, T,
    detail::transform2_result<
      const NumericRange,
      detail::constant_range<T>,
      detail::arc_tangent2<
        typename detail::range_value<NumericRange>::type
      >
    >
  >::type
    atan2(const NumericRange &rng, const T &c)
{
  typedef typename detail::range_value<NumericRange>::type value_type;
  return detail::transform(rng, make_constant_range(c, rng.size()), detail::arc_tangent2<value_type>());
} // end atan2


template<typename T, typename NumericRange>
inline __host__ __device__
  typename detail::lazy_enable_if_scalar_and_range<
    T, NumericRange,
    detail::transform2_result<
      detail::constant_range<T>,
      const NumericRange,
      detail::arc_tangent2<
        typename detail::range_value<T>::type
      >
    >
  >::type
    atan2(const T &c, const NumericRange &rng)
{
  typedef typename detail::range_value<T>::type value_type;
  return detail::transform(make_constant_range(c, rng.size()), rng, detail::arc_tangent2<value_type>());
} // end atan2


template<typename NumericRange>
inline __host__ __device__
  typename detail::lazy_enable_if_range<
    NumericRange,
    detail::transform1_result<
      const NumericRange,
      detail::cosine<
        typename detail::range_value<NumericRange>::type
      >
    >
  >::type
    cos(const NumericRange &rng)
{
  typedef typename detail::range_value<NumericRange>::type value_type;
  return detail::transform(rng, detail::cosine<value_type>());
} // end cos


template<typename NumericRange>
inline __host__ __device__
  typename detail::lazy_enable_if_range<
    NumericRange,
    detail::transform1_result<
      const NumericRange,
      detail::hyperbolic_cosine<
        typename detail::range_value<NumericRange>::type
      >
    >
  >::type
    cosh(const NumericRange &rng)
{
  typedef typename detail::range_value<NumericRange>::type value_type;
  return detail::transform(rng, detail::hyperbolic_cosine<value_type>());
} // end cosh


template<typename NumericRange>
inline __host__ __device__
  typename detail::lazy_enable_if_range<
    NumericRange,
    detail::transform1_result<
      const NumericRange,
      detail::exponential<
        typename detail::range_value<NumericRange>::type
      >
    >
  >::type
    exp(const NumericRange &rng)
{
  typedef typename detail::range_value<NumericRange>::type value_type;
  return detail::transform(rng, detail::exponential<value_type>());
} // end exp


template<typename NumericRange>
inline __host__ __device__
  typename detail::lazy_enable_if_range<
    NumericRange,
    detail::transform1_result<
      const NumericRange,
      detail::logarithm<
        typename detail::range_value<NumericRange>::type
      >
    >
  >::type
    log(const NumericRange &rng)
{
  typedef typename detail::range_value<NumericRange>::type value_type;
  return detail::transform(rng, detail::logarithm<value_type>());
} // end log


template<typename NumericRange>
inline __host__ __device__
  typename detail::lazy_enable_if_range<
    NumericRange,
    detail::transform1_result<
      const NumericRange,
      detail::logarithm10<
        typename detail::range_value<NumericRange>::type
      >
    >
  >::type
    log10(const NumericRange &rng)
{
  typedef typename detail::range_value<NumericRange>::type value_type;
  return detail::transform(rng, detail::logarithm10<value_type>());
} // end log10


template<typename NumericRange1, typename NumericRange2>
inline __host__ __device__
  typename detail::lazy_enable_if_ranges<
    NumericRange1, NumericRange2,
    detail::transform2_result<
      const NumericRange1,
      const NumericRange2,
      detail::power<
        typename detail::range_value<NumericRange1>::type
      >
    >
  >::type
    pow(const NumericRange1 &rng1, const NumericRange2 &rng2)
{
  typedef typename detail::range_value<NumericRange1>::type value_type1;
  return detail::transform(rng1, rng2, detail::power<value_type1>());
} // end pow


template<typename NumericRange, typename T>
inline __host__ __device__
  typename detail::lazy_enable_if_range_and_scalar<
    NumericRange, T,
    detail::transform2_result<
      const NumericRange,
      detail::constant_range<T>,
      detail::power<
        typename detail::range_value<NumericRange>::type
      >
    >
  >::type
    pow(const NumericRange &rng, const T &c)
{
  typedef typename detail::range_value<NumericRange>::type value_type;
  return detail::transform(rng, make_constant_range(c, rng.size()), detail::power<value_type>());
} // end pow


template<typename T, typename NumericRange>
inline __host__ __device__
  typename detail::lazy_enable_if_scalar_and_range<
    T, NumericRange,
    detail::transform2_result<
      detail::constant_range<T>,
      const NumericRange,
      detail::arc_tangent<
        typename detail::range_value<T>::type
      >
    >
  >::type
    pow(const T &c, const NumericRange &rng)
{
  typedef typename detail::range_value<T>::type value_type;
  return detail::transform(make_constant_range(c, rng.size()), rng, detail::power<value_type>());
} // end pow


template<typename NumericRange>
inline __host__ __device__
  typename detail::lazy_enable_if_range<
    NumericRange,
    detail::transform1_result<
      const NumericRange,
      detail::sine<
        typename detail::range_value<NumericRange>::type
      >
    >
  >::type
    sin(const NumericRange &rng)
{
  typedef typename detail::range_value<NumericRange>::type value_type;
  return detail::transform(rng, detail::sine<value_type>());
} // end sin


template<typename NumericRange>
inline __host__ __device__
  typename detail::lazy_enable_if_range<
    NumericRange,
    detail::transform1_result<
      const NumericRange,
      detail::hyperbolic_sine<
        typename detail::range_value<NumericRange>::type
      >
    >
  >::type
    sinh(const NumericRange &rng)
{
  typedef typename detail::range_value<NumericRange>::type value_type;
  return detail::transform(rng, detail::hyperbolic_sine<value_type>());
} // end sinh


template<typename NumericRange>
inline __host__ __device__
  typename detail::lazy_enable_if_range<
    NumericRange,
    detail::transform1_result<
      const NumericRange,
      detail::square_root<
        typename detail::range_value<NumericRange>::type
      >
    >
  >::type
    sqrt(const NumericRange &rng)
{
  typedef typename detail::range_value<NumericRange>::type value_type;
  return detail::transform(rng, detail::square_root<value_type>());
} // end sqrt


template<typename NumericRange>
inline __host__ __device__
  typename detail::lazy_enable_if_range<
    NumericRange,
    detail::transform1_result<
      const NumericRange,
      detail::tangent<
        typename detail::range_value<NumericRange>::type
      >
    >
  >::type
    tan(const NumericRange &rng)
{
  typedef typename detail::range_value<NumericRange>::type value_type;
  return detail::transform(rng, detail::tangent<value_type>());
} // end tan


template<typename NumericRange>
inline __host__ __device__
  typename detail::lazy_enable_if_range<
    NumericRange,
    detail::transform1_result<
      const NumericRange,
      detail::hyperbolic_tangent<
        typename detail::range_value<NumericRange>::type
      >
    >
  >::type
    tanh(const NumericRange &rng)
{
  typedef typename detail::range_value<NumericRange>::type value_type;
  return detail::transform(rng, detail::hyperbolic_tangent<value_type>());
} // end tanh


} // end newton

