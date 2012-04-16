#pragma once

#include <newton/detail/range/type_traits.hpp>
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
      detail::arc_tangent<
        typename detail::range_value<NumericRange1>::type
      >
    >
  >::type
    atan2(const NumericRange1 &rng1, const NumericRange2 &rng2)
{
  typedef typename detail::range_value<NumericRange1>::type value_type1;
  return detail::transform(rng1, rng2, detail::arc_tangent2<value_type1>());
} // end atan


template<typename NumericRange, typename T>
inline __host__ __device__
  typename detail::lazy_enable_if_range_and_scalar<
    NumericRange, T,
    detail::transform2_result<
      const NumericRange,
      detail::constant_range<T>,
      detail::arc_tangent<
        typename detail::range_value<NumericRange>::type
      >
    >
  >::type
    atan2(const NumericRange &rng, const T &c)
{
  typedef typename detail::range_value<NumericRange>::type value_type;
  return detail::transform(rng, make_constant_range(c, rng.size()), detail::arc_tangent2<value_type>());
} // end atan

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
    atan2(const T &c, const NumericRange &rng)
{
  typedef typename detail::range_value<T>::type value_type;
  return detail::transform(make_constant_range(c, rng.size()), rng, detail::arc_tangent2<value_type>());
} // end atan


} // end newton

