#pragma once

#include <newton/detail/range/type_traits.hpp>
#include <newton/detail/functional.hpp>

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


} // end newton

