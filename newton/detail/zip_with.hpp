#pragma once

#include <newton/detail/zip.hpp>
#include <newton/detail/transform.hpp>
#include <thrust/tuple.h>

namespace newton
{
namespace detail
{

template<typename AdaptableBinaryFunction>
  struct unpack_and_apply2
{
  typedef typename AdaptableBinaryFunction::result_type result_type;

  AdaptableBinaryFunction m_f;

  inline __host__ __device__
  unpack_and_apply2(AdaptableBinaryFunction f)
    : m_f(f)
  {}

  template<typename Tuple>
  inline __host__ __device__
  result_type operator()(const Tuple &t) const
  {
    return m_f(thrust::get<0>(t), thrust::get<1>(t));
  }
}; // end unpack_and_apply2


template<typename Range1, typename Range2, typename AdaptableBinaryFunction>
  class zip_with2_result
{
  typedef typename zip2_result<Range1,Range2>::type  zipped_range_type;
  typedef unpack_and_apply2<AdaptableBinaryFunction> functor_type;
  
  public:
    typedef typename transform1_result<zipped_range_type, functor_type>::type type;
}; // end zip_with2_result


template<typename Range1, typename Range2, typename AdaptableBinaryFunction>
inline __host__ __device__
  typename zip_with2_result<Range1,Range2,AdaptableBinaryFunction>::type
    zip_with(Range1 &rng1, Range2 &rng2, AdaptableBinaryFunction f)
{
  return transform(zip(rng1, rng2), unpack_and_apply2<AdaptableBinaryFunction>(f));
} // end zip_with()

template<typename Range1, typename Range2, typename AdaptableBinaryFunction>
inline __host__ __device__
  typename zip_with2_result<Range1,const Range2,AdaptableBinaryFunction>::type
    zip_with(Range1 &rng1, const Range2 &rng2, AdaptableBinaryFunction f)
{
  return transform(zip(rng1, rng2), unpack_and_apply2<AdaptableBinaryFunction>(f));
} // end zip_with()


template<typename Range1, typename Range2, typename AdaptableBinaryFunction>
inline __host__ __device__
  typename zip_with2_result<const Range1,Range2,AdaptableBinaryFunction>::type
    zip_with(const Range1 &rng1, Range2 &rng2, AdaptableBinaryFunction f)
{
  return transform(zip(rng1, rng2), unpack_and_apply2<AdaptableBinaryFunction>(f));
} // end zip_with()


template<typename Range1, typename Range2, typename AdaptableBinaryFunction>
inline __host__ __device__
  typename zip_with2_result<const Range1,const Range2,AdaptableBinaryFunction>::type
    zip_with(const Range1 &rng1, const Range2 &rng2, AdaptableBinaryFunction f)
{
  return transform(zip(rng1, rng2), unpack_and_apply2<AdaptableBinaryFunction>(f));
} // end zip_with()


} // end detail
} // end newton

