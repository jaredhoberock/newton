#pragma once

#include <newton/detail/type_traits.hpp>
#include <newton/detail/zip_with.hpp>
#include <thrust/functional.h>

namespace newton
{
namespace detail
{

template<typename Range1, typename Range2>
  class sum_ranges_result
{
  typedef typename range_value<Range1>::type value_type1;

  public:
    typedef typename zip_with2_result<const Range1, const Range2, thrust::plus<value_type1> >::type type;
};

template<typename Range1, typename Range2>
  inline __host__ __device__
    typename sum_ranges_result<Range1,Range2>::type
      sum_ranges(const Range1 &lhs, const Range2 &rhs)
{
  typedef typename range_value<Range1>::type value_type1;

  return zip_with(lhs, rhs, thrust::plus<value_type1>());
}

} // end detail
} // end newton

