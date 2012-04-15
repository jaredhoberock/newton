#pragma once

#include <newton/detail/type_traits.hpp>
#include <newton/detail/transform_range.hpp>

namespace newton
{
namespace detail
{


template<typename Range, typename AdaptableUnaryFunction>
  class transform1_result
{
  typedef typename range_iterator<Range>::type iterator;

  public:
    typedef transform_range<AdaptableUnaryFunction, iterator> type;
};


template<typename Range, typename AdaptableUnaryFunction>
inline __host__ __device__
typename transform1_result<Range,AdaptableUnaryFunction>::type
transform(Range &rng, AdaptableUnaryFunction f)
{
  return make_transform_range(rng,f);
}


template<typename Range, typename AdaptableUnaryFunction>
inline __host__ __device__
typename transform1_result<const Range,AdaptableUnaryFunction>::type
transform(const Range &rng, AdaptableUnaryFunction f)
{
  return make_transform_range(rng,f);
}


} // end detail
} // end newton

