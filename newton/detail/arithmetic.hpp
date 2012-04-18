#pragma once

#include <newton/detail/range/range_traits.hpp>
#include <newton/detail/promoting_transform.hpp>

namespace newton
{
namespace detail
{


template<typename RangeOrScalar>
  struct range_value_or_identity
    : thrust::detail::eval_if<
        is_range<RangeOrScalar>::value,
        range_value<RangeOrScalar>,
        thrust::detail::identity_<RangeOrScalar>
      >
{};


} // end detail

namespace ops
{


#define __NEWTON_DEFINE_RANGE_BINARY_OPERATOR(op, functor) \
template<typename RangeOrScalar1, typename RangeOrScalar2> \
inline __host__ __device__ \
  typename detail::lazy_enable_if_at_least_one_is_range< \
    RangeOrScalar1, RangeOrScalar2, \
    detail::promoting_transform_result< \
      const RangeOrScalar1, \
      const RangeOrScalar2, \
      functor<typename detail::range_value_or_identity<RangeOrScalar1>::type> \
    > \
  >::type \
    operator op(const RangeOrScalar1 &lhs, const RangeOrScalar2 &rhs) \
{ \
  return detail::promoting_transform(lhs, rhs, functor<typename detail::range_value_or_identity<RangeOrScalar1>::type>()); \
}


__NEWTON_DEFINE_RANGE_BINARY_OPERATOR(+, thrust::plus);
__NEWTON_DEFINE_RANGE_BINARY_OPERATOR(*, thrust::multiplies);


} // end ops


using namespace ops;

} // end newton

