#pragma once

#include <newton/detail/range/range_traits.hpp>
#include <newton/detail/scalar_math.hpp>
#include <newton/detail/transform.hpp>
#include <newton/detail/promoting_transform.hpp>
#include <thrust/functional.h>

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


// introduce this version of modulus which calls mod instead of operator%
// mod is itself implemented with operator% for integral types
template<typename T>
  struct modulus
    : thrust::binary_function<T,T,T>
{
  inline __host__ __device__
  T operator()(const T &lhs, const T &rhs) const
  {
    return mod(lhs,rhs);
  }
};


} // end detail


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


#define __NEWTON_DEFINE_RANGE_UNARY_OPERATOR(op, functor) \
template<typename Range> \
inline __host__ __device__ \
  typename detail::lazy_enable_if_range< \
    Range, \
    detail::transform1_result< \
      const Range, \
      functor<typename detail::range_value<Range>::type> \
    > \
  >::type \
    operator op(const Range &rng) \
{ \
  return detail::transform(rng, functor<typename detail::range_value<Range>::type>()); \
}

// unary arithmetic
__NEWTON_DEFINE_RANGE_UNARY_OPERATOR(+, thrust::identity);
__NEWTON_DEFINE_RANGE_UNARY_OPERATOR(-, thrust::negate);

// binary arithmetic
__NEWTON_DEFINE_RANGE_BINARY_OPERATOR(/, thrust::divides);
__NEWTON_DEFINE_RANGE_BINARY_OPERATOR(-, thrust::minus);
__NEWTON_DEFINE_RANGE_BINARY_OPERATOR(%, detail::modulus);
__NEWTON_DEFINE_RANGE_BINARY_OPERATOR(*, thrust::multiplies);
__NEWTON_DEFINE_RANGE_BINARY_OPERATOR(+, thrust::plus);

// binary relational
__NEWTON_DEFINE_RANGE_BINARY_OPERATOR(==, thrust::equal_to);
__NEWTON_DEFINE_RANGE_BINARY_OPERATOR(>,  thrust::greater);
__NEWTON_DEFINE_RANGE_BINARY_OPERATOR(>=, thrust::greater_equal);
__NEWTON_DEFINE_RANGE_BINARY_OPERATOR(<,  thrust::less);
__NEWTON_DEFINE_RANGE_BINARY_OPERATOR(<=, thrust::less_equal);
__NEWTON_DEFINE_RANGE_BINARY_OPERATOR(!=, thrust::not_equal_to);

// unary logical
__NEWTON_DEFINE_RANGE_UNARY_OPERATOR(~, thrust::logical_not);

// binary logical
__NEWTON_DEFINE_RANGE_BINARY_OPERATOR(&&, thrust::logical_and);
__NEWTON_DEFINE_RANGE_BINARY_OPERATOR(||, thrust::logical_or);

// binary bitwise
__NEWTON_DEFINE_RANGE_BINARY_OPERATOR(&,  thrust::bit_and);
__NEWTON_DEFINE_RANGE_BINARY_OPERATOR(|,  thrust::bit_or);
__NEWTON_DEFINE_RANGE_BINARY_OPERATOR(^,  thrust::bit_xor);

// put these in namespace ops so that the user can easily make
// arithmetic available to all ranges with a single using directive
namespace ops
{

// arithmetic
using newton::operator/;
using newton::operator-;
using newton::operator%;
using newton::operator*;
using newton::operator+;

// relational
using newton::operator==;
using newton::operator>;
using newton::operator>=;
using newton::operator<;
using newton::operator<=;
using newton::operator!=;

// logical
using newton::operator~;
using newton::operator&&;
using newton::operator||;

// bitwise
using newton::operator&;
using newton::operator|;
using newton::operator^;

} // end namespace ops

// put these in namespace detail so that arithmetic is
// available to detail::transform_range and detail::constant_range
namespace detail
{

// arithmetic
using newton::operator/;
using newton::operator-;
using newton::operator%;
using newton::operator*;
using newton::operator+;

// relational
using newton::operator==;
using newton::operator>;
using newton::operator>=;
using newton::operator<;
using newton::operator<=;
using newton::operator!=;

// logical
using newton::operator~;
using newton::operator&&;
using newton::operator||;

// bitwise
using newton::operator&;
using newton::operator|;
using newton::operator^;

} // end namespace detail

} // end namespace newton

