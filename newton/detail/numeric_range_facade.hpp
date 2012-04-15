#pragma once

#include <newton/detail/range/type_traits.hpp>
#include <newton/detail/arithmetic.hpp>
#include <newton/detail/range/constant_range.hpp>

namespace newton
{
namespace detail
{

template<typename Derived>
  class numeric_range_facade
{
  public:
    typedef Derived derived_type;

    inline __host__ __device__
    derived_type &derived()
    {
      return static_cast<derived_type&>(*this);
    }

    inline __host__ __device__
    const derived_type &derived() const
    {
      return static_cast<const derived_type&>(*this);
    }
};

} // end detail


// binary arithmetic operators follow
// each operator has five versions:
//   0. numeric_range_facade, numeric_range_facade
//   1. numeric_range_facade, generic Range
//   2. generic Range, numeric_range_facade
//   3. numeric_range_facade, arithmetic Scalar
//   4. arithmetic Scalar, numeric_range_facade


// + case 0
template<typename Derived1, typename Derived2>
inline __host__ __device__
  typename detail::sum_ranges_result<
    Derived1,
    Derived2
  >::type
  operator+(const detail::numeric_range_facade<Derived1> &lhs, const detail::numeric_range_facade<Derived2> &rhs)
{
  return sum_ranges(lhs.derived(),rhs.derived());
} // end operator+()

// + case 1
template<typename Derived, typename Range>
inline __host__ __device__
  typename detail::lazy_enable_if_range<
    Range,
    detail::sum_ranges_result<
      Derived,
      Range
    >
  >::type
    operator+(const detail::numeric_range_facade<Derived> &lhs, const Range &rhs)
{
  return sum_ranges(lhs.derived(),rhs);
} // end operator+()

// + case 2
template<typename Range, typename Derived>
inline __host__ __device__
  typename detail::lazy_enable_if_range<
    Range,
    detail::sum_ranges_result<
      Range,
      Derived
    >
  >::type
    operator+(const Range &lhs, const detail::numeric_range_facade<Derived> &rhs)
{
  return sum_ranges(lhs,rhs.derived());
} // end operator+()


// + case 3
template<typename Derived, typename Scalar>
inline __host__ __device__
  typename detail::lazy_disable_if_range<
    Scalar,
    detail::sum_ranges_result<
      Derived,
      detail::constant_range<Scalar>
    >
  >::type
    operator+(const detail::numeric_range_facade<Derived> &lhs, const Scalar &rhs)
{
  return sum_ranges(lhs.derived(),detail::make_constant_range(rhs,lhs.derived().size()));
} // end operator+()

// + case 4
template<typename Scalar, typename Derived>
inline __host__ __device__
  typename detail::lazy_disable_if_range<
    Scalar,
    detail::sum_ranges_result<
      detail::constant_range<Scalar>,
      Derived
    >
  >::type
    operator+(const Scalar &lhs, const detail::numeric_range_facade<Derived> &rhs)
{
  return sum_ranges(detail::make_constant_range(lhs,rhs.derived().size()),rhs.derived());
} // end operator+()


// * case 0
template<typename Derived1, typename Derived2>
inline __host__ __device__
  typename detail::multiply_ranges_result<
    Derived1,
    Derived2
  >::type
  operator*(const detail::numeric_range_facade<Derived1> &lhs, const detail::numeric_range_facade<Derived2> &rhs)
{
  return multiply_ranges(lhs.derived(),rhs.derived());
} // end operator*()

// * case 1
template<typename Derived, typename Range>
inline __host__ __device__
  typename detail::lazy_enable_if_range<
    Range,
    detail::multiply_ranges_result<
      Derived,
      Range
    >
  >::type
    operator*(const detail::numeric_range_facade<Derived> &lhs, const Range &rhs)
{
  return multiply_ranges(lhs.derived(),rhs);
} // end operator*()

// * case 2
template<typename Range, typename Derived>
inline __host__ __device__
  typename detail::lazy_enable_if_range<
    Range,
    detail::multiply_ranges_result<
      Range,
      Derived
    >
  >::type
    operator*(const Range &lhs, const detail::numeric_range_facade<Derived> &rhs)
{
  return multiply_ranges(lhs,rhs.derived());
} // end operator*()


// * case 3
template<typename Derived, typename Scalar>
inline __host__ __device__
  typename detail::lazy_disable_if_range<
    Scalar,
    detail::multiply_ranges_result<
      Derived,
      detail::constant_range<Scalar>
    >
  >::type
    operator*(const detail::numeric_range_facade<Derived> &lhs, const Scalar &rhs)
{
  return multiply_ranges(lhs.derived(),detail::make_constant_range(rhs,lhs.derived().size()));
} // end operator*()

// * case 4
template<typename Scalar, typename Derived>
inline __host__ __device__
  typename detail::lazy_disable_if_range<
    Scalar,
    detail::multiply_ranges_result<
      detail::constant_range<Scalar>,
      Derived
    >
  >::type
    operator*(const Scalar &lhs, const detail::numeric_range_facade<Derived> &rhs)
{
  return multiply_ranges(detail::make_constant_range(lhs,rhs.derived().size()),rhs.derived());
} // end operator*()


} // end newton

