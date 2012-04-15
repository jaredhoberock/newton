#pragma once

#include <newton/detail/range/range.hpp>
#include <newton/detail/arithmetic.hpp>
#include <newton/detail/type_traits.hpp>
#include <newton/detail/range/constant_range.hpp>

namespace newton
{

// XXX assumes is_aritmetic<typename value_type<Iterator>::type>
template<typename Iterator>
  class numeric_range
    : public newton::detail::range<Iterator>
{
  private:
    typedef newton::detail::range<Iterator> super_t;

  public:
    typedef typename super_t::iterator iterator;

    inline __host__ __device__
    numeric_range(iterator first, iterator last)
      : super_t(first,last)
    {}

    template<typename Range>
    inline __host__ __device__
    numeric_range(Range &rng)
      : super_t(rng)
    {}

    template<typename Range>
    inline __host__ __device__
    numeric_range(const Range &rng)
      : super_t(rng)
    {}
};

template<typename Range>
inline __host__ __device__
  numeric_range<typename detail::range_iterator<Range> >
    make_numeric_range(Range &rng)
{
  return numeric_range<typename detail::range_iterator<Range> >(rng);
}

template<typename Range>
inline __host__ __device__
  numeric_range<typename detail::range_iterator<const Range> >
    make_numeric_range(const Range &rng)
{
  return numeric_range<typename detail::range_iterator<const Range> >(rng);
}

// binary arithmetic operators follow
// each operator has five versions:
//   0. numeric_range, numeric_range
//   1. numeric_range, generic Range
//   2. generic Range, numeric_range
//   3. numeric_range, arithmetic Scalar
//   4. arithmetic Scalar, numeric_range
//
// XXX should probably disambiguate within
// enable_if with is_range instead of is_arithmetic
// e.g. std::complex fails is_arithmetic

// + case 0
template<typename Iterator1, typename Iterator2>
inline __host__ __device__
  typename detail::sum_ranges_result<
    numeric_range<Iterator1>,
    numeric_range<Iterator2>
  >::type
  operator+(const numeric_range<Iterator1> &lhs, const numeric_range<Iterator2> &rhs)
{
  return sum_ranges(lhs,rhs);
} // end operator+()

// + case 1
template<typename Iterator, typename Range>
inline __host__ __device__
  typename detail::lazy_disable_if<
    detail::is_arithmetic<Range>::value,
    detail::sum_ranges_result<
      numeric_range<Iterator>,
      Range
    >
  >::type
    operator+(const numeric_range<Iterator> &lhs, const Range &rhs)
{
  return sum_ranges(lhs,rhs);
} // end operator+()

// + case 2
template<typename Range, typename Iterator>
inline __host__ __device__
  typename detail::lazy_disable_if<
    detail::is_arithmetic<Range>::value,
    detail::sum_ranges_result<
      Range,
      numeric_range<Iterator>
    >
  >::type
    operator+(const Range &lhs, const numeric_range<Iterator> &rhs)
{
  return sum_ranges(lhs,rhs);
} // end operator+()


// + case 3
template<typename Iterator, typename Scalar>
inline __host__ __device__
  typename detail::lazy_enable_if<
    detail::is_arithmetic<Scalar>::value,
    detail::sum_ranges_result<
      numeric_range<Iterator>,
      detail::constant_range<Scalar>
    >
  >::type
    operator+(const numeric_range<Iterator> &lhs, const Scalar &rhs)
{
  return sum_ranges(lhs,detail::make_constant_range(rhs,lhs.size()));
} // end operator+()

// + case 4
template<typename Scalar, typename Iterator>
inline __host__ __device__
  typename detail::lazy_enable_if<
    detail::is_arithmetic<Scalar>::value,
    detail::sum_ranges_result<
      detail::constant_range<Scalar>,
      numeric_range<Iterator>
    >
  >::type
    operator+(const Scalar &lhs, const numeric_range<Iterator> &rhs)
{
  return sum_ranges(detail::make_constant_range(lhs,rhs.size()),rhs);
} // end operator+()


// * case 0
template<typename Iterator1, typename Iterator2>
inline __host__ __device__
  typename detail::multiply_ranges_result<
    numeric_range<Iterator1>,
    numeric_range<Iterator2>
  >::type
    operator*(const numeric_range<Iterator1> &lhs, const numeric_range<Iterator2> &rhs)
{
  return multiply_ranges(lhs,rhs);
} // end operator*()

// * case 1
template<typename Iterator, typename Range>
inline __host__ __device__
  typename detail::lazy_disable_if<
    detail::is_arithmetic<Range>::value,
    detail::multiply_ranges_result<
      numeric_range<Iterator>,
      Range
    >
  >::type
    operator*(const numeric_range<Iterator> &lhs, const Range &rhs)
{
  return multiply_ranges(lhs,rhs);
} // end operator*()

// * case 2
template<typename Range, typename Iterator>
inline __host__ __device__
  typename detail::lazy_disable_if<
    detail::is_arithmetic<Range>::value,
    detail::multiply_ranges_result<
      Range,
      numeric_range<Iterator>
    >
  >::type
    operator*(const Range &lhs, const numeric_range<Iterator> &rhs)
{
  return multiply_ranges(lhs,rhs);
} // end operator*()

// * case 3
template<typename Iterator, typename Scalar>
inline __host__ __device__
  typename detail::lazy_enable_if<
    detail::is_arithmetic<Scalar>::value,
    detail::multiply_ranges_result<
      numeric_range<Iterator>,
      detail::constant_range<Scalar>
    >
  >::type
    operator*(const numeric_range<Iterator> &lhs, const Scalar &rhs)
{
  return multiply_ranges(lhs,detail::make_constant_range(rhs,lhs.size()));
} // end operator*()

// * case 4
template<typename Scalar, typename Iterator>
inline __host__ __device__
  typename detail::lazy_enable_if<
    detail::is_arithmetic<Scalar>::value,
    detail::multiply_ranges_result<
      detail::constant_range<Scalar>,
      numeric_range<Iterator>
    >
  >::type
    operator*(const Scalar &lhs, const numeric_range<Iterator> &rhs)
{
  return multiply_ranges(detail::make_constant_range(lhs,rhs.size()),rhs);
} // end operator*()


} // end newton

