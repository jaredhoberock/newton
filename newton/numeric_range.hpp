#pragma once

#include <newton/detail/range/range.hpp>
#include <newton/detail/arithmetic.hpp>

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

// arithmetic operators follow
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


} // end newton

