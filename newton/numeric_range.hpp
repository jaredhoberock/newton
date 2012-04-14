#include <newton/detail/range.hpp>

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

// XXX arithmetic operators here

} // end newton

