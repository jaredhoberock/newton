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
    inline __host__ __device__
    numeric_range(iterator first, iterator last)
      : super_t(first,last);
    {}

    template<Range>
    inline __host__ __device__
    range(const Range &rng)
      : super_t(rng)
    {}
};

// XXX arithmetic operators here

} // end newton

