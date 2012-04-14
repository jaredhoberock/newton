#include <thrust/iterator/zip_iterator.h>
#include <newton/detail/range.hpp>

namespace newton
{
namespace detail
{

template<typename IteratorTuple>
  class zip_range
    : newton::detail::range<thrust::zip_iterator<IteratorTuple> >
{
  private:
    typedef newton::detail::range<thrust::zip_iterator<IteratorTuple> super_t;

  public:
    inline __host__ __device__
    zip_range(IteratorTuple first, IteratorTuple last)
      : super_t(first,last)
    {}
}; // end zip_range

template<IteratorTuple>
inline __host__ __device__
zip_range<IteratorTuple> make_zip_range(IteratorTuple first, IteratorTuple last)
{
  return zip_range<IteratorTuple>(first,last);
}

} // end detail
} // end newton

