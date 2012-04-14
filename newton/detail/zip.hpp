#include <newton/detail/zip_range.hpp>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>

namespace newton
{
namespace detail
{


template<typename Range>
inline __host__ __device__
  zip_range<thrust::tuple<typename range_iterator<Range>::type> >
    zip(Range &rng)
{
  return zip_range<thrust::tuple<typename range_iterator<Range>::type> >(thrust::make_tuple(rng));
}

template<typename Range>
inline __host__ __device__
  zip_range<thrust::tuple<typename range_iterator<const Range>::type> >
    zip(const Range &rng)
{
  return zip_range<thrust::tuple<typename range_iterator<const Range>::type> >(thrust::make_tuple(rng));
}


template<typename Range1, typename Range2>
inline __host__ __device__
  zip_range<
    thrust::tuple<
      typename range_iterator<Range1>::type,
      typename range_iterator<Range2>::type,
    >
  >
    zip(Range1 &rng1, Range2 &rng2)
{
  return zip_range<thrust::tuple<typename range_iterator<Range1>::type, typename range_iterator<Range2>::type> >(thrust::make_tuple(rng1,rng2));
}

template<typename Range1, typename Range2>
inline __host__ __device__
  zip_range<
    thrust::tuple<
      typename range_iterator<Range1>::type,
      typename range_iterator<const Range2>::type,
    >
  >
    zip(Range1 &rng1, const Range2 &rng2)
{
  return zip_range<thrust::tuple<typename range_iterator<Range1>::type, typename range_iterator<const Range2>::type> >(thrust::make_tuple(rng1,rng2));
}

template<typename Range1, typename Range2>
inline __host__ __device__
  zip_range<
    thrust::tuple<
      typename range_iterator<const Range1>::type,
      typename range_iterator<Range2>::type,
    >
  >
    zip(const Range1 &rng1, Range2 &rng2)
{
  return zip_range<thrust::tuple<typename range_iterator<const Range1>::type, typename range_iterator<Range2>::type> >(thrust::make_tuple(rng1,rng2));
}

template<typename Range1, typename Range2>
inline __host__ __device__
  zip_range<
    thrust::tuple<
      typename range_iterator<const Range1>::type,
      typename range_iterator<const Range2>::type,
    >
  >
    zip(const Range1 &rng1, const Range2 &rng2)
{
  return zip_range<thrust::tuple<typename range_iterator<const Range1>::type, typename range_iterator<const Range2>::type> >(thrust::make_tuple(rng1,rng2));
}


} // end detail
} // end newton

