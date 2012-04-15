#pragma once

#include <thrust/detail/vector_base.h>
#include <thrust/device_malloc_allocator.h>
#include <newton/detail/numeric_range_facade.hpp>
#include <newton/detail/type_traits.hpp>
#include <vector>


namespace newton
{


// XXX for now, use malloc_allocator, but we'll probably
//     want to use a better default, e.g. an allocator
//     that does not initialize elements
//     see http://stackoverflow.com/questions/7218574/avoiding-default-construction-of-elements-in-standard-containers
template<typename T, typename Allocator = thrust::device_malloc_allocator<T> >
  class numeric_vector
    : public thrust::detail::vector_base<T,Allocator>,
      public newton::detail::numeric_range_facade<numeric_vector<T,Allocator> >
{
  private:
    typedef thrust::detail::vector_base<T,Allocator> super_t;

  public:
    typedef typename super_t::size_type  size_type;
    typedef typename super_t::value_type value_type;

    inline numeric_vector()
      : super_t()
    {}

    inline explicit numeric_vector(size_type n, const value_type &value = value_type())
      : super_t(n,value)
    {}

    inline numeric_vector(const numeric_vector &x)
      : super_t(x)
    {}

    template<typename Range>
    inline numeric_vector(const Range &rng, typename detail::enable_if_range<Range>::type * = 0)
      : super_t(rng.begin(), rng.end())
    {}

    template<typename InputIterator>
    inline numeric_vector(InputIterator first, InputIterator last)
      : super_t(first,last)
    {}

    inline numeric_vector &operator=(const numeric_vector &x)
    {
      super_t::operator=(x);
      return *this;
    }

    template<typename Range>
    typename detail::enable_if_range<
      Range,
      numeric_vector &
    >::type
    operator=(const Range &rng)
    {
      this->assign(rng.begin(),rng.end());
      return *this;
    }
}; // end numeric_vector


} // end newton

