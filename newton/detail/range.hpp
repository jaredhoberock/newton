#pragma once

#include <thrust/iterator/iterator_traits.h>
#include <newton/detail/type_traits.hpp>

namespace newton
{
namespace detail
{

template<typename Range>
inline __host__ __device__
  typename newton::detail::range_iterator<Range>::type
    begin(Range &rng)
{
  return rng.begin();
}

template<typename Range>
inline __host__ __device__
  typename range_iterator<const Range>::type
    begin(const Range &rng)
{
  return rng.begin();
}

template<typename Range>
inline __host__ __device__
  typename range_iterator<Range>::type
    end(Range &rng)
{
  return rng.end();
}

template<typename Range>
inline __host__ __device__
  typename range_iterator<const Range>::type
    end(const Range &rng)
{
  return rng.end();
}


// helpers to dispatch begin & end via adl
// when in the presence of a similarly-named overload
template<typename Range>
inline __host__ __device__
  typename range_iterator<Range>::type
    adl_begin(Range &rng)
{
  return begin(rng);
}

template<typename Range>
inline __host__ __device__
  typename range_iterator<const Range>::type
    adl_begin(const Range &rng)
{
  return begin(rng);
}

template<typename Range>
inline __host__ __device__
  typename range_iterator<Range>::type
    adl_end(Range &rng)
{
  return end(rng);
}

template<typename Range>
inline __host__ __device__
  typename range_iterator<const Range>::type
    adl_end(const Range &rng)
{
  return end(rng);
}

// XXX specialize for RandomAccessIterator
//     to avoid storing two iterators
template<typename Iterator>
  class range
{
  public:
    typedef Iterator                                             iterator;
    typedef typename thrust::iterator_value<iterator>::type      value_type;
    typedef typename thrust::iterator_reference<iterator>::type  reference;
    typedef typename thrust::iterator_difference<iterator>::type difference_type;

    inline __host__ __device__
    range(iterator first, iterator last)
      : m_begin(first), m_end(last)
    {}

    template<typename Range>
    inline __host__ __device__
    range(Range &rng)
      : m_begin(adl_begin(rng)),
        m_end(adl_end(rng))
    {}

    template<typename Range>
    inline __host__ __device__
    range(const Range &rng)
      : m_begin(adl_begin(rng)),
        m_end(adl_end(rng))
    {}

    inline __host__ __device__
    iterator begin() const
    {
      return m_begin;
    }

    inline __host__ __device__
    iterator end() const
    {
      return m_end;
    }

    inline __host__ __device__
    difference_type size() const
    {
      return end() - begin();
    }

    inline __host__ __device__
    bool empty() const
    {
      return begin() == end();
    }

    inline __host__ __device__
    reference operator[](const difference_type &i) const
    {
      return begin()[i];
    }

  private:
    iterator m_begin, m_end;
};

// specialize range_iterator for const range<Iterator>
template<typename Iterator>
  struct range_iterator<const range<Iterator> >
{
  typedef typename range<Iterator>::iterator type;
};

template<typename Iterator>
__host__ __device__
range<Iterator> make_range(Iterator first, Iterator last)
{
  return range<Iterator>(first,last);
}

template<typename Range>
__host__ __device__
range<typename range_iterator<Range>::type> make_range(Range &rng)
{
  return range<typename range_iterator<Range>::type>(rng);
}

template<typename Range>
__host__ __device__
range<typename range_iterator<const Range>::type> make_range(const Range &rng)
{
  return range<typename range_iterator<const Range>::type>(rng);
}

} // end detail
} // end newton

