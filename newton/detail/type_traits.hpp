#pragma once

namespace newton
{
namespace detail
{

template<typename Range>
  struct range_iterator
{
  typedef typename Range::iterator type;
};

template<typename Container>
  struct range_iterator<const Container>
{
  typedef typename Container::const_iterator type;
};

} // end detail
} // end newton

