#pragma once

#include <thrust/detail/type_traits.h>

namespace newton
{
namespace detail
{
namespace type_traits_detail
{

template<typename T>
  struct has_const_iterator
{
  typedef char yes_type;
  typedef int  no_type;
  template<typename S> static yes_type test(typename S::const_iterator *);
  template<typename S> static no_type  test(...);
  static bool const value = sizeof(test<T>(0)) == sizeof(yes_type);
  typedef thrust::detail::integral_constant<bool, value> type;
}; // end has_const_iterator

template<typename Range>
  struct nested_const_iterator
{
  typedef typename Range::const_iterator type;
};

template<typename Range>
  struct nested_iterator
{
  typedef typename Range::iterator type;
};

// if the Range has a nested const_iterator type, return it
// else, return its nested iterator type
template<typename Range>
  struct range_const_iterator
    : thrust::detail::eval_if<
        has_const_iterator<Range>::value,
        nested_const_iterator<Range>,
        nested_iterator<Range>
      >
{};

} // end type_traits_detail


template<typename Range>
  struct range_iterator
{
  typedef typename Range::iterator type;
};

// for const Ranges, we need to return the const_iterator
// type, if it exists
template<typename Range>
  struct range_iterator<const Range>
    : type_traits_detail::range_const_iterator<Range>
{
};

template<typename Range>
  struct range_value
{
  typedef typename Range::value_type type;
};

} // end detail
} // end newton

