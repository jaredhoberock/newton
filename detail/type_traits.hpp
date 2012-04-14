namespace newton
{
namespace detail
{

template<typename Range>
  struct range_iterator
{
  typename Range::iterator type;
};

template<typename Range>
  struct range_iterator<const Range>
{
  typename Range::const_iterator type;
};

} // end detail
} // end newton

