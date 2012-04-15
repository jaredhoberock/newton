#pragma once

namespace newton
{
namespace detail
{

template<typename _Tp, _Tp __v>
  struct integral_constant
{
  static const _Tp                      value = __v;
  typedef _Tp                           value_type;
  typedef integral_constant<_Tp, __v>   type;
};

typedef integral_constant<bool, true>     true_type;
typedef integral_constant<bool, false>    false_type;


template<bool, typename T = void> struct enable_if {};
template<typename T>              struct enable_if<true, T> {typedef T type;};
template<bool, typename T> struct lazy_enable_if {};
template<typename T>       struct lazy_enable_if<true, T> {typedef typename T::type type;};

template<bool condition, typename T = void> struct disable_if : enable_if<!condition, T> {};
template<bool condition, typename T>        struct lazy_disable_if : lazy_enable_if<!condition, T> {};


template<typename T> struct is_integral                           : public false_type {};
template<>           struct is_integral<bool>                     : public true_type {};
template<>           struct is_integral<char>                     : public true_type {};
template<>           struct is_integral<signed char>              : public true_type {};
template<>           struct is_integral<unsigned char>            : public true_type {};
template<>           struct is_integral<short>                    : public true_type {};
template<>           struct is_integral<unsigned short>           : public true_type {};
template<>           struct is_integral<int>                      : public true_type {};
template<>           struct is_integral<unsigned int>             : public true_type {};
template<>           struct is_integral<long>                     : public true_type {};
template<>           struct is_integral<unsigned long>            : public true_type {};
template<>           struct is_integral<long long>                : public true_type {};
template<>           struct is_integral<unsigned long long>       : public true_type {};
template<>           struct is_integral<const bool>               : public true_type {};
template<>           struct is_integral<const char>               : public true_type {};
template<>           struct is_integral<const unsigned char>      : public true_type {};
template<>           struct is_integral<const short>              : public true_type {};
template<>           struct is_integral<const unsigned short>     : public true_type {};
template<>           struct is_integral<const int>                : public true_type {};
template<>           struct is_integral<const unsigned int>       : public true_type {};
template<>           struct is_integral<const long>               : public true_type {};
template<>           struct is_integral<const unsigned long>      : public true_type {};
template<>           struct is_integral<const long long>          : public true_type {};
template<>           struct is_integral<const unsigned long long> : public true_type {};

template<typename T> struct is_floating_point              : public false_type {};
template<>           struct is_floating_point<float>       : public true_type {};
template<>           struct is_floating_point<double>      : public true_type {};
template<>           struct is_floating_point<long double> : public true_type {};


template<typename T>
  struct is_arithmetic
    : integral_constant<
        bool,
        is_integral<T>::value || is_floating_point<T>::value
      >
{};


} // end detail
} // end newton

