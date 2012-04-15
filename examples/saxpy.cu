#include <thrust/device_vector.h>
#include <newton/numeric_range.hpp>
#include <iostream>
#include <typeinfo>

int main()
{
  thrust::device_vector<float> vec_x(100), vec_y(100);

  typedef thrust::device_vector<float> range;

  typedef thrust::device_vector<float>::iterator iterator;

  newton::numeric_range<iterator> x = vec_x;
  newton::numeric_range<iterator> y = vec_y;
  
  x + y;

  return 0;
}

