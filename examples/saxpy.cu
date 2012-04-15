#include <thrust/device_vector.h>
#include <newton/numeric_range.hpp>
#include <iostream>
#include <typeinfo>

int main()
{
  // initialize host arrays
  float h_vec_x[4] = {1.0, 1.0, 1.0, 1.0};
  float h_vec_y[4] = {1.0, 2.0, 3.0, 4.0};

  thrust::device_vector<float> d_vec_x(h_vec_x,h_vec_x + 4);
  thrust::device_vector<float> d_vec_y(h_vec_y,h_vec_y + 4);

  typedef thrust::device_vector<float>::iterator iterator;

  // create a numeric "view" of the vectors
  newton::numeric_range<iterator> x = d_vec_x;
  newton::numeric_range<iterator> y = d_vec_y;

  const float a = 2.0f;
  
  a * x + y;

  return 0;
}

