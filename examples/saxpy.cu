#include <newton/newton.hpp>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <iostream>
#include <iterator>

int main()
{
  // initialize host arrays
  float h_vec_x[4] = {1.0, 1.0, 1.0, 1.0};
  float h_vec_y[4] = {1.0, 2.0, 3.0, 4.0};

  thrust::device_vector<float> d_vec_x(h_vec_x,h_vec_x + 4);

  typedef thrust::device_vector<float>::iterator iterator;

  // create a numeric "view" vector of vector x
  newton::numeric_range<iterator> x = d_vec_x;

  // create a numeric_vector (a container) from h_vec_y
  // by default, numeric_vector is associated with thrust's device system
  newton::numeric_vector<float> y = h_vec_y;

  const float a = 2.0f;
  
  // saxpy to y 
  y = a * x + y;

  std::cout << "y: ";
  thrust::copy(y.begin(), y.end(), std::ostream_iterator<float>(std::cout, " "));
  std::cout << std::endl;

  return 0;
}

