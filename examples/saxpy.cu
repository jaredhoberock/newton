#include <newton/newton.hpp>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <iostream>
#include <iterator>

int main()
{
  // initialize host arrays
  float h_x[4] = {1.0, 1.0, 1.0, 1.0};
  float h_y[4] = {1.0, 2.0, 3.0, 4.0};

  thrust::device_vector<float> x(h_x,h_x + 4);

  // create a numeric_vector (a container) from h_y
  // by default, numeric_vector is associated with thrust's device system
  newton::numeric_vector<float> y = h_y;

  const float a = 2.0f;

  // importing this namespace enables arithmetic on ranges like device_vector
  // numeric_vector has arithmetic operators either way
  using namespace newton::ops;
  
  // saxpy to y 
  y = a * x + y;

  std::cout << "y: ";
  thrust::copy(y.begin(), y.end(), std::ostream_iterator<float>(std::cout, " "));
  std::cout << std::endl;

  return 0;
}

