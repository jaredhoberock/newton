#include <newton/newton.hpp>
#include <thrust/device_vector.h>

static constexpr float sqrt2 = 1.41421356237; 
static constexpr float inv_sqrt2 = 1.f / sqrt2;

template<typename T>
inline __host__ __device__
  auto cnd(const T &x)
    -> decltype(0.5f * (1.f + erf(inv_sqrt2 * x)))
{
  return 0.5f * (1.f + erf(inv_sqrt2 * x));
}

void black_scholes(const thrust::device_vector<float> &stock_price,
                   const thrust::device_vector<float> &option_strike,
                   const thrust::device_vector<float> &option_years,
                   const thrust::device_vector<float> &riskless_rate,
                   const thrust::device_vector<float> &volatility,
                   newton::numeric_vector<float> &call_result,
                   newton::numeric_vector<float> &put_result)
{
  // to apply math functions on device_vector, 
  // we can use a using namespace newton directive
  using namespace newton;

  auto sqrt_option_years = sqrt(option_years);
  auto d1 = (log(stock_price / option_strike) + (riskless_rate + 0.5f * volatility * volatility) * option_years) / (volatility * sqrt_option_years);
  auto d2 = d1 - volatility * sqrt_option_years;

  auto expRT = exp(-riskless_rate * option_years);

  // we receive results as numeric_vector so that we can assign the result of numeric expressions to them
  // note that all of the operations above and below are "fused" into two kernels which are evaluated on
  // each of the two assignments below
  call_result = stock_price * cnd(d1) - option_strike * expRT * cnd(d2);
  put_result  = option_strike * expRT * (1.0f - cnd(d2)) - stock_price * (1.0f - cnd(d1));
}

int main()
{
  size_t num_options = 1 << 20;

  // initialize problem
  thrust::device_vector<float> stock_price(num_options,   100);
  thrust::device_vector<float> option_strike(num_options, 98);
  thrust::device_vector<float> option_years(num_options,  2);
  thrust::device_vector<float> riskless_rate(num_options, 0.02);
  thrust::device_vector<float> volatility(num_options,    5);

  // storage for result
  newton::numeric_vector<float> call_result(num_options), put_result(num_options);

  black_scholes(stock_price, option_strike, option_years, riskless_rate, volatility, call_result, put_result);

  return 0;
}

