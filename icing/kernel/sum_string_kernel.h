#ifndef _SUM_STRING_KERNEL_H_
#define _SUM_STRING_KERNEL_H_

#include <string>
#include <vector>
#include <cassert>
#include <cmath>
#include "data_set.h"
#include "string_kernel.h"

template<class k_type>
class SumStringKernel {
 public:
  /** Constructor, sets kernel parameters. */
  SumStringKernel(size_t min_kn, size_t max_kn,
                  const int normalize, const int symbol_size,
                  const size_t max_length, double lambda, const int hard_matching)
      : _min_kn(min_kn), _max_kn(max_kn), _normalize(normalize),
        _symbol_size(symbol_size), _max_length(max_length),
        _lambda(lambda),
        _hard_matching(hard_matching) {
            _num_subseq_length = max_kn - min_kn + 1;
        }

  ~SumStringKernel() {
        delete [] _kernel;
        delete _string_data;
  }

  /** Set the dataset to be used by the kernel. */
  void set_data(const std::vector<std::string> &strings);

  /** Calculate the kernel. */
  void compute_kernel();
  void copy_kernel(k_type * copy);

  /** Return pointer to kernel matrix. */
  k_type * values() const {
    assert(_kernel);
    return _kernel;
  }

  /** Return the size of the array of StringKernels. */
  size_t size() const {
    return _num_subseq_length;
  }

 protected:
  const size_t _min_kn;
  const size_t _max_kn;
  const int _normalize;
  const int _symbol_size;
  const size_t _max_length;
  const double _lambda;
  const int _hard_matching;
  size_t _num_subseq_length;
  DataSet *_string_data;
  k_type *_kernel;
};


template<class k_type>
void SumStringKernel<k_type>::set_data(const std::vector<std::string> &strings) {
  assert(strings.size() > 0);
  _string_data = new DataSet(_max_length, _symbol_size);
  _string_data->load_strings(strings);
}

template<class k_type>
void SumStringKernel<k_type>::copy_kernel(k_type * copy) {
    size_t kernel_dim_2 = _string_data->size() * _string_data->size();
    for (size_t i = 0; i < kernel_dim_2; i++) {
        copy[i] = _kernel[i];
    }
}

template<class k_type>
void SumStringKernel<k_type>::compute_kernel() {
  assert(_string_data);

  size_t i, j;
  size_t kernel_dim = _string_data->size();

  // kernel is just the sum of kernels
  // after having the sum, we can normalise with the diagonal
  _kernel = new k_type [kernel_dim*kernel_dim];

  // initialise to 0
  for (i = 0; i < kernel_dim * kernel_dim; i++) {
      _kernel[i] = 0;
  }

  // compute the sum of unnormalised kernels
  StringKernel<k_type> * string_kernel;
  for(i = 0; i < _num_subseq_length; i++) {
      string_kernel = new StringKernel<k_type>(0, _symbol_size,
            _max_length, _min_kn + i, _lambda, _hard_matching);
      string_kernel -> set_data(_string_data); //avoid copying
      string_kernel -> compute_kernel();

      // sum to the kernel
      for (j = 0; j < kernel_dim * kernel_dim; j++) {
          _kernel[j] += string_kernel -> _kernel[j];
      }
      delete string_kernel;
  }

  if(_normalize) {
      for (i = 0; i < kernel_dim; i++) {
          for (j = i + 1; j < kernel_dim; j++) {
            // K[i,j] /= sqrt(K[i,i] + K[j,j])
            _kernel[i*kernel_dim + j] /= sqrt(_kernel[i*kernel_dim + i] *
                                              _kernel[j*kernel_dim + j]);
            _kernel[j*kernel_dim + i] = _kernel[i*kernel_dim + j];
          }
      }
      // normalise the diagonal
      for (i = 0; i < kernel_dim; i++) {
          _kernel[i * kernel_dim + i] = 1;
      }
  }
}


#endif
