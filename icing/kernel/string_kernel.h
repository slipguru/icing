/*********************************************************************
* Software License Agreement (BSD License)
*
*  Copyright (c) 2012, Marianna Madry
*  All rights reserved.
*
*  Contact: marianna.madry@gmail.com
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * The name of contributors may not be used to endorse or promote products
*     derived from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

#ifndef _STRING_KERNEL_H_
#define _STRING_KERNEL_H_

#include <string>
#include <vector>
#include <cassert>
#include <cmath>
#include "data_set.h"
#include "models.h"

size_t get_idx_aa(char x) {
    if (x >= 'A' && x <= 'Z') {
        return x - 'A' + 3;
    }
    switch (x) {
        case '.': return 0;
        case '-': return 1;
        case '*': return 2;
        default: throw 20; return -1;
    }
    return 0; // never reached
}

template<class k_type>
class StringKernel {
 public:
  /** Constructor, sets kernel parameters. */
  StringKernel(int normalize, int symbol_size,
               size_t max_length, size_t kn, double lambda, int hard_matching)
        {
            _normalize = normalize;
            _symbol_size = symbol_size;
            _max_length = max_length;
            _kn = kn;
            _lambda = lambda;
            _hard_matching = hard_matching;
            _string_data = 0;
            _kernel = 0;
        }

  ~StringKernel() {
    delete [] _kernel;
    if(_normalize) {
        delete [] norms;
    }
    if(_private_dataset) {
        delete _string_data;
    }
  }

  /** Set the dataset to be used by the kernel. */
  void set_data(const std::vector<std::string> &strings);
  void set_data(DataSet * dataset);

  /** Calculate the kernel and the norms. */
  void compute_kernel();
  void compute_norms();

  /** Return pointer to kernel matrix. */
  k_type * values() const {
    assert(_kernel);
    return _kernel;
  }

  /** Return the size of the NxN kernel. */
  size_t size() const {
    assert(_string_data);
    return _string_data->size();
  }

 // protected:
  int _hard_matching;
  int _normalize;
  int _symbol_size;
  size_t _max_length;
  size_t _kn;
  double _lambda;
  DataSet *_string_data;
  k_type *_kernel;
  k_type * norms;

 private:
  k_type kernel(const DataElement &x, const DataElement &y) const;
  int _private_dataset;
};


template<class k_type>
void StringKernel<k_type>::set_data(const std::vector<std::string> &strings) {
  assert(strings.size() > 0);
  _string_data = new DataSet(_max_length, _symbol_size);
  _string_data->load_strings(strings);

  _private_dataset = 1;
}

template<class k_type>
void StringKernel<k_type>::set_data(DataSet * dataset) {
  _string_data = dataset; // copy the pointer, to avoid copying the same data
  _private_dataset = 0;
}


template<class k_type>
k_type StringKernel<k_type>::kernel(const DataElement &x, const DataElement &y) const {
    size_t i, j, k;
    k_type * Kd[2];
    size_t x_dim = x.length + 1;
    size_t y_dim = y.length + 1;
    if(x.length < _kn || y.length < _kn) {
        if(x.length == y.length) {
            for(i = 0; i < x.length; i++) {
                if(x.attributes[i] != y.attributes[i]) {
                    return 0;
                }
            }
            return 1;
        }
        return 0;
    }

    // Allocate and initialise Kd
    for (i = 0; i < 2; i++) {
      Kd[i] = new k_type [x_dim * y_dim];
      for (j = 0; j < x_dim * y_dim; j++) {
          Kd[i][j] = (i + 1) % 2;
      }
    }

  // Kd now contains two matrices, that are n+1 x m+1 (empty string included)
  // Kd[0] is composed by 1s (follows the definition of K_0)
  // Kd[1] is composed by 0s -> it starts to be filled

  // start with i = kn = 1, 2, 3 ...
  for (i = 1; i < _kn; i++) {
    /* Set the Kd to zero for those lengths of s and t
    where s (or t) has exactly length i-1 and t (or s)
    has length >= i-1. L-shaped upside down matrix */
    for (j = (i - 1); j < x.length; j++) {
      Kd[i % 2][j * y_dim + i - 1] = 0;
    }
    for (j = (i - 1); j < y.length; j++) {
      Kd[i % 2][(i - 1) * y_dim + j] = 0;
    }

    // // print matrix, DEBUG
    // for (int zzz = 0; zzz < 2; zzz++) {
    //     printf("########### Kd[%d] ##########\n", zzz);
    //     int _i;
    //     for (_i = 0, printf("  e "); _i < y.length; _i++) {
    //           printf("%c ", y.attributes[_i]);
    //     }
    //     printf("\n");
    //
    //     for (int j = 0; j < (x.length + 1); j++) {
    //         if(j==0) printf("e ");
    //         else printf("%c ", x.attributes[j-1]);
    //         for (int k = 0; k < (y.length + 1); k++) {
    //             printf("%.9f ", Kd[zzz][j*y_dim + k]);
    //         }
    //         printf("\n");
    //     }
    // }

    for (j = i; j < x.length; j++) {
      // Kdd maintains the contribution of the left and diagonal terms
      // that is, ONLY the contribution of the left (not influenced by the
      // upper terms) and the eventual contibution of lambda^2 in case the
      // chars are the same
      k_type Kdd = 0;
      for (k = i; k < y.length; k++) {
        if (x.attributes[j - 1] != y.attributes[k - 1]) {
          // ((.))-1 is because indices start with 0 (not with 1)
          Kdd = _lambda * Kdd;
        } else {
          Kdd = _lambda * (Kdd + (_lambda * Kd[(i + 1) % 2][(j - 1)*y_dim + k - 1]));
        }
        Kd[i % 2][j*y_dim+k] = _lambda * Kd[i % 2][(j - 1) * y_dim + k] + Kdd;
      }
    }
  }

  // Calculate K
  k_type sum = 0;
  try {
    for (i = _kn - 1; i < x.length; i++) {
      for (j = _kn - 1; j < y.length; j++) {
          // hard matching
          if(_hard_matching) {
              if (x.attributes[i] == y.attributes[j]) {
                  sum += _lambda * _lambda * Kd[(_kn - 1) % 2][i*y_dim + j];
              }
          } else {
              // soft matching, regulated from models.h, amminoacidic model
              sum += _lambda * _lambda * \
                  aa_model[get_idx_aa(x.attributes[i])*29 +
                           get_idx_aa(y.attributes[j])] * \
                  Kd[(_kn - 1) % 2][i*y_dim + j];
          }
      }
    }
  } catch (int e) {
    printf("Exception occurred. n. %d\n", e);
    sum = -1;
  }

  for (i = 0; i < 2; i++) {
      delete[] Kd[i];
  }
  return sum;
}

template<class k_type>
void StringKernel<k_type>::compute_kernel() {
  assert(_string_data);

  size_t i, j;
  size_t kernel_dim = _string_data->size();

  // if computing normalised kernel, then compute norms
  if (_normalize) {
      compute_norms();
  }

  // Compute kernel using dynamic programming
  _kernel = new k_type [kernel_dim * kernel_dim];
  size_t offset = _normalize ? 1 : 0;
  for (i = 0; i < kernel_dim; i++) {
      if(_normalize) {
         _kernel[i*kernel_dim+i] = 1;
      }
      for (j = i + offset; j < kernel_dim; j++) {
        _kernel[i*kernel_dim+j] = kernel(_string_data->elements()[i],
                                         _string_data->elements()[j]);
        if (_normalize) {
            _kernel[i*kernel_dim+j] /= sqrt(norms[i] * norms[j]);
        }
        _kernel[j*kernel_dim+i] = _kernel[i*kernel_dim+j];
    }
  }
}

template<class k_type>
void StringKernel<k_type>::compute_norms() {
    // Get values for normalization, it is computed for elements in diagonal
    norms = new k_type [_string_data->size()];
    for (size_t i = 0; i < _string_data->size(); i++) {
      norms[i] = kernel(_string_data->elements()[i],
                        _string_data->elements()[i]);
    }
}
#endif
