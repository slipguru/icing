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

#ifndef _FILE_IO_H_
#define _FILE_IO_H_

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include "string_kernel.h"
#include "sum_string_kernel.h"


template <class k_type>
bool write_libsvm_kernel(const std::string &file_name,
                         const std::vector<int> &labels,
                         const StringKernel<k_type> &kernel) {
  assert(labels.size() == kernel.size());
  int size = labels.size();

  std::ofstream file;
  file.open(file_name.c_str());
  if (file.is_open()) {
    for (int i = 0; i < size; i++) {
      file << labels[i] << " 0:" << i << " ";
      for (int j = 0; j < size; j++)
        file << j + 1 << ":" << kernel.values()[i][j] << " ";

      file << std::endl;
    }
    file.close();
    return true;
  }
  return false;
}

template <class k_type>
bool write_kernel(const std::string &file_name,
                  const std::vector<std::string> &labels,
                  const StringKernel<k_type> &kernel) {
  assert(labels.size() == kernel.size());
  int size = labels.size();

  std::ofstream file;
  file.open(file_name.c_str());
  if (file.is_open()) {
    for (int i = 0; i < size; i++) {
    //   file << labels[i] << " 0:" << i << " ";
      file << labels[i] << "\t";
      for (int j = 0; j < size; j++)
        file << kernel.values()[i][j] << "\t";

      file << std::endl;
    }
    file.close();
    return true;
  }
  return false;
}

template <class k_type>
bool write_kernel(const std::string &file_name,
                  const std::vector<std::string> &labels,
                  const SumStringKernel<k_type> &kernel) {
  assert(labels.size() == kernel.size());
  size_t size = labels.size();

  std::ofstream file;
  file.open(file_name.c_str());
  if (file.is_open()) {
      for (size_t i = 0; i < size; i++) {
          file << "," << labels[i];
      }
      file << std::endl;
      for (size_t i = 0; i < size; i++) {
          file << labels[i];
          for (size_t j = 0; j < size; j++) {
              file << "," << kernel.values()[i*size+j];
          }
          file << std::endl;
      }
      file.close();
      return true;
  }
  return false;
}

template <class k_type>
bool write_kernel_cout(const std::vector<std::string> &labels,
                       const StringKernel<k_type> &kernel) {
    // assert(labels.size() == kernel.size());
    int size = labels.size();

    for (int i = 0; i < size; i++) {
      std::cout << labels[i] << "\t";
      for (int j = 0; j < size; j++)
        std::cout << kernel.values()[i*size+j] << "\t";

      std::cout << std::endl;
    }
    return true;
  }

template <class k_type>
bool write_kernel_cout(const std::vector<std::string> &labels,
                       const SumStringKernel<k_type> &kernel) {
    // assert(labels.size() == kernel.size());
    int size = labels.size();

    for (int i = 0; i < size; i++) {
      std::cout << labels[i] << "\t";
      for (int j = 0; j < size; j++)
        std::cout << kernel.values()[i*size+j] << "\t";

      std::cout << std::endl;
    }
    return true;
  }

#endif
