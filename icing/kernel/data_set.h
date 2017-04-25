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

#ifndef _DATA_SET_H_
#define _DATA_SET_H_

#include <cstddef>
#include <cassert>
#include <vector>
#include <string>

struct DataElement {
  size_t length;
  int * attributes;

  DataElement() : length(0), attributes(0) {}

  void allocate(size_t length_) {
    length = length_;
    attributes = new int[length];
  }

  ~DataElement() {
    if (attributes)
      delete [] attributes;
  }
};


class DataSet {

 public:
  DataSet(size_t max_length, int symbol_size):
    _max_length(max_length), _symbol_size(symbol_size), _size(0), _elements(0)
    {}

  void load_strings(const std::vector<std::string> &strings) {
    _size = strings.size();
    _elements = new DataElement[_size];

    for (size_t i = 0; i < _size; i++) {
      assert(strings[i].length() < _max_length);

      size_t str_len = strings[i].length();
      _elements[i].allocate(str_len);

      for (size_t j = 0; j < str_len; j++) {
        char temp = *(strings[i].substr(j, 1).c_str());
        assert(static_cast<int>(temp) < _symbol_size);

        // work with uppercase letters
        _elements[i].attributes[j] = static_cast<int>(toupper(temp));
      }
    }
  }

  ~DataSet() {
    delete [] _elements;
  }

  const DataElement * elements() const {
    return _elements;
  }

  size_t size() const {
    return _size;
  }

 private:
  size_t _max_length;
  int _symbol_size;
  size_t _size;
  DataElement *_elements;
};

#endif
