// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "sh/default_image.h"

namespace sh {

template <typename T>
DefaultImage<T>::DefaultImage(int width, int height)
    : width_(width), height_(height) {
  pixels_.reset(new Eigen::Array<T,3,1>[width * height]);
}

template <typename T>
int DefaultImage<T>::width() const { return width_; }

template <typename T>
int DefaultImage<T>::height() const { return height_; }

template <typename T>
Eigen::Array<T,3,1> DefaultImage<T>::GetPixel(int x, int y) const {
  int index = x + y * width_;
  return pixels_[index];
}

template <typename T>
void DefaultImage<T>::SetPixel(int x, int y, const Eigen::Array<T,3,1>& v) {
  int index = x + y * width_;
  pixels_[index] = v;
}

// Explicit template instantiation.
template class DefaultImage<float>;
template class DefaultImage<double>;

}  // namespace sh
