// Copyright 2009-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/platform.h"

namespace oidn {

  using std::isfinite;
  using std::isnan;

  // Returns ceil(a / b) for non-negative integers
  template<typename Int, typename IntB>
  __forceinline constexpr Int ceil_div(Int a, IntB b)
  {
    //assert(a >= 0);
    //assert(b > 0);
    return (a + b - 1) / b;
  }

  // Returns a rounded up to multiple of b
  template<typename Int, typename IntB>
  __forceinline constexpr Int round_up(Int a, IntB b)
  {
    return ceil_div(a, b) * b;
  }

} // namespace oidn
