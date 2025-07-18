// Copyright (c) 2021, Viktor Larsson
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of the copyright holder nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <cmath>

#include "eigen_typedefs.h"

#define SWITCH_LOSS_FUNCTIONS                   \
    case BundleOptions::LossType::TRIVIAL:      \
        SWITCH_LOSS_FUNCTION_CASE(TrivialLoss); \
        break;                                  \
    case BundleOptions::LossType::HUBER:        \
        SWITCH_LOSS_FUNCTION_CASE(HuberLoss);   \
        break;                                  \
    case BundleOptions::LossType::CAUCHY:       \
        SWITCH_LOSS_FUNCTION_CASE(CauchyLoss);  \
        break;

// Robust Loss functions
class TrivialLoss {
   public:
    TrivialLoss(Float) {
    }  // dummy to ensure we have consistent calling interface
    TrivialLoss() {}
    Float Loss(Float r2) const { return r2; }
    Float Weight(Float) const { return 1.0; }
    Float Curvature(Float) const { return 0.0; }
};

class HuberLoss {
   public:
    HuberLoss(Float threshold) : thr(threshold) {}
    Float Loss(Float r2) const {
        if (r2 <= thr * thr) {
            return r2;
        } else {
            const Float r = std::sqrt(r2);
            return thr * (2.0 * r - thr);
        }
    }
    Float Weight(Float r2) const {
        if (r2 <= thr * thr) {
            return 1.0;
        } else {
            const Float r = std::sqrt(r2);
            return thr / r;
        }
    }
    Float Curvature(Float r2) const {
        if (r2 < thr * thr) {
            return 0.0;
        } else {
            return -Weight(r2) / (2.0 * r2);
        }
    }

   private:
    const Float thr;
};

class CauchyLoss {
   public:
    CauchyLoss(Float threshold)
        : sq_thr(threshold * threshold), inv_sq_thr(1.0 / sq_thr) {}
    Float Loss(Float r2) const { return sq_thr * std::log1p(r2 * inv_sq_thr); }
    Float Weight(Float r2) const {
        return std::max(std::numeric_limits<Float>::min(),
                        Float(1.0) / (Float(1.0) + r2 * inv_sq_thr));
    }
    Float Curvature(Float r2) const {
        return -inv_sq_thr * std::pow(1.0 / (1.0 + r2 * inv_sq_thr), 2);
    }

   private:
    const Float sq_thr;
    const Float inv_sq_thr;
};
