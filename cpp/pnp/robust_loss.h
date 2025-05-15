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
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <cmath>

#include "eigen_typedefs.h"

#define SWITCH_LOSS_FUNCTIONS                           \
    case BundleOptions::LossType::TRIVIAL:              \
        SWITCH_LOSS_FUNCTION_CASE(TrivialLoss);         \
        break;                                          \
    case BundleOptions::LossType::TRUNCATED:            \
        SWITCH_LOSS_FUNCTION_CASE(TruncatedLoss);       \
        break;                                          \
    case BundleOptions::LossType::HUBER:                \
        SWITCH_LOSS_FUNCTION_CASE(HuberLoss);           \
        break;                                          \
    case BundleOptions::LossType::CAUCHY:               \
        SWITCH_LOSS_FUNCTION_CASE(CauchyLoss);          \
        break;                                          \
    case BundleOptions::LossType::TRUNCATED_LE_ZACH:    \
        SWITCH_LOSS_FUNCTION_CASE(TruncatedLossLeZach); \
        break;

// Robust Loss functions
class TrivialLoss {
   public:
    TrivialLoss(Float) {}  // dummy to ensure we have consistent calling interface
    TrivialLoss() {}
    Float Loss(Float r2) const { return r2; }
    Float Weight(Float) const { return 1.0; }
};

class TruncatedLoss {
   public:
    TruncatedLoss(Float threshold) : squared_thr(threshold * threshold) {}
    Float Loss(Float r2) const { return std::min(r2, squared_thr); }
    Float Weight(Float r2) const { return (r2 < squared_thr) ? 1.0 : 0.0; }

   private:
    const Float squared_thr;
};

// The method from
//  Le and Zach, Robust Fitting with Truncated Least Squares: A Bilevel Optimization Approach, 3DV 2021
// for truncated least squares optimization with IRLS.
class TruncatedLossLeZach {
   public:
    TruncatedLossLeZach(Float threshold) : squared_thr(threshold * threshold), mu(0.5) {}
    Float Loss(Float r2) const { return std::min(r2, squared_thr); }
    Float Weight(Float r2) const {
        Float r2_hat = r2 / squared_thr;
        Float zstar = std::min(r2_hat, Float(1.0));

        if (r2_hat < 1.0) {
            return 0.5;
        } else {
            // assumes mu > 0.5
            Float r2m1 = r2_hat - 1.0;
            Float rho = (2.0 * r2m1 + std::sqrt(4.0 * r2m1 * r2m1 * mu * mu + 2 * mu * r2m1)) / mu;
            Float a = (r2_hat + mu * rho * zstar - 0.5 * rho) / (1 + mu * rho);
            Float zbar = std::max(Float(0.0), std::min(a, Float(1.0)));
            return (zstar - zbar) / rho;
        }
    }

   private:
    const Float squared_thr;

   public:
    // hyper-parameter for penalty strength
    Float mu;
    // schedule for increasing mu in each iteration
    static constexpr Float alpha = 1.5;
};

class HuberLoss {
   public:
    HuberLoss(Float threshold) : thr(threshold) {}
    Float Loss(Float r2) const {
        const Float r = std::sqrt(r2);
        if (r <= thr) {
            return r2;
        } else {
            return thr * (2.0 * r - thr);
        }
    }
    Float Weight(Float r2) const {
        const Float r = std::sqrt(r2);
        if (r <= thr) {
            return 1.0;
        } else {
            return thr / r;
        }
    }

   private:
    const Float thr;
};

class CauchyLoss {
   public:
    CauchyLoss(Float threshold) : inv_sq_thr(1.0 / (threshold * threshold)) {}
    Float Loss(Float r2) const { return std::log1p(r2 * inv_sq_thr); }
    Float Weight(Float r2) const {
        return std::max(std::numeric_limits<Float>::min(), inv_sq_thr / (Float(1.0) + r2 * inv_sq_thr));
    }

   private:
    const Float inv_sq_thr;
};
