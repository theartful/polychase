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

#include "PoseLib/types.h"
#include "eigen_typedefs.h"

/*
 Templated implementation of Levenberg-Marquadt.

 The Problem class must provide
    Problem::num_params - number of parameters to optimize over
    Problem::params_t - type for the parameters which optimize over
    Problem::accumulate(param, JtJ, Jtr) - compute jacobians
    Problem::residual(param) - compute the current residuals
    Problem::step(delta_params, param) - take a step in parameter space

    Check jacobian_impl.h for examples
*/

using IterationCallback = std::function<void(const poselib::BundleStats &stats)>;

template <typename Problem, typename Param = typename Problem::param_t>
poselib::BundleStats lm_impl(Problem &problem, Param *parameters, const poselib::BundleOptions &opt,
                             IterationCallback callback = nullptr) {
    constexpr int n_params = Problem::num_params;
    RowMajorMatrixf<n_params, n_params> JtJ;
    RowMajorMatrixf<n_params, 1> Jtr;
    RowMajorMatrixf<n_params, 1> sol;
    RowMajorMatrixf<n_params, 1> JtJ_diag;

    // Initialize
    poselib::BundleStats stats;
    stats.cost = problem.residual(*parameters);
    stats.initial_cost = stats.cost;
    stats.grad_norm = -1;
    stats.step_norm = -1;
    stats.invalid_steps = 0;
    stats.lambda = opt.initial_lambda;

    using Solver =
        Eigen::LLT<typename Eigen::SelfAdjointView<RowMajorMatrixf<n_params, n_params>, Eigen::Lower>::PlainObject,
                   Eigen::Lower>;

    Solver linear_solver;

    bool recompute_jac = true;
    for (stats.iterations = 0; stats.iterations < opt.max_iterations; ++stats.iterations) {
        // We only recompute jacobian and residual vector if last step was successful
        if (recompute_jac) {
            JtJ.setZero();
            Jtr.setZero();
            problem.accumulate(*parameters, JtJ, Jtr);
            stats.grad_norm = Jtr.norm();
            if (stats.grad_norm < opt.gradient_tol) {
                break;
            }
            JtJ_diag = JtJ.diagonal().cwiseMax(1e-6).cwiseMin(1e32);
        }

        // Add dampening
        // TODO: Further investigate Marquardt dampening vs Levenberg dampening
        // In my experiments, the Marquardt dampening outperformed Levenberg dampening
        JtJ.diagonal() = JtJ_diag * (1.0 + stats.lambda);
        // JtJ.diagonal() = JtJ_diag.array() + stats.lambda;

        linear_solver.compute(JtJ.template selfadjointView<Eigen::Lower>());
        sol = -linear_solver.solve(Jtr);

        stats.step_norm = sol.norm();
        if (stats.step_norm < opt.step_tol) {
            break;
        }

        Param parameters_new = problem.step(sol, *parameters);

        float cost_new = problem.residual(parameters_new);

        if (cost_new < stats.cost) {
            *parameters = parameters_new;
            stats.lambda = std::max(opt.min_lambda, stats.lambda / 10);
            stats.cost = cost_new;
            recompute_jac = true;
        } else {
            stats.invalid_steps++;
            stats.lambda = std::min(opt.max_lambda, stats.lambda * 10);
            recompute_jac = false;
        }
        if (callback != nullptr) {
            callback(stats);
        }
    }
    return stats;
}
