#pragma once

#include <cstddef>

#include "eigen_typedefs.h"
#include "pnp/types.h"

// This file is heavily inspired by PoseLib's lm_impl.h
// https://github.com/PoseLib/PoseLib/blob/master/PoseLib/robust/lm_impl.h
// Which is written by Viktor Larsson

/*
   struct LevMarqDenseProblem {
        using Parameters = ...;

        // If number of params is unknown, it should be -1, otherwise, same as NumParams, but constexpr
        static constexpr int kNumParams = ...;
        // The vector length of each residual
        static constexpr int kResidualLength = ...;

        size_t NumParams() const; // Actual number of params

        size_t NumResiduals() const;

        RowMajorMatrixf<kResidualLength, 1>
        Evaluate(const Parameters& params, size_t residual_idx);

        void EvaluateWithJacobian(
            const Parameters& param,
            size_t residual_idx,
            RowMajorMatrixf<kResidualLength, kNumParams>& J,
            RowMajorMatrixf<kResidualLength, 1>& residual
        );

        Parameters Step(const Parameters& param, const RowMajorMatrixf<kNumParams, 1>& step) const;
   };


   struct LossFunction {
        Float Loss(Float squared_error);
        Float Weight(Float squared_error);
   };
*/

template <typename Problem, typename LossFunction, typename WeightVector>
class LevMarqDenseSolver {
    using Parameters = typename Problem::Parameters;
    using Hessian = RowMajorMatrixf<Problem::kNumParams, Problem::kNumParams>;
    using Solver = Eigen::LLT<typename Eigen::SelfAdjointView<Hessian, Eigen::Lower>::PlainObject, Eigen::Lower>;

   public:
    LevMarqDenseSolver(Problem& problem, LossFunction& loss_fn, WeightVector& weights, const BundleOptions& opts)
        : problem(problem), loss_fn(loss_fn), weights(weights), opts(opts) {
        // In case NumParams is dynamic
        if constexpr (Problem::kNumParams == Eigen::Dynamic) {
            const size_t num_params = problem.NumParams();

            J.resize(Problem::kResidualLength, num_params);
            JtJ.resize(num_params, num_params);
            Jtr.resize(num_params, 1);
            JtJ_diag.resize(num_params, 1);
            step.resize(num_params, 1);
        }
    }

    BundleStats Solve(Parameters* params) {
        // Initialize stats
        BundleStats stats;
        stats.cost = TotalCost(*params);
        stats.initial_cost = stats.cost;
        stats.grad_norm = -1;
        stats.step_norm = -1;
        stats.invalid_steps = 0;
        stats.lambda = opts.initial_lambda;

        Float v = 2.0f;
        bool rebuild = true;
        for (stats.iterations = 0; stats.iterations < opts.max_iterations; ++stats.iterations) {
            if (rebuild) {
                BuildNormalEquations(*params);

                stats.grad_norm = Jtr.norm();
                if (stats.grad_norm < opts.gradient_tol) {
                    break;
                }
            }

            // Add dampening
            JtJ.diagonal() = JtJ_diag * (1.0 + stats.lambda);

            linear_solver.compute(JtJ.template selfadjointView<Eigen::Lower>());
            if (linear_solver.info() != Eigen::Success) {
                stats.invalid_steps++;
                stats.lambda = std::min(opts.max_lambda, stats.lambda * v);
                v = 2 * v;

                rebuild = false;
                continue;
            }

            step = linear_solver.solve(-Jtr);

            stats.step_norm = step.norm();
            if (stats.step_norm < opts.step_tol) {
                break;
            }

            const Parameters params_new = problem.Step(*params, step);
            const Float cost_new = TotalCost(params_new);

            if (cost_new < stats.cost) {
                // Remove dampening, so that we can compute expected_cost_change
                JtJ.diagonal() = JtJ_diag;

                // See: http://www2.imm.dtu.dk/pubdb/edoc/imm3215.pdf
                const Float actual_cost_change = cost_new - stats.cost;
                const Float expected_cost_change =
                    step.transpose() * (2.0 * Jtr + JtJ.template selfadjointView<Eigen::Lower>() * step);

                const Float rho = actual_cost_change / expected_cost_change;

                // Mathematically, this should always be the case, but numerically it might happen
                // when the condition number of JtJ is high due to floating point precision errors.
                if (rho > 0) {
                    const Float factor = std::max(1.0 / 3.0, 1.0 - pow(2.0 * rho - 1.0, 3));
                    stats.lambda = std::max(opts.min_lambda, stats.lambda * factor);
                }

                *params = params_new;
                stats.cost = cost_new;
                v = 2;

                rebuild = true;
            } else {
                stats.invalid_steps++;
                stats.lambda = std::min(opts.max_lambda, stats.lambda * v);
                v = 2 * v;

                rebuild = false;
            }
        }

        return stats;
    }

   private:
    void BuildNormalEquations(const Parameters& params) {
        JtJ.setZero();
        Jtr.setZero();

        const size_t num_residuals = problem.NumResiduals();

        // TODO: Parallelize?
        for (size_t idx = 0; idx < num_residuals; ++idx) {
            if (weights[idx] == 0.0) {
                continue;
            }

            problem.EvaluateWithJacobian(params, idx, J, res);

            Float r_squared = res.squaredNorm();
            const Float weight = weights[idx] * loss_fn.Weight(r_squared);

            if (weight == 0.0) {
                continue;
            }

            JtJ.template selfadjointView<Eigen::Lower>().rankUpdate(J.transpose(), weight);
            Jtr += J.transpose() * (weight * res);
        }

        JtJ_diag = JtJ.diagonal().cwiseMax(1e-6).cwiseMin(1e32);
    }

    Float TotalCost(const Parameters& params) {
        const size_t num_residuals = problem.NumResiduals();
        Float cost = 0.0f;

        // TODO: Parallelize?
        for (size_t idx = 0; idx < num_residuals; idx++) {
            if (weights[idx] == 0.0f) {
                continue;
            }
            auto residual = problem.Evaluate(params, idx);
            cost += weights[idx] * loss_fn.Loss(residual.squaredNorm());
        }

        return cost;
    }

   private:
    Problem& problem;
    LossFunction& loss_fn;
    WeightVector& weights;
    Solver linear_solver;

    // FIXME: Since we have our custom bundle adjuster, maybe create our custom bundle types?
    const BundleOptions& opts;

    RowMajorMatrixf<Problem::kResidualLength, Problem::kNumParams> J;
    RowMajorMatrixf<Problem::kResidualLength, 1> res;
    RowMajorMatrixf<Problem::kNumParams, Problem::kNumParams> JtJ;
    RowMajorMatrixf<Problem::kNumParams, 1> Jtr;
    RowMajorMatrixf<Problem::kNumParams, 1> step;
    RowMajorMatrixf<Problem::kNumParams, 1> JtJ_diag;
};

template <typename Problem, typename LossFunction, typename WeightVector,
          typename Parameters = typename Problem::Parameters>
BundleStats LevMarqDenseSolve(Problem& problem, const LossFunction& loss_fn, const WeightVector& weights,
                              Parameters* params, const BundleOptions& opts) {
    LevMarqDenseSolver solver(problem, loss_fn, weights, opts);
    return solver.Solve(params);
}
