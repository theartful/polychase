#pragma once

#include <fmt/base.h>
#include <spdlog/spdlog.h>

#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>
#include <cstddef>
#include <unordered_set>

#include "eigen_typedefs.h"
#include "pnp/types.h"
#include "utils.h"

// This file is heavily inspired by PoseLib's lm_impl.h
// https://github.com/PoseLib/PoseLib/blob/master/PoseLib/robust/lm_impl.h
// Which is written by Viktor Larsson

/*
   struct LevMarqDenseProblem {
        using Parameters = ...;

        // If number of params is unknown, it should be -1, otherwise, same as NumParams, but constexpr
        static constexpr int kNumParams;
        // The vector length of each residual
        static constexpr int kResidualLength;

        size_t NumParams() const; // Total number of params

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

   struct LevMarqSparseProblem {
        using Parameters = ...;

        static constexpr bool kShouldNormalize;
        // If number of params is unknown, it should be -1, otherwise, same as NumParams, but constexpr
        static constexpr int kNumParams;
        // The vector length of each residual
        static constexpr int kResidualLength;

        size_t NumParams() const; // Actual number of params

        size_t NumParamBlocks() const;

        size_t ParamBlockLength() const;

        size_t NumResiduals(size_t edge_idx) const;

        size_t NumEdges() const;

        std::pair<size_t, size_t> GetEdge(size_t edge_idx) const;

        std::optional<RowMajorMatrixf<kResidualLength, 1>>
        Evaluate(const Parameters& params, size_t edge_idx, size_t residual_idx);

        bool EvaluateWithJacobian(
            const Parameters& param,
            size_t edge_idx,
            size_t residual_idx,
            std::vector<Eigen::Triplet<Float>>& J,
            RowMajorMatrixf<kResidualLength, 1>& residual
        );

        Parameters Step(const Parameters& param, const RowMajorMatrixf<kNumParams, 1>& step) const;
   };

   struct LossFunction {
        Float Loss(Float squared_error);
        Float Weight(Float squared_error);
   };
*/

using IterationCallback = std::function<bool(const BundleStats& stats)>;

template <typename Problem, typename LossFunction, typename WeightVector>
class LevMarqDenseSolver {
    using Parameters = typename Problem::Parameters;
    using Solver =
        Eigen::LLT<typename Eigen::SelfAdjointView<RowMajorMatrixf<Problem::kNumParams, Problem::kNumParams>,
                                                   Eigen::Lower>::PlainObject,
                   Eigen::Lower>;

   public:
    LevMarqDenseSolver(Problem& problem, const LossFunction& loss_fn, const WeightVector& weights,
                       const BundleOptions& opts, IterationCallback callback)
        : problem(problem), loss_fn(loss_fn), weights(weights), callback(callback), opts(opts) {
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
        stats = {};
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

            const Eigen::ComputationInfo status = ComputeStep();
            if (status != Eigen::Success) {
                stats.invalid_steps++;
                stats.lambda = std::min(opts.max_lambda, stats.lambda * v);
                v = 2 * v;

                rebuild = false;
                continue;
            }

            stats.step_norm = step.norm();
            if (stats.step_norm < opts.step_tol) {
                break;
            }

            const Parameters params_new = problem.Step(*params, step);
            const Float cost_new = TotalCost(params_new);

            if (cost_new < stats.cost) {
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

            if (callback != nullptr) {
                if (!callback(stats)) {
                    break;
                }
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

            J.setZero();
            RowMajorMatrixf<Problem::kResidualLength, 1> res;

            problem.EvaluateWithJacobian(params, idx, J, res);

            const Float r_squared = res.squaredNorm();
            const Float weight = weights[idx] * loss_fn.Weight(r_squared);

            if (weight == 0.0) {
                continue;
            }

            JtJ.template selfadjointView<Eigen::Lower>().rankUpdate(J.transpose(), weight);
            Jtr += J.transpose() * (weight * res);
        }

        JtJ_diag = JtJ.diagonal().cwiseMax(1e-6).cwiseMin(1e32);
    }

    Eigen::ComputationInfo ComputeStep() {
        // Add dampening
        JtJ.diagonal() = JtJ_diag * (1.0 + stats.lambda);

        linear_solver.compute(JtJ.template selfadjointView<Eigen::Lower>());

        // Remove dampening
        JtJ.diagonal() = JtJ_diag;

        if (linear_solver.info() != Eigen::Success) {
            return linear_solver.info();
        }

        step = linear_solver.solve(-Jtr);
        return linear_solver.info();
    }

    Float TotalCost(const Parameters& params) {
        const size_t num_residuals = problem.NumResiduals();
        Float cost = 0.0f;

        // TODO: Parallelize?
        for (size_t idx = 0; idx < num_residuals; idx++) {
            if (weights[idx] == 0.0f) {
                continue;
            }
            RowMajorMatrixf<Problem::kResidualLength, 1> res = problem.Evaluate(params, idx);
            cost += weights[idx] * loss_fn.Loss(res.squaredNorm());
        }

        return cost;
    }

   private:
    Problem& problem;
    const LossFunction& loss_fn;
    const WeightVector& weights;
    Solver linear_solver;
    IterationCallback callback;

    // FIXME: Since we have our custom bundle adjuster, maybe create our custom bundle types?
    const BundleOptions& opts;

    BundleStats stats;

    RowMajorMatrixf<Problem::kResidualLength, Problem::kNumParams> J;
    RowMajorMatrixf<Problem::kNumParams, Problem::kNumParams> JtJ;
    RowMajorMatrixf<Problem::kNumParams, 1> JtJ_diag;
    RowMajorMatrixf<Problem::kNumParams, 1> Jtr;
    RowMajorMatrixf<Problem::kNumParams, 1> step;
};

template <typename Problem, typename LossFunction, typename WeightVector>
class LevMarqSparseSolver {
    using Parameters = typename Problem::Parameters;
    using Solver =
        Eigen::SimplicialLLT<typename Eigen::SelfAdjointView<Eigen::SparseMatrix<Float>, Eigen::Lower>::PlainObject,
                             Eigen::Lower>;

   public:
    LevMarqSparseSolver(Problem& problem, const LossFunction& loss_fn, const WeightVector& weights,
                        const BundleOptions& opts, IterationCallback callback)
        : problem(problem), loss_fn(loss_fn), weights(weights), opts(opts), callback(callback) {
        const size_t param_block_length = problem.ParamBlockLength();
        const size_t num_param_blocks = problem.NumParamBlocks();
        const size_t num_params = num_param_blocks * param_block_length;

        CHECK_EQ(num_params, problem.NumParams());

        J_pair.resize(Problem::kResidualLength, 2 * param_block_length);
        JtJ.resize(num_params, num_params);
        JtJ_pair.resize(2 * param_block_length, 2 * param_block_length);
        Jtr.resize(num_params, 1);
        Jtr_pair.resize(2 * param_block_length, 1);
        JtJ_diag.resize(num_params, 1);
        step.resize(num_params, 1);

        SPDLOG_INFO("Creating JtJ pattern...");

        struct pair_hash {
            std::size_t operator()(const std::pair<int, int>& v) const { return v.first * 31 + v.second; }
        };
        std::unordered_set<std::pair<size_t, size_t>, pair_hash> unique_edges;

        for (size_t i = 0; i < problem.NumEdges(); i++) {
            auto [b1, b2] = problem.GetEdge(i);

            CHECK_LT(b1, num_param_blocks);
            CHECK_LT(b2, num_param_blocks);
            CHECK_NE(b1, b2);

            if (b1 > b2) {
                std::swap(b1, b2);
            }

            if (unique_edges.contains(std::make_pair(b1, b2))) {
                continue;
            }

            unique_edges.insert({b1, b2});
        }

        std::vector<Eigen::Triplet<float>> triplets;
        triplets.reserve(unique_edges.size() * param_block_length * param_block_length +
                         num_param_blocks * param_block_length * (param_block_length + 1) / 2);

        for (auto [b1, b2] : unique_edges) {
            const size_t b1_start_idx = b1 * param_block_length;
            const size_t b2_start_idx = b2 * param_block_length;

            for (size_t j = 0; j < param_block_length; j++) {
                for (size_t k = 0; k < param_block_length; k++) {
                    const size_t r = b2_start_idx + k;
                    const size_t c = b1_start_idx + j;

                    triplets.emplace_back(r, c, 0.0f);
                }
            }
        }

        for (size_t i = 0; i < num_param_blocks; i++) {
            const size_t start_idx = i * param_block_length;

            for (size_t j = 0; j < param_block_length; j++) {
                for (size_t k = j; k < param_block_length; k++) {
                    const size_t r = start_idx + k;
                    const size_t c = start_idx + j;

                    triplets.emplace_back(r, c, 0.0f);
                }
            }
        }

        CHECK_EQ(triplets.size(), triplets.capacity());

        JtJ.resize(num_params, num_params);
        JtJ.setFromTriplets(triplets.begin(), triplets.end());
        JtJ.makeCompressed();

        SPDLOG_INFO("Analyzing JtJ pattern...");
        linear_solver.analyzePattern(JtJ.template selfadjointView<Eigen::Lower>());
    }

    BundleStats Solve(Parameters* params) {
        // Initialize stats
        stats = {};
        stats.cost = TotalCost(*params, current_num_valid_residuals);
        stats.initial_cost = stats.cost;
        stats.grad_norm = -1;
        stats.step_norm = -1;
        stats.invalid_steps = 0;
        stats.lambda = opts.initial_lambda;

        params_new = *params;

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

            const Eigen::ComputationInfo status = ComputeStep();
            if (status != Eigen::Success) {
                stats.invalid_steps++;
                stats.lambda = std::min(opts.max_lambda, stats.lambda * v);
                v = 2 * v;

                rebuild = false;
                continue;
            }

            stats.step_norm = step.norm();
            if (stats.step_norm < opts.step_tol) {
                break;
            }

            size_t num_valid_residuals = 0;

            problem.Step(*params, step, params_new);
            const Float cost_new = TotalCost(params_new, num_valid_residuals);

            if (cost_new < stats.cost) {
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

                std::swap(*params, params_new);
                stats.cost = cost_new;
                v = 2;

                current_num_valid_residuals = num_valid_residuals;
                rebuild = true;
            } else {
                stats.invalid_steps++;
                stats.lambda = std::min(opts.max_lambda, stats.lambda * v);
                v = 2 * v;

                rebuild = false;
            }

            if (callback != nullptr) {
                if (!callback(stats)) {
                    break;
                }
            }
        }

        return stats;
    }

   private:
    void AccumBlockInSparseMatrix(Eigen::SparseMatrix<Float>& sparse_mat, Eigen::Ref<RowMajorMatrixXf> block, int row,
                                  int col) {
        Float* values = sparse_mat.valuePtr();
        const int* inner_indices = sparse_mat.innerIndexPtr();
        const int* outer_starts = sparse_mat.outerIndexPtr();

        // Delta is the distance from the last element in the column, to the element in the column at the row that we
        // want (+ 1 because it's actually from the first element in the next column).
        //
        // So delta is negative. This assumes that sparse_mat consists of blocks, so delta should be the same.
        //
        // I can probably cache this value in an array of size NumParamBlocks x NumParamBlocks
        const int* first_ptr =
            std::lower_bound(inner_indices + outer_starts[col], inner_indices + outer_starts[col + 1], row);

        const int delta = std::distance(inner_indices + outer_starts[col + 1], first_ptr);

        CHECK_LE(delta, 0);

        for (int src_col = 0; src_col < block.cols(); ++src_col) {
            const int target_col = col + src_col;

            // Index of the first element is:
            // - Block is not at the diagonal: Index of the last element in the column + delta that we computed before.
            // - Block is at the diagonal:     Index of the last element in the column + delta + src_col. The extra
            //   offset is because otherwise, we would lie at the upper part.
            const int first_idx = outer_starts[target_col + 1] + delta + ((row == col) ? src_col : 0);
            const int first_target_row = inner_indices[first_idx];
            Float* row_ptr = values + first_idx;

            // Check that we're within target_col and didn't go over to the next column.
            CHECK_LT(first_idx, outer_starts[target_col + 1]);
            // Check that we're in the lower half.
            CHECK_GE(first_target_row, target_col);
            // Check that we're within the target block [row, row + nRows)
            CHECK_GE(first_target_row, row);

            // Process all rows in the block range that exist in this column
            for (int src_row = first_target_row - row; src_row < block.rows(); src_row++) {
                // We're assuming that block is a lower triangular matrix
                if (src_row >= src_col) {
                    *row_ptr += block(src_row, src_col);
                } else {
                    *row_ptr += block(src_col, src_row);
                }

                row_ptr++;
            }
        }
    }

    void BuildNormalEquations(const Parameters& params) {
        std::fill(JtJ.valuePtr(), JtJ.valuePtr() + JtJ.nonZeros(), 0.0);
        Jtr.setZero();

        const size_t num_edges = problem.NumEdges();
        size_t actual_num_valid_residuals = 0;

        // TODO: Parallelize?
        for (size_t edge_idx = 0; edge_idx < num_edges; edge_idx++) {
            const float edge_weight = weights[edge_idx];
            if (edge_weight == 0.0f) {
                continue;
            }

            JtJ_pair.setZero();
            Jtr_pair.setZero();

            const size_t num_residuals = problem.NumResiduals(edge_idx);
            for (size_t res_idx = 0; res_idx < num_residuals; res_idx++) {
                J_pair.setZero();
                RowMajorMatrixf<Problem::kResidualLength, 1> res;

                const bool ok = problem.EvaluateWithJacobian(params, edge_idx, res_idx, J_pair, res);
                if (ok) {
                    const Float r_squared = res.squaredNorm();
                    const Float weight = edge_weight * loss_fn.Weight(r_squared);

                    JtJ_pair.template selfadjointView<Eigen::Lower>().rankUpdate(J_pair.transpose(), weight);
                    Jtr_pair += J_pair.transpose() * (weight * res);

                    actual_num_valid_residuals++;
                }
            }

            // INVESTIGATE!!
            // Dividing here instead of at the end (even though it's more expensive),
            // because I think it's more numerically stable.
            if constexpr (Problem::kShouldNormalize) {
                JtJ_pair /= current_num_valid_residuals;
                Jtr_pair /= current_num_valid_residuals;
            }

            /* JtJ_pair consists of four blocks:
             *  | J1.transpose() * J1         J1.transpose() * J2 |
             *  | J2.transpose() * J1         J2.transpose() * J2 |
             */
            const auto [param_block_1, param_block_2] = problem.GetEdge(edge_idx);
            CHECK_NE(param_block_1, param_block_2);

            const int param_block_length = problem.ParamBlockLength();
            const int param_block_1_idx = param_block_1 * param_block_length;
            const int param_block_2_idx = param_block_2 * param_block_length;

            AccumBlockInSparseMatrix(JtJ, JtJ_pair.block(0, 0, param_block_length, param_block_length),
                                     param_block_1_idx, param_block_1_idx);

            AccumBlockInSparseMatrix(
                JtJ, JtJ_pair.block(param_block_length, param_block_length, param_block_length, param_block_length),
                param_block_2_idx, param_block_2_idx);

            if (param_block_1_idx > param_block_2_idx) {
                AccumBlockInSparseMatrix(JtJ,
                                         JtJ_pair.block(0, param_block_length, param_block_length, param_block_length),
                                         param_block_1_idx, param_block_2_idx);
            } else {
                AccumBlockInSparseMatrix(JtJ,
                                         JtJ_pair.block(param_block_length, 0, param_block_length, param_block_length),
                                         param_block_2_idx, param_block_1_idx);
            }

            // Set Jtr
            Jtr.block(param_block_1_idx, 0, param_block_length, 1) += Jtr_pair.block(0, 0, param_block_length, 1);
            Jtr.block(param_block_2_idx, 0, param_block_length, 1) +=
                Jtr_pair.block(param_block_length, 0, param_block_length, 1);
        }

        CHECK_EQ(current_num_valid_residuals, actual_num_valid_residuals);

        JtJ_diag = JtJ.diagonal().cwiseMax(1e-6).cwiseMin(1e32);
    }

    Float TotalCost(const Parameters& params, size_t& num_valid_residuals) {
        const size_t num_edges = problem.NumEdges();
        num_valid_residuals = 0;
        Float cost = 0.0f;

        // TODO: Parallelize?
        for (size_t edge_idx = 0; edge_idx < num_edges; edge_idx++) {
            const float edge_weight = weights[edge_idx];
            if (edge_weight == 0.0f) {
                continue;
            }

            Float edge_cost = 0.0f;

            const size_t num_residuals = problem.NumResiduals(edge_idx);
            for (size_t res_idx = 0; res_idx < num_residuals; res_idx++) {
                auto maybe_res = problem.Evaluate(params, edge_idx, res_idx);
                if (maybe_res.has_value()) {
                    const auto& res = *maybe_res;
                    edge_cost += loss_fn.Loss(res.squaredNorm());
                    num_valid_residuals++;
                }
            }

            cost += edge_weight * edge_cost;
        }

        if constexpr (Problem::kShouldNormalize) {
            return cost / num_valid_residuals;
        } else {
            return cost;
        }
    }

    Eigen::ComputationInfo ComputeStep() {
        // Add dampening
        JtJ.diagonal() = JtJ_diag * (1.0 + stats.lambda);

        linear_solver.compute(JtJ.template selfadjointView<Eigen::Lower>());

        // Remove dampening
        JtJ.diagonal() = JtJ_diag;

        if (linear_solver.info() != Eigen::Success) {
            return linear_solver.info();
        }

        step = linear_solver.solve(-Jtr);
        return linear_solver.info();
    }

   private:
    Problem& problem;
    Parameters params_new;
    const LossFunction& loss_fn;
    const WeightVector& weights;
    const BundleOptions& opts;
    IterationCallback callback;

    Solver linear_solver;

    RowMajorMatrixf<Problem::kResidualLength, Eigen::Dynamic> J_pair;
    Eigen::SparseMatrix<Float> JtJ;
    RowMajorMatrixf<Eigen::Dynamic, Eigen::Dynamic> JtJ_pair;
    RowMajorMatrixf<Eigen::Dynamic, 1> JtJ_diag;
    RowMajorMatrixf<Eigen::Dynamic, 1> Jtr;
    RowMajorMatrixf<Eigen::Dynamic, 1> Jtr_pair;
    RowMajorMatrixf<Eigen::Dynamic, 1> step;

    size_t current_num_valid_residuals;

    BundleStats stats;
};

template <typename Problem, typename LossFunction, typename WeightVector,
          typename Parameters = typename Problem::Parameters>
BundleStats LevMarqDenseSolve(Problem& problem, const LossFunction& loss_fn, const WeightVector& weights,
                              Parameters* params, const BundleOptions& opts, IterationCallback callback = nullptr) {
    LevMarqDenseSolver solver(problem, loss_fn, weights, opts, callback);
    return solver.Solve(params);
}

template <typename Problem, typename LossFunction, typename WeightVector,
          typename Parameters = typename Problem::Parameters>
BundleStats LevMarqSparseSolve(Problem& problem, const LossFunction& loss_fn, const WeightVector& weights,
                               Parameters* params, const BundleOptions& opts, IterationCallback callback = nullptr) {
    LevMarqSparseSolver solver(problem, loss_fn, weights, opts, callback);
    return solver.Solve(params);
    return {};
}
