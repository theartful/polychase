#pragma once

#include <spdlog/spdlog.h>
#include <tbb/tbb.h>

#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>
#include <atomic>
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

        static constexpr bool kShouldNormalize;
        // Should be -1 if num of params is dynamic
        constexpr static constexpr int kNumParams;
        // The vector length of each residual
        static constexpr int kResidualLength;

        size_t NumParams() const; // Total number of params

        size_t NumResiduals() const;

        Float Weight(size_t res_idx) const;

        std::optional<RowMajorMatrixf<kResidualLength, 1>>
        Evaluate(const Parameters& params, size_t residual_idx);

        bool EvaluateWithJacobian(
            const Parameters& param,
            size_t residual_idx,
            RowMajorMatrixf<kResidualLength, kNumParams>& J,
            RowMajorMatrixf<kResidualLength, 1>& residual
        );

        Parameters Step(const Parameters& param, const
                        RowMajorMatrixf<kNumParams, 1>& step) const;
   };

   struct LevMarqSparseProblem {
        using Parameters = ...;

        static constexpr bool kShouldNormalize;
        // Should be -1 if num of params is dynamic
        constexpr static constexpr int kNumParams;
        // The vector length of each residual
        static constexpr int kResidualLength;

        size_t NumParams() const; // Actual number of params

        size_t NumParamBlocks() const;

        size_t ParamBlockLength() const;

        size_t NumResiduals(size_t edge_idx) const;

        size_t NumEdges() const;

        std::pair<size_t, size_t> GetEdge(size_t edge_idx) const;

        Float EdgeWeight(size_t edge_idx) const;

        Float ResidualWeight(size_t edge_idx, size_t res_idx) const;

        std::optional<RowMajorMatrixf<kResidualLength, 1>>
        Evaluate(const Parameters& params, size_t edge_idx, size_t res_idx);

        bool EvaluateWithJacobian(
            const Parameters& param,
            size_t edge_idx,
            size_t residual_idx,
            std::vector<Eigen::Triplet<Float>>& J,
            RowMajorMatrixf<kResidualLength, 1>& residual
        );

        Parameters Step(const Parameters& param, const
                        RowMajorMatrixf<kNumParams, 1>& step) const;
   };

   struct LossFunction {
        Float Loss(Float squared_error);
        Float Weight(Float squared_error);
   };
*/

using IterationCallback = std::function<bool(const BundleStats& stats)>;

template <typename Problem, typename LossFunction>
class LevMarqDenseSolver {
    using Parameters = typename Problem::Parameters;
    using Solver = Eigen::LLT<
        typename Eigen::SelfAdjointView<
            RowMajorMatrixf<Problem::kNumParams, Problem::kNumParams>,
            Eigen::Lower>::PlainObject,
        Eigen::Lower>;

   public:
    LevMarqDenseSolver(Problem& problem, const LossFunction& loss_fn,
                       const BundleOptions& opts, IterationCallback callback)
        : problem(problem), loss_fn(loss_fn), callback(callback), opts(opts) {
        // In case NumParams is dynamic
        if constexpr (Problem::kNumParams == Eigen::Dynamic) {
            const size_t num_params = problem.NumParams();

            JtJ.resize(num_params, num_params);
            Jtr.resize(num_params, 1);
            JtJ_diag.resize(num_params, 1);
            step.resize(num_params, 1);

            jac_data = tbb::enumerable_thread_specific<JacData>{[=]() {
                JacData data;
                data.J.resize(Problem::kResidualLength, num_params);
                data.JtJ.resize(num_params, num_params);
                data.Jtr.resize(num_params, 1);

                return data;
            }};
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

        params_new = *params;

        Float v = 2.0f;
        bool rebuild = true;
        for (stats.iterations = 0; stats.iterations < opts.max_iterations;
             ++stats.iterations) {
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
                if (stats.lambda == opts.max_lambda) {
                    break;
                }

                stats.lambda = std::min(opts.max_lambda, stats.lambda * v);
                v = 2 * v;

                rebuild = false;
                continue;
            }

            stats.step_norm = step.norm();
            if (stats.step_norm < opts.step_tol) {
                break;
            }

            problem.Step(*params, step, params_new);
            const Float cost_new = TotalCost(params_new);

            if (cost_new < stats.cost) {
                // See: http://www2.imm.dtu.dk/pubdb/edoc/imm3215.pdf
                const Float actual_cost_change = cost_new - stats.cost;
                const Float expected_cost_change =
                    step.transpose() *
                    (2.0 * Jtr +
                     JtJ.template selfadjointView<Eigen::Lower>() * step);

                const Float rho = actual_cost_change / expected_cost_change;

                // Mathematically, this should always be the case, but
                // numerically it might happen when the condition number of JtJ
                // is high due to floating point precision errors.
                if (rho > 0) {
                    const Float factor =
                        std::max(1.0 / 3.0, 1.0 - pow(2.0 * rho - 1.0, 3));
                    stats.lambda = std::clamp(stats.lambda * factor,
                                              opts.min_lambda, opts.max_lambda);
                }

                *params = params_new;
                stats.cost = cost_new;
                v = 2;

                rebuild = true;
            } else {
                stats.invalid_steps++;
                if (stats.lambda == opts.max_lambda) {
                    break;
                }

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

        if (callback != nullptr) {
            callback(stats);
        }

        return stats;
    }

   private:
    void BuildNormalEquations(const Parameters& params) {
        JtJ.setZero();
        Jtr.setZero();

        std::atomic<size_t> num_valid_residuals = 0;

        const size_t n = problem.NumResiduals();

        // TODO: Make max_allowed_parallelism customizable
        tbb::global_control tbb_global_control(
            tbb::global_control::max_allowed_parallelism, 8);

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, n, 32),
            [&](const tbb::blocked_range<size_t>& range) {
                JacData& data = jac_data.local();
                data.JtJ.setZero();
                data.Jtr.setZero();
                size_t local_valid_residuals = 0;

                for (size_t idx = range.begin(); idx != range.end(); idx++) {
                    const Float weight = problem.Weight(idx);
                    if (weight == 0.0) {
                        continue;
                    }

                    data.J.setZero();
                    RowMajorMatrixf<Problem::kResidualLength, 1> res;

                    const bool ok =
                        problem.EvaluateWithJacobian(params, idx, data.J, res);
                    if (ok) {
                        const Float r_squared = res.squaredNorm();
                        const Float total_weight =
                            weight * loss_fn.Weight(r_squared);

                        data.JtJ.template selfadjointView<Eigen::Lower>()
                            .rankUpdate(data.J.transpose(), total_weight);
                        data.Jtr += data.J.transpose() * (total_weight * res);

                        local_valid_residuals++;
                    }
                }

                num_valid_residuals.fetch_add(local_valid_residuals,
                                              std::memory_order_relaxed);

                // Accumulate JtJ and Jtr
                for (int col = 0; col < data.JtJ.cols(); col++) {
                    for (int row = col; row < data.JtJ.rows(); row++) {
                        std::atomic_ref<Float> elem{JtJ(row, col)};
                        elem.fetch_add(data.JtJ(row, col),
                                       std::memory_order_relaxed);
                    }

                    std::atomic_ref<Float> elem{Jtr(col, 0)};
                    elem.fetch_add(data.Jtr(col, 0), std::memory_order_relaxed);
                }
            });

        if constexpr (Problem::kShouldNormalize) {
            if (num_valid_residuals > 0) {
                JtJ /= num_valid_residuals;
                Jtr /= num_valid_residuals;
            }
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

        step = -linear_solver.solve(Jtr);
        return linear_solver.info();
    }

    Float TotalCost(const Parameters& params) {
        const size_t n = problem.NumResiduals();
        std::atomic<Float> cost = 0.0f;

        std::atomic<size_t> num_valid_residuals = 0;

        // TODO: Make max_allowed_parallelism customizable
        tbb::global_control tbb_global_control(
            tbb::global_control::max_allowed_parallelism, 8);

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, n, 32),
            [&](const tbb::blocked_range<size_t>& range) {
                size_t local_valid_residuals = 0;
                Float local_cost = 0.0f;

                for (size_t idx = range.begin(); idx != range.end(); idx++) {
                    const Float weight = problem.Weight(idx);
                    if (weight == 0.0f) {
                        continue;
                    }
                    const auto res = problem.Evaluate(params, idx);
                    if (res.has_value()) {
                        local_cost += weight * loss_fn.Loss(res->squaredNorm());
                        local_valid_residuals++;
                    }
                }

                num_valid_residuals.fetch_add(local_valid_residuals,
                                              std::memory_order_relaxed);
                cost.fetch_add(local_cost, std::memory_order_relaxed);
            });

        if constexpr (Problem::kShouldNormalize) {
            if (num_valid_residuals > 0) {
                cost = cost / num_valid_residuals;
            }
        }

        return cost;
    }

   private:
    Problem& problem;
    Parameters params_new;

    const LossFunction& loss_fn;
    Solver linear_solver;
    IterationCallback callback;

    const BundleOptions& opts;

    BundleStats stats;

    RowMajorMatrixf<Problem::kNumParams, Problem::kNumParams> JtJ;
    RowMajorMatrixf<Problem::kNumParams, 1> JtJ_diag;
    RowMajorMatrixf<Problem::kNumParams, 1> Jtr;
    RowMajorMatrixf<Problem::kNumParams, 1> step;

    struct JacData {
        RowMajorMatrixf<Problem::kResidualLength, Problem::kNumParams> J;
        RowMajorMatrixf<Problem::kNumParams, Problem::kNumParams> JtJ;
        RowMajorMatrixf<Problem::kNumParams, 1> Jtr;
    };
    tbb::enumerable_thread_specific<JacData> jac_data;
};

template <typename Problem, typename LossFunction>
class LevMarqSparseSolver {
    using Parameters = typename Problem::Parameters;
    using Solver = Eigen::SimplicialLLT<
        typename Eigen::SelfAdjointView<Eigen::SparseMatrix<Float>,
                                        Eigen::Lower>::PlainObject,
        Eigen::Lower>;

   public:
    LevMarqSparseSolver(Problem& problem, const LossFunction& loss_fn,
                        const BundleOptions& opts, IterationCallback callback)
        : problem(problem), loss_fn(loss_fn), opts(opts), callback(callback) {
        const size_t param_block_length = problem.ParamBlockLength();
        const size_t num_param_blocks = problem.NumParamBlocks();
        const size_t num_params = num_param_blocks * param_block_length;

        CHECK_EQ(num_params, problem.NumParams());

        jac_pair_data = tbb::enumerable_thread_specific<JacPairData>{[=]() {
            JacPairData data;
            data.J_pair.resize(Problem::kResidualLength,
                               2 * param_block_length);
            data.JtJ_pair.resize(2 * param_block_length,
                                 2 * param_block_length);
            data.Jtr_pair.resize(2 * param_block_length, 1);

            return data;
        }};

        JtJ.resize(num_params, num_params);
        Jtr.resize(num_params, 1);
        JtJ_diag.resize(num_params, 1);
        step.resize(num_params, 1);

        SPDLOG_INFO("Creating JtJ pattern...");

        struct pair_hash {
            std::size_t operator()(const std::pair<int, int>& v) const {
                return v.first * 31 + v.second;
            }
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

            unique_edges.insert({b1, b2});
        }

        std::vector<Eigen::Triplet<float>> triplets;
        triplets.reserve(unique_edges.size() * param_block_length *
                             param_block_length +
                         num_param_blocks * param_block_length *
                             (param_block_length + 1) / 2);

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
        linear_solver.analyzePattern(
            JtJ.template selfadjointView<Eigen::Lower>());
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

        params_new = *params;

        Float v = 2.0f;
        bool rebuild = true;
        for (stats.iterations = 0; stats.iterations < opts.max_iterations;
             ++stats.iterations) {
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
                if (stats.lambda == opts.max_lambda) {
                    break;
                }

                stats.lambda = std::min(opts.max_lambda, stats.lambda * v);
                v = 2 * v;

                rebuild = false;
                continue;
            }

            stats.step_norm = step.norm();
            if (stats.step_norm < opts.step_tol) {
                break;
            }

            problem.Step(*params, step, params_new);
            const Float cost_new = TotalCost(params_new);

            if (cost_new < stats.cost) {
                // See: http://www2.imm.dtu.dk/pubdb/edoc/imm3215.pdf
                const Float actual_cost_change = cost_new - stats.cost;
                const Float expected_cost_change =
                    step.transpose() *
                    (2.0 * Jtr +
                     JtJ.template selfadjointView<Eigen::Lower>() * step);

                const Float rho = actual_cost_change / expected_cost_change;

                // Mathematically, this should always be the case, but
                // numerically it might happen when the condition number of JtJ
                // is high due to floating point precision errors.
                if (rho > 0) {
                    const Float factor =
                        std::max(1.0 / 3.0, 1.0 - pow(2.0 * rho - 1.0, 3));
                    stats.lambda = std::clamp(stats.lambda * factor,
                                              opts.min_lambda, opts.max_lambda);
                }

                std::swap(*params, params_new);
                stats.cost = cost_new;
                v = 2;

                rebuild = true;
            } else {
                stats.invalid_steps++;
                if (stats.lambda == opts.max_lambda) {
                    break;
                }

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

        if (callback != nullptr) {
            callback(stats);
        }

        return stats;
    }

   private:
    void AccumBlockInSparseMatrix(Eigen::SparseMatrix<Float>& sparse_mat,
                                  Eigen::Ref<Eigen::MatrixX<Float>> block,
                                  int row, int col) {
        Float* values = sparse_mat.valuePtr();
        const int* inner_indices = sparse_mat.innerIndexPtr();
        const int* outer_starts = sparse_mat.outerIndexPtr();

        // Delta is the distance from the last element in the column, to the
        // element in the column at the row that we want (+ 1 because it's
        // actually from the first element in the next column).
        //
        // So delta is negative. This assumes that sparse_mat consists of
        // blocks, so delta should be the same.
        //
        // I can probably cache this value in an array of size NumParamBlocks x
        // NumParamBlocks
        const int* first_ptr =
            std::lower_bound(inner_indices + outer_starts[col],
                             inner_indices + outer_starts[col + 1], row);

        // Check that the found row is less than row + nRows
        CHECK_LT(*first_ptr, row + block.rows());

        const int delta =
            std::distance(inner_indices + outer_starts[col + 1], first_ptr);

        CHECK_LE(delta, 0);

        for (int src_col = 0; src_col < block.cols(); ++src_col) {
            const int target_col = col + src_col;

            // Index of the first element is:
            // - If block is not at the diagonal: column + delta
            // - Block is at the diagonal:        column + delta + src_col.
            //   the extra offset is to stay in the lower part of the matrix.
            const int first_idx = outer_starts[target_col + 1] + delta +
                                  ((row == col) ? src_col : 0);
            const int first_target_row = inner_indices[first_idx];
            Float* row_ptr = values + first_idx;

            // Check that we're within target_col and didn't go over to the next
            // column.
            CHECK_LT(first_idx, outer_starts[target_col + 1]);
            // Check that we're in the lower half.
            CHECK_GE(first_target_row, target_col);
            // Check that we're within the target block [row, row + nRows)
            CHECK_GE(first_target_row, row);

            // Process all rows in the block range that exist in this column
            for (int src_row = first_target_row - row; src_row < block.rows();
                 src_row++) {
                std::atomic_ref<Float> element_atomic{*row_ptr};

                element_atomic.fetch_add(block(src_row, src_col),
                                         std::memory_order_relaxed);
                row_ptr++;
            }
        }
    }

    void BuildNormalEquations(const Parameters& params) {
        std::fill(JtJ.valuePtr(), JtJ.valuePtr() + JtJ.nonZeros(), 0.0);
        Jtr.setZero();

        const size_t num_edges = problem.NumEdges();

        // TODO: Make max_allowed_parallelism customizable
        tbb::global_control tbb_global_control(
            tbb::global_control::max_allowed_parallelism, 8);

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, num_edges),
            [&](const tbb::blocked_range<size_t>& range) {
                JacPairData& data = jac_pair_data.local();

                for (size_t edge_idx = range.begin(); edge_idx != range.end();
                     edge_idx++) {
                    const float edge_weight = problem.EdgeWeight(edge_idx);
                    if (edge_weight == 0.0f) {
                        continue;
                    }

                    data.JtJ_pair.setZero();
                    data.Jtr_pair.setZero();

                    const size_t num_residuals = problem.NumResiduals(edge_idx);
                    size_t num_valid_residuals = 0;

                    for (size_t res_idx = 0; res_idx < num_residuals;
                         res_idx++) {
                        data.J_pair.setZero();
                        RowMajorMatrixf<Problem::kResidualLength, 1> res;

                        const bool ok = problem.EvaluateWithJacobian(
                            params, edge_idx, res_idx, data.J_pair, res);
                        if (ok) {
                            const Float r_squared = res.squaredNorm();
                            const Float res_weight =
                                problem.ResidualWeight(edge_idx, res_idx);
                            const Float weight = edge_weight * res_weight *
                                                 loss_fn.Weight(r_squared);

                            data.JtJ_pair +=
                                data.J_pair.transpose() * data.J_pair * weight;

                            data.Jtr_pair +=
                                data.J_pair.transpose() * (weight * res);

                            num_valid_residuals++;
                        }
                    }

                    if constexpr (Problem::kShouldNormalize) {
                        if (num_valid_residuals > 0) {
                            data.JtJ_pair /= num_valid_residuals;
                            data.Jtr_pair /= num_valid_residuals;
                        }
                    }

                    const auto [block_1, block_2] = problem.GetEdge(edge_idx);
                    CHECK_NE(block_1, block_2);

                    const int block_length = problem.ParamBlockLength();
                    const int block_1_idx = block_1 * block_length;
                    const int block_2_idx = block_2 * block_length;

                    /* JtJ_pair consists of four blocks:
                     *  | J1.transpose() * J1         J1.transpose() * J2 |
                     *  | J2.transpose() * J1         J2.transpose() * J2 |
                     */
                    AccumBlockInSparseMatrix(
                        JtJ,
                        data.JtJ_pair.block(0, 0, block_length, block_length),
                        block_1_idx, block_1_idx);

                    AccumBlockInSparseMatrix(
                        JtJ,
                        data.JtJ_pair.block(block_length, block_length,
                                            block_length, block_length),
                        block_2_idx, block_2_idx);

                    if (block_1_idx > block_2_idx) {
                        AccumBlockInSparseMatrix(
                            JtJ,
                            data.JtJ_pair.block(0, block_length, block_length,
                                                block_length),
                            block_1_idx, block_2_idx);
                    } else {
                        AccumBlockInSparseMatrix(
                            JtJ,
                            data.JtJ_pair.block(block_length, 0, block_length,
                                                block_length),
                            block_2_idx, block_1_idx);
                    }

                    // Accum Jtr for first parameter block.
                    // Equivalent to:
                    //  Jtr.block(block_1_idx, 0, block_length, 1) +=
                    //    data.Jtr_pair.block(0, 0, block_length, 1);
                    for (int i = 0; i < block_length; ++i) {
                        std::atomic_ref<Float> target(Jtr(block_1_idx + i, 0));
                        target.fetch_add(data.Jtr_pair(i, 0),
                                         std::memory_order_relaxed);
                    }

                    // Accum Jtr for second parameter block parameter block.
                    // Equivalent to:
                    //  Jtr.block(block_2_idx, 0, block_length, 1) +=
                    //    data.Jtr_pair.block(block_length, 0, block_length, 1);
                    for (int i = 0; i < block_length; ++i) {
                        std::atomic_ref<Float> target(Jtr(block_2_idx + i, 0));
                        target.fetch_add(data.Jtr_pair(block_length + i, 0),
                                         std::memory_order_relaxed);
                    }
                }
            });

        JtJ_diag = JtJ.diagonal().cwiseMax(1e-6).cwiseMin(1e32);
    }

    Float TotalCost(const Parameters& params) {
        const size_t num_edges = problem.NumEdges();
        std::atomic<Float> cost = 0.0f;

        // TODO: Make max_allowed_parallelism customizable
        tbb::global_control tbb_global_control(
            tbb::global_control::max_allowed_parallelism, 8);

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, num_edges),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t edge_idx = range.begin(); edge_idx != range.end();
                     edge_idx++) {
                    const Float edge_weight = problem.EdgeWeight(edge_idx);
                    if (edge_weight == 0.0f) {
                        continue;
                    }

                    Float edge_cost = 0.0f;

                    const size_t num_residuals = problem.NumResiduals(edge_idx);
                    size_t num_valid_residuals = 0;

                    for (size_t res_idx = 0; res_idx < num_residuals;
                         res_idx++) {
                        auto maybe_res =
                            problem.Evaluate(params, edge_idx, res_idx);
                        if (maybe_res.has_value()) {
                            const auto& res = *maybe_res;

                            const Float res_weight =
                                problem.ResidualWeight(edge_idx, res_idx);
                            edge_cost +=
                                res_weight * loss_fn.Loss(res.squaredNorm());

                            num_valid_residuals++;
                        }
                    }

                    if constexpr (Problem::kShouldNormalize) {
                        if (num_valid_residuals > 0) {
                            edge_cost /= num_valid_residuals;
                        }
                    }

                    cost.fetch_add(edge_weight * edge_cost,
                                   std::memory_order_relaxed);
                }
            });

        return cost;
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

        step = -linear_solver.solve(Jtr);
        return linear_solver.info();
    }

   private:
    Problem& problem;
    Parameters params_new;

    const LossFunction& loss_fn;
    const BundleOptions& opts;
    IterationCallback callback;

    Solver linear_solver;

    Eigen::SparseMatrix<Float> JtJ;
    RowMajorMatrixf<Eigen::Dynamic, 1> JtJ_diag;
    RowMajorMatrixf<Eigen::Dynamic, 1> Jtr;
    RowMajorMatrixf<Eigen::Dynamic, 1> step;

    struct JacPairData {
        RowMajorMatrixf<Problem::kResidualLength, Eigen::Dynamic> J_pair;
        // Eigen::SparseMatrix is in compressed sparse column format, which
        // makes it easier in AccumBlockInSparseMatrix to traverse the block
        // as if it is column major. If JtJ_pair is row major, this would be
        // cache infriendly. Hence JtJ_pair is column major.
        //
        // Maybe deciding to use row major everywhere is not such a good idea.
        Eigen::Matrix<Float, Eigen::Dynamic, Eigen::Dynamic> JtJ_pair;
        RowMajorMatrixf<Eigen::Dynamic, 1> Jtr_pair;
    };
    tbb::enumerable_thread_specific<JacPairData> jac_pair_data;

    BundleStats stats;
};

template <typename Problem, typename LossFunction,
          typename Parameters = typename Problem::Parameters>
BundleStats LevMarqDenseSolve(Problem& problem, const LossFunction& loss_fn,
                              Parameters* params, const BundleOptions& opts,
                              IterationCallback callback = nullptr) {
    LevMarqDenseSolver solver(problem, loss_fn, opts, callback);
    return solver.Solve(params);
}

template <typename Problem, typename LossFunction,
          typename Parameters = typename Problem::Parameters>
BundleStats LevMarqSparseSolve(Problem& problem, const LossFunction& loss_fn,
                               Parameters* params, const BundleOptions& opts,
                               IterationCallback callback = nullptr) {
    LevMarqSparseSolver solver(problem, loss_fn, opts, callback);
    return solver.Solve(params);
}
