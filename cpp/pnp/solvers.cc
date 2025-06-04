#include "solvers.h"

#include "lev_marq.h"
#include "pnp_problem.h"
#include "robust_loss.h"
#include "utils.h"

template <typename LossFunction>
static inline void SolvePnPIterative(
    const RefConstRowMajorMatrixX3f &object_points,
    const RefConstRowMajorMatrixX2f &image_points,
    const RefConstArrayXf &weights, const LossFunction &loss_fn,
    const PnPOptions &opts, PnPResult &result) {
    PnPProblem problem(
        image_points, object_points, weights, opts.optimize_focal_length,
        opts.optimize_principal_point, result.camera.intrinsics.GetBounds());

    PnPProblem::Parameters params{
        .cam = result.camera,
        .R = result.camera.pose.R(),
    };

    result.bundle_stats =
        LevMarqDenseSolve(problem, loss_fn, &params, opts.bundle_opts);
    result.camera = params.cam;

    // Calculate inlier ratio
    size_t num_inliers = 0;

    if (opts.max_inlier_error > 0.0f) {
        const Float max_inlier_error_sq =
            opts.max_inlier_error * opts.max_inlier_error;
        for (size_t i = 0; i < problem.NumResiduals(); i++) {
            const auto maybe_result = problem.Evaluate(params, i);
            CHECK(maybe_result.has_value());

            const Float err_sq = maybe_result->squaredNorm();
            if (err_sq < max_inlier_error_sq) {
                num_inliers++;
            }
        }
    }
    result.inlier_ratio =
        static_cast<Float>(num_inliers) / problem.NumResiduals();
}

void SolvePnPIterative(const RefConstRowMajorMatrixX3f &object_points,
                       const RefConstRowMajorMatrixX2f &image_points,
                       const RefConstArrayXf &weights, const PnPOptions &opts,
                       PnPResult &result) {
    CHECK_EQ(object_points.rows(), image_points.rows());
    CHECK_GE(object_points.rows(), 3);

    switch (opts.bundle_opts.loss_type) {
#define SWITCH_LOSS_FUNCTION_CASE(LossFunction)                                \
    {                                                                          \
        LossFunction loss_fn(opts.bundle_opts.loss_scale);                     \
        SolvePnPIterative(object_points, image_points, weights, loss_fn, opts, \
                          result);                                             \
    }
        SWITCH_LOSS_FUNCTIONS
#undef SWITCH_LOSS_FUNCTION_CASE
        default:
            throw std::runtime_error(
                fmt::format("Unknown loss type: {}",
                            static_cast<int>(opts.bundle_opts.loss_type)));
    }
}

void SolvePnPIterative(const RefConstRowMajorMatrixX3f &object_points,
                       const RefConstRowMajorMatrixX2f &image_points,
                       const PnPOptions &opts, PnPResult &result) {
    ArrayXf weights;
    SolvePnPIterative(object_points, image_points, weights, opts, result);
}
