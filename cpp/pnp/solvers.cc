#include "solvers.h"

#include <PoseLib/robust.h>
#include <PoseLib/robust/jacobian_impl.h>

#include "lev_marq.h"
#include "pnp_problem.h"
#include "robust_loss.h"
#include "utils.h"

template <typename LossFunction>
static inline void SolvePnPIterative(const ConstRefRowMajorMatrixX3f &object_points,
                                     const ConstRefRowMajorMatrixX2f &image_points, const ConstRefArrayXf &weights,
                                     const LossFunction &loss_fn, const BundleOptions &bundle_opts,
                                     bool optimize_focal_length, bool optimize_principal_point, PnPResult &result) {
    PnPProblem problem(image_points, object_points, weights, optimize_focal_length, optimize_principal_point,
                       result.camera.intrinsics.GetBounds());

    PnPProblem::Parameters params{
        .cam = result.camera,
        .R = result.camera.pose.R(),
    };

    result.bundle_stats = LevMarqDenseSolve(problem, loss_fn, &params, bundle_opts);
    result.camera = params.cam;
}

void SolvePnPIterative(const ConstRefRowMajorMatrixX3f &object_points, const ConstRefRowMajorMatrixX2f &image_points,
                       const ConstRefArrayXf &weights, const BundleOptions &bundle_opts, bool optimize_focal_length,
                       bool optimize_principal_point, PnPResult &result) {
    CHECK_EQ(object_points.rows(), image_points.rows());
    CHECK_GE(object_points.rows(), 3);

    switch (bundle_opts.loss_type) {
#define SWITCH_LOSS_FUNCTION_CASE(LossFunction)                                                              \
    {                                                                                                        \
        LossFunction loss_fn(bundle_opts.loss_scale);                                                        \
        SolvePnPIterative(object_points, image_points, weights, loss_fn, bundle_opts, optimize_focal_length, \
                          optimize_principal_point, result);                                                 \
    }
        SWITCH_LOSS_FUNCTIONS
#undef SWITCH_LOSS_FUNCTION_CASE
        default:
            throw std::runtime_error(fmt::format("Unknown loss type: {}", static_cast<int>(bundle_opts.loss_type)));
    }
}

void SolvePnPIterative(const ConstRefRowMajorMatrixX3f &object_points, const ConstRefRowMajorMatrixX2f &image_points,
                       const BundleOptions &bundle_opts, bool optimize_focal_length, bool optimize_principal_point,
                       PnPResult &result) {
    ArrayXf weights{};
    SolvePnPIterative(object_points, image_points, weights, bundle_opts, optimize_focal_length,
                      optimize_principal_point, result);
}
