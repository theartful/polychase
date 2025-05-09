#include <Eigen/Core>
#include <iostream>
#include <random>

#include "jacobian.h"
#include "pnp/lm_impl.h"
#include "pnp/robust_loss.h"
#include "ray_casting.h"

Eigen::Vector3f RandomDir() { return Eigen::Vector3f::Random().normalized(); }

Ray RandomRay() {
    return Ray{
        .origin = Eigen::Vector3f::Zero(),
        .dir = RandomDir(),
    };
}

Plane RandomPlane() {
    return Plane{
        .point = Eigen::Vector3f::Random(),
        .normal = RandomDir(),
    };
}

Triangle RandomTriangle() {
    return Triangle{
        .p1 = Eigen::Vector3f::Random(),
        .p2 = Eigen::Vector3f::Random(),
        .p3 = Eigen::Vector3f::Random(),
    };
}

Eigen::Vector3f SamplePoint(const Triangle& triangle) {
    using Dist = std::uniform_real_distribution<float>;

    Dist dist;
    std::random_device rd;

    const float u = dist(rd, Dist::param_type{0.0f, 0.99f});
    const float v = dist(rd, Dist::param_type{0.0f, 0.99f - u});

    CHECK_LT(u + v, 1.0f);

    return triangle.Barycentric(u, v);
}

RowMajorMatrix3f skew(const Eigen::Vector3f& v) {
    // clang-format off
        return (RowMajorMatrix3f() <<
            0,     -v(2), v(1),
            v(2),  0,     -v(0),
            -v(1), v(0),  0
        ).finished();
    // clang-format on
}

void TestIntersectPlaneWithJac() {
    std::cout << "\n\nTestIntersectPlaneWithJac\n";
    auto ray = RandomRay();
    auto plane = RandomPlane();

    RowMajorMatrixf<3, 3> jac_o;
    RowMajorMatrixf<3, 3> jac_d;
    Eigen::Vector3f inter;

    const bool ok = IntersectWithJac(ray, plane, &inter, &jac_o, &jac_d);
    CHECK(ok);

    std::cout << "ERROR = " << (plane.point - inter).dot(plane.normal) << '\n';
    std::cout << "JACOBIAN_O = \n" << jac_o << '\n';
    std::cout << "JACOBIAN_d = \n" << jac_d << '\n';

    RowMajorMatrixf<3, 3> numerical_jac_o;
    RowMajorMatrixf<3, 3> numerical_jac_d;
    for (int i = 0; i < 3; i++) {
        const float epsilon = 1e-5;
        auto ray_copy = ray;
        ray_copy.origin(i) += epsilon;

        std::optional<Eigen::Vector3f> maybe_inter2 = Intersect(ray_copy, plane);
        CHECK(maybe_inter2.has_value());

        Eigen::Vector3f inter2 = *maybe_inter2;

        numerical_jac_o.block<3, 1>(0, i) = (inter2 - inter) / epsilon;

        ray_copy = ray;
        ray_copy.dir(i) += epsilon;

        maybe_inter2 = Intersect(ray_copy, plane);
        CHECK(maybe_inter2.has_value());
        inter2 = *maybe_inter2;

        numerical_jac_d.block<3, 1>(0, i) = (inter2 - inter) / epsilon;
    }

    std::cout << "NUMERICAL_JAC_O = \n" << numerical_jac_o << '\n';
    std::cout << "NUMERICAL_JAC_d = \n" << numerical_jac_d << '\n';
    std::cout << "JACOBIAN ERROR O = " << (jac_o - numerical_jac_o).norm() << '\n';
    std::cout << "JACOBIAN ERROR d = " << (jac_d - numerical_jac_d).norm() << '\n';
}

void TestIntersectTriangleWithJac() {
    std::cout << "\n\nTestIntersectTriangleWithJac\n";
    const Triangle tri = RandomTriangle();
    const Eigen::Vector3f o = Eigen::Vector3f::Random();
    const Eigen::Vector3f p = SamplePoint(tri);

    const Ray ray = Ray{.origin = o, .dir = (p - o)};

    RowMajorMatrixf<3, 3> jac_o;
    RowMajorMatrixf<3, 3> jac_d;
    Eigen::Vector3f inter;

    const bool ok = IntersectWithJac(ray, tri, &inter, &jac_o, &jac_d);
    CHECK(ok);

    // const Eigen::Vector3f plane_normal = (tri.p2 - tri.p1).cross(tri.p3 - tri.p1).normalized();
    std::cout << "ERROR = " << (inter - p).norm() << '\n';
    std::cout << "JACOBIAN_O = \n" << jac_o << '\n';
    std::cout << "JACOBIAN_d = \n" << jac_d << '\n';

    RowMajorMatrixf<3, 3> numerical_jac_o;
    RowMajorMatrixf<3, 3> numerical_jac_d;
    for (int i = 0; i < 3; i++) {
        const float epsilon = 1e-5;
        auto ray_copy = ray;
        ray_copy.origin(i) += epsilon;

        std::optional<Eigen::Vector3f> maybe_inter2 = Intersect(ray_copy, tri);
        CHECK(maybe_inter2.has_value());

        Eigen::Vector3f inter2 = *maybe_inter2;

        numerical_jac_o.block<3, 1>(0, i) = (inter2 - inter) / epsilon;

        ray_copy = ray;
        ray_copy.dir(i) += epsilon;

        maybe_inter2 = Intersect(ray_copy, tri);
        CHECK(maybe_inter2.has_value());
        inter2 = *maybe_inter2;

        numerical_jac_d.block<3, 1>(0, i) = (inter2 - inter) / epsilon;
    }

    std::cout << "NUMERICAL_JAC_O = \n" << numerical_jac_o << '\n';
    std::cout << "NUMERICAL_JAC_d = \n" << numerical_jac_d << '\n';
    std::cout << "JACOBIAN ERROR O = " << (jac_o - numerical_jac_o).norm() << '\n';
    std::cout << "JACOBIAN ERROR d = " << (jac_d - numerical_jac_d).norm() << '\n';
}

void TestCenterWithJac() {
    std::cout << "\n\nTestCenterWithJac\n";
    // Generate random pose
    Eigen::Quaternionf quat = Eigen::Quaternionf::UnitRandom();
    CameraPose pose(RowMajorMatrix3f(quat.toRotationMatrix()), Eigen::Vector3f::Random());

    Eigen::Vector3f center;
    RowMajorMatrixf<3, 3> jac_R, jac_t;

    pose.CenterWithJac(&center, &jac_R, &jac_t);
    std::cout << "JACOBIAN_R = \n" << jac_R << '\n';
    std::cout << "JACOBIAN_t = \n" << jac_t << '\n';

    RowMajorMatrixf<3, 3> numerical_jac_R;
    RowMajorMatrixf<3, 3> numerical_jac_t;
    for (int i = 0; i < 3; i++) {
        const float epsilon = 1e-4;
        CameraPose pose_copy = pose;

        Eigen::Vector3f w_delta = Eigen::Vector3f::Zero();
        w_delta(i) += epsilon;

        pose_copy.q = quat_step_post(pose_copy.q, w_delta);
        Eigen::Vector3f center2 = pose_copy.Center();

        numerical_jac_R.block<3, 1>(0, i) = (center2 - center) / epsilon;

        pose_copy = pose;
        pose_copy.t(i) += epsilon;
        center2 = pose_copy.Center();

        numerical_jac_t.block<3, 1>(0, i) = (center2 - center) / epsilon;
    }

    std::cout << "NUMERICAL_JAC_R = \n" << numerical_jac_R << '\n';
    std::cout << "NUMERICAL_JAC_t = \n" << numerical_jac_t << '\n';
    std::cout << "JACOBIAN ERROR R = " << (jac_R - numerical_jac_R).norm() << '\n';
    std::cout << "JACOBIAN ERROR t = " << (jac_t - numerical_jac_t).norm() << '\n';
}

void TestDifferentPnPExtrinsics() {
    std::cout << "\n\nTestDifferentPnPExtrinsics\n";
    CameraIntrinsics intrin = {
        .fx = 1920.0f * 1.2f,
        .fy = 1920.0f * 1.2f,
        .cx = 1920.0f / 2.0f,
        .cy = 1080.0f / 2.0f,
        .aspect_ratio = 1.0f,
        .width = 1920.0f,
        .height = 1080.0f,
        .convention = CameraConvention::OpenCV,
    };

    Eigen::Quaternionf quat = Eigen::Quaternionf::UnitRandom();
    Eigen::AngleAxisf target_diff(10.0 * M_PI / 180.0, RandomDir());

    RowMajorMatrix3f rot_src = quat.toRotationMatrix();
    RowMajorMatrix3f rot_tgt = rot_src * target_diff;

    Eigen::Vector3f t_src = Eigen::Vector3f::Random();
    Eigen::Vector3f t_tgt = Eigen::Vector3f::Random();

    const CameraState camera_tgt{
        .intrinsics = intrin,
        .pose = CameraPose(rot_tgt, t_tgt),
    };

    const CameraState camera_src_gt{
        .intrinsics = intrin,
        .pose = CameraPose(rot_src, t_src),
    };

    CameraState camera_src = camera_tgt;

    constexpr size_t num_points = 100;
    RowMajorMatrixX2f points2D_src(num_points, 2);
    RowMajorMatrixX2f points2D_tgt(num_points, 2);
    std::vector<Plane> planes;

    std::uniform_real_distribution<float> dist(1.0, 10.0);
    std::random_device prng;

    for (size_t i = 0; i < num_points; i++) {
        const Eigen::Vector2f p_0_1 = (Eigen::Vector2f::Random() + Eigen::Vector2f(1.0f, 1.0f)) / 2.0f;
        const Eigen::Vector2f p_tgt = (p_0_1.array() * Eigen::Vector2f(intrin.width, intrin.height).array()).matrix();

        const Eigen::Vector3f p_tgt_dir = intrin.Unproject(p_tgt);

        p_tgt_dir.normalized();
        float depth = dist(prng);

        const Eigen::Vector3f point_worldspace = camera_tgt.pose.Inverse().Apply(p_tgt_dir * depth);
        const Eigen::Vector2f p_src = camera_src_gt.intrinsics.Project(camera_src_gt.pose.Apply(point_worldspace));

        points2D_src.row(i) = p_src;
        points2D_tgt.row(i) = p_tgt;
        planes.push_back(Plane{
            .point = point_worldspace,
            .normal =
                Eigen::AngleAxisf(10.0 * M_PI / 180.0, RandomDir()) * (camera_tgt.pose.Center() - point_worldspace),
        });

        // std::cout << "P_TGT = " << p_tgt.x() << " " << p_tgt.y() << "\t P_SRC = " << p_src.x() << " " << p_src.y()
        //           << '\n';
    }

    TrivialLoss loss_fn;
    DifferentPnPJacobianAccumulator<TrivialLoss> accum(points2D_src, points2D_tgt, camera_tgt, planes, false, false,
                                                       loss_fn, poselib::UniformWeightVector());

    BundleOptions bundle_opts;
    auto bundle_stats = lm_impl<decltype(accum)>(accum, &camera_src, bundle_opts, nullptr);

    std::cout << "BUNDLE_STATS INITIAL COST = " << bundle_stats.initial_cost << '\n';
    std::cout << "BUNDLE_STATS COST = " << bundle_stats.cost << '\n';
    std::cout << "BUNDLE_STATS ITERS = " << bundle_stats.iterations << '\n';
    std::cout << "BUNDLE_STATS INVALID = " << bundle_stats.invalid_steps << '\n';
}

void TestDifferentPnPIntrinsics() {
    std::cout << "\n\nTestDifferentPnPIntrinsics\n";
    CameraIntrinsics intrin_tgt = {
        .fx = 1920.0f * 1.2f,
        .fy = 1920.0f * 1.2f,
        .cx = 1920.0f / 2.0f,
        .cy = 1080.0f / 2.0f,
        .aspect_ratio = 1.0f,
        .width = 1920.0f,
        .height = 1080.0f,
        .convention = CameraConvention::OpenCV,
    };

    CameraIntrinsics intrin_src = {
        .fx = 1920.0f * 2.0f,
        .fy = 1920.0f * 2.0f,
        .cx = 1920.0f / 4.0f,
        .cy = 1080.0f / 4.0f,
        .aspect_ratio = 1.0f,
        .width = 1920.0f,
        .height = 1080.0f,
        .convention = CameraConvention::OpenCV,
    };

    CameraPose pose(RowMajorMatrix3f(Eigen::Quaternionf::UnitRandom().toRotationMatrix()), Eigen::Vector3f::Random());

    const CameraState camera_tgt{
        .intrinsics = intrin_tgt,
        .pose = pose,
    };

    const CameraState camera_src_gt{
        .intrinsics = intrin_src,
        .pose = pose,
    };

    CameraState camera_src = camera_tgt;

    constexpr size_t num_points = 100;
    RowMajorMatrixX2f points2D_src(num_points, 2);
    RowMajorMatrixX2f points2D_tgt(num_points, 2);
    std::vector<Plane> planes;

    std::uniform_real_distribution<float> dist(1.0, 10.0);
    std::mt19937 prng;

    for (size_t i = 0; i < num_points; i++) {
        const Eigen::Vector2f p_0_1 = (Eigen::Vector2f::Random() + Eigen::Vector2f(1.0f, 1.0f)) / 2.0f;
        const Eigen::Vector2f p_tgt =
            (p_0_1.array() * Eigen::Vector2f(intrin_tgt.width, intrin_tgt.height).array()).matrix();

        const Eigen::Vector3f p_tgt_dir = intrin_tgt.Unproject(p_tgt);

        p_tgt_dir.normalized();
        float depth = dist(prng);

        const Eigen::Vector3f point_worldspace = camera_tgt.pose.Inverse().Apply(p_tgt_dir * depth);
        const Eigen::Vector2f p_src = camera_src_gt.intrinsics.Project(camera_src_gt.pose.Apply(point_worldspace));

        points2D_src.row(i) = p_src;
        points2D_tgt.row(i) = p_tgt;
        planes.push_back(Plane{
            .point = point_worldspace,
            .normal =
                Eigen::AngleAxisf(10.0 * M_PI / 180.0, RandomDir()) * (camera_tgt.pose.Center() - point_worldspace),
        });
    }

    TrivialLoss loss_fn;
    DifferentPnPJacobianAccumulator<TrivialLoss> accum(points2D_src, points2D_tgt, camera_tgt, planes, true, true,
                                                       loss_fn, poselib::UniformWeightVector());

    std::cout << "ESTIMATED FX = " << camera_src.intrinsics.fx << '\n';
    std::cout << "ESTIMATED FY = " << camera_src.intrinsics.fy << '\n';
    std::cout << "ESTIMATED CX = " << camera_src.intrinsics.cx << '\n';
    std::cout << "ESTIMATED CY = " << camera_src.intrinsics.cy << '\n';

    BundleOptions bundle_opts;
    auto bundle_stats = lm_impl<decltype(accum)>(accum, &camera_src, bundle_opts, nullptr);

    std::cout << "BUNDLE_STATS INITIAL COST = " << bundle_stats.initial_cost << '\n';
    std::cout << "BUNDLE_STATS COST = " << bundle_stats.cost << '\n';
    std::cout << "BUNDLE_STATS ITERS = " << bundle_stats.iterations << '\n';
    std::cout << "BUNDLE_STATS INVALID = " << bundle_stats.invalid_steps << '\n';
    std::cout << "ESTIMATED FX = " << camera_src.intrinsics.fx << '\n';
    std::cout << "GT FX = " << camera_src_gt.intrinsics.fx << '\n';
    std::cout << "ESTIMATED FY = " << camera_src.intrinsics.fy << '\n';
    std::cout << "GT FY = " << camera_src_gt.intrinsics.fy << '\n';
    std::cout << "ESTIMATED CX = " << camera_src.intrinsics.cx << '\n';
    std::cout << "GT CX = " << camera_src_gt.intrinsics.cx << '\n';
    std::cout << "ESTIMATED CY = " << camera_src.intrinsics.cy << '\n';
    std::cout << "GT CY = " << camera_src_gt.intrinsics.cy << '\n';
}

int main() {
    srand(time(NULL));
    TestCenterWithJac();
    TestIntersectPlaneWithJac();
    TestIntersectTriangleWithJac();
    TestDifferentPnPExtrinsics();
    TestDifferentPnPIntrinsics();
}
