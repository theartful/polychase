#include <Eigen/Dense>
#include <iomanip>
#include <iostream>

int main() {
    using Mat9f = Eigen::Matrix<float, 9, 9, Eigen::RowMajor>;
    using Vec9f = Eigen::Matrix<float, 9, 1>;
    using Solver = Eigen::LLT<
        typename Eigen::SelfAdjointView<Mat9f, Eigen::Lower>::PlainObject,
        Eigen::Lower>;

    std::cout << std::setprecision(9);

    // These exact values were found by trying to track a geometry in an
    // example, and inspecting LevMarqDenseSolver
    const float lambda = 1.5607382e-06f;
    // clang-format off
    const Mat9f JtJ = (Mat9f() <<
        557551.4375,        0.0,                0.0,                0.0,                0.0,                0.0,                0.0,                0.0,                0.0,
        296441.21875,       657639.8125,        0.0,                0.0,                0.0,                0.0,                0.0,                0.0,                0.0,
        -4293.4072265625,   -5085.32958984375,  364752.1875,        0.0,                0.0,                0.0,                0.0,                0.0,                0.0,
        42399.52734375,     131392.296875,      31440.83984375,     70597.6328125,      0.0,                0.0,                0.0,                0.0,                0.0,
        -27725.1328125,     44876.76953125,     -105931.8828125,    0.0,                70597.6328125,      0.0,                0.0,                0.0,                0.0,
        -43429.875,         -83350.875,         -62037.90625,       -55166.17578125,    25584.125,          52518.796875,       0.0,                0.0,                0.0,
        1993.02294921875,   3831.88916015625,   2867.069091796875,  2574.660400390625,  -1193.505981445312, -2450.312255859375, 114.358093261719,   0.0,                0.0,
        1947.6396484375,    6048.806640625,     1454.197631835938,  3295.457763671875,  0.0,                -2574.660400390625, 120.201538085938,   153.880432128906,   0.0,
        -1262.969848632812, 2073.468505859375,  -4891.965820312500, 0.0,                3295.457763671875,  1193.505981445312,  -55.693786621094,   0.0,                153.880432128906
    ).finished();

    const Vec9f Jtr = (Vec9f() <<
        -2.338238716125,
        -4.207848548889,
        3.598472595215,
        -1.105026721954,
        -1.491069078445,
        0.368796110153,
        -0.017316624522,
        -0.051174595952,
        -0.068564474583
    ).finished();
    // clang-format on

    const Mat9f JtJ_damp = JtJ + lambda * Mat9f::Identity();

    const Solver linear_solver(
        JtJ_damp.template selfadjointView<Eigen::Lower>());

    // Solve the linear system:  (J^T J + lambda I) step = -J^T r
    const Vec9f step = -linear_solver.solve(Jtr);

    // Compute residual norm: || (J^T J + lambda I) step + J^T r ||
    const float residual =
        (JtJ_damp.template selfadjointView<Eigen::Lower>() * step + Jtr).norm();

    const float expected_cost_change =
        step.transpose() *
        (2.0 * Jtr + JtJ.template selfadjointView<Eigen::Lower>() * step);

    std::cout << std::setprecision(9);
    // Should be cloes to 0, but instead we get 0.0028946274
    std::cout << "Residual norm = \t" << residual << "\n";
    // Should be negative, but instead we get 0.000244110823
    std::cout << "Expected cost change = \t" << expected_cost_change << "\n";

    return 0;
}
