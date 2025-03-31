#include <iostream>
#include <Eigen/Core>

int main() {
    std::cout << "Hello, world!\n";

    std::cout << "sizeof(Eigen::Vector4f) = " << sizeof(Eigen::Vector4f) << '\n';
    std::cout << "expected sizeof(Eigen::Vector4f) = " << sizeof(float) * 4 << '\n';
}
