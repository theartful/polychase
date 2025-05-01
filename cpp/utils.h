#pragma once

#include <fmt/format.h>

#include <stdexcept>

#define CHECK(expr)                                                  \
    if (!static_cast<bool>(expr)) {                                  \
        ::asserts::assert_fail(#expr, __FILE__, __LINE__, __func__); \
    }

#define CHECK_EQ(expr1, expr2) CHECK((expr1) == (expr2))
#define CHECK_GT(expr1, expr2) CHECK((expr1) > (expr2))
#define CHECK_LT(expr1, expr2) CHECK((expr1) < (expr2))
#define CHECK_GE(expr1, expr2) CHECK((expr1) >= (expr2))
#define CHECK_LE(expr1, expr2) CHECK((expr1) <= (expr2))

namespace asserts {
[[noreturn]] static inline void assert_fail(const char* assertion, const char* file, unsigned int line,
                                            const char* function) {
    throw std::logic_error(fmt::format("[{}:{} {}] Assertion failed: {}", file, line, function, assertion));
}

}  // namespace asserts
