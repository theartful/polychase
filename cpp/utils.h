// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Ahmed Essam <aessam.dahy@gmail.com>

#pragma once

#include <fmt/format.h>

#define FMT_FORMAT(f, ...) fmt::format(FMT_STRING(f) __VA_OPT__(, ) __VA_ARGS__)

#define STRINGIFY(x) #x

#define CHECK_EXPR_STR(expr, expr_str, ...)                                 \
    do {                                                                    \
        if (!static_cast<bool>(expr)) {                                     \
            __VA_OPT__(auto message = FMT_FORMAT(__VA_ARGS__);)             \
            ::asserts::assert_fail(expr_str, __FILE__, __LINE__,            \
                                   __func__ __VA_OPT__(, message.c_str())); \
        }                                                                   \
    } while (0)

// clang-format off
#define CHECK_OP(expr1, op, expr2)                                        \
    do {                                                                  \
        auto&& v1__ = expr1;                                              \
        auto&& v2__ = expr2;                                              \
        const bool result__ = static_cast<bool>((expr1)op(expr2));        \
        CHECK_EXPR_STR(result__, STRINGIFY((expr1) op (expr2)),           \
                       "({}) = {} while ({}) = {}", #expr1, v1__, #expr2, \
                       v2__);                                             \
    } while (0)
// clang-format on

#define CHECK(expr, ...) CHECK_EXPR_STR(expr, #expr __VA_OPT__(, ) __VA_ARGS__)
#define CHECK_EQ(expr1, expr2) CHECK_OP(expr1, ==, expr2)
#define CHECK_NE(expr1, expr2) CHECK_OP(expr1, !=, expr2)
#define CHECK_GT(expr1, expr2) CHECK_OP(expr1, >, expr2)
#define CHECK_LT(expr1, expr2) CHECK_OP(expr1, <, expr2)
#define CHECK_GE(expr1, expr2) CHECK_OP(expr1, >=, expr2)
#define CHECK_LE(expr1, expr2) CHECK_OP(expr1, <=, expr2)

namespace asserts {
[[noreturn]] void assert_fail(const char* assertion, const char* file,
                              unsigned int line, const char* function,
                              const char* message = nullptr);
}  // namespace asserts
