#include "utils.h"

#include <fmt/format.h>

#include <stdexcept>

namespace asserts {
void assert_fail(const char* assertion, const char* file, unsigned int line,
                 const char* function, const char* message) {
    if (message) {
        throw std::logic_error(
            fmt::format(FMT_STRING("[{}:{} {}] Assertion failed: {} -- {}"),
                        file, line, function, assertion, message));
    } else {
        throw std::logic_error(
            fmt::format(FMT_STRING("[{}:{} {}] Assertion failed: {}"), file,
                        line, function, assertion));
    }
}
}  // namespace asserts
