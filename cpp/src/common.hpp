#pragma once

#include <cstdarg>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace dtt
{

/** Every internal failure is thrown as this; the public API turns it into an error value. */
struct Failure : std::runtime_error {
    using std::runtime_error::runtime_error;
};

#if defined(__GNUC__)
__attribute__((format(printf, 1, 2)))
#endif
inline std::string format(const char* fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    va_list ap2;
    va_copy(ap2, ap);
    const int n = std::vsnprintf(nullptr, 0, fmt, ap);
    va_end(ap);
    std::string s(n > 0 ? static_cast<size_t>(n) : 0, '\0');
    if (n > 0) {
        std::vsnprintf(s.data(), s.size() + 1, fmt, ap2);
    }
    va_end(ap2);
    return s;
}

#define DTT_CHECK(cond, ...)                                                                                          \
    do {                                                                                                              \
        if (!(cond)) {                                                                                                \
            throw ::dtt::Failure(::dtt::format(__VA_ARGS__));                                                         \
        }                                                                                                             \
    } while (0)

} // namespace dtt
