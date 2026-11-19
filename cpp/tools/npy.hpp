#pragma once

// Minimal .npy reader for the parity tests: little-endian f4 / f8 / f2 / i8 / i4 / i1 / u1 / b1, C order.

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace npy
{

struct Array {
    std::vector<int64_t> shape;
    std::vector<double> data; // every element as double (exact for the stored types)
    size_t size() const { return data.size(); }
    std::vector<float> f32() const { return std::vector<float>(data.begin(), data.end()); }
};

inline float half_to_float(uint16_t h)
{
    const uint32_t s = (h & 0x8000u) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    uint32_t f;
    if (e == 0) {
        if (m == 0) {
            f = s;
        } else {
            int ee = -1;
            uint32_t mm = m;
            do {
                ee++;
                mm <<= 1;
            } while ((mm & 0x400) == 0);
            f = s | ((127 - 15 - ee) << 23) | ((mm & 0x3ff) << 13);
        }
    } else if (e == 31) {
        f = s | 0x7f800000u | (m << 13);
    } else {
        f = s | ((e + 127 - 15) << 23) | (m << 13);
    }
    float r;
    std::memcpy(&r, &f, 4);
    return r;
}

inline Array load(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("cannot open " + path);
    }
    char magic[6];
    f.read(magic, 6);
    uint8_t ver[2];
    f.read(reinterpret_cast<char*>(ver), 2);
    uint32_t hlen = 0;
    if (ver[0] == 1) {
        uint16_t h16;
        f.read(reinterpret_cast<char*>(&h16), 2);
        hlen = h16;
    } else {
        f.read(reinterpret_cast<char*>(&hlen), 4);
    }
    std::string hdr(hlen, '\0');
    f.read(hdr.data(), hlen);
    auto field = [&](const std::string& key) {
        const size_t p = hdr.find("'" + key + "'");
        if (p == std::string::npos) {
            throw std::runtime_error(path + ": no " + key);
        }
        return p + key.size() + 3;
    };
    size_t p = field("descr");
    p = hdr.find('\'', p) + 1;
    const std::string descr = hdr.substr(p, hdr.find('\'', p) - p);
    if (hdr.find("'fortran_order': True") != std::string::npos) {
        throw std::runtime_error(path + ": fortran order");
    }
    p = hdr.find('(', field("shape")) + 1;
    const std::string sh = hdr.substr(p, hdr.find(')', p) - p);
    Array a;
    int64_t n = 1;
    size_t q = 0;
    while (q < sh.size()) {
        while (q < sh.size() && (sh[q] == ' ' || sh[q] == ',')) {
            q++;
        }
        if (q >= sh.size()) {
            break;
        }
        const int64_t d = std::stoll(sh.substr(q));
        a.shape.push_back(d);
        n *= d;
        while (q < sh.size() && sh[q] != ',') {
            q++;
        }
    }
    a.data.resize(static_cast<size_t>(n));
    const std::string t = descr.substr(1);
    auto read = [&](auto tag) {
        using T = decltype(tag);
        std::vector<T> buf(static_cast<size_t>(n));
        f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(n * sizeof(T)));
        for (size_t i = 0; i < buf.size(); i++) {
            a.data[i] = static_cast<double>(buf[i]);
        }
    };
    if (t == "f4") {
        read(float{});
    } else if (t == "f8") {
        read(double{});
    } else if (t == "i8") {
        read(int64_t{});
    } else if (t == "i4") {
        read(int32_t{});
    } else if (t == "i1") {
        read(int8_t{});
    } else if (t == "u1" || t == "b1") {
        read(uint8_t{});
    } else if (t == "f2") {
        std::vector<uint16_t> buf(static_cast<size_t>(n));
        f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(n * 2));
        for (size_t i = 0; i < buf.size(); i++) {
            a.data[i] = half_to_float(buf[i]);
        }
    } else {
        throw std::runtime_error(path + ": unsupported dtype " + descr);
    }
    if (!f) {
        throw std::runtime_error(path + ": short read");
    }
    return a;
}

} // namespace npy
