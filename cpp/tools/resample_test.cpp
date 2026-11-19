// dettato_resample_test IN.npy SR_IN SR_OUT OUT.npy : resample with dtt::resample and write float32 .npy
#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>

#include "dsp/resample.hpp"
#include "npy.hpp"

int main(int argc, char** argv)
{
    if (argc < 5) {
        std::fprintf(stderr, "usage: %s IN.npy SR_IN SR_OUT OUT.npy\n", argv[0]);
        return 2;
    }
    const auto x = npy::load(argv[1]).f32();
    const auto t0 = std::chrono::steady_clock::now();
    const auto y = dtt::resample(x.data(), x.size(), std::stod(argv[2]), std::stod(argv[3]));
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "%zu -> %zu samples in %.0f ms\n", x.size(), y.size(), ms);
    std::ofstream f(argv[4], std::ios::binary);
    std::string hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + std::to_string(y.size()) + ",), }";
    while ((10 + hdr.size() + 1) % 64 != 0) {
        hdr += ' ';
    }
    hdr += '\n';
    const unsigned short hl = static_cast<unsigned short>(hdr.size());
    f.write("\x93NUMPY\x01\x00", 8);
    f.write(reinterpret_cast<const char*>(&hl), 2);
    f.write(hdr.data(), static_cast<std::streamsize>(hdr.size()));
    f.write(reinterpret_cast<const char*>(y.data()), static_cast<std::streamsize>(y.size() * 4));
    return 0;
}
