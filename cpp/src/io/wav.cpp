#include "io/wav.hpp"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <cstring>
#include <fstream>

#include "common.hpp"

namespace dtt
{

Audio readWav(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    DTT_CHECK(f.good(), "cannot open %s", path.c_str());
    std::vector<unsigned char> d((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    auto u16 = [&](size_t o) { return static_cast<uint32_t>(d[o] | (d[o + 1] << 8)); };
    auto u32 = [&](size_t o) { return u16(o) | (u16(o + 2) << 16); };
    DTT_CHECK(d.size() >= 12 && std::memcmp(d.data(), "RIFF", 4) == 0 && std::memcmp(d.data() + 8, "WAVE", 4) == 0,
              "%s is not a WAV file (convert other formats with: ffmpeg -i in.mp3 out.wav)", path.c_str());
    uint32_t fmt = 0, ch = 0, rate = 0, bits = 0;
    size_t data = 0, data_len = 0;
    for (size_t o = 12; o + 8 <= d.size();) {
        const uint32_t len = u32(o + 4);
        if (std::memcmp(d.data() + o, "fmt ", 4) == 0) {
            fmt = u16(o + 8);
            ch = u16(o + 10);
            rate = u32(o + 12);
            bits = u16(o + 22);
            if (fmt == 0xfffe && len >= 26) {
                fmt = u16(o + 8 + 24); // WAVE_FORMAT_EXTENSIBLE: the subformat's first two bytes
            }
        } else if (std::memcmp(d.data() + o, "data", 4) == 0) {
            data = o + 8;
            data_len = std::min<size_t>(len, d.size() - data);
        }
        o += 8 + len + (len & 1);
    }
    DTT_CHECK(data != 0 && ch > 0 && rate > 0, "%s: no fmt/data chunk", path.c_str());
    DTT_CHECK((fmt == 1 && (bits == 8 || bits == 16 || bits == 24 || bits == 32)) || (fmt == 3 && (bits == 32 || bits == 64)),
              "%s: unsupported WAV encoding (format %u, %u bits)", path.c_str(), fmt, bits);
    const size_t bps = bits / 8, frames = data_len / (bps * ch);
    Audio a;
    a.rate = rate;
    a.samples.assign(frames, 0.0f);
    for (size_t i = 0; i < frames; i++) {
        double acc = 0;
        for (uint32_t c = 0; c < ch; c++) {
            const unsigned char* p = d.data() + data + (i * ch + c) * bps;
            double v;
            if (fmt == 3) {
                if (bits == 32) {
                    float x;
                    std::memcpy(&x, p, 4);
                    v = x;
                } else {
                    double x;
                    std::memcpy(&x, p, 8);
                    v = x;
                }
            } else if (bits == 8) {
                v = (p[0] - 128) / 128.0;
            } else if (bits == 16) {
                v = static_cast<int16_t>(p[0] | (p[1] << 8)) / 32768.0;
            } else if (bits == 24) {
                int32_t x = p[0] | (p[1] << 8) | (p[2] << 16);
                if (x & 0x800000) {
                    x |= ~0xffffff;
                }
                v = x / 8388608.0;
            } else {
                int32_t x;
                std::memcpy(&x, p, 4);
                v = x / 2147483648.0;
            }
            acc += v;
        }
        a.samples[i] = static_cast<float>(acc / ch);
    }
    return a;
}

} // namespace dtt
