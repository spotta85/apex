
// Purpose: read the A2 dumps. Only what dump_reference.py writes:
// version 1.0, C order, dtype <f4 or <i8.
#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace apex {

struct Npy {
    std::vector<int64_t> shape;
    std::string dtype;               // "<f4" or "<i8"
    std::vector<char> raw;

    const float*   f32() const { return reinterpret_cast<const float*>(raw.data()); }
    const int64_t* i64() const { return reinterpret_cast<const int64_t*>(raw.data()); }
    size_t count() const { size_t n = 1; for (auto d : shape) n *= d; return n; }
};

inline Npy loadNpy(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("npy: cannot open " + path);

    char magic[6]; uint8_t major, minor;
    in.read(magic, 6); in.read(reinterpret_cast<char*>(&major), 1); in.read(reinterpret_cast<char*>(&minor), 1);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0) throw std::runtime_error("npy: bad magic " + path);

    uint32_t header_len = 0;
    if (major == 1) { uint16_t h; in.read(reinterpret_cast<char*>(&h), 2); header_len = h; }
    else            { in.read(reinterpret_cast<char*>(&header_len), 4); }
    std::string header(header_len, '\0');
    in.read(header.data(), header_len);

    Npy out;
    // dtype
    size_t d = header.find("'descr':");
    size_t q1 = header.find('\'', d + 8), q2 = header.find('\'', q1 + 1);
    out.dtype = header.substr(q1 + 1, q2 - q1 - 1);
    if (out.dtype != "<f4" && out.dtype != "<i8") throw std::runtime_error("npy: unsupported dtype " + out.dtype);
    if (header.find("'fortran_order': True") != std::string::npos) throw std::runtime_error("npy: fortran order");
    // shape
    size_t s = header.find("'shape':");
    size_t p1 = header.find('(', s), p2 = header.find(')', p1);
    std::string dims = header.substr(p1 + 1, p2 - p1 - 1);
    size_t pos = 0;
    while (pos < dims.size()) {
        size_t comma = dims.find(',', pos);
        std::string tok = dims.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        if (!tok.empty() && tok.find_first_not_of(" ") != std::string::npos) out.shape.push_back(std::stoll(tok));
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    size_t item = out.dtype == "<f4" ? 4 : 8;
    out.raw.resize(out.count() * item);
    in.read(out.raw.data(), out.raw.size());
    if (!in) throw std::runtime_error("npy: truncated " + path);
    return out;
}

}  // namespace apex