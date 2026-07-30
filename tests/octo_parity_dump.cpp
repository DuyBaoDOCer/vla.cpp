// Copyright 2026 VinRobotics
//
// Licensed under the Apache License, Version 2.0 (the "License");

#include "models/octo.h"

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace {

// Minimal NPY v1.0 writer for a C-contiguous float32 array (TIP-009: cpp_samples.npy
// / cpp_noise.npy need to be loadable by numpy in scripts/compare_action_dist.py).
bool write_npy_f32(const std::string& path, const std::vector<float>& data, const std::vector<int64_t>& shape) {
    std::string shape_str = "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        shape_str += std::to_string(shape[i]);
        if (shape.size() == 1 || i + 1 < shape.size()) shape_str += ", ";
    }
    shape_str += ")";
    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': " + shape_str + ", }";
    const size_t total_before_pad = 10 + header.size() + 1;  // +1 for trailing '\n'
    const size_t pad = (64 - (total_before_pad % 64)) % 64;
    header.append(pad, ' ');
    header.push_back('\n');
    const uint16_t hlen = (uint16_t) header.size();

    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "cannot write %s\n", path.c_str());
        return false;
    }
    f.write("\x93NUMPY", 6);
    const char ver[2] = {1, 0};
    f.write(ver, 2);
    f.write(reinterpret_cast<const char *>(&hlen), 2);
    f.write(header.data(), (std::streamsize) header.size());
    f.write(reinterpret_cast<const char *>(data.data()), (std::streamsize) (data.size() * sizeof(float)));
    return (bool) f;
}

}  // namespace

int main(int argc, char ** argv) {
    std::string ckpt;
    std::string case_dir;
    std::string dump_dir;
    std::string t5_inject_path;
    std::string unnorm_dataset = "bridge_dataset";
    bool resident = false;  // TIP-BUILD-OCTO-GPU-B: opt-in GPU-wired dump path
    int free_sample_n = 0;
    uint32_t seed = 0;
    std::string out_samples_path = "cpp_samples.npy";
    std::string out_noise_path = "cpp_noise.npy";
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto need = [&](const char * opt) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s requires a value\n", opt);
                return nullptr;
            }
            return argv[++i];
        };
        if (a == "--ckpt") {
            const char * v = need("--ckpt");
            if (!v) return 1;
            ckpt = v;
        } else if (a == "--case") {
            const char * v = need("--case");
            if (!v) return 1;
            case_dir = v;
        } else if (a == "--out") {
            const char * v = need("--out");
            if (!v) return 1;
            dump_dir = v;
        } else if (a == "--t5-inject") {
            const char * v = need("--t5-inject");
            if (!v) return 1;
            t5_inject_path = v;
        } else if (a == "--unnorm-dataset") {
            const char * v = need("--unnorm-dataset");
            if (!v) return 1;
            unnorm_dataset = v;
        } else if (a == "--free-sample") {
            const char * v = need("--free-sample");
            if (!v) return 1;
            free_sample_n = std::atoi(v);
        } else if (a == "--seed") {
            const char * v = need("--seed");
            if (!v) return 1;
            seed = (uint32_t) std::strtoul(v, nullptr, 10);
        } else if (a == "--out-samples") {
            const char * v = need("--out-samples");
            if (!v) return 1;
            out_samples_path = v;
        } else if (a == "--out-noise") {
            const char * v = need("--out-noise");
            if (!v) return 1;
            out_noise_path = v;
        } else if (a == "--resident") {
            resident = true;
        } else {
            std::fprintf(stderr, "unknown arg: %s\n", a.c_str());
            return 1;
        }
    }

    if (free_sample_n > 0) {
        // TIP-009: statistical action-distribution parity mode. Runs the full
        // pipeline free_sample_n times end-to-end on --case's own observation.
        if (ckpt.empty() || case_dir.empty()) {
            std::fprintf(stderr,
                "usage: %s --ckpt octo-small-1.5-f32.gguf --case <golden-case-dir> --free-sample N\n"
                "          [--seed S] [--out-samples cpp_samples.npy] [--out-noise cpp_noise.npy]\n", argv[0]);
            return 1;
        }
        std::vector<float> samples, noise;
        if (!vla::octo_free_sample_case(ckpt, case_dir, free_sample_n, seed, samples, noise)) return 2;
        if (!write_npy_f32(out_samples_path, samples, {free_sample_n, 4, 7})) return 2;
        if (!write_npy_f32(out_noise_path, noise, {free_sample_n, 2, 28})) return 2;
        std::printf("octo_parity_dump free-sample: N=%d seed=%u -> %s, %s\n",
                    free_sample_n, seed, out_samples_path.c_str(), out_noise_path.c_str());
        return 0;
    }

    if (dump_dir.empty()) {
        if (const char * env = std::getenv("VLA_OCTO_DUMP")) dump_dir = env;
    }
    if (ckpt.empty() || case_dir.empty() || dump_dir.empty()) {
        std::fprintf(stderr,
            "usage: %s --ckpt octo-small-1.5-f32.gguf --case <golden-case-dir> [--t5-inject <npy>]\n"
            "          [--unnorm-dataset <key>] [--out <dump-dir>|VLA_OCTO_DUMP=<dump-dir>] [--resident]\n"
            "       %s --ckpt <gguf> --case <dir> --free-sample N [--seed S]\n"
            "          [--out-samples <npy>] [--out-noise <npy>]\n"
            "  --resident  compute via the resident/GPU-wired path (TIP-BUILD-OCTO-GPU-B) --\n"
            "              same code OctoModelArch::predict() runs, instead of the default\n"
            "              CPU-only host-vector dump path.\n", argv[0], argv[0]);
        return 1;
    }
    const bool ok = resident
        ? vla::octo_dump_tokenizer_case_resident(ckpt, case_dir, dump_dir, t5_inject_path, unnorm_dataset)
        : vla::octo_dump_tokenizer_case(ckpt, case_dir, dump_dir, t5_inject_path, unnorm_dataset);
    if (!ok) return 2;
    std::printf("octo_parity_dump wrote %s%s\n", dump_dir.c_str(), resident ? " (resident/GPU path)" : "");
    return 0;
}
