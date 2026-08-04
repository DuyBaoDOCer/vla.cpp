// Copyright 2026 VinRobotics
//
// Licensed under the Apache License, Version 2.0 (the "License");

// TIP-06: CLI wrapper around vla::octo_dump_l1_stagewise_case_resident, mirroring
// octo_parity_dump.cpp's role for the diffusion path -- one small binary, all the real work
// lives in octo.cpp so this stays a thin driver ctest can shell out to.
//
//   octo_l1_parity_dump --ckpt <l1-gguf> --case <case-dir> --out <dump-dir> [--unnorm-dataset <key>]

#include "models/octo.h"

#include <cstdlib>
#include <cstdio>
#include <string>

int main(int argc, char ** argv) {
    std::string ckpt;
    std::string case_dir;
    std::string dump_dir;
    std::string unnorm_dataset;
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
        } else if (a == "--unnorm-dataset") {
            const char * v = need("--unnorm-dataset");
            if (!v) return 1;
            unnorm_dataset = v;
        } else {
            std::fprintf(stderr, "unknown arg: %s\n", a.c_str());
            return 1;
        }
    }
    if (dump_dir.empty()) {
        if (const char * env = std::getenv("VLA_OCTO_L1_DUMP")) dump_dir = env;
    }
    if (ckpt.empty() || case_dir.empty() || dump_dir.empty()) {
        std::fprintf(stderr,
            "usage: %s --ckpt <l1-gguf> --case <case-dir> [--out <dump-dir>|VLA_OCTO_L1_DUMP=<dump-dir>]\n"
            "          [--unnorm-dataset <key>]\n", argv[0]);
        return 1;
    }
    if (!vla::octo_dump_l1_stagewise_case_resident(ckpt, case_dir, dump_dir, unnorm_dataset)) return 2;
    std::printf("octo_l1_parity_dump wrote %s\n", dump_dir.c_str());
    return 0;
}
