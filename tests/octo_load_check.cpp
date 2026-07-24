// Copyright 2026 VinRobotics
//
// Licensed under the Apache License, Version 2.0 (the "License");

#include "model.h"
#include "models/octo.h"

#include <cstdio>
#include <string>

int main(int argc, char ** argv) {
    if (argc != 3 || std::string(argv[1]) != "--ckpt") {
        std::fprintf(stderr, "usage: %s --ckpt octo-small-1.5-f32.gguf\n", argv[0]);
        return 1;
    }
    const std::string ckpt = argv[2];
    if (!vla::octo_dump_gguf_inventory(ckpt)) return 2;

    vla::Model * m = vla::model_load("", ckpt, "");
    if (!m) {
        std::fprintf(stderr, "octo model_load failed\n");
        return 3;
    }
    const vla::Config& cfg = vla::model_config(m);
    std::printf("loaded_octo hidden=%lld layers=%lld action_horizon=%lld action_dim=%lld\n",
                (long long) cfg.hidden, (long long) cfg.n_layers,
                (long long) cfg.n_suffix, (long long) cfg.real_action_dim);
    vla::model_free(m);
    return 0;
}
