// Copyright 2026 VinRobotics
//
// Licensed under the Apache License, Version 2.0 (the "License");

// TIP-BUILD-OCTO-GPU-B AC3: per-call latency of the live vla::predict() path for Octo.
// predict_check.cpp (the generic multi-arch harness) hardcodes a 6-token, no-mask language
// input; Octo's predict() rejects anything but exactly 16 lang tokens + a 16-entry attention
// mask (see the "expects lang_tokens AND attention_mask of exactly 16 entries" check in
// octo.cpp), so it can't be reused here without changing behavior for every other arch that
// binary also exercises. This is a small Octo-only sibling: same model.h API, same timing
// loop shape, real 16-token/mask input built via octo_tokenize_text (the tokenizer already
// verified against golden in the M-series parity tests), so it hits the exact
// OctoModelArch::predict() -> octo_run_pipeline_resident() -> 5 *_resident stage functions
// live path with no separate reimplementation.
//
//   octo_bench <ckpt.gguf> [iters]
//   env: VLA_BENCH_ITERS overrides the CLI iters arg if set.

#include "model.h"
#include "models/octo.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>
#include <vector>

using namespace vla;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <octo-ckpt.gguf> [iters]\n", argv[0]);
        return 1;
    }
    const std::string ckpt = argv[1];
    int iters = argc > 2 ? std::atoi(argv[2]) : 20;
    if (const char* env = std::getenv("VLA_BENCH_ITERS")) iters = std::atoi(env);

    Model* m = model_load("", ckpt, "");
    if (!m) {
        std::fprintf(stderr, "model_load failed\n");
        return 1;
    }
    const Config& cfg = model_config(m);

    std::vector<int32_t> input_ids, attention_mask;
    if (!octo_tokenize_text(ckpt, "pick up the block and place it on the plate", input_ids, attention_mask)) {
        std::fprintf(stderr, "octo_tokenize_text failed\n");
        return 1;
    }
    std::fprintf(stderr, "tokenized: n_lang=%zu n_mask=%zu\n", input_ids.size(), attention_mask.size());

    const int primary_w = 256, primary_h = 256;
    const int wrist_w = 128, wrist_h = 128;
    std::vector<uint8_t> primary_buf((size_t) 3 * primary_w * primary_h);
    std::vector<uint8_t> wrist_buf((size_t) 3 * wrist_w * wrist_h);
    for (int y = 0; y < primary_h; ++y)
        for (int x = 0; x < primary_w; ++x)
            for (int c = 0; c < 3; ++c)
                primary_buf[((size_t) y * primary_w + x) * 3 + c] = (uint8_t) ((x + 2 * y + 40 * c) & 0xFF);
    for (int y = 0; y < wrist_h; ++y)
        for (int x = 0; x < wrist_w; ++x)
            for (int c = 0; c < 3; ++c)
                wrist_buf[((size_t) y * wrist_w + x) * 3 + c] = (uint8_t) ((x + 3 * y + 60 * c) & 0xFF);

    ImageView views[2];
    views[0] = ImageView{primary_buf.data(), primary_w, primary_h, PixelFormat::U8};
    views[1] = ImageView{wrist_buf.data(), wrist_w, wrist_h, PixelFormat::U8};

    std::vector<float> state((size_t) cfg.max_state_dim, 0.0f);
    const size_t noise_n = (size_t) cfg.max_action_dim * (size_t) cfg.n_suffix;
    std::vector<float> noise(noise_n);
    for (size_t i = 0; i < noise_n; ++i) noise[i] = 0.001f * (float) ((i * 2654435761u) % 1000) - 0.5f;

    Inputs in{};
    in.images           = views;
    in.n_images         = 2;
    in.lang_tokens      = input_ids.data();
    in.n_lang           = (int) input_ids.size();
    in.attention_mask   = attention_mask.data();
    in.attention_mask_n = (int) attention_mask.size();
    in.state            = state.data();
    in.noise            = noise.data();
    in.timing_detail    = TimingDetail::NONE;

    std::vector<float> act = predict(m, in);
    std::printf("action_len=%zu\n", act.size());
    if (act.empty()) {
        std::fprintf(stderr, "predict() returned empty action -- see stderr above for the reason\n");
        model_free(m);
        return 2;
    }

    for (int w = 0; w < 3; ++w) (void) predict(m, in);
    double best = 1e30, sum = 0.0;
    for (int i = 0; i < iters; ++i) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        std::vector<float> a = predict(m, in);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        const double ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
        best = ms < best ? ms : best;
        sum += ms;
    }
    std::fprintf(stderr, "octo predict() over %d iters: min=%.3f ms  avg=%.3f ms\n", iters, best, sum / iters);

    model_free(m);
    return 0;
}
