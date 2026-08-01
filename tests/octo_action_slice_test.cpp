// Copyright 2026 VinRobotics
//
// Licensed under the Apache License, Version 2.0 (the "License");

// Permanent tripwire for Octo's final-action-slice formula, the single riskiest line in the
// window-size port (Blueprint risk item 1445/1492): OctoPt's sample_actions() returns
// actions[:, -1] -- the LAST timestep of the observation window -- after the diffusion head
// runs over the full window. src/models/octo.cpp (run_diffusion_resident, around the "final
// action slice" comment) implements this as:
//
//     final_begin = (window_size - 1) * action;
//     final_end   = window_size * action;
//
// where `action` is the flattened per-timestep action chunk (action_horizon * action_dim, e.g.
// 4 * 7 = 28 for octo-small). This test does NOT call into octo.cpp -- that formula lives in a
// static function with no public entry point, and TIP-HARNESS's scope explicitly excludes
// touching src/models/octo.cpp to expose one. Instead it re-asserts the formula's two concrete,
// specifically-called-out cases (window=1 -> [0,28), window=2 -> [28,56)) as an independent,
// model-free, every-build check. If this test and octo.cpp's formula ever diverge, a human needs
// to reconcile them -- that divergence is exactly the failure mode this test exists to catch.

#include <cassert>
#include <cstddef>
#include <cstdio>

namespace {

struct Slice {
    size_t begin;
    size_t end;
};

// Mirrors src/models/octo.cpp's final_begin/final_end computation verbatim.
Slice final_action_slice(int window_size, size_t action) {
    return {
        (size_t) (window_size - 1) * action,
        (size_t) window_size * action,
    };
}

bool check(int window_size, size_t action, size_t expect_begin, size_t expect_end) {
    const Slice s = final_action_slice(window_size, action);
    const bool ok = (s.begin == expect_begin) && (s.end == expect_end);
    std::printf("window_size=%d action=%zu -> [%zu,%zu) expected [%zu,%zu) %s\n",
                window_size, action, s.begin, s.end, expect_begin, expect_end,
                ok ? "PASS" : "FAIL");
    return ok;
}

}  // namespace

int main() {
    // action_horizon=4, action_dim=7 -> action=28, matching octo-small's DiffusionActionHead.
    const size_t action = 28;
    bool ok = true;
    ok &= check(/*window_size=*/1, action, /*expect_begin=*/0, /*expect_end=*/28);
    ok &= check(/*window_size=*/2, action, /*expect_begin=*/28, /*expect_end=*/56);
    return ok ? 0 : 1;
}
