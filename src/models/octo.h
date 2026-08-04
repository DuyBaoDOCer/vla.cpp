// Copyright 2026 VinRobotics
//
// Licensed under the Apache License, Version 2.0 (the "License");

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vla {

bool octo_dump_gguf_inventory(const std::string& ckpt_path);

// TIP-ND1-B: sole tokenizer-case dump path (was octo_dump_tokenizer_case / _resident before
// TIP-ND1-A proved the resident/GPU-wired path is bit-exact with the deleted disk-read
// original -- 20/20 golden cases). Dumps every M0-M8 boundary needed by
// verify_octo_parity.py, computed via the resident stage functions on the model's real
// backend (CUDA when available, CPU otherwise) -- the literal code OctoModelArch::predict()
// runs.
bool octo_dump_tokenizer_case_resident(const std::string& ckpt_path,
                                       const std::string& case_dir,
                                       const std::string& dump_dir,
                                       const std::string& t5_inject_path = "",
                                       const std::string& unnorm_dataset = "bridge_dataset");

// T5 SentencePiece-unigram tokenization (vocab embedded in the GGUF at convert
// time). Pads/truncates to octo.tokens.language (16), appends EOS, matching
// HFTokenizer(t5-base, max_length=16, padding="max_length", truncation=True).
bool octo_tokenize_text(const std::string& ckpt_path,
                        const std::string& text,
                        std::vector<int32_t>& input_ids,
                        std::vector<int32_t>& attention_mask);

// Action chunk produced by one live end-to-end Octo forward pass.
struct OctoCliAction {
    std::vector<float> normalized;    ///< [4,7] normalized action; verified vs golden (M1-M5 parity).
    std::vector<float> unnormalized;  ///< [4,7] world-unit action via octo.dataset_statistics
                                       ///< ("bridge_dataset"). The unnormalize formula itself is
                                       ///< verified vs golden (M8, see action_final_unnormalized in
                                       ///< the ctest harness); this specific *live* CLI call is not,
                                       ///< since it samples fresh diffusion noise each run (see
                                       ///< TIP-007's Completion Report for why an exact-match golden
                                       ///< comparison isn't meaningful for the live/stochastic path).
};

// Runs the full Octo pipeline (SmallStem x2 -> T5 encoder -> block transformer
// -> diffusion action head) from a live image pair + raw instruction text.
// primary/wrist images are single current frames (interleaved RGB8, arbitrary
// size); resized internally to 256x256 / 128x128 and duplicated across the
// window with timestep_pad_mask=[0,1] (matches OctoPt's HistoryWrapper cold
// start: history filled with the first frame, only the latest slot valid). No
// goal image (language-only conditioning): task image is zero-filled, matching
// OctoModelPt.create_tasks(texts=...).
// unnorm_dataset: octo.dataset_statistics key to un-normalize against; "" (default)
// auto-resolves (VLA_OCTO_UNNORM_DATASET env var, else the sole key if unambiguous,
// else "bridge_dataset" if present) -- see resolve_unnorm_dataset_key in octo.cpp.
bool octo_predict_from_images(const std::string& ckpt_path,
                              const uint8_t* primary_rgb, int primary_w, int primary_h,
                              const uint8_t* wrist_rgb, int wrist_w, int wrist_h,
                              const std::string& instruction,
                              OctoCliAction& out,
                              const std::string& unnorm_dataset = "");

// TIP-009: statistical action-distribution parity. Loads the golden case's own
// observation (SmallStem images, task-language input_ids/attention_mask, all
// four pad masks) once, then runs the full pipeline (tokenizer -> T5 -> block
// transformer -> diffusion) n_samples times end-to-end, each with fresh
// N(0,1) diffusion noise seeded from std::mt19937(seed + i). Does not touch
// any already-verified graph function's behavior; reuses them unchanged.
// samples_out: [n_samples,4,7] normalized action, row-major, sample-major.
// noise_out:   [n_samples,2,28] initial DDPM noise actually consumed, same layout.
bool octo_free_sample_case(const std::string& ckpt_path,
                           const std::string& case_dir,
                           int n_samples,
                           uint32_t seed,
                           std::vector<float>& samples_out,
                           std::vector<float>& noise_out);

// TIP-06: stagewise (T0-T5) golden-parity dump for the L1/proprio forward path (TIP-05),
// mirroring octo_dump_tokenizer_case_resident's role for the diffusion path. ckpt_path must
// be a head_type=l1 + proprio GGUF. case_dir must contain (written by
// octo-pytorch-kamusarj's scripts/dump_l1_stagewise_golden.py):
//   input.top_hwc_u8.npy (256,256,3) uint8, input.wrist_hwc_u8.npy (128,128,3) uint8,
//   input.proprio_raw.npy (7,) float32, input.instruction.txt (plain text).
// Runs octo.cpp's own real T5/SentencePiece tokenizer (octo_tokenize_text) and the exact
// head_type=l1 resident stage functions (run_proprio_tokenizer_graph_resident,
// run_l1_action_head_graph_resident, ...) -- the same code OctoModelArch::predict() runs for
// an L1 checkpoint, not a separate reimplementation. Writes T0-T5 to dump_dir as .f32 (same
// write_f32_dump format octo_dump_tokenizer_case_resident already uses) + manifest.txt.
bool octo_dump_l1_stagewise_case_resident(const std::string& ckpt_path,
                                          const std::string& case_dir,
                                          const std::string& dump_dir,
                                          const std::string& unnorm_dataset = "");

}  // namespace vla
