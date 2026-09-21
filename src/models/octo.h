// Copyright 2026 VinRobotics
//
// Licensed under the Apache License, Version 2.0 (the "License");

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vla {

// T5 SentencePiece-unigram tokenization (vocab embedded in the GGUF at convert
// time). Pads/truncates to octo.tokens.language (16), appends EOS, matching
// HFTokenizer(t5-base, max_length=16, padding="max_length", truncation=True).
bool octo_tokenize_text(const std::string& ckpt_path,
                        const std::string& text,
                        std::vector<int32_t>& input_ids,
                        std::vector<int32_t>& attention_mask);

// Action chunk produced by one live end-to-end Octo forward pass.
struct OctoCliAction {
    std::vector<float> normalized;    ///< [action_horizon, 7] normalized action.
    std::vector<float> unnormalized;  ///< [action_horizon, 7] world-unit action, via the
                                      ///< octo.dataset_statistics block named by
                                      ///< @p unnorm_dataset below.
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

}  // namespace vla
