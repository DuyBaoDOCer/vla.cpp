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

}  // namespace vla
