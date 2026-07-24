// Copyright 2026 VinRobotics
//
// Licensed under the Apache License, Version 2.0 (the "License");

#pragma once

#include <string>

namespace vla {

bool octo_dump_gguf_inventory(const std::string& ckpt_path);
bool octo_dump_tokenizer_case(const std::string& ckpt_path,
                              const std::string& case_dir,
                              const std::string& dump_dir);

}  // namespace vla
