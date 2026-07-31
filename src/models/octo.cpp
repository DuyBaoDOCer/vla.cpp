// Copyright 2026 VinRobotics
//
// Licensed under the Apache License, Version 2.0 (the "License");

#include "arch.h"
#include "model.h"
#include "models/gguf_reader.h"
#include "models/octo.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif
#include "gguf.h"

#include "nlohmann/json.hpp"
#include "sentencepiece_processor.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cfloat>
#include <cstring>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace vla {
namespace {

// Per-timestep token count in the block-transformer sequence: primary(256) + wrist(64)
// + repeated-language(16) + readout(1) = 337. Constant across window_size -- only the
// number of timesteps (window_size) and the derived total seq length change.
constexpr int kStepTokens = 337;
// task_language prefix tokens (once, not repeated per timestep).
constexpr int kTaskTokens = 16;
// Shared max-horizon slab size of the obs/task pos-embedding tables written by
// scripts/convert_octo_to_gguf.py (same for every checkpoint regardless of the
// window_size actually trained/used) -- see TIP-C1. window_size must fit within it.
constexpr int kMaxHorizon = 10;

// seq = kTaskTokens + window_size * kStepTokens (window=2 -> 690, window=1 -> 353).
inline int octo_seq_len(int64_t window_size) {
    return kTaskTokens + (int) window_size * kStepTokens;
}

struct OctoModelArch : public ModelArchBase {
    OctoModelArch() : ModelArchBase(Arch::OCTO) {}
    ~OctoModelArch() override {
        if (weight_buf)  ggml_backend_buffer_free(weight_buf);
        if (ctx_weights) ggml_free(ctx_weights);
        if (backend)     ggml_backend_free(backend);
    }

    std::string gguf_path;
    ggml_backend_t backend = nullptr;
    ggml_context * ctx_weights = nullptr;
    ggml_backend_buffer_t weight_buf = nullptr;
    ggml_type matmul_type = GGML_TYPE_F32;
    int n_threads = default_cpu_threads();

    // TIP-BUILD-OCTO-GPU-B: predict()'s 5 stages now compute on `backend` directly (CUDA on
    // GPU-capable builds, CPU otherwise -- see octo_create) and reference weight tensors
    // resident on `weight_buf`/`ctx_weights` (populated once by load_all_tensors, same call
    // as before TIP-A/B). TIP-A's separate CPU-only residency buffer
    // (weight_cpu_backend/ctx_weights_cpu/weight_buf_cpu/tensors_cpu/load_all_tensors_cpu)
    // has been removed: it's no longer needed now that compute and residency both key off
    // the same real `backend` -- one residency path, matching the model's actual backend,
    // exactly like every sibling.

    int64_t hidden = 384;
    int64_t blocks = 12;
    int64_t heads = 6;
    int64_t ffn = 1536;
    int64_t window_size = 2;
    int64_t action_horizon = 4;
    int64_t action_dim = 7;
    int64_t primary_tokens = 256;
    int64_t wrist_tokens = 64;
    int64_t language_tokens = 16;
    int64_t diffusion_steps = 20;

    gguf_reader io{"octo"};
    std::vector<ggml_tensor *> tensors;

    std::vector<float> predict(const Inputs& in) override;
};

bool require_key(const gguf_reader& g, const char * key) {
    if (!g.has(key)) {
        std::fprintf(stderr, "vla(octo): missing metadata %s\n", key);
        return false;
    }
    return true;
}

bool load_config(const gguf_reader& g, OctoModelArch& m) {
    const char * keys[] = {
        "octo.architecture",
        "octo.embedding_length",
        "octo.block_count",
        "octo.attention.head_count",
        "octo.feed_forward_length",
        "octo.attention.layer_norm_eps",
        "octo.window_size",
        "octo.action.horizon",
        "octo.action.dim",
        "octo.readout.count",
        "octo.tokens.primary",
        "octo.tokens.wrist",
        "octo.tokens.language",
        "octo.image.primary_size",
        "octo.image.wrist_size",
        "octo.diffusion.steps",
        "octo.diffusion.beta_schedule",
        "octo.diffusion.s",
        "octo.diffusion.max_action",
        "octo.diffusion.time_dim",
        "octo.diffusion.hidden",
        "octo.diffusion.num_blocks",
        "octo.dataset_statistics",
    };
    for (const char * key : keys) {
        if (!require_key(g, key)) return false;
    }
    if (g.str("octo.architecture") != "octo-small-1.5") {
        std::fprintf(stderr, "vla(octo): octo.architecture=%s, expected octo-small-1.5\n", g.str("octo.architecture").c_str());
        return false;
    }

    m.hidden = g.u32("octo.embedding_length");
    m.blocks = g.u32("octo.block_count");
    m.heads = g.u32("octo.attention.head_count");
    m.ffn = g.u32("octo.feed_forward_length");
    m.window_size = g.u32("octo.window_size");
    m.action_horizon = g.u32("octo.action.horizon");
    m.action_dim = g.u32("octo.action.dim");
    m.primary_tokens = g.u32("octo.tokens.primary");
    m.wrist_tokens = g.u32("octo.tokens.wrist");
    m.language_tokens = g.u32("octo.tokens.language");
    m.diffusion_steps = g.u32("octo.diffusion.steps");

    if (m.hidden != 384 || m.blocks != 12 || m.heads != 6 || m.ffn != 1536 ||
        m.action_horizon != 4 || m.action_dim != 7) {
        std::fprintf(stderr, "vla(octo): metadata does not match octo-small-1.5 M0 constants\n");
        return false;
    }
    // window_size is per-checkpoint (bridge pretrain=2, LIBERO finetunes such as
    // cyrusneary/octo-finetuned-libero=1); the pos-embedding table is a shared
    // max_horizon=10 slab (see scripts/convert_octo_to_gguf.py), so any value in
    // [1, kMaxHorizon] is a legal slice of it.
    if (m.window_size < 1 || m.window_size > kMaxHorizon) {
        std::fprintf(stderr, "vla(octo): octo.window_size=%lld out of supported range [1, %d]\n",
                     (long long) m.window_size, kMaxHorizon);
        return false;
    }

    m.cfg.n_img = m.primary_tokens + m.wrist_tokens;
    m.cfg.n_lang = m.language_tokens;
    m.cfg.n_state = 0;
    m.cfg.n_prefix = m.language_tokens + m.window_size * (m.primary_tokens + m.wrist_tokens + m.language_tokens);
    m.cfg.n_suffix = m.action_horizon;
    m.cfg.n_full = m.cfg.n_prefix + m.window_size;
    m.cfg.hidden = m.hidden;
    m.cfg.expert_h = 256;
    m.cfg.intermediate = m.ffn;
    m.cfg.expert_inter = 256;
    m.cfg.n_q_heads = m.heads;
    m.cfg.n_kv_heads = m.heads;
    m.cfg.head_dim = m.hidden / m.heads;
    m.cfg.q_full_dim = m.hidden;
    m.cfg.kv_full_dim = m.hidden;
    m.cfg.n_layers = m.blocks;
    m.cfg.self_attn_every_n = 1;
    m.cfg.max_state_dim = 0;
    m.cfg.max_action_dim = m.action_dim;
    m.cfg.real_state_dim = 0;
    m.cfg.real_action_dim = m.action_dim;
    m.cfg.norm_eps = g.f32("octo.attention.layer_norm_eps");
    m.cfg.num_steps = (int) m.diffusion_steps;
    return true;
}

bool is_matmul_tensor(const char * name) {
    return std::strstr(name, ".weight") != nullptr &&
           std::strstr(name, ".gn.") == nullptr &&
           std::strstr(name, "_norm.") == nullptr &&
           std::strstr(name, ".ln.") == nullptr &&
           std::strstr(name, "pos_embd") == nullptr &&
           std::strstr(name, "time_fourier") == nullptr;
}

bool load_all_tensors(OctoModelArch& m, gguf_reader& g) {
    ggml_init_params wp = { (size_t) 16 * 1024 * 1024, nullptr, true };
    m.ctx_weights = ggml_init(wp);
    if (!m.ctx_weights) {
        std::fprintf(stderr, "vla(octo): ggml_init(ctx_weights) failed\n");
        return false;
    }

    const int64_t n = gguf_get_n_tensors(g.gctx);
    m.tensors.reserve((size_t) n);
    ggml_context * W = m.ctx_weights;
    auto mk = [&](const char * name, ggml_type type) -> ggml_tensor * {
        const ggml_tensor * gt = g.meta(name);
        if (!gt) {
            std::fprintf(stderr, "vla(octo): missing tensor %s\n", name);
            return nullptr;
        }
        if (gt->type != GGML_TYPE_F32) {
            std::fprintf(stderr, "vla(octo): tensor %s type=%d, expected F32 for M0\n", name, (int) gt->type);
            return nullptr;
        }
        ggml_tensor * t = ggml_new_tensor(W, type, ggml_n_dims(gt), gt->ne);
        ggml_set_name(t, name);
        return t;
    };
    auto mk_f32 = [&](const char * name) { return mk(name, GGML_TYPE_F32); };
    auto mk_mm = [&](const char * name) { return mk(name, m.matmul_type); };

    for (int64_t i = 0; i < n; ++i) {
        const char * name = gguf_get_tensor_name(g.gctx, i);
        ggml_tensor * t = is_matmul_tensor(name) ? mk_mm(name) : mk_f32(name);
        if (!t) return false;
        m.tensors.push_back(t);
    }

    m.weight_buf = ggml_backend_alloc_ctx_tensors(m.ctx_weights, m.backend);
    if (!m.weight_buf) {
        std::fprintf(stderr, "vla(octo): ggml_backend_alloc_ctx_tensors failed\n");
        return false;
    }

    for (ggml_tensor * t : m.tensors) {
        std::vector<uint8_t> bytes = g.read_convert(t->name, t->type);
        if (bytes.empty()) return false;
        ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
    }
    return true;
}

// TIP-BUILD-OCTO-GPU-A: looks up an already-resident weight tensor by its GGUF name (same
// name strings the *_resident graph-building functions below already used as ggml_set_name
// literals, so this is a drop-in replacement for the old "build a fresh tensor + upload from
// a freshly-disk-read host vector" pattern). Used only by predict()'s live pipeline; the
// original host-vector-based run_*_graph functions (unchanged, still disk-reading) remain in
// use by octo_dump_tokenizer_case / octo_free_sample_case (test/CLI tooling).
static ggml_tensor * wt(ggml_context * ctx_w, const char * name) {
    ggml_tensor * t = ggml_get_tensor(ctx_w, name);
    if (!t) std::fprintf(stderr, "vla(octo): missing resident weight %s\n", name);
    return t;
}

// Host-side copy of a resident weight tensor's data, for the handful of weights that need a
// per-call CPU-side transform (standardize_conv_weight, expand_1d, pos-embed slicing) before
// they become graph operands -- those transforms are unchanged, just fed from the resident
// tensor's bytes instead of a freshly-read host vector. TIP-BUILD-OCTO-GPU-B: uses
// ggml_backend_tensor_get (not a raw ->data dereference) so this stays correct now that the
// resident tensor may live on a CUDA device buffer, not just host memory.
static std::vector<float> tensor_to_vec(const ggml_tensor * t) {
    std::vector<float> out((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
    return out;
}

struct NpyU8 {
    std::vector<int64_t> shape;
    std::vector<uint8_t> data;
};

struct NpyF32 {
    std::vector<int64_t> shape;
    std::vector<float> data;
};

struct NpyBool {
    std::vector<int64_t> shape;
    std::vector<uint8_t> data;
};

struct NpyI32 {
    std::vector<int64_t> shape;
    std::vector<int32_t> data;
};

static bool read_file_all(const std::string& path, std::vector<uint8_t>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "vla(octo): cannot open %s\n", path.c_str());
        return false;
    }
    f.seekg(0, std::ios::end);
    const std::streamoff n = f.tellg();
    f.seekg(0, std::ios::beg);
    out.resize((size_t) n);
    return n == 0 || (bool) f.read(reinterpret_cast<char *>(out.data()), n);
}

static bool parse_npy_u8(const std::string& path, NpyU8& out) {
    std::vector<uint8_t> bytes;
    if (!read_file_all(path, bytes)) return false;
    if (bytes.size() < 16 || std::memcmp(bytes.data(), "\x93NUMPY", 6) != 0) {
        std::fprintf(stderr, "vla(octo): %s is not a .npy file\n", path.c_str());
        return false;
    }
    const int major = bytes[6];
    size_t pos = 8;
    uint32_t hlen = 0;
    if (major == 1) {
        hlen = (uint32_t) bytes[pos] | ((uint32_t) bytes[pos + 1] << 8);
        pos += 2;
    } else if (major == 2 || major == 3) {
        hlen = (uint32_t) bytes[pos] | ((uint32_t) bytes[pos + 1] << 8) |
               ((uint32_t) bytes[pos + 2] << 16) | ((uint32_t) bytes[pos + 3] << 24);
        pos += 4;
    } else {
        std::fprintf(stderr, "vla(octo): unsupported .npy version %d in %s\n", major, path.c_str());
        return false;
    }
    if (pos + hlen > bytes.size()) return false;
    const std::string header(reinterpret_cast<const char *>(bytes.data() + pos), hlen);
    pos += hlen;
    if (header.find("'descr': '|u1'") == std::string::npos &&
        header.find("\"descr\": \"|u1\"") == std::string::npos) {
        std::fprintf(stderr, "vla(octo): %s expected uint8 .npy\n", path.c_str());
        return false;
    }
    if (header.find("'fortran_order': False") == std::string::npos &&
        header.find("\"fortran_order\": False") == std::string::npos) {
        std::fprintf(stderr, "vla(octo): %s expected C-order .npy\n", path.c_str());
        return false;
    }
    const size_t l = header.find('(');
    const size_t r = header.find(')', l == std::string::npos ? 0 : l);
    if (l == std::string::npos || r == std::string::npos) return false;
    out.shape.clear();
    size_t s = l + 1;
    while (s < r) {
        while (s < r && (header[s] == ' ' || header[s] == ',')) ++s;
        size_t e = s;
        while (e < r && header[e] >= '0' && header[e] <= '9') ++e;
        if (e > s) out.shape.push_back(std::strtoll(header.substr(s, e - s).c_str(), nullptr, 10));
        s = e + 1;
    }
    int64_t ne = 1;
    for (int64_t d : out.shape) ne *= d;
    if (pos + (size_t) ne > bytes.size()) {
        std::fprintf(stderr, "vla(octo): %s truncated .npy payload\n", path.c_str());
        return false;
    }
    out.data.assign(bytes.begin() + (ptrdiff_t) pos, bytes.begin() + (ptrdiff_t) pos + ne);
    return true;
}

static bool parse_npy_f32(const std::string& path, NpyF32& out) {
    std::vector<uint8_t> bytes;
    if (!read_file_all(path, bytes)) return false;
    if (bytes.size() < 16 || std::memcmp(bytes.data(), "\x93NUMPY", 6) != 0) {
        std::fprintf(stderr, "vla(octo): %s is not a .npy file\n", path.c_str());
        return false;
    }
    const int major = bytes[6];
    size_t pos = 8;
    uint32_t hlen = 0;
    if (major == 1) {
        hlen = (uint32_t) bytes[pos] | ((uint32_t) bytes[pos + 1] << 8);
        pos += 2;
    } else if (major == 2 || major == 3) {
        hlen = (uint32_t) bytes[pos] | ((uint32_t) bytes[pos + 1] << 8) |
               ((uint32_t) bytes[pos + 2] << 16) | ((uint32_t) bytes[pos + 3] << 24);
        pos += 4;
    } else {
        std::fprintf(stderr, "vla(octo): unsupported .npy version %d in %s\n", major, path.c_str());
        return false;
    }
    if (pos + hlen > bytes.size()) return false;
    const std::string header(reinterpret_cast<const char *>(bytes.data() + pos), hlen);
    pos += hlen;
    if (header.find("'descr': '<f4'") == std::string::npos &&
        header.find("\"descr\": \"<f4\"") == std::string::npos &&
        header.find("'descr': '|f4'") == std::string::npos &&
        header.find("\"descr\": \"|f4\"") == std::string::npos) {
        std::fprintf(stderr, "vla(octo): %s expected float32 .npy\n", path.c_str());
        return false;
    }
    if (header.find("'fortran_order': False") == std::string::npos &&
        header.find("\"fortran_order\": False") == std::string::npos) {
        std::fprintf(stderr, "vla(octo): %s expected C-order .npy\n", path.c_str());
        return false;
    }
    const size_t l = header.find('(');
    const size_t r = header.find(')', l == std::string::npos ? 0 : l);
    if (l == std::string::npos || r == std::string::npos) return false;
    out.shape.clear();
    size_t s = l + 1;
    while (s < r) {
        while (s < r && (header[s] == ' ' || header[s] == ',')) ++s;
        size_t e = s;
        while (e < r && header[e] >= '0' && header[e] <= '9') ++e;
        if (e > s) out.shape.push_back(std::strtoll(header.substr(s, e - s).c_str(), nullptr, 10));
        s = e + 1;
    }
    int64_t ne = 1;
    for (int64_t d : out.shape) ne *= d;
    if (pos + (size_t) ne * sizeof(float) > bytes.size()) {
        std::fprintf(stderr, "vla(octo): %s truncated .npy payload\n", path.c_str());
        return false;
    }
    out.data.resize((size_t) ne);
    std::memcpy(out.data.data(), bytes.data() + pos, (size_t) ne * sizeof(float));
    return true;
}

static bool parse_npy_bool(const std::string& path, NpyBool& out) {
    std::vector<uint8_t> bytes;
    if (!read_file_all(path, bytes)) return false;
    if (bytes.size() < 16 || std::memcmp(bytes.data(), "\x93NUMPY", 6) != 0) {
        std::fprintf(stderr, "vla(octo): %s is not a .npy file\n", path.c_str());
        return false;
    }
    const int major = bytes[6];
    size_t pos = 8;
    uint32_t hlen = 0;
    if (major == 1) {
        hlen = (uint32_t) bytes[pos] | ((uint32_t) bytes[pos + 1] << 8);
        pos += 2;
    } else if (major == 2 || major == 3) {
        hlen = (uint32_t) bytes[pos] | ((uint32_t) bytes[pos + 1] << 8) |
               ((uint32_t) bytes[pos + 2] << 16) | ((uint32_t) bytes[pos + 3] << 24);
        pos += 4;
    } else {
        std::fprintf(stderr, "vla(octo): unsupported .npy version %d in %s\n", major, path.c_str());
        return false;
    }
    if (pos + hlen > bytes.size()) return false;
    const std::string header(reinterpret_cast<const char *>(bytes.data() + pos), hlen);
    pos += hlen;
    if (header.find("'descr': '|b1'") == std::string::npos &&
        header.find("\"descr\": \"|b1\"") == std::string::npos) {
        std::fprintf(stderr, "vla(octo): %s expected bool .npy\n", path.c_str());
        return false;
    }
    if (header.find("'fortran_order': False") == std::string::npos &&
        header.find("\"fortran_order\": False") == std::string::npos) {
        std::fprintf(stderr, "vla(octo): %s expected C-order .npy\n", path.c_str());
        return false;
    }
    const size_t l = header.find('(');
    const size_t r = header.find(')', l == std::string::npos ? 0 : l);
    if (l == std::string::npos || r == std::string::npos) return false;
    out.shape.clear();
    size_t s = l + 1;
    while (s < r) {
        while (s < r && (header[s] == ' ' || header[s] == ',')) ++s;
        size_t e = s;
        while (e < r && header[e] >= '0' && header[e] <= '9') ++e;
        if (e > s) out.shape.push_back(std::strtoll(header.substr(s, e - s).c_str(), nullptr, 10));
        s = e + 1;
    }
    int64_t ne = 1;
    for (int64_t d : out.shape) ne *= d;
    if (pos + (size_t) ne > bytes.size()) {
        std::fprintf(stderr, "vla(octo): %s truncated .npy payload\n", path.c_str());
        return false;
    }
    out.data.assign(bytes.begin() + (ptrdiff_t) pos, bytes.begin() + (ptrdiff_t) pos + ne);
    return true;
}

// Some golden cases (e.g. tier2/bridge_debug, real-robot language-only conditioning)
// never recorded a task-image tensor at all: no goal image was ever provided for
// that dump, matching OctoModelPt.create_tasks(texts=...)'s own zero-fill for
// absent task modalities (octo-pytorch/octo/model/octo_model_pt.py:139-152). Parse
// the file if present (tier1's synthetic zero-content task image); zero-fill to
// match obs's spatial shape if the file is simply absent.
static bool parse_npy_u8_or_zero_task(const std::string& path, const NpyU8& obs, NpyU8& task) {
    std::error_code ec;
    if (std::filesystem::exists(path, ec)) return parse_npy_u8(path, task);
    if (obs.shape.size() != 5) {
        std::fprintf(stderr, "vla(octo): cannot infer zero-fill task shape from obs\n");
        return false;
    }
    task.shape = {1, obs.shape[2], obs.shape[3], obs.shape[4]};
    task.data.assign((size_t) obs.shape[2] * obs.shape[3] * obs.shape[4], 0);
    return true;
}

// Observation-side counterpart to parse_npy_u8_or_zero_task: some golden cases (the
// cyrusneary LIBERO checkpoint, TIP-GOLD) come from a genuinely single-camera
// finetune -- its example_batch/finetune_config never fed a wrist observation at all,
// so tensors/input.observation.image_wrist.npy simply doesn't exist. Zero-fill to
// {1, window_size, 3, side, side} if absent (matches
// scripts/patch_libero_golden_for_harness.py's WRIST_IMAGE_SIZE=128 placeholder, now
// built in so that script is no longer required). `used_real` reports which happened,
// so the caller can mark the corresponding wrist_valid pad-mask entries false instead
// of pretending a placeholder camera is a real, valid observation.
static bool parse_npy_u8_or_zero_obs(const std::string& path, int window_size, int side,
                                     NpyU8& obs, bool& used_real) {
    std::error_code ec;
    if (std::filesystem::exists(path, ec)) {
        used_real = true;
        return parse_npy_u8(path, obs);
    }
    used_real = false;
    obs.shape = {1, window_size, 3, side, side};
    obs.data.assign((size_t) window_size * 3 * side * side, 0);
    return true;
}

// Bool-array counterpart: zero-fill (all-False = "not valid") {1, window_size} if the
// golden case has no wrist pad-mask file at all (same single-camera scenario above).
static bool parse_npy_bool_or_zero(const std::string& path, int window_size, NpyBool& out) {
    std::error_code ec;
    if (std::filesystem::exists(path, ec)) return parse_npy_bool(path, out);
    out.shape = {1, window_size};
    out.data.assign((size_t) window_size, 0);
    return true;
}

static bool parse_npy_i32(const std::string& path, NpyI32& out) {
    std::vector<uint8_t> bytes;
    if (!read_file_all(path, bytes)) return false;
    if (bytes.size() < 16 || std::memcmp(bytes.data(), "\x93NUMPY", 6) != 0) {
        std::fprintf(stderr, "vla(octo): %s is not a .npy file\n", path.c_str());
        return false;
    }
    const int major = bytes[6];
    size_t pos = 8;
    uint32_t hlen = 0;
    if (major == 1) {
        hlen = (uint32_t) bytes[pos] | ((uint32_t) bytes[pos + 1] << 8);
        pos += 2;
    } else if (major == 2 || major == 3) {
        hlen = (uint32_t) bytes[pos] | ((uint32_t) bytes[pos + 1] << 8) |
               ((uint32_t) bytes[pos + 2] << 16) | ((uint32_t) bytes[pos + 3] << 24);
        pos += 4;
    } else {
        std::fprintf(stderr, "vla(octo): unsupported .npy version %d in %s\n", major, path.c_str());
        return false;
    }
    if (pos + hlen > bytes.size()) return false;
    const std::string header(reinterpret_cast<const char *>(bytes.data() + pos), hlen);
    pos += hlen;
    // Golden dump scripts aren't consistent about token-id width (tier1 uses int32,
    // tier2/bridge_debug uses torch's default int64) -- accept either and downcast;
    // token ids are always well within int32 range.
    bool is_i32 = header.find("'descr': '<i4'") != std::string::npos ||
                  header.find("\"descr\": \"<i4\"") != std::string::npos ||
                  header.find("'descr': '=i4'") != std::string::npos ||
                  header.find("\"descr\": \"=i4\"") != std::string::npos;
    bool is_i64 = header.find("'descr': '<i8'") != std::string::npos ||
                  header.find("\"descr\": \"<i8\"") != std::string::npos ||
                  header.find("'descr': '=i8'") != std::string::npos ||
                  header.find("\"descr\": \"=i8\"") != std::string::npos;
    if (!is_i32 && !is_i64) {
        std::fprintf(stderr, "vla(octo): %s expected int32 or int64 .npy\n", path.c_str());
        return false;
    }
    if (header.find("'fortran_order': False") == std::string::npos &&
        header.find("\"fortran_order\": False") == std::string::npos) {
        std::fprintf(stderr, "vla(octo): %s expected C-order .npy\n", path.c_str());
        return false;
    }
    const size_t l = header.find('(');
    const size_t r = header.find(')', l == std::string::npos ? 0 : l);
    if (l == std::string::npos || r == std::string::npos) return false;
    out.shape.clear();
    size_t s = l + 1;
    while (s < r) {
        while (s < r && (header[s] == ' ' || header[s] == ',')) ++s;
        size_t e = s;
        while (e < r && header[e] >= '0' && header[e] <= '9') ++e;
        if (e > s) out.shape.push_back(std::strtoll(header.substr(s, e - s).c_str(), nullptr, 10));
        s = e + 1;
    }
    int64_t ne = 1;
    for (int64_t d : out.shape) ne *= d;
    const size_t elsz = is_i64 ? sizeof(int64_t) : sizeof(int32_t);
    if (pos + (size_t) ne * elsz > bytes.size()) {
        std::fprintf(stderr, "vla(octo): %s truncated .npy payload\n", path.c_str());
        return false;
    }
    out.data.resize((size_t) ne);
    if (is_i64) {
        std::vector<int64_t> tmp((size_t) ne);
        std::memcpy(tmp.data(), bytes.data() + pos, (size_t) ne * elsz);
        for (size_t i = 0; i < tmp.size(); ++i) out.data[i] = (int32_t) tmp[i];
    } else {
        std::memcpy(out.data.data(), bytes.data() + pos, (size_t) ne * elsz);
    }
    return true;
}

static bool write_f32_dump(const std::string& dir,
                           const char * name,
                           const std::vector<float>& data,
                           const std::vector<int64_t>& shape,
                           std::ofstream& manifest) {
    static std::string made;
    if (made != dir) {
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        if (ec) {
            std::fprintf(stderr, "vla(octo): cannot create %s: %s\n", dir.c_str(), ec.message().c_str());
            return false;
        }
        made = dir;
    }
    const std::string path = dir + "/" + name + ".f32";
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "vla(octo): cannot write %s\n", path.c_str());
        return false;
    }
    f.write(reinterpret_cast<const char *>(data.data()), (std::streamsize) (data.size() * sizeof(float)));
    manifest << name << " " << path << " float32";
    for (int64_t d : shape) manifest << " " << d;
    manifest << "\n";
    return (bool) f;
}

static bool write_bool_dump(const std::string& dir,
                            const char * name,
                            const std::vector<uint8_t>& data,
                            const std::vector<int64_t>& shape,
                            std::ofstream& manifest) {
    const std::string path = dir + "/" + name + ".bool";
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "vla(octo): cannot write %s\n", path.c_str());
        return false;
    }
    f.write(reinterpret_cast<const char *>(data.data()), (std::streamsize) data.size());
    manifest << name << " " << path << " bool";
    for (int64_t d : shape) manifest << " " << d;
    manifest << "\n";
    return (bool) f;
}

struct OctoObsWeights {
    std::vector<float> conv_w[4], conv_b[4], gn_w[4], gn_b[4];
    std::vector<float> patch_w, patch_b, proj_w, proj_b, pos;
};

struct OctoLanguageWeights {
    std::vector<float> proj_w, proj_b, pos;
};

struct OctoBlockWeights {
    std::vector<float> attn_norm_w, attn_norm_b;
    std::vector<float> qkv_w, qkv_b, attn_o_w, attn_o_b;
    std::vector<float> ffn_norm_w, ffn_norm_b;
    std::vector<float> ffn_up_w, ffn_up_b, ffn_down_w, ffn_down_b;
};

struct OctoTransformerWeights {
    std::array<OctoBlockWeights, 12> blocks;
    std::vector<float> output_norm_w, output_norm_b, readout_pos;
};

struct OctoT5BlockWeights {
    std::vector<float> attn_norm_w;
    std::vector<float> q_w, k_w, v_w, o_w;
    std::vector<float> ffn_norm_w;
    std::vector<float> ffn_up_w, ffn_down_w;
};

struct OctoT5Weights {
    std::array<OctoT5BlockWeights, 12> blocks;
    std::vector<float> attn_rel_b;    // shared bucket->head bias table, block 0 only
    std::vector<float> output_norm_w;
};

struct OctoDiffusionBlockWeights {
    std::vector<float> ln_w, ln_b;
    std::vector<float> fc1_w, fc1_b;
    std::vector<float> fc2_w, fc2_b;
};

struct OctoDiffusionWeights {
    std::vector<float> time_fourier_w;
    std::vector<float> cond0_w, cond0_b;
    std::vector<float> cond1_w, cond1_b;
    std::vector<float> reverse_in_w, reverse_in_b;
    std::array<OctoDiffusionBlockWeights, 3> blocks;
    std::vector<float> reverse_out_w, reverse_out_b;
};

static bool read_obs_weights(gguf_reader& g, const char * view, OctoObsWeights& w) {
    char name[160];
    for (int i = 0; i < 4; ++i) {
        std::snprintf(name, sizeof(name), "octo.obs.%s.stem.%d.conv.weight", view, i); w.conv_w[i] = g.read_f32(name);
        std::snprintf(name, sizeof(name), "octo.obs.%s.stem.%d.conv.bias",   view, i); w.conv_b[i] = g.read_f32(name);
        std::snprintf(name, sizeof(name), "octo.obs.%s.stem.%d.gn.weight",   view, i); w.gn_w[i]   = g.read_f32(name);
        std::snprintf(name, sizeof(name), "octo.obs.%s.stem.%d.gn.bias",     view, i); w.gn_b[i]   = g.read_f32(name);
        if (w.conv_w[i].empty() || w.conv_b[i].empty() || w.gn_w[i].empty() || w.gn_b[i].empty()) return false;
    }
    std::snprintf(name, sizeof(name), "octo.obs.%s.patch_embd.weight", view); w.patch_w = g.read_f32(name);
    std::snprintf(name, sizeof(name), "octo.obs.%s.patch_embd.bias",   view); w.patch_b = g.read_f32(name);
    std::snprintf(name, sizeof(name), "octo.obs.%s.proj.weight",       view); w.proj_w  = g.read_f32(name);
    std::snprintf(name, sizeof(name), "octo.obs.%s.proj.bias",         view); w.proj_b  = g.read_f32(name);
    std::snprintf(name, sizeof(name), "octo.obs.%s.pos_embd",          view); w.pos     = g.read_f32(name);
    return !w.patch_w.empty() && !w.patch_b.empty() && !w.proj_w.empty() && !w.proj_b.empty() && !w.pos.empty();
}

static bool read_language_weights(gguf_reader& g, OctoLanguageWeights& w) {
    w.proj_w = g.read_f32("octo.task.language.proj.weight");
    w.proj_b = g.read_f32("octo.task.language.proj.bias");
    w.pos    = g.read_f32("octo.task.language.pos_embd");
    return !w.proj_w.empty() && !w.proj_b.empty() && !w.pos.empty();
}

static bool read_transformer_weights(gguf_reader& g, OctoTransformerWeights& w) {
    char name[160];
    for (int i = 0; i < 12; ++i) {
        OctoBlockWeights& b = w.blocks[(size_t) i];
        auto read = [&](const char * leaf, std::vector<float>& dst) {
            std::snprintf(name, sizeof(name), "octo.blk.%d.%s", i, leaf);
            dst = g.read_f32(name);
            return !dst.empty();
        };
        if (!read("attn_norm.weight", b.attn_norm_w) || !read("attn_norm.bias", b.attn_norm_b) ||
            !read("attn_qkv.weight", b.qkv_w) || !read("attn_qkv.bias", b.qkv_b) ||
            !read("attn_o.weight", b.attn_o_w) || !read("attn_o.bias", b.attn_o_b) ||
            !read("ffn_norm.weight", b.ffn_norm_w) || !read("ffn_norm.bias", b.ffn_norm_b) ||
            !read("ffn_up.weight", b.ffn_up_w) || !read("ffn_up.bias", b.ffn_up_b) ||
            !read("ffn_down.weight", b.ffn_down_w) || !read("ffn_down.bias", b.ffn_down_b)) return false;
    }
    w.output_norm_w = g.read_f32("octo.output_norm.weight");
    w.output_norm_b = g.read_f32("octo.output_norm.bias");
    w.readout_pos   = g.read_f32("octo.readout.action.pos_embd");
    return !w.output_norm_w.empty() && !w.output_norm_b.empty() && w.readout_pos.size() >= 2 * 384;
}

static bool read_t5_weights(gguf_reader& g, OctoT5Weights& w) {
    char name[160];
    for (int i = 0; i < 12; ++i) {
        OctoT5BlockWeights& b = w.blocks[(size_t) i];
        auto read = [&](const char * leaf, std::vector<float>& dst) {
            std::snprintf(name, sizeof(name), "octo.t5.blk.%d.%s", i, leaf);
            dst = g.read_f32(name);
            return !dst.empty();
        };
        if (!read("attn_norm.weight", b.attn_norm_w) ||
            !read("attn_q.weight", b.q_w) || !read("attn_k.weight", b.k_w) ||
            !read("attn_v.weight", b.v_w) || !read("attn_o.weight", b.o_w) ||
            !read("ffn_norm.weight", b.ffn_norm_w) ||
            !read("ffn_up.weight", b.ffn_up_w) || !read("ffn_down.weight", b.ffn_down_w)) return false;
        if (b.attn_norm_w.size() != 768 || b.q_w.size() != 768 * 768 || b.k_w.size() != 768 * 768 ||
            b.v_w.size() != 768 * 768 || b.o_w.size() != 768 * 768 || b.ffn_norm_w.size() != 768 ||
            b.ffn_up_w.size() != 768 * 3072 || b.ffn_down_w.size() != 3072 * 768) {
            std::fprintf(stderr, "vla(octo): T5 block %d weight has unexpected size\n", i);
            return false;
        }
    }
    w.attn_rel_b = g.read_f32("octo.t5.blk.0.attn_rel_b.weight");
    w.output_norm_w = g.read_f32("octo.t5.output_norm.weight");
    return w.attn_rel_b.size() == 12 * 32 && w.output_norm_w.size() == 768;
}

static bool read_diffusion_weights(gguf_reader& g, OctoDiffusionWeights& w) {
    w.time_fourier_w = g.read_f32("octo.head.diffusion.time_fourier.weight");
    w.cond0_w = g.read_f32("octo.head.diffusion.cond.0.weight");
    w.cond0_b = g.read_f32("octo.head.diffusion.cond.0.bias");
    w.cond1_w = g.read_f32("octo.head.diffusion.cond.1.weight");
    w.cond1_b = g.read_f32("octo.head.diffusion.cond.1.bias");
    w.reverse_in_w = g.read_f32("octo.head.diffusion.reverse.in.weight");
    w.reverse_in_b = g.read_f32("octo.head.diffusion.reverse.in.bias");
    char name[160];
    for (int i = 0; i < 3; ++i) {
        OctoDiffusionBlockWeights& b = w.blocks[(size_t) i];
        std::snprintf(name, sizeof(name), "octo.head.diffusion.reverse.blk.%d.ln.weight", i); b.ln_w = g.read_f32(name);
        std::snprintf(name, sizeof(name), "octo.head.diffusion.reverse.blk.%d.ln.bias", i);   b.ln_b = g.read_f32(name);
        std::snprintf(name, sizeof(name), "octo.head.diffusion.reverse.blk.%d.fc1.weight", i); b.fc1_w = g.read_f32(name);
        std::snprintf(name, sizeof(name), "octo.head.diffusion.reverse.blk.%d.fc1.bias", i);   b.fc1_b = g.read_f32(name);
        std::snprintf(name, sizeof(name), "octo.head.diffusion.reverse.blk.%d.fc2.weight", i); b.fc2_w = g.read_f32(name);
        std::snprintf(name, sizeof(name), "octo.head.diffusion.reverse.blk.%d.fc2.bias", i);   b.fc2_b = g.read_f32(name);
        if (b.ln_w.empty() || b.ln_b.empty() || b.fc1_w.empty() || b.fc1_b.empty() ||
            b.fc2_w.empty() || b.fc2_b.empty()) return false;
    }
    w.reverse_out_w = g.read_f32("octo.head.diffusion.reverse.out.weight");
    w.reverse_out_b = g.read_f32("octo.head.diffusion.reverse.out.bias");
    return w.time_fourier_w.size() == 16 && w.cond0_w.size() == 64 * 32 && w.cond0_b.size() == 64 &&
           w.cond1_w.size() == 32 * 64 && w.cond1_b.size() == 32 &&
           w.reverse_in_w.size() == 256 * 444 && w.reverse_in_b.size() == 256 &&
           w.reverse_out_w.size() == 28 * 256 && w.reverse_out_b.size() == 28;
}

static void standardize_conv_weight(const std::vector<float>& src,
                                    int oc, int ic, int kh, int kw,
                                    std::vector<float>& dst) {
    dst.resize(src.size());
    const int n = ic * kh * kw;
    for (int o = 0; o < oc; ++o) {
        double mean = 0.0, var = 0.0;
        const size_t base = (size_t) o * n;
        for (int i = 0; i < n; ++i) mean += src[base + i];
        mean /= n;
        for (int i = 0; i < n; ++i) {
            const double d = (double) src[base + i] - mean;
            var += d * d;
        }
        const float inv = 1.0f / std::sqrt((float) (var / n) + 1e-10f);
        for (int i = 0; i < n; ++i) dst[base + i] = ((float) src[base + i] - (float) mean) * inv;
    }
}

static ggml_tensor * make_4d(ggml_context * ctx, const char * name, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    ggml_tensor * t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne0, ne1, ne2, ne3);
    ggml_set_name(t, name);
    return t;
}

static bool run_one_obs_tokenizer_graph(const OctoObsWeights& w,
                                        const NpyU8& obs,
                                        const NpyU8& task,
                                        int side,
                                        int n_tok,
                                        int window_size,
                                        std::vector<float>& tok,
                                        std::vector<float>& proj,
                                        std::vector<float>& pos) {
    if (obs.shape.size() != 5 || task.shape.size() != 4 ||
        obs.shape[0] != 1 || obs.shape[1] != window_size || obs.shape[2] != 3 ||
        task.shape[0] != 1 || task.shape[1] != 3 ||
        obs.shape[3] != side || obs.shape[4] != side ||
        task.shape[2] != side || task.shape[3] != side) {
        std::fprintf(stderr, "vla(octo): unexpected input image shape for side=%d\n", side);
        return false;
    }
    tok.assign((size_t) 1 * window_size * n_tok * 512, 0.0f);
    proj.assign((size_t) 1 * window_size * n_tok * 384, 0.0f);
    pos.assign((size_t) 1 * window_size * n_tok * 384, 0.0f);

    const int stem_oc[4] = {32, 96, 192, 384};
    const int stem_ic[4] = {6, 32, 96, 192};
    std::vector<float> input((size_t) side * side * 6 * window_size, 0.0f);
    for (int t = 0; t < window_size; ++t) {
        for (int c = 0; c < 3; ++c) {
            for (int yy = 0; yy < side; ++yy) {
                for (int xx = 0; xx < side; ++xx) {
                    const size_t oi = ((((size_t) t * 3 + c) * side + yy) * side + xx);
                    const size_t ti = (((size_t) c * side + yy) * side + xx);
                    input[(((size_t) t * 6 + c) * side + yy) * side + xx] = (float) obs.data[oi] / 127.5f - 1.0f;
                    input[(((size_t) t * 6 + c + 3) * side + yy) * side + xx] = (float) task.data[ti] / 127.5f - 1.0f;
                }
            }
        }
    }

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        std::fprintf(stderr, "vla(octo): ggml_backend_cpu_init failed for tokenizer graph\n");
        return false;
    }
    ggml_backend_cpu_set_n_threads(backend, default_cpu_threads());

    ggml_init_params gp = {(size_t) 96 * 1024 * 1024, nullptr, true};
    ggml_context * ctx = ggml_init(gp);
    if (!ctx) {
        std::fprintf(stderr, "vla(octo): ggml_init(tokenizer graph ctx) failed\n");
        ggml_backend_free(backend);
        return false;
    }

    std::vector<ggml_tensor *> tensors;
    std::vector<std::vector<float>> payloads;
    auto add_payload = [&](ggml_tensor * t, std::vector<float> data) {
        tensors.push_back(t);
        payloads.push_back(std::move(data));
    };
    auto expand_1d = [](const std::vector<float>& src, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
        std::vector<float> out((size_t) ne0 * ne1 * ne2 * ne3, 0.0f);
        for (int64_t i3 = 0; i3 < ne3; ++i3)
            for (int64_t i2 = 0; i2 < ne2; ++i2)
                for (int64_t i1 = 0; i1 < ne1; ++i1)
                    for (int64_t i0 = 0; i0 < ne0; ++i0)
                        out[((size_t) i3 * ne2 * ne1 * ne0) + (size_t) i2 * ne1 * ne0 + (size_t) i1 * ne0 + i0] = src[(size_t) i2];
        return out;
    };

    ggml_tensor * x = make_4d(ctx, "octo.obs.input_norm", side, side, 6, window_size);
    add_payload(x, std::move(input));

    for (int li = 0; li < 4; ++li) {
        std::vector<float> ws;
        standardize_conv_weight(w.conv_w[li], stem_oc[li], stem_ic[li], 3, 3, ws);
        char name[64];
        std::snprintf(name, sizeof(name), "octo.obs.stem.%d.conv.weight_std", li);
        ggml_tensor * cw = make_4d(ctx, name, 3, 3, stem_ic[li], stem_oc[li]);
        add_payload(cw, std::move(ws));
        std::snprintf(name, sizeof(name), "octo.obs.stem.%d.conv.bias", li);
        ggml_tensor * cb = make_4d(ctx, name, 1, 1, stem_oc[li], 1);
        add_payload(cb, expand_1d(w.conv_b[li], 1, 1, stem_oc[li], 1));
        std::snprintf(name, sizeof(name), "octo.obs.stem.%d.gn.weight", li);
        ggml_tensor * gw = make_4d(ctx, name, 1, 1, stem_oc[li], 1);
        add_payload(gw, expand_1d(w.gn_w[li], 1, 1, stem_oc[li], 1));
        std::snprintf(name, sizeof(name), "octo.obs.stem.%d.gn.bias", li);
        ggml_tensor * gb = make_4d(ctx, name, 1, 1, stem_oc[li], 1);
        add_payload(gb, expand_1d(w.gn_b[li], 1, 1, stem_oc[li], 1));

        x = ggml_conv_2d(ctx, cw, x, 2, 2, 1, 1, 1, 1);
        x = ggml_add(ctx, x, cb);
        x = ggml_group_norm(ctx, x, 32, 1e-5f);
        x = ggml_add(ctx, ggml_mul(ctx, x, gw), gb);
        x = ggml_relu(ctx, x);
    }

    ggml_tensor * pw = make_4d(ctx, "octo.obs.patch_embd.weight", 1, 1, 384, 512);
    add_payload(pw, w.patch_w);
    ggml_tensor * pb = make_4d(ctx, "octo.obs.patch_embd.bias", 1, 1, 512, 1);
    add_payload(pb, expand_1d(w.patch_b, 1, 1, 512, 1));
    ggml_tensor * jw = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 384);
    ggml_set_name(jw, "octo.obs.proj.weight");
    add_payload(jw, w.proj_w);
    ggml_tensor * jb = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 384, 1, 1);
    ggml_set_name(jb, "octo.obs.proj.bias");
    add_payload(jb, w.proj_b);
    std::vector<float> pos2(w.pos.begin(), w.pos.begin() + (size_t) window_size * n_tok * 384);
    ggml_tensor * pe = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 384, n_tok, window_size);
    ggml_set_name(pe, "octo.obs.pos_embd.window");
    add_payload(pe, std::move(pos2));

    ggml_tensor * patch = ggml_conv_2d(ctx, pw, x, 1, 1, 0, 0, 1, 1);
    patch = ggml_add(ctx, patch, pb);
    ggml_tensor * tok_t = ggml_cont(ctx, ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_permute(ctx, patch, 1, 2, 0, 3)), 512, n_tok, window_size));
    ggml_set_name(tok_t, "obs.tokenizer.tok");
    ggml_set_output(tok_t);

    ggml_tensor * proj_t = ggml_add(ctx, ggml_mul_mat(ctx, jw, tok_t), jb);
    ggml_set_name(proj_t, "obs.tokenizer.proj");
    ggml_set_output(proj_t);
    ggml_tensor * pos_t = ggml_add(ctx, proj_t, pe);
    ggml_set_name(pos_t, "obs.tokenizer.pos");
    ggml_set_output(pos_t);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 8192, false);
    ggml_build_forward_expand(graph, pos_t);
    ggml_gallocr_t gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!gallocr || !ggml_gallocr_alloc_graph(gallocr, graph)) {
        std::fprintf(stderr, "vla(octo): tokenizer ggml_gallocr_alloc_graph failed\n");
        if (gallocr) ggml_gallocr_free(gallocr);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }
    for (size_t i = 0; i < tensors.size(); ++i) {
        ggml_backend_tensor_set(tensors[i], payloads[i].data(), 0, ggml_nbytes(tensors[i]));
    }
    const ggml_status st = ggml_backend_graph_compute(backend, graph);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "vla(octo): tokenizer ggml_backend_graph_compute failed (%d)\n", (int) st);
        ggml_gallocr_free(gallocr);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    ggml_backend_tensor_get(tok_t, tok.data(), 0, ggml_nbytes(tok_t));
    ggml_backend_tensor_get(proj_t, proj.data(), 0, ggml_nbytes(proj_t));
    ggml_backend_tensor_get(pos_t, pos.data(), 0, ggml_nbytes(pos_t));

    ggml_gallocr_free(gallocr);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return true;
}

// TIP-BUILD-OCTO-GPU-B: predict()-only variant of run_one_obs_tokenizer_graph, computing on
// the model's real `backend` instead of a fresh per-call CPU backend. standardize_conv_weight
// / expand_1d / the pos-embed slice are unchanged (same math), fed from tensor_to_vec() reads
// of the resident tensors (now backend-safe via ggml_backend_tensor_get) -- their outputs are
// still per-call host payloads uploaded via ggml_backend_tensor_set, same as any other input,
// just computed from resident bytes instead of a freshly-read-from-disk vector. `backend` is
// NOT owned by this function -- never freed here.
static bool run_one_obs_tokenizer_graph_resident(ggml_context * ctx_w,
                                                  ggml_backend_t backend,
                                                  const char * view,
                                                  const NpyU8& obs,
                                                  const NpyU8& task,
                                                  int side,
                                                  int n_tok,
                                                  int window_size,
                                                  std::vector<float>& tok,
                                                  std::vector<float>& proj,
                                                  std::vector<float>& pos) {
    if (obs.shape.size() != 5 || task.shape.size() != 4 ||
        obs.shape[0] != 1 || obs.shape[1] != window_size || obs.shape[2] != 3 ||
        task.shape[0] != 1 || task.shape[1] != 3 ||
        obs.shape[3] != side || obs.shape[4] != side ||
        task.shape[2] != side || task.shape[3] != side) {
        std::fprintf(stderr, "vla(octo): unexpected input image shape for side=%d\n", side);
        return false;
    }
    tok.assign((size_t) 1 * window_size * n_tok * 512, 0.0f);
    proj.assign((size_t) 1 * window_size * n_tok * 384, 0.0f);
    pos.assign((size_t) 1 * window_size * n_tok * 384, 0.0f);

    const int stem_oc[4] = {32, 96, 192, 384};
    const int stem_ic[4] = {6, 32, 96, 192};
    std::vector<float> input((size_t) side * side * 6 * window_size, 0.0f);
    for (int t = 0; t < window_size; ++t) {
        for (int c = 0; c < 3; ++c) {
            for (int yy = 0; yy < side; ++yy) {
                for (int xx = 0; xx < side; ++xx) {
                    const size_t oi = ((((size_t) t * 3 + c) * side + yy) * side + xx);
                    const size_t ti = (((size_t) c * side + yy) * side + xx);
                    input[(((size_t) t * 6 + c) * side + yy) * side + xx] = (float) obs.data[oi] / 127.5f - 1.0f;
                    input[(((size_t) t * 6 + c + 3) * side + yy) * side + xx] = (float) task.data[ti] / 127.5f - 1.0f;
                }
            }
        }
    }

    ggml_init_params gp = {(size_t) 96 * 1024 * 1024, nullptr, true};
    ggml_context * ctx = ggml_init(gp);
    if (!ctx) {
        std::fprintf(stderr, "vla(octo): ggml_init(tokenizer graph ctx) failed\n");
        return false;
    }

    std::vector<ggml_tensor *> tensors;
    std::vector<std::vector<float>> payloads;
    auto add_payload = [&](ggml_tensor * t, std::vector<float> data) {
        tensors.push_back(t);
        payloads.push_back(std::move(data));
    };
    auto expand_1d = [](const std::vector<float>& src, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
        std::vector<float> out((size_t) ne0 * ne1 * ne2 * ne3, 0.0f);
        for (int64_t i3 = 0; i3 < ne3; ++i3)
            for (int64_t i2 = 0; i2 < ne2; ++i2)
                for (int64_t i1 = 0; i1 < ne1; ++i1)
                    for (int64_t i0 = 0; i0 < ne0; ++i0)
                        out[((size_t) i3 * ne2 * ne1 * ne0) + (size_t) i2 * ne1 * ne0 + (size_t) i1 * ne0 + i0] = src[(size_t) i2];
        return out;
    };

    ggml_tensor * x = make_4d(ctx, "octo.obs.input_norm", side, side, 6, window_size);
    add_payload(x, std::move(input));

    char rname[160];
    bool ok = true;
    for (int li = 0; li < 4 && ok; ++li) {
        std::snprintf(rname, sizeof(rname), "octo.obs.%s.stem.%d.conv.weight", view, li);
        ggml_tensor * conv_w_r = wt(ctx_w, rname);
        std::snprintf(rname, sizeof(rname), "octo.obs.%s.stem.%d.conv.bias", view, li);
        ggml_tensor * conv_b_r = wt(ctx_w, rname);
        std::snprintf(rname, sizeof(rname), "octo.obs.%s.stem.%d.gn.weight", view, li);
        ggml_tensor * gn_w_r = wt(ctx_w, rname);
        std::snprintf(rname, sizeof(rname), "octo.obs.%s.stem.%d.gn.bias", view, li);
        ggml_tensor * gn_b_r = wt(ctx_w, rname);
        if (!conv_w_r || !conv_b_r || !gn_w_r || !gn_b_r) { ok = false; break; }

        std::vector<float> ws;
        standardize_conv_weight(tensor_to_vec(conv_w_r), stem_oc[li], stem_ic[li], 3, 3, ws);
        char name[64];
        std::snprintf(name, sizeof(name), "octo.obs.stem.%d.conv.weight_std", li);
        ggml_tensor * cw = make_4d(ctx, name, 3, 3, stem_ic[li], stem_oc[li]);
        add_payload(cw, std::move(ws));
        std::snprintf(name, sizeof(name), "octo.obs.stem.%d.conv.bias", li);
        ggml_tensor * cb = make_4d(ctx, name, 1, 1, stem_oc[li], 1);
        add_payload(cb, expand_1d(tensor_to_vec(conv_b_r), 1, 1, stem_oc[li], 1));
        std::snprintf(name, sizeof(name), "octo.obs.stem.%d.gn.weight", li);
        ggml_tensor * gw = make_4d(ctx, name, 1, 1, stem_oc[li], 1);
        add_payload(gw, expand_1d(tensor_to_vec(gn_w_r), 1, 1, stem_oc[li], 1));
        std::snprintf(name, sizeof(name), "octo.obs.stem.%d.gn.bias", li);
        ggml_tensor * gb = make_4d(ctx, name, 1, 1, stem_oc[li], 1);
        add_payload(gb, expand_1d(tensor_to_vec(gn_b_r), 1, 1, stem_oc[li], 1));

        x = ggml_conv_2d(ctx, cw, x, 2, 2, 1, 1, 1, 1);
        x = ggml_add(ctx, x, cb);
        x = ggml_group_norm(ctx, x, 32, 1e-5f);
        x = ggml_add(ctx, ggml_mul(ctx, x, gw), gb);
        x = ggml_relu(ctx, x);
    }
    if (!ok) {
        ggml_free(ctx);
        return false;
    }

    std::snprintf(rname, sizeof(rname), "octo.obs.%s.patch_embd.weight", view);
    ggml_tensor * patch_w_r = wt(ctx_w, rname);
    std::snprintf(rname, sizeof(rname), "octo.obs.%s.patch_embd.bias", view);
    ggml_tensor * patch_b_r = wt(ctx_w, rname);
    std::snprintf(rname, sizeof(rname), "octo.obs.%s.proj.weight", view);
    ggml_tensor * proj_w_r = wt(ctx_w, rname);
    std::snprintf(rname, sizeof(rname), "octo.obs.%s.proj.bias", view);
    ggml_tensor * proj_b_r = wt(ctx_w, rname);
    std::snprintf(rname, sizeof(rname), "octo.obs.%s.pos_embd", view);
    ggml_tensor * pos_r = wt(ctx_w, rname);
    if (!patch_w_r || !patch_b_r || !proj_w_r || !proj_b_r || !pos_r) {
        ggml_free(ctx);
        return false;
    }

    ggml_tensor * pw = make_4d(ctx, "octo.obs.patch_embd.weight", 1, 1, 384, 512);
    add_payload(pw, tensor_to_vec(patch_w_r));
    ggml_tensor * pb = make_4d(ctx, "octo.obs.patch_embd.bias", 1, 1, 512, 1);
    add_payload(pb, expand_1d(tensor_to_vec(patch_b_r), 1, 1, 512, 1));
    ggml_tensor * jw = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 384);
    ggml_set_name(jw, "octo.obs.proj.weight");
    add_payload(jw, tensor_to_vec(proj_w_r));
    ggml_tensor * jb = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 384, 1, 1);
    ggml_set_name(jb, "octo.obs.proj.bias");
    add_payload(jb, tensor_to_vec(proj_b_r));
    const std::vector<float> pos_full = tensor_to_vec(pos_r);
    std::vector<float> pos2(pos_full.begin(), pos_full.begin() + (size_t) window_size * n_tok * 384);
    ggml_tensor * pe = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 384, n_tok, window_size);
    ggml_set_name(pe, "octo.obs.pos_embd.window");
    add_payload(pe, std::move(pos2));

    ggml_tensor * patch = ggml_conv_2d(ctx, pw, x, 1, 1, 0, 0, 1, 1);
    patch = ggml_add(ctx, patch, pb);
    ggml_tensor * tok_t = ggml_cont(ctx, ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_permute(ctx, patch, 1, 2, 0, 3)), 512, n_tok, window_size));
    ggml_set_name(tok_t, "obs.tokenizer.tok");
    ggml_set_output(tok_t);

    ggml_tensor * proj_t = ggml_add(ctx, ggml_mul_mat(ctx, jw, tok_t), jb);
    ggml_set_name(proj_t, "obs.tokenizer.proj");
    ggml_set_output(proj_t);
    ggml_tensor * pos_t = ggml_add(ctx, proj_t, pe);
    ggml_set_name(pos_t, "obs.tokenizer.pos");
    ggml_set_output(pos_t);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 8192, false);
    ggml_build_forward_expand(graph, pos_t);
    ggml_gallocr_t gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!gallocr || !ggml_gallocr_alloc_graph(gallocr, graph)) {
        std::fprintf(stderr, "vla(octo): tokenizer ggml_gallocr_alloc_graph failed\n");
        if (gallocr) ggml_gallocr_free(gallocr);
        ggml_free(ctx);
        return false;
    }
    for (size_t i = 0; i < tensors.size(); ++i) {
        ggml_backend_tensor_set(tensors[i], payloads[i].data(), 0, ggml_nbytes(tensors[i]));
    }
    const ggml_status st = ggml_backend_graph_compute(backend, graph);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "vla(octo): tokenizer ggml_backend_graph_compute failed (%d)\n", (int) st);
        ggml_gallocr_free(gallocr);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_get(tok_t, tok.data(), 0, ggml_nbytes(tok_t));
    ggml_backend_tensor_get(proj_t, proj.data(), 0, ggml_nbytes(proj_t));
    ggml_backend_tensor_get(pos_t, pos.data(), 0, ggml_nbytes(pos_t));

    ggml_gallocr_free(gallocr);
    ggml_free(ctx);
    return true;
}

static bool run_language_graph(const OctoLanguageWeights& w,
                               const NpyF32& t5,
                               int window_size,
                               std::vector<float>& proj,
                               std::vector<float>& pos,
                               std::vector<float>& repeated) {
    if (t5.shape.size() != 3 || t5.shape[0] != 1 || t5.shape[1] != 16 || t5.shape[2] != 768) {
        std::fprintf(stderr, "vla(octo): expected T5 inject shape [1,16,768]\n");
        return false;
    }
    proj.assign((size_t) 1 * 16 * 384, 0.0f);
    pos.assign((size_t) 1 * 16 * 384, 0.0f);
    repeated.assign((size_t) 1 * window_size * 16 * 384, 0.0f);

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        std::fprintf(stderr, "vla(octo): ggml_backend_cpu_init failed for language graph\n");
        return false;
    }
    ggml_backend_cpu_set_n_threads(backend, default_cpu_threads());

    ggml_init_params gp = {(size_t) 8 * 1024 * 1024, nullptr, true};
    ggml_context * ctx = ggml_init(gp);
    if (!ctx) {
        std::fprintf(stderr, "vla(octo): ggml_init(language graph ctx) failed\n");
        ggml_backend_free(backend);
        return false;
    }

    std::vector<ggml_tensor *> tensors;
    std::vector<std::vector<float>> payloads;
    auto add_payload = [&](ggml_tensor * t, std::vector<float> data) {
        tensors.push_back(t);
        payloads.push_back(std::move(data));
    };

    ggml_tensor * inp = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 768, 16, 1);
    ggml_set_name(inp, "octo.task.language.t5_inject");
    add_payload(inp, t5.data);
    ggml_tensor * jw = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 768, 384);
    ggml_set_name(jw, "octo.task.language.proj.weight");
    add_payload(jw, w.proj_w);
    ggml_tensor * jb = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 384, 1, 1);
    ggml_set_name(jb, "octo.task.language.proj.bias");
    add_payload(jb, w.proj_b);
    ggml_tensor * pe = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 384, 16, 1);
    ggml_set_name(pe, "octo.task.language.pos_embd");
    add_payload(pe, w.pos);

    ggml_tensor * proj_t = ggml_add(ctx, ggml_mul_mat(ctx, jw, inp), jb);
    ggml_set_name(proj_t, "task_language.proj");
    ggml_set_output(proj_t);
    ggml_tensor * pos_t = ggml_add(ctx, proj_t, pe);
    ggml_set_name(pos_t, "task_language.pos");
    ggml_set_output(pos_t);
    ggml_tensor * repeated_t = ggml_repeat_4d(ctx, pos_t, 384, 16, window_size, 1);
    ggml_set_name(repeated_t, "obs_task_language.repeated");
    ggml_set_output(repeated_t);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 1024, false);
    ggml_build_forward_expand(graph, repeated_t);
    ggml_gallocr_t gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!gallocr || !ggml_gallocr_alloc_graph(gallocr, graph)) {
        std::fprintf(stderr, "vla(octo): language ggml_gallocr_alloc_graph failed\n");
        if (gallocr) ggml_gallocr_free(gallocr);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }
    for (size_t i = 0; i < tensors.size(); ++i) {
        ggml_backend_tensor_set(tensors[i], payloads[i].data(), 0, ggml_nbytes(tensors[i]));
    }
    const ggml_status st = ggml_backend_graph_compute(backend, graph);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "vla(octo): language ggml_backend_graph_compute failed (%d)\n", (int) st);
        ggml_gallocr_free(gallocr);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    ggml_backend_tensor_get(proj_t, proj.data(), 0, ggml_nbytes(proj_t));
    ggml_backend_tensor_get(pos_t, pos.data(), 0, ggml_nbytes(pos_t));
    ggml_backend_tensor_get(repeated_t, repeated.data(), 0, ggml_nbytes(repeated_t));

    ggml_gallocr_free(gallocr);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return true;
}

// TIP-BUILD-OCTO-GPU-A: predict()-only variant of run_language_graph. proj_w/proj_b/pos are
// used directly as-is by the original (no host-side transform), so they're referenced straight
// from the resident context -- no copy, no per-call upload. `inp` (the T5 encoder's per-call
// output) is genuinely new data each call and still gets uploaded exactly as before.
static bool run_language_graph_resident(ggml_context * ctx_w, ggml_backend_t backend,
                                        const NpyF32& t5,
                                        int window_size,
                                        std::vector<float>& proj,
                                        std::vector<float>& pos,
                                        std::vector<float>& repeated) {
    if (t5.shape.size() != 3 || t5.shape[0] != 1 || t5.shape[1] != 16 || t5.shape[2] != 768) {
        std::fprintf(stderr, "vla(octo): expected T5 inject shape [1,16,768]\n");
        return false;
    }
    proj.assign((size_t) 1 * 16 * 384, 0.0f);
    pos.assign((size_t) 1 * 16 * 384, 0.0f);
    repeated.assign((size_t) 1 * window_size * 16 * 384, 0.0f);

    ggml_tensor * jw = wt(ctx_w, "octo.task.language.proj.weight");
    ggml_tensor * jb = wt(ctx_w, "octo.task.language.proj.bias");
    ggml_tensor * pe = wt(ctx_w, "octo.task.language.pos_embd");
    if (!jw || !jb || !pe) return false;

    ggml_init_params gp = {(size_t) 8 * 1024 * 1024, nullptr, true};
    ggml_context * ctx = ggml_init(gp);
    if (!ctx) {
        std::fprintf(stderr, "vla(octo): ggml_init(language graph ctx) failed\n");
        return false;
    }

    ggml_tensor * inp = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 768, 16, 1);
    ggml_set_name(inp, "octo.task.language.t5_inject");

    ggml_tensor * proj_t = ggml_add(ctx, ggml_mul_mat(ctx, jw, inp), jb);
    ggml_set_name(proj_t, "task_language.proj");
    ggml_set_output(proj_t);
    ggml_tensor * pos_t = ggml_add(ctx, proj_t, pe);
    ggml_set_name(pos_t, "task_language.pos");
    ggml_set_output(pos_t);
    ggml_tensor * repeated_t = ggml_repeat_4d(ctx, pos_t, 384, 16, window_size, 1);
    ggml_set_name(repeated_t, "obs_task_language.repeated");
    ggml_set_output(repeated_t);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 1024, false);
    ggml_build_forward_expand(graph, repeated_t);
    ggml_gallocr_t gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!gallocr || !ggml_gallocr_alloc_graph(gallocr, graph)) {
        std::fprintf(stderr, "vla(octo): language ggml_gallocr_alloc_graph failed\n");
        if (gallocr) ggml_gallocr_free(gallocr);
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_set(inp, t5.data.data(), 0, ggml_nbytes(inp));
    const ggml_status st = ggml_backend_graph_compute(backend, graph);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "vla(octo): language ggml_backend_graph_compute failed (%d)\n", (int) st);
        ggml_gallocr_free(gallocr);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_get(proj_t, proj.data(), 0, ggml_nbytes(proj_t));
    ggml_backend_tensor_get(pos_t, pos.data(), 0, ggml_nbytes(pos_t));
    ggml_backend_tensor_get(repeated_t, repeated.data(), 0, ggml_nbytes(repeated_t));

    ggml_gallocr_free(gallocr);
    ggml_free(ctx);
    return true;
}

// T5 relative-position bucket, encoder self-attention (bidirectional=true).
// Matches HF T5Attention._relative_position_bucket / llama.cpp's llama_relative_position_bucket:
// relative_position = key_pos - query_pos, 32 buckets, max_distance 128.
static int32_t t5_relative_position_bucket(int32_t query_pos, int32_t key_pos, int32_t n_buckets, int32_t max_distance) {
    const int32_t nb = n_buckets / 2;
    const int32_t relative_position = key_pos - query_pos;
    int32_t bucket = (relative_position > 0) ? nb : 0;
    const int32_t rp = std::abs(relative_position);
    const int32_t max_exact = nb / 2;
    if (rp < max_exact) {
        bucket += rp;
    } else {
        const float v = (float) max_exact + std::log((float) rp / (float) max_exact) /
                         std::log((float) max_distance / (float) max_exact) * (float) (nb - max_exact);
        int32_t rp_large = (int32_t) std::floor(v);
        rp_large = std::min(rp_large, nb - 1);
        bucket += rp_large;
    }
    return bucket;
}

// T5-base encoder-only forward as a ggml graph: embedding lookup (host-side row fetch) -> 12x
// [T5LayerNorm(RMS) -> self-attn (shared relative-position bias + padding mask, no query scaling)
// -> residual -> T5LayerNorm -> DenseReluDense(ReLU) -> residual] -> final T5LayerNorm.
static bool run_t5_encoder_graph(gguf_reader& g,
                                 OctoT5Weights& w,
                                 const std::vector<int32_t>& input_ids,
                                 const std::vector<int32_t>& attention_mask,
                                 std::vector<float>& t5_out) {
    constexpr int hidden = 768;
    constexpr int heads = 12;
    constexpr int head_dim = 64;
    constexpr int ffn = 3072;
    constexpr int seq = 16;
    constexpr int n_buckets = 32;
    constexpr int max_distance = 128;
    constexpr float ln_eps = 1e-6f;
    if (input_ids.size() != seq || attention_mask.size() != seq) {
        std::fprintf(stderr, "vla(octo): T5 encoder expected %d input_ids/attention_mask\n", seq);
        return false;
    }

    std::vector<float> embed((size_t) hidden * seq);
    if (!g.fetch_rows_f32("octo.t5.tok_embd.weight", input_ids, embed.data(), hidden)) return false;

    std::vector<int32_t> bucket_idx((size_t) seq * seq);
    std::vector<float> padmask((size_t) seq * seq);
    for (int j = 0; j < seq; ++j) {          // query
        for (int i = 0; i < seq; ++i) {      // key
            bucket_idx[(size_t) j * seq + i] = t5_relative_position_bucket(j, i, n_buckets, max_distance);
            padmask[(size_t) j * seq + i] = attention_mask[(size_t) i] != 0 ? 0.0f : -FLT_MAX;
        }
    }

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        std::fprintf(stderr, "vla(octo): ggml_backend_cpu_init failed for T5 encoder graph\n");
        return false;
    }
    ggml_backend_cpu_set_n_threads(backend, default_cpu_threads());
    ggml_init_params gp = {(size_t) 32 * 1024 * 1024, nullptr, true};
    ggml_context * ctx = ggml_init(gp);
    if (!ctx) {
        ggml_backend_free(backend);
        return false;
    }

    std::vector<ggml_tensor *> tensors;
    std::vector<std::vector<float>> payloads_f32;
    std::vector<std::vector<int32_t>> payloads_i32;
    auto add_f32 = [&](ggml_tensor * t, std::vector<float> data) {
        tensors.push_back(t);
        payloads_f32.push_back(std::move(data));
        return t;
    };
    auto add_i32 = [&](ggml_tensor * t, std::vector<int32_t> data) {
        tensors.push_back(t);
        payloads_i32.push_back(std::move(data));
        return t;
    };

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, seq);
    ggml_set_name(x, "octo.t5.input_embed");
    add_f32(x, embed);

    ggml_tensor * bucket = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, seq, seq);
    ggml_set_name(bucket, "octo.t5.pos_bucket");
    add_i32(bucket, bucket_idx);
    ggml_tensor * rel_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, heads, n_buckets);
    ggml_set_name(rel_b, "octo.t5.attn_rel_b");
    add_f32(rel_b, std::move(w.attn_rel_b));
    ggml_tensor * padmask_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, seq, seq);
    ggml_set_name(padmask_t, "octo.t5.padmask");
    add_f32(padmask_t, padmask);

    ggml_tensor * pos_bucket_1d = ggml_reshape_1d(ctx, bucket, (int64_t) seq * seq);
    ggml_tensor * pos_bias = ggml_get_rows(ctx, rel_b, pos_bucket_1d);
    pos_bias = ggml_reshape_3d(ctx, pos_bias, heads, seq, seq);
    pos_bias = ggml_cont(ctx, ggml_permute(ctx, pos_bias, 2, 0, 1, 3));
    ggml_tensor * mask = ggml_add(ctx, pos_bias, padmask_t);

    char name[160];
    for (int i = 0; i < 12; ++i) {
        OctoT5BlockWeights& b = w.blocks[(size_t) i];
        auto n1w = add_f32(ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden), std::move(b.attn_norm_w));
        ggml_set_name(n1w, (std::snprintf(name, sizeof(name), "octo.t5.blk.%d.attn_norm.weight", i), name));
        auto Wq = add_f32(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, hidden), std::move(b.q_w));
        ggml_set_name(Wq, (std::snprintf(name, sizeof(name), "octo.t5.blk.%d.attn_q.weight", i), name));
        auto Wk = add_f32(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, hidden), std::move(b.k_w));
        ggml_set_name(Wk, (std::snprintf(name, sizeof(name), "octo.t5.blk.%d.attn_k.weight", i), name));
        auto Wv = add_f32(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, hidden), std::move(b.v_w));
        ggml_set_name(Wv, (std::snprintf(name, sizeof(name), "octo.t5.blk.%d.attn_v.weight", i), name));
        auto Wo = add_f32(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, hidden), std::move(b.o_w));
        ggml_set_name(Wo, (std::snprintf(name, sizeof(name), "octo.t5.blk.%d.attn_o.weight", i), name));
        auto n2w = add_f32(ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden), std::move(b.ffn_norm_w));
        ggml_set_name(n2w, (std::snprintf(name, sizeof(name), "octo.t5.blk.%d.ffn_norm.weight", i), name));
        auto Wup = add_f32(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, ffn), std::move(b.ffn_up_w));
        ggml_set_name(Wup, (std::snprintf(name, sizeof(name), "octo.t5.blk.%d.ffn_up.weight", i), name));
        auto Wdown = add_f32(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ffn, hidden), std::move(b.ffn_down_w));
        ggml_set_name(Wdown, (std::snprintf(name, sizeof(name), "octo.t5.blk.%d.ffn_down.weight", i), name));

        ggml_tensor * n1 = ggml_mul(ctx, ggml_rms_norm(ctx, x, ln_eps), n1w);
        ggml_tensor * Q = ggml_mul_mat(ctx, Wq, n1);
        ggml_tensor * K = ggml_mul_mat(ctx, Wk, n1);
        ggml_tensor * V = ggml_mul_mat(ctx, Wv, n1);
        ggml_tensor * Qh = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, Q, head_dim, heads, seq), 0, 2, 1, 3));
        ggml_tensor * Kh = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, K, head_dim, heads, seq), 0, 2, 1, 3));
        ggml_tensor * Vh = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, V, head_dim, heads, seq), 1, 2, 0, 3));
        ggml_tensor * scores = ggml_mul_mat(ctx, Kh, Qh);
        ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
        ggml_tensor * probs = ggml_soft_max_ext(ctx, scores, mask, 1.0f, 0.0f);  // T5: no 1/sqrt(d_k) scaling
        ggml_tensor * attended = ggml_mul_mat(ctx, Vh, probs);
        ggml_tensor * merged = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, attended, 0, 2, 1, 3)), hidden, seq);
        ggml_tensor * attn_out = ggml_mul_mat(ctx, Wo, merged);
        x = ggml_add(ctx, x, attn_out);

        ggml_tensor * n2 = ggml_mul(ctx, ggml_rms_norm(ctx, x, ln_eps), n2w);
        ggml_tensor * h = ggml_relu(ctx, ggml_mul_mat(ctx, Wup, n2));
        ggml_tensor * ffn_out = ggml_mul_mat(ctx, Wdown, h);
        x = ggml_add(ctx, x, ffn_out);
    }

    ggml_tensor * outw = add_f32(ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden), std::move(w.output_norm_w));
    ggml_set_name(outw, "octo.t5.output_norm.weight");
    ggml_tensor * out = ggml_mul(ctx, ggml_rms_norm(ctx, x, ln_eps), outw);
    ggml_set_name(out, "t5.out");
    ggml_set_output(out);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 4096, false);
    ggml_build_forward_expand(graph, out);
    ggml_gallocr_t gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!gallocr || !ggml_gallocr_alloc_graph(gallocr, graph)) {
        std::fprintf(stderr, "vla(octo): T5 encoder ggml_gallocr_alloc_graph failed\n");
        if (gallocr) ggml_gallocr_free(gallocr);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }
    size_t fi = 0, ii = 0;
    for (ggml_tensor * t : tensors) {
        if (t->type == GGML_TYPE_I32) {
            ggml_backend_tensor_set(t, payloads_i32[ii].data(), 0, ggml_nbytes(t));
            ++ii;
        } else {
            ggml_backend_tensor_set(t, payloads_f32[fi].data(), 0, ggml_nbytes(t));
            ++fi;
        }
    }
    const ggml_status st = ggml_backend_graph_compute(backend, graph);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "vla(octo): T5 encoder ggml_backend_graph_compute failed (%d)\n", (int) st);
        ggml_gallocr_free(gallocr);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }
    t5_out.resize((size_t) hidden * seq);
    ggml_backend_tensor_get(out, t5_out.data(), 0, ggml_nbytes(out));

    ggml_gallocr_free(gallocr);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return true;
}

// TIP-BUILD-OCTO-GPU-B: predict()-only variant of run_t5_encoder_graph, computing on the
// model's real `backend` (CUDA when available, CPU otherwise) instead of a fresh per-call
// CPU backend. The 12 blocks' attn/ffn weights, attn_rel_b, and output_norm are referenced
// directly from the resident context (no copy, no per-call upload, on whatever device
// `backend` lives on). The embedding lookup is now an in-graph ggml_get_rows against the
// resident octo.t5.tok_embd.weight tensor (same op already used for the relative-position-
// bias gather two lines below) instead of a host-side row-copy loop -- this runs on `backend`
// too, so on a CUDA build the gather itself happens on-device, no host round-trip. `backend`
// is NOT owned by this function (it's the model's shared backend) -- never freed here.
static bool run_t5_encoder_graph_resident(ggml_context * ctx_w, ggml_backend_t backend,
                                          const std::vector<int32_t>& input_ids,
                                          const std::vector<int32_t>& attention_mask,
                                          std::vector<float>& t5_out) {
    constexpr int hidden = 768;
    constexpr int heads = 12;
    constexpr int head_dim = 64;
    constexpr int seq = 16;
    constexpr int n_buckets = 32;
    constexpr int max_distance = 128;
    constexpr float ln_eps = 1e-6f;
    if (input_ids.size() != seq || attention_mask.size() != seq) {
        std::fprintf(stderr, "vla(octo): T5 encoder expected %d input_ids/attention_mask\n", seq);
        return false;
    }

    ggml_tensor * tok_embd_r = wt(ctx_w, "octo.t5.tok_embd.weight");
    ggml_tensor * rel_b_r = wt(ctx_w, "octo.t5.blk.0.attn_rel_b.weight");
    ggml_tensor * outw_r = wt(ctx_w, "octo.t5.output_norm.weight");
    if (!tok_embd_r || !rel_b_r || !outw_r) return false;
    char rname[160];
    ggml_tensor * blk_w[12][8];  // attn_norm, q, k, v, o, ffn_norm, ffn_up, ffn_down
    const char * leaves[8] = {"attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
                              "attn_o.weight", "ffn_norm.weight", "ffn_up.weight", "ffn_down.weight"};
    for (int i = 0; i < 12; ++i) {
        for (int j = 0; j < 8; ++j) {
            std::snprintf(rname, sizeof(rname), "octo.t5.blk.%d.%s", i, leaves[j]);
            blk_w[i][j] = wt(ctx_w, rname);
            if (!blk_w[i][j]) return false;
        }
    }

    std::vector<int32_t> bucket_idx((size_t) seq * seq);
    std::vector<float> padmask((size_t) seq * seq);
    for (int j = 0; j < seq; ++j) {          // query
        for (int i = 0; i < seq; ++i) {      // key
            bucket_idx[(size_t) j * seq + i] = t5_relative_position_bucket(j, i, n_buckets, max_distance);
            padmask[(size_t) j * seq + i] = attention_mask[(size_t) i] != 0 ? 0.0f : -FLT_MAX;
        }
    }

    ggml_init_params gp = {(size_t) 32 * 1024 * 1024, nullptr, true};
    ggml_context * ctx = ggml_init(gp);
    if (!ctx) return false;

    std::vector<ggml_tensor *> tensors;
    std::vector<std::vector<float>> payloads_f32;
    std::vector<std::vector<int32_t>> payloads_i32;
    auto add_f32 = [&](ggml_tensor * t, std::vector<float> data) {
        tensors.push_back(t);
        payloads_f32.push_back(std::move(data));
        return t;
    };
    auto add_i32 = [&](ggml_tensor * t, std::vector<int32_t> data) {
        tensors.push_back(t);
        payloads_i32.push_back(std::move(data));
        return t;
    };

    ggml_tensor * input_ids_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq);
    ggml_set_name(input_ids_t, "octo.t5.input_ids");
    add_i32(input_ids_t, input_ids);
    ggml_tensor * x = ggml_get_rows(ctx, tok_embd_r, input_ids_t);
    ggml_set_name(x, "octo.t5.input_embed");

    ggml_tensor * bucket = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, seq, seq);
    ggml_set_name(bucket, "octo.t5.pos_bucket");
    add_i32(bucket, bucket_idx);
    ggml_tensor * rel_b = rel_b_r;
    ggml_tensor * padmask_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, seq, seq);
    ggml_set_name(padmask_t, "octo.t5.padmask");
    add_f32(padmask_t, padmask);

    ggml_tensor * pos_bucket_1d = ggml_reshape_1d(ctx, bucket, (int64_t) seq * seq);
    ggml_tensor * pos_bias = ggml_get_rows(ctx, rel_b, pos_bucket_1d);
    pos_bias = ggml_reshape_3d(ctx, pos_bias, heads, seq, seq);
    pos_bias = ggml_cont(ctx, ggml_permute(ctx, pos_bias, 2, 0, 1, 3));
    ggml_tensor * mask = ggml_add(ctx, pos_bias, padmask_t);

    for (int i = 0; i < 12; ++i) {
        ggml_tensor * n1w = blk_w[i][0];
        ggml_tensor * Wq  = blk_w[i][1];
        ggml_tensor * Wk  = blk_w[i][2];
        ggml_tensor * Wv  = blk_w[i][3];
        ggml_tensor * Wo  = blk_w[i][4];
        ggml_tensor * n2w = blk_w[i][5];
        ggml_tensor * Wup = blk_w[i][6];
        ggml_tensor * Wdown = blk_w[i][7];

        ggml_tensor * n1 = ggml_mul(ctx, ggml_rms_norm(ctx, x, ln_eps), n1w);
        ggml_tensor * Q = ggml_mul_mat(ctx, Wq, n1);
        ggml_tensor * K = ggml_mul_mat(ctx, Wk, n1);
        ggml_tensor * V = ggml_mul_mat(ctx, Wv, n1);
        ggml_tensor * Qh = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, Q, head_dim, heads, seq), 0, 2, 1, 3));
        ggml_tensor * Kh = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, K, head_dim, heads, seq), 0, 2, 1, 3));
        ggml_tensor * Vh = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, V, head_dim, heads, seq), 1, 2, 0, 3));
        ggml_tensor * scores = ggml_mul_mat(ctx, Kh, Qh);
        ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
        ggml_tensor * probs = ggml_soft_max_ext(ctx, scores, mask, 1.0f, 0.0f);  // T5: no 1/sqrt(d_k) scaling
        ggml_tensor * attended = ggml_mul_mat(ctx, Vh, probs);
        ggml_tensor * merged = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, attended, 0, 2, 1, 3)), hidden, seq);
        ggml_tensor * attn_out = ggml_mul_mat(ctx, Wo, merged);
        x = ggml_add(ctx, x, attn_out);

        ggml_tensor * n2 = ggml_mul(ctx, ggml_rms_norm(ctx, x, ln_eps), n2w);
        ggml_tensor * h = ggml_relu(ctx, ggml_mul_mat(ctx, Wup, n2));
        ggml_tensor * ffn_out = ggml_mul_mat(ctx, Wdown, h);
        x = ggml_add(ctx, x, ffn_out);
    }

    ggml_tensor * out = ggml_mul(ctx, ggml_rms_norm(ctx, x, ln_eps), outw_r);
    ggml_set_name(out, "t5.out");
    ggml_set_output(out);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 4096, false);
    ggml_build_forward_expand(graph, out);
    ggml_gallocr_t gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!gallocr || !ggml_gallocr_alloc_graph(gallocr, graph)) {
        std::fprintf(stderr, "vla(octo): T5 encoder ggml_gallocr_alloc_graph failed\n");
        if (gallocr) ggml_gallocr_free(gallocr);
        ggml_free(ctx);
        return false;
    }
    size_t fi = 0, ii = 0;
    for (ggml_tensor * t : tensors) {
        if (t->type == GGML_TYPE_I32) {
            ggml_backend_tensor_set(t, payloads_i32[ii].data(), 0, ggml_nbytes(t));
            ++ii;
        } else {
            ggml_backend_tensor_set(t, payloads_f32[fi].data(), 0, ggml_nbytes(t));
            ++fi;
        }
    }
    const ggml_status st = ggml_backend_graph_compute(backend, graph);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "vla(octo): T5 encoder ggml_backend_graph_compute failed (%d)\n", (int) st);
        ggml_gallocr_free(gallocr);
        ggml_free(ctx);
        return false;
    }
    t5_out.resize((size_t) hidden * seq);
    ggml_backend_tensor_get(out, t5_out.data(), 0, ggml_nbytes(out));

    ggml_gallocr_free(gallocr);
    ggml_free(ctx);
    return true;
}

enum class OctoTokenKind { TASK, OBS, READOUT };

struct OctoTokenMetadata {
    OctoTokenKind kind;
    int timestep;
};

struct OctoTransformerResult {
    std::vector<float> input;
    std::vector<uint8_t> blocked_mask;
    std::array<std::vector<float>, 12> block_outputs;
    std::vector<float> output;
    std::vector<float> task_language;
    std::vector<float> obs_primary;
    std::vector<float> obs_wrist;
    std::vector<float> obs_task_language;
    std::vector<float> readout_action;
};

struct OctoDiffusionSchedule {
    std::array<float, 20> betas{};
    std::array<float, 20> alphas{};
    std::array<float, 20> alpha_hats{};
};

struct OctoDiffusionResult {
    std::vector<float> initial_noise;
    std::vector<uint8_t> action_mask;
    std::vector<uint8_t> flat_action_mask;
    std::array<std::vector<float>, 20> current_x_before;
    std::array<std::vector<float>, 20> pred_eps;
    std::array<std::vector<float>, 20> z;
    std::array<std::vector<float>, 20> after_denoise;
    std::array<std::vector<float>, 20> after_noise_add;
    std::array<std::vector<float>, 20> after_clip;
    std::array<std::vector<float>, 20> after_mask;
    std::vector<float> actions_all_timesteps;
    std::vector<float> final_actions;
};

static bool assemble_transformer_input(const std::vector<float>& task_language,
                                       const std::vector<float>& obs_primary,
                                       const std::vector<float>& obs_wrist,
                                       const std::vector<float>& repeated_language,
                                       const std::vector<float>& readout_pos,
                                       int window_size,
                                       std::vector<float>& input) {
    constexpr int hidden = 384;
    const int seq = octo_seq_len(window_size);
    constexpr int step_tokens = kStepTokens;
    if (task_language.size() != (size_t) 16 * hidden ||
        obs_primary.size() != (size_t) window_size * 256 * hidden ||
        obs_wrist.size() != (size_t) window_size * 64 * hidden ||
        repeated_language.size() != (size_t) window_size * 16 * hidden ||
        readout_pos.size() < (size_t) window_size * hidden) {
        std::fprintf(stderr, "vla(octo): invalid tensor size while assembling block transformer input\n");
        return false;
    }
    input.assign((size_t) seq * hidden, 0.0f);
    std::copy(task_language.begin(), task_language.end(), input.begin());
    for (int t = 0; t < window_size; ++t) {
        const size_t dst = (size_t) (16 + t * step_tokens) * hidden;
        std::copy_n(obs_primary.begin() + (size_t) t * 256 * hidden, (size_t) 256 * hidden, input.begin() + dst);
        std::copy_n(obs_wrist.begin() + (size_t) t * 64 * hidden, (size_t) 64 * hidden, input.begin() + dst + (size_t) 256 * hidden);
        std::copy_n(repeated_language.begin() + (size_t) t * 16 * hidden, (size_t) 16 * hidden,
                    input.begin() + dst + (size_t) 320 * hidden);
        std::copy_n(readout_pos.begin() + (size_t) t * hidden, hidden,
                    input.begin() + dst + (size_t) 336 * hidden);
    }
    return true;
}

static OctoDiffusionSchedule make_cosine_schedule() {
    OctoDiffusionSchedule s;
    constexpr int steps = 20;
    constexpr double ds = 0.008;
    constexpr double pi = 3.141592653589793238462643383279502884;
    std::array<double, steps + 1> alpha_cum{};
    for (int i = 0; i <= steps; ++i) {
        const double t = (double) i / (double) steps;
        const double v = std::cos((t + ds) / (1.0 + ds) * pi * 0.5);
        alpha_cum[(size_t) i] = v * v;
    }
    const double first = alpha_cum[0];
    float cum = 1.0f;
    for (int i = 0; i < steps; ++i) {
        const double a0 = alpha_cum[(size_t) i] / first;
        const double a1 = alpha_cum[(size_t) i + 1] / first;
        const float beta = (float) std::min(std::max(1.0 - a1 / a0, 0.0), 0.999);
        s.betas[(size_t) i] = beta;
        s.alphas[(size_t) i] = 1.0f - beta;
        cum *= s.alphas[(size_t) i];
        s.alpha_hats[(size_t) i] = cum;
    }
    return s;
}

static bool run_score_actor_graph(const OctoDiffusionWeights& w,
                                  const std::vector<float>& readout_action,
                                  const std::vector<float>& noisy_action,
                                  int time_value,
                                  int window_size,
                                  std::vector<float>& pred_eps) {
    constexpr int hidden = 384;
    constexpr int action = 28;
    const int width = window_size;
    constexpr float ln_eps = 1e-6f;
    constexpr float two_pi = 6.2831853071795864769f;
    if (readout_action.size() != (size_t) hidden * width || noisy_action.size() != (size_t) action * width) return false;

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        std::fprintf(stderr, "vla(octo): ggml_backend_cpu_init failed for score actor graph\n");
        return false;
    }
    ggml_backend_cpu_set_n_threads(backend, default_cpu_threads());
    ggml_init_params gp = {(size_t) 16 * 1024 * 1024, nullptr, true};
    ggml_context * ctx = ggml_init(gp);
    if (!ctx) {
        ggml_backend_free(backend);
        return false;
    }

    std::vector<ggml_tensor *> tensors;
    std::vector<std::vector<float>> payloads;
    auto add_payload = [&](ggml_tensor * t, const std::vector<float>& data) {
        tensors.push_back(t);
        payloads.push_back(data);
        return t;
    };
    auto make_1d = [&](const char * name, int64_t ne0, const std::vector<float>& data) {
        ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne0);
        ggml_set_name(t, name);
        return add_payload(t, data);
    };
    auto make_2d = [&](const char * name, int64_t ne0, int64_t ne1, const std::vector<float>& data) {
        ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
        ggml_set_name(t, name);
        return add_payload(t, data);
    };

    ggml_tensor * time_w = make_2d("octo.head.diffusion.time_fourier.weight", 1, 16, w.time_fourier_w);
    ggml_tensor * c0w = make_2d("octo.head.diffusion.cond.0.weight", 32, 64, w.cond0_w);
    ggml_tensor * c0b = make_1d("octo.head.diffusion.cond.0.bias", 64, w.cond0_b);
    ggml_tensor * c1w = make_2d("octo.head.diffusion.cond.1.weight", 64, 32, w.cond1_w);
    ggml_tensor * c1b = make_1d("octo.head.diffusion.cond.1.bias", 32, w.cond1_b);
    ggml_tensor * rinw = make_2d("octo.head.diffusion.reverse.in.weight", 444, 256, w.reverse_in_w);
    ggml_tensor * rinb = make_1d("octo.head.diffusion.reverse.in.bias", 256, w.reverse_in_b);
    ggml_tensor * routw = make_2d("octo.head.diffusion.reverse.out.weight", 256, 28, w.reverse_out_w);
    ggml_tensor * routb = make_1d("octo.head.diffusion.reverse.out.bias", 28, w.reverse_out_b);

    std::vector<float> time_data(width, (float) time_value);
    ggml_tensor * time = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, width);
    ggml_set_name(time, "action_head.time");
    add_payload(time, time_data);
    ggml_tensor * obs = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, width);
    ggml_set_name(obs, "action_head.readout_embedding");
    add_payload(obs, readout_action);
    ggml_tensor * actions = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, action, width);
    ggml_set_name(actions, "action_head.noisy_action");
    add_payload(actions, noisy_action);

    ggml_tensor * f = ggml_scale(ctx, ggml_mul_mat(ctx, time_w, time), two_pi);
    ggml_tensor * time_ff = ggml_concat(ctx, ggml_cos(ctx, f), ggml_sin(ctx, f), 0);
    ggml_tensor * cond = ggml_silu(ctx, ggml_add(ctx, ggml_mul_mat(ctx, c0w, time_ff), c0b));
    cond = ggml_add(ctx, ggml_mul_mat(ctx, c1w, cond), c1b);
    ggml_tensor * reverse_input = ggml_concat(ctx, ggml_concat(ctx, cond, obs, 0), actions, 0);
    ggml_tensor * x = ggml_add(ctx, ggml_mul_mat(ctx, rinw, reverse_input), rinb);
    char name[160];
    for (int i = 0; i < 3; ++i) {
        const OctoDiffusionBlockWeights& b = w.blocks[(size_t) i];
        auto lnw = make_1d((std::snprintf(name, sizeof(name), "octo.head.diffusion.reverse.blk.%d.ln.weight", i), name), 256, b.ln_w);
        auto lnb = make_1d((std::snprintf(name, sizeof(name), "octo.head.diffusion.reverse.blk.%d.ln.bias", i), name), 256, b.ln_b);
        auto fc1w = make_2d((std::snprintf(name, sizeof(name), "octo.head.diffusion.reverse.blk.%d.fc1.weight", i), name), 256, 1024, b.fc1_w);
        auto fc1b = make_1d((std::snprintf(name, sizeof(name), "octo.head.diffusion.reverse.blk.%d.fc1.bias", i), name), 1024, b.fc1_b);
        auto fc2w = make_2d((std::snprintf(name, sizeof(name), "octo.head.diffusion.reverse.blk.%d.fc2.weight", i), name), 1024, 256, b.fc2_w);
        auto fc2b = make_1d((std::snprintf(name, sizeof(name), "octo.head.diffusion.reverse.blk.%d.fc2.bias", i), name), 256, b.fc2_b);
        ggml_tensor * residual = x;
        ggml_tensor * h = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, ln_eps), lnw), lnb);
        h = ggml_silu(ctx, ggml_add(ctx, ggml_mul_mat(ctx, fc1w, h), fc1b));
        h = ggml_add(ctx, ggml_mul_mat(ctx, fc2w, h), fc2b);
        x = ggml_add(ctx, residual, h);
    }
    ggml_tensor * out = ggml_add(ctx, ggml_mul_mat(ctx, routw, ggml_silu(ctx, x)), routb);
    ggml_set_name(out, "action_head.pred_eps");
    ggml_set_output(out);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 2048, false);
    ggml_build_forward_expand(graph, out);
    ggml_gallocr_t gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!gallocr || !ggml_gallocr_alloc_graph(gallocr, graph)) {
        std::fprintf(stderr, "vla(octo): score actor ggml_gallocr_alloc_graph failed\n");
        if (gallocr) ggml_gallocr_free(gallocr);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }
    for (size_t i = 0; i < tensors.size(); ++i) {
        ggml_backend_tensor_set(tensors[i], payloads[i].data(), 0, ggml_nbytes(tensors[i]));
    }
    const ggml_status st = ggml_backend_graph_compute(backend, graph);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "vla(octo): score actor ggml_backend_graph_compute failed (%d)\n", (int) st);
        ggml_gallocr_free(gallocr);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }
    pred_eps.resize((size_t) action * width);
    ggml_backend_tensor_get(out, pred_eps.data(), 0, ggml_nbytes(out));

    ggml_gallocr_free(gallocr);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return true;
}

// TIP-BUILD-OCTO-GPU-A: predict()-only variant of run_score_actor_graph -- called once per
// denoising step (20x per predict()), so this is the highest call-frequency of the 5 stages.
// All 9 fixed weights + the 3 residual blocks' 6 weights each are referenced directly from
// the resident context; only the 3 genuinely-per-call input tensors (time, readout embedding,
// noisy action) still get built + uploaded fresh each call. Graph topology unchanged.
static bool run_score_actor_graph_resident(ggml_context * ctx_w, ggml_backend_t backend,
                                           const std::vector<float>& readout_action,
                                           const std::vector<float>& noisy_action,
                                           int time_value,
                                           int window_size,
                                           std::vector<float>& pred_eps) {
    constexpr int hidden = 384;
    constexpr int action = 28;
    const int width = window_size;
    constexpr float ln_eps = 1e-6f;
    constexpr float two_pi = 6.2831853071795864769f;
    if (readout_action.size() != (size_t) hidden * width || noisy_action.size() != (size_t) action * width) return false;

    ggml_tensor * time_w = wt(ctx_w, "octo.head.diffusion.time_fourier.weight");
    ggml_tensor * c0w = wt(ctx_w, "octo.head.diffusion.cond.0.weight");
    ggml_tensor * c0b = wt(ctx_w, "octo.head.diffusion.cond.0.bias");
    ggml_tensor * c1w = wt(ctx_w, "octo.head.diffusion.cond.1.weight");
    ggml_tensor * c1b = wt(ctx_w, "octo.head.diffusion.cond.1.bias");
    ggml_tensor * rinw = wt(ctx_w, "octo.head.diffusion.reverse.in.weight");
    ggml_tensor * rinb = wt(ctx_w, "octo.head.diffusion.reverse.in.bias");
    ggml_tensor * routw = wt(ctx_w, "octo.head.diffusion.reverse.out.weight");
    ggml_tensor * routb = wt(ctx_w, "octo.head.diffusion.reverse.out.bias");
    if (!time_w || !c0w || !c0b || !c1w || !c1b || !rinw || !rinb || !routw || !routb) return false;

    char rname[160];
    ggml_tensor * blk_w[3][6];  // ln.weight, ln.bias, fc1.weight, fc1.bias, fc2.weight, fc2.bias
    const char * leaves[6] = {"ln.weight", "ln.bias", "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 6; ++j) {
            std::snprintf(rname, sizeof(rname), "octo.head.diffusion.reverse.blk.%d.%s", i, leaves[j]);
            blk_w[i][j] = wt(ctx_w, rname);
            if (!blk_w[i][j]) return false;
        }
    }

    ggml_init_params gp = {(size_t) 16 * 1024 * 1024, nullptr, true};
    ggml_context * ctx = ggml_init(gp);
    if (!ctx) return false;

    std::vector<ggml_tensor *> in_tensors;
    std::vector<std::vector<float>> in_payloads;
    auto add_input = [&](ggml_tensor * t, const std::vector<float>& data) {
        in_tensors.push_back(t);
        in_payloads.push_back(data);
        return t;
    };

    std::vector<float> time_data(width, (float) time_value);
    ggml_tensor * time = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, width);
    ggml_set_name(time, "action_head.time");
    add_input(time, time_data);
    ggml_tensor * obs = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, width);
    ggml_set_name(obs, "action_head.readout_embedding");
    add_input(obs, readout_action);
    ggml_tensor * actions = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, action, width);
    ggml_set_name(actions, "action_head.noisy_action");
    add_input(actions, noisy_action);

    ggml_tensor * f = ggml_scale(ctx, ggml_mul_mat(ctx, time_w, time), two_pi);
    ggml_tensor * time_ff = ggml_concat(ctx, ggml_cos(ctx, f), ggml_sin(ctx, f), 0);
    ggml_tensor * cond = ggml_silu(ctx, ggml_add(ctx, ggml_mul_mat(ctx, c0w, time_ff), c0b));
    cond = ggml_add(ctx, ggml_mul_mat(ctx, c1w, cond), c1b);
    ggml_tensor * reverse_input = ggml_concat(ctx, ggml_concat(ctx, cond, obs, 0), actions, 0);
    ggml_tensor * x = ggml_add(ctx, ggml_mul_mat(ctx, rinw, reverse_input), rinb);
    for (int i = 0; i < 3; ++i) {
        ggml_tensor * lnw = blk_w[i][0];
        ggml_tensor * lnb = blk_w[i][1];
        ggml_tensor * fc1w = blk_w[i][2];
        ggml_tensor * fc1b = blk_w[i][3];
        ggml_tensor * fc2w = blk_w[i][4];
        ggml_tensor * fc2b = blk_w[i][5];
        ggml_tensor * residual = x;
        ggml_tensor * h = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, ln_eps), lnw), lnb);
        h = ggml_silu(ctx, ggml_add(ctx, ggml_mul_mat(ctx, fc1w, h), fc1b));
        h = ggml_add(ctx, ggml_mul_mat(ctx, fc2w, h), fc2b);
        x = ggml_add(ctx, residual, h);
    }
    ggml_tensor * out = ggml_add(ctx, ggml_mul_mat(ctx, routw, ggml_silu(ctx, x)), routb);
    ggml_set_name(out, "action_head.pred_eps");
    ggml_set_output(out);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 2048, false);
    ggml_build_forward_expand(graph, out);
    ggml_gallocr_t gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!gallocr || !ggml_gallocr_alloc_graph(gallocr, graph)) {
        std::fprintf(stderr, "vla(octo): score actor ggml_gallocr_alloc_graph failed\n");
        if (gallocr) ggml_gallocr_free(gallocr);
        ggml_free(ctx);
        return false;
    }
    for (size_t i = 0; i < in_tensors.size(); ++i) {
        ggml_backend_tensor_set(in_tensors[i], in_payloads[i].data(), 0, ggml_nbytes(in_tensors[i]));
    }
    const ggml_status st = ggml_backend_graph_compute(backend, graph);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "vla(octo): score actor ggml_backend_graph_compute failed (%d)\n", (int) st);
        ggml_gallocr_free(gallocr);
        ggml_free(ctx);
        return false;
    }
    pred_eps.resize((size_t) action * width);
    ggml_backend_tensor_get(out, pred_eps.data(), 0, ggml_nbytes(out));

    ggml_gallocr_free(gallocr);
    ggml_free(ctx);
    return true;
}

static bool load_f32_shape(const std::string& path, const std::vector<int64_t>& shape, std::vector<float>& dst) {
    NpyF32 npy;
    if (!parse_npy_f32(path, npy)) return false;
    if (npy.shape != shape) {
        std::fprintf(stderr, "vla(octo): %s unexpected shape\n", path.c_str());
        return false;
    }
    dst = std::move(npy.data);
    return true;
}

static bool run_diffusion_replay(const OctoDiffusionWeights& w,
                                 const std::string& case_dir,
                                 const std::vector<float>& readout_action,
                                 int window_size,
                                 OctoDiffusionResult& result) {
    constexpr int steps = 20;
    const int width = window_size;
    constexpr int action = 28;
    constexpr float max_action = 5.0f;
    if (readout_action.size() != (size_t) 384 * width) return false;
    const std::string prefix = case_dir + "/tensors/action_head.predict_action.";
    if (!load_f32_shape(prefix + "initial_noise.npy", {1, window_size, 28}, result.initial_noise)) return false;
    NpyBool action_mask;
    if (!parse_npy_bool(prefix + "action_mask.npy", action_mask) || action_mask.shape != std::vector<int64_t>({1, window_size, 4, 7})) return false;
    result.action_mask = std::move(action_mask.data);
    NpyBool flat_mask;
    if (!parse_npy_bool(prefix + "flat_action_mask.npy", flat_mask) || flat_mask.shape != std::vector<int64_t>({1, window_size, 28})) return false;
    result.flat_action_mask = std::move(flat_mask.data);

    OctoDiffusionSchedule sched = make_cosine_schedule();
    std::vector<float> x = result.initial_noise;
    for (int step = 0; step < steps; ++step) {
        const int time_value = steps - 1 - step;
        char stem[96];
        std::snprintf(stem, sizeof(stem), "step_%02d.t_%02d.", step, time_value);
        result.current_x_before[(size_t) step] = x;
        if (!run_score_actor_graph(w, readout_action, x, time_value, window_size, result.pred_eps[(size_t) step])) return false;
        if (!load_f32_shape(prefix + stem + "z.npy", {1, window_size, 28}, result.z[(size_t) step])) return false;

        std::vector<float> y((size_t) width * action);
        const float alpha = sched.alphas[(size_t) time_value];
        const float beta = sched.betas[(size_t) time_value];
        const float alpha_hat = sched.alpha_hats[(size_t) time_value];
        const float alpha_1 = 1.0f / std::sqrt(alpha);
        const float alpha_2 = (1.0f - alpha) / std::sqrt(1.0f - alpha_hat);
        for (size_t i = 0; i < y.size(); ++i) y[i] = alpha_1 * (x[i] - alpha_2 * result.pred_eps[(size_t) step][i]);
        result.after_denoise[(size_t) step] = y;
        if (time_value > 0) {
            const float sigma = std::sqrt(beta);
            for (size_t i = 0; i < y.size(); ++i) y[i] += sigma * result.z[(size_t) step][i];
        }
        result.after_noise_add[(size_t) step] = y;
        for (float& v : y) v = std::min(std::max(v, -max_action), max_action);
        result.after_clip[(size_t) step] = y;
        const float masked_noise_scale = std::sqrt(1.0f - alpha_hat);
        for (size_t i = 0; i < y.size(); ++i) {
            if (!result.flat_action_mask[i]) y[i] = masked_noise_scale * result.z[(size_t) step][i];
        }
        result.after_mask[(size_t) step] = y;
        x = std::move(y);
    }

    result.actions_all_timesteps = x;
    // Final action chunk = the LAST timestep's readout (index window_size-1), matching
    // OctoPt's sample_actions() which returns actions[:, -1] after the block-transformer
    // diffusion head runs over the full observation window. For window_size=2 this is
    // exactly the old hardcoded [action, 2*action) slice; for window_size=1 it is the
    // sole timestep's block [0, action).
    const size_t final_begin = (size_t) (window_size - 1) * action;
    const size_t final_end   = (size_t) window_size * action;
    if (final_end > x.size()) {
        std::fprintf(stderr, "vla(octo): final action slice [%zu,%zu) out of bounds (x.size()=%zu)\n",
                     final_begin, final_end, x.size());
        return false;
    }
    result.final_actions.assign(x.begin() + (ptrdiff_t) final_begin, x.begin() + (ptrdiff_t) final_end);
    return true;
}

// TIP-BUILD-OCTO-GPU-B: resident/GPU-wired counterpart of run_diffusion_replay, used ONLY by
// octo_dump_tokenizer_case_resident (the --resident dump path) to get real, noise-matched
// GPU-vs-CPU-golden drift numbers for the diffusion boundaries. Same DDPM math, same golden
// z.npy/initial_noise.npy replay (deterministic, not the live path's fresh RNG) -- the only
// difference from run_diffusion_replay is calling run_score_actor_graph_resident(ctx_w,
// backend, ...) instead of run_score_actor_graph(w, ...), so the diffusion head compute runs
// on `backend` (CUDA when available) using the SAME resident weights live predict() uses.
static bool run_diffusion_replay_resident(ggml_context * ctx_w, ggml_backend_t backend,
                                          const std::string& case_dir,
                                          const std::vector<float>& readout_action,
                                          int window_size,
                                          OctoDiffusionResult& result) {
    constexpr int steps = 20;
    const int width = window_size;
    constexpr int action = 28;
    constexpr float max_action = 5.0f;
    if (readout_action.size() != (size_t) 384 * width) return false;
    const std::string prefix = case_dir + "/tensors/action_head.predict_action.";
    if (!load_f32_shape(prefix + "initial_noise.npy", {1, window_size, 28}, result.initial_noise)) return false;
    NpyBool action_mask;
    if (!parse_npy_bool(prefix + "action_mask.npy", action_mask) || action_mask.shape != std::vector<int64_t>({1, window_size, 4, 7})) return false;
    result.action_mask = std::move(action_mask.data);
    NpyBool flat_mask;
    if (!parse_npy_bool(prefix + "flat_action_mask.npy", flat_mask) || flat_mask.shape != std::vector<int64_t>({1, window_size, 28})) return false;
    result.flat_action_mask = std::move(flat_mask.data);

    OctoDiffusionSchedule sched = make_cosine_schedule();
    std::vector<float> x = result.initial_noise;
    for (int step = 0; step < steps; ++step) {
        const int time_value = steps - 1 - step;
        char stem[96];
        std::snprintf(stem, sizeof(stem), "step_%02d.t_%02d.", step, time_value);
        result.current_x_before[(size_t) step] = x;
        if (!run_score_actor_graph_resident(ctx_w, backend, readout_action, x, time_value, window_size, result.pred_eps[(size_t) step])) return false;
        if (!load_f32_shape(prefix + stem + "z.npy", {1, window_size, 28}, result.z[(size_t) step])) return false;

        std::vector<float> y((size_t) width * action);
        const float alpha = sched.alphas[(size_t) time_value];
        const float beta = sched.betas[(size_t) time_value];
        const float alpha_hat = sched.alpha_hats[(size_t) time_value];
        const float alpha_1 = 1.0f / std::sqrt(alpha);
        const float alpha_2 = (1.0f - alpha) / std::sqrt(1.0f - alpha_hat);
        for (size_t i = 0; i < y.size(); ++i) y[i] = alpha_1 * (x[i] - alpha_2 * result.pred_eps[(size_t) step][i]);
        result.after_denoise[(size_t) step] = y;
        if (time_value > 0) {
            const float sigma = std::sqrt(beta);
            for (size_t i = 0; i < y.size(); ++i) y[i] += sigma * result.z[(size_t) step][i];
        }
        result.after_noise_add[(size_t) step] = y;
        for (float& v : y) v = std::min(std::max(v, -max_action), max_action);
        result.after_clip[(size_t) step] = y;
        const float masked_noise_scale = std::sqrt(1.0f - alpha_hat);
        for (size_t i = 0; i < y.size(); ++i) {
            if (!result.flat_action_mask[i]) y[i] = masked_noise_scale * result.z[(size_t) step][i];
        }
        result.after_mask[(size_t) step] = y;
        x = std::move(y);
    }

    result.actions_all_timesteps = x;
    const size_t final_begin = (size_t) (window_size - 1) * action;
    const size_t final_end   = (size_t) window_size * action;
    if (final_end > x.size()) {
        std::fprintf(stderr, "vla(octo): final action slice [%zu,%zu) out of bounds (x.size()=%zu)\n",
                     final_begin, final_end, x.size());
        return false;
    }
    result.final_actions.assign(x.begin() + (ptrdiff_t) final_begin, x.begin() + (ptrdiff_t) final_end);
    return true;
}

// Same DDPM reverse process as run_diffusion_replay, but samples initial noise
// and per-step z ~ N(0,1) fresh instead of replaying golden-dump noise (live
// inference has no golden dir to replay from). action_mask/flat_action_mask are
// all-true for octo-small-1.5 (real_action_dim == max_action_dim == 7, full
// action_horizon), so the golden masked-noise-substitution step is a no-op here
// and is omitted.
static bool run_diffusion_live(const OctoDiffusionWeights& w,
                               const std::vector<float>& readout_action,
                               int window_size,
                               std::mt19937& rng,
                               std::vector<float>& final_actions,
                               std::vector<float>* initial_noise_out = nullptr) {
    constexpr int steps = 20;
    const int width = window_size;
    constexpr int action = 28;
    constexpr float max_action = 5.0f;
    if (readout_action.size() != (size_t) 384 * width) return false;

    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> x((size_t) width * action);
    for (float& v : x) v = normal(rng);
    if (initial_noise_out) *initial_noise_out = x;  // diagnostic only; sampled above, doesn't perturb rng

    OctoDiffusionSchedule sched = make_cosine_schedule();
    for (int step = 0; step < steps; ++step) {
        const int time_value = steps - 1 - step;
        std::vector<float> pred_eps;
        if (!run_score_actor_graph(w, readout_action, x, time_value, window_size, pred_eps)) return false;

        std::vector<float> y((size_t) width * action);
        const float alpha = sched.alphas[(size_t) time_value];
        const float beta = sched.betas[(size_t) time_value];
        const float alpha_hat = sched.alpha_hats[(size_t) time_value];
        const float alpha_1 = 1.0f / std::sqrt(alpha);
        const float alpha_2 = (1.0f - alpha) / std::sqrt(1.0f - alpha_hat);
        for (size_t i = 0; i < y.size(); ++i) y[i] = alpha_1 * (x[i] - alpha_2 * pred_eps[i]);
        if (time_value > 0) {
            const float sigma = std::sqrt(beta);
            for (size_t i = 0; i < y.size(); ++i) y[i] += sigma * normal(rng);
        }
        for (float& v : y) v = std::min(std::max(v, -max_action), max_action);
        x = std::move(y);
    }

    // See run_diffusion_replay's matching comment: last-timestep readout, generalized
    // from the old hardcoded [action, 2*action) slice.
    const size_t final_begin = (size_t) (window_size - 1) * action;
    const size_t final_end   = (size_t) window_size * action;
    if (final_end > x.size()) {
        std::fprintf(stderr, "vla(octo): final action slice [%zu,%zu) out of bounds (x.size()=%zu)\n",
                     final_begin, final_end, x.size());
        return false;
    }
    final_actions.assign(x.begin() + (ptrdiff_t) final_begin, x.begin() + (ptrdiff_t) final_end);
    return true;
}

// TIP-BUILD-OCTO-GPU-A: predict()-only variant of run_diffusion_live -- same DDPM reverse
// process (noise sampling, denoise, clip), just calls run_score_actor_graph_resident inside
// the 20-step loop instead of run_score_actor_graph, so the diffusion head's weights come
// from the resident context instead of a per-call host-vector re-read.
static bool run_diffusion_live_resident(ggml_context * ctx_w, ggml_backend_t backend,
                                        const std::vector<float>& readout_action,
                                        int window_size,
                                        std::mt19937& rng,
                                        std::vector<float>& final_actions) {
    constexpr int steps = 20;
    const int width = window_size;
    constexpr int action = 28;
    constexpr float max_action = 5.0f;
    if (readout_action.size() != (size_t) 384 * width) return false;

    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> x((size_t) width * action);
    for (float& v : x) v = normal(rng);

    OctoDiffusionSchedule sched = make_cosine_schedule();
    for (int step = 0; step < steps; ++step) {
        const int time_value = steps - 1 - step;
        std::vector<float> pred_eps;
        if (!run_score_actor_graph_resident(ctx_w, backend, readout_action, x, time_value, window_size, pred_eps)) return false;

        std::vector<float> y((size_t) width * action);
        const float alpha = sched.alphas[(size_t) time_value];
        const float beta = sched.betas[(size_t) time_value];
        const float alpha_hat = sched.alpha_hats[(size_t) time_value];
        const float alpha_1 = 1.0f / std::sqrt(alpha);
        const float alpha_2 = (1.0f - alpha) / std::sqrt(1.0f - alpha_hat);
        for (size_t i = 0; i < y.size(); ++i) y[i] = alpha_1 * (x[i] - alpha_2 * pred_eps[i]);
        if (time_value > 0) {
            const float sigma = std::sqrt(beta);
            for (size_t i = 0; i < y.size(); ++i) y[i] += sigma * normal(rng);
        }
        for (float& v : y) v = std::min(std::max(v, -max_action), max_action);
        x = std::move(y);
    }

    // See run_diffusion_replay's matching comment: last-timestep readout, generalized
    // from the old hardcoded [action, 2*action) slice.
    const size_t final_begin = (size_t) (window_size - 1) * action;
    const size_t final_end   = (size_t) window_size * action;
    if (final_end > x.size()) {
        std::fprintf(stderr, "vla(octo): final action slice [%zu,%zu) out of bounds (x.size()=%zu)\n",
                     final_begin, final_end, x.size());
        return false;
    }
    final_actions.assign(x.begin() + (ptrdiff_t) final_begin, x.begin() + (ptrdiff_t) final_end);
    return true;
}

static bool read_kv_u8_array(const gguf_reader& g, const char * key, std::vector<uint8_t>& out) {
    const int64_t id = gguf_find_key(g.gctx, key);
    if (id < 0) {
        std::fprintf(stderr, "vla(octo): missing metadata %s\n", key);
        return false;
    }
    if (gguf_get_kv_type(g.gctx, id) != GGUF_TYPE_ARRAY || gguf_get_arr_type(g.gctx, id) != GGUF_TYPE_UINT8) {
        std::fprintf(stderr, "vla(octo): %s is not a UINT8 array\n", key);
        return false;
    }
    const size_t n = gguf_get_arr_n(g.gctx, id);
    const uint8_t * data = (const uint8_t *) gguf_get_arr_data(g.gctx, id);
    out.assign(data, data + n);
    return true;
}

// Nearest-neighbor resize from interleaved HWC RGB8 (arbitrary size) to a
// dside x dside planar CHW RGB8 buffer, matching the [C,H,W] layout
// run_one_obs_tokenizer_graph expects for its obs/task tensors.
static void resize_to_planar_chw(const uint8_t * src_hwc, int sw, int sh, int dside, std::vector<uint8_t>& dst_chw) {
    dst_chw.assign((size_t) 3 * dside * dside, 0);
    for (int yy = 0; yy < dside; ++yy) {
        const int sy = std::min(sh - 1, (int) ((int64_t) yy * sh / dside));
        for (int xx = 0; xx < dside; ++xx) {
            const int sx = std::min(sw - 1, (int) ((int64_t) xx * sw / dside));
            const uint8_t * px = src_hwc + ((size_t) sy * sw + sx) * 3;
            for (int c = 0; c < 3; ++c) {
                dst_chw[((size_t) c * dside + yy) * dside + xx] = px[c];
            }
        }
    }
}

// Resolves which top-level key of octo.dataset_statistics to un-normalize against when
// the caller didn't pin one down explicitly (dataset_key_in empty). Priority:
//   1. VLA_OCTO_UNNORM_DATASET env var -- the per-checkpoint-config idiom this codebase
//      already uses for things a loaded GGUF can't self-describe (c.f. gr00t's
//      VLA_GR00T_EMBODIMENT); the only channel available to server predict(), which has
//      no per-call CLI flag.
//   2. dataset_statistics has exactly one top-level key -> unambiguous, use it.
//   3. "bridge_dataset" if present -- preserves the historical hardcoded default for the
//      rail-berkeley bridge pretrain checkpoint (whose dataset_statistics has ~25 OXE-mix
//      keys, so rule 2 doesn't apply to it).
// Otherwise: fail loudly rather than silently guessing among several real candidates
// (e.g. cyrusneary's libero_object/libero_spatial/libero_goal/liber_o10).
static bool resolve_unnorm_dataset_key(const nlohmann::json& stats, std::string& key) {
    if (const char* env = std::getenv("VLA_OCTO_UNNORM_DATASET"); env && env[0] != '\0') {
        key = env;
        return true;
    }
    if (stats.is_object() && stats.size() == 1) {
        key = stats.begin().key();
        return true;
    }
    if (stats.is_object() && stats.contains("bridge_dataset")) {
        key = "bridge_dataset";
        return true;
    }
    std::fprintf(stderr,
                 "vla(octo): cannot auto-resolve unnorm dataset key (%zu candidate keys in "
                 "octo.dataset_statistics); set VLA_OCTO_UNNORM_DATASET or pass an explicit key "
                 "(e.g. --unnorm-dataset libero_object)\n",
                 stats.is_object() ? stats.size() : (size_t) 0);
    return false;
}

// Un-normalizes a [4,7] flattened action using octo.dataset_statistics's
// per-dataset action mean/std/mask: unnorm[d] = mask[d] ? norm[d]*std[d]+mean[d]
// : norm[d] (dims with mask=false, e.g. bridge_dataset's/libero_object's gripper dim 6,
// are left as-is -- confirmed against golden: final_action_unnormalized[...,6] ==
// sample_actions.final_action_normalized[...,6] exactly, while masked-in dims match
// a*std+mean to ~1e-5). dataset_key_in must match golden's metadata.unnormalization.dataset
// when comparing against a golden trace (all shipped bridge golden cases use
// "bridge_dataset"); pass "" to auto-resolve via resolve_unnorm_dataset_key (live/serving
// paths, where there's no golden to match against).
static bool unnormalize_action(gguf_reader& g, const std::string& dataset_key_in,
                               const std::vector<float>& normalized_28, std::vector<float>& unnorm_28) {
    const std::string stats_json = g.str("octo.dataset_statistics");
    if (stats_json.empty()) {
        std::fprintf(stderr, "vla(octo): missing octo.dataset_statistics\n");
        return false;
    }
    nlohmann::json j = nlohmann::json::parse(stats_json, nullptr, false);
    if (j.is_discarded()) {
        std::fprintf(stderr, "vla(octo): octo.dataset_statistics is not valid JSON\n");
        return false;
    }
    std::string dataset_key = dataset_key_in;
    if (dataset_key.empty() && !resolve_unnorm_dataset_key(j, dataset_key)) return false;
    if (!j.contains(dataset_key) || !j[dataset_key].contains("action")) {
        std::fprintf(stderr, "vla(octo): dataset_statistics missing %s.action\n", dataset_key.c_str());
        return false;
    }
    const auto& act = j[dataset_key]["action"];
    std::vector<float> mean = act.at("mean").get<std::vector<float>>();
    std::vector<float> stdv = act.at("std").get<std::vector<float>>();
    std::vector<bool> mask = act.at("mask").get<std::vector<bool>>();
    if (mean.size() != 7 || stdv.size() != 7 || mask.size() != 7 || normalized_28.size() != 28) {
        std::fprintf(stderr, "vla(octo): unexpected dataset_statistics/action shape\n");
        return false;
    }
    unnorm_28.resize(28);
    for (int t = 0; t < 4; ++t) {
        for (int d = 0; d < 7; ++d) {
            const float norm = normalized_28[(size_t) t * 7 + d];
            unnorm_28[(size_t) t * 7 + d] = mask[(size_t) d] ? (norm * stdv[(size_t) d] + mean[(size_t) d]) : norm;
        }
    }
    return true;
}

static bool build_transformer_mask(const NpyBool& task_valid,
                                   const NpyBool& primary_valid,
                                   const NpyBool& wrist_valid,
                                   const NpyBool& timestep_valid,
                                   int window_size,
                                   std::vector<uint8_t>& blocked,
                                   std::vector<float>& blocked_f32) {
    const int seq = octo_seq_len(window_size);
    constexpr int heads = 6;
    constexpr int step_tokens = kStepTokens;
    if (task_valid.shape != std::vector<int64_t>{1} ||
        primary_valid.shape != std::vector<int64_t>({1, window_size}) ||
        wrist_valid.shape != std::vector<int64_t>({1, window_size}) ||
        timestep_valid.shape != std::vector<int64_t>({1, window_size})) {
        std::fprintf(stderr, "vla(octo): unexpected input pad-mask shape\n");
        return false;
    }

    std::vector<OctoTokenMetadata> metadata;
    std::vector<uint8_t> key_valid;
    metadata.reserve(seq);
    key_valid.reserve(seq);
    for (int i = 0; i < 16; ++i) {
        metadata.push_back({OctoTokenKind::TASK, -1});
        key_valid.push_back(task_valid.data[0] != 0);
    }
    for (int t = 0; t < window_size; ++t) {
        const uint8_t timestep_ok = timestep_valid.data[(size_t) t] != 0;
        for (int i = 0; i < 256; ++i) {
            metadata.push_back({OctoTokenKind::OBS, t});
            key_valid.push_back(timestep_ok && primary_valid.data[(size_t) t]);
        }
        for (int i = 0; i < 64; ++i) {
            metadata.push_back({OctoTokenKind::OBS, t});
            key_valid.push_back(timestep_ok && wrist_valid.data[(size_t) t]);
        }
        for (int i = 0; i < 16; ++i) {
            metadata.push_back({OctoTokenKind::OBS, t});
            key_valid.push_back(task_valid.data[0] != 0);
        }
        metadata.push_back({OctoTokenKind::READOUT, t});
        key_valid.push_back(1);
    }
    if (metadata.size() != (size_t) seq || key_valid.size() != (size_t) seq ||
        16 + window_size * step_tokens != seq) return false;

    const size_t plane = (size_t) seq * seq;
    std::vector<uint8_t> blocked_one_head(plane, 0);
    blocked_f32.assign(plane, 0.0f);
    for (int q = 0; q < seq; ++q) {
        const OctoTokenMetadata qm = metadata[(size_t) q];
        for (int k = 0; k < seq; ++k) {
            const OctoTokenMetadata km = metadata[(size_t) k];
            bool allowed = false;
            if (qm.kind == OctoTokenKind::TASK) {
                allowed = km.kind == OctoTokenKind::TASK;
            } else if (qm.kind == OctoTokenKind::OBS) {
                allowed = km.kind == OctoTokenKind::TASK ||
                          (km.kind == OctoTokenKind::OBS && km.timestep <= qm.timestep);
            } else {
                allowed = km.kind == OctoTokenKind::TASK ||
                          (km.kind == OctoTokenKind::OBS && km.timestep <= qm.timestep) ||
                          (km.kind == OctoTokenKind::READOUT && km.timestep <= qm.timestep);
            }
            const size_t index = (size_t) q * seq + k;
            const bool is_blocked = !allowed || !key_valid[(size_t) k];
            blocked_one_head[index] = is_blocked ? 1 : 0;
            blocked_f32[index] = is_blocked ? 1.0f : 0.0f;
        }
    }
    blocked.resize((size_t) heads * plane);
    for (int h = 0; h < heads; ++h) {
        std::copy(blocked_one_head.begin(), blocked_one_head.end(), blocked.begin() + (size_t) h * plane);
    }
    return true;
}

static bool run_transformer_graph(OctoTransformerWeights& w,
                                  const std::vector<float>& input,
                                  const std::vector<float>& blocked_mask,
                                  int window_size,
                                  OctoTransformerResult& result) {
    constexpr int hidden = 384;
    constexpr int ffn = 1536;
    constexpr int heads = 6;
    constexpr int head_dim = 64;
    const int seq = octo_seq_len(window_size);
    constexpr float ln_eps = 1e-6f;
    constexpr float attn_scale = 0.125f;
    if (input.size() != (size_t) hidden * seq || blocked_mask.size() != (size_t) seq * seq) return false;

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        std::fprintf(stderr, "vla(octo): ggml_backend_cpu_init failed for transformer graph\n");
        return false;
    }
    ggml_backend_cpu_set_n_threads(backend, default_cpu_threads());
    ggml_init_params gp = {(size_t) 32 * 1024 * 1024, nullptr, true};
    ggml_context * ctx = ggml_init(gp);
    if (!ctx) {
        ggml_backend_free(backend);
        return false;
    }

    std::vector<ggml_tensor *> tensors;
    std::vector<std::vector<float>> payloads;
    auto add_payload = [&](ggml_tensor * t, std::vector<float> data) {
        tensors.push_back(t);
        payloads.push_back(std::move(data));
        return t;
    };
    auto make_1d_payload = [&](const char * name, int64_t ne0, std::vector<float>& data) {
        ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne0);
        ggml_set_name(t, name);
        return add_payload(t, std::move(data));
    };
    auto make_2d_payload = [&](const char * name, int64_t ne0, int64_t ne1, std::vector<float>& data) {
        ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
        ggml_set_name(t, name);
        return add_payload(t, std::move(data));
    };

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, seq);
    ggml_set_name(x, "octo.block_transformer.input");
    add_payload(x, input);
    ggml_tensor * mask_blocked = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, seq, seq);
    ggml_set_name(mask_blocked, "octo.block_transformer.blocked_mask");
    add_payload(mask_blocked, blocked_mask);
    ggml_tensor * mask = ggml_scale(ctx, ggml_repeat_4d(ctx, mask_blocked, seq, seq, heads, 1), -FLT_MAX);
    ggml_set_name(mask, "octo.block_transformer.additive_mask");

    std::array<ggml_tensor *, 12> block_out{};
    char name[160];
    for (int i = 0; i < 12; ++i) {
        OctoBlockWeights& b = w.blocks[(size_t) i];
        auto n1w = make_1d_payload((std::snprintf(name, sizeof(name), "octo.blk.%d.attn_norm.weight", i), name), hidden, b.attn_norm_w);
        auto n1b = make_1d_payload((std::snprintf(name, sizeof(name), "octo.blk.%d.attn_norm.bias", i), name), hidden, b.attn_norm_b);
        auto Wqkv = make_2d_payload((std::snprintf(name, sizeof(name), "octo.blk.%d.attn_qkv.weight", i), name), hidden, 3 * hidden, b.qkv_w);
        auto bqkv = make_1d_payload((std::snprintf(name, sizeof(name), "octo.blk.%d.attn_qkv.bias", i), name), 3 * hidden, b.qkv_b);
        auto Wo = make_2d_payload((std::snprintf(name, sizeof(name), "octo.blk.%d.attn_o.weight", i), name), hidden, hidden, b.attn_o_w);
        auto bo = make_1d_payload((std::snprintf(name, sizeof(name), "octo.blk.%d.attn_o.bias", i), name), hidden, b.attn_o_b);
        auto n2w = make_1d_payload((std::snprintf(name, sizeof(name), "octo.blk.%d.ffn_norm.weight", i), name), hidden, b.ffn_norm_w);
        auto n2b = make_1d_payload((std::snprintf(name, sizeof(name), "octo.blk.%d.ffn_norm.bias", i), name), hidden, b.ffn_norm_b);
        auto Wup = make_2d_payload((std::snprintf(name, sizeof(name), "octo.blk.%d.ffn_up.weight", i), name), hidden, ffn, b.ffn_up_w);
        auto bup = make_1d_payload((std::snprintf(name, sizeof(name), "octo.blk.%d.ffn_up.bias", i), name), ffn, b.ffn_up_b);
        auto Wdown = make_2d_payload((std::snprintf(name, sizeof(name), "octo.blk.%d.ffn_down.weight", i), name), ffn, hidden, b.ffn_down_w);
        auto bdown = make_1d_payload((std::snprintf(name, sizeof(name), "octo.blk.%d.ffn_down.bias", i), name), hidden, b.ffn_down_b);

        ggml_tensor * n1 = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, ln_eps), n1w), n1b);
        ggml_tensor * qkv = ggml_add(ctx, ggml_mul_mat(ctx, Wqkv, n1), bqkv);
        ggml_tensor * q = ggml_cont(ctx, ggml_view_2d(ctx, qkv, hidden, seq, qkv->nb[1], 0));
        ggml_tensor * k = ggml_cont(ctx, ggml_view_2d(ctx, qkv, hidden, seq, qkv->nb[1], (size_t) hidden * qkv->nb[0]));
        ggml_tensor * v = ggml_cont(ctx, ggml_view_2d(ctx, qkv, hidden, seq, qkv->nb[1], (size_t) 2 * hidden * qkv->nb[0]));
        ggml_tensor * Q = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, q, head_dim, heads, seq), 0, 2, 1, 3));
        ggml_tensor * K = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, k, head_dim, heads, seq), 0, 2, 1, 3));
        ggml_tensor * V = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, v, head_dim, heads, seq), 1, 2, 0, 3));
        ggml_tensor * scores = ggml_mul_mat(ctx, K, Q);
        ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
        ggml_tensor * probs = ggml_soft_max_ext(ctx, scores, mask, attn_scale, 0.0f);
        ggml_tensor * attended = ggml_mul_mat(ctx, V, probs);
        ggml_tensor * merged = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, attended, 0, 2, 1, 3)), hidden, seq);
        ggml_tensor * attn_out = ggml_add(ctx, ggml_mul_mat(ctx, Wo, merged), bo);
        ggml_tensor * residual = ggml_add(ctx, x, attn_out);
        ggml_tensor * n2 = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, residual, ln_eps), n2w), n2b);
        ggml_tensor * mlp = ggml_add(ctx, ggml_mul_mat(ctx, Wup, n2), bup);
        mlp = ggml_gelu_erf(ctx, mlp);
        mlp = ggml_add(ctx, ggml_mul_mat(ctx, Wdown, mlp), bdown);
        x = ggml_add(ctx, residual, mlp);
        std::snprintf(name, sizeof(name), "bt.blk%d.out", i);
        ggml_set_name(x, name);
        ggml_set_output(x);
        block_out[(size_t) i] = x;
    }

    ggml_tensor * out_w = make_1d_payload("octo.output_norm.weight", hidden, w.output_norm_w);
    ggml_tensor * out_b = make_1d_payload("octo.output_norm.bias", hidden, w.output_norm_b);
    ggml_tensor * output = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, ln_eps), out_w), out_b);
    ggml_set_name(output, "bt.output");
    ggml_set_output(output);
    constexpr int step_tokens = kStepTokens;
    ggml_tensor * split_task = ggml_cont(ctx, ggml_view_2d(ctx, output, hidden, 16, output->nb[1], 0));
    ggml_tensor * split_primary = ggml_cont(ctx, ggml_view_3d(ctx, output, hidden, 256, window_size, output->nb[1], (size_t) step_tokens * output->nb[1], (size_t) 16 * output->nb[1]));
    ggml_tensor * split_wrist = ggml_cont(ctx, ggml_view_3d(ctx, output, hidden, 64, window_size, output->nb[1], (size_t) step_tokens * output->nb[1], (size_t) 272 * output->nb[1]));
    ggml_tensor * split_language = ggml_cont(ctx, ggml_view_3d(ctx, output, hidden, 16, window_size, output->nb[1], (size_t) step_tokens * output->nb[1], (size_t) 336 * output->nb[1]));
    ggml_tensor * split_readout = ggml_cont(ctx, ggml_view_3d(ctx, output, hidden, 1, window_size, output->nb[1], (size_t) step_tokens * output->nb[1], (size_t) 352 * output->nb[1]));
    for (ggml_tensor * t : {split_task, split_primary, split_wrist, split_language, split_readout}) ggml_set_output(t);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 8192, false);
    ggml_build_forward_expand(graph, split_task);
    ggml_build_forward_expand(graph, split_primary);
    ggml_build_forward_expand(graph, split_wrist);
    ggml_build_forward_expand(graph, split_language);
    ggml_build_forward_expand(graph, split_readout);
    ggml_gallocr_t gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!gallocr || !ggml_gallocr_alloc_graph(gallocr, graph)) {
        std::fprintf(stderr, "vla(octo): transformer ggml_gallocr_alloc_graph failed\n");
        if (gallocr) ggml_gallocr_free(gallocr);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }
    for (size_t i = 0; i < tensors.size(); ++i) {
        ggml_backend_tensor_set(tensors[i], payloads[i].data(), 0, ggml_nbytes(tensors[i]));
    }
    const ggml_status st = ggml_backend_graph_compute(backend, graph);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "vla(octo): transformer ggml_backend_graph_compute failed (%d)\n", (int) st);
        ggml_gallocr_free(gallocr);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    for (int i = 0; i < 12; ++i) {
        result.block_outputs[(size_t) i].resize((size_t) hidden * seq);
        ggml_backend_tensor_get(block_out[(size_t) i], result.block_outputs[(size_t) i].data(), 0, ggml_nbytes(block_out[(size_t) i]));
    }
    auto get = [&](ggml_tensor * t, std::vector<float>& dst) {
        dst.resize(ggml_nelements(t));
        ggml_backend_tensor_get(t, dst.data(), 0, ggml_nbytes(t));
    };
    get(output, result.output);
    get(split_task, result.task_language);
    get(split_primary, result.obs_primary);
    get(split_wrist, result.obs_wrist);
    get(split_language, result.obs_task_language);
    get(split_readout, result.readout_action);

    ggml_gallocr_free(gallocr);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return true;
}

// TIP-BUILD-OCTO-GPU-A: predict()-only variant of run_transformer_graph. All 12 blocks'
// attn/ffn weights + the final output_norm are referenced directly from the resident context
// (no copy, no per-call upload); `input`/`blocked_mask` (this call's assembled sequence +
// mask) are genuinely new per-call data and still get uploaded exactly as before. Graph
// topology and op order are unchanged.
static bool run_transformer_graph_resident(ggml_context * ctx_w, ggml_backend_t backend,
                                           const std::vector<float>& input,
                                           const std::vector<float>& blocked_mask,
                                           int window_size,
                                           OctoTransformerResult& result) {
    constexpr int hidden = 384;
    constexpr int heads = 6;
    constexpr int head_dim = 64;
    const int seq = octo_seq_len(window_size);
    constexpr float ln_eps = 1e-6f;
    constexpr float attn_scale = 0.125f;
    if (input.size() != (size_t) hidden * seq || blocked_mask.size() != (size_t) seq * seq) return false;

    char rname[160];
    ggml_tensor * blk_w[12][10];  // attn_norm.{w,b}, attn_qkv.{w,b}, attn_o.{w,b}, ffn_norm.{w,b}, ffn_up.{w,b}, ffn_down.{w,b}
    const char * leaves[10] = {"attn_norm.weight", "attn_norm.bias", "attn_qkv.weight", "attn_qkv.bias",
                               "attn_o.weight", "attn_o.bias", "ffn_norm.weight", "ffn_norm.bias",
                               "ffn_up.weight", "ffn_up.bias"};
    for (int i = 0; i < 12; ++i) {
        for (int j = 0; j < 10; ++j) {
            std::snprintf(rname, sizeof(rname), "octo.blk.%d.%s", i, leaves[j]);
            blk_w[i][j] = wt(ctx_w, rname);
            if (!blk_w[i][j]) return false;
        }
    }
    // ffn_down.{weight,bias} didn't fit the 10-wide table above cleanly (name pattern differs
    // only in the leaf, kept separate to avoid a confusing 12-wide array); fetched per-block below.
    ggml_tensor * ffn_down_w[12];
    ggml_tensor * ffn_down_b[12];
    for (int i = 0; i < 12; ++i) {
        std::snprintf(rname, sizeof(rname), "octo.blk.%d.ffn_down.weight", i);
        ffn_down_w[i] = wt(ctx_w, rname);
        std::snprintf(rname, sizeof(rname), "octo.blk.%d.ffn_down.bias", i);
        ffn_down_b[i] = wt(ctx_w, rname);
        if (!ffn_down_w[i] || !ffn_down_b[i]) return false;
    }
    ggml_tensor * out_w_r = wt(ctx_w, "octo.output_norm.weight");
    ggml_tensor * out_b_r = wt(ctx_w, "octo.output_norm.bias");
    if (!out_w_r || !out_b_r) return false;

    ggml_init_params gp = {(size_t) 32 * 1024 * 1024, nullptr, true};
    ggml_context * ctx = ggml_init(gp);
    if (!ctx) return false;

    std::vector<ggml_tensor *> tensors;
    std::vector<std::vector<float>> payloads;
    auto add_payload = [&](ggml_tensor * t, std::vector<float> data) {
        tensors.push_back(t);
        payloads.push_back(std::move(data));
        return t;
    };

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, seq);
    ggml_set_name(x, "octo.block_transformer.input");
    add_payload(x, input);
    ggml_tensor * mask_blocked = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, seq, seq);
    ggml_set_name(mask_blocked, "octo.block_transformer.blocked_mask");
    add_payload(mask_blocked, blocked_mask);
    ggml_tensor * mask = ggml_scale(ctx, ggml_repeat_4d(ctx, mask_blocked, seq, seq, heads, 1), -FLT_MAX);
    ggml_set_name(mask, "octo.block_transformer.additive_mask");

    std::array<ggml_tensor *, 12> block_out{};
    for (int i = 0; i < 12; ++i) {
        ggml_tensor * n1w = blk_w[i][0];
        ggml_tensor * n1b = blk_w[i][1];
        ggml_tensor * Wqkv = blk_w[i][2];
        ggml_tensor * bqkv = blk_w[i][3];
        ggml_tensor * Wo = blk_w[i][4];
        ggml_tensor * bo = blk_w[i][5];
        ggml_tensor * n2w = blk_w[i][6];
        ggml_tensor * n2b = blk_w[i][7];
        ggml_tensor * Wup = blk_w[i][8];
        ggml_tensor * bup = blk_w[i][9];
        ggml_tensor * Wdown = ffn_down_w[i];
        ggml_tensor * bdown = ffn_down_b[i];

        ggml_tensor * n1 = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, ln_eps), n1w), n1b);
        ggml_tensor * qkv = ggml_add(ctx, ggml_mul_mat(ctx, Wqkv, n1), bqkv);
        ggml_tensor * q = ggml_cont(ctx, ggml_view_2d(ctx, qkv, hidden, seq, qkv->nb[1], 0));
        ggml_tensor * k = ggml_cont(ctx, ggml_view_2d(ctx, qkv, hidden, seq, qkv->nb[1], (size_t) hidden * qkv->nb[0]));
        ggml_tensor * v = ggml_cont(ctx, ggml_view_2d(ctx, qkv, hidden, seq, qkv->nb[1], (size_t) 2 * hidden * qkv->nb[0]));
        ggml_tensor * Q = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, q, head_dim, heads, seq), 0, 2, 1, 3));
        ggml_tensor * K = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, k, head_dim, heads, seq), 0, 2, 1, 3));
        ggml_tensor * V = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, v, head_dim, heads, seq), 1, 2, 0, 3));
        ggml_tensor * scores = ggml_mul_mat(ctx, K, Q);
        ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
        ggml_tensor * probs = ggml_soft_max_ext(ctx, scores, mask, attn_scale, 0.0f);
        ggml_tensor * attended = ggml_mul_mat(ctx, V, probs);
        ggml_tensor * merged = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, attended, 0, 2, 1, 3)), hidden, seq);
        ggml_tensor * attn_out = ggml_add(ctx, ggml_mul_mat(ctx, Wo, merged), bo);
        ggml_tensor * residual = ggml_add(ctx, x, attn_out);
        ggml_tensor * n2 = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, residual, ln_eps), n2w), n2b);
        ggml_tensor * mlp = ggml_add(ctx, ggml_mul_mat(ctx, Wup, n2), bup);
        mlp = ggml_gelu_erf(ctx, mlp);
        mlp = ggml_add(ctx, ggml_mul_mat(ctx, Wdown, mlp), bdown);
        x = ggml_add(ctx, residual, mlp);
        char name[64];
        std::snprintf(name, sizeof(name), "bt.blk%d.out", i);
        ggml_set_name(x, name);
        ggml_set_output(x);
        block_out[(size_t) i] = x;
    }

    ggml_tensor * output = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, ln_eps), out_w_r), out_b_r);
    ggml_set_name(output, "bt.output");
    ggml_set_output(output);
    constexpr int step_tokens = kStepTokens;
    ggml_tensor * split_task = ggml_cont(ctx, ggml_view_2d(ctx, output, hidden, 16, output->nb[1], 0));
    ggml_tensor * split_primary = ggml_cont(ctx, ggml_view_3d(ctx, output, hidden, 256, window_size, output->nb[1], (size_t) step_tokens * output->nb[1], (size_t) 16 * output->nb[1]));
    ggml_tensor * split_wrist = ggml_cont(ctx, ggml_view_3d(ctx, output, hidden, 64, window_size, output->nb[1], (size_t) step_tokens * output->nb[1], (size_t) 272 * output->nb[1]));
    ggml_tensor * split_language = ggml_cont(ctx, ggml_view_3d(ctx, output, hidden, 16, window_size, output->nb[1], (size_t) step_tokens * output->nb[1], (size_t) 336 * output->nb[1]));
    ggml_tensor * split_readout = ggml_cont(ctx, ggml_view_3d(ctx, output, hidden, 1, window_size, output->nb[1], (size_t) step_tokens * output->nb[1], (size_t) 352 * output->nb[1]));
    for (ggml_tensor * t : {split_task, split_primary, split_wrist, split_language, split_readout}) ggml_set_output(t);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 8192, false);
    ggml_build_forward_expand(graph, split_task);
    ggml_build_forward_expand(graph, split_primary);
    ggml_build_forward_expand(graph, split_wrist);
    ggml_build_forward_expand(graph, split_language);
    ggml_build_forward_expand(graph, split_readout);
    ggml_gallocr_t gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!gallocr || !ggml_gallocr_alloc_graph(gallocr, graph)) {
        std::fprintf(stderr, "vla(octo): transformer ggml_gallocr_alloc_graph failed\n");
        if (gallocr) ggml_gallocr_free(gallocr);
        ggml_free(ctx);
        return false;
    }
    for (size_t i = 0; i < tensors.size(); ++i) {
        ggml_backend_tensor_set(tensors[i], payloads[i].data(), 0, ggml_nbytes(tensors[i]));
    }
    const ggml_status st = ggml_backend_graph_compute(backend, graph);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "vla(octo): transformer ggml_backend_graph_compute failed (%d)\n", (int) st);
        ggml_gallocr_free(gallocr);
        ggml_free(ctx);
        return false;
    }

    for (int i = 0; i < 12; ++i) {
        result.block_outputs[(size_t) i].resize((size_t) hidden * seq);
        ggml_backend_tensor_get(block_out[(size_t) i], result.block_outputs[(size_t) i].data(), 0, ggml_nbytes(block_out[(size_t) i]));
    }
    auto get = [&](ggml_tensor * t, std::vector<float>& dst) {
        dst.resize(ggml_nelements(t));
        ggml_backend_tensor_get(t, dst.data(), 0, ggml_nbytes(t));
    };
    get(output, result.output);
    get(split_task, result.task_language);
    get(split_primary, result.obs_primary);
    get(split_wrist, result.obs_wrist);
    get(split_language, result.obs_task_language);
    get(split_readout, result.readout_action);

    ggml_gallocr_free(gallocr);
    ggml_free(ctx);
    return true;
}

}  // namespace

std::unique_ptr<ModelArchBase> octo_create(const std::string& mmproj_path,
                                           const std::string& ckpt_path,
                                           const std::string&) {
    if (!mmproj_path.empty()) {
        std::printf("vla(octo): note - mmproj '%s' is ignored (Octo M0 uses one GGUF)\n", mmproj_path.c_str());
    }

    auto m = std::make_unique<OctoModelArch>();
    m->gguf_path = ckpt_path;
    m->matmul_type = GGML_TYPE_F32;

    if (!m->io.open(ckpt_path)) return nullptr;
    if (!m->io.has("octo.architecture")) {
        std::fprintf(stderr, "vla(octo): %s is not an Octo GGUF\n", ckpt_path.c_str());
        return nullptr;
    }
    if (!load_config(m->io, *m)) return nullptr;

#ifdef GGML_USE_CUDA
    m->backend = ggml_backend_cuda_init(0);
    if (m->backend) std::printf("vla(octo): backend = CUDA (device 0)\n");
    else            std::fprintf(stderr, "vla(octo): ggml_backend_cuda_init failed; falling back to CPU\n");
#elif defined(GGML_USE_METAL)
    m->backend = ggml_backend_metal_init();
    if (m->backend) std::printf("vla(octo): backend = Metal\n");
    else            std::fprintf(stderr, "vla(octo): ggml_backend_metal_init failed; falling back to CPU\n");
#endif
    if (!m->backend) {
        m->backend = ggml_backend_cpu_init();
        if (!m->backend) {
            std::fprintf(stderr, "vla(octo): ggml_backend_cpu_init failed\n");
            return nullptr;
        }
        ggml_backend_cpu_set_n_threads(m->backend, m->n_threads);
        std::printf("vla(octo): backend = CPU (%d threads)\n", m->n_threads);
    }

    // TIP-BUILD-OCTO-GPU-B: predict()'s 5 stages now consume these same weights (resident on
    // `m->backend` -- CUDA when available, CPU otherwise) directly; no separate CPU-only copy.
    if (!load_all_tensors(*m, m->io)) return nullptr;
    std::printf("vla(octo): loaded %lld F32 tensors, hidden=%lld blocks=%lld heads=%lld horizon=%lld action_dim=%lld\n",
                (long long) m->tensors.size(), (long long) m->hidden, (long long) m->blocks,
                (long long) m->heads, (long long) m->action_horizon, (long long) m->action_dim);
    return m;
}

bool octo_dump_gguf_inventory(const std::string& ckpt_path) {
    gguf_init_params p{};
    p.no_alloc = true;
    ggml_context * meta = nullptr;
    p.ctx = &meta;
    gguf_context * gctx = gguf_init_from_file(ckpt_path.c_str(), p);
    if (!gctx) {
        std::fprintf(stderr, "vla(octo): gguf_init_from_file failed for %s\n", ckpt_path.c_str());
        return false;
    }

    const int64_t n_kv = gguf_get_n_kv(gctx);
    const int64_t n_tensors = gguf_get_n_tensors(gctx);
    std::printf("gguf_kv=%lld\n", (long long) n_kv);
    std::printf("gguf_tensors=%lld\n", (long long) n_tensors);
    for (int64_t i = 0; i < n_kv; ++i) {
        const char * key = gguf_get_key(gctx, i);
        std::printf("kv\t%s\ttype=%s\n", key, gguf_type_name(gguf_get_kv_type(gctx, i)));
    }
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(gctx, i);
        const ggml_tensor * t = ggml_get_tensor(meta, name);
        std::printf("tensor\t%s\ttype=%s\tshape=[", name, ggml_type_name(gguf_get_tensor_type(gctx, i)));
        const int nd = t ? ggml_n_dims(t) : 0;
        for (int d = nd - 1; d >= 0; --d) {
            std::printf("%lld%s", (long long) t->ne[d], d == 0 ? "" : ",");
        }
        std::printf("]\tbytes=%zu\n", gguf_get_tensor_size(gctx, i));
    }
    gguf_free(gctx);
    if (meta) ggml_free(meta);
    return true;
}

bool octo_dump_tokenizer_case(const std::string& ckpt_path,
                              const std::string& case_dir,
                              const std::string& dump_dir,
                              const std::string& t5_inject_path,
                              const std::string& unnorm_dataset) {
    gguf_reader g{"octo"};
    if (!g.open(ckpt_path)) return false;
    const int window_size = (int) g.u32("octo.window_size");
    if (window_size < 1 || window_size > kMaxHorizon) {
        std::fprintf(stderr, "vla(octo): octo.window_size=%d out of supported range [1, %d]\n",
                     window_size, kMaxHorizon);
        return false;
    }
    const int seq = octo_seq_len(window_size);

    NpyU8 primary_obs, wrist_obs, primary_task, wrist_task;
    bool wrist_obs_real = true;
    if (!parse_npy_u8(case_dir + "/tensors/input.observation.image_primary.npy", primary_obs)) return false;
    if (!parse_npy_u8_or_zero_obs(case_dir + "/tensors/input.observation.image_wrist.npy",
                                  window_size, 128, wrist_obs, wrist_obs_real)) return false;
    if (!parse_npy_u8_or_zero_task(case_dir + "/tensors/input.task.image_primary.npy", primary_obs, primary_task)) return false;
    if (!parse_npy_u8_or_zero_task(case_dir + "/tensors/input.task.image_wrist.npy", wrist_obs, wrist_task)) return false;

    OctoObsWeights primary_w, wrist_w;
    if (!read_obs_weights(g, "primary", primary_w)) return false;
    if (!read_obs_weights(g, "wrist", wrist_w)) return false;

    std::ofstream mf(dump_dir + "/manifest.txt");
    if (!mf) {
        std::error_code ec;
        std::filesystem::create_directories(dump_dir, ec);
        if (ec) {
            std::fprintf(stderr, "vla(octo): cannot create %s: %s\n", dump_dir.c_str(), ec.message().c_str());
            return false;
        }
        mf.open(dump_dir + "/manifest.txt");
    }
    if (!mf) {
        std::fprintf(stderr, "vla(octo): cannot write %s/manifest.txt\n", dump_dir.c_str());
        return false;
    }

    std::vector<float> tok, proj, pos, primary_pos, wrist_pos;
    if (!run_one_obs_tokenizer_graph(primary_w, primary_obs, primary_task, 256, 256, window_size, tok, proj, pos)) return false;
    if (!write_f32_dump(dump_dir, "obs.primary.tok",  tok,  {1, window_size, 256, 512}, mf)) return false;
    if (!write_f32_dump(dump_dir, "obs.primary.proj", proj, {1, window_size, 256, 384}, mf)) return false;
    if (!write_f32_dump(dump_dir, "obs.primary.pos",  pos,  {1, window_size, 256, 384}, mf)) return false;
    primary_pos = pos;
    if (!run_one_obs_tokenizer_graph(wrist_w, wrist_obs, wrist_task, 128, 64, window_size, tok, proj, pos)) return false;
    if (!write_f32_dump(dump_dir, "obs.wrist.tok",  tok,  {1, window_size, 64, 512}, mf)) return false;
    if (!write_f32_dump(dump_dir, "obs.wrist.proj", proj, {1, window_size, 64, 384}, mf)) return false;
    if (!write_f32_dump(dump_dir, "obs.wrist.pos",  pos,  {1, window_size, 64, 384}, mf)) return false;
    wrist_pos = pos;
    // T5-base encoder: always computed natively (M5). --t5-inject, when given, overrides the
    // tokens fed downstream (oracle/fallback path); the native "t5.out" boundary is always
    // dumped so it can be checked against golden regardless.
    OctoT5Weights t5_w;
    if (!read_t5_weights(g, t5_w)) return false;
    NpyI32 input_ids, attn_mask;
    if (!parse_npy_i32(case_dir + "/tensors/input.task.language_instruction.input_ids.npy", input_ids)) return false;
    if (!parse_npy_i32(case_dir + "/tensors/input.task.language_instruction.attention_mask.npy", attn_mask)) return false;
    if (input_ids.shape != std::vector<int64_t>({1, 16}) || attn_mask.shape != std::vector<int64_t>({1, 16})) {
        std::fprintf(stderr, "vla(octo): unexpected input_ids/attention_mask shape\n");
        return false;
    }
    std::vector<float> t5_native_out;
    if (!run_t5_encoder_graph(g, t5_w, input_ids.data, attn_mask.data, t5_native_out)) return false;
    if (!write_f32_dump(dump_dir, "t5.out", t5_native_out, {1, 16, 768}, mf)) return false;

    NpyF32 t5;
    if (!t5_inject_path.empty()) {
        if (!parse_npy_f32(t5_inject_path, t5)) return false;
    } else {
        t5.shape = {1, 16, 768};
        t5.data = t5_native_out;
    }
    OctoLanguageWeights lang_w;
    if (!read_language_weights(g, lang_w)) return false;
    std::vector<float> lang_proj, lang_pos, repeated;
    if (!run_language_graph(lang_w, t5, window_size, lang_proj, lang_pos, repeated)) return false;
    if (!write_f32_dump(dump_dir, "lang.proj",         lang_proj, {1, 16, 384}, mf)) return false;
    if (!write_f32_dump(dump_dir, "lang.pos",          lang_pos,  {1, 16, 384}, mf)) return false;
    if (!write_f32_dump(dump_dir, "repeated_language", repeated,  {1, window_size, 16, 384}, mf)) return false;

    OctoTransformerWeights transformer_w;
    if (!read_transformer_weights(g, transformer_w)) return false;
    NpyBool task_valid, primary_valid, wrist_valid, timestep_valid;
    if (!parse_npy_bool(case_dir + "/tensors/input.task.pad_mask_dict.language_instruction.npy", task_valid) ||
        !parse_npy_bool(case_dir + "/tensors/input.observation.pad_mask_dict.image_primary.npy", primary_valid) ||
        !parse_npy_bool_or_zero(case_dir + "/tensors/input.observation.pad_mask_dict.image_wrist.npy", window_size, wrist_valid) ||
        !parse_npy_bool(case_dir + "/tensors/input.observation.timestep_pad_mask.npy", timestep_valid)) return false;
    if (!wrist_obs_real) {
        // Single-camera checkpoint: the wrist image is a zero-fill placeholder, not a
        // real observation, regardless of what (if anything) the golden's own wrist
        // pad-mask says -- never let the causal mask treat it as valid.
        std::fill(wrist_valid.data.begin(), wrist_valid.data.end(), (uint8_t) 0);
    }

    OctoTransformerResult bt;
    if (!assemble_transformer_input(lang_pos, primary_pos, wrist_pos, repeated, transformer_w.readout_pos, window_size, bt.input)) return false;
    std::vector<float> blocked_mask;
    if (!build_transformer_mask(task_valid, primary_valid, wrist_valid, timestep_valid, window_size, bt.blocked_mask, blocked_mask)) return false;
    if (!write_f32_dump(dump_dir, "bt.input", bt.input, {1, seq, 384}, mf)) return false;
    if (!write_bool_dump(dump_dir, "bt.mask", bt.blocked_mask, {6, seq, seq}, mf)) return false;
    if (!run_transformer_graph(transformer_w, bt.input, blocked_mask, window_size, bt)) return false;
    for (int i = 0; i < 12; ++i) {
        char boundary[32];
        std::snprintf(boundary, sizeof(boundary), "bt.blk%d.out", i);
        if (!write_f32_dump(dump_dir, boundary, bt.block_outputs[(size_t) i], {1, seq, 384}, mf)) return false;
    }
    if (!write_f32_dump(dump_dir, "bt.output", bt.output, {1, seq, 384}, mf)) return false;
    if (!write_f32_dump(dump_dir, "bt.task_language", bt.task_language, {1, 16, 384}, mf)) return false;
    if (!write_f32_dump(dump_dir, "bt.obs_primary", bt.obs_primary, {1, window_size, 256, 384}, mf)) return false;
    if (!write_f32_dump(dump_dir, "bt.obs_wrist", bt.obs_wrist, {1, window_size, 64, 384}, mf)) return false;
    if (!write_f32_dump(dump_dir, "bt.obs_task_language", bt.obs_task_language, {1, window_size, 16, 384}, mf)) return false;
    if (!write_f32_dump(dump_dir, "bt.readout_action", bt.readout_action, {1, window_size, 1, 384}, mf)) return false;

    OctoDiffusionWeights diffusion_w;
    if (!read_diffusion_weights(g, diffusion_w)) return false;
    OctoDiffusionResult diff;
    if (!run_diffusion_replay(diffusion_w, case_dir, bt.readout_action, window_size, diff)) return false;
    if (!write_f32_dump(dump_dir, "diff.initial_noise", diff.initial_noise, {1, window_size, 28}, mf)) return false;
    if (!write_bool_dump(dump_dir, "diff.action_mask", diff.action_mask, {1, window_size, 4, 7}, mf)) return false;
    if (!write_bool_dump(dump_dir, "diff.flat_action_mask", diff.flat_action_mask, {1, window_size, 28}, mf)) return false;
    for (int step = 0; step < 20; ++step) {
        char boundary[64];
        std::snprintf(boundary, sizeof(boundary), "diff.step%02d.current_x_before", step);
        if (!write_f32_dump(dump_dir, boundary, diff.current_x_before[(size_t) step], {1, window_size, 28}, mf)) return false;
        std::snprintf(boundary, sizeof(boundary), "diff.step%02d.pred_eps", step);
        if (!write_f32_dump(dump_dir, boundary, diff.pred_eps[(size_t) step], {1, window_size, 28}, mf)) return false;
        std::snprintf(boundary, sizeof(boundary), "diff.step%02d.z", step);
        if (!write_f32_dump(dump_dir, boundary, diff.z[(size_t) step], {1, window_size, 28}, mf)) return false;
        std::snprintf(boundary, sizeof(boundary), "diff.step%02d.x_after_denoise", step);
        if (!write_f32_dump(dump_dir, boundary, diff.after_denoise[(size_t) step], {1, window_size, 28}, mf)) return false;
        std::snprintf(boundary, sizeof(boundary), "diff.step%02d.x_after_noise_add", step);
        if (!write_f32_dump(dump_dir, boundary, diff.after_noise_add[(size_t) step], {1, window_size, 28}, mf)) return false;
        std::snprintf(boundary, sizeof(boundary), "diff.step%02d.x_after_clip", step);
        if (!write_f32_dump(dump_dir, boundary, diff.after_clip[(size_t) step], {1, window_size, 28}, mf)) return false;
        std::snprintf(boundary, sizeof(boundary), "diff.step%02d.x_after_mask", step);
        if (!write_f32_dump(dump_dir, boundary, diff.after_mask[(size_t) step], {1, window_size, 28}, mf)) return false;
    }
    if (!write_f32_dump(dump_dir, "diff.actions_all_timesteps", diff.actions_all_timesteps, {1, window_size, 4, 7}, mf)) return false;
    if (!write_f32_dump(dump_dir, "sample_actions.final_action_normalized", diff.final_actions, {1, 4, 7}, mf)) return false;
    if (!write_f32_dump(dump_dir, "action_final", diff.final_actions, {1, 4, 7}, mf)) return false;

    std::vector<float> unnorm_actions;
    if (!unnormalize_action(g, unnorm_dataset, diff.final_actions, unnorm_actions)) return false;
    if (!write_f32_dump(dump_dir, "action_final_unnormalized", unnorm_actions, {1, 4, 7}, mf)) return false;
    return true;
}

// TIP-BUILD-OCTO-GPU-B: --resident counterpart of octo_dump_tokenizer_case. Dumps the exact
// same boundary set, in the same layout, so it's a drop-in replacement for
// verify_octo_parity.py -- but computes every stage via the *_resident functions (on `backend`,
// CUDA when available) instead of the host-vector/CPU-only path above, using the SAME weight
// residency setup octo_create() builds for the live server (load_all_tensors onto a backend
// selected the identical way). This is what makes the drift measurement real: it exercises the
// literal code OctoModelArch::predict() runs, not a separate parallel implementation. The
// original octo_dump_tokenizer_case above is untouched and remains the default (CPU) dump path.
bool octo_dump_tokenizer_case_resident(const std::string& ckpt_path,
                                       const std::string& case_dir,
                                       const std::string& dump_dir,
                                       const std::string& t5_inject_path,
                                       const std::string& unnorm_dataset) {
    gguf_reader g{"octo"};
    if (!g.open(ckpt_path)) return false;
    const int window_size = (int) g.u32("octo.window_size");
    if (window_size < 1 || window_size > kMaxHorizon) {
        std::fprintf(stderr, "vla(octo): octo.window_size=%d out of supported range [1, %d]\n",
                     window_size, kMaxHorizon);
        return false;
    }
    const int seq = octo_seq_len(window_size);

    // Backend selection + weight residency: byte-for-byte the same sequence octo_create()
    // uses for the live server, so this dump path is built exactly like a real model load.
    OctoModelArch m;
    m.matmul_type = GGML_TYPE_F32;
#ifdef GGML_USE_CUDA
    m.backend = ggml_backend_cuda_init(0);
    if (m.backend) std::printf("vla(octo): [--resident dump] backend = CUDA (device 0)\n");
    else           std::fprintf(stderr, "vla(octo): [--resident dump] ggml_backend_cuda_init failed; falling back to CPU\n");
#elif defined(GGML_USE_METAL)
    m.backend = ggml_backend_metal_init();
    if (m.backend) std::printf("vla(octo): [--resident dump] backend = Metal\n");
    else           std::fprintf(stderr, "vla(octo): [--resident dump] ggml_backend_metal_init failed; falling back to CPU\n");
#endif
    if (!m.backend) {
        m.backend = ggml_backend_cpu_init();
        if (!m.backend) {
            std::fprintf(stderr, "vla(octo): [--resident dump] ggml_backend_cpu_init failed\n");
            return false;
        }
        ggml_backend_cpu_set_n_threads(m.backend, default_cpu_threads());
        std::printf("vla(octo): [--resident dump] backend = CPU (%d threads)\n", default_cpu_threads());
    }
    if (!load_all_tensors(m, g)) return false;
    ggml_context * ctx_w = m.ctx_weights;
    ggml_backend_t backend = m.backend;

    NpyU8 primary_obs, wrist_obs, primary_task, wrist_task;
    bool wrist_obs_real = true;
    if (!parse_npy_u8(case_dir + "/tensors/input.observation.image_primary.npy", primary_obs)) return false;
    if (!parse_npy_u8_or_zero_obs(case_dir + "/tensors/input.observation.image_wrist.npy",
                                  window_size, 128, wrist_obs, wrist_obs_real)) return false;
    if (!parse_npy_u8_or_zero_task(case_dir + "/tensors/input.task.image_primary.npy", primary_obs, primary_task)) return false;
    if (!parse_npy_u8_or_zero_task(case_dir + "/tensors/input.task.image_wrist.npy", wrist_obs, wrist_task)) return false;

    std::ofstream mf(dump_dir + "/manifest.txt");
    if (!mf) {
        std::error_code ec;
        std::filesystem::create_directories(dump_dir, ec);
        if (ec) {
            std::fprintf(stderr, "vla(octo): cannot create %s: %s\n", dump_dir.c_str(), ec.message().c_str());
            return false;
        }
        mf.open(dump_dir + "/manifest.txt");
    }
    if (!mf) {
        std::fprintf(stderr, "vla(octo): cannot write %s/manifest.txt\n", dump_dir.c_str());
        return false;
    }

    std::vector<float> tok, proj, pos, primary_pos, wrist_pos;
    if (!run_one_obs_tokenizer_graph_resident(ctx_w, backend, "primary", primary_obs, primary_task, 256, 256, window_size, tok, proj, pos)) return false;
    if (!write_f32_dump(dump_dir, "obs.primary.tok",  tok,  {1, window_size, 256, 512}, mf)) return false;
    if (!write_f32_dump(dump_dir, "obs.primary.proj", proj, {1, window_size, 256, 384}, mf)) return false;
    if (!write_f32_dump(dump_dir, "obs.primary.pos",  pos,  {1, window_size, 256, 384}, mf)) return false;
    primary_pos = pos;
    if (!run_one_obs_tokenizer_graph_resident(ctx_w, backend, "wrist", wrist_obs, wrist_task, 128, 64, window_size, tok, proj, pos)) return false;
    if (!write_f32_dump(dump_dir, "obs.wrist.tok",  tok,  {1, window_size, 64, 512}, mf)) return false;
    if (!write_f32_dump(dump_dir, "obs.wrist.proj", proj, {1, window_size, 64, 384}, mf)) return false;
    if (!write_f32_dump(dump_dir, "obs.wrist.pos",  pos,  {1, window_size, 64, 384}, mf)) return false;
    wrist_pos = pos;

    NpyI32 input_ids, attn_mask;
    if (!parse_npy_i32(case_dir + "/tensors/input.task.language_instruction.input_ids.npy", input_ids)) return false;
    if (!parse_npy_i32(case_dir + "/tensors/input.task.language_instruction.attention_mask.npy", attn_mask)) return false;
    if (input_ids.shape != std::vector<int64_t>({1, 16}) || attn_mask.shape != std::vector<int64_t>({1, 16})) {
        std::fprintf(stderr, "vla(octo): unexpected input_ids/attention_mask shape\n");
        return false;
    }
    std::vector<float> t5_native_out;
    if (!run_t5_encoder_graph_resident(ctx_w, backend, input_ids.data, attn_mask.data, t5_native_out)) return false;
    if (!write_f32_dump(dump_dir, "t5.out", t5_native_out, {1, 16, 768}, mf)) return false;

    NpyF32 t5;
    if (!t5_inject_path.empty()) {
        if (!parse_npy_f32(t5_inject_path, t5)) return false;
    } else {
        t5.shape = {1, 16, 768};
        t5.data = t5_native_out;
    }
    std::vector<float> lang_proj, lang_pos, repeated;
    if (!run_language_graph_resident(ctx_w, backend, t5, window_size, lang_proj, lang_pos, repeated)) return false;
    if (!write_f32_dump(dump_dir, "lang.proj",         lang_proj, {1, 16, 384}, mf)) return false;
    if (!write_f32_dump(dump_dir, "lang.pos",          lang_pos,  {1, 16, 384}, mf)) return false;
    if (!write_f32_dump(dump_dir, "repeated_language", repeated,  {1, window_size, 16, 384}, mf)) return false;

    ggml_tensor * readout_pos_r = wt(ctx_w, "octo.readout.action.pos_embd");
    if (!readout_pos_r) return false;
    const std::vector<float> readout_pos = tensor_to_vec(readout_pos_r);

    NpyBool task_valid, primary_valid, wrist_valid, timestep_valid;
    if (!parse_npy_bool(case_dir + "/tensors/input.task.pad_mask_dict.language_instruction.npy", task_valid) ||
        !parse_npy_bool(case_dir + "/tensors/input.observation.pad_mask_dict.image_primary.npy", primary_valid) ||
        !parse_npy_bool_or_zero(case_dir + "/tensors/input.observation.pad_mask_dict.image_wrist.npy", window_size, wrist_valid) ||
        !parse_npy_bool(case_dir + "/tensors/input.observation.timestep_pad_mask.npy", timestep_valid)) return false;
    if (!wrist_obs_real) {
        std::fill(wrist_valid.data.begin(), wrist_valid.data.end(), (uint8_t) 0);
    }

    OctoTransformerResult bt;
    if (!assemble_transformer_input(lang_pos, primary_pos, wrist_pos, repeated, readout_pos, window_size, bt.input)) return false;
    std::vector<float> blocked_mask;
    if (!build_transformer_mask(task_valid, primary_valid, wrist_valid, timestep_valid, window_size, bt.blocked_mask, blocked_mask)) return false;
    if (!write_f32_dump(dump_dir, "bt.input", bt.input, {1, seq, 384}, mf)) return false;
    if (!write_bool_dump(dump_dir, "bt.mask", bt.blocked_mask, {6, seq, seq}, mf)) return false;
    if (!run_transformer_graph_resident(ctx_w, backend, bt.input, blocked_mask, window_size, bt)) return false;
    for (int i = 0; i < 12; ++i) {
        char boundary[32];
        std::snprintf(boundary, sizeof(boundary), "bt.blk%d.out", i);
        if (!write_f32_dump(dump_dir, boundary, bt.block_outputs[(size_t) i], {1, seq, 384}, mf)) return false;
    }
    if (!write_f32_dump(dump_dir, "bt.output", bt.output, {1, seq, 384}, mf)) return false;
    if (!write_f32_dump(dump_dir, "bt.task_language", bt.task_language, {1, 16, 384}, mf)) return false;
    if (!write_f32_dump(dump_dir, "bt.obs_primary", bt.obs_primary, {1, window_size, 256, 384}, mf)) return false;
    if (!write_f32_dump(dump_dir, "bt.obs_wrist", bt.obs_wrist, {1, window_size, 64, 384}, mf)) return false;
    if (!write_f32_dump(dump_dir, "bt.obs_task_language", bt.obs_task_language, {1, window_size, 16, 384}, mf)) return false;
    if (!write_f32_dump(dump_dir, "bt.readout_action", bt.readout_action, {1, window_size, 1, 384}, mf)) return false;

    OctoDiffusionResult diff;
    if (!run_diffusion_replay_resident(ctx_w, backend, case_dir, bt.readout_action, window_size, diff)) return false;
    if (!write_f32_dump(dump_dir, "diff.initial_noise", diff.initial_noise, {1, window_size, 28}, mf)) return false;
    if (!write_bool_dump(dump_dir, "diff.action_mask", diff.action_mask, {1, window_size, 4, 7}, mf)) return false;
    if (!write_bool_dump(dump_dir, "diff.flat_action_mask", diff.flat_action_mask, {1, window_size, 28}, mf)) return false;
    for (int step = 0; step < 20; ++step) {
        char boundary[64];
        std::snprintf(boundary, sizeof(boundary), "diff.step%02d.current_x_before", step);
        if (!write_f32_dump(dump_dir, boundary, diff.current_x_before[(size_t) step], {1, window_size, 28}, mf)) return false;
        std::snprintf(boundary, sizeof(boundary), "diff.step%02d.pred_eps", step);
        if (!write_f32_dump(dump_dir, boundary, diff.pred_eps[(size_t) step], {1, window_size, 28}, mf)) return false;
        std::snprintf(boundary, sizeof(boundary), "diff.step%02d.z", step);
        if (!write_f32_dump(dump_dir, boundary, diff.z[(size_t) step], {1, window_size, 28}, mf)) return false;
        std::snprintf(boundary, sizeof(boundary), "diff.step%02d.x_after_denoise", step);
        if (!write_f32_dump(dump_dir, boundary, diff.after_denoise[(size_t) step], {1, window_size, 28}, mf)) return false;
        std::snprintf(boundary, sizeof(boundary), "diff.step%02d.x_after_noise_add", step);
        if (!write_f32_dump(dump_dir, boundary, diff.after_noise_add[(size_t) step], {1, window_size, 28}, mf)) return false;
        std::snprintf(boundary, sizeof(boundary), "diff.step%02d.x_after_clip", step);
        if (!write_f32_dump(dump_dir, boundary, diff.after_clip[(size_t) step], {1, window_size, 28}, mf)) return false;
        std::snprintf(boundary, sizeof(boundary), "diff.step%02d.x_after_mask", step);
        if (!write_f32_dump(dump_dir, boundary, diff.after_mask[(size_t) step], {1, window_size, 28}, mf)) return false;
    }
    if (!write_f32_dump(dump_dir, "diff.actions_all_timesteps", diff.actions_all_timesteps, {1, window_size, 4, 7}, mf)) return false;
    if (!write_f32_dump(dump_dir, "sample_actions.final_action_normalized", diff.final_actions, {1, 4, 7}, mf)) return false;
    if (!write_f32_dump(dump_dir, "action_final", diff.final_actions, {1, 4, 7}, mf)) return false;

    std::vector<float> unnorm_actions;
    if (!unnormalize_action(g, unnorm_dataset, diff.final_actions, unnorm_actions)) return false;
    if (!write_f32_dump(dump_dir, "action_final_unnormalized", unnorm_actions, {1, 4, 7}, mf)) return false;
    return true;
}

bool octo_tokenize_text(const std::string& ckpt_path,
                        const std::string& text,
                        std::vector<int32_t>& input_ids,
                        std::vector<int32_t>& attention_mask) {
    gguf_reader g{"octo"};
    if (!g.open(ckpt_path)) return false;

    std::vector<uint8_t> spm_bytes;
    if (!read_kv_u8_array(g, "octo.tokenizer.spm_model", spm_bytes)) return false;
    const uint32_t eos_id = g.has("octo.tokenizer.eos_id") ? g.u32("octo.tokenizer.eos_id") : 1;
    const uint32_t pad_id = g.has("octo.tokenizer.pad_id") ? g.u32("octo.tokenizer.pad_id") : 0;
    const int64_t max_length = g.has("octo.tokens.language") ? g.u32("octo.tokens.language") : 16;

    sentencepiece::SentencePieceProcessor sp;
    const auto status = sp.LoadFromSerializedProto(
        absl::string_view(reinterpret_cast<const char *>(spm_bytes.data()), spm_bytes.size()));
    if (!status.ok()) {
        std::fprintf(stderr, "vla(octo): sentencepiece LoadFromSerializedProto failed: %s\n", status.ToString().c_str());
        return false;
    }

    std::vector<int> ids = sp.EncodeAsIds(text);
    if ((int64_t) ids.size() > max_length - 1) ids.resize((size_t) (max_length - 1));  // reserve 1 slot for EOS
    input_ids.assign(ids.begin(), ids.end());
    input_ids.push_back((int32_t) eos_id);
    attention_mask.assign(input_ids.size(), 1);
    input_ids.resize((size_t) max_length, (int32_t) pad_id);
    attention_mask.resize((size_t) max_length, 0);
    return true;
}

// Shared tail of the live/cold-start prediction path: given already-assembled per-view
// observation+task images (obs.shape={1,window_size,3,side,side}, task.shape={1,3,side,side})
// and already-tokenized language, runs obs tokenizer x2 -> T5 encoder -> language ->
// assemble -> causal mask -> block transformer -> diffusion head -> unnormalize. Used by
// both octo_predict_from_images (CLI: tokenizes its own instruction via SentencePiece) and
// OctoModelArch::predict (server: receives already-tokenized Inputs::lang_tokens from the
// client, matching every other arch's client-tokenizes/server-embeds convention).
// wrist_real=false marks the wrist observation as a zero-fill placeholder (single-camera
// checkpoint or a caller that supplied no wrist view) -- its causal-mask key_valid entries
// are forced false so the transformer treats it as absent padding, not a real observation.
static bool octo_run_pipeline(gguf_reader& g, int window_size,
                              const NpyU8& primary_obs, const NpyU8& primary_task,
                              const NpyU8& wrist_obs, const NpyU8& wrist_task, bool wrist_real,
                              const std::vector<int32_t>& input_ids,
                              const std::vector<int32_t>& attention_mask,
                              const std::string& unnorm_dataset,
                              std::vector<float>& normalized_out,
                              std::vector<float>& unnormalized_out) {
    OctoObsWeights primary_w_g, wrist_w_g;
    if (!read_obs_weights(g, "primary", primary_w_g)) return false;
    if (!read_obs_weights(g, "wrist", wrist_w_g)) return false;

    std::vector<float> tok, primary_proj, primary_pos, wrist_proj, wrist_pos;
    if (!run_one_obs_tokenizer_graph(primary_w_g, primary_obs, primary_task, 256, 256, window_size, tok, primary_proj, primary_pos)) return false;
    if (!run_one_obs_tokenizer_graph(wrist_w_g, wrist_obs, wrist_task, 128, 64, window_size, tok, wrist_proj, wrist_pos)) return false;

    OctoT5Weights t5_w;
    if (!read_t5_weights(g, t5_w)) return false;
    std::vector<float> t5_out;
    if (!run_t5_encoder_graph(g, t5_w, input_ids, attention_mask, t5_out)) return false;

    OctoLanguageWeights lang_w;
    if (!read_language_weights(g, lang_w)) return false;
    NpyF32 t5;
    t5.shape = {1, 16, 768};
    t5.data = t5_out;
    std::vector<float> lang_proj, lang_pos, repeated;
    if (!run_language_graph(lang_w, t5, window_size, lang_proj, lang_pos, repeated)) return false;

    OctoTransformerWeights transformer_w;
    if (!read_transformer_weights(g, transformer_w)) return false;
    NpyBool task_valid, primary_valid, wrist_valid, timestep_valid;
    task_valid.shape = {1};
    task_valid.data = {1};
    primary_valid.shape = {1, window_size};
    primary_valid.data.assign((size_t) window_size, 1);
    wrist_valid.shape = {1, window_size};
    wrist_valid.data.assign((size_t) window_size, wrist_real ? 1 : 0);
    timestep_valid.shape = {1, window_size};
    timestep_valid.data.assign((size_t) window_size, 0);
    timestep_valid.data.back() = 1;  // cold start: only the last slot is the live frame (matches HistoryWrapper.reset)

    OctoTransformerResult bt;
    if (!assemble_transformer_input(lang_pos, primary_pos, wrist_pos, repeated, transformer_w.readout_pos, window_size, bt.input)) return false;
    std::vector<float> blocked_mask;
    if (!build_transformer_mask(task_valid, primary_valid, wrist_valid, timestep_valid, window_size, bt.blocked_mask, blocked_mask)) return false;
    if (!run_transformer_graph(transformer_w, bt.input, blocked_mask, window_size, bt)) return false;

    OctoDiffusionWeights diffusion_w;
    if (!read_diffusion_weights(g, diffusion_w)) return false;
    std::random_device rd;
    std::mt19937 rng(rd());
    if (!run_diffusion_live(diffusion_w, bt.readout_action, window_size, rng, normalized_out)) return false;

    return unnormalize_action(g, unnorm_dataset, normalized_out, unnormalized_out);
}

// TIP-BUILD-OCTO-GPU-B: predict()-only variant of octo_run_pipeline -- uses the 5 *_resident
// graph functions, now computing on `backend` (the model's real backend: CUDA when available,
// CPU otherwise) instead of a fresh per-call CPU backend, referencing weights resident on
// `ctx_w` (m->ctx_weights, populated once by load_all_tensors -- the same call used before
// TIP-A/B, on whatever device `backend` is). `io` is the model's persistent, already-open
// gguf_reader (m->io, opened once in octo_create) -- unnormalize_action only reads the small
// in-memory "octo.dataset_statistics" metadata string via it (metadata is parsed once at
// gguf_reader::open() time, so this is not a disk touch as long as `io` is already open,
// which it is here). octo_run_pipeline above stays untouched for octo_predict_from_images
// (vla-cli --model octo, a one-shot CLI invocation) and octo_dump_tokenizer_case/octo_free_sample_case.
static bool octo_run_pipeline_resident(ggml_context * ctx_w, ggml_backend_t backend, gguf_reader& io, int window_size,
                                       const NpyU8& primary_obs, const NpyU8& primary_task,
                                       const NpyU8& wrist_obs, const NpyU8& wrist_task, bool wrist_real,
                                       const std::vector<int32_t>& input_ids,
                                       const std::vector<int32_t>& attention_mask,
                                       const std::string& unnorm_dataset,
                                       std::vector<float>& normalized_out,
                                       std::vector<float>& unnormalized_out,
                                       float& ms_vision_out,
                                       float& ms_inference_out) {
    using clock = std::chrono::steady_clock;

    const auto t_vision0 = clock::now();
    std::vector<float> tok, primary_proj, primary_pos, wrist_proj, wrist_pos;
    if (!run_one_obs_tokenizer_graph_resident(ctx_w, backend, "primary", primary_obs, primary_task, 256, 256, window_size, tok, primary_proj, primary_pos)) return false;
    if (!run_one_obs_tokenizer_graph_resident(ctx_w, backend, "wrist", wrist_obs, wrist_task, 128, 64, window_size, tok, wrist_proj, wrist_pos)) return false;
    ms_vision_out = std::chrono::duration<float, std::milli>(clock::now() - t_vision0).count();

    const auto t_inference0 = clock::now();
    std::vector<float> t5_out;
    if (!run_t5_encoder_graph_resident(ctx_w, backend, input_ids, attention_mask, t5_out)) return false;

    NpyF32 t5;
    t5.shape = {1, 16, 768};
    t5.data = t5_out;
    std::vector<float> lang_proj, lang_pos, repeated;
    if (!run_language_graph_resident(ctx_w, backend, t5, window_size, lang_proj, lang_pos, repeated)) return false;

    ggml_tensor * readout_pos_r = wt(ctx_w, "octo.readout.action.pos_embd");
    if (!readout_pos_r) return false;
    const std::vector<float> readout_pos = tensor_to_vec(readout_pos_r);

    NpyBool task_valid, primary_valid, wrist_valid, timestep_valid;
    task_valid.shape = {1};
    task_valid.data = {1};
    primary_valid.shape = {1, window_size};
    primary_valid.data.assign((size_t) window_size, 1);
    wrist_valid.shape = {1, window_size};
    wrist_valid.data.assign((size_t) window_size, wrist_real ? 1 : 0);
    timestep_valid.shape = {1, window_size};
    timestep_valid.data.assign((size_t) window_size, 0);
    timestep_valid.data.back() = 1;  // cold start: only the last slot is the live frame (matches HistoryWrapper.reset)

    OctoTransformerResult bt;
    if (!assemble_transformer_input(lang_pos, primary_pos, wrist_pos, repeated, readout_pos, window_size, bt.input)) return false;
    std::vector<float> blocked_mask;
    if (!build_transformer_mask(task_valid, primary_valid, wrist_valid, timestep_valid, window_size, bt.blocked_mask, blocked_mask)) return false;
    if (!run_transformer_graph_resident(ctx_w, backend, bt.input, blocked_mask, window_size, bt)) return false;

    std::random_device rd;
    std::mt19937 rng(rd());
    if (!run_diffusion_live_resident(ctx_w, backend, bt.readout_action, window_size, rng, normalized_out)) return false;
    ms_inference_out = std::chrono::duration<float, std::milli>(clock::now() - t_inference0).count();

    return unnormalize_action(io, unnorm_dataset, normalized_out, unnormalized_out);
}

// Cold start: only ONE live frame is available (a single vla-cli snapshot or server
// request), but the model expects `window_size` observation timesteps. Repeat the live
// frame into every slot; the caller's timestep_valid marks all-but-the-last as padding
// (see octo_run_pipeline) so the causal mask treats only the last slot as "real" -- matches
// OctoPt's HistoryWrapper.reset() cold start for window_size=2, and generalizes to
// window_size=1 (no padding slot at all, the single slot IS the live frame). No goal image
// (language-only conditioning): task is zero-filled, matching OctoModelPt.create_tasks(texts=...).
static void octo_build_cold_start_obs_task(const uint8_t* rgb, int sw, int sh, int side,
                                           int window_size, NpyU8& obs, NpyU8& task) {
    std::vector<uint8_t> frame;
    resize_to_planar_chw(rgb, sw, sh, side, frame);
    obs.shape = {1, window_size, 3, side, side};
    obs.data.resize((size_t) window_size * 3 * side * side);
    for (int t = 0; t < window_size; ++t) {
        std::copy(frame.begin(), frame.end(), obs.data.begin() + (ptrdiff_t) t * (ptrdiff_t) frame.size());
    }
    task.shape = {1, 3, side, side};
    task.data.assign((size_t) 3 * side * side, 0);
}

bool octo_predict_from_images(const std::string& ckpt_path,
                              const uint8_t* primary_rgb, int primary_w, int primary_h,
                              const uint8_t* wrist_rgb, int wrist_w, int wrist_h,
                              const std::string& instruction,
                              OctoCliAction& out,
                              const std::string& unnorm_dataset) {
    gguf_reader g{"octo"};
    if (!g.open(ckpt_path)) return false;
    const int window_size = (int) g.u32("octo.window_size");
    if (window_size < 1 || window_size > kMaxHorizon) {
        std::fprintf(stderr, "vla(octo): octo.window_size=%d out of supported range [1, %d]\n",
                     window_size, kMaxHorizon);
        return false;
    }

    std::vector<int32_t> input_ids, attention_mask;
    if (!octo_tokenize_text(ckpt_path, instruction, input_ids, attention_mask)) return false;

    NpyU8 primary_obs, primary_task, wrist_obs, wrist_task;
    octo_build_cold_start_obs_task(primary_rgb, primary_w, primary_h, 256, window_size, primary_obs, primary_task);
    octo_build_cold_start_obs_task(wrist_rgb, wrist_w, wrist_h, 128, window_size, wrist_obs, wrist_task);

    return octo_run_pipeline(g, window_size, primary_obs, primary_task, wrist_obs, wrist_task,
                             /*wrist_real=*/true, input_ids, attention_mask, unnorm_dataset,
                             out.normalized, out.unnormalized);
}

// Server-facing entry point (vla-server, TIP-CLIENT): in.images[0] is the primary view,
// in.images[1] the wrist view if the client sent one (single-camera checkpoints/clients
// omit it -- zero-filled and masked invalid via octo_run_pipeline's wrist_real=false, same
// fallback as the golden-case loaders above). Unlike most other archs' predict(), this
// returns the UN-normalized (world-unit) action rather than the normalized one: Octo's
// dataset_statistics lives embedded in the multi-hundred-MB checkpoint GGUF itself (not a
// small sibling stats.json a client can cheaply hold locally the way gr00t's
// --stats-json works), so un-normalizing server-side and reusing the already
// golden-verified unnormalize_action() is the only practical option without shipping the
// whole GGUF to the client just to read its JSON metadata. See TIP-CLIENT report for the
// full rationale; the client (adapters.py) only needs to invert+binarize the gripper dim,
// not repeat the mean/std un-normalization.
std::vector<float> OctoModelArch::predict(const Inputs& in) {
    const auto t_total0 = std::chrono::steady_clock::now();
    if (in.n_images < 1 || !in.images) {
        std::fprintf(stderr, "vla(octo): predict needs at least 1 image (primary)\n");
        return {};
    }
    if (in.images[0].format != PixelFormat::U8) {
        std::fprintf(stderr, "vla(octo): predict only supports PixelFormat::U8 images "
                              "(client must send RGB_U8, already rotated+resized -- TIP-P)\n");
        return {};
    }
    const bool wrist_real = in.n_images >= 2;
    if (wrist_real && in.images[1].format != PixelFormat::U8) {
        std::fprintf(stderr, "vla(octo): predict only supports PixelFormat::U8 images\n");
        return {};
    }
    if (in.n_lang != (int) language_tokens || in.attention_mask_n != (int) language_tokens || !in.attention_mask) {
        std::fprintf(stderr,
                     "vla(octo): predict expects lang_tokens AND attention_mask of exactly %lld "
                     "entries each (client tokenizes with t5-base, max_length=%lld, "
                     "padding=\"max_length\" -- Octo's T5 encoder needs real padding info, unlike "
                     "archs that derive their own mask); got n_lang=%d attention_mask=%s attention_mask_n=%d\n",
                     (long long) language_tokens, (long long) language_tokens, in.n_lang,
                     in.attention_mask ? "set" : "null", in.attention_mask_n);
        return {};
    }

    // TIP-BUILD-OCTO-GPU-B: no per-call GGUF reopen -- weights are resident on `backend`
    // (loaded once in octo_create via load_all_tensors), and the 5 stages now compute on
    // that same real backend (CUDA when available, CPU otherwise) instead of a fresh
    // per-call CPU backend. `io` (opened once in octo_create too) covers the one remaining
    // in-memory metadata read (octo.dataset_statistics, inside unnormalize_action).

    NpyU8 primary_obs, primary_task, wrist_obs, wrist_task;
    octo_build_cold_start_obs_task((const uint8_t*) in.images[0].data, in.images[0].w, in.images[0].h,
                                   256, (int) window_size, primary_obs, primary_task);
    if (wrist_real) {
        octo_build_cold_start_obs_task((const uint8_t*) in.images[1].data, in.images[1].w, in.images[1].h,
                                       128, (int) window_size, wrist_obs, wrist_task);
    } else {
        wrist_obs.shape = {1, window_size, 3, 128, 128};
        wrist_obs.data.assign((size_t) window_size * 3 * 128 * 128, 0);
        wrist_task.shape = {1, 3, 128, 128};
        wrist_task.data.assign((size_t) 3 * 128 * 128, 0);
    }

    const std::vector<int32_t> input_ids(in.lang_tokens, in.lang_tokens + in.n_lang);
    const std::vector<int32_t> attention_mask(in.attention_mask, in.attention_mask + in.attention_mask_n);

    std::vector<float> normalized, unnormalized;
    float ms_vision = 0.f, ms_inference = 0.f;
    if (!octo_run_pipeline_resident(ctx_weights, backend, io, (int) window_size, primary_obs, primary_task, wrist_obs, wrist_task,
                                    wrist_real, input_ids, attention_mask, /*unnorm_dataset=*/"",
                                    normalized, unnormalized, ms_vision, ms_inference)) {
        return {};
    }
    stats.ms_vision = ms_vision;
    stats.ms_inference = ms_inference;
    stats.ms_total = std::chrono::duration<float, std::milli>(
        std::chrono::steady_clock::now() - t_total0).count();
    return unnormalized;
}

bool octo_free_sample_case(const std::string& ckpt_path,
                           const std::string& case_dir,
                           int n_samples,
                           uint32_t seed,
                           std::vector<float>& samples_out,
                           std::vector<float>& noise_out) {
    if (n_samples <= 0) {
        std::fprintf(stderr, "vla(octo): free-sample n_samples must be > 0\n");
        return false;
    }
    gguf_reader g{"octo"};
    if (!g.open(ckpt_path)) return false;
    const int window_size = (int) g.u32("octo.window_size");
    if (window_size < 1 || window_size > kMaxHorizon) {
        std::fprintf(stderr, "vla(octo): octo.window_size=%d out of supported range [1, %d]\n",
                     window_size, kMaxHorizon);
        return false;
    }

    // Observation: load once from the golden case, exactly as octo_dump_tokenizer_case
    // does (same helpers, same zero-fill fallback for cases with no task image or,
    // for a single-camera checkpoint, no wrist observation at all).
    NpyU8 primary_obs, wrist_obs, primary_task, wrist_task;
    bool wrist_obs_real = true;
    if (!parse_npy_u8(case_dir + "/tensors/input.observation.image_primary.npy", primary_obs)) return false;
    if (!parse_npy_u8_or_zero_obs(case_dir + "/tensors/input.observation.image_wrist.npy",
                                  window_size, 128, wrist_obs, wrist_obs_real)) return false;
    if (!parse_npy_u8_or_zero_task(case_dir + "/tensors/input.task.image_primary.npy", primary_obs, primary_task)) return false;
    if (!parse_npy_u8_or_zero_task(case_dir + "/tensors/input.task.image_wrist.npy", wrist_obs, wrist_task)) return false;
    NpyI32 input_ids, attention_mask;
    if (!parse_npy_i32(case_dir + "/tensors/input.task.language_instruction.input_ids.npy", input_ids)) return false;
    if (!parse_npy_i32(case_dir + "/tensors/input.task.language_instruction.attention_mask.npy", attention_mask)) return false;
    NpyBool task_valid, primary_valid, wrist_valid, timestep_valid;
    if (!parse_npy_bool(case_dir + "/tensors/input.task.pad_mask_dict.language_instruction.npy", task_valid) ||
        !parse_npy_bool(case_dir + "/tensors/input.observation.pad_mask_dict.image_primary.npy", primary_valid) ||
        !parse_npy_bool_or_zero(case_dir + "/tensors/input.observation.pad_mask_dict.image_wrist.npy", window_size, wrist_valid) ||
        !parse_npy_bool(case_dir + "/tensors/input.observation.timestep_pad_mask.npy", timestep_valid)) return false;
    if (!wrist_obs_real) {
        std::fill(wrist_valid.data.begin(), wrist_valid.data.end(), (uint8_t) 0);
    }
    std::vector<uint8_t> blocked_bytes;
    std::vector<float> blocked_mask;
    if (!build_transformer_mask(task_valid, primary_valid, wrist_valid, timestep_valid, window_size, blocked_bytes, blocked_mask)) return false;

    // Weights: read once (matches how OctoPt holds a loaded model in memory across
    // repeated sample_actions() calls, rather than reloading per sample).
    // run_t5_encoder_graph/run_transformer_graph move their weight arg's vectors into
    // the ggml payload, so each of the N passes gets a fresh copy of just those two
    // (T5, transformer) -- the move-based signatures are unchanged/reused as-is.
    OctoObsWeights primary_w_g, wrist_w_g;
    if (!read_obs_weights(g, "primary", primary_w_g)) return false;
    if (!read_obs_weights(g, "wrist", wrist_w_g)) return false;
    OctoT5Weights t5_w_master;
    if (!read_t5_weights(g, t5_w_master)) return false;
    OctoLanguageWeights lang_w;
    if (!read_language_weights(g, lang_w)) return false;
    OctoTransformerWeights transformer_w_master;
    if (!read_transformer_weights(g, transformer_w_master)) return false;
    OctoDiffusionWeights diffusion_w;
    if (!read_diffusion_weights(g, diffusion_w)) return false;

    constexpr int action = 28;
    samples_out.assign((size_t) n_samples * 4 * 7, 0.0f);
    noise_out.assign((size_t) n_samples * (size_t) window_size * action, 0.0f);

    for (int i = 0; i < n_samples; ++i) {
        std::vector<float> tok, primary_proj, primary_pos, wrist_proj, wrist_pos;
        if (!run_one_obs_tokenizer_graph(primary_w_g, primary_obs, primary_task, 256, 256, window_size, tok, primary_proj, primary_pos)) return false;
        if (!run_one_obs_tokenizer_graph(wrist_w_g, wrist_obs, wrist_task, 128, 64, window_size, tok, wrist_proj, wrist_pos)) return false;

        OctoT5Weights t5_w = t5_w_master;
        std::vector<float> t5_out;
        if (!run_t5_encoder_graph(g, t5_w, input_ids.data, attention_mask.data, t5_out)) return false;

        NpyF32 t5;
        t5.shape = {1, 16, 768};
        t5.data = t5_out;
        std::vector<float> lang_proj, lang_pos, repeated;
        if (!run_language_graph(lang_w, t5, window_size, lang_proj, lang_pos, repeated)) return false;

        OctoTransformerWeights transformer_w = transformer_w_master;
        OctoTransformerResult bt;
        if (!assemble_transformer_input(lang_pos, primary_pos, wrist_pos, repeated, transformer_w.readout_pos, window_size, bt.input)) return false;
        if (!run_transformer_graph(transformer_w, bt.input, blocked_mask, window_size, bt)) return false;

        std::mt19937 rng(seed + (uint32_t) i);
        std::vector<float> sample, noise;
        if (!run_diffusion_live(diffusion_w, bt.readout_action, window_size, rng, sample, &noise)) return false;
        std::copy(sample.begin(), sample.end(), samples_out.begin() + (size_t) i * 28);
        std::copy(noise.begin(), noise.end(), noise_out.begin() + (size_t) i * (size_t) window_size * action);
    }
    return true;
}

}  // namespace vla
