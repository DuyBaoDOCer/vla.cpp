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
// TIP-05: an L1/proprio checkpoint inserts a 3rd obs group (7 proprio tokens) right after
// wrist, per-timestep: primary(256) + wrist(64) + proprio(0 or 7) + repeated-language(16)
// + readout(1). kStepTokens is the historical no-proprio constant (still what every
// diffusion checkpoint uses, n_proprio=0); octo_step_tokens/octo_seq_len generalize it.
constexpr int kStepTokens = 337;
// task_language prefix tokens (once, not repeated per timestep).
constexpr int kTaskTokens = 16;
// Shared max-horizon slab size of the obs/task pos-embedding tables written by
// scripts/convert_octo_to_gguf.py (same for every checkpoint regardless of the
// window_size actually trained/used) -- see TIP-C1. window_size must fit within it.
constexpr int kMaxHorizon = 10;
// LowdimObsTokenizerPt(obs_keys=["proprio"]) always emits exactly 1 token per state
// dimension (discretize=False or True doesn't change the token *count*, only how each
// dimension is encoded before the shared Linear projection) -- octo-small-1.5's proprio
// is 7-dim, so this is fixed regardless of checkpoint.
constexpr int kProprioTokens = 7;

inline int octo_step_tokens(int n_proprio_tokens) {
    return kStepTokens + n_proprio_tokens;
}

// seq = kTaskTokens + window_size * octo_step_tokens(n_proprio_tokens). n_proprio_tokens
// defaults to 0 (every pre-TIP-05 call site: diffusion checkpoints have no proprio group)
// so this stays byte-identical to the old window_size-only signature for them.
inline int octo_seq_len(int64_t window_size, int n_proprio_tokens = 0) {
    return kTaskTokens + (int) window_size * octo_step_tokens(n_proprio_tokens);
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
    // "diffusion" (default, backward-compat for GGUFs converted before this key existed)
    // or "l1" (aloha jitter-adapted L1 head; forward wired in TIP-05).
    std::string head_type = "diffusion";
    // TIP-05: proprio tokenizer is detected from tensor presence (no dedicated GGUF KV
    // key), independent of head_type -- see detect_proprio(). proprio_in_dim is
    // octo.obs.proprio.proj.weight's in-dim: 1 (LowdimObsTokenizerPt discretize=False,
    // the expected/documented case) or 256 (discretize=True, bin one-hot).
    bool has_proprio = false;
    int64_t proprio_in_dim = 0;
    int64_t primary_tokens = 256;
    int64_t wrist_tokens = 64;
    int64_t language_tokens = 16;
    int64_t diffusion_steps = 20;

    gguf_reader io{"octo"};
    std::vector<ggml_tensor *> tensors;

    std::vector<float> predict(const Inputs& in) override;
};

// TIP-05: proprio has no dedicated GGUF metadata key (scripts/convert_octo_to_gguf.py only
// writes the octo.obs.proprio.* tensors when the source checkpoint's config has a "proprio"
// observation_tokenizer) -- detect it from tensor presence instead, same way every other
// per-checkpoint-optional group in this file is handled. Called from both load_config (the
// resident server/dump/free-sample paths, which already have a gguf_reader) and
// octo_predict_from_images (the CLI path, which builds its OctoModelArch by hand rather than
// through load_config).
static void detect_proprio(const gguf_reader& g, bool& has_proprio, int64_t& proprio_in_dim) {
    const ggml_tensor * proj = g.meta("octo.obs.proprio.proj.weight");
    has_proprio = proj != nullptr;
    proprio_in_dim = has_proprio ? proj->ne[0] : 0;
}

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
    // octo.action.head_type is optional: GGUFs converted before this key existed (all
    // pre-TIP-03 diffusion checkpoints) default to "diffusion", their sole prior behavior.
    m.head_type = g.has("octo.action.head_type") ? g.str("octo.action.head_type") : "diffusion";
    m.primary_tokens = g.u32("octo.tokens.primary");
    m.wrist_tokens = g.u32("octo.tokens.wrist");
    m.language_tokens = g.u32("octo.tokens.language");
    m.diffusion_steps = g.u32("octo.diffusion.steps");
    detect_proprio(g, m.has_proprio, m.proprio_in_dim);
    if (m.has_proprio && m.proprio_in_dim != 1 && m.proprio_in_dim != 256) {
        std::fprintf(stderr, "vla(octo): unexpected octo.obs.proprio.proj.weight in-dim=%lld (expected 1 or 256)\n",
                     (long long) m.proprio_in_dim);
        return false;
    }

    // action_horizon is per-checkpoint (diffusion libero=4, L1 aloha jitter-adapted=20);
    // action_dim stays fixed at 7 (the octo-small-1.5 backbone's action-dim constant,
    // shared by every head type) along with the other M0 backbone consts below.
    if (m.hidden != 384 || m.blocks != 12 || m.heads != 6 || m.ffn != 1536 || m.action_dim != 7) {
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

    // TIP-05: n_state/max_state_dim/real_state_dim were 0 (Octo had no proprio input)
    // before this TIP; a proprio-tokenizer checkpoint has exactly 7 state dims (one
    // token/dim -- see kProprioTokens), reported here so callers (server.cpp) know to
    // supply Inputs::state.
    const int64_t proprio_dim = m.has_proprio ? kProprioTokens : 0;
    m.cfg.n_img = m.primary_tokens + m.wrist_tokens;
    m.cfg.n_lang = m.language_tokens;
    m.cfg.n_state = proprio_dim;
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
    m.cfg.max_state_dim = proprio_dim;
    m.cfg.max_action_dim = m.action_dim;
    m.cfg.real_state_dim = proprio_dim;
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

// TIP-ND1-B: shared backend-selection logic (CUDA -> Metal -> CPU fallback), factored out of
// octo_create so every caller that needs its own resident model load (octo_create itself,
// octo_dump_tokenizer_case_resident, octo_predict_from_images, octo_free_sample_case) shares
// the identical selection order instead of re-duplicating it. `log_prefix` distinguishes the
// log lines of the non-octo_create callers (e.g. "[dump] ") from the live server's.
static ggml_backend_t octo_select_backend(int n_threads, const char * log_prefix) {
    ggml_backend_t backend = nullptr;
#ifdef GGML_USE_CUDA
    backend = ggml_backend_cuda_init(0);
    if (backend) std::printf("vla(octo): %sbackend = CUDA (device 0)\n", log_prefix);
    else         std::fprintf(stderr, "vla(octo): %sggml_backend_cuda_init failed; falling back to CPU\n", log_prefix);
#elif defined(GGML_USE_METAL)
    backend = ggml_backend_metal_init();
    if (backend) std::printf("vla(octo): %sbackend = Metal\n", log_prefix);
    else         std::fprintf(stderr, "vla(octo): %sggml_backend_metal_init failed; falling back to CPU\n", log_prefix);
#endif
    if (!backend) {
        backend = ggml_backend_cpu_init();
        if (!backend) {
            std::fprintf(stderr, "vla(octo): %sggml_backend_cpu_init failed\n", log_prefix);
            return nullptr;
        }
        ggml_backend_cpu_set_n_threads(backend, n_threads);
        std::printf("vla(octo): %sbackend = CPU (%d threads)\n", log_prefix, n_threads);
    }
    return backend;
}

// TIP-BUILD-OCTO-GPU-A: looks up an already-resident weight tensor by its GGUF name (same
// name strings the *_resident graph-building functions below already used as ggml_set_name
// literals, so this is a drop-in replacement for the old "build a fresh tensor + upload from
// a freshly-disk-read host vector" pattern).
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

// TIP-ND1-B: sole obs-tokenizer implementation (was run_one_obs_tokenizer_graph /
// run_one_obs_tokenizer_graph_resident, unified -- TIP-ND1-A proved resident-on-CPU is
// bit-exact with the deleted disk-read/cpu_init original, 20/20 golden cases). Weights come
// from the resident context (`wt()` + `tensor_to_vec()` for the handful needing a host-side
// transform); compute runs on the model's real `backend` (CUDA when available, CPU
// otherwise). `backend` is NOT owned by this function -- never freed here.
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

// TIP-05 Part A: LowdimObsTokenizerPt(obs_keys=["proprio"]) forward. p_norm (z-scored by
// proprio-stats, done by the caller before this graph -- see load_proprio_stats) is turned
// into one token per state dim (7 total): continuous (in_dim=1) feeds p_norm straight into
// the shared Linear(1,384) projection; discretize (in_dim=256) bins p_norm through
// octo.obs.proprio.bin_thresholds into a one-hot(256) vector first. Either way the
// projection's weight/bias is the SAME [in_dim,384] matrix applied independently per
// dim-token (matches BinTokenizerPt/LowdimObsTokenizerPt's nn.Linear applied on the last
// axis of a (...,7,in_dim) tensor -- one shared projection, not 7 per-dim ones), so both
// branches converge on an identical graph shape ([in_dim,7,window_size] -> [384,7,window_size])
// once the host-side `tokens_in` buffer is built.
static bool run_proprio_tokenizer_graph_resident(ggml_context * ctx_w, ggml_backend_t backend,
                                                  const std::vector<float>& proprio_norm,  // [window_size,7], z-scored
                                                  int in_dim,                              // 1 or 256
                                                  int window_size,
                                                  std::vector<float>& pos_out) {            // [window_size,7,384]
    constexpr int n_dims = kProprioTokens;
    if (proprio_norm.size() != (size_t) window_size * n_dims) return false;
    if (in_dim != 1 && in_dim != 256) {
        std::fprintf(stderr, "vla(octo): proprio tokenizer in_dim=%d unsupported (expected 1 or 256)\n", in_dim);
        return false;
    }
    pos_out.assign((size_t) window_size * n_dims * 384, 0.0f);

    ggml_tensor * proj_w_r = wt(ctx_w, "octo.obs.proprio.proj.weight");
    ggml_tensor * proj_b_r = wt(ctx_w, "octo.obs.proprio.proj.bias");
    ggml_tensor * pos_r = wt(ctx_w, "octo.obs.proprio.pos_embd");
    if (!proj_w_r || !proj_b_r || !pos_r) return false;

    std::vector<float> tokens_in((size_t) in_dim * n_dims * window_size, 0.0f);
    if (in_dim == 1) {
        // Continuous: token = p_norm.unsqueeze(-1) -- the raw z-scored scalar itself.
        for (size_t i = 0; i < proprio_norm.size(); ++i) tokens_in[i] = proprio_norm[i];
    } else {
        // Discretize: BinTokenizerPt buckets each dim's p_norm against the fixed (n_bins-1)
        // thresholds octo.obs.proprio.bin_thresholds and one-hot encodes the bucket index --
        // torch.bucketize(x, boundaries) semantics (index = count of boundaries <= x, i.e.
        // TIP-05's "argmax_j (p_norm >= thr[j] & p_norm < thr[j+1])" with thr padded by
        // implicit +/-inf at the ends). NOT exercised by any GGUF available in this sandbox
        // (the only observed checkpoint config is discretize=False/in_dim=1) -- implemented
        // per spec for completeness; TIP-06 parity should confirm bucket-boundary behavior
        // against a real discretize=True golden trace if one ever exists.
        ggml_tensor * thresholds_r = wt(ctx_w, "octo.obs.proprio.bin_thresholds");
        if (!thresholds_r) return false;
        const std::vector<float> thresholds = tensor_to_vec(thresholds_r);
        const int n_thresh = (int) thresholds.size();
        for (int t = 0; t < window_size; ++t) {
            for (int d = 0; d < n_dims; ++d) {
                const float v = proprio_norm[(size_t) t * n_dims + d];
                int bucket = 0;
                while (bucket < n_thresh && thresholds[(size_t) bucket] <= v) ++bucket;
                bucket = std::min(bucket, in_dim - 1);
                tokens_in[((size_t) t * n_dims + d) * in_dim + bucket] = 1.0f;
            }
        }
    }

    ggml_init_params gp = {(size_t) 8 * 1024 * 1024, nullptr, true};
    ggml_context * ctx = ggml_init(gp);
    if (!ctx) return false;

    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, in_dim, n_dims, window_size);
    ggml_set_name(x, "octo.obs.proprio.tokens_in");

    ggml_tensor * pb = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 384, 1, 1);
    ggml_set_name(pb, "octo.obs.proprio.proj.bias3d");

    const std::vector<float> pos_full = tensor_to_vec(pos_r);
    if (pos_full.size() < (size_t) window_size * n_dims * 384) {
        std::fprintf(stderr, "vla(octo): octo.obs.proprio.pos_embd too small for window_size=%d\n", window_size);
        ggml_free(ctx);
        return false;
    }
    std::vector<float> pos_slice(pos_full.begin(), pos_full.begin() + (size_t) window_size * n_dims * 384);
    ggml_tensor * pe = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 384, n_dims, window_size);
    ggml_set_name(pe, "octo.obs.proprio.pos_embd.window");

    ggml_tensor * proj_t = ggml_add(ctx, ggml_mul_mat(ctx, proj_w_r, x), pb);
    ggml_tensor * pos_t = ggml_add(ctx, proj_t, pe);
    ggml_set_name(pos_t, "obs.proprio.pos");
    ggml_set_output(pos_t);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 256, false);
    ggml_build_forward_expand(graph, pos_t);
    ggml_gallocr_t gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!gallocr || !ggml_gallocr_alloc_graph(gallocr, graph)) {
        std::fprintf(stderr, "vla(octo): proprio tokenizer ggml_gallocr_alloc_graph failed\n");
        if (gallocr) ggml_gallocr_free(gallocr);
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_set(x, tokens_in.data(), 0, ggml_nbytes(x));
    ggml_backend_tensor_set(pb, tensor_to_vec(proj_b_r).data(), 0, ggml_nbytes(pb));
    ggml_backend_tensor_set(pe, pos_slice.data(), 0, ggml_nbytes(pe));
    const ggml_status st = ggml_backend_graph_compute(backend, graph);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "vla(octo): proprio tokenizer ggml_backend_graph_compute failed (%d)\n", (int) st);
        ggml_gallocr_free(gallocr);
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_get(pos_t, pos_out.data(), 0, ggml_nbytes(pos_t));

    ggml_gallocr_free(gallocr);
    ggml_free(ctx);
    return true;
}

// TIP-ND1-B: sole language-stage implementation (was run_language_graph /
// run_language_graph_resident, unified). proj_w/proj_b/pos are referenced straight from the
// resident context (no copy, no per-call upload); `inp` (the T5 encoder's per-call output) is
// genuinely new data each call and still gets uploaded fresh.
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
// TIP-ND1-B: sole T5-encoder implementation (was run_t5_encoder_graph /
// run_t5_encoder_graph_resident, unified). The 12 blocks' attn/ffn weights, attn_rel_b, and
// output_norm are referenced directly from the resident context; compute runs on the model's
// real `backend` (CUDA when available, CPU otherwise). The embedding lookup is an in-graph
// ggml_get_rows against the resident octo.t5.tok_embd.weight tensor (same op already used for
// the relative-position-bias gather two lines below), so on a CUDA build the gather itself
// happens on-device, no host round-trip. `backend` is NOT owned by this function -- never
// freed here.
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
    std::vector<float> obs_proprio;  // TIP-05: empty unless n_proprio_tokens > 0.
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

// TIP-05: obs_proprio/n_proprio_tokens are additive -- n_proprio_tokens=0 (obs_proprio
// ignored/may be empty) reproduces the pre-TIP-05 layout byte-for-byte (every existing call
// site keeps doing exactly that). When n_proprio_tokens=7, the proprio group is inserted
// right after wrist and before the repeated-language group, per timestep: primary(256) +
// wrist(64) + proprio(7) + repeated-language(16) + readout(1) -- see kStepTokens's TIP-05
// comment. (Order chosen from the "primary -> wrist -> proprio" obs-group ordering TIP-05
// specifies, placed before the task-derived repeated-language/readout entries exactly like
// primary/wrist already are; TIP-06 should confirm this ordering against a golden L1 dump.)
static bool assemble_transformer_input(const std::vector<float>& task_language,
                                       const std::vector<float>& obs_primary,
                                       const std::vector<float>& obs_wrist,
                                       const std::vector<float>& obs_proprio,
                                       const std::vector<float>& repeated_language,
                                       const std::vector<float>& readout_pos,
                                       int window_size,
                                       int n_proprio_tokens,
                                       std::vector<float>& input) {
    constexpr int hidden = 384;
    const int seq = octo_seq_len(window_size, n_proprio_tokens);
    const int step_tokens = octo_step_tokens(n_proprio_tokens);
    if (task_language.size() != (size_t) 16 * hidden ||
        obs_primary.size() != (size_t) window_size * 256 * hidden ||
        obs_wrist.size() != (size_t) window_size * 64 * hidden ||
        (n_proprio_tokens > 0 && obs_proprio.size() != (size_t) window_size * n_proprio_tokens * hidden) ||
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
        size_t off = (size_t) 320 * hidden;
        if (n_proprio_tokens > 0) {
            std::copy_n(obs_proprio.begin() + (size_t) t * n_proprio_tokens * hidden, (size_t) n_proprio_tokens * hidden,
                        input.begin() + dst + off);
            off += (size_t) n_proprio_tokens * hidden;
        }
        std::copy_n(repeated_language.begin() + (size_t) t * 16 * hidden, (size_t) 16 * hidden,
                    input.begin() + dst + off);
        off += (size_t) 16 * hidden;
        std::copy_n(readout_pos.begin() + (size_t) t * hidden, hidden,
                    input.begin() + dst + off);
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

// TIP-ND1-B: sole score-actor implementation (was run_score_actor_graph /
// run_score_actor_graph_resident, unified) -- called once per denoising step (20x per
// predict()), so this is the highest call-frequency of the 5 stages. All 9 fixed weights + the
// 3 residual blocks' 6 weights each are referenced directly from the resident context; only
// the 3 genuinely-per-call input tensors (time, readout embedding, noisy action) get built +
// uploaded fresh each call.
static bool run_score_actor_graph_resident(ggml_context * ctx_w, ggml_backend_t backend,
                                           const std::vector<float>& readout_action,
                                           const std::vector<float>& noisy_action,
                                           int time_value,
                                           int window_size,
                                           int action_total,
                                           std::vector<float>& pred_eps) {
    constexpr int hidden = 384;
    const int action = action_total;
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

// TIP-ND1-B: which noise the diffusion reverse process consumes. REPLAY reads golden
// z.npy/initial_noise.npy from a case dir (ctest parity -- deterministic, must match exactly).
// RANDOM samples N(0,1) from a caller-owned RNG (live predict()/free-sample -- stochastic by
// design, not compared bit-exact anywhere). CLIENT_NOISE is reserved for a future round
// (client-supplied deterministic noise, e.g. a PredictRequest.noise field) -- NOT implemented
// here; the slot exists so this signature doesn't need to change again when that lands.
enum class OctoNoiseSourceKind {
    REPLAY,
    RANDOM,
    // CLIENT_NOISE,  // TODO(in.noise): caller-supplied noise buffer, not implemented yet.
};

struct OctoNoiseSource {
    OctoNoiseSourceKind kind;
    std::string case_dir;          // REPLAY only: golden case directory to read z.npy/initial_noise.npy from.
    std::mt19937 * rng = nullptr;  // RANDOM only: not owned by this struct.
};

// TIP-ND1-B: sole diffusion implementation (was run_diffusion_replay / _resident /
// run_diffusion_live / _resident, unified -- those 4 differed along two orthogonal axes:
// compute-source [gone now, resident-only] and noise-source [now `noise`]). REPLAY behavior is
// byte-for-byte the same as the deleted run_diffusion_replay_resident (same npy reads, same
// unconditional per-step z.npy read, same masked-noise-substitution using flat_action_mask).
// RANDOM behavior matches the deleted run_diffusion_live_resident exactly: z is only drawn
// from `noise.rng` when time_value > 0 (same RNG draw count/order as before -- the old code
// never drew a step-20 noise sample either, since DDPM's reverse process adds no noise at
// t=0), and flat_action_mask is all-valid so the masked-noise-substitution step below is
// structurally a no-op for it, matching the old live path that omitted that step outright.
static bool run_diffusion_resident(ggml_context * ctx_w, ggml_backend_t backend,
                                   const OctoNoiseSource& noise,
                                   const std::vector<float>& readout_action,
                                   int window_size,
                                   int action_total,
                                   OctoDiffusionResult& result) {
    constexpr int steps = 20;
    const int width = window_size;
    const int action = action_total;
    constexpr float max_action = 5.0f;
    if (readout_action.size() != (size_t) 384 * width) return false;

    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::string prefix;
    if (noise.kind == OctoNoiseSourceKind::REPLAY) {
        prefix = noise.case_dir + "/tensors/action_head.predict_action.";
        if (!load_f32_shape(prefix + "initial_noise.npy", {1, window_size, action}, result.initial_noise)) return false;
        NpyBool action_mask;
        if (!parse_npy_bool(prefix + "action_mask.npy", action_mask) || action_mask.shape != std::vector<int64_t>({1, window_size, 4, 7})) return false;
        result.action_mask = std::move(action_mask.data);
        NpyBool flat_mask;
        if (!parse_npy_bool(prefix + "flat_action_mask.npy", flat_mask) || flat_mask.shape != std::vector<int64_t>({1, window_size, action})) return false;
        result.flat_action_mask = std::move(flat_mask.data);
    } else {
        result.initial_noise.resize((size_t) width * action);
        for (float& v : result.initial_noise) v = normal(*noise.rng);
        // All-true: octo-small-1.5 has real_action_dim == max_action_dim == 7 (full
        // action_horizon), so this makes the masked-noise-substitution step below a
        // structural no-op, matching the old live path that omitted it outright.
        result.action_mask.assign((size_t) width * 4 * 7, 1);
        result.flat_action_mask.assign((size_t) width * action, 1);
    }

    OctoDiffusionSchedule sched = make_cosine_schedule();
    std::vector<float> x = result.initial_noise;
    for (int step = 0; step < steps; ++step) {
        const int time_value = steps - 1 - step;
        result.current_x_before[(size_t) step] = x;
        if (!run_score_actor_graph_resident(ctx_w, backend, readout_action, x, time_value, window_size, action_total, result.pred_eps[(size_t) step])) return false;

        if (noise.kind == OctoNoiseSourceKind::REPLAY) {
            char stem[96];
            std::snprintf(stem, sizeof(stem), "step_%02d.t_%02d.", step, time_value);
            if (!load_f32_shape(prefix + stem + "z.npy", {1, window_size, action}, result.z[(size_t) step])) return false;
        } else if (time_value > 0) {
            result.z[(size_t) step].resize((size_t) width * action);
            for (float& v : result.z[(size_t) step]) v = normal(*noise.rng);
        } else {
            result.z[(size_t) step].assign((size_t) width * action, 0.0f);  // never read below (time_value == 0)
        }

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

// TIP-05 Part B: L1 action head forward -- ContinuousActionHeadPt.forward/predict_action,
// replacing the diffusion head entirely when head_type=="l1" (routed in
// octo_run_pipeline_resident). Dump points for TIP-06 parity: map_attn_out (post out_proj,
// pre-LN/MLP), map_emb (post residual+MLP -- MAPHeadPt's final per-timestep embedding),
// mean_normalized (post mean_proj+tanh*max_action, ALL window timesteps), final_actions
// (mean_normalized's last window timestep only, what predict_action returns).
struct OctoL1HeadResult {
    std::vector<float> map_attn_out;    // [384,window_size]
    std::vector<float> map_emb;         // [384,window_size]
    std::vector<float> mean_normalized; // [action_total,window_size]
    std::vector<float> final_actions;   // [action_total] -- mean_normalized[:, window_size-1]
};

// readout_action: [384,window_size] (the block-transformer's readout_action output, one
// 384-dim token per timestep -- readouts.action=1 so there is exactly one token/timestep
// already, matching MAPHeadPt's (b,w,1,384) input shape with the "1" dim implicit). No
// attention happens ACROSS window_size or across batch: MAPHeadPt's nn.MultiheadAttention
// runs independently per (batch,timestep) pair (query=probe, key=value=that timestep's
// single token) -- window_size is therefore modeled as a batch axis (ggml ne3) here, heads
// as a second batch axis (ne2), never mixed via the ne0/ne1 axes mul_mat actually contracts
// over/compares. See TIP-05 Completion Report for the full ggml shape derivation.
static bool run_l1_action_head_graph_resident(ggml_context * ctx_w, ggml_backend_t backend,
                                              const std::vector<float>& readout_action,
                                              int window_size,
                                              int action_total,
                                              OctoL1HeadResult& result) {
    constexpr int hidden = 384;
    constexpr int map_heads = 8;          // TIP-05: 8, NOT the block-transformer's 6.
    constexpr int map_head_dim = hidden / map_heads;  // 48
    constexpr float ln_eps = 1e-6f;
    constexpr float max_action = 5.0f;
    const int width = window_size;
    if (readout_action.size() != (size_t) hidden * width || action_total <= 0) return false;

    ggml_tensor * probe_r      = wt(ctx_w, "octo.head.l1.map.probe");
    ggml_tensor * qkv_w_r      = wt(ctx_w, "octo.head.l1.map.attn_qkv.weight");
    ggml_tensor * qkv_b_r      = wt(ctx_w, "octo.head.l1.map.attn_qkv.bias");
    ggml_tensor * o_w_r        = wt(ctx_w, "octo.head.l1.map.attn_o.weight");
    ggml_tensor * o_b_r        = wt(ctx_w, "octo.head.l1.map.attn_o.bias");
    ggml_tensor * norm_w_r     = wt(ctx_w, "octo.head.l1.map.norm.weight");
    ggml_tensor * norm_b_r     = wt(ctx_w, "octo.head.l1.map.norm.bias");
    ggml_tensor * ffn_up_w_r   = wt(ctx_w, "octo.head.l1.map.ffn_up.weight");
    ggml_tensor * ffn_up_b_r   = wt(ctx_w, "octo.head.l1.map.ffn_up.bias");
    ggml_tensor * ffn_down_w_r = wt(ctx_w, "octo.head.l1.map.ffn_down.weight");
    ggml_tensor * ffn_down_b_r = wt(ctx_w, "octo.head.l1.map.ffn_down.bias");
    ggml_tensor * mean_w_r     = wt(ctx_w, "octo.head.l1.mean_proj.weight");
    ggml_tensor * mean_b_r     = wt(ctx_w, "octo.head.l1.mean_proj.bias");
    if (!probe_r || !qkv_w_r || !qkv_b_r || !o_w_r || !o_b_r || !norm_w_r || !norm_b_r ||
        !ffn_up_w_r || !ffn_up_b_r || !ffn_down_w_r || !ffn_down_b_r || !mean_w_r || !mean_b_r) {
        return false;
    }

    ggml_init_params gp = {(size_t) 16 * 1024 * 1024, nullptr, true};
    ggml_context * ctx = ggml_init(gp);
    if (!ctx) return false;

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, width);
    ggml_set_name(x, "l1_head.readout_action");

    // probe is resident weight octo.head.l1.map.probe, PyTorch shape (1,1,384) -> GGUF
    // drops the leading size-1 dims at load (ggml_n_dims), so it's already a plain
    // ne=[384,1,1,1] resident tensor -- exactly the "[384,1] single query token" shape
    // needed below, used directly with no reshape (same convention as every other
    // resident weight in this file: mul_mat/add operands reference wt()'s result as-is
    // unless a host-side transform is needed, per house style -- see conv weights below
    // needing tensor_to_vec vs. attention weights that don't).
    //
    // Q comes from probe (query), K/V come from x (key=value=this timestep's readout
    // token) -- two DIFFERENT inputs through the SAME combined in_proj_weight, so (unlike
    // the block-transformer's self-attention, where q/k/v all come from one input and can
    // share one mul_mat) each needs its own mul_mat against the full [384,1152] qkv
    // weight; the slice not needed from each (k/v from the probe pass, q from the x pass)
    // is simply left unused, same in_proj_weight both passes (matches
    // nn.MultiheadAttention where in_proj_weight's 3 row-blocks are always [Wq;Wk;Wv]
    // regardless of what query/key/value tensors get fed through it).
    ggml_tensor * qkv_probe = ggml_add(ctx, ggml_mul_mat(ctx, qkv_w_r, probe_r), qkv_b_r);
    ggml_tensor * q = ggml_cont(ctx, ggml_view_2d(ctx, qkv_probe, hidden, 1, qkv_probe->nb[1], 0));
    ggml_tensor * qkv_x = ggml_add(ctx, ggml_mul_mat(ctx, qkv_w_r, x), qkv_b_r);
    ggml_tensor * k = ggml_cont(ctx, ggml_view_2d(ctx, qkv_x, hidden, width, qkv_x->nb[1], (size_t) hidden * qkv_x->nb[0]));
    ggml_tensor * v = ggml_cont(ctx, ggml_view_2d(ctx, qkv_x, hidden, width, qkv_x->nb[1], (size_t) 2 * hidden * qkv_x->nb[0]));

    // Split into heads with heads on ne2 (a batch axis mul_mat loops over, never
    // cross-multiplied) and width on ne3 (a SECOND batch axis) -- this is what keeps each
    // timestep's attention independent (no cross-timestep mixing) while still doing
    // per-head dot products correctly. Qh's ne3=1 broadcasts into Kh/Vh's ne3=width per
    // ggml_can_mul_mat (t1->ne3 % t0->ne3 == 0, t0=the smaller/A operand).
    ggml_tensor * Qh = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, q, map_head_dim, map_heads, 1, 1), 0, 2, 1, 3));
    ggml_tensor * Kh = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, k, map_head_dim, map_heads, 1, width), 0, 2, 1, 3));
    ggml_tensor * Vh = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, v, map_head_dim, map_heads, 1, width), 1, 2, 0, 3));

    ggml_tensor * scores = ggml_mul_mat(ctx, Qh, Kh);  // [1(seq_q),1(seq_k),heads,width]
    ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
    // seq_k==1 always (readouts.action==1 token/timestep) so this softmax is over a
    // single logit -- always exactly 1.0 -- computed via the real op (not hand-simplified
    // to "skip attention") so this stays correct if that ever changes, and so TIP-06 can
    // dump a genuine attn-probability boundary.
    ggml_tensor * probs = ggml_soft_max_ext(ctx, scores, nullptr, 1.0f / std::sqrt((float) map_head_dim), 0.0f);
    ggml_tensor * attended = ggml_mul_mat(ctx, Vh, probs);  // [head_dim,1,heads,width]
    ggml_tensor * merged = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, attended, 0, 2, 1, 3)), hidden, width);

    ggml_tensor * attn_out = ggml_add(ctx, ggml_mul_mat(ctx, o_w_r, merged), o_b_r);
    ggml_set_name(attn_out, "l1_head.map.attn_out");
    ggml_set_output(attn_out);

    ggml_tensor * y = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, attn_out, ln_eps), norm_w_r), norm_b_r);
    ggml_tensor * h = ggml_gelu_erf(ctx, ggml_add(ctx, ggml_mul_mat(ctx, ffn_up_w_r, y), ffn_up_b_r));
    h = ggml_add(ctx, ggml_mul_mat(ctx, ffn_down_w_r, h), ffn_down_b_r);
    // Residual is onto attn_out (pre-LN), not onto y -- per TIP-05 spec, matches
    // MlpBlockPt's usage inside MAPHeadPt (out = attn_out + MlpBlock(LayerNorm(attn_out))).
    ggml_tensor * emb = ggml_add(ctx, attn_out, h);
    ggml_set_name(emb, "l1_head.map.emb");
    ggml_set_output(emb);

    ggml_tensor * mean_raw = ggml_add(ctx, ggml_mul_mat(ctx, mean_w_r, emb), mean_b_r);
    ggml_tensor * mean_final = ggml_scale(ctx, ggml_tanh(ctx, ggml_scale(ctx, mean_raw, 1.0f / max_action)), max_action);
    ggml_set_name(mean_final, "l1_head.mean_normalized");
    ggml_set_output(mean_final);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 512, false);
    ggml_build_forward_expand(graph, attn_out);
    ggml_build_forward_expand(graph, emb);
    ggml_build_forward_expand(graph, mean_final);
    ggml_gallocr_t gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!gallocr || !ggml_gallocr_alloc_graph(gallocr, graph)) {
        std::fprintf(stderr, "vla(octo): L1 head ggml_gallocr_alloc_graph failed\n");
        if (gallocr) ggml_gallocr_free(gallocr);
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_set(x, readout_action.data(), 0, ggml_nbytes(x));
    const ggml_status st = ggml_backend_graph_compute(backend, graph);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "vla(octo): L1 head ggml_backend_graph_compute failed (%d)\n", (int) st);
        ggml_gallocr_free(gallocr);
        ggml_free(ctx);
        return false;
    }

    result.map_attn_out.resize((size_t) hidden * width);
    ggml_backend_tensor_get(attn_out, result.map_attn_out.data(), 0, ggml_nbytes(attn_out));
    result.map_emb.resize((size_t) hidden * width);
    ggml_backend_tensor_get(emb, result.map_emb.data(), 0, ggml_nbytes(emb));
    result.mean_normalized.resize((size_t) action_total * width);
    if ((size_t) ggml_nelements(mean_final) != result.mean_normalized.size()) {
        std::fprintf(stderr, "vla(octo): L1 head mean_proj out-dim=%lld does not match action_horizon*action_dim=%d\n",
                     (long long) mean_final->ne[0], action_total);
        ggml_gallocr_free(gallocr);
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_get(mean_final, result.mean_normalized.data(), 0, ggml_nbytes(mean_final));

    // predict_action: last window timestep only (matches diffusion's own final-action-slice
    // convention -- see run_diffusion_resident's "final action slice" comment/the
    // octo_action_slice tripwire test).
    result.final_actions.assign(result.mean_normalized.end() - action_total, result.mean_normalized.end());

    ggml_gallocr_free(gallocr);
    ggml_free(ctx);
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

// TIP-05V: octo.dataset_statistics comes in two shapes depending on how many datasets the
// checkpoint's OctoModelPt.dataset_statistics covers -- nested-by-dataset-key (bridge/libero:
// {"bridge_dataset": {"action":..., "proprio":...}, "<other dataset>": {...}, ...}, needing
// resolve_unnorm_dataset_key to pick one) vs. FLAT single-dataset (the real
// octo-aloha-jitter2525.gguf checkpoint: {"action":..., "proprio":..., "num_transitions":...,
// "num_trajectories":...} directly, no dataset-name wrapper at all -- confirmed by inspecting
// the GGUF's raw KV bytes; resolve_unnorm_dataset_key previously misread this flat object's 4
// members as 4 candidate dataset NAMES and failed loudly, since none of them is a
// single-key/bridge_dataset/env-var match). Detected by whether the object itself already has
// an "action" member shaped like a stats block (has "mean") -- real dataset names never
// collide with that. Every existing (nested) GGUF this codebase ships/tests against is
// unaffected: their top level never has a member literally named "action".
static bool resolve_stats_block(const nlohmann::json& j, const std::string& dataset_key_in,
                                const nlohmann::json** out) {
    if (j.is_object() && j.contains("action") && j["action"].is_object() && j["action"].contains("mean")) {
        *out = &j;
        return true;
    }
    std::string dataset_key = dataset_key_in;
    if (dataset_key.empty() && !resolve_unnorm_dataset_key(j, dataset_key)) return false;
    if (!j.contains(dataset_key)) {
        std::fprintf(stderr, "vla(octo): dataset_statistics missing key %s\n", dataset_key.c_str());
        return false;
    }
    *out = &j[dataset_key];
    return true;
}

// TIP-05 Part A step 1: proprio z-score stats -- octo.dataset_statistics[dataset_key].proprio
// (NOT .action; a sibling stat block in the same per-dataset JSON object). Convention: the
// caller (harness/TIP-07) sends RAW proprio (RLDS state, original units) via Inputs::state;
// this engine normalizes it here, once, before the proprio tokenizer runs -- mirrors
// unnormalize_action's dataset-key resolution (same resolve_stats_block, so both stat blocks
// always come from the same resolved dataset), but is otherwise a separate, self-contained
// JSON read (no mask field -- TIP-05 doesn't mention one for proprio, unlike action's
// gripper-dim mask).
static bool load_proprio_stats(gguf_reader& g, const std::string& dataset_key_in,
                               std::vector<float>& mean, std::vector<float>& stdv) {
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
    const nlohmann::json* block = nullptr;
    if (!resolve_stats_block(j, dataset_key_in, &block)) return false;
    if (!block->contains("proprio")) {
        std::fprintf(stderr, "vla(octo): dataset_statistics stats block missing .proprio\n");
        return false;
    }
    const auto& p = (*block)["proprio"];
    mean = p.at("mean").get<std::vector<float>>();
    stdv = p.at("std").get<std::vector<float>>();
    if (mean.size() != (size_t) kProprioTokens || stdv.size() != (size_t) kProprioTokens) {
        std::fprintf(stderr, "vla(octo): unexpected dataset_statistics/proprio shape\n");
        return false;
    }
    return true;
}

// Un-normalizes a [horizon,7] flattened action (horizon = normalized_flat.size()/7 --
// action_dim is fixed at 7 across every head type, only action_horizon varies: 4 for
// diffusion libero, 20 for the L1 aloha jitter-adapted head) using octo.dataset_statistics's
// per-dataset action mean/std/mask: unnorm[d] = mask[d] ? norm[d]*std[d]+mean[d]
// : norm[d] (dims with mask=false, e.g. bridge_dataset's/libero_object's gripper dim 6,
// are left as-is -- confirmed against golden: final_action_unnormalized[...,6] ==
// sample_actions.final_action_normalized[...,6] exactly, while masked-in dims match
// a*std+mean to ~1e-5). dataset_key_in must match golden's metadata.unnormalization.dataset
// when comparing against a golden trace (all shipped bridge golden cases use
// "bridge_dataset"); pass "" to auto-resolve via resolve_stats_block (live/serving paths,
// where there's no golden to match against; also transparently handles a flat/single-dataset
// stats blob with no dataset-name wrapper, e.g. octo-aloha-jitter2525.gguf -- TIP-05V).
static bool unnormalize_action(gguf_reader& g, const std::string& dataset_key_in,
                               const std::vector<float>& normalized_flat, std::vector<float>& unnorm_flat) {
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
    const nlohmann::json* block = nullptr;
    if (!resolve_stats_block(j, dataset_key_in, &block)) return false;
    if (!block->contains("action")) {
        std::fprintf(stderr, "vla(octo): dataset_statistics stats block missing .action\n");
        return false;
    }
    const auto& act = (*block)["action"];
    std::vector<float> mean = act.at("mean").get<std::vector<float>>();
    std::vector<float> stdv = act.at("std").get<std::vector<float>>();
    std::vector<bool> mask = act.at("mask").get<std::vector<bool>>();
    const size_t dim = mask.size();
    if (mean.size() != dim || stdv.size() != dim || dim != 7 ||
        normalized_flat.empty() || normalized_flat.size() % dim != 0) {
        std::fprintf(stderr, "vla(octo): unexpected dataset_statistics/action shape\n");
        return false;
    }
    const size_t horizon = normalized_flat.size() / dim;
    unnorm_flat.resize(normalized_flat.size());
    for (size_t t = 0; t < horizon; ++t) {
        for (size_t d = 0; d < dim; ++d) {
            const float norm = normalized_flat[t * dim + d];
            unnorm_flat[t * dim + d] = mask[d] ? (norm * stdv[d] + mean[d]) : norm;
        }
    }
    return true;
}

// TIP-05: n_proprio_tokens=0 (the default every pre-TIP-05 call site passes) reproduces the
// old metadata/key_valid sequence byte-for-byte. n_proprio_tokens=7 inserts a proprio OBS
// group right after wrist (same position assemble_transformer_input uses -- the two must
// stay in lock-step, both keyed off the same n_proprio_tokens). Proprio validity is gated
// solely by timestep_valid (CAUSAL, same rule as primary/wrist/readout): unlike a camera,
// proprioception has no "was this sensor even connected" ambiguity to model with a separate
// per-token pad mask, so there's no proprio_valid input here (matches TIP-05 Part A step 4:
// "mask = timestep_pad_mask").
static bool build_transformer_mask(const NpyBool& task_valid,
                                   const NpyBool& primary_valid,
                                   const NpyBool& wrist_valid,
                                   const NpyBool& timestep_valid,
                                   int window_size,
                                   int n_proprio_tokens,
                                   std::vector<uint8_t>& blocked,
                                   std::vector<float>& blocked_f32) {
    const int seq = octo_seq_len(window_size, n_proprio_tokens);
    constexpr int heads = 6;
    const int step_tokens = octo_step_tokens(n_proprio_tokens);
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
        for (int i = 0; i < n_proprio_tokens; ++i) {
            metadata.push_back({OctoTokenKind::OBS, t});
            key_valid.push_back(timestep_ok);
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

// TIP-ND1-B: sole block-transformer implementation (was run_transformer_graph /
// run_transformer_graph_resident, unified). All 12 blocks' attn/ffn weights + the final
// output_norm are referenced directly from the resident context; `input`/`blocked_mask` (this
// call's assembled sequence + mask) are genuinely new per-call data and still get uploaded
// fresh. Graph topology and op order are unchanged.
static bool run_transformer_graph_resident(ggml_context * ctx_w, ggml_backend_t backend,
                                           const std::vector<float>& input,
                                           const std::vector<float>& blocked_mask,
                                           int window_size,
                                           int n_proprio_tokens,
                                           OctoTransformerResult& result) {
    constexpr int hidden = 384;
    constexpr int heads = 6;
    constexpr int head_dim = 64;
    const int seq = octo_seq_len(window_size, n_proprio_tokens);
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
    const int step_tokens = octo_step_tokens(n_proprio_tokens);
    // TIP-05: proprio (when present) sits right after wrist, before repeated-language --
    // same offsets assemble_transformer_input used to place it. n_proprio_tokens=0
    // collapses off_language/off_readout back to the original 336/352 constants.
    constexpr size_t off_primary = 16;
    constexpr size_t off_wrist = off_primary + 256;
    constexpr size_t off_proprio = off_wrist + 64;
    const size_t off_language = off_proprio + (size_t) n_proprio_tokens;
    const size_t off_readout = off_language + 16;
    ggml_tensor * split_task = ggml_cont(ctx, ggml_view_2d(ctx, output, hidden, 16, output->nb[1], 0));
    ggml_tensor * split_primary = ggml_cont(ctx, ggml_view_3d(ctx, output, hidden, 256, window_size, output->nb[1], (size_t) step_tokens * output->nb[1], off_primary * output->nb[1]));
    ggml_tensor * split_wrist = ggml_cont(ctx, ggml_view_3d(ctx, output, hidden, 64, window_size, output->nb[1], (size_t) step_tokens * output->nb[1], off_wrist * output->nb[1]));
    ggml_tensor * split_proprio = nullptr;
    if (n_proprio_tokens > 0) {
        split_proprio = ggml_cont(ctx, ggml_view_3d(ctx, output, hidden, n_proprio_tokens, window_size, output->nb[1], (size_t) step_tokens * output->nb[1], off_proprio * output->nb[1]));
    }
    ggml_tensor * split_language = ggml_cont(ctx, ggml_view_3d(ctx, output, hidden, 16, window_size, output->nb[1], (size_t) step_tokens * output->nb[1], off_language * output->nb[1]));
    ggml_tensor * split_readout = ggml_cont(ctx, ggml_view_3d(ctx, output, hidden, 1, window_size, output->nb[1], (size_t) step_tokens * output->nb[1], off_readout * output->nb[1]));
    for (ggml_tensor * t : {split_task, split_primary, split_wrist, split_language, split_readout}) ggml_set_output(t);
    if (split_proprio) ggml_set_output(split_proprio);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 8192, false);
    ggml_build_forward_expand(graph, split_task);
    ggml_build_forward_expand(graph, split_primary);
    ggml_build_forward_expand(graph, split_wrist);
    if (split_proprio) ggml_build_forward_expand(graph, split_proprio);
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
    if (split_proprio) get(split_proprio, result.obs_proprio);
    else result.obs_proprio.clear();
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

    m->backend = octo_select_backend(m->n_threads, "");
    if (!m->backend) return nullptr;

    // TIP-BUILD-OCTO-GPU-B: predict()'s 5 stages now consume these same weights (resident on
    // `m->backend` -- CUDA when available, CPU otherwise) directly; no separate CPU-only copy.
    if (!load_all_tensors(*m, m->io)) return nullptr;
    std::printf("vla(octo): loaded %lld F32 tensors, hidden=%lld blocks=%lld heads=%lld horizon=%lld "
                "action_dim=%lld head_type=%s window_size=%lld has_proprio=%s proprio_in_dim=%lld "
                "proprio_tokens=%d\n",
                (long long) m->tensors.size(), (long long) m->hidden, (long long) m->blocks,
                (long long) m->heads, (long long) m->action_horizon, (long long) m->action_dim,
                m->head_type.c_str(), (long long) m->window_size, m->has_proprio ? "true" : "false",
                (long long) m->proprio_in_dim, m->has_proprio ? kProprioTokens : 0);
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

// TIP-ND1-B: sole dump implementation (was octo_dump_tokenizer_case / _resident, unified).
// Dumps every M0-M8 boundary needed by verify_octo_parity.py, computed via the *_resident
// stage functions (on `backend`, CUDA when available) using the SAME weight residency setup
// octo_create() builds for the live server (load_all_tensors onto a backend selected the
// identical way) -- this exercises the literal code OctoModelArch::predict() runs, not a
// separate parallel implementation.
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
    m.action_horizon = g.u32("octo.action.horizon");
    m.action_dim = g.u32("octo.action.dim");
    const int action_total = (int) (m.action_horizon * m.action_dim);
    m.backend = octo_select_backend(default_cpu_threads(), "[dump] ");
    if (!m.backend) return false;
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
    if (!assemble_transformer_input(lang_pos, primary_pos, wrist_pos, {}, repeated, readout_pos, window_size, /*n_proprio_tokens=*/0, bt.input)) return false;
    std::vector<float> blocked_mask;
    if (!build_transformer_mask(task_valid, primary_valid, wrist_valid, timestep_valid, window_size, /*n_proprio_tokens=*/0, bt.blocked_mask, blocked_mask)) return false;
    if (!write_f32_dump(dump_dir, "bt.input", bt.input, {1, seq, 384}, mf)) return false;
    if (!write_bool_dump(dump_dir, "bt.mask", bt.blocked_mask, {6, seq, seq}, mf)) return false;
    if (!run_transformer_graph_resident(ctx_w, backend, bt.input, blocked_mask, window_size, /*n_proprio_tokens=*/0, bt)) return false;
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
    const OctoNoiseSource replay_noise{OctoNoiseSourceKind::REPLAY, case_dir, nullptr};
    if (!run_diffusion_resident(ctx_w, backend, replay_noise, bt.readout_action, window_size, action_total, diff)) return false;
    if (!write_f32_dump(dump_dir, "diff.initial_noise", diff.initial_noise, {1, window_size, action_total}, mf)) return false;
    if (!write_bool_dump(dump_dir, "diff.action_mask", diff.action_mask, {1, window_size, 4, 7}, mf)) return false;
    if (!write_bool_dump(dump_dir, "diff.flat_action_mask", diff.flat_action_mask, {1, window_size, action_total}, mf)) return false;
    for (int step = 0; step < 20; ++step) {
        char boundary[64];
        std::snprintf(boundary, sizeof(boundary), "diff.step%02d.current_x_before", step);
        if (!write_f32_dump(dump_dir, boundary, diff.current_x_before[(size_t) step], {1, window_size, action_total}, mf)) return false;
        std::snprintf(boundary, sizeof(boundary), "diff.step%02d.pred_eps", step);
        if (!write_f32_dump(dump_dir, boundary, diff.pred_eps[(size_t) step], {1, window_size, action_total}, mf)) return false;
        std::snprintf(boundary, sizeof(boundary), "diff.step%02d.z", step);
        if (!write_f32_dump(dump_dir, boundary, diff.z[(size_t) step], {1, window_size, action_total}, mf)) return false;
        std::snprintf(boundary, sizeof(boundary), "diff.step%02d.x_after_denoise", step);
        if (!write_f32_dump(dump_dir, boundary, diff.after_denoise[(size_t) step], {1, window_size, action_total}, mf)) return false;
        std::snprintf(boundary, sizeof(boundary), "diff.step%02d.x_after_noise_add", step);
        if (!write_f32_dump(dump_dir, boundary, diff.after_noise_add[(size_t) step], {1, window_size, action_total}, mf)) return false;
        std::snprintf(boundary, sizeof(boundary), "diff.step%02d.x_after_clip", step);
        if (!write_f32_dump(dump_dir, boundary, diff.after_clip[(size_t) step], {1, window_size, action_total}, mf)) return false;
        std::snprintf(boundary, sizeof(boundary), "diff.step%02d.x_after_mask", step);
        if (!write_f32_dump(dump_dir, boundary, diff.after_mask[(size_t) step], {1, window_size, action_total}, mf)) return false;
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

// TIP-ND1-B: sole pipeline implementation (was octo_run_pipeline / _resident, unified). Shared
// tail of the live/cold-start prediction path: given already-assembled per-view
// observation+task images (obs.shape={1,window_size,3,side,side}, task.shape={1,3,side,side})
// and already-tokenized language, runs obs tokenizer x2 -> T5 encoder -> language ->
// assemble -> causal mask -> block transformer -> diffusion head -> unnormalize, entirely via
// the resident stage functions on `backend` (the model's real backend: CUDA when available,
// CPU otherwise), referencing weights resident on `ctx_w` (populated once by load_all_tensors).
// `io` is the caller's already-open gguf_reader -- unnormalize_action only reads the small
// in-memory "octo.dataset_statistics" metadata string via it (metadata is parsed once at
// gguf_reader::open() time, so this is not a disk touch as long as `io` is already open). Used
// by both octo_predict_from_images (CLI: tokenizes its own instruction via SentencePiece) and
// OctoModelArch::predict (server: receives already-tokenized Inputs::lang_tokens from the
// client, matching every other arch's client-tokenizes/server-embeds convention).
// wrist_real=false marks the wrist observation as a zero-fill placeholder (single-camera
// checkpoint or a caller that supplied no wrist view) -- its causal-mask key_valid entries
// are forced false so the transformer treats it as absent padding, not a real observation.
//
// TIP-05: has_proprio/proprio_in_dim/proprio_state_raw are additive -- has_proprio=false
// (every pre-TIP-05 checkpoint) skips Part A entirely (n_proprio=0, byte-identical sequence
// geometry to before). head_type routes Part B: "l1" runs the L1 action head instead of
// diffusion (bypassing the whole noise/denoising-loop pipeline per TIP-05's routing spec);
// "diffusion" keeps the untouched pre-TIP-05 path. The two are independent switches (a
// checkpoint could in principle have proprio without an L1 head or vice versa; every actual
// checkpoint pairs them, but nothing here assumes that).
static bool octo_run_pipeline_resident(ggml_context * ctx_w, ggml_backend_t backend, gguf_reader& io, int window_size,
                                       int action_total, const std::string& head_type,
                                       bool has_proprio, int proprio_in_dim,
                                       const NpyU8& primary_obs, const NpyU8& primary_task,
                                       const NpyU8& wrist_obs, const NpyU8& wrist_task, bool wrist_real,
                                       const std::vector<int32_t>& input_ids,
                                       const std::vector<int32_t>& attention_mask,
                                       const std::vector<float>& proprio_state_raw,
                                       const std::string& unnorm_dataset,
                                       std::vector<float>& normalized_out,
                                       std::vector<float>& unnormalized_out,
                                       float& ms_vision_out,
                                       float& ms_inference_out) {
    using clock = std::chrono::steady_clock;

    if (head_type != "diffusion" && head_type != "l1") {
        std::fprintf(stderr, "vla(octo): head_type=%s forward not implemented\n", head_type.c_str());
        return false;
    }
    const int n_proprio = has_proprio ? kProprioTokens : 0;

    const auto t_vision0 = clock::now();
    std::vector<float> tok, primary_proj, primary_pos, wrist_proj, wrist_pos;
    if (!run_one_obs_tokenizer_graph_resident(ctx_w, backend, "primary", primary_obs, primary_task, 256, 256, window_size, tok, primary_proj, primary_pos)) return false;
    if (!run_one_obs_tokenizer_graph_resident(ctx_w, backend, "wrist", wrist_obs, wrist_task, 128, 64, window_size, tok, wrist_proj, wrist_pos)) return false;
    std::vector<float> proprio_pos;
    if (has_proprio) {
        std::vector<float> proprio_mean, proprio_std;
        if (!load_proprio_stats(io, unnorm_dataset, proprio_mean, proprio_std)) return false;
        std::vector<float> proprio_norm((size_t) window_size * kProprioTokens);
        for (int t = 0; t < window_size; ++t) {
            for (int d = 0; d < kProprioTokens; ++d) {
                const float raw = (d < (int) proprio_state_raw.size()) ? proprio_state_raw[(size_t) d] : 0.0f;
                proprio_norm[(size_t) t * kProprioTokens + d] = (raw - proprio_mean[(size_t) d]) / proprio_std[(size_t) d];
            }
        }
        if (!run_proprio_tokenizer_graph_resident(ctx_w, backend, proprio_norm, proprio_in_dim, window_size, proprio_pos)) return false;
    }
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
    if (!assemble_transformer_input(lang_pos, primary_pos, wrist_pos, proprio_pos, repeated, readout_pos, window_size, n_proprio, bt.input)) return false;
    std::vector<float> blocked_mask;
    if (!build_transformer_mask(task_valid, primary_valid, wrist_valid, timestep_valid, window_size, n_proprio, bt.blocked_mask, blocked_mask)) return false;
    if (!run_transformer_graph_resident(ctx_w, backend, bt.input, blocked_mask, window_size, n_proprio, bt)) return false;

    if (head_type == "l1") {
        OctoL1HeadResult l1;
        if (!run_l1_action_head_graph_resident(ctx_w, backend, bt.readout_action, window_size, action_total, l1)) return false;
        normalized_out = std::move(l1.final_actions);
    } else {
        std::random_device rd;
        std::mt19937 rng(rd());
        OctoDiffusionResult diff;
        const OctoNoiseSource live_noise{OctoNoiseSourceKind::RANDOM, "", &rng};
        if (!run_diffusion_resident(ctx_w, backend, live_noise, bt.readout_action, window_size, action_total, diff)) return false;
        normalized_out = std::move(diff.final_actions);
    }
    ms_inference_out = std::chrono::duration<float, std::milli>(clock::now() - t_inference0).count();

    return unnormalize_action(io, unnorm_dataset, normalized_out, unnormalized_out);
}

// Cold start: only ONE live frame is available (a single vla-cli snapshot or server
// request), but the model expects `window_size` observation timesteps. Repeat the live
// frame into every slot; the caller's timestep_valid marks all-but-the-last as padding
// (see octo_run_pipeline_resident) so the causal mask treats only the last slot as "real" -- matches
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

    // TIP-ND1-B: CLI one-shot resident load -- same backend-selection + load_all_tensors
    // sequence octo_create()/octo_dump_tokenizer_case_resident() use, scoped to this single
    // call (no persistent OctoModelArch survives past this function, unlike the live server).
    OctoModelArch m;
    m.matmul_type = GGML_TYPE_F32;
    m.action_horizon = g.u32("octo.action.horizon");
    m.action_dim = g.u32("octo.action.dim");
    m.head_type = g.has("octo.action.head_type") ? g.str("octo.action.head_type") : "diffusion";
    detect_proprio(g, m.has_proprio, m.proprio_in_dim);
    m.backend = octo_select_backend(default_cpu_threads(), "[cli] ");
    if (!m.backend) return false;
    if (!load_all_tensors(m, g)) return false;

    std::vector<int32_t> input_ids, attention_mask;
    if (!octo_tokenize_text(ckpt_path, instruction, input_ids, attention_mask)) return false;

    NpyU8 primary_obs, primary_task, wrist_obs, wrist_task;
    octo_build_cold_start_obs_task(primary_rgb, primary_w, primary_h, 256, window_size, primary_obs, primary_task);
    octo_build_cold_start_obs_task(wrist_rgb, wrist_w, wrist_h, 128, window_size, wrist_obs, wrist_task);

    // TIP-05: octo_predict_from_images (the images-only CLI entry point, octo.h) has no
    // proprio parameter -- an L1/proprio checkpoint driven from the CLI gets an all-zero
    // proprio reading (z-scored to (0-mean)/std, NOT world-unit zero) rather than a real
    // robot state. Fine for the CLI's smoke-test purpose; the server path (predict() below)
    // is what actually threads Inputs::state through.
    const std::vector<float> proprio_state_raw;

    float ms_vision = 0.f, ms_inference = 0.f;
    const int action_total = (int) (m.action_horizon * m.action_dim);
    return octo_run_pipeline_resident(m.ctx_weights, m.backend, g, window_size, action_total, m.head_type,
                                      m.has_proprio, (int) m.proprio_in_dim,
                                      primary_obs, primary_task, wrist_obs, wrist_task,
                                      /*wrist_real=*/true, input_ids, attention_mask, proprio_state_raw, unnorm_dataset,
                                      out.normalized, out.unnormalized, ms_vision, ms_inference);
}

// TIP-06: sole L1/proprio stagewise dump implementation. Mirrors
// octo_dump_tokenizer_case_resident's structure/boilerplate (backend selection,
// load_all_tensors, manifest handling) but drives the head_type=l1 resident stage functions
// (TIP-05) instead of the diffusion ones, and reads its input from a case_dir written by the
// Part A python dump script rather than a JAX/OctoPt golden trace directory.
bool octo_dump_l1_stagewise_case_resident(const std::string& ckpt_path,
                                          const std::string& case_dir,
                                          const std::string& dump_dir,
                                          const std::string& unnorm_dataset) {
    gguf_reader g{"octo"};
    if (!g.open(ckpt_path)) return false;
    const int window_size = (int) g.u32("octo.window_size");
    if (window_size < 1 || window_size > kMaxHorizon) {
        std::fprintf(stderr, "vla(octo): octo.window_size=%d out of supported range [1, %d]\n",
                     window_size, kMaxHorizon);
        return false;
    }

    OctoModelArch m;
    m.matmul_type = GGML_TYPE_F32;
    m.action_horizon = g.u32("octo.action.horizon");
    m.action_dim = g.u32("octo.action.dim");
    m.head_type = g.has("octo.action.head_type") ? g.str("octo.action.head_type") : "diffusion";
    detect_proprio(g, m.has_proprio, m.proprio_in_dim);
    if (m.head_type != "l1" || !m.has_proprio) {
        std::fprintf(stderr, "vla(octo): octo_dump_l1_stagewise_case_resident requires head_type=l1 "
                              "and a proprio tokenizer (got head_type=%s has_proprio=%s)\n",
                     m.head_type.c_str(), m.has_proprio ? "true" : "false");
        return false;
    }
    const int action_total = (int) (m.action_horizon * m.action_dim);
    m.backend = octo_select_backend(default_cpu_threads(), "[l1-dump] ");
    if (!m.backend) return false;
    if (!load_all_tensors(m, g)) return false;
    ggml_context * ctx_w = m.ctx_weights;
    ggml_backend_t backend = m.backend;

    NpyU8 top_hwc, wrist_hwc;
    if (!parse_npy_u8(case_dir + "/input.top_hwc_u8.npy", top_hwc)) return false;
    if (!parse_npy_u8(case_dir + "/input.wrist_hwc_u8.npy", wrist_hwc)) return false;
    if (top_hwc.shape != std::vector<int64_t>{256, 256, 3} || wrist_hwc.shape != std::vector<int64_t>{128, 128, 3}) {
        std::fprintf(stderr, "vla(octo): unexpected l1-dump input image shape\n");
        return false;
    }
    NpyF32 proprio_raw_npy;
    if (!parse_npy_f32(case_dir + "/input.proprio_raw.npy", proprio_raw_npy) ||
        proprio_raw_npy.data.size() != (size_t) kProprioTokens) {
        std::fprintf(stderr, "vla(octo): unexpected l1-dump input.proprio_raw.npy shape\n");
        return false;
    }
    std::ifstream instr_f(case_dir + "/input.instruction.txt");
    if (!instr_f) {
        std::fprintf(stderr, "vla(octo): cannot open %s/input.instruction.txt\n", case_dir.c_str());
        return false;
    }
    const std::string instruction((std::istreambuf_iterator<char>(instr_f)), std::istreambuf_iterator<char>());

    std::vector<int32_t> input_ids, attention_mask;
    if (!octo_tokenize_text(ckpt_path, instruction, input_ids, attention_mask)) return false;

    NpyU8 primary_obs, primary_task, wrist_obs, wrist_task;
    octo_build_cold_start_obs_task(top_hwc.data.data(), 256, 256, 256, window_size, primary_obs, primary_task);
    octo_build_cold_start_obs_task(wrist_hwc.data.data(), 128, 128, 128, window_size, wrist_obs, wrist_task);

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

    // T0: proprio z-score normalize (same formula/stats source as octo_run_pipeline_resident)
    // + proprio tokenizer forward (proj + pos_embd, PRE-transformer).
    std::vector<float> proprio_mean, proprio_std;
    if (!load_proprio_stats(g, unnorm_dataset, proprio_mean, proprio_std)) return false;
    std::vector<float> proprio_norm((size_t) kProprioTokens);
    for (int d = 0; d < kProprioTokens; ++d) {
        proprio_norm[(size_t) d] = (proprio_raw_npy.data[(size_t) d] - proprio_mean[(size_t) d]) / proprio_std[(size_t) d];
    }
    if (!write_f32_dump(dump_dir, "t0.proprio_normalized", proprio_norm, {kProprioTokens}, mf)) return false;

    std::vector<float> proprio_tokens;
    if (!run_proprio_tokenizer_graph_resident(ctx_w, backend, proprio_norm, (int) m.proprio_in_dim, window_size, proprio_tokens)) return false;
    if (!write_f32_dump(dump_dir, "t0.proprio_tokens", proprio_tokens, {1, window_size, kProprioTokens, 384}, mf)) return false;

    std::vector<float> tok, primary_proj, primary_pos, wrist_proj, wrist_pos;
    if (!run_one_obs_tokenizer_graph_resident(ctx_w, backend, "primary", primary_obs, primary_task, 256, 256, window_size, tok, primary_proj, primary_pos)) return false;
    if (!run_one_obs_tokenizer_graph_resident(ctx_w, backend, "wrist", wrist_obs, wrist_task, 128, 64, window_size, tok, wrist_proj, wrist_pos)) return false;

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
    wrist_valid.data.assign((size_t) window_size, 1);  // real wrist image, always valid
    timestep_valid.shape = {1, window_size};
    timestep_valid.data.assign((size_t) window_size, 0);
    timestep_valid.data.back() = 1;  // cold start: only the last slot is the live frame

    const int n_proprio = kProprioTokens;
    OctoTransformerResult bt;
    if (!assemble_transformer_input(lang_pos, primary_pos, wrist_pos, proprio_tokens, repeated, readout_pos, window_size, n_proprio, bt.input)) return false;
    std::vector<float> blocked_mask;
    if (!build_transformer_mask(task_valid, primary_valid, wrist_valid, timestep_valid, window_size, n_proprio, bt.blocked_mask, blocked_mask)) return false;
    if (!run_transformer_graph_resident(ctx_w, backend, bt.input, blocked_mask, window_size, n_proprio, bt)) return false;

    // T1/T2: readout_action tokens -- transformer OUTPUT, also the L1 head's INPUT.
    if (!write_f32_dump(dump_dir, "t1t2.readout_action", bt.readout_action, {1, window_size, 1, 384}, mf)) return false;

    OctoL1HeadResult l1;
    if (!run_l1_action_head_graph_resident(ctx_w, backend, bt.readout_action, window_size, action_total, l1)) return false;
    // T3: MAPHead internals (post out_proj pre-LN/MLP; post residual+MLP).
    if (!write_f32_dump(dump_dir, "t3.map_attn_out", l1.map_attn_out, {1, window_size, 384}, mf)) return false;
    if (!write_f32_dump(dump_dir, "t3.map_emb", l1.map_emb, {1, window_size, 1, 384}, mf)) return false;
    // T4: mean_proj + tanh*max_action, ALL window timesteps.
    if (!write_f32_dump(dump_dir, "t4.mean_normalized", l1.mean_normalized,
                        {1, window_size, (int) m.action_horizon, (int) m.action_dim}, mf)) return false;

    // T5: final unnormalized action (last window timestep only).
    std::vector<float> unnorm_actions;
    if (!unnormalize_action(g, unnorm_dataset, l1.final_actions, unnorm_actions)) return false;
    if (!write_f32_dump(dump_dir, "t5.action_unnormalized", unnorm_actions,
                        {1, (int) m.action_horizon, (int) m.action_dim}, mf)) return false;
    return true;
}

// Server-facing entry point (vla-server, TIP-CLIENT): in.images[0] is the primary view,
// in.images[1] the wrist view if the client sent one (single-camera checkpoints/clients
// omit it -- zero-filled and masked invalid via octo_run_pipeline_resident's wrist_real=false, same
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

    // TIP-05: Inputs::state is RAW proprio (RLDS state, original units) per the doc comment
    // in model.h -- octo_run_pipeline_resident z-scores it internally via load_proprio_stats
    // (harness/TIP-07 must NOT pre-normalize, to avoid double-normalizing). Missing
    // Inputs::state on a proprio checkpoint zero-fills rather than hard-failing (matches
    // pi0.cpp's `in.state ? in.state[i] : 0.f` convention for the same field), but is
    // surfaced loudly since a silently-zeroed proprio reading will silently skew every
    // predicted action.
    std::vector<float> proprio_state_raw;
    if (has_proprio) {
        proprio_state_raw.assign((size_t) kProprioTokens, 0.0f);
        if (in.state) {
            for (int64_t d = 0; d < kProprioTokens; ++d) proprio_state_raw[(size_t) d] = in.state[d];
        } else {
            std::fprintf(stderr, "vla(octo): predict: proprio checkpoint but Inputs::state is null -- "
                                  "using all-zero proprio (z-scored, not world-unit zero)\n");
        }
    }

    std::vector<float> normalized, unnormalized;
    float ms_vision = 0.f, ms_inference = 0.f;
    const int action_total = (int) (action_horizon * action_dim);
    if (!octo_run_pipeline_resident(ctx_weights, backend, io, (int) window_size, action_total, head_type,
                                    has_proprio, (int) proprio_in_dim,
                                    primary_obs, primary_task, wrist_obs, wrist_task,
                                    wrist_real, input_ids, attention_mask, proprio_state_raw, /*unnorm_dataset=*/"",
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
    if (!build_transformer_mask(task_valid, primary_valid, wrist_valid, timestep_valid, window_size, /*n_proprio_tokens=*/0, blocked_bytes, blocked_mask)) return false;

    // TIP-ND1-B: weights resident, loaded once (same load_all_tensors() sequence every other
    // resident caller uses) -- matches how OctoPt holds a loaded model in memory across
    // repeated sample_actions() calls. Unlike the deleted disk-read path, the resident stage
    // functions only ever READ tensor bytes via wt()/tensor_to_vec() -- nothing is
    // moved/consumed out of ctx_w, so every one of the N passes below sees the identical,
    // uncorrupted weight set with no per-iteration "copy the master" step needed (that copy
    // existed only because the deleted run_t5_encoder_graph/run_transformer_graph MOVED their
    // weight-struct argument's vectors into the ggml payload, consuming them after one use --
    // that move-semantics hazard no longer exists once the weight source is a read-only
    // resident lookup).
    OctoModelArch m;
    m.matmul_type = GGML_TYPE_F32;
    m.action_horizon = g.u32("octo.action.horizon");
    m.action_dim = g.u32("octo.action.dim");
    m.backend = octo_select_backend(default_cpu_threads(), "[free-sample] ");
    if (!m.backend) return false;
    if (!load_all_tensors(m, g)) return false;
    ggml_context * ctx_w = m.ctx_weights;
    ggml_backend_t backend = m.backend;

    ggml_tensor * readout_pos_r = wt(ctx_w, "octo.readout.action.pos_embd");
    if (!readout_pos_r) return false;
    const std::vector<float> readout_pos = tensor_to_vec(readout_pos_r);

    const int action = (int) (m.action_horizon * m.action_dim);
    samples_out.assign((size_t) n_samples * (size_t) action, 0.0f);
    noise_out.assign((size_t) n_samples * (size_t) window_size * action, 0.0f);

    for (int i = 0; i < n_samples; ++i) {
        std::vector<float> tok, primary_proj, primary_pos, wrist_proj, wrist_pos;
        if (!run_one_obs_tokenizer_graph_resident(ctx_w, backend, "primary", primary_obs, primary_task, 256, 256, window_size, tok, primary_proj, primary_pos)) return false;
        if (!run_one_obs_tokenizer_graph_resident(ctx_w, backend, "wrist", wrist_obs, wrist_task, 128, 64, window_size, tok, wrist_proj, wrist_pos)) return false;

        std::vector<float> t5_out;
        if (!run_t5_encoder_graph_resident(ctx_w, backend, input_ids.data, attention_mask.data, t5_out)) return false;

        NpyF32 t5;
        t5.shape = {1, 16, 768};
        t5.data = t5_out;
        std::vector<float> lang_proj, lang_pos, repeated;
        if (!run_language_graph_resident(ctx_w, backend, t5, window_size, lang_proj, lang_pos, repeated)) return false;

        OctoTransformerResult bt;
        if (!assemble_transformer_input(lang_pos, primary_pos, wrist_pos, {}, repeated, readout_pos, window_size, /*n_proprio_tokens=*/0, bt.input)) return false;
        if (!run_transformer_graph_resident(ctx_w, backend, bt.input, blocked_mask, window_size, /*n_proprio_tokens=*/0, bt)) return false;

        std::mt19937 rng(seed + (uint32_t) i);
        OctoDiffusionResult diff;
        const OctoNoiseSource sample_noise{OctoNoiseSourceKind::RANDOM, "", &rng};
        if (!run_diffusion_resident(ctx_w, backend, sample_noise, bt.readout_action, window_size, action, diff)) return false;
        std::copy(diff.final_actions.begin(), diff.final_actions.end(), samples_out.begin() + (size_t) i * (size_t) action);
        std::copy(diff.initial_noise.begin(), diff.initial_noise.end(), noise_out.begin() + (size_t) i * (size_t) window_size * action);
    }
    return true;
}

}  // namespace vla
