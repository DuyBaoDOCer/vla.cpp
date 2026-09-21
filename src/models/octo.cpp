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
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace vla {
namespace {

// task_language prefix tokens (once, not repeated per timestep).
constexpr int kTaskTokens = 16;
// Shared max-horizon slab size of the obs/task pos-embedding tables written by
// scripts/convert_octo_to_gguf.py (same for every checkpoint regardless of the
// window_size actually trained/used). window_size must fit within it.
constexpr int kMaxHorizon = 10;
// LowdimObsTokenizerPt(obs_keys=["proprio"]) always emits exactly 1 token per state
// dimension (discretize=False or True doesn't change the token *count*, only how each
// dimension is encoded before the shared Linear projection) -- octo-small-1.5's proprio
// is 7-dim, so this is fixed regardless of checkpoint.
constexpr int kProprioTokens = 7;

// Every forward stage below builds a throwaway graph. Both halves of that are expensive to
// redo per frame: ggml_init() malloc()s its whole mem_size even with no_alloc (the graph and
// tensor metadata are the only things that live there), and ggml_gallocr_new()/_free()
// reallocates the backend compute buffer instead of reusing one sized for the same graph.
// OctoRuntime keeps both alive for the model's lifetime: one host arena the stages take turns
// borrowing (they never nest), and one allocator per stage so each keeps its own steady-state
// buffer.
enum class OctoStage { OBS_PRIMARY, OBS_WRIST, PROPRIO, T5, LANGUAGE, TRANSFORMER, SCORE_ACTOR, L1_HEAD, COUNT };

struct OctoRuntime {
    ggml_backend_t backend = nullptr;   ///< Not owned.
    ggml_context * ctx_w = nullptr;     ///< Resident weights, not owned.
    std::vector<uint8_t> arena;
    std::array<ggml_gallocr_t, (size_t) OctoStage::COUNT> allocs{};
    std::unordered_map<std::string, ggml_tensor *> by_name;

    // The T5 encoder and the projection after it depend only on the instruction's token ids
    // and attention mask, which stay fixed for as long as the robot is pursuing one task --
    // so across a rollout this is the same ~2 ms of work repeated on every frame. Cache the
    // result, keyed on the tokens so a task switch still recomputes.
    std::vector<int32_t> lang_key;
    std::vector<float> lang_pos;       ///< [16,384] projected task tokens.
    std::vector<float> lang_repeated;  ///< [steps,16,384] repeated into each timestep.
    int lang_steps = -1;

    // octo.dataset_statistics is a JSON blob in the GGUF metadata (~25 datasets wide for the
    // bridge pretrain checkpoint). Parsing it per request, twice, to read 21 floats is pure
    // overhead -- and it defers a malformed-metadata failure to inference time. Resolved once,
    // on the first call that needs it, and keyed on the dataset name in case a caller asks
    // for a different block.
    struct ActionStats {
        std::vector<float> mean, stdv;
        std::vector<uint8_t> mask;
    };
    struct ProprioStats {
        std::vector<float> mean, stdv;
    };
    std::string stats_key;
    bool stats_loaded = false;
    ActionStats action_stats;
    ProprioStats proprio_stats;
    bool has_proprio_stats = false;

    // Graph metadata only (no_alloc), so this is far more than any stage needs.
    static constexpr size_t kArenaBytes = 32u * 1024 * 1024;

    void init(ggml_backend_t b, ggml_context * w) {
        backend = b;
        ctx_w = w;
        arena.resize(kArenaBytes);
        // ggml_get_tensor is a linear strcmp scan over every resident tensor, and building
        // the stage graphs looks up a few hundred weights by name per frame. Index them once.
        by_name.clear();
        for (ggml_tensor * t = ggml_get_first_tensor(w); t; t = ggml_get_next_tensor(w, t)) {
            by_name.emplace(t->name, t);
        }
    }
    ggml_tensor * weight(const char * name) const {
        auto it = by_name.find(name);
        if (it == by_name.end()) {
            std::fprintf(stderr, "vla(octo): missing resident weight %s\n", name);
            return nullptr;
        }
        return it->second;
    }
    ggml_context * open_ctx() {
        ggml_init_params gp = {arena.size(), arena.data(), /*no_alloc=*/true};
        return ggml_init(gp);
    }
    ggml_gallocr_t alloc(OctoStage stage) {
        ggml_gallocr_t& a = allocs[(size_t) stage];
        if (!a) a = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        return a;
    }
    // Must run before the backend it allocated from is freed.
    void reset() {
        for (ggml_gallocr_t& a : allocs) {
            if (a) ggml_gallocr_free(a);
            a = nullptr;
        }
    }
    ~OctoRuntime() { reset(); }
};

struct OctoModelArch : public ModelArchBase {
    OctoModelArch() : ModelArchBase(Arch::OCTO) {}
    ~OctoModelArch() override {
        rt.reset();
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
    OctoRuntime rt;

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

// SmallStem's StdConv standardizes its kernel per output channel on every forward pass
// (StdConvPt.forward: (w-mean)/sqrt(var+1e-10), population variance over [in,kh,kw]). The
// kernel is frozen at inference, so the result is too -- bake it in while the weights are
// still passing through host memory on their way to the backend, instead of paying a
// device->host->device round trip per camera per frame.
static bool is_stem_conv_weight(const char * name) {
    return std::strstr(name, "octo.obs.") == name &&
           std::strstr(name, ".stem.") != nullptr &&
           std::strstr(name, ".conv.weight") != nullptr;
}

bool is_matmul_tensor(const char * name) {
    return std::strstr(name, ".weight") != nullptr &&
           std::strstr(name, ".gn.") == nullptr &&
           std::strstr(name, "_norm.") == nullptr &&
           std::strstr(name, ".ln.") == nullptr &&
           std::strstr(name, "pos_embd") == nullptr &&
           std::strstr(name, "time_fourier") == nullptr;
}

// In-place per-output-channel weight standardization; see is_stem_conv_weight.
static void standardize_conv_weight(float * w, int64_t oc, int64_t n) {
    for (int64_t o = 0; o < oc; ++o) {
        float * row = w + o * n;
        double mean = 0.0, var = 0.0;
        for (int64_t i = 0; i < n; ++i) mean += row[i];
        mean /= (double) n;
        for (int64_t i = 0; i < n; ++i) {
            const double d = (double) row[i] - mean;
            var += d * d;
        }
        const float inv = 1.0f / std::sqrt((float) (var / (double) n) + 1e-10f);
        for (int64_t i = 0; i < n; ++i) row[i] = ((float) row[i] - (float) mean) * inv;
    }
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
        if (is_stem_conv_weight(t->name)) {
            // ggml ne = [kw, kh, in, out]; one contiguous block of kw*kh*in per output channel.
            standardize_conv_weight(reinterpret_cast<float *>(bytes.data()),
                                    t->ne[3], t->ne[0] * t->ne[1] * t->ne[2]);
        }
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

// SmallStem16 image tokenizer for one camera view: 4 standardized-conv + GroupNorm + ReLU
// stages (stride 2 each, so side/16 x side/16 cells), a 1x1 patch embedding to 512, a linear
// projection to the 384-wide model width, and the per-timestep position embedding.
//
// `obs` holds the frames for `steps` (obs.shape = {1, steps, 3, side, side}) and `task` the
// single goal frame concatenated onto every one of them as channels 3..5. Every weight is
// referenced straight out of the resident context -- reshaped where the graph needs a
// different rank, never copied through host memory.
static bool run_obs_tokenizer_graph(OctoRuntime& rt,
                                    OctoStage stage,
                                    const char * view,
                                    const NpyU8& obs,
                                    const NpyU8& task,
                                    int side,
                                    int n_tok,
                                    int steps,
                                    const std::vector<int32_t>& pos_rows,
                                    std::vector<float>& pos) {
    if (obs.shape.size() != 5 || task.shape.size() != 4 ||
        obs.shape[0] != 1 || obs.shape[1] != steps || obs.shape[2] != 3 ||
        task.shape[0] != 1 || task.shape[1] != 3 ||
        obs.shape[3] != side || obs.shape[4] != side ||
        task.shape[2] != side || task.shape[3] != side) {
        std::fprintf(stderr, "vla(octo): unexpected input image shape for side=%d\n", side);
        return false;
    }
    if ((int) pos_rows.size() != steps) return false;
    pos.assign((size_t) steps * n_tok * 384, 0.0f);

    std::vector<float> input((size_t) side * side * 6 * steps, 0.0f);
    for (int t = 0; t < steps; ++t) {
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

    ggml_context * ctx = rt.open_ctx();
    if (!ctx) {
        std::fprintf(stderr, "vla(octo): ggml_init(tokenizer graph ctx) failed\n");
        return false;
    }

    ggml_tensor * input_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, side, side, 6, steps);
    ggml_set_name(input_t, "octo.obs.input_norm");
    ggml_tensor * x = input_t;

    char rname[160];
    auto weight = [&](const char * suffix, int li) -> ggml_tensor * {
        if (li < 0) std::snprintf(rname, sizeof(rname), "octo.obs.%s.%s", view, suffix);
        else        std::snprintf(rname, sizeof(rname), "octo.obs.%s.stem.%d.%s", view, li, suffix);
        return rt.weight(rname);
    };
    // conv/GroupNorm biases and scales arrive as [ch]; the graph applies them across a
    // [w,h,ch,steps] feature map, so reshape (a view, not a copy) to [1,1,ch,1] and broadcast.
    auto per_channel = [&](ggml_tensor * t) {
        return t ? ggml_reshape_4d(ctx, t, 1, 1, t->ne[0], 1) : nullptr;
    };

    for (int li = 0; li < 4; ++li) {
        ggml_tensor * cw = weight("conv.weight", li);   // already standardized at load
        ggml_tensor * cb = per_channel(weight("conv.bias", li));
        ggml_tensor * gw = per_channel(weight("gn.weight", li));
        ggml_tensor * gb = per_channel(weight("gn.bias", li));
        if (!cw || !cb || !gw || !gb) { ggml_free(ctx); return false; }

        x = ggml_conv_2d(ctx, cw, x, 2, 2, 1, 1, 1, 1);
        x = ggml_add(ctx, x, cb);
        x = ggml_group_norm(ctx, x, 32, 1e-5f);
        x = ggml_add(ctx, ggml_mul(ctx, x, gw), gb);
        x = ggml_relu(ctx, x);
    }

    ggml_tensor * pw = weight("patch_embd.weight", -1);
    ggml_tensor * pb = per_channel(weight("patch_embd.bias", -1));
    ggml_tensor * jw = weight("proj.weight", -1);
    ggml_tensor * jb = weight("proj.bias", -1);
    ggml_tensor * pos_r = weight("pos_embd", -1);
    if (!pw || !pb || !jw || !jb || !pos_r) { ggml_free(ctx); return false; }

    ggml_tensor * patch = ggml_add(ctx, ggml_conv_2d(ctx, pw, x, 1, 1, 0, 0, 1, 1), pb);
    ggml_tensor * tok_t = ggml_cont(ctx, ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_permute(ctx, patch, 1, 2, 0, 3)), 512, n_tok, steps));

    // pos_r is [384, n_tok, max_horizon]; gather the rows for the timesteps actually in the
    // sequence (which need not be the leading ones -- see OctoSeqLayout).
    ggml_tensor * rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, steps);
    ggml_set_name(rows, "octo.obs.pos_embd.rows");
    ggml_tensor * pe = ggml_reshape_3d(ctx,
        ggml_get_rows(ctx, ggml_reshape_2d(ctx, pos_r, 384 * n_tok, pos_r->ne[2]), rows),
        384, n_tok, steps);

    ggml_tensor * pos_t = ggml_add(ctx, ggml_add(ctx, ggml_mul_mat(ctx, jw, tok_t), jb), pe);
    ggml_set_name(pos_t, "obs.tokenizer.pos");
    ggml_set_output(pos_t);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 8192, false);
    ggml_build_forward_expand(graph, pos_t);
    ggml_gallocr_t gallocr = rt.alloc(stage);
    if (!gallocr || !ggml_gallocr_alloc_graph(gallocr, graph)) {
        std::fprintf(stderr, "vla(octo): tokenizer ggml_gallocr_alloc_graph failed\n");
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_set(input_t, input.data(), 0, ggml_nbytes(input_t));
    ggml_backend_tensor_set(rows, pos_rows.data(), 0, ggml_nbytes(rows));
    const ggml_status st = ggml_backend_graph_compute(rt.backend, graph);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "vla(octo): tokenizer ggml_backend_graph_compute failed (%d)\n", (int) st);
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_get(pos_t, pos.data(), 0, ggml_nbytes(pos_t));
    ggml_free(ctx);
    return true;
}

// LowdimObsTokenizerPt(obs_keys=["proprio"]): one token per state dimension. p_norm (already
// z-scored against the checkpoint's proprio stats by the caller) either feeds the shared
// Linear(1,384) directly (in_dim=1, discretize=False) or is bucketed against
// octo.obs.proprio.bin_thresholds into a one-hot(256) first (in_dim=256, discretize=True).
// Both converge on the same [in_dim,7,steps] -> [384,7,steps] graph once tokens_in is built.
static bool run_proprio_tokenizer_graph(OctoRuntime& rt,
                                        const std::vector<float>& proprio_norm,  // [steps,7], z-scored
                                        int in_dim,                              // 1 or 256
                                        int steps,
                                        const std::vector<int32_t>& pos_rows,
                                        std::vector<float>& pos_out) {           // [steps,7,384]
    constexpr int n_dims = kProprioTokens;
    if (proprio_norm.size() != (size_t) steps * n_dims) return false;
    if ((int) pos_rows.size() != steps) return false;
    if (in_dim != 1 && in_dim != 256) {
        std::fprintf(stderr, "vla(octo): proprio tokenizer in_dim=%d unsupported (expected 1 or 256)\n", in_dim);
        return false;
    }
    pos_out.assign((size_t) steps * n_dims * 384, 0.0f);

    ggml_tensor * proj_w_r = rt.weight("octo.obs.proprio.proj.weight");
    ggml_tensor * proj_b_r = rt.weight("octo.obs.proprio.proj.bias");
    ggml_tensor * pos_r = rt.weight("octo.obs.proprio.pos_embd");
    if (!proj_w_r || !proj_b_r || !pos_r) return false;

    std::vector<float> tokens_in((size_t) in_dim * n_dims * steps, 0.0f);
    if (in_dim == 1) {
        // Continuous: the token IS the z-scored scalar.
        for (size_t i = 0; i < proprio_norm.size(); ++i) tokens_in[i] = proprio_norm[i];
    } else {
        // BinTokenizerPt: torch.bucketize(x, boundaries) -- index = count of boundaries <= x --
        // then one-hot. No checkpoint observed so far sets discretize=True; implemented per spec.
        ggml_tensor * thresholds_r = rt.weight("octo.obs.proprio.bin_thresholds");
        if (!thresholds_r) return false;
        const std::vector<float> thresholds = tensor_to_vec(thresholds_r);
        const int n_thresh = (int) thresholds.size();
        for (int t = 0; t < steps; ++t) {
            for (int d = 0; d < n_dims; ++d) {
                const float v = proprio_norm[(size_t) t * n_dims + d];
                int bucket = 0;
                while (bucket < n_thresh && thresholds[(size_t) bucket] <= v) ++bucket;
                bucket = std::min(bucket, in_dim - 1);
                tokens_in[((size_t) t * n_dims + d) * in_dim + bucket] = 1.0f;
            }
        }
    }

    if (pos_r->ne[2] < steps) {
        std::fprintf(stderr, "vla(octo): octo.obs.proprio.pos_embd too small for %d timesteps\n", steps);
        return false;
    }

    ggml_context * ctx = rt.open_ctx();
    if (!ctx) return false;

    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, in_dim, n_dims, steps);
    ggml_set_name(x, "octo.obs.proprio.tokens_in");
    ggml_tensor * rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, steps);
    ggml_set_name(rows, "octo.obs.proprio.pos_embd.rows");

    ggml_tensor * pe = ggml_reshape_3d(ctx,
        ggml_get_rows(ctx, ggml_reshape_2d(ctx, pos_r, 384 * n_dims, pos_r->ne[2]), rows),
        384, n_dims, steps);
    ggml_tensor * pos_t = ggml_add(ctx, ggml_add(ctx, ggml_mul_mat(ctx, proj_w_r, x), proj_b_r), pe);
    ggml_set_name(pos_t, "obs.proprio.pos");
    ggml_set_output(pos_t);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 256, false);
    ggml_build_forward_expand(graph, pos_t);
    ggml_gallocr_t gallocr = rt.alloc(OctoStage::PROPRIO);
    if (!gallocr || !ggml_gallocr_alloc_graph(gallocr, graph)) {
        std::fprintf(stderr, "vla(octo): proprio tokenizer ggml_gallocr_alloc_graph failed\n");
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_set(x, tokens_in.data(), 0, ggml_nbytes(x));
    ggml_backend_tensor_set(rows, pos_rows.data(), 0, ggml_nbytes(rows));
    const ggml_status st = ggml_backend_graph_compute(rt.backend, graph);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "vla(octo): proprio tokenizer ggml_backend_graph_compute failed (%d)\n", (int) st);
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_get(pos_t, pos_out.data(), 0, ggml_nbytes(pos_t));
    ggml_free(ctx);
    return true;
}

// task_language projection + position embedding, then repeat_task_tokens: the same 16
// projected language tokens are duplicated into every observation timestep of the sequence.
// `t5` is the encoder output for this call; the projection weight/bias and position embedding
// are referenced in place from the resident context.
static bool run_language_graph(OctoRuntime& rt,
                               const std::vector<float>& t5,     // [16,768]
                               int steps,
                               std::vector<float>& pos,          // [16,384]
                               std::vector<float>& repeated) {   // [steps,16,384]
    if (t5.size() != (size_t) 16 * 768) {
        std::fprintf(stderr, "vla(octo): expected T5 output of 16x768\n");
        return false;
    }
    pos.assign((size_t) 16 * 384, 0.0f);
    repeated.assign((size_t) steps * 16 * 384, 0.0f);

    ggml_tensor * jw = rt.weight("octo.task.language.proj.weight");
    ggml_tensor * jb = rt.weight("octo.task.language.proj.bias");
    ggml_tensor * pe = rt.weight("octo.task.language.pos_embd");
    if (!jw || !jb || !pe) return false;

    ggml_context * ctx = rt.open_ctx();
    if (!ctx) {
        std::fprintf(stderr, "vla(octo): ggml_init(language graph ctx) failed\n");
        return false;
    }

    ggml_tensor * inp = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 768, 16, 1);
    ggml_set_name(inp, "octo.task.language.t5_inject");

    ggml_tensor * pos_t = ggml_add(ctx, ggml_add(ctx, ggml_mul_mat(ctx, jw, inp), jb), pe);
    ggml_set_name(pos_t, "task_language.pos");
    ggml_set_output(pos_t);
    ggml_tensor * repeated_t = ggml_repeat_4d(ctx, pos_t, 384, 16, steps, 1);
    ggml_set_name(repeated_t, "obs_task_language.repeated");
    ggml_set_output(repeated_t);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 1024, false);
    ggml_build_forward_expand(graph, repeated_t);
    ggml_gallocr_t gallocr = rt.alloc(OctoStage::LANGUAGE);
    if (!gallocr || !ggml_gallocr_alloc_graph(gallocr, graph)) {
        std::fprintf(stderr, "vla(octo): language ggml_gallocr_alloc_graph failed\n");
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_set(inp, t5.data(), 0, ggml_nbytes(inp));
    const ggml_status st = ggml_backend_graph_compute(rt.backend, graph);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "vla(octo): language ggml_backend_graph_compute failed (%d)\n", (int) st);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_get(pos_t, pos.data(), 0, ggml_nbytes(pos_t));
    ggml_backend_tensor_get(repeated_t, repeated.data(), 0, ggml_nbytes(repeated_t));
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
static bool run_t5_encoder_graph(OctoRuntime& rt,
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

    ggml_tensor * tok_embd_r = rt.weight("octo.t5.tok_embd.weight");
    ggml_tensor * rel_b_r = rt.weight("octo.t5.blk.0.attn_rel_b.weight");
    ggml_tensor * outw_r = rt.weight("octo.t5.output_norm.weight");
    if (!tok_embd_r || !rel_b_r || !outw_r) return false;
    char rname[160];
    ggml_tensor * blk_w[12][8];  // attn_norm, q, k, v, o, ffn_norm, ffn_up, ffn_down
    const char * leaves[8] = {"attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
                              "attn_o.weight", "ffn_norm.weight", "ffn_up.weight", "ffn_down.weight"};
    for (int i = 0; i < 12; ++i) {
        for (int j = 0; j < 8; ++j) {
            std::snprintf(rname, sizeof(rname), "octo.t5.blk.%d.%s", i, leaves[j]);
            blk_w[i][j] = rt.weight(rname);
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

    ggml_context * ctx = rt.open_ctx();
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
    ggml_gallocr_t gallocr = rt.alloc(OctoStage::T5);
    if (!gallocr || !ggml_gallocr_alloc_graph(gallocr, graph)) {
        std::fprintf(stderr, "vla(octo): T5 encoder ggml_gallocr_alloc_graph failed\n");
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
    const ggml_status st = ggml_backend_graph_compute(rt.backend, graph);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "vla(octo): T5 encoder ggml_backend_graph_compute failed (%d)\n", (int) st);
        ggml_free(ctx);
        return false;
    }
    t5_out.resize((size_t) hidden * seq);
    ggml_backend_tensor_get(out, t5_out.data(), 0, ggml_nbytes(out));
    ggml_free(ctx);
    return true;
}

// Which tokenizer a run of sequence tokens came from. TASK is the once-only language prefix;
// PRIMARY/WRIST/PROPRIO/LANGUAGE are per-timestep observation groups (the repeated task
// tokens count as observation, matching repeat_task_tokens); READOUT is the action query.
enum class OctoGroup { TASK, PRIMARY, WRIST, PROPRIO, LANGUAGE, READOUT };

// One contiguous run of same-group, same-timestep tokens in the block-transformer sequence.
struct OctoSeqRun {
    OctoGroup group;
    int timestep;    ///< -1 for the task prefix.
    int n_tokens;
    int src_step;    ///< Index of this timestep inside that group's own (compacted) buffer.
    bool key_valid;  ///< False => masked out as a key for every query.
};

// The sequence Octo's block transformer actually sees.
//
// Observation tokens belonging to a padded timestep are left OUT of the sequence rather than
// emitted and then masked. They cannot affect the result: the pad mask makes them invalid
// keys for every query, and the only output read downstream is the readout token's, so their
// own rows go nowhere. Dropping them shortens the cold-start window sequence from 690 to 370
// tokens (window_size=2) and lets the image tokenizer run its conv stem over the frames that
// are actually live instead of over `window_size` duplicates of one frame.
struct OctoSeqLayout {
    std::vector<OctoSeqRun> runs;
    int seq = 0;
    std::vector<int32_t> primary_steps;   ///< Original timestep of each emitted primary batch entry.
    std::vector<int32_t> wrist_steps;
    std::vector<int32_t> proprio_steps;
    std::vector<int32_t> readout_seq_idx; ///< Sequence position of each timestep's readout token.
};

struct OctoDiffusionSchedule {
    std::array<float, 20> betas{};
    std::array<float, 20> alphas{};
    std::array<float, 20> alpha_hats{};
};

// Per-timestep group order is primary -> wrist -> proprio -> repeated-language -> readout,
// with the task prefix once at the front.
static OctoSeqLayout build_seq_layout(bool task_valid,
                                      const std::vector<uint8_t>& primary_valid,
                                      const std::vector<uint8_t>& wrist_valid,
                                      const std::vector<uint8_t>& timestep_valid,
                                      int window_size,
                                      int n_primary, int n_wrist, int n_proprio) {
    OctoSeqLayout L;
    auto emit = [&](OctoGroup g, int t, int n, int src, bool valid) {
        if (n <= 0) return;
        L.runs.push_back({g, t, n, src, valid});
        L.seq += n;
    };

    emit(OctoGroup::TASK, -1, kTaskTokens, 0, task_valid);
    for (int t = 0; t < window_size; ++t) {
        const bool live = timestep_valid[(size_t) t] != 0;
        if (live && primary_valid[(size_t) t]) {
            emit(OctoGroup::PRIMARY, t, n_primary, (int) L.primary_steps.size(), true);
            L.primary_steps.push_back(t);
        }
        if (live && wrist_valid[(size_t) t]) {
            emit(OctoGroup::WRIST, t, n_wrist, (int) L.wrist_steps.size(), true);
            L.wrist_steps.push_back(t);
        }
        if (live && n_proprio > 0) {
            emit(OctoGroup::PROPRIO, t, n_proprio, (int) L.proprio_steps.size(), true);
            L.proprio_steps.push_back(t);
        }
        // The repeated task tokens and the readout query stay in the sequence for every
        // timestep: both are valid keys for later timesteps regardless of the pad mask.
        emit(OctoGroup::LANGUAGE, t, kTaskTokens, t, task_valid);
        L.readout_seq_idx.push_back(L.seq);
        emit(OctoGroup::READOUT, t, 1, t, true);
    }
    return L;
}

// Lays the per-group token buffers out in sequence order. Each group's buffer is indexed by
// the run's src_step, which for the image/proprio groups is the compacted batch index (only
// live timesteps were tokenized) and for language/readout is the original timestep.
static bool assemble_transformer_input(const OctoSeqLayout& layout,
                                       const std::vector<float>& task_language,
                                       const std::vector<float>& obs_primary,
                                       const std::vector<float>& obs_wrist,
                                       const std::vector<float>& obs_proprio,
                                       const std::vector<float>& repeated_language,
                                       const std::vector<float>& readout_pos,
                                       std::vector<float>& input) {
    constexpr int hidden = 384;
    input.assign((size_t) layout.seq * hidden, 0.0f);
    size_t dst = 0;
    for (const OctoSeqRun& r : layout.runs) {
        const std::vector<float> * src = nullptr;
        switch (r.group) {
            case OctoGroup::TASK:     src = &task_language;     break;
            case OctoGroup::PRIMARY:  src = &obs_primary;       break;
            case OctoGroup::WRIST:    src = &obs_wrist;         break;
            case OctoGroup::PROPRIO:  src = &obs_proprio;       break;
            case OctoGroup::LANGUAGE: src = &repeated_language; break;
            case OctoGroup::READOUT:  src = &readout_pos;       break;
        }
        const size_t n = (size_t) r.n_tokens * hidden;
        const size_t off = (size_t) r.src_step * n;
        if (src->size() < off + n) {
            std::fprintf(stderr, "vla(octo): invalid tensor size while assembling block transformer input\n");
            return false;
        }
        std::copy_n(src->begin() + (ptrdiff_t) off, n, input.begin() + (ptrdiff_t) dst);
        dst += n;
    }
    return true;
}

// Additive attention mask, 0 where attention is allowed and -FLT_MAX where it is blocked.
// Block-wise rules: a task token sees only task tokens; an observation token sees the task
// prefix plus every observation token at its own timestep or earlier; a readout token sees
// those plus the readout tokens up to its own timestep. A key whose pad mask says it is not
// real is blocked for everyone. Shape is [seq,seq] with no head axis -- ggml_soft_max_ext
// broadcasts a mask with ne2 == 1 over all heads.
static void build_transformer_mask(const OctoSeqLayout& layout, std::vector<float>& mask) {
    const int seq = layout.seq;
    std::vector<OctoGroup> group((size_t) seq);
    std::vector<int> timestep((size_t) seq);
    std::vector<uint8_t> key_valid((size_t) seq);
    int i = 0;
    for (const OctoSeqRun& r : layout.runs) {
        for (int k = 0; k < r.n_tokens; ++k, ++i) {
            group[(size_t) i] = r.group;
            timestep[(size_t) i] = r.timestep;
            key_valid[(size_t) i] = r.key_valid ? 1 : 0;
        }
    }

    mask.assign((size_t) seq * seq, 0.0f);
    for (int q = 0; q < seq; ++q) {
        const OctoGroup qg = group[(size_t) q];
        const int qt = timestep[(size_t) q];
        for (int k = 0; k < seq; ++k) {
            const OctoGroup kg = group[(size_t) k];
            const bool k_task = kg == OctoGroup::TASK;
            const bool k_readout = kg == OctoGroup::READOUT;
            const bool k_obs = !k_task && !k_readout;
            bool allowed;
            if (qg == OctoGroup::TASK) {
                allowed = k_task;
            } else if (qg == OctoGroup::READOUT) {
                allowed = k_task || ((k_obs || k_readout) && timestep[(size_t) k] <= qt);
            } else {
                allowed = k_task || (k_obs && timestep[(size_t) k] <= qt);
            }
            if (!allowed || !key_valid[(size_t) k]) mask[(size_t) q * seq + k] = -FLT_MAX;
        }
    }
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

// One reverse-process step of the diffusion action head, as graph nodes: Fourier time
// embedding -> conditioning MLP -> concat(cond, readout, noisy action) -> 3 residual blocks
// -> eps prediction. `weights` is the pre-resolved weight table (see OctoScoreActorWeights)
// so a 20-step chain can be built without re-looking-up 27 tensors per step.
struct OctoScoreActorWeights {
    ggml_tensor * time_w;
    ggml_tensor * c0w; ggml_tensor * c0b;
    ggml_tensor * c1w; ggml_tensor * c1b;
    ggml_tensor * rinw; ggml_tensor * rinb;
    ggml_tensor * routw; ggml_tensor * routb;
    ggml_tensor * blk[3][6];  // ln.{w,b}, fc1.{w,b}, fc2.{w,b}
};

static bool resolve_score_actor_weights(const OctoRuntime& rt, OctoScoreActorWeights& w) {
    w.time_w = rt.weight("octo.head.diffusion.time_fourier.weight");
    w.c0w = rt.weight("octo.head.diffusion.cond.0.weight");
    w.c0b = rt.weight("octo.head.diffusion.cond.0.bias");
    w.c1w = rt.weight("octo.head.diffusion.cond.1.weight");
    w.c1b = rt.weight("octo.head.diffusion.cond.1.bias");
    w.rinw = rt.weight("octo.head.diffusion.reverse.in.weight");
    w.rinb = rt.weight("octo.head.diffusion.reverse.in.bias");
    w.routw = rt.weight("octo.head.diffusion.reverse.out.weight");
    w.routb = rt.weight("octo.head.diffusion.reverse.out.bias");
    if (!w.time_w || !w.c0w || !w.c0b || !w.c1w || !w.c1b || !w.rinw || !w.rinb || !w.routw || !w.routb) return false;

    char rname[160];
    const char * leaves[6] = {"ln.weight", "ln.bias", "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 6; ++j) {
            std::snprintf(rname, sizeof(rname), "octo.head.diffusion.reverse.blk.%d.%s", i, leaves[j]);
            w.blk[i][j] = rt.weight(rname);
            if (!w.blk[i][j]) return false;
        }
    }
    return true;
}

static ggml_tensor * build_score_actor(ggml_context * ctx,
                                       const OctoScoreActorWeights& w,
                                       ggml_tensor * time,      // [1,width]
                                       ggml_tensor * obs,       // [384,width]
                                       ggml_tensor * actions) { // [action,width]
    constexpr float ln_eps = 1e-6f;
    constexpr float two_pi = 6.2831853071795864769f;

    ggml_tensor * f = ggml_scale(ctx, ggml_mul_mat(ctx, w.time_w, time), two_pi);
    ggml_tensor * time_ff = ggml_concat(ctx, ggml_cos(ctx, f), ggml_sin(ctx, f), 0);
    ggml_tensor * cond = ggml_silu(ctx, ggml_add(ctx, ggml_mul_mat(ctx, w.c0w, time_ff), w.c0b));
    cond = ggml_add(ctx, ggml_mul_mat(ctx, w.c1w, cond), w.c1b);
    ggml_tensor * reverse_input = ggml_concat(ctx, ggml_concat(ctx, cond, obs, 0), actions, 0);
    ggml_tensor * x = ggml_add(ctx, ggml_mul_mat(ctx, w.rinw, reverse_input), w.rinb);
    for (int i = 0; i < 3; ++i) {
        ggml_tensor * residual = x;
        ggml_tensor * h = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, ln_eps), w.blk[i][0]), w.blk[i][1]);
        h = ggml_silu(ctx, ggml_add(ctx, ggml_mul_mat(ctx, w.blk[i][2], h), w.blk[i][3]));
        h = ggml_add(ctx, ggml_mul_mat(ctx, w.blk[i][4], h), w.blk[i][5]);
        x = ggml_add(ctx, residual, h);
    }
    return ggml_add(ctx, ggml_mul_mat(ctx, w.routw, ggml_silu(ctx, x)), w.routb);
}

// DDPM reverse process: `steps` denoising iterations over the score actor, seeded from
// `rng` (N(0,1) initial noise, plus one fresh N(0,1) draw per step while time_value > 0 --
// the reverse process adds no noise at t=0). Each step is dispatched on its own: the steps
// are sequentially dependent so they cannot be batched, and chaining all 20 into a single
// graph measured slower on both CUDA and CPU (a 20x larger graph to build and allocate,
// which costs more than the device syncs it saves for a head this small).
static bool run_diffusion(OctoRuntime& rt,
                          std::mt19937& rng,
                          const std::vector<float>& readout_action,
                          int window_size,
                          int action_total,
                          std::vector<float>& final_actions) {
    constexpr int steps = 20;
    const int width = window_size;
    const int action = action_total;
    constexpr float max_action = 5.0f;
    if (readout_action.size() != (size_t) 384 * width || width < 1 || action < 1) return false;

    // Resolved once rather than per step: wt() is a linear scan of the resident context, and
    // the head needs 27 of them.
    OctoScoreActorWeights w{};
    if (!resolve_score_actor_weights(rt, w)) return false;

    const OctoDiffusionSchedule sched = make_cosine_schedule();
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> x((size_t) width * action);
    for (float& v : x) v = normal(rng);
    std::vector<float> z((size_t) width * action);
    std::vector<float> eps((size_t) width * action);
    std::vector<float> time_data((size_t) width);

    for (int step = 0; step < steps; ++step) {
        const int time_value = steps - 1 - step;

        ggml_context * ctx = rt.open_ctx();
        if (!ctx) return false;
        ggml_tensor * time = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, width);
        ggml_set_name(time, "action_head.time");
        ggml_tensor * obs = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 384, width);
        ggml_set_name(obs, "action_head.readout_embedding");
        ggml_tensor * actions = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, action, width);
        ggml_set_name(actions, "action_head.noisy_action");

        ggml_tensor * out = build_score_actor(ctx, w, time, obs, actions);
        ggml_set_name(out, "action_head.pred_eps");
        ggml_set_output(out);

        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 2048, false);
        ggml_build_forward_expand(graph, out);
        ggml_gallocr_t gallocr = rt.alloc(OctoStage::SCORE_ACTOR);
        if (!gallocr || !ggml_gallocr_alloc_graph(gallocr, graph)) {
            std::fprintf(stderr, "vla(octo): score actor ggml_gallocr_alloc_graph failed\n");
            ggml_free(ctx);
            return false;
        }
        std::fill(time_data.begin(), time_data.end(), (float) time_value);
        ggml_backend_tensor_set(time, time_data.data(), 0, ggml_nbytes(time));
        ggml_backend_tensor_set(obs, readout_action.data(), 0, ggml_nbytes(obs));
        ggml_backend_tensor_set(actions, x.data(), 0, ggml_nbytes(actions));
        const ggml_status st = ggml_backend_graph_compute(rt.backend, graph);
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "vla(octo): score actor ggml_backend_graph_compute failed (%d)\n", (int) st);
            ggml_free(ctx);
            return false;
        }
        ggml_backend_tensor_get(out, eps.data(), 0, ggml_nbytes(out));
        ggml_free(ctx);

        if (time_value > 0) {
            for (float& v : z) v = normal(rng);
        }
        const float alpha = sched.alphas[(size_t) time_value];
        const float beta = sched.betas[(size_t) time_value];
        const float alpha_hat = sched.alpha_hats[(size_t) time_value];
        const float alpha_1 = 1.0f / std::sqrt(alpha);
        const float alpha_2 = (1.0f - alpha) / std::sqrt(1.0f - alpha_hat);
        for (size_t i = 0; i < x.size(); ++i) x[i] = alpha_1 * (x[i] - alpha_2 * eps[i]);
        if (time_value > 0) {
            const float sigma = std::sqrt(beta);
            for (size_t i = 0; i < x.size(); ++i) x[i] += sigma * z[i];
        }
        for (float& v : x) v = std::min(std::max(v, -max_action), max_action);
    }

    // sample_actions() returns the last window timestep's chunk.
    final_actions.assign(x.end() - action, x.end());
    return true;
}

// TIP-05 Part B: L1 action head forward -- ContinuousActionHeadPt.forward/predict_action,
// replacing the diffusion head entirely when head_type=="l1" (routed in
// octo_run_pipeline).
struct OctoL1HeadResult {
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
static bool run_l1_action_head_graph(OctoRuntime& rt,
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

    ggml_tensor * probe_r      = rt.weight("octo.head.l1.map.probe");
    ggml_tensor * qkv_w_r      = rt.weight("octo.head.l1.map.attn_qkv.weight");
    ggml_tensor * qkv_b_r      = rt.weight("octo.head.l1.map.attn_qkv.bias");
    ggml_tensor * o_w_r        = rt.weight("octo.head.l1.map.attn_o.weight");
    ggml_tensor * o_b_r        = rt.weight("octo.head.l1.map.attn_o.bias");
    ggml_tensor * norm_w_r     = rt.weight("octo.head.l1.map.norm.weight");
    ggml_tensor * norm_b_r     = rt.weight("octo.head.l1.map.norm.bias");
    ggml_tensor * ffn_up_w_r   = rt.weight("octo.head.l1.map.ffn_up.weight");
    ggml_tensor * ffn_up_b_r   = rt.weight("octo.head.l1.map.ffn_up.bias");
    ggml_tensor * ffn_down_w_r = rt.weight("octo.head.l1.map.ffn_down.weight");
    ggml_tensor * ffn_down_b_r = rt.weight("octo.head.l1.map.ffn_down.bias");
    ggml_tensor * mean_w_r     = rt.weight("octo.head.l1.mean_proj.weight");
    ggml_tensor * mean_b_r     = rt.weight("octo.head.l1.mean_proj.bias");
    if (!probe_r || !qkv_w_r || !qkv_b_r || !o_w_r || !o_b_r || !norm_w_r || !norm_b_r ||
        !ffn_up_w_r || !ffn_up_b_r || !ffn_down_w_r || !ffn_down_b_r || !mean_w_r || !mean_b_r) {
        return false;
    }

    ggml_context * ctx = rt.open_ctx();
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

    ggml_tensor * y = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, attn_out, ln_eps), norm_w_r), norm_b_r);
    ggml_tensor * h = ggml_gelu_erf(ctx, ggml_add(ctx, ggml_mul_mat(ctx, ffn_up_w_r, y), ffn_up_b_r));
    h = ggml_add(ctx, ggml_mul_mat(ctx, ffn_down_w_r, h), ffn_down_b_r);
    // Residual is onto attn_out (pre-LN), not onto y -- per TIP-05 spec, matches
    // MlpBlockPt's usage inside MAPHeadPt (out = attn_out + MlpBlock(LayerNorm(attn_out))).
    ggml_tensor * emb = ggml_add(ctx, attn_out, h);
    ggml_set_name(emb, "l1_head.map.emb");

    ggml_tensor * mean_raw = ggml_add(ctx, ggml_mul_mat(ctx, mean_w_r, emb), mean_b_r);
    ggml_tensor * mean_final = ggml_scale(ctx, ggml_tanh(ctx, ggml_scale(ctx, mean_raw, 1.0f / max_action)), max_action);
    ggml_set_name(mean_final, "l1_head.mean_normalized");
    ggml_set_output(mean_final);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 512, false);
    ggml_build_forward_expand(graph, mean_final);
    ggml_gallocr_t gallocr = rt.alloc(OctoStage::L1_HEAD);
    if (!gallocr || !ggml_gallocr_alloc_graph(gallocr, graph)) {
        std::fprintf(stderr, "vla(octo): L1 head ggml_gallocr_alloc_graph failed\n");
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_set(x, readout_action.data(), 0, ggml_nbytes(x));
    const ggml_status st = ggml_backend_graph_compute(rt.backend, graph);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "vla(octo): L1 head ggml_backend_graph_compute failed (%d)\n", (int) st);
        ggml_free(ctx);
        return false;
    }

    result.mean_normalized.resize((size_t) action_total * width);
    if ((size_t) ggml_nelements(mean_final) != result.mean_normalized.size()) {
        std::fprintf(stderr, "vla(octo): L1 head mean_proj out-dim=%lld does not match action_horizon*action_dim=%d\n",
                     (long long) mean_final->ne[0], action_total);
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_get(mean_final, result.mean_normalized.data(), 0, ggml_nbytes(mean_final));

    // predict_action: last window timestep only (matches diffusion's own final-action-slice
    // convention -- see run_diffusion_resident's "final action slice" comment/the
    // octo_action_slice tripwire test).
    result.final_actions.assign(result.mean_normalized.end() - action_total, result.mean_normalized.end());
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

// Parses octo.dataset_statistics and caches the action (and, when present, proprio) blocks
// on `rt`. Subsequent calls with the same dataset key are a no-op.
static bool ensure_stats(OctoRuntime& rt, gguf_reader& g, const std::string& dataset_key_in) {
    if (rt.stats_loaded && rt.stats_key == dataset_key_in) return true;

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
    OctoRuntime::ActionStats a;
    a.mean = act.at("mean").get<std::vector<float>>();
    a.stdv = act.at("std").get<std::vector<float>>();
    for (bool b : act.at("mask").get<std::vector<bool>>()) a.mask.push_back(b ? 1 : 0);
    if (a.mask.size() != 7 || a.mean.size() != 7 || a.stdv.size() != 7) {
        std::fprintf(stderr, "vla(octo): unexpected dataset_statistics/action shape\n");
        return false;
    }

    OctoRuntime::ProprioStats pr;
    bool has_pr = false;
    if (block->contains("proprio")) {
        const auto& p = (*block)["proprio"];
        pr.mean = p.at("mean").get<std::vector<float>>();
        pr.stdv = p.at("std").get<std::vector<float>>();
        if (pr.mean.size() != (size_t) kProprioTokens || pr.stdv.size() != (size_t) kProprioTokens) {
            std::fprintf(stderr, "vla(octo): unexpected dataset_statistics/proprio shape\n");
            return false;
        }
        has_pr = true;
    }

    rt.action_stats = std::move(a);
    rt.proprio_stats = std::move(pr);
    rt.has_proprio_stats = has_pr;
    rt.stats_key = dataset_key_in;
    rt.stats_loaded = true;
    return true;
}

// Applies the cached action statistics: unnorm[d] = mask[d] ? norm[d]*std[d] + mean[d]
// : norm[d]. Dims with mask=false (e.g. bridge_dataset's and libero_object's gripper dim)
// are passed through untouched, matching OctoPt.
static bool unnormalize_action(const OctoRuntime& rt,
                               const std::vector<float>& normalized_flat,
                               std::vector<float>& unnorm_flat) {
    const OctoRuntime::ActionStats& a = rt.action_stats;
    const size_t dim = a.mask.size();
    if (dim != 7 || normalized_flat.empty() || normalized_flat.size() % dim != 0) {
        std::fprintf(stderr, "vla(octo): unexpected action shape for un-normalization\n");
        return false;
    }
    const size_t horizon = normalized_flat.size() / dim;
    unnorm_flat.resize(normalized_flat.size());
    for (size_t t = 0; t < horizon; ++t) {
        for (size_t d = 0; d < dim; ++d) {
            const float norm = normalized_flat[t * dim + d];
            unnorm_flat[t * dim + d] = a.mask[d] ? (norm * a.stdv[d] + a.mean[d]) : norm;
        }
    }
    return true;
}

// Octo's block transformer: 12 pre-norm encoder blocks over the assembled sequence, with the
// block-wise attention mask built by build_transformer_mask. Only the readout tokens are
// gathered back out -- everything else the blocks compute is intermediate.
static bool run_transformer_graph(OctoRuntime& rt,
                                  const OctoSeqLayout& layout,
                                  const std::vector<float>& input,
                                  const std::vector<float>& blocked_mask,
                                  std::vector<float>& readout_action) {
    constexpr int hidden = 384;
    constexpr int heads = 6;
    constexpr int head_dim = 64;
    constexpr float ln_eps = 1e-6f;
    constexpr float attn_scale = 0.125f;
    const int seq = layout.seq;
    const int n_readout = (int) layout.readout_seq_idx.size();
    if (input.size() != (size_t) hidden * seq || blocked_mask.size() != (size_t) seq * seq) return false;

    char rname[160];
    ggml_tensor * blk_w[12][10];  // attn_norm.{w,b}, attn_qkv.{w,b}, attn_o.{w,b}, ffn_norm.{w,b}, ffn_up.{w,b}
    const char * leaves[10] = {"attn_norm.weight", "attn_norm.bias", "attn_qkv.weight", "attn_qkv.bias",
                               "attn_o.weight", "attn_o.bias", "ffn_norm.weight", "ffn_norm.bias",
                               "ffn_up.weight", "ffn_up.bias"};
    ggml_tensor * ffn_down_w[12];
    ggml_tensor * ffn_down_b[12];
    for (int i = 0; i < 12; ++i) {
        for (int j = 0; j < 10; ++j) {
            std::snprintf(rname, sizeof(rname), "octo.blk.%d.%s", i, leaves[j]);
            blk_w[i][j] = rt.weight(rname);
            if (!blk_w[i][j]) return false;
        }
        std::snprintf(rname, sizeof(rname), "octo.blk.%d.ffn_down.weight", i);
        ffn_down_w[i] = rt.weight(rname);
        std::snprintf(rname, sizeof(rname), "octo.blk.%d.ffn_down.bias", i);
        ffn_down_b[i] = rt.weight(rname);
        if (!ffn_down_w[i] || !ffn_down_b[i]) return false;
    }
    ggml_tensor * out_w_r = rt.weight("octo.output_norm.weight");
    ggml_tensor * out_b_r = rt.weight("octo.output_norm.bias");
    if (!out_w_r || !out_b_r) return false;

    ggml_context * ctx = rt.open_ctx();
    if (!ctx) return false;

    ggml_tensor * input_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, seq);
    ggml_set_name(input_t, "octo.block_transformer.input");
    ggml_tensor * x = input_t;
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, seq, seq);
    ggml_set_name(mask, "octo.block_transformer.additive_mask");
    ggml_tensor * readout_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_readout);
    ggml_set_name(readout_idx, "octo.block_transformer.readout_idx");

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
        mlp = ggml_add(ctx, ggml_mul_mat(ctx, ffn_down_w[i], mlp), ffn_down_b[i]);
        x = ggml_add(ctx, residual, mlp);
    }

    ggml_tensor * output = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, ln_eps), out_w_r), out_b_r);
    // The readout tokens are not evenly spaced once padded timesteps drop their observation
    // groups, so gather them by sequence index rather than striding.
    ggml_tensor * split_readout = ggml_get_rows(ctx, output, readout_idx);
    ggml_set_name(split_readout, "bt.readout_action");
    ggml_set_output(split_readout);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 8192, false);
    ggml_build_forward_expand(graph, split_readout);
    ggml_gallocr_t gallocr = rt.alloc(OctoStage::TRANSFORMER);
    if (!gallocr || !ggml_gallocr_alloc_graph(gallocr, graph)) {
        std::fprintf(stderr, "vla(octo): transformer ggml_gallocr_alloc_graph failed\n");
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_set(input_t, input.data(), 0, ggml_nbytes(input_t));
    ggml_backend_tensor_set(mask, blocked_mask.data(), 0, ggml_nbytes(mask));
    ggml_backend_tensor_set(readout_idx, layout.readout_seq_idx.data(), 0, ggml_nbytes(readout_idx));
    const ggml_status st = ggml_backend_graph_compute(rt.backend, graph);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "vla(octo): transformer ggml_backend_graph_compute failed (%d)\n", (int) st);
        ggml_free(ctx);
        return false;
    }

    readout_action.resize((size_t) hidden * n_readout);
    ggml_backend_tensor_get(split_readout, readout_action.data(), 0, ggml_nbytes(split_readout));
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
    m->rt.init(m->backend, m->ctx_weights);
    std::printf("vla(octo): loaded %lld F32 tensors, hidden=%lld blocks=%lld heads=%lld horizon=%lld "
                "action_dim=%lld head_type=%s window_size=%lld has_proprio=%s proprio_in_dim=%lld "
                "proprio_tokens=%d\n",
                (long long) m->tensors.size(), (long long) m->hidden, (long long) m->blocks,
                (long long) m->heads, (long long) m->action_horizon, (long long) m->action_dim,
                m->head_type.c_str(), (long long) m->window_size, m->has_proprio ? "true" : "false",
                (long long) m->proprio_in_dim, m->has_proprio ? kProprioTokens : 0);
    return m;
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
// Picks the window slots named by `steps` out of a {1, window_size, 3, side, side} buffer.
static NpyU8 slice_obs_steps(const NpyU8& obs, const std::vector<int32_t>& steps, int side) {
    const size_t frame = (size_t) 3 * side * side;
    NpyU8 out;
    out.shape = {1, (int64_t) steps.size(), 3, side, side};
    out.data.resize(steps.size() * frame);
    for (size_t i = 0; i < steps.size(); ++i) {
        std::copy_n(obs.data.begin() + (ptrdiff_t) ((size_t) steps[i] * frame), frame,
                    out.data.begin() + (ptrdiff_t) (i * frame));
    }
    return out;
}

static bool octo_run_pipeline(OctoRuntime& rt, gguf_reader& io, int window_size,
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
    if (!ensure_stats(rt, io, unnorm_dataset)) return false;

    // Cold start: history is filled with copies of the one live frame and every slot but the
    // last is marked padding, matching OctoPt's HistoryWrapper.reset().
    std::vector<uint8_t> primary_valid((size_t) window_size, 1);
    std::vector<uint8_t> wrist_valid((size_t) window_size, wrist_real ? 1 : 0);
    std::vector<uint8_t> timestep_valid((size_t) window_size, 0);
    if (window_size < 1) return false;
    timestep_valid[(size_t) window_size - 1] = 1;

    const OctoSeqLayout layout = build_seq_layout(/*task_valid=*/true, primary_valid, wrist_valid,
                                                  timestep_valid, window_size, 256, 64, n_proprio);

    const auto t_vision0 = clock::now();
    std::vector<float> primary_pos, wrist_pos, proprio_pos;
    if (!layout.primary_steps.empty()) {
        NpyU8 obs = slice_obs_steps(primary_obs, layout.primary_steps, 256);
        if (!run_obs_tokenizer_graph(rt, OctoStage::OBS_PRIMARY, "primary", obs, primary_task, 256, 256,
                                     (int) layout.primary_steps.size(), layout.primary_steps, primary_pos)) return false;
    }
    if (!layout.wrist_steps.empty()) {
        NpyU8 obs = slice_obs_steps(wrist_obs, layout.wrist_steps, 128);
        if (!run_obs_tokenizer_graph(rt, OctoStage::OBS_WRIST, "wrist", obs, wrist_task, 128, 64,
                                     (int) layout.wrist_steps.size(), layout.wrist_steps, wrist_pos)) return false;
    }
    if (!layout.proprio_steps.empty()) {
        if (!rt.has_proprio_stats) {
            std::fprintf(stderr, "vla(octo): proprio checkpoint but dataset_statistics has no .proprio block\n");
            return false;
        }
        const std::vector<float>& proprio_mean = rt.proprio_stats.mean;
        const std::vector<float>& proprio_std = rt.proprio_stats.stdv;
        const int n_steps = (int) layout.proprio_steps.size();
        std::vector<float> proprio_norm((size_t) n_steps * kProprioTokens);
        for (int t = 0; t < n_steps; ++t) {
            for (int d = 0; d < kProprioTokens; ++d) {
                const float raw = (d < (int) proprio_state_raw.size()) ? proprio_state_raw[(size_t) d] : 0.0f;
                proprio_norm[(size_t) t * kProprioTokens + d] = (raw - proprio_mean[(size_t) d]) / proprio_std[(size_t) d];
            }
        }
        if (!run_proprio_tokenizer_graph(rt, proprio_norm, proprio_in_dim, n_steps,
                                         layout.proprio_steps, proprio_pos)) return false;
    }
    ms_vision_out = std::chrono::duration<float, std::milli>(clock::now() - t_vision0).count();

    const auto t_inference0 = clock::now();
    std::vector<int32_t> lang_key = input_ids;
    lang_key.insert(lang_key.end(), attention_mask.begin(), attention_mask.end());
    if (rt.lang_key != lang_key || rt.lang_steps != window_size) {
        std::vector<float> t5_out;
        if (!run_t5_encoder_graph(rt, input_ids, attention_mask, t5_out)) return false;
        if (!run_language_graph(rt, t5_out, window_size, rt.lang_pos, rt.lang_repeated)) return false;
        rt.lang_key = std::move(lang_key);
        rt.lang_steps = window_size;
    }
    const std::vector<float>& lang_pos = rt.lang_pos;
    const std::vector<float>& repeated = rt.lang_repeated;

    ggml_tensor * readout_pos_r = rt.weight("octo.readout.action.pos_embd");
    if (!readout_pos_r) return false;
    const std::vector<float> readout_pos = tensor_to_vec(readout_pos_r);

    std::vector<float> input, mask;
    if (!assemble_transformer_input(layout, lang_pos, primary_pos, wrist_pos, proprio_pos,
                                    repeated, readout_pos, input)) return false;
    build_transformer_mask(layout, mask);

    std::vector<float> readout_action;
    if (!run_transformer_graph(rt, layout, input, mask, readout_action)) return false;

    if (head_type == "l1") {
        OctoL1HeadResult l1;
        if (!run_l1_action_head_graph(rt, readout_action, window_size, action_total, l1)) return false;
        normalized_out = std::move(l1.final_actions);
    } else {
        std::random_device rd;
        std::mt19937 rng(rd());
        if (!run_diffusion(rt, rng, readout_action, window_size, action_total, normalized_out)) return false;
    }
    ms_inference_out = std::chrono::duration<float, std::milli>(clock::now() - t_inference0).count();

    return unnormalize_action(rt, normalized_out, unnormalized_out);
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
    m.rt.init(m.backend, m.ctx_weights);

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
    return octo_run_pipeline(m.rt, g, window_size, action_total, m.head_type,
                             m.has_proprio, (int) m.proprio_in_dim,
                             primary_obs, primary_task, wrist_obs, wrist_task,
                             /*wrist_real=*/true, input_ids, attention_mask, proprio_state_raw, unnorm_dataset,
                             out.normalized, out.unnormalized, ms_vision, ms_inference);
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
    if (!octo_run_pipeline(rt, io, (int) window_size, action_total, head_type,
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

}  // namespace vla
