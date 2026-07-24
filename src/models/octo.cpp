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

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace vla {
namespace {

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

    std::vector<float> predict(const Inputs&) override {
        std::fprintf(stderr, "vla(octo): predict is not implemented in M0; this loader only validates GGUF load/shape metadata\n");
        return {};
    }
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
        m.window_size != 2 || m.action_horizon != 4 || m.action_dim != 7) {
        std::fprintf(stderr, "vla(octo): metadata does not match octo-small-1.5 M0 constants\n");
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

struct NpyU8 {
    std::vector<int64_t> shape;
    std::vector<uint8_t> data;
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

struct OctoObsWeights {
    std::vector<float> conv_w[4], conv_b[4], gn_w[4], gn_b[4];
    std::vector<float> patch_w, patch_b, proj_w, proj_b, pos;
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
                                        std::vector<float>& tok,
                                        std::vector<float>& proj,
                                        std::vector<float>& pos) {
    if (obs.shape.size() != 5 || task.shape.size() != 4 ||
        obs.shape[0] != 1 || obs.shape[1] != 2 || obs.shape[2] != 3 ||
        task.shape[0] != 1 || task.shape[1] != 3 ||
        obs.shape[3] != side || obs.shape[4] != side ||
        task.shape[2] != side || task.shape[3] != side) {
        std::fprintf(stderr, "vla(octo): unexpected input image shape for side=%d\n", side);
        return false;
    }
    tok.assign((size_t) 1 * 2 * n_tok * 512, 0.0f);
    proj.assign((size_t) 1 * 2 * n_tok * 384, 0.0f);
    pos.assign((size_t) 1 * 2 * n_tok * 384, 0.0f);

    const int stem_oc[4] = {32, 96, 192, 384};
    const int stem_ic[4] = {6, 32, 96, 192};
    std::vector<float> input((size_t) side * side * 6 * 2, 0.0f);
    for (int t = 0; t < 2; ++t) {
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

    ggml_tensor * x = make_4d(ctx, "octo.obs.input_norm", side, side, 6, 2);
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
    std::vector<float> pos2(w.pos.begin(), w.pos.begin() + (size_t) 2 * n_tok * 384);
    ggml_tensor * pe = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 384, n_tok, 2);
    ggml_set_name(pe, "octo.obs.pos_embd.window2");
    add_payload(pe, std::move(pos2));

    ggml_tensor * patch = ggml_conv_2d(ctx, pw, x, 1, 1, 0, 0, 1, 1);
    patch = ggml_add(ctx, patch, pb);
    ggml_tensor * tok_t = ggml_cont(ctx, ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_permute(ctx, patch, 1, 2, 0, 3)), 512, n_tok, 2));
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
                              const std::string& dump_dir) {
    gguf_reader g{"octo"};
    if (!g.open(ckpt_path)) return false;

    NpyU8 primary_obs, wrist_obs, primary_task, wrist_task;
    if (!parse_npy_u8(case_dir + "/tensors/input.observation.image_primary.npy", primary_obs)) return false;
    if (!parse_npy_u8(case_dir + "/tensors/input.observation.image_wrist.npy", wrist_obs)) return false;
    if (!parse_npy_u8(case_dir + "/tensors/input.task.image_primary.npy", primary_task)) return false;
    if (!parse_npy_u8(case_dir + "/tensors/input.task.image_wrist.npy", wrist_task)) return false;

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

    std::vector<float> tok, proj, pos;
    if (!run_one_obs_tokenizer_graph(primary_w, primary_obs, primary_task, 256, 256, tok, proj, pos)) return false;
    if (!write_f32_dump(dump_dir, "obs.primary.tok",  tok,  {1, 2, 256, 512}, mf)) return false;
    if (!write_f32_dump(dump_dir, "obs.primary.proj", proj, {1, 2, 256, 384}, mf)) return false;
    if (!write_f32_dump(dump_dir, "obs.primary.pos",  pos,  {1, 2, 256, 384}, mf)) return false;
    if (!run_one_obs_tokenizer_graph(wrist_w, wrist_obs, wrist_task, 128, 64, tok, proj, pos)) return false;
    if (!write_f32_dump(dump_dir, "obs.wrist.tok",  tok,  {1, 2, 64, 512}, mf)) return false;
    if (!write_f32_dump(dump_dir, "obs.wrist.proj", proj, {1, 2, 64, 384}, mf)) return false;
    if (!write_f32_dump(dump_dir, "obs.wrist.pos",  pos,  {1, 2, 64, 384}, mf)) return false;
    return true;
}

}  // namespace vla
