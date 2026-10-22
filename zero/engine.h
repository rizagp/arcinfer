// arcinferzero: the engine on the Level Zero runtime (zero/clrt.h); it began as a copy of the OpenCL engine's src/engine.h (removed 2026-10-20).
// arcinfer engine: Qwen3.5 / Qwen3.6 / Qwen3.8 dense (hybrid Gated DeltaNet + gated attention, GGUF arch
// "qwen35"), their mixture-of-experts variant ("qwen35moe", GSQ 2-bit experts from tools/convert_gsq.py) and
// Gemma 4 on Intel Xe-LPG integrated GPUs via OpenCL.
//
// Weights: the raw GGUF blocks of every matrix live on the GPU (used by the single-token GEMV path). For prompt
// processing each matrix is multiplied in the int8 "W8" form: decoded once at load when it fits the memory
// budget, otherwise decoded just in time before each GEMM (zero/kernels/decode.cl). Q8_0 uses its own repacked form.
#pragma once
#include "clrt.h"
#include "xcache.h"
#include "../src/gguf.h"
#include "../src/quants.h"
#include "../src/json.h"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <set>

struct GpuWeight {
  uint32_t type = 0;  // GGUF type of the raw rows (BF16/F16/F32 matrices are requantized to Q8_0)
  int N = 0, K = 0;
  cl_mem raw = nullptr;               // raw GGUF rows
  cl_mem wq = nullptr, wh = nullptr;  // resident GEMM form (W8, or repacked Q8_0); null = decode just in time
  int gemm_qtype = 0;                 // zero/kernels/gemm.cl QTYPE
  int role = 0;                       // just-in-time W8 staging slot (one per matrix role within a layer)
};

struct Hparams {
  std::string arch = "qwen35";
  bool gemma = false;  // Gemma 4: sliding-window + global attention layers, GeGLU, sandwich norms
  int n_layer = 0, n_embd = 0, n_ff = 0, n_head = 0, n_head_kv = 0, head_dim = 0, n_rot = 0, n_vocab = 0;
  int ssm_d_state = 0, ssm_nk = 0, ssm_nv = 0, ssm_d_inner = 0, conv_k = 0, full_attn_interval = 4;
  float eps = 1e-6f, rope_base = 1e7f;
  int rope_sect[4] = {0, 0, 0, 0};  // M-RoPE sections (Qwen3.5 / 3.6 / 3.8 with vision: 11, 11, 10, 0)
  // Gemma 4 (head_dim above = the largest head size, for buffer sizing)
  int n_swa = 0, hd_swa = 0, hd_full = 0;
  int n_ple = 0, n_layer_kv = 0;  // per-layer embedding size (E2B/E4B), layers that own a KV cache
  float rope_base_swa = 1e4f, softcap = 0.0f;
  std::vector<int> swa, kv_heads;  // per layer
  // mixture of experts (qwen35moe): routed experts (2-bit, zero/kernels/moe.cl) + a sigmoid-gated shared expert
  bool moe = false, moe_std = false;  // moe_std: standard GGUF expert tensors (not GSQ)
  int n_expert = 0, n_expert_used = 0, n_ff_exp = 0, n_ff_shexp = 0;
  int n_ff_exp_pad = 0;  // expert FFN width as the down projection reads it (Flash Next: 640 padded to 768)
  int router_rows() const { return (n_expert + 1 + 15) / 16 * 16; }  // router + shared-expert gate, padded
  // Qwen3.8 Flash Next (qwen4exp): hyper-connections (hc_n residual streams of n_embd, low rank hc_lr) and the PLE
  // n-gram embedding block at layer ple_layer (docs/flash-next-spec.md)
  bool hc = false;
  int hc_n = 1, hc_lr = 0, ple_layer = -1, ple_dim = 0, ple_ng = 0, ple_per = 0, ple_taps = 0;
  int64_t ple_eos = -1;
  std::vector<int64_t> ple_mult, ple_off, ple_voc;
  // QSA sparse attention (Flash Next): block size per layer (0: dense), indexer heads / key size, keys selected
  std::vector<int> qsa_ratio;
  int qsa_heads = 0, qsa_dim = 0, qsa_topk = 0, n_qsa = 0;
  int hce() const { return hc_n * n_embd; }  // residual floats per token
  int conv_dim() const { return 2 * ssm_nk * ssm_d_state + ssm_d_inner; }
  bool recurrent(int il) const { return !gemma && (il + 1) % full_attn_interval != 0; }
};

struct EngineOptions {
  int ctx = 8192;
  int chunk = 0;              // prompt tokens per forward step; 0 = auto
  int min_chunk = 0;          // at least this many (Gemma 4 images: one non-causal block of up to the image limit)
  double w8_budget_gb = 9.0;  // keep decoded W8 weights resident if raw + W8 fit in this budget
  bool act32_all = true;      // per-32 activation scales for every GEMM (false: only for the sensitive inputs)
  bool attn_int8 = true;      // int8 (DP4a) attention scores; false = f32 scores
  bool v4 = false;            // + a 4-bit copy of V for generation (Qwen; prompts read the int8 V; opt-in)
  bool q2 = false;            // Gemma 4: two-pass int8 activations + float attention (KL < 0.01, slower; opt-in for
                              // now, also ARC_Q2=1; other models ignore it)
  bool mtp = false;           // load the MTP (nextn) block for speculative decoding (mtp_draft / verify)
  int mtp_drafts = 0;         // drafts per step (1..7; later ones chained on the MTP block's own output); 0 = auto:
                              // 3 for wide dense Qwen models (27B: code 7.8 -> 9.5 tok/s, text +2.5 %), else 2
  float mtp_min_p = 0.5f;     // draft a second token only if the MTP block gives the first this probability
  std::string mtp_model;      // Gemma 4: assistant drafter GGUF (empty: an mtp-*.gguf next to the model, if any)
  std::string dflash;         // DFlash2 block drafter (a z-lab DFlash2 folder or its model.safetensors): with mtp, it
                              // drafts instead of the MTP block (Qwen3.8 27B)
  bool mtp_block = true;      // with dflash: false = the MTP block is not loaded, DFlash2 drafts every step (--draft
                              // dflash); true = both, picked per step by how many DFlash2 drafts were taken
  std::string kernel_dir = "zero/kernels";  // (arcinferzero: its own copy, with the -DXCACHE kernels)
  bool dry = false;  // only add up the GPU memory a load takes (no allocations, uploads or kernels): --ctx auto
  // arcinferzero expert streaming (standard-type MoE GGUFs): routed experts read from the SSD into an LRU cache on
  // the GPU. Cache size: expert_cache_gb when > 0, else expert_cache_pct (0..100) of all experts when > 0, else
  // automatic: what is left after the rest of the model while the process stays within RAM - windows_reserve_gb
  // and within the GPU's memory share.
  bool stream_experts = true;
  double expert_cache_gb = 0, expert_cache_pct = 0, windows_reserve_gb = 14;
  size_t stream_reserve = 0;  // bytes the automatic cache size leaves for GPU memory allocated later (image encoders)
  int slots = 1;  // conversations held at once (server --slots): each further one its own KV caches and states
};

class VisionQwen3VL;

class Engine {
  friend class VisionQwen3VL;
  friend class VisionGemma4V;
  friend class AudioGemma4A;  // the vision encoders use the engine's OpenCL context, kernels and GEMM

 public:
  Hparams hp;
  EngineOptions opt;
  int ctx = 0, max_chunk = 0, pos = 0;
  // rotary position = cache position + rope_shift for text (Qwen M-RoPE: an image of gw x gh merged tokens advances
  // the rotary position by max(gw, gh) instead of gw * gh); kept by the caller with the conversation
  int rope_shift = 0;
  bool profile = false, w8_resident = false, has_mtp = false;
  // weight bytes the GEMVs read (token generation; bin/runz.exe reports GB/s against the measured ~75 GB/s)
  uint64_t rd_bytes = 0;
  static uint64_t raw_bytes(const GpuWeight& w) { return (uint64_t)w.N * (w.K / 256) * gguf_sb_bytes(w.type); }
  std::map<std::string, double> prof_ms;
  std::map<std::string, int> prof_n;
  size_t gpu_bytes = 0;
  double load_ms = 0;

  Engine(const std::string& path, const EngineOptions& o) : opt(o), f_(path), path_(path) {
    if (!cl_.init(true)) throw std::runtime_error("OpenCL init failed");
    ctx = (o.ctx + 15) / 16 * 16;
    load_hparams();
    if (getenv("ARC_V4") && atoi(getenv("ARC_V4")) > 0) opt.v4 = true;
    if (opt.v4 && (hp.gemma || !opt.attn_int8))
      throw std::runtime_error("the 4-bit V cache (--v4) is for Qwen models with int8 attention");
    if (opt.v4 && kv4_opts().find("-DKV4V") != std::string::npos)
      throw std::runtime_error("ARC_KV4 simulates a 4-bit V inside the int8 cache: not together with --v4 (K only: ARC_KV4=K32)");
    // (MoE, Qwen3.6 35B: 1 and 2 drafts equally fast, 20.2 vs 20.3 tok/s; 1 keeps one DeltaNet state copy less)
    // (measured on arcinferzero, server answers at temperature 0, 2026-10-16: 35B 1 -> 3 drafts 23.9 -> 26.1 tok/s,
    //  4B 2 -> 3 22.9 -> 24.8; 9B 2 and 3 alike (15.8 / 15.6); 27B 3 = 5 = 7; Gemma's drafter unchanged at 2)
    if (!opt.dflash.empty()) opt.mtp = true;
    if (!opt.mtp_block && opt.dflash.empty()) {  // (no drafter to draft instead)
      if (opt.mtp) fprintf(stderr, "arcinferzero: --draft dflash without a DFlash2 drafter: the MTP block drafts\n");
      opt.mtp_block = true;
    }
    if (opt.mtp_drafts <= 0) opt.mtp_drafts = !opt.dflash.empty() ? 7 : mtp_block_drafts();
    auto t0 = std::chrono::steady_clock::now();
    plan_weights();
    stream_ = hp.moe_std && opt.stream_experts && !getenv("ARC_ZERO_NO_STREAM");
    if (f_.has("arcinferzero.package") && !stream_)  // (a package keeps its experts only in the expert file)
      throw std::runtime_error("this is an arcinferzero package: it runs with expert streaming only (no --no-stream)");
    if (stream_ && !opt.dry) open_experts();
    // (streamed experts: 8192-token chunks, each chunk reads a layer's missing experts once and decodes them once;
    // Qwen3.6 35B at ctx 262K, chunk 2048 / 4096 / 8192: expert cache 83 / 81 / 77 %, 32K prompt 485 / 533 / 604
    // tok/s, 128K prompt 284 / 306 / 324, generation at 128K 13.0 / 12.8 / 12.6 tok/s)
    // (Flash Next: 2048, its residual is 4 streams wide)
    max_chunk = o.chunk > 0 ? o.chunk : stream_ ? (hp.hc ? 2048 : 8192) : hp.moe ? 2048 : (w8_resident ? 512 : 4096);
    max_chunk = (std::max(max_chunk, o.min_chunk) + 127) / 128 * 128;
    q2_init();
    if (!opt.dry) build_kernels();
    load_weights();
    {
      const size_t g0 = gpu_bytes;
      alloc_buffers();
      mem_bufs_ = gpu_bytes - g0;
    }
    if (opt.slots > 1) add_slots(opt.slots);  // (before the expert cache takes what is left)
    if (xc_) size_expert_cache();
    // (--ctx auto: room for an expert cache of 90 % of the experts, as far as what the weights and buffers leave
    //  allows with 128K tokens of context (32K until 18 Oct), or at least 3/4 of it. The context took the memory
    //  before (35B: 254K tokens and a 60 % cache: generation 33 / 19 instead of 43 / 23 tok/s, with pauses for the
    //  SSD); Flash Next, whose experts are 3x the GPU share, was left at the minimum context of 4096. Context sweep
    //  18 Oct (docs/speed-progress.md): 35B UD-IQ4_XS at 131K keeps 81 % of the experts and answers as fast as at
    //  64K; Flash Next loses 2-6 % from 32K to 131K.)
    if (stream_ && opt.dry) {
      const double left = (double)device_mem() - 1.17 * 1073741824.0 - (double)gpu_bytes + (double)ctx_bytes(ctx) -
                          (double)snapshot_bytes() - (double)opt.stream_reserve;
      // (--slots N: 128K of context in all, each slot 128K / N; the floor of 3/4 only as far as it leaves every slot
      //  32K: with a plain floor Gemma 4 26B-A4B kept 8,192 tokens per slot, its sliding-window rings ~1.4 GB each.
      //  One slot: as before)
      const int ns = std::max(opt.slots, 1), per = (131072 / ns + 4095) / 4096 * 4096;
      auto ctx_all = [&](int c) { return (double)ctx_bytes(c) + (double)(ns - 1) * (double)slot_bytes(c); };
      const double room = std::max(left - ctx_all(per), std::min(0.75 * left, left - ctx_all(kSlotCtxFloor)));
      gpu_bytes += (size_t)std::max(0.0, std::min(0.9 * (double)stream_bytes_, room));
    }
    mem_load_ = gpu_bytes;
    // the GGUF pages read while loading leave the working set (standby: Windows reuses them when it needs RAM)
    SetProcessWorkingSetSize(GetCurrentProcess(), (SIZE_T)-1, (SIZE_T)-1);
    clFinish(cl_.queue);
    load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    reset();
  }

  // Frees everything on the GPU (the server switches models in one process).
  ~Engine() {
    if (cl_.queue) clFinish(cl_.queue);
    for (auto& [m, p] : orig_ptrs_) m->ptr = p;  // (slots: each buffer object frees its own memory)
    for (auto& kv : kcache_) clReleaseKernel(kv.second);
    for (cl_program p : progs_) clReleaseProgram(p);
    for (auto it = mems_.rbegin(); it != mems_.rend(); ++it)  // sub-buffers (created later) before their parents
      if (*it) clReleaseMemObject(*it);
    if (cl_.queue) clReleaseCommandQueue(cl_.queue);
    if (cl_.ctx) clReleaseContext(cl_.ctx);
  }
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  // Clears the KV cache and recurrent states.
  void reset() {
    pos = 0;
    written_hi_ = 0;
    swa_lo_ = 0;
    pend_toks_.clear();
    last_drafts_.clear();  // (a verification pass never settled, a batch that threw: nothing left of them)
    vrows_ = dfl_used_ = spec_ = false;
    batch_.clear();
    dfl_cap_lo_ = dfl_cap_base_ = dfl_cap_end_ = 0;
    if (hlast_) fill(hlast_, (size_t)hp.n_embd * 4);
    for (auto& L : layers_)
      if (hp.recurrent(L.il)) {
        fill(L.conv_state, conv_bytes());
        fill(L.S, (size_t)hp.ssm_nv * hp.ssm_d_state * hp.ssm_d_state * 4);
      }
    clFinish(cl_.queue);
  }

  // Feeds tokens at the current position; returns the logits for the last token.
  const std::vector<float>& forward(const int* tokens, int n) {
    if (pos + n > ctx) throw std::runtime_error("context overflow");
    if (has_mtp) flush_mtp_pending();
    if (hp.ple_layer >= 0) ple_sync();
    struct DflCommit { Engine& e; int p0; ~DflCommit() { if (e.dfl_) e.dfl_commit_prompt(p0); } } dfl_commit{*this, pos};
    if (dfl_) dfl_window(pos, n);
    if ((!w8_resident || stream_) && n > max_chunk && !getenv("ARC_CHUNK_MAJOR") && !hp.n_ple) {
      // layer-major in segments of up to lm_seg_ tokens (each matrix is decoded once per segment)
      prog_total_ = n;
      for (int s0 = 0; s0 < n; s0 += lm_seg_) {
        prog_base_ = s0;
        const auto& l = forward_layer_major(tokens + s0, std::min(lm_seg_, n - s0));
        if (s0 + lm_seg_ >= n) {
          prog_base_ = prog_total_ = 0;
          return l;
        }
      }
    }
    const bool by_layer = cancel && !w8_resident;  // a chunk takes seconds: stop after any layer
    for (int c0 = 0; c0 < n; c0 += max_chunk) {
      if (c0 > 0 && cancelled()) throw Cancelled();  // between chunks the state is complete
      const int T = std::min(max_chunk, n - c0);
      // pad to a multiple of 128 tokens so GEMM work-groups always get 8 token tiles (rows beyond T are zero)
      const int Mp = T >= 128 ? std::min((T + 127) / 128 * 128, max_chunk) : (T + 15) / 16 * 16;
      if (by_layer) keep_block();
      embed(tokens + c0, T, Mp);
      for (auto& L : layers_) {
        run_layer(L, T, Mp);
        if (by_layer && &L != &layers_.back() && cancelled()) cancel_block(pos);
      }
      pos += T;
      if (mtp_block_) mtp_prefill(tokens + c0, T, Mp, pos - T);
      if (c0 + T >= n) head(T);
      if (profile) collect_profile();
    }
    CL_CHECK(clEnqueueReadBuffer(cl_.queue, logits_, CL_TRUE, 0, (size_t)hp.n_vocab * 4, logits_host_.data(), 0,
                                 nullptr, nullptr));
    collect_profile();
    return logits_host_;
  }

  // Prompt processing with just-in-time weights: every chunk goes through a layer before the next layer starts,
  // so each matrix is decoded once per prompt instead of once per chunk. The residual stream of all chunks lives
  // in x_all_ (x_ points at the current chunk's view).
  const std::vector<float>& forward_layer_major(const int* tokens, int n) {
    const int pos0 = pos, nch = (n + max_chunk - 1) / max_chunk;
    struct SegN { int& v; ~SegN() { v = 0; } } seg_n{seg_n_};  // (the experts' deep prefetch, see stream_layer)
    seg_n_ = n;
    cl_mem x_saved = x_;
    auto chunk_T = [&](int c) { return std::min(max_chunk, n - c * max_chunk); };
    auto chunk_Mp = [&](int c) {
      const int T = chunk_T(c);
      return T >= 128 ? std::min((T + 127) / 128 * 128, max_chunk) : (T + 15) / 16 * 16;
    };
    if (cancel) {
      if (cancelled()) throw Cancelled();  // (between segments)
      keep_block();
    }
    cl_mem ple_saved = ple_emb_;
    struct PleBack { cl_mem& p; cl_mem s; ~PleBack() { p = s; } } ple_back{ple_emb_, ple_saved};
    for (int c = 0; c < nch; ++c) {
      x_ = x_views_[c];
      if (!ple_views_.empty()) ple_emb_ = ple_views_[c];
      embed(tokens + (size_t)c * max_chunk, chunk_T(c), chunk_Mp(c));
    }
    for (size_t li = 0; li < layers_.size(); ++li) {
      Layer& L = layers_[li];
      for (int c = 0; c < nch; ++c) {
        x_ = x_views_[c];
        if (!ple_views_.empty()) ple_emb_ = ple_views_[c];
        cur_T_ = chunk_T(c);
        pos = pos0 + c * max_chunk;
        run_layer(L, cur_T_, chunk_Mp(c));
      }
      if (progress) {
        const int total = prog_total_ ? prog_total_ : n;
        progress(prog_base_ + (int)((double)n * (li + 1) / layers_.size()), total);
      }
      if (profile) collect_profile();
      if (cancel && li + 1 < layers_.size() && cancelled()) {
        x_ = x_saved;
        cancel_block(pos0);
      }
    }
    pos = pos0 + n;
    if (mtp_block_)
      for (int c = 0; c < nch; ++c) {
        x_ = x_views_[c];
        mtp_prefill(tokens + (size_t)c * max_chunk, chunk_T(c), chunk_Mp(c), pos0 + c * max_chunk);
      }
    x_ = x_views_[nch - 1];
    cur_T_ = chunk_T(nch - 1);
    head(cur_T_);
    // leave the last chunk in x_ as the chunk-by-chunk path does (logits_at() reads its rows)
    CL_CHECK(clEnqueueCopyBuffer(cl_.queue, x_, x_saved, 0, 0, (size_t)chunk_Mp(nch - 1) * hp.hce() * 4, 0, nullptr,
                                 nullptr));
    x_ = x_saved;
    CL_CHECK(clEnqueueReadBuffer(cl_.queue, logits_, CL_TRUE, 0, (size_t)hp.n_vocab * 4, logits_host_.data(), 0,
                                 nullptr, nullptr));
    collect_profile();
    return logits_host_;
  }

  // Feeds n rows of precomputed input embeddings (an image) at the current position as one chunk; with
  // non_causal the rows attend to each other in both directions in the sliding-window layers. Returns the logits
  // after the last row when `logits` is set.
  // With rpos (3 ints per row: t, h, w) the rows get those rotary positions (Qwen M-RoPE image tokens).
  const std::vector<float>& forward_embd(const float* embd, int n, bool non_causal, bool logits,
                                         const int* rpos = nullptr) {
    if (n <= 0 || n > max_chunk) throw std::runtime_error("forward_embd: 1..max_chunk rows");
    if (rpos) {
      if (!rpos_) rpos_ = buf((size_t)max_chunk * 3 * 4);
      CL_CHECK(clEnqueueWriteBuffer(cl_.queue, rpos_, CL_TRUE, 0, (size_t)n * 3 * 4, rpos, 0, nullptr, nullptr));
      use_rpos_ = 1;
    }
    struct NoRpos { int& u; ~NoRpos() { u = 0; } } no_rpos{use_rpos_};
    if (pos + n > ctx) throw std::runtime_error("context overflow");
    const int Mp = n >= 128 ? std::min((n + 127) / 128 * 128, max_chunk) : (n + 15) / 16 * 16;
    cur_T_ = n;
    std::fill(x_host_.begin(), x_host_.begin() + (size_t)Mp * hp.n_embd, 0.0f);
    std::copy(embd, embd + (size_t)n * hp.n_embd, x_host_.begin());
    CL_CHECK(clEnqueueWriteBuffer(cl_.queue, x_, CL_TRUE, 0, (size_t)Mp * hp.n_embd * 4, x_host_.data(), 0, nullptr,
                                  nullptr));
    if (hp.n_ple) {
      // E2B / E4B: every embedding row takes the per-layer token embedding of token 0 (padding), combined as usual
      // with the projection of the row itself (llama.cpp models/gemma4.cpp, TAG_GEMMA4_IMG_PADDING)
      const size_t np = (size_t)hp.n_ple * hp.n_layer;
      std::vector<float> pad(np);
      dequant_row(ple_tok_embd_->type, ple_tok_embd_->data, (int)np, pad.data());
      const float sc = std::sqrt((float)hp.n_ple);
      for (size_t t = 0; t < (size_t)n; ++t)
        for (size_t i = 0; i < np; ++i) ple_host_[t * np + i] = pad[i] * sc;
      CL_CHECK(clEnqueueWriteBuffer(cl_.queue, ple_tok_, CL_TRUE, 0, (size_t)n * np * 4, ple_host_.data(), 0, nullptr,
                                    nullptr));
      ple_inputs(n, Mp);
    }
    if (non_causal) { nc_lo_ = pos; nc_hi_ = pos + n - 1; }
    for (auto& L : layers_) run_layer(L, n, Mp);
    nc_lo_ = nc_hi_ = -1;
    pos += n;
    if (logits) {
      head(n);
      CL_CHECK(clEnqueueReadBuffer(cl_.queue, logits_, CL_TRUE, 0, (size_t)hp.n_vocab * 4, logits_host_.data(), 0,
                                   nullptr, nullptr));
    }
    collect_profile();
    return logits_host_;
  }

  // Optional progress callback for long prompts: (tokens-equivalent done, total).
  std::function<void(int, int)> progress;
  // Optional cancellation of prompt processing (the server: its client went away), polled between chunks and, with
  // just-in-time weights (where a block of the prompt takes seconds to minutes), after every layer. forward() then
  // throws Cancelled with the engine at the start of the block it was in: complete blocks stay processed. The
  // DeltaNet states come back from a host copy taken at the block start (27B: 157 MB, ~20 ms per block); Gemma moves
  // back while its sliding-window rings allow it (blocks up to max_chunk + 4096 tokens), else it is reset. `pos`
  // tells where it stands. Not for generation (the copy would cost every token).
  std::function<bool()> cancel;
  struct Cancelled {};

  const GgufFile& gguf() const { return f_; }
  // Runs the generation kernels once so that a first request does not wait for their just-in-time compilation: a
  // short prompt and a token, then with MTP / DFlash2 drafts and a verification pass of every row count (DFlash2 and
  // the MTP block each); leaves the engine reset. Only when this model, these options and these kernel sources have
  // not been warmed before (a marker in the kernel cache): after an update the first requests took seconds per new
  // kernel (27B with DFlash2: a 27-token prompt 37 s, then 1-3 tok/s), later starts skip it. Returns true when it ran.
  bool warm_up_once(int tok) {
    const std::string& dir = cl_.cache_dir;
    char hex[17];
    snprintf(hex, sizeof hex, "%016llx",
             (unsigned long long)ClRt::fnv1a(gemm_src_ + "\x01" + ops_src_ + "\x01" + dec_src_ + "\x01" + moe_src_ + "\x01" + f_.path +
                                             "|" + std::to_string(opt.mtp) + "|" + opt.dflash + (opt.mtp_block ? "" : "|nomtp") +
                                             "|" + std::to_string(opt.mtp_drafts) +
                                             "|" + std::to_string(max_chunk) + (opt.v4 ? "|v4" : "") + "|" + cl_.device_id));
    const std::string marker = dir.empty() ? std::string() : dir + "\\warm-" + hex;
    if (!marker.empty() && std::filesystem::exists(marker)) {
      forward(&tok, 1);  // (the kernels are in the cache: building them is quick)
      reset();
      return false;
    }
    std::vector<int> t(std::max(1, std::min(32, ctx - 12)), tok);
    forward(t.data(), (int)t.size());
    forward(&tok, 1);
    if (has_mtp && !gemma_mtp_) {
      for (int round = 0; round < (dfl_ ? 2 : 1); ++round) {
        for (int k = 1; k <= max_drafts() && pos + k + 2 < ctx; ++k) {
          dfl_ema_ = round == 0 ? 4.0 : 0.0;  // (round 0 with DFlash2: its drafts; round 1: the MTP block's)
          dfl_wait_ = 0;
          dfl_gap_ = 1 << 16;
          mtp_draft(tok, k);
          verify(tok, std::vector<int>(k, tok));
          verify_done(0);
        }
      }
    }
    reset();
    dfl_ema_ = 4.0;
    dfl_wait_ = 0;
    dfl_gap_ = 8;
    dfl_probe_ = false;
    if (!marker.empty()) {
      std::error_code ec;
      std::filesystem::create_directories(dir, ec);
      if (FILE* f = fopen(marker.c_str(), "w")) fclose(f);
    }
    return true;
  }

  // Moves back to position P to reuse a cached prefix (models without recurrent state, i.e. Gemma 4): possible
  // while the sliding-window rings still hold the n_swa positions before P. Qwen3.5 uses snapshot()/restore().
  bool rewind(int P) {
    if (!hp.gemma || P < 0 || P > pos) return false;
    if (written_hi_ - 1 - swa_cap_ >= P - hp.n_swa) return false;  // those slots were overwritten since
    if (swa_lo_ > 0 && P - hp.n_swa < swa_lo_) return false;  // a loaded conversation keeps K/V from there
    pos = P;
    return true;
  }

  // Logits predicting the token after row `row` of the last processed chunk (for perplexity evaluation).
  const std::vector<float>& logits_at(int row) {
    head_rows(row, 1, out_norm_, nullptr);
    CL_CHECK(clEnqueueReadBuffer(cl_.queue, logits_, CL_TRUE, 0, (size_t)hp.n_vocab * 4, logits_host_.data(), 0,
                                 nullptr, nullptr));
    return logits_host_;
  }
  // Recurrent-state snapshots for prefix reuse: the attention KV cache is position-indexed, so restoring
  // the DeltaNet states (and the position) is enough to resume from an earlier point of the sequence.
  void snapshot(int slot) {
    if (slot >= (int)snaps_.size()) snaps_.resize(slot + 1);
    Snap& s = snaps_[slot];
    const size_t sb = (size_t)hp.ssm_nv * hp.ssm_d_state * hp.ssm_d_state * 4, cb = conv_bytes();
    if (s.S.empty())
      for (auto& L : layers_)
        if (hp.recurrent(L.il)) { s.S.push_back(buf(sb)); s.conv.push_back(buf(cb)); }
    size_t k = 0;
    for (auto& L : layers_)
      if (hp.recurrent(L.il)) {
        CL_CHECK(clEnqueueCopyBuffer(cl_.queue, L.S, s.S[k], 0, 0, sb, 0, nullptr, nullptr));
        CL_CHECK(clEnqueueCopyBuffer(cl_.queue, L.conv_state, s.conv[k], 0, 0, cb, 0, nullptr, nullptr));
        ++k;
      }
    if (has_mtp) {
      if (!s.h) s.h = buf((size_t)hp.n_embd * 4);
      flush_mtp_pending();
      CL_CHECK(clEnqueueCopyBuffer(cl_.queue, hlast_, s.h, 0, 0, (size_t)hp.n_embd * 4, 0, nullptr, nullptr));
    }
    s.pos = pos;
  }
  int snapshot_pos(int slot) const { return slot < (int)snaps_.size() ? snaps_[slot].pos : -1; }
  void drop_snapshot(int slot) { if (slot < (int)snaps_.size()) snaps_[slot].pos = -1; }
  bool restore(int slot) {
    if (snapshot_pos(slot) < 0) return false;
    Snap& s = snaps_[slot];
    const size_t sb = (size_t)hp.ssm_nv * hp.ssm_d_state * hp.ssm_d_state * 4, cb = conv_bytes();
    size_t k = 0;
    for (auto& L : layers_)
      if (hp.recurrent(L.il)) {
        CL_CHECK(clEnqueueCopyBuffer(cl_.queue, s.S[k], L.S, 0, 0, sb, 0, nullptr, nullptr));
        CL_CHECK(clEnqueueCopyBuffer(cl_.queue, s.conv[k], L.conv_state, 0, 0, cb, 0, nullptr, nullptr));
        ++k;
      }
    if (has_mtp) {
      CL_CHECK(clEnqueueCopyBuffer(cl_.queue, s.h, hlast_, 0, 0, (size_t)hp.n_embd * 4, 0, nullptr, nullptr));
      pend_toks_.clear();
    }
    pos = s.pos;
    return true;
  }

  // ---------------------------------------------------------------- sizing the context (server --ctx auto)
  // GPU bytes that depend on the context size (same formulas as the allocations: the KV caches of load_attn and
  // load_weights_gemma4, the decode partials and the layer-major residual stream of alloc_buffers); the rest of
  // gpu_bytes does not change with ctx.
  // (the part of the context's memory that is working buffers, allocated with the others: decode partials, the
  // residual stream of layer-major prompt segments)
  size_t ctx_buf_bytes(int c) const {
    c = (c + 15) / 16 * 16;
    size_t n = 4 * (size_t)std::max(c / 256 + 1, 512) * hp.n_head * (hp.head_dim + 2) * 4;  // apart_
    if (hp.n_qsa) n += (size_t)256 * qsa_blocks(c) * 4 + (size_t)max_chunk * qsa_mask_words(c) * 4;  // QSA scores, masks
    if (!w8_resident || stream_) {
      const int seg = lm_seg_for(c);
      n += (size_t)(seg + max_chunk) * (hp.hce() + (hp.ple_layer >= 0 ? hp.n_embd : 0)) * 4;  // x_all_ (+ PLE rows)
    }
    return n;
  }
  // KV cache bytes of the MTP block's attention layer (allocated with the block)
  // Qwen KV bytes per position and kv head: K and V int8 + their scales (--v4: + the 4-bit V copy and its scales)
  size_t kv_pos_bytes() const {
    if (!opt.attn_int8) return 4 * (size_t)hp.head_dim;  // (--attn-f32: f16 K and V rows, load_attn's kc / vc)
    return 2 * (size_t)hp.head_dim + 8 + (opt.v4 ? hp.head_dim / 2 + 4 * (hp.head_dim / 64) : 0);
  }
  size_t mtp_kv_bytes(int c) const {
    c = (c + 15) / 16 * 16;
    return mtp_block_ ? (size_t)hp.n_head_kv * c * kv_pos_bytes() : 0;
  }
  // GPU memory by part for a context of c tokens (as loaded, this engine's ctx; a dry load gives the same numbers):
  // weights (incl. DeltaNet states), MTP block / drafter (with its state copies, without its KV), working buffers,
  // the context (KV caches + the buffers that grow with it). Encoders loaded later: gpu_bytes - loaded_bytes().
  struct MemParts { double weights = 0, mtp = 0, buffers = 0, context = 0; };
  MemParts mem_parts(int c) const {
    MemParts m;
    const double here = (double)ctx_bytes(ctx), bufc = (double)ctx_buf_bytes(ctx), mkv = (double)mtp_kv_bytes(ctx);
    m.mtp = std::max(0.0, (double)mem_mtp_ - mkv);
    m.buffers = std::max(0.0, (double)mem_bufs_ - bufc);
    m.weights = std::max(0.0, (double)mem_load_ - here - m.mtp - m.buffers);
    m.context = (double)ctx_bytes(c);
    return m;
  }
  size_t loaded_bytes() const { return mem_load_; }
  size_t ctx_bytes(int c) const {
    c = (c + 15) / 16 * 16;
    size_t n = ctx_buf_bytes(c);
    if (hp.gemma) {
      const int swa_cap = hp.n_layer_kv < hp.n_layer ? c : std::min(c, (hp.n_swa + max_chunk + 4096 + 255) / 256 * 256);
      for (const Layer& L : layers_)
        if (L.kv_src < 0) {
          const size_t cap = L.swa ? swa_cap : c;
          n += 2 * (size_t)L.nkv * cap * L.hd + 2 * (size_t)L.nkv * cap * (L.hd / 64) * 4;
          if (q2_) n += (q2_kf32_ ? 6 : 4) * (size_t)L.nkv * cap * L.hd;  // (two-pass mode: attn_fa_g's K / V rows)
        }
    } else {
      int nl = mtp_block_ ? 1 : 0;
      for (const Layer& L : layers_) nl += !hp.recurrent(L.il);
      n += (size_t)nl * hp.n_head_kv * c * kv_pos_bytes();
      n += (size_t)hp.n_qsa * qsa_blocks(c) * hp.qsa_dim * 4;  // QSA pooled keys
    }
    return n;
  }
  // GPU bytes of the server's two recurrent-state snapshots
  size_t snapshot_bytes() const {
    size_t n = 0;
    for (const Layer& L : layers_)
      if (hp.recurrent(L.il)) n += state_bytes() + conv_bytes();
    return 2 * (n + (has_mtp ? (size_t)hp.n_embd * 4 : 0));
  }
  uint64_t device_mem() const {
    cl_ulong v = 0;
    clGetDeviceInfo(cl_.device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(v), &v, nullptr);
    if (stream_) {  // (streamed experts: the process also stays within RAM - the Windows reserve)
      const uint64_t ram = ram_total(), res = (uint64_t)(opt.windows_reserve_gb * 1073741824.0);
      v = std::min<uint64_t>(v, ram > res ? ram - res : 0);
    }
    return v;
  }

  // The first position rewind() accepts (Gemma): the sliding-window rings must still hold the n_swa positions before it.
  int rewind_floor() const {
    return hp.gemma ? std::max({0, written_hi_ - swa_cap_ + hp.n_swa, swa_lo_ > 0 ? swa_lo_ + hp.n_swa : 0}) : 0;
  }

  // A recurrent-state snapshot kept in host memory (no GPU memory): the server takes it where every conversation
  // of a client shares the prompt (the end of the system prompt), so a new conversation resumes there.
  void snapshot_host() {
    hsnap_.S.clear();
    for (auto& L : layers_)
      if (hp.recurrent(L.il)) {
        hsnap_.S.push_back(read_buf(L.S, state_bytes()));
        hsnap_.S.push_back(read_buf(L.conv_state, conv_bytes()));
      }
    if (has_mtp) {
      flush_mtp_pending();
      hsnap_.h = read_buf(hlast_, (size_t)hp.n_embd * 4);
    }
    hsnap_.pos = pos;
  }
  int host_snapshot_pos() const { return hsnap_.pos; }
  void drop_host_snapshot() { hsnap_ = HostSnap(); }
  bool restore_host() {
    if (hsnap_.pos < 0) return false;
    size_t k = 0;
    for (auto& L : layers_)
      if (hp.recurrent(L.il)) {
        write_buf(L.S, hsnap_.S[k++]);
        write_buf(L.conv_state, hsnap_.S[k++]);
      }
    if (has_mtp) {
      write_buf(hlast_, hsnap_.h);
      pend_toks_.clear();
    }
    pos = hsnap_.pos;
    return true;
  }

  // ---------------------------------------------------------------- conversation slots (server --slots)
  // Several conversations held at once, one step at a time: each slot has its own KV caches, DeltaNet and conv
  // states (with Flash Next's PLE history and QSA tails inside), QSA pooled keys, the MTP block's KV cache and
  // hidden states, DFlash2's context rings, snapshots, position and drafter choice. bind(s) points the engine's state
  // buffers at slot s's memory without copying: the buffer objects stay the same, only their device addresses
  // change, and kernel arguments are set at every launch. Weights, working buffers and the expert cache are shared.
  // Bind only between steps (a verification pass settled with verify_done()).
  int slot_count() const { return std::max(1, (int)slot_ptrs_.size()); }
  // the drafter choice back to its start (DFlash2 first): a conversation that starts over in a slot
  void reset_drafter_choice() {
    dfl_ema_ = 4.0;
    dfl_wait_ = 0;
    dfl_gap_ = 8;
    dfl_probe_ = false;
  }
  int bound_slot() const { return slot_; }
  static constexpr int kMaxSlots = 8;
  static constexpr int kSlotCtxFloor = 32768;  // (--ctx auto, slots of streamed MoE models: context kept per slot)
  void add_slots(int n) {
    if (n > kMaxSlots) throw std::out_of_range("add_slots: at most " + std::to_string(kMaxSlots) + " slots");
    if (n <= 1 || !slot_ptrs_.empty() || opt.dry) return;
    std::set<cl_mem> seen;
    auto add = [&](cl_mem& f) {
      if (f && seen.insert(f).second) roles_.push_back(&f);
    };
    // (with the verification pass's state, so that a slot's drafts can wait for verify_done while other slots run:
    //  S_next and the inputs verify_done replays, the pass's hidden states, its DFlash2 rows)
    auto add_layer = [&](Layer& L) {
      for (cl_mem* f : {&L.kq, &L.vq, &L.ksc, &L.vsc, &L.vq4, &L.vsc4, &L.conv_state, &L.S, &L.pk, &L.S_next, &L.conv_pre,
                        &L.xin, &L.qkvin, &L.abin})
        add(*f);
      if (!opt.attn_int8 || q2_) { add(L.kc); add(L.vc); }
    };
    for (auto& L : layers_) add_layer(L);
    if (mtp_block_) add_layer(mtp_);
    add(hlast_);
    add(hpend_);
    add(hver_);
    for (auto& D : dfl_l_) { add(D.rk); add(D.rv); }
    if (dfl_) {
      dfl_vrows_rows_ = batch_rows_;
      dfl_vrows_ = buf((size_t)dfl_vrows_rows_ * hp.n_embd * 4);
      add(dfl_vrows_);
    }
    // views into those buffers (the PLE history and QSA tails in a recurrent state) follow their buffer
    for (ZMem* m : zrt::mems) {
      if (m->owner) continue;
      for (size_t i = 0; i < roles_.size(); ++i) {
        const ZMem* r = *roles_[i];
        if (m != r && m->ptr >= r->ptr && m->ptr < r->ptr + r->size) {
          subs_.push_back({m, (int)i, (size_t)(m->ptr - r->ptr)});
          break;
        }
      }
    }
    // the addresses the buffer objects had (put back before they are freed); S_next trades places with S
    for (cl_mem* r : roles_) orig_ptrs_.push_back({*r, (*r)->ptr});
    for (auto& v : subs_) orig_ptrs_.push_back({v.m, v.m->ptr});
    {  // (slot_bytes, which --ctx auto counts with, must be what a slot really takes)
      size_t per = 0;
      for (cl_mem* r : roles_) per += (*r)->size;
      const size_t est = slot_bytes(ctx) - snapshot_bytes();
      slot_real_bytes_ = per;
      slot_est_bytes_ = est;
      if (per > est + est / 100 || getenv("ARC_SLOT_BYTES"))
        fprintf(stderr, "arcinferzero: a slot takes %.1f MiB, slot_bytes() counts %.1f MiB\n", per / 1048576.0, est / 1048576.0);
    }
    slot_ptrs_.assign(n, {});
    slot_host_.assign(n, SlotHost());
    for (cl_mem* r : roles_) slot_ptrs_[0].push_back((*r)->ptr);
    for (int s = 1; s < n; ++s)
      for (cl_mem* r : roles_) {
        cl_mem b = buf((*r)->size);
        fill(b, (*r)->size);
        slot_ptrs_[s].push_back(b->ptr);
      }
    clFinish(cl_.queue);
  }
  void bind(int s) {
    if (s < 0 || s >= slot_count()) throw std::out_of_range("bind: slot " + std::to_string(s) + " out of range");
    if (s == slot_) return;
    if (!last_drafts_.empty() && !vrows_) throw std::runtime_error("bind: a verification pass is not settled");
    clFinish(cl_.queue);  // (cancellation reads and host snapshots work on the bound buffers)
    park();
    unpark(s);
  }

  // ---------------------------------------------------------------- batched decoding over slots (QUEUE 20.4)
  // One generation step for several slots at once: row b is the next token of slot slots[b] (at that slot's position).
  // The parts that work row by row (norms, every projection and FFN / MoE GEMV, the output head) run once for all rows,
  // so the weights are read once per step instead of once per slot; the parts that follow a sequence (DeltaNet conv
  // and recurrence, RoPE and the K/V write, attention) run per slot on its rows with its state (point_to). Then each
  // slot's MTP block KV / hidden state and DFlash2 context take their row. Returns the logits rows (row b: slot b).
  // Dense and MoE Qwen models (not Flash Next or Gemma); every slot's drafts settled; up to 1 + max(drafts, slots - 1)
  // rows. The bound slot stays bound.
  bool batch_ok() const { return !hp.hc && opt.attn_int8 && slot_ptrs_.size() > 1; }
  bool batch_verify_ok() const { return batch_ok() && has_mtp && !gemma_mtp_; }
  const std::vector<float>& forward_batch(const std::vector<int>& slots, const std::vector<int>& toks) {
    const int B = (int)slots.size();
    if (!batch_ok() || B < 1 || B != (int)toks.size() || B > batch_rows_)
      throw std::runtime_error("forward_batch: not for this model or too many rows");
    check_batch(slots, std::vector<int>(B, 1));
    BatchGuard guard{*this, slot_};
    for (int b = 0; b < B; ++b) {  // (pending MTP rows of accepted drafts first, each in its slot)
      bind(slots[b]);
      if (has_mtp) flush_mtp_pending();
    }
    bind(guard.bound);
    clFinish(cl_.queue);
    park();
    guard.parked = true;
    batch_.clear();
    for (int b = 0; b < B; ++b) {
      const SlotHost& h = slot_host_[slots[b]];
      batch_.push_back({slots[b], b, 1, h.pos, h.rope_shift});
    }
    const int Mp = 16;
    embed(toks.data(), B, Mp);
    // (DFlash2: every row's features go to the first accumulator; each slot's ring takes its row below)
    pos = 0;
    dfl_cap_lo_ = dfl_cap_base_ = 0;
    dfl_cap_end_ = B;
    for (auto& L : layers_) run_layer(L, B, Mp);
    fail_point();
    const std::vector<BatchSeq> seqs = batch_;
    batch_.clear();
    head_rows(0, B, out_norm_, nullptr);
    if (batch_logits_host_.size() < (size_t)hp.n_vocab * B) batch_logits_host_.resize((size_t)hp.n_vocab * batch_rows_);
    CL_CHECK(clEnqueueReadBuffer(cl_.queue, logits_, CL_TRUE, 0, (size_t)hp.n_vocab * B * 4, batch_logits_host_.data(), 0,
                                 nullptr, nullptr));
    for (const BatchSeq& q : seqs) {
      point_to(q.slot);
      pos = q.pos;
      rope_shift = q.rs;
      if (mtp_block_) {  // as mtp_prefill for its row: hlast_, then the row's final hidden state, then the MTP block
        const size_t E4 = (size_t)hp.n_embd * 4;
        CL_CHECK(clEnqueueCopyBuffer(cl_.queue, hlast_, hfin_, 0, 0, E4, 0, nullptr, nullptr));
        float eps = hp.eps;
        int E = hp.n_embd, M = Mp, TT = 1, r0 = q.row, wh = 1, a32 = 1;
        set_args(k_rmsnorm_, 0, x_, r0, out_norm_, E, M, TT, eps, aq_, asc_, absum_, hfin1_, wh, a32);
        size_t gws[2] = {256, 1}, lws[2] = {256, 1};
        run("mtp_concat", k_rmsnorm_, 2, gws, lws);
        CL_CHECK(clEnqueueCopyBuffer(cl_.queue, hfin_, hlast_, E4, 0, E4, 0, nullptr, nullptr));
        mtp_rows(&toks[q.row], hfin_, 1, q.pos, false);
      }
      if (has_mtp && hlast_) {  // (as head() leaves it: the normed hidden state of the row; the drafters start there)
        float eps = hp.eps;
        int E = hp.n_embd, M = Mp, TT = 1, r0 = q.row, wh = 1, a32 = 1;
        set_args(k_rmsnorm_, 0, x_, r0, out_norm_, E, M, TT, eps, aq_, asc_, absum_, hlast_, wh, a32);
        size_t gws[2] = {256, 1}, lws[2] = {256, 1};
        run("rmsnorm_quant", k_rmsnorm_, 2, gws, lws);
      }
      if (dfl_) dfl_commit(0, q.row, 1, q.pos);
    }
    clFinish(cl_.queue);
    for (const BatchSeq& q : seqs) slot_host_[q.slot].pos = q.pos + 1;  // (the slots advance once the pass is done)
    dfl_cap_lo_ = dfl_cap_end_ = 0;
    guard.parked = false;
    unpark(guard.bound);
    collect_profile();
    return batch_logits_host_;
  }
  int batch_rows() const { return batch_rows_; }
  // (tests of the error paths: the fail_after-th verification or batched pass from now throws after its layers have
  //  written their state; 0 = never. The server sets it from ARC_FAIL_PASS)
  long fail_after = 0;
  void fail_point() {
    if (fail_after > 0 && --fail_after == 0) throw std::runtime_error("injected failure (ARC_FAIL_PASS)");
  }
  size_t slot_real_bytes_ = 0, slot_est_bytes_ = 0;  // (add_slots: one slot's buffers, and what slot_bytes() counted)
 private:
  // a batch's slots: in range, each once, none with a verification pass not settled, n[b] more positions within ctx
  void check_batch(const std::vector<int>& slots, const std::vector<int>& n) const {
    std::vector<char> seen(slot_count(), 0);
    for (size_t b = 0; b < slots.size(); ++b) {
      const int s = slots[b];
      if (s < 0 || s >= slot_count()) throw std::out_of_range("batch: slot " + std::to_string(s) + " out of range");
      if (seen[s]++) throw std::runtime_error("batch: slot " + std::to_string(s) + " twice");
      const bool bound = s == slot_;
      const bool pending = bound ? !last_drafts_.empty() || vrows_ : !slot_host_[s].last_drafts.empty() || slot_host_[s].vrows;
      if (pending) throw std::runtime_error("batch: slot " + std::to_string(s) + " has a verification pass not settled");
      if ((bound ? pos : slot_host_[s].pos) + n[b] > ctx) throw std::runtime_error("context overflow");
    }
  }
  // a batched pass's state, cleared however the pass ends, and the slot bound before it bound again. A pass that
  // threw leaves its slots' positions and pending drafts as they were (they change only once it is done), but their
  // device state (KV rows past the position, DeltaNet states) may be partly written: the caller starts those
  // conversations over (reset).
  struct BatchGuard {
    Engine& e;
    int bound;
    bool parked = false;
    ~BatchGuard() {
      e.batch_.clear();
      e.spec_ = false;
      e.dfl_cap_lo_ = e.dfl_cap_base_ = e.dfl_cap_end_ = 0;
      if (parked) e.unpark(bound);
      else if (e.slot_ != bound) {  // (bind or flush_mtp_pending threw in the loop before park: host work only,
        e.park();                   // no queue wait or pending-drafts check that could throw again)
        e.unpark(bound);
      }
    }
  };
 public:
  // verify() for several slots in one pass: rows[b] = {token, drafts...} of slot slots[b] (drafts from mtp_draft in
  // that slot). Returns the logits rows of every slot one after the other; then each slot settles its own with
  // verify_done() (bound to it, in any order, other slots' passes may run in between). At most batch_rows() rows
  // (8 or fewer keep the single-token kernels: the rows bit-identical to verify() of each slot alone).
  const std::vector<float>& verify_batch(const std::vector<int>& slots, const std::vector<std::vector<int>>& rows) {
    const int B = (int)slots.size();
    int R = 0;
    for (const auto& r : rows) R += (int)r.size();
    if (!batch_verify_ok() || B < 1 || B != (int)rows.size() || R > batch_rows_ || (dfl_ && R > dfl_vrows_rows_))
      throw std::runtime_error("verify_batch: not for this model or too many rows");
    std::vector<int> n(B);
    for (int b = 0; b < B; ++b) {
      if (rows[b].size() < 2 || (int)rows[b].size() > 1 + max_drafts()) throw std::runtime_error("verify_batch: drafts");
      n[b] = (int)rows[b].size();
    }
    check_batch(slots, n);
    BatchGuard guard{*this, slot_};
    for (int b = 0; b < B; ++b) {
      bind(slots[b]);
      flush_mtp_pending();
    }
    bind(guard.bound);
    clFinish(cl_.queue);
    park();
    guard.parked = true;
    batch_.clear();
    std::vector<int> toks;
    for (int b = 0; b < B; ++b) {
      const SlotHost& h = slot_host_[slots[b]];
      batch_.push_back({slots[b], (int)toks.size(), (int)rows[b].size(), h.pos, h.rope_shift});
      toks.insert(toks.end(), rows[b].begin(), rows[b].end());
    }
    const int Mp = 16;
    embed(toks.data(), R, Mp);
    pos = 0;
    dfl_cap_lo_ = dfl_cap_base_ = 0;
    dfl_cap_end_ = R;
    spec_ = true;
    for (auto& L : layers_) run_layer(L, R, Mp);
    fail_point();
    spec_ = false;
    const std::vector<BatchSeq> seqs = batch_;
    batch_.clear();
    head_rows(0, R, out_norm_, nullptr);
    if (batch_logits_host_.size() < (size_t)hp.n_vocab * R) batch_logits_host_.resize((size_t)hp.n_vocab * batch_rows_);
    CL_CHECK(clEnqueueReadBuffer(cl_.queue, logits_, CL_TRUE, 0, (size_t)hp.n_vocab * R * 4, batch_logits_host_.data(), 0,
                                 nullptr, nullptr));
    const size_t E4 = (size_t)hp.n_embd * 4;
    for (int b = 0; b < B; ++b) {  // each slot's hidden states (verify_done) and DFlash2 rows of the pass
      const BatchSeq& q = seqs[b];
      point_to(q.slot);
      float eps = hp.eps;
      int E = hp.n_embd, M = Mp, nn = q.n, r0 = q.row, wh = 1, a32 = 1;
      set_args(k_rmsnorm_, 0, x_, r0, out_norm_, E, M, nn, eps, aq_, asc_, absum_, hver_, wh, a32);
      size_t gws[2] = {256, (size_t)q.n}, lws[2] = {256, 1};
      run("rmsnorm_quant", k_rmsnorm_, 2, gws, lws);
      if (dfl_)
        CL_CHECK(clEnqueueCopyBuffer(cl_.queue, dfl_acc_[0], dfl_vrows_, (size_t)q.row * E4, 0, (size_t)q.n * E4, 0, nullptr, nullptr));
    }
    std::vector<std::vector<int>> pend(B);  // (allocated before any slot changes: the commit below cannot throw)
    for (int b = 0; b < B; ++b) pend[b].assign(rows[b].begin() + 1, rows[b].end());
    clFinish(cl_.queue);
    for (int b = 0; b < B; ++b) {  // (the slots advance once the pass is done)
      SlotHost& h = slot_host_[seqs[b].slot];
      h.pos = seqs[b].pos + seqs[b].n;
      h.last_drafts.swap(pend[b]);  // (now pending: verify_done settles it)
      h.vrows = true;
    }
    dfl_cap_lo_ = dfl_cap_end_ = 0;
    guard.parked = false;
    unpark(guard.bound);
    collect_profile();
    return batch_logits_host_;
  }

 private:
  // the bound slot's host state and device addresses back into its records
  void park() {
    SlotHost& h = slot_host_[slot_];
    h.pos = pos;
    h.rope_shift = rope_shift;
    h.written_hi = written_hi_;
    h.swa_lo = swa_lo_;
    h.pend = std::move(pend_toks_);
    h.ple_prev[0] = ple_prev_[0];
    h.ple_prev[1] = ple_prev_[1];
    h.snaps = std::move(snaps_);
    h.hsnap = std::move(hsnap_);
    h.dfl_ema = dfl_ema_;
    h.dfl_wait = dfl_wait_;
    h.dfl_gap = dfl_gap_;
    h.dfl_probe = dfl_probe_;
    h.dfl_used = dfl_used_;
    h.vrows = vrows_;
    h.last_drafts = std::move(last_drafts_);
    for (size_t i = 0; i < roles_.size(); ++i) slot_ptrs_[slot_][i] = (*roles_[i])->ptr;
  }
  // slot s's host state and device addresses into the engine
  void unpark(int s) {
    slot_ = s;
    SlotHost& g = slot_host_[s];
    pos = g.pos;
    rope_shift = g.rope_shift;
    written_hi_ = g.written_hi;
    swa_lo_ = g.swa_lo;
    pend_toks_ = std::move(g.pend);
    ple_prev_[0] = g.ple_prev[0];
    ple_prev_[1] = g.ple_prev[1];
    snaps_ = std::move(g.snaps);
    hsnap_ = std::move(g.hsnap);
    dfl_ema_ = g.dfl_ema;
    dfl_wait_ = g.dfl_wait;
    dfl_gap_ = g.dfl_gap;
    dfl_probe_ = g.dfl_probe;
    dfl_used_ = g.dfl_used;
    vrows_ = g.vrows;
    last_drafts_ = std::move(g.last_drafts);
    point_to(s);
    attn_in_ready_ = ffn_in_ready_ = false;
    rt_pos_[0] = rt_pos_[1] = -1;
  }
  // the state buffers at slot s's memory (device addresses only; inside a batched pass, every slot parked)
  void point_to(int s) {
    for (size_t i = 0; i < roles_.size(); ++i) (*roles_[i])->ptr = slot_ptrs_[s][i];
    for (auto& v : subs_) v.m->ptr = (*roles_[v.role])->ptr + v.off;
  }
  // a batched pass: each sequence's rows (slot, first row, rows, position, rotary shift); empty otherwise
  struct BatchSeq { int slot, row, n, pos, rs; };
  std::vector<BatchSeq> batch_;
  bool vrows_ = false;            // the pending verification is a batched one: its DFlash2 rows in dfl_vrows_
  int dfl_vrows_rows_ = 0;
  cl_mem dfl_vrows_ = nullptr;    // (slots with DFlash2: a slot's rows of a batched verification pass)
  int batch_rows_ = 1;
  std::vector<float> batch_logits_host_;
  // a working buffer from byte `off` on (rows of a sequence inside a batched pass), made once
  std::map<std::pair<ZMem*, size_t>, cl_mem> views_;
  cl_mem from(cl_mem b, size_t off) {
    if (!off || !b) return b;
    cl_mem& v = views_[{b, off}];
    if (!v) {
      if (off >= b->size) throw std::runtime_error("from: offset past the buffer");
      for (cl_mem* role : roles_) {  // (a view of a slot's buffer would not follow point_to: working buffers only)
        const ZMem* r = *role;
        if (b->ptr >= r->ptr && b->ptr < r->ptr + r->size) throw std::logic_error("from: a view of a slot's state buffer");
      }
      cl_buffer_region r{off, b->size - off};
      cl_int err;
      v = clCreateSubBuffer(b, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &r, &err);
      CL_CHECK(err);
      mems_.push_back(v);
    }
    return v;
  }

 public:
  // GPU bytes of one more slot with a context of c tokens: the KV caches (and QSA pooled keys), the recurrent states,
  // the MTP block's hidden states, DFlash2's rings, the snapshots
  // (the roles add_slots copies: the KV caches and QSA keys of ctx_bytes; per recurrent layer S and conv_state, with
  //  drafts also S_next, conv_pre, xin, qkvin, abin; hlast_ / hpend_ / hver_; DFlash2's rings and rows)
  size_t slot_bytes(int c) const {
    size_t n = ctx_bytes(c) - ctx_buf_bytes(c);
    const bool spec = opt.mtp && !hp.gemma;
    for (const Layer& L : layers_)
      if (hp.recurrent(L.il)) {
        n += state_bytes() + conv_bytes();
        if (spec)
          n += state_bytes() + conv_bytes() + 2 * (size_t)(1 + kMaxDrafts) * hp.conv_dim() * 4 +
               (size_t)(1 + kMaxDrafts) * 2 * hp.ssm_nv * 4;
      }
    if (has_mtp) n += (size_t)hp.n_embd * 4 * (1 + max_drafts() + 1 + max_drafts());  // (hlast_, hpend_, hver_)
    n += dfl_l_.size() * 2 * (size_t)dfl_R_ * dfl_nkv_ * 128 * 4;
    if (dfl_) n += (size_t)batch_rows_ * hp.n_embd * 4;
    return n + snapshot_bytes();
  }

  // ---------------------------------------------------------------- saved conversations (the server's session cache)
  // Everything a conversation leaves in the engine after `pos` positions: the K/V blocks and scales of every layer
  // that owns a KV cache (full-attention layers: positions [0, pos); sliding-window layers only from n_swa positions
  // before `keep_from`, the earliest position the conversation should be resumable at), the MTP block's cache and
  // hidden state, the DeltaNet states and all snapshots. Device data is copied in batches of up to 256 MB.
  // load_state() returns false (the engine left reset) for a stream from another model or an incompatible layout.
  // Gemma: the first position a conversation saved with keep_from can be resumed at (see rewind()).
  int saved_floor(int keep_from) const {
    return hp.gemma ? std::max(rewind_floor(), swa_save_lo(keep_from) + hp.n_swa) : 0;
  }
  // save_state() writes through `out` (a file, or memory for a write in the background); saved_size() is its length.
  using Sink = std::function<bool(const void*, size_t)>;
  // host_snapshot = false: without the host snapshot (a state saved exactly where it was taken holds it already)
  bool save_state(FILE* f, int keep_from, bool host_snapshot = true) {
    return save_state([f](const void* p, size_t n) { return fwrite(p, 1, n, f) == n; }, keep_from, host_snapshot);
  }
  size_t saved_size(int keep_from) {
    if (!opt.attn_int8) return 0;
    std::vector<int32_t> hdr;
    std::vector<Seg> segs;
    save_layout(keep_from, hdr, segs);
    size_t n = hdr.size() * 4;
    for (const Seg& sg : segs) n += sg.n;
    if (hsnap_.pos >= 0) {
      for (const auto& v : hsnap_.S) n += v.size();
      if (has_mtp) n += hsnap_.h.size();
    }
    return n;
  }
  bool save_state(const Sink& out, int keep_from, bool host_snapshot = true) {
    if (!opt.attn_int8) return false;
    flush_mtp_pending();
    std::vector<int32_t> hdr;
    std::vector<Seg> segs;
    save_layout(keep_from, hdr, segs, host_snapshot);
    bool ok = out(hdr.data(), hdr.size() * 4) && segs_out(segs, out);
    if (ok && host_snapshot && hsnap_.pos >= 0) {
      for (const auto& v : hsnap_.S) ok = ok && out(v.data(), v.size());
      if (has_mtp) ok = ok && out(hsnap_.h.data(), hsnap_.h.size());
    }
    return ok;
  }
  bool load_state(FILE* f) {
    reset();
    for (int i = 0; i < (int)snaps_.size(); ++i) drop_snapshot(i);
    drop_host_snapshot();
    if (!opt.attn_int8) return false;
    const auto kl = kv_layers();
    std::vector<int32_t> hdr(11);
    if (!get(f, hdr.data(), hdr.size() * 4)) return false;
    const int spos = hdr[7], swh = hdr[8], slo = hdr[9];
    if (hdr[0] != kStateMagic || hdr[1] != 2 || hdr[2] != hp.n_layer || hdr[3] != hp.n_embd || hdr[4] != hp.n_vocab ||
        hdr[5] != state_flags() || hdr[6] != (has_mtp ? 1 : 0) || hdr[10] != (int)kl.size() || spos < 0 ||
        spos > ctx || swh > ctx || slo < 0)
      return false;
    std::vector<int32_t> geo(kl.size() * 5);
    if (!get(f, geo.data(), geo.size() * 4)) return false;
    std::vector<std::pair<int, int>> br;
    for (size_t i = 0; i < kl.size(); ++i) {
      const int* q = &geo[i * 5];
      // same heads and scale layout; the blocks must fit this cache (a sliding-window ring: its capacity)
      if (q[0] != kl[i].nkv || q[1] != kl[i].hd || q[2] != kl[i].sb || q[3] < 0 || q[4] < q[3] ||
          q[4] - q[3] > kl[i].cap / 16 || (!kl[i].L->swa && (q[3] != 0 || q[4] > kl[i].cap / 16)))
        return false;
      br.push_back({q[3], q[4]});
    }
    int32_t ns = 0, hpos = -1;
    if (!get(f, &ns, 4) || ns < 0 || ns > 8) return false;
    std::vector<int> sp(ns);
    if (!get(f, sp.data(), (size_t)ns * 4) || !get(f, &hpos, 4)) return false;
    for (int i = 0; i < ns; ++i)
      if (sp[i] >= 0) snap_slot(i);
    bool ok = segs_in(state_segs(kl, br, sp), f);
    if (ok && hpos >= 0) {
      for (auto& L : layers_)
        if (hp.recurrent(L.il)) {
          hsnap_.S.emplace_back(state_bytes());
          hsnap_.S.emplace_back(conv_bytes());
        }
      for (auto& v : hsnap_.S) ok = ok && get(f, v.data(), v.size());
      if (has_mtp) {
        hsnap_.h.resize((size_t)hp.n_embd * 4);
        ok = ok && get(f, hsnap_.h.data(), hsnap_.h.size());
      }
    }
    if (!ok) {
      reset();
      for (int i = 0; i < (int)snaps_.size(); ++i) drop_snapshot(i);
      drop_host_snapshot();
      return false;
    }
    for (int i = 0; i < ns; ++i)
      if (sp[i] >= 0) snaps_[i].pos = sp[i];
    hsnap_.pos = hpos;
    pos = spos;
    written_hi_ = swh;
    swa_lo_ = slo;
    pend_toks_.clear();
    last_drafts_.clear();
    attn_in_ready_ = ffn_in_ready_ = false;
    rt_pos_[0] = rt_pos_[1] = -1;
    return true;
  }

 private:
  struct Layer {
    int il = 0;
    cl_mem attn_norm = nullptr, post_norm = nullptr;
    // Gated DeltaNet
    GpuWeight qkv, z, ssm_out, ab;
    // single tokens: qkv + z (+ ab when they are Q8_0) rows stacked, one GEMV (qkv / z / ab are views of it)
    GpuWeight qz;
    bool qz_ab = false;
    cl_mem dt_bias = nullptr, ssm_a = nullptr, conv_w = nullptr, ssm_norm = nullptr;
    cl_mem conv_state = nullptr, S = nullptr;
    // speculative verification (MTP): the pass writes its states to S_next and keeps S, its inputs and the conv history
    // before it, so verify_done() takes S_next (every draft accepted) or replays the accepted tokens from S
    cl_mem S_next = nullptr, conv_pre = nullptr, xin = nullptr, abin = nullptr, qkvin = nullptr;
    // attention
    GpuWeight wq, wk, wv, wo;
    cl_mem q_norm = nullptr, k_norm = nullptr, kc = nullptr, vc = nullptr, kq = nullptr, ksc = nullptr,
           vq = nullptr, vsc = nullptr, vq4 = nullptr, vsc4 = nullptr;  // (vq4 / vsc4: --v4's copy of V)
    // FFN
    GpuWeight gate, up, down;
    // MoE FFN: router (+ shared-expert gate row), shared expert, routed experts (gate/up/down codes and scales)
    GpuWeight router, gate_sh, up_sh, down_sh;
    // single tokens: router rows (padded to rgu_rows()) + shared-expert gate + up rows, one Q8_0 GEMV (they read
    // the same input; three small GEMVs each paid a launch's fixed cost)
    GpuWeight rgu;
    cl_mem ex_q[3] = {}, ex_s[3] = {};
    uint32_t ex_type[3] = {};  // (standard expert tensors: their GGUF types; ex_q = the GPU raw form)
    std::vector<cl_mem> ex_sub[3];  // views of ex_q per group of moe_group_ experts (prompt GEMM path, made on use)
    // Gemma 4
    bool swa = false, has_v = true;
    int nkv = 0, hd = 0, cap = 0;  // kv heads, head size, KV-cache capacity (positions; ring for sliding window)
    float out_scale = 1.0f;
    cl_mem post_attn = nullptr, ffn_norm = nullptr, post_ffw = nullptr, rope_ff = nullptr;
    int kv_src = -1;            // shared KV: the layer whose cache this layer reads (no K/V of its own)
    // Gemma 4 with experts: the dense FFN's output norm, the experts' input norm and output norm, the router's input
    // norm weights (its scale / sqrt(E)), the scale of each expert
    cl_mem post_ffw1 = nullptr, ffn_norm2 = nullptr, post_ffw2 = nullptr, rnorm = nullptr, escale = nullptr,
           router_f32 = nullptr;
    GpuWeight ple_gate, ple_proj;
    cl_mem ple_post = nullptr;  // per-layer-embedding block
    // Flash Next hyper-connection mixers before the attention / DeltaNet block (0) and the FFN (1): stream norm,
    // low-rank down projection stacked with the inject rows (hc_lr + hc_n, padded), up projection (K padded to 512)
    cl_mem hc_norm[2] = {};
    GpuWeight hc_down[2], hc_up[2];
    // Flash Next PLE block (its layer only): key / value projections, norms, depthwise conv weights [HCE][taps]
    GpuWeight ple_key, ple_value;
    cl_mem ple_nk = nullptr, ple_nq = nullptr, ple_nc = nullptr, ple_cw = nullptr;
    // Flash Next QSA (attention layers): indexer key + query rows stacked, their norms, the pooled block keys
    // ([ctx / qr][dim], by block: position-indexed like the KV cache), the raw keys of the tail block (state)
    int qr = 0, qsa_i = -1;
    GpuWeight ix;
    cl_mem ix_kn = nullptr, ix_qn = nullptr, pk = nullptr, qtail = nullptr;
  };

  struct Snap {
    int pos = -1;
    std::vector<cl_mem> S, conv;
    cl_mem h = nullptr;  // MTP: main hidden state of the last position
  };
  std::vector<Snap> snaps_;
  static constexpr int32_t kStateMagic = 0x53435241;  // "ARCS"
  struct HostSnap {
    int pos = -1;
    std::vector<std::vector<uint8_t>> S;  // per recurrent layer: state, conv state
    std::vector<uint8_t> h;
  };
  HostSnap hsnap_;
  // ---- conversation slots (see bind): the state buffers' fields, each slot's device addresses for them, the views
  // inside them, the addresses the buffer objects were created with, the host state of the slots not bound
  struct SlotHost {
    int pos = 0, rope_shift = 0, written_hi = 0, swa_lo = 0, ple_prev[2] = {-1, -1};
    std::vector<int> pend;
    std::vector<Snap> snaps;
    HostSnap hsnap;
    double dfl_ema = 4.0;
    int dfl_wait = 0, dfl_gap = 8;
    bool dfl_probe = false, dfl_used = false, vrows = false;
    std::vector<int> last_drafts;
  };
  struct SlotView { ZMem* m; int role; size_t off; };
  std::vector<cl_mem*> roles_;
  std::vector<std::vector<char*>> slot_ptrs_;
  std::vector<SlotView> subs_;
  std::vector<std::pair<ZMem*, char*>> orig_ptrs_;
  std::vector<SlotHost> slot_host_;
  int slot_ = 0;
  // ---- cancellation (see cancel)
  std::vector<std::vector<uint8_t>> blk_;  // DeltaNet states at the start of the current block
  bool cancelled() {
    if (!cancel) return false;
    clFinish(cl_.queue);  // the check follows the work done, not just enqueued
    return cancel();
  }
  void keep_block() {
    size_t k = 0;
    for (auto& L : layers_)
      if (hp.recurrent(L.il))
        for (auto [b, n] : {std::pair{L.S, state_bytes()}, std::pair{L.conv_state, conv_bytes()}}) {
          if (k == blk_.size()) blk_.emplace_back(n);
          CL_CHECK(clEnqueueReadBuffer(cl_.queue, b, CL_FALSE, 0, n, blk_[k++].data(), 0, nullptr, nullptr));
        }
    clFinish(cl_.queue);
  }
  // back to p0, the start of the block being processed, then throw Cancelled
  [[noreturn]] void cancel_block(int p0) {
    clFinish(cl_.queue);
    size_t k = 0;
    for (auto& L : layers_)
      if (hp.recurrent(L.il)) {
        write_buf(L.S, blk_[k++]);
        write_buf(L.conv_state, blk_[k++]);
      }
    prog_base_ = prog_total_ = 0;
    if (hp.gemma && !rewind(p0)) reset();  // the rings no longer hold the n_swa positions before p0
    else pos = p0;
    throw Cancelled();
  }
  int swa_lo_ = 0;  // sliding-window layers hold no K/V below this position (a loaded conversation, trimmed)
  size_t state_bytes() const { return (size_t)hp.ssm_nv * hp.ssm_d_state * hp.ssm_d_state * 4; }
  // (DeltaNet conv history of 3 rows; Flash Next: every recurrent layer's state also has room for the PLE block's,
  // used at its layer: ple_hist() rows of the conv input and the last two token ids, so snapshots, saved
  // conversations and cancellation carry it like the other recurrent state)
  int ple_hist() const { return hp.ple_layer >= 0 ? (hp.ple_taps - 1) * hp.ple_ng : 0; }
  size_t ple_hist_off() const { return (size_t)3 * hp.conv_dim() * 4; }
  size_t ple_tok_off() const { return ple_hist_off() + (size_t)ple_hist() * hp.hce() * 4; }
  // (QSA: the raw keys of each QSA layer's tail block, 4 KB apart)
  size_t qsa_tail_off(int i) const { return (ple_tok_off() + 16 + 4095) / 4096 * 4096 + (size_t)i * 4096; }
  size_t conv_bytes() const {
    return hp.n_qsa ? qsa_tail_off(hp.n_qsa) : ple_tok_off() + (hp.ple_layer >= 0 ? 16 : 0);
  }
  int qsa_blocks(int c) const { return hp.n_qsa ? (c + 15) / 16 * 16 / hp.qsa_ratio[hp.full_attn_interval - 1] : 0; }
  int qsa_mask_words(int c) const { return (qsa_blocks(c) + 31) / 32; }
  std::vector<uint8_t> read_buf(cl_mem b, size_t n, size_t off = 0) {
    std::vector<uint8_t> v(n);
    if (n) CL_CHECK(clEnqueueReadBuffer(cl_.queue, b, CL_TRUE, off, n, v.data(), 0, nullptr, nullptr));
    return v;
  }
  void write_buf(cl_mem b, const std::vector<uint8_t>& v, size_t off = 0) {
    if (!v.empty()) CL_CHECK(clEnqueueWriteBuffer(cl_.queue, b, CL_TRUE, off, v.size(), v.data(), 0, nullptr, nullptr));
  }
  static bool put(FILE* f, const void* p, size_t n) { return fwrite(p, 1, n, f) == n; }
  static bool put(FILE* f, const std::vector<uint8_t>& v) { return put(f, v.data(), v.size()); }
  static bool get(FILE* f, void* p, size_t n) { return fread(p, 1, n, f) == n; }
  Snap& snap_slot(int slot) {  // GPU snapshot slot with its buffers allocated (as snapshot() does)
    if (slot >= (int)snaps_.size()) snaps_.resize(slot + 1);
    Snap& s = snaps_[slot];
    if (s.S.empty())
      for (auto& L : layers_)
        if (hp.recurrent(L.il)) { s.S.push_back(buf(state_bytes())); s.conv.push_back(buf(conv_bytes())); }
    if (has_mtp && !s.h) s.h = buf((size_t)hp.n_embd * 4);
    return s;
  }
  // layers with a KV cache of their own and its geometry (int8 cache: K/V blocks of 16 keys x hd bytes per kv head,
  // sb scales per key and block)
  struct KvGeom {
    Layer* L;
    int nkv, hd, cap, sb;
  };
  std::vector<KvGeom> kv_layers() {
    std::vector<KvGeom> v;
    for (auto& L : layers_) {
      if (hp.gemma) {
        if (L.kv_src < 0) v.push_back({&L, L.nkv, L.hd, L.cap, L.hd / 64});
      } else if (!hp.recurrent(L.il)) {
        v.push_back({&L, hp.n_head_kv, hp.head_dim, ctx, 1});
      }
    }
    if (mtp_block_) v.push_back({&mtp_, hp.n_head_kv, hp.head_dim, ctx, 1});
    return v;
  }
  int kv_hi() const { return hp.gemma ? std::max(pos, written_hi_) : pos; }  // positions the caches hold
  // first position the sliding-window layers keep when saving: n_swa before keep_from (block-aligned), no older than
  // what the ring still holds
  int swa_save_lo(int keep_from) const {
    const int need = std::max(0, std::min(keep_from, pos) - hp.n_swa) / 16 * 16;
    const int ring = (std::max(0, kv_hi() - swa_cap_) + 15) / 16 * 16;
    return std::max({need, ring, (swa_lo_ + 15) / 16 * 16});
  }
  struct Seg {
    cl_mem b;
    size_t off, n;
  };
  // device ranges of a saved state, in stream order: per KV layer and kv head K, V, K scales, V scales (blocks
  // [b0, b1) at their ring slots), the DeltaNet states, the GPU snapshots that hold a position, the MTP hidden state
  std::vector<Seg> state_segs(const std::vector<KvGeom>& kl, const std::vector<std::pair<int, int>>& br,
                              const std::vector<int>& sp) {
    std::vector<Seg> v;
    for (size_t i = 0; i < kl.size(); ++i) {
      const KvGeom& k = kl[i];
      const int nbc = k.cap / 16, b0 = br[i].first, n = br[i].second - br[i].first, s0 = b0 % nbc;
      const int first = std::min(n, nbc - s0);
      for (int g = 0; g < k.nkv; ++g)
        for (int t = 0; t < (k.L->vq4 ? 6 : 4); ++t) {  // (--v4: + the 4-bit V copy and its scales)
          cl_mem b = t == 0 ? k.L->kq : t == 1 ? k.L->vq : t == 2 ? k.L->ksc : t == 3 ? k.L->vsc : t == 4 ? k.L->vq4 : k.L->vsc4;
          const size_t unit = t < 2 ? (size_t)16 * k.hd : t < 4 ? (size_t)k.sb * 16 * 4 : t == 4 ? (size_t)8 * k.hd : (size_t)(k.hd / 64) * 16 * 4;
          const size_t base = (size_t)g * nbc;
          if (first > 0) v.push_back({b, (base + s0) * unit, (size_t)first * unit});
          if (n > first) v.push_back({b, base * unit, (size_t)(n - first) * unit});
        }
      if (q2_)  // (two-pass mode: the float K / V rows attn_fa_g reads, [kv head][cap][hd], the same 16-row blocks)
        for (int g = 0; g < k.nkv; ++g)
          for (int t = 0; t < 2; ++t) {
            const size_t unit = (size_t)16 * k.hd * (t == 0 ? (q2_kf32_ ? 4 : 2) : 2), base = (size_t)g * nbc;
            cl_mem b = t == 0 ? k.L->kc : k.L->vc;
            if (first > 0) v.push_back({b, (base + s0) * unit, (size_t)first * unit});
            if (n > first) v.push_back({b, base * unit, (size_t)(n - first) * unit});
          }
    }
    if (hp.n_qsa && !br.empty())  // QSA pooled keys of the blocks below the saved positions (the KV blocks' range)
      for (auto& L : layers_)
        if (L.pk) v.push_back({L.pk, 0, (size_t)std::min(qsa_blocks(ctx), br[0].second * 16 / L.qr) * hp.qsa_dim * 4});
    for (auto& L : layers_)
      if (hp.recurrent(L.il)) {
        v.push_back({L.S, 0, state_bytes()});
        v.push_back({L.conv_state, 0, conv_bytes()});
      }
    for (size_t i = 0; i < sp.size(); ++i) {
      if (sp[i] < 0) continue;
      const Snap& sn = snaps_[i];
      for (size_t k = 0; k < sn.S.size(); ++k) {
        v.push_back({sn.S[k], 0, state_bytes()});
        v.push_back({sn.conv[k], 0, conv_bytes()});
      }
      if (has_mtp) v.push_back({sn.h, 0, (size_t)hp.n_embd * 4});
    }
    if (has_mtp) v.push_back({hlast_, 0, (size_t)hp.n_embd * 4});
    return v;
  }
  // (a saved state loads only into the same layout: Gemma, the 4-bit V copy, the two-pass mode's float K / V rows)
  int state_flags() const { return (hp.gemma ? 1 : 0) | (opt.v4 ? 2 : 0) | (q2_ ? 4 : 0); }
  // header and device ranges of save_state()
  void save_layout(int keep_from, std::vector<int32_t>& hdr, std::vector<Seg>& segs, bool host_snapshot = true) {
    const auto kl = kv_layers();
    // positions [0, pos) (beyond pos a Gemma cache may still hold a longer, rewound conversation: not needed)
    const int hi = pos, slo = hp.gemma ? swa_save_lo(keep_from) : 0;
    hdr = {kStateMagic, 2, hp.n_layer, hp.n_embd, hp.n_vocab, state_flags(), has_mtp ? 1 : 0, pos, pos, slo,
           (int)kl.size()};
    std::vector<std::pair<int, int>> br;  // blocks [b0, b1) saved per KV layer
    for (const KvGeom& k : kl) {
      const int b0 = k.L->swa ? slo / 16 : 0, b1 = std::max(b0, (hi + 15) / 16);
      br.push_back({b0, b1});
      hdr.insert(hdr.end(), {k.nkv, k.hd, k.sb, b0, b1});
    }
    std::vector<int> sp;
    for (const Snap& sn : snaps_) sp.push_back(sn.pos);
    hdr.push_back((int)sp.size());
    hdr.insert(hdr.end(), sp.begin(), sp.end());
    hdr.push_back(host_snapshot ? hsnap_.pos : -1);
    segs = state_segs(kl, br, sp);
  }
  // device ranges -> sink / stream -> device ranges, through a host buffer of up to 256 MB: non-blocking copies, one
  // wait per batch
  template <class Fn>
  static void batches(const std::vector<Seg>& segs, Fn fn) {  // fn(first, last, bytes) per batch
    const size_t cap = (size_t)256 << 20;
    for (size_t i = 0; i < segs.size();) {
      size_t j = i, n = 0;
      while (j < segs.size() && (j == i || n + segs[j].n <= cap)) n += segs[j++].n;
      if (!fn(i, j, n)) return;
      i = j;
    }
  }
  bool segs_out(const std::vector<Seg>& segs, const Sink& out) {
    std::vector<uint8_t> hb;
    bool ok = true;
    batches(segs, [&](size_t i, size_t j, size_t n) {
      hb.resize(n);
      size_t o = 0;
      for (size_t k = i; k < j; o += segs[k++].n)
        if (segs[k].n)
          CL_CHECK(clEnqueueReadBuffer(cl_.queue, segs[k].b, CL_FALSE, segs[k].off, segs[k].n, hb.data() + o, 0, nullptr, nullptr));
      CL_CHECK(clFinish(cl_.queue));
      return ok = out(hb.data(), n);
    });
    return ok;
  }
  bool segs_in(const std::vector<Seg>& segs, FILE* f) {
    std::vector<uint8_t> hb;
    bool ok = true;
    batches(segs, [&](size_t i, size_t j, size_t n) {
      hb.resize(n);
      if (!(ok = get(f, hb.data(), n))) return false;
      size_t o = 0;
      for (size_t k = i; k < j; o += segs[k++].n)
        if (segs[k].n)
          CL_CHECK(clEnqueueWriteBuffer(cl_.queue, segs[k].b, CL_FALSE, segs[k].off, segs[k].n, hb.data() + o, 0, nullptr, nullptr));
      CL_CHECK(clFinish(cl_.queue));
      return true;
    });
    return ok;
  }
  GgufFile f_;
  std::string path_;
  ClRt cl_;
  std::string gemm_src_, ops_src_, dec_src_;
  std::map<std::string, cl_kernel> kcache_;
  std::vector<cl_program> progs_;  // released with the engine
  std::unique_ptr<XCache> xc_;     // arcinferzero: streamed experts (null: experts on the GPU as in arcinfer)
  bool stream_ = false;            // experts streamed (also set in dry runs, where xc_ stays null)
  uint64_t stream_bytes_ = 0;      // all routed experts in the GPU form
  std::vector<cl_mem> mems_;       // every buffer and sub-buffer, released with the engine
  cl_kernel k_rmsnorm_ = nullptr, k_swiglu_ = nullptr, k_gdn_conv_ = nullptr, k_conv_upd_ = nullptr,
            k_gdn_recur_ = nullptr, k_gnorm_ = nullptr, k_attn_prep_ = nullptr, k_attn_ = nullptr,
            k_attn_gate_ = nullptr, k_gdn_gates_ = nullptr, k_attn_q8_ = nullptr, k_attn_dec_ = nullptr,
            k_attn_comb_ = nullptr, k_gdn_recur1_ = nullptr;
  std::vector<Layer> layers_;
  cl_kernel k_add_norm_ = nullptr, k_ple_gate_ = nullptr, k_ple_comb_ = nullptr, k_add_nq_ = nullptr;
  cl_kernel k_rope_tab_ = nullptr;
  // Gemma 4 RoPE tables (sliding, global) and the rows they hold (pos, T)
  cl_mem rope_tab_[2] = {nullptr, nullptr};
  int rt_pos_[2] = {-1, -1}, rt_T_[2] = {0, 0};
  // single-token passes: the next layer's attention input was already normed and quantized (add_norm_quant)
  bool attn_in_ready_ = false, ffn_in_ready_ = false;
  const GgufTensor* ple_tok_embd_ = nullptr;
  GpuWeight ple_model_proj_;
  cl_mem ple_proj_norm_ = nullptr, ple_ = nullptr, ple_tok_ = nullptr, ple_g_ = nullptr;
  std::vector<float> ple_host_;
  int nc_lo_ = -1, nc_hi_ = -1;  // positions processed non-causally in the current forward (an image span)
  cl_mem rpos_ = nullptr;          // per-row (t, h, w) rotary positions of an image chunk (Qwen M-RoPE)
  int use_rpos_ = 0;
  int swa_cap_ = 0, written_hi_ = 0;  // Gemma: sliding-window ring capacity, highest position written + 1
  // MTP block and speculative-decoding state
  Layer mtp_;
  GpuWeight eh_proj_;
  cl_mem enorm_ = nullptr, hnorm_ = nullptr, shnorm_ = nullptr;
  cl_kernel k_mtp_concat_ = nullptr;
  std::vector<int> last_drafts_, pend_toks_;
  cl_mem mtp_x_ = nullptr, mtp_e_ = nullptr, hfin_ = nullptr, hfin1_ = nullptr, hlast_ = nullptr, hpend_ = nullptr,
         hver_ = nullptr, mtp_h_ = nullptr;
  bool spec_ = false;  // verification pass: DeltaNet states to S_next, the inputs kept (verify_done)
  std::vector<float> mtp_logits_host_, spec_logits_host_;
  std::vector<int> draft_ids_;     // vocabulary rows of the drafts' output head (empty: the whole head)
  cl_mem draft_idx_ = nullptr;
  int conv_ct_ = 1;  // tokens per sub-group in gdn_conv (1 is fastest: more sub-groups in flight)
  GpuWeight lm_head_;
  cl_mem out_norm_ = nullptr, tabs_ = nullptr;
  // Flash Next: the final mixer (the output norm), the mixers' buffers (normed streams, gates, low-rank + inject
  // outputs, the block output / mixed input), the PLE block's n-gram embeddings (per chunk view in layer-major
  // prefill) and the token ids before the current position (host copy of the ones kept in the PLE layer's state)
  cl_mem hc_rinv_ = nullptr;  // (single-token mixers: the streams' 1 / rms)
  cl_mem out_hc_norm_ = nullptr, hc_xn_ = nullptr, hc_gate_ = nullptr, hc_lo_ = nullptr, hco_ = nullptr,
         ple_emb_ = nullptr, ple_hist_ = nullptr;
  GpuWeight out_hc_down_, out_hc_up_;
  std::vector<cl_mem> ple_views_;
  const GgufTensor* ple_table_ = nullptr;
  int ple_prev_[2] = {-1, -1};
  // QSA: indexer rows of the pass, normed / rotated queries, block scores of up to 256 rows, the selection masks
  cl_mem qsa_ixo_ = nullptr, qsa_qx_ = nullptr, qsa_sc_ = nullptr, qsa_mask_ = nullptr;
  int qsa_n_ = 0;
  const GgufTensor* tok_embd_ = nullptr;
  // activations
  cl_mem x_ = nullptr, aq_ = nullptr, asc_ = nullptr, absum_ = nullptr, big_ = nullptr, z_ = nullptr,
         conv_out_ = nullptr, gb_ = nullptr, o_ = nullptr, qrope_ = nullptr, logits_ = nullptr, qi_ = nullptr,
         qsc_ = nullptr, apart_ = nullptr;
  // Gemma, T <= 8: the fused gate/up GEMV (gemv_glu) writes the down projection's int8 input here
  cl_mem aqB_ = nullptr, ascB_ = nullptr, absumB_ = nullptr;
  // just-in-time W8 staging, one slot per matrix role (attn_qkv, ffn_up, ...): a layer's matrices stay decoded
  // while several chunks go through it (layer-major prefill)
  static constexpr int kRoles = 16;
  cl_mem stage_q_[kRoles] = {}, stage_h_[kRoles] = {};
  cl_mem splitk_ = nullptr;  // split-K GEMV partials (up to 4 x 65536 floats)
  bool has_rgu_ = false;      // MoE: stacked router + shared-expert gate / up weights (Layer::rgu)
  cl_mem rgu_out_ = nullptr, rgu_gu_ = nullptr;
  cl_mem qz_out_ = nullptr, qz_z_ = nullptr, qz_ab_ = nullptr;  // stacked DeltaNet input projections' output, its views
  bool has_qz_ = false;
  size_t mem_mtp_ = 0, mem_bufs_ = 0, mem_load_ = 0;  // GPU bytes: MTP block / drafter, working buffers, after loading
  // router rows padded to 128 (512 bytes: the shared-expert part of rgu_out_ starts at a sub-buffer alignment)
  int rgu_rows() const { return (hp.router_rows() + 127) / 128 * 128; }
  size_t stage_need_q_[kRoles] = {}, stage_need_h_[kRoles] = {};
  const GpuWeight* stage_owner_[kRoles] = {};
  // Gemma 4 assistant drafter (speculative decoding): small layers of its own whose attention reads the main model's
  // KV cache (its last sliding-window and last global layer); input [embedding(tok) * sqrt(E) | main hidden state]
  struct DraftLayer {
    bool swa = false;
    int hd = 0;
    float out_scale = 1.0f;
    GpuWeight wq, wo, gate, up, down;
    cl_mem attn_norm = nullptr, q_norm = nullptr, post_attn = nullptr, ffn_norm = nullptr, post_ffw = nullptr,
           rope_ff = nullptr;
  };
  std::unique_ptr<GgufFile> df_;
  std::vector<DraftLayer> dl_;
  GpuWeight d_pre_, d_post_, d_lm_;
  cl_mem d_norm_ = nullptr, d_x_ = nullptr, d_cat_ = nullptr, d_h_ = nullptr;
  int d_E_ = 0, d_F_ = 0, d_nh_ = 0, d_swa_ = 0;  // drafter width, FFN size, q heads, sliding window
  float d_base_ = 1e6f, d_base_swa_ = 1e4f;
  bool gemma_mtp_ = false;
  bool mtp_block_ = false;  // the model's own MTP (nextn) block is loaded (Qwen)
  // MoE: router logits, routing, slots (token, expert pairs grouped by expert), expert activations and outputs
  std::string moe_src_;
  cl_mem gmlp_ = nullptr, gmoe_ = nullptr;  // (Gemma 4 experts, see layer_gemma_ffn)
  cl_mem rl_ = nullptr, mids_ = nullptr, mwts_ = nullptr, sgate_ = nullptr, mcount_ = nullptr, moffs_ = nullptr,
         mcursor_ = nullptr, mtiles_ = nullptr, stok_ = nullptr, swt_ = nullptr, minv_ = nullptr, mh_ = nullptr,
         mhq_ = nullptr, mhsc_ = nullptr, my_ = nullptr, msh_ = nullptr, mtstart_ = nullptr;
  int slots_ = 0;  // slot capacity (max_chunk * n_expert_used): row stride of mhsc_
  // prompt chunks of standard-type experts: a group of moe_group_ experts at a time decoded into W8 (moe_wq_ /
  // moe_wh_), then the int8 GEMM over that group's slot tiles (gemm.cl MOE_TILES)
  cl_mem moe_wq_ = nullptr, moe_wh_ = nullptr;
  cl_mem pf_rl_ = nullptr, pf_tiles_ = nullptr, pf_dummy_ = nullptr;  // next-layer routing guess (generation)
  bool pf_next_ = false;
  // one token, streamed experts: the tiles of its cached and of its missing experts (layer_moe, ARC_XSPLIT)
  cl_mem xt_hit_ = nullptr, xt_miss_ = nullptr;
  std::vector<int> xt_hit_h_, xt_miss_h_;
  int moe_group_ = 0;
  // residual stream of a prompt segment (up to lm_seg_ tokens) for layer-major prefill, and its per-chunk views
  cl_mem x_all_ = nullptr;
  int lm_seg_ = 16384, prog_base_ = 0, prog_total_ = 0;
  std::vector<cl_mem> x_views_;
  std::vector<float> logits_host_, x_host_;
  std::vector<std::pair<std::string, cl_event>> events_;
  int cur_T_ = 0;

  // ---------------------------------------------------------------- setup
  void load_hparams() {
    const std::string a = f_.get_str("general.architecture");
    hp.arch = a;
    if (a == "gemma4") return load_hparams_gemma4();
    if (a != "qwen35" && a != "qwen35moe" && a != "qwen4exp") throw std::runtime_error("unsupported architecture: " + a);
    hp.moe = a == "qwen35moe" || a == "qwen4exp";
    hp.hc = a == "qwen4exp";
    auto I = [&](const std::string& k, int64_t def = -1) {
      int64_t v = f_.get_int(a + "." + k, def);
      if (v < 0) throw std::runtime_error("missing hparam " + k);
      return (int)v;
    };
    {
      const int blocks = I("block_count"), nextn = (int)f_.get_int(a + ".nextn_predict_layers", 0);
      if (blocks <= 0 || blocks > 4096 || nextn < 0 || nextn >= blocks)
        throw std::runtime_error("invalid block_count / nextn_predict_layers in the GGUF");
      hp.n_layer = blocks - nextn;
    }
    hp.n_embd = I("embedding_length");
    if (hp.n_embd <= 0) throw std::runtime_error("invalid embedding_length in the GGUF");
    hp.n_ff = hp.moe ? 0 : I("feed_forward_length");
    if (hp.moe) {
      hp.n_expert = I("expert_count");
      hp.n_expert_used = I("expert_used_count");
      hp.n_ff_exp = I("expert_feed_forward_length");
      hp.n_ff_shexp = I("expert_shared_feed_forward_length");
      // experts: GSQ 2-bit (tools/convert_gsq.py, zero/kernels/moe.cl) or standard GGUF types (decode.cl moe_*_q)
      hp.moe_std = f_.get_str("arcinfer.expert_format", "") != "gsq2-v1";
      hp.n_ff_exp_pad = padded_k(hp.n_ff_exp);
      // (Flash Next: FFN widths of 640, its down projections padded to 768 and its shared expert to 768 rows)
      if (hp.hc) hp.n_ff_shexp = padded_k(hp.n_ff_shexp);
      if ((hp.n_expert != 256 && !(hp.hc && hp.n_expert == 512)) || hp.n_expert_used > 16 ||
          (hp.n_ff_exp % 256 && !hp.hc) || hp.n_ff_shexp % 256 || (hp.hc && !hp.moe_std))
        throw std::runtime_error("unsupported MoE dimensions");
    }
    if (hp.hc) {
      hp.hc_n = I("hyper_connection.count");
      hp.hc_lr = I("hyper_connection.low_rank");
      if (const auto* pl = f_.get_ints(a + ".ple.layers"))
        if (!pl->empty()) hp.ple_layer = (int)(*pl)[0];
      if (hp.ple_layer >= 0) {
        hp.ple_dim = I("embedding_length_per_layer_input");
        hp.ple_ng = I("ple.ngram_size");
        hp.ple_per = I("ple.heads_per_ngram");
        hp.ple_taps = I("ple.conv_kernel");
        hp.ple_eos = f_.get_int(a + ".ple.eos_token_id", -1);
        for (auto [key, v] : {std::pair{"ple.layer_multipliers", &hp.ple_mult}, std::pair{"ple.head_offsets", &hp.ple_off},
                              std::pair{"ple.head_vocab_sizes", &hp.ple_voc}})
          if (const auto* x = f_.get_ints(a + "." + key)) v->assign(x->begin(), x->end());
        const int nph = (hp.ple_ng - 1) * hp.ple_per;
        if ((int)hp.ple_mult.size() < hp.ple_ng || (int)hp.ple_off.size() < nph || (int)hp.ple_voc.size() < nph ||
            nph * hp.ple_dim != I("embedding_length") || hp.ple_taps != 4 || (hp.ple_taps - 1) * hp.ple_ng > 16)
          throw std::runtime_error("unsupported PLE n-gram block");
      }
      hp.full_attn_interval = I("full_attention_interval", 4);
      if (const auto* cr = f_.get_ints(a + ".attention.compress_ratios")) {
        const int nl = I("block_count");
        hp.qsa_ratio.assign(nl, 0);
        for (int il = 0; il < nl && il < (int)cr->size(); ++il)
          if (f_.find("blk." + std::to_string(il) + ".indexer.k_proj.weight")) hp.qsa_ratio[il] = (int)(*cr)[il];
        for (int r : hp.qsa_ratio) hp.n_qsa += r > 0;
        if (hp.n_qsa) {
          hp.qsa_heads = I("attention.indexer.head_count");
          hp.qsa_dim = I("attention.indexer.key_length");
          hp.qsa_topk = I("attention.indexer.top_k");
          if (const char* e = getenv("ARC_QSA_TOPK")) hp.qsa_topk = atoi(e);  // (tests: fewer keys, sparse sooner)
          for (int r : hp.qsa_ratio)
            if (r > 0 && (r < 2 || r > 8 || hp.qsa_topk % r || r != hp.qsa_ratio[hp.full_attn_interval - 1]))
              throw std::runtime_error("unsupported QSA block size");
          if (hp.qsa_dim != 128 || hp.qsa_heads < 1 || hp.qsa_heads > 16 || hp.ple_layer < 0)
            throw std::runtime_error("unsupported QSA indexer");
          // (attn_fa, the f32 path, has no QSA masks: a Flash Next 8K perplexity of 42.6 instead of 2.20)
          if (!opt.attn_int8)
            throw std::runtime_error("f32 attention (--attn-f32) does not support QSA sparse attention (Qwen3.8 Flash Next)");
        }
      }
    }
    hp.n_head = I("attention.head_count");
    hp.n_head_kv = I("attention.head_count_kv");
    hp.head_dim = I("attention.key_length");
    hp.n_rot = I("rope.dimension_count");
    hp.ssm_d_state = I("ssm.state_size");
    hp.ssm_nk = I("ssm.group_count");
    hp.ssm_nv = I("ssm.time_step_rank");
    hp.ssm_d_inner = I("ssm.inner_size");
    hp.conv_k = I("ssm.conv_kernel");
    hp.full_attn_interval = I("full_attention_interval", 4);
    hp.eps = (float)f_.get_float(a + ".attention.layer_norm_rms_epsilon", 1e-6);
    hp.rope_base = (float)f_.get_float(a + ".rope.freq_base", 1e7);
    if (const auto* rs = f_.get_ints(a + ".rope.dimension_sections"))
      for (int i = 0; i < 4 && i < (int)rs->size(); ++i) hp.rope_sect[i] = (int)(*rs)[i];
    if (hp.rope_sect[0] + hp.rope_sect[1] + hp.rope_sect[2] + hp.rope_sect[3] == 0) hp.rope_sect[0] = hp.n_rot / 2;
    tok_embd_ = &f_.get("token_embd.weight");
    hp.n_vocab = (int)tok_embd_->dims[1];
    if (hp.head_dim != 256 || hp.ssm_d_state != 128 || hp.conv_k != 4 || hp.n_rot > 64 || hp.n_embd % 256 ||
        hp.n_ff % 256 || hp.ssm_d_inner % 256 || hp.n_head % hp.n_head_kv)
      throw std::runtime_error("unsupported Qwen3.5 dimensions");
    if (hp.hc && (hp.hc_n < 1 || hp.hc_n > 16 || hp.hc_lr <= 0 || hp.hc_lr + hp.hc_n > 512 ||
                  (hp.ple_layer >= 0 && !hp.recurrent(hp.ple_layer))))
      throw std::runtime_error("unsupported hyper-connection shape");
  }

  void load_hparams_gemma4() {
    const std::string a = "gemma4";
    auto I = [&](const std::string& k, int64_t def = -1) {
      int64_t v = f_.get_int(a + "." + k, def);
      if (v < 0) throw std::runtime_error("missing hparam " + k);
      return (int)v;
    };
    hp.gemma = true;
    hp.n_layer = I("block_count");
    if (hp.n_layer <= 0 || hp.n_layer > 4096) throw std::runtime_error("invalid block_count in the GGUF");
    hp.n_embd = I("embedding_length");
    hp.n_ff = I("feed_forward_length");
    hp.n_head = I("attention.head_count");
    hp.hd_full = I("attention.key_length");
    hp.hd_swa = I("attention.key_length_swa", hp.hd_full);
    hp.head_dim = std::max(hp.hd_full, hp.hd_swa);
    hp.n_swa = I("attention.sliding_window", 0);
    hp.eps = (float)f_.get_float(a + ".attention.layer_norm_rms_epsilon", 1e-6);
    hp.rope_base = (float)f_.get_float(a + ".rope.freq_base", 1e6);
    hp.rope_base_swa = (float)f_.get_float(a + ".rope.freq_base_swa", 1e4);
    hp.softcap = (float)f_.get_float(a + ".final_logit_softcapping", 0.0);
    hp.n_ple = (int)f_.get_int(a + ".embedding_length_per_layer_input", 0);
    hp.n_layer_kv = hp.n_layer - (int)f_.get_int(a + ".attention.shared_kv_layers", 0);
    if (hp.n_ple != 0 && hp.n_ple != 256) throw std::runtime_error("unsupported Gemma 4 per-layer embedding size");
    if (hp.n_layer_kv < 2) throw std::runtime_error("unsupported Gemma 4 KV sharing");
    hp.swa.assign(hp.n_layer, 0);
    hp.kv_heads.assign(hp.n_layer, 0);
    const auto* pat = f_.get_ints(a + ".attention.sliding_window_pattern");
    const auto* kvh = f_.get_ints(a + ".attention.head_count_kv");
    for (int il = 0; il < hp.n_layer; ++il) {
      hp.swa[il] = pat && il < (int)pat->size() ? (int)(*pat)[il] : 0;
      hp.kv_heads[il] = kvh && il < (int)kvh->size() ? (int)(*kvh)[il] : I("attention.head_count_kv");
    }
    hp.n_head_kv = *std::max_element(hp.kv_heads.begin(), hp.kv_heads.end());
    if (hp.n_embd <= 0 || hp.n_head <= 0 || hp.n_head_kv <= 0 || hp.n_layer_kv > hp.n_layer)
      throw std::runtime_error("invalid Gemma 4 shape in the GGUF");
    tok_embd_ = &f_.get("token_embd.weight");
    hp.n_vocab = (int)tok_embd_->dims[1];
    for (int il = 0; il < hp.n_layer; ++il) {
      const int hd = hp.swa[il] ? hp.hd_swa : hp.hd_full;
      if ((hd != 256 && hd != 512) || hp.n_head % hp.kv_heads[il])
        throw std::runtime_error("unsupported Gemma 4 attention shape");
    }
    // (26B-A4B: the dense FFN is 2112 wide; its gate / up get zero rows and its down rows zero columns up to whole
    //  superblocks, 2304: the extra values are gelu(0) * 0 = 0, the results the same)
    hp.n_ff = padded_k(hp.n_ff);
    if (hp.n_embd % 256) throw std::runtime_error("unsupported Gemma 4 dimensions");
    // Gemma 4 26B-A4B: every layer has the dense FFN and routed experts next to it (expert gate and up in one tensor,
    // GELU, a router of its own input norm, a scale per expert)
    hp.n_expert = (int)f_.get_int(a + ".expert_count", 0);
    if (hp.n_expert > 0) {
      hp.moe = hp.moe_std = true;
      hp.n_expert_used = I("expert_used_count");
      hp.n_ff_exp = I("expert_feed_forward_length");
      hp.n_ff_exp_pad = padded_k(hp.n_ff_exp);
      hp.n_ff_shexp = 0;
      if (hp.n_expert % 16 || hp.n_expert_used > 16 || hp.n_ff_exp % 32 || hp.n_ple)
        throw std::runtime_error("unsupported Gemma 4 expert shape");
    }
  }

  // q heads per attention sub-group: the largest of 4,3,2,1 dividing the GQA ratio.
  // int8 attention tiling: each sub-group handles hs of the q heads sharing a kv head (nsp = hpg / hs sub-groups
  // per token, a power of two), and a work-group covers tpw tokens with 512 work-items.
  // Qwen prompt attention (attn_q8, 256-dim heads): 4 q heads per sub-group where the GQA ratio allows, despite a few
  // spilled registers: 16384-token prompt (bench/attn_bench), 8 q heads per kv head (Qwen3.6 35B) 544 -> 479 ms per
  // layer, 4 per kv head (Qwen3.5 9B) 531 -> 468 ms; 6 per kv head (27B) stays at 3 (6 per sub-group spills: slower)
  int attn_hs() const {
    const int hpg = hp.n_head / hp.n_head_kv;
    static const bool old = getenv("ARC_ATTN_HS_OLD") != nullptr;  // (A/B)
    if (!hp.gemma && !old && hpg % 4 == 0 && hpg / 4 <= 32 && ((hpg / 4) & (hpg / 4 - 1)) == 0) return 4;
    return attn_hs_for(hpg);
  }
  static int attn_hs_for(int hpg) {
    for (int h : {3, 2, 1})
      if (hpg % h == 0 && ((hpg / h) & (hpg / h - 1)) == 0 && hpg / h <= 32) return h;
    return 1;
  }
  int attn_nsp() const { return hp.n_head / hp.n_head_kv / attn_hs(); }
  // Qwen prompt attention with the head dims split over sub-groups (attn_ps) for chunks ending past `from`: dim
  // groups, q heads and tokens per sub-group, tokens per work-group. pns = 0: attn_q8 everywhere. ARC_QWEN_PS =
  // "pns,phg,ptt,tpw[,from]" overrides (bench/attn_bench ATT_PS / ATT_TPW for the search).
  struct PsCfg { int pns = 0, phg = 0, ptt = 0, tpw = 0, from = 0; };
  PsCfg qwen_ps(int hd, int hpg) const {
    PsCfg c;
    if (hp.gemma || hd != 256) return c;
    if (const char* e = getenv("ARC_QWEN_PS")) {
      sscanf(e, "%d,%d,%d,%d,%d", &c.pns, &c.phg, &c.ptt, &c.tpw, &c.from);
      if (c.pns && (hpg % c.phg || c.tpw % c.ptt)) c.pns = 0;
      return c;
    }
    (void)hpg;
    return c;  // (defaults per GQA ratio: set from the measurements)
  }
  int attn_tpw() const { return 32 / attn_nsp(); }
  // build options of the int8 attention kernels for a head size and GQA ratio
  std::string attn_opts(int hd, int hpg) const {
    const int hs = attn_hs_for(hpg), nsp = hpg / hs;
    return "-cl-intel-256-GRF-per-thread -DHD=" + std::to_string(hd) + " -DHPG=" + std::to_string(hpg) + " -DHS=" +
           std::to_string(hs) + " -DTPW=" + std::to_string(32 / nsp) + sb_opt(hd);
  }
  // int8 attention scale blocks per row: Gemma 4 one per 64 dims (its V rows have a few large dims, which a single
  // row scale lets swamp the others: measured against an f32 reference), Qwen one per row (as accurate there, faster)
  std::string sb_opt(int hd) const { return " -DSB=" + std::to_string(hp.gemma ? hd / 64 : 1); }
  // --v4: attn_prep also writes the 4-bit copy of V, and the generation / verification kernels (attn_decode,
  // attn_split, attn_dsplit: up to 8 rows) read it; prompt chunks (attn_q8, attn_ps) read the int8 V at full speed
  std::string v4_opt() const { return opt.v4 && !hp.gemma ? " -DV4" : ""; }
  cl_mem vq_dec(const Layer& L) const { return opt.v4 && L.vq4 ? L.vq4 : L.vq; }
  cl_mem vsc_dec(const Layer& L) const { return opt.v4 && L.vsc4 ? L.vsc4 : L.vsc; }
  // --v4: the first keys (ARC_V4_SINK, default 128) and each row's most recent ones (ARC_V4_RECENT, default 4096) are
  // read from the int8 V: attention puts most of its weight there, and they are a few % of a long context (131K: 3 %).
  // 35B, 8K generation path, KL vs int8: 0 / 0 0.023, 128 / 1024 0.019, 128 / 2048 0.012, 128 / 4096 0.008
  static int v4_sink() {
    static const int v = getenv("ARC_V4_SINK") ? std::max(0, atoi(getenv("ARC_V4_SINK"))) : 128;
    return v;
  }
  static int v4_recent() {
    static const int v = getenv("ARC_V4_RECENT") ? std::max(0, atoi(getenv("ARC_V4_RECENT"))) : 4096;
    return v;
  }
  // the extra arguments of attn_decode / attn_dsplit under --v4 (from argument `at`), for the row at position p
  void v4_dec_args(cl_kernel k, int at, const Layer& L, int p) {
    if (!opt.v4) return;
    int lo = v4_sink(), hi = p + 1 - v4_recent();
    set_args(k, at, L.vq, L.vsc, lo, hi);
  }
  // Experiment, not a feature: ARC_KV4 = "K32V32", "V32", "K64" ... (block sizes 16-256) simulates a 4-bit K and / or V
  // cache inside the int8 one (ops.cl kv4_fake), to measure its accuracy. Anything else is refused.
  static std::string kv4_opts() {
    const char* e = getenv("ARC_KV4");
    if (!e || !*e) return "";
    std::string out;
    const std::string s = e;
    for (size_t i = 0; i < s.size();) {
      const char c = s[i];
      size_t j = i + 1;
      while (j < s.size() && isdigit((unsigned char)s[j])) ++j;
      const int b = j > i + 1 ? atoi(s.substr(i + 1, j - i - 1).c_str()) : 0;
      if ((c != 'K' && c != 'V') || (b != 16 && b != 32 && b != 64 && b != 128 && b != 256))
        throw std::runtime_error("ARC_KV4: expected e.g. K32V32 or V32 (block sizes 16, 32, 64, 128, 256), got " + s);
      out += std::string(" -DKV4") + c + "=" + std::to_string(b);
      i = j;
      if (i < s.size() && s[i] == 'S') {  // ("V64S": 16 symmetric levels with an unsigned scale instead of q4_0's)
        out += " -DKV4SYM";
        ++i;
      } else if (i < s.size() && (s[i] == 'Z' || s[i] == 'E')) {  // ("V64Z" / "V64E": (n - 8) * amax / 7 or / 8)
        out += s[i] == 'Z' ? " -DKV4ZD=7" : " -DKV4ZD=8";
        ++i;
      } else if (i < s.size() && s[i] == 'O') {  // ("V64O": (n - 8) * d, d chosen per block by squared error)
        out += " -DKV4OPT";
        ++i;
      }
    }
    return out;
  }

  int gq() const {
    const int r = hp.n_head / hp.n_head_kv;
    for (int g : {4, 3, 2, 1})
      if (r % g == 0) return g;
    return 1;
  }

  // Decides whether decoded W8 weights stay resident.
  void plan_weights() {
    if (hp.moe) {  // Q8_0 matrices (repacked at load) and 2-bit experts: nothing is decoded just in time
      w8_resident = true;
      // streamed experts that cannot all be cached (more than the GPU memory share; Qwen3.8 Flash Next: 52 GB): the
      // dense weights' W8 copies (~3 GB) go to the expert cache instead, decoded just in time per prompt segment
      uint64_t ex = 0;
      for (auto& t : f_.tensors)
        if (t.name.find("_exps.weight") != std::string::npos) ex += t.nbytes;
      static const bool keep = getenv("ARC_W8_RESIDENT") != nullptr;  // (A/B)
      if (hp.moe_std && opt.stream_experts && !keep && zrt::global_mem && ex > zrt::global_mem) w8_resident = false;
      return;
    }
    size_t raw = 0, w8 = 0;
    for (auto& t : f_.tensors) {
      int il = -1;
      if (t.dims.size() != 2 || sscanf(t.name.c_str(), "blk.%d.", &il) != 1 || il >= hp.n_layer) continue;
      raw += t.nbytes;
      w8 += (size_t)t.dims[0] * t.dims[1] * 41 / 40;
    }
    w8_resident = (raw + w8) / 1073741824.0 <= opt.w8_budget_gb;
  }

  cl_kernel kernel(const std::string& which, const std::string& defs, const char* name) {
    const std::string key = which + defs + "|" + name;
    auto it = kcache_.find(key);
    if (it != kcache_.end()) return it->second;
    const std::string& src = which == "gemm" ? gemm_src_ : which == "decode" ? dec_src_ : which == "moe" ? moe_src_ : ops_src_;
    auto p = cl_.build(src, "-cl-std=CL3.0 -cl-mad-enable " + defs + (which == "ops" ? q2_defs_ : std::string()));
    progs_.push_back(p);
    return kcache_[key] = cl_.kernel(p, name);
  }
  // (experiments: ARC_GEMM_DEFS adds build options, ARC_GEMM_TM sets the tokens per sub-group)
  static const std::string& gemm_extra() {
    static const std::string e = getenv("ARC_GEMM_DEFS") ? std::string(" ") + getenv("ARC_GEMM_DEFS") : std::string();
    return e;
  }
  static int gemm_tm() {
    static const int tm = getenv("ARC_GEMM_TM") ? std::max(1, atoi(getenv("ARC_GEMM_TM"))) : 16;
    return tm;
  }
  cl_kernel gemm_kernel(int qtype, bool act32) {
    return kernel("gemm", "-cl-intel-256-GRF-per-thread -DQTYPE=" + std::to_string(qtype) + (act32 ? " -DACT32" : "") +
                              (gemm_tm() != 16 ? " -DTM=" + std::to_string(gemm_tm()) : std::string()) + gemm_extra(),
                  "gemm");
  }

  void build_kernels() {
    const std::string dir = opt.kernel_dir;
    gemm_src_ = read_file(dir + "/gemm.cl");
    ops_src_ = read_file(dir + "/ops.cl");
    dec_src_ = read_file(dir + "/decode.cl");
    if (hp.moe) moe_src_ = read_file(dir + "/moe.cl");
    auto p = [&](const char* n) { return kernel("ops", "", n); };
    k_rmsnorm_ = p("rmsnorm_quant");
    k_swiglu_ = p("swiglu_quant");
    if (getenv("ARC_CT")) conv_ct_ = std::max(1, atoi(getenv("ARC_CT")));
    {
      const char* g = getenv("ARC_CONV_GRF");
      const bool big = g != nullptr && std::string(g) == "256";
      k_gdn_conv_ = kernel("ops", (big ? std::string("-cl-intel-256-GRF-per-thread ") : std::string()) + "-DCT=" +
                                      std::to_string(conv_ct_), "gdn_conv");
    }
    k_conv_upd_ = p("conv_state_update");
    k_gnorm_ = hp.hc ? kernel("ops", "-DGATE_SIGMOID", "gated_norm_quant") : p("gated_norm_quant");
    k_attn_prep_ = kernel("ops", sb_opt(256) + v4_opt() + kv4_opts(), "attn_prep");
    k_attn_gate_ = p("attn_gate_quant");
    k_gdn_gates_ = p("gdn_gates");
    const std::string big = "-cl-intel-256-GRF-per-thread -DGQ=" + std::to_string(gq());
    k_attn_ = kernel("ops", big, "attn_fa");
    k_gdn_recur_ = kernel("ops", big, "gdn_recur");
    k_gdn_recur1_ = p("gdn_recur1");  // (T = 1: 128 GRF, 4 lanes per state column)
    const std::string att = "-cl-intel-256-GRF-per-thread -DHPG=" + std::to_string(hp.n_head / hp.n_head_kv) +
                            " -DHS=" + std::to_string(attn_hs()) + " -DTPW=" + std::to_string(attn_tpw()) + sb_opt(256);
    k_attn_q8_ = kernel("ops", att, "attn_q8");
    {
      const int hpg = hp.n_head / hp.n_head_kv, hs = attn_hs_for(hpg);
      k_attn_dec_ = kernel("ops", "-cl-intel-256-GRF-per-thread -DHPG=" + std::to_string(hpg) + " -DHS=" + std::to_string(hs) +
                                      " -DTPW=" + std::to_string(32 / (hpg / hs)) + sb_opt(256) + v4_opt(), "attn_decode");
    }
    k_attn_comb_ = kernel("ops", "", "attn_combine");
    k_mtp_concat_ = kernel("ops", "", "mtp_concat");
    auto tabs = build_iq_tables();
    tabs_ = buf(tabs.size() * 4, tabs.data());
  }

  cl_mem buf(size_t bytes, const void* host = nullptr) {
    gpu_bytes += bytes;
    if (opt.dry) return nullptr;
    mems_.push_back(cl_.buffer(CL_MEM_READ_WRITE, bytes, host));
    return mems_.back();
  }
  // Raw GGUF rows uploaded with each superblock zero-padded to a multiple of 8 bytes (zero/kernels/decode.cl BBP),
  // written in parallel straight into a host-visible buffer (zero-copy on an integrated GPU).
  // Zero bytes after the last superblock: gemv reads two superblocks per step (rounded up to 64-byte block reads),
  // so with an odd superblock count its last row reads up to one superblock + 63 bytes past the data (lanes whose
  // block is past the row contribute 0 x those bytes, which must be finite: zeros). 1024 covers every type.
  static constexpr size_t kRawSlack = 1024;
  cl_mem upload_padded(uint32_t type, const uint8_t* data, size_t nbytes) {
    const size_t bb = gguf_sb_bytes(type), bp = raw_padded_sb_bytes(type), nsb = nbytes / bb;
    const size_t total = nsb * bp + kRawSlack;
    if (opt.dry) {
      gpu_bytes += total;
      return nullptr;
    }
    cl_int err;
    cl_mem m = clCreateBuffer(cl_.ctx, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, total, nullptr, &err);
    CL_CHECK(err);
    mems_.push_back(m);
    gpu_bytes += total;
    uint8_t* dst = (uint8_t*)clEnqueueMapBuffer(cl_.queue, m, CL_TRUE, CL_MAP_WRITE, 0, total, 0, nullptr, nullptr, &err);
    CL_CHECK(err);
    const int64_t chunks = (int64_t)((nsb + 4095) / 4096);
#pragma omp parallel for schedule(static)
    for (int64_t ch = 0; ch < chunks; ++ch) {
      const size_t s0 = (size_t)ch * 4096, s1 = std::min(nsb, s0 + 4096);
      if (bb == bp && !gpu_regrouped(type)) {
        memcpy(dst + s0 * bp, data + s0 * bb, (s1 - s0) * bb);
      } else {
        for (size_t i = s0; i < s1; ++i) gpu_raw_sb(type, data + i * bb, dst + i * bp);
      }
    }
    memset(dst + nsb * bp, 0, kRawSlack);
    CL_CHECK(clEnqueueUnmapMemObject(cl_.queue, m, dst, 0, nullptr, nullptr));
    return m;
  }

  void fill(cl_mem b, size_t bytes) {
    if (!b) return;  // (a dry load)
    const uint32_t zero = 0;
    CL_CHECK(clEnqueueFillBuffer(cl_.queue, b, &zero, 4, 0, bytes, 0, nullptr, nullptr));
  }

  // Uploads a matrix: raw rows always, plus its prompt-processing form when resident.
  // (view: the raw rows already on the GPU, a sub-buffer of a stacked upload)
  GpuWeight make_weight(uint32_t type, const uint8_t* data, size_t nbytes, int N, int K, int role = kRoles - 1,
                        const cl_mem* view = nullptr) {
    GpuWeight w;
    w.role = role;
    w.type = type;
    w.N = N;
    w.K = K;
    w.raw = view ? *view : upload_padded(type, data, nbytes);
    // Q8_0 (per-32 float scales) is fastest in its own repacked kernel; everything else goes through W8. The
    // repacked copy is kept only with resident weights of a dense model; otherwise (MoE: ~1.5 GB of Q8_0 matrices
    // stored twice, or just-in-time models) prompt chunks decode the raw rows into the same form (decode_w8, PER32).
    if (type == GGML_Q8_0 && (!w8_resident || hp.moe)) {
      w.gemm_qtype = (int)type;
      stage_need_q_[w.role] = std::max(stage_need_q_[w.role], w8_bytes_q(N, K));
      stage_need_h_[w.role] = std::max(stage_need_h_[w.role], w8_bytes_h(N, K));
      return w;
    }
    if (type == GGML_Q8_0) {
      RepackedWeight r;
      repack_rows(type, data, K, 0, N, r);
      w.wq = buf(r.wq.size() * 4, r.wq.data());
      w.wh = buf(r.wh.size() * 4, r.wh.data());
      w.gemm_qtype = (int)type;
      return w;
    }
    if (!is_w8_type(type)) throw std::runtime_error(std::string("unsupported weight type ") + ggml_type_name(type));
    // per-32-scale types (Q4_0, IQ4_NL, Q2_0) decode into the native Q8_0 GEMM form (per-32 f32 scales; there is no
    // such W8 variant)
    w.gemm_qtype = w8_variant(type) == 4 ? (int)GGML_Q8_0 : 100 + w8_variant(type);
    const size_t qb = w8_bytes_q(N, K), hb = w8_bytes_h(N, K);
    if (w8_resident) {
      w.wq = buf(qb);
      w.wh = buf(hb);
      if (!opt.dry) decode_w8(w, w.wq, w.wh);
    } else {
      stage_need_q_[w.role] = std::max(stage_need_q_[w.role], qb);
      stage_need_h_[w.role] = std::max(stage_need_h_[w.role], hb);
    }
    return w;
  }
  static int weight_role(const std::string& name) {
    static const char* roles[] = {"attn_qkv.", "attn_gate.", "ssm_out.", "attn_q.", "attn_k.", "attn_v.",
                                  "attn_output.", "ffn_gate.", "ffn_up.", "ffn_down.", "nextn.eh_proj."};
    for (int r = 0; r < (int)(sizeof(roles) / sizeof(roles[0])); ++r)
      if (name.find(roles[r]) != std::string::npos) return r;
    return kRoles - 1;
  }
  GpuWeight load_w(const std::string& name) {
    const GgufTensor& t = f_.get(name);
    const int K = (int)t.dims[0], N = (int)t.dims[1];
    if (t.type == GGML_F32 || t.type == GGML_F16 || t.type == GGML_BF16) {
      std::vector<float> v((size_t)N * K);
      dequant_row(t.type, t.data, N * K, v.data());
      auto q8 = quantize_rows_q8_0(v.data(), N, K);
      return make_weight(GGML_Q8_0, q8.data(), q8.size(), N, K, weight_role(name));
    }
    return make_weight(t.type, t.data, t.nbytes, N, K, weight_role(name));
  }
  // DeltaNet input projections (qkv, z = attn_gate, ab = alpha + beta). When qkv and z have the same quantized type
  // they are uploaded as one stack of rows (with ab, already Q8_0, when they are Q8_0 too): a single token runs one GEMV
  // over all of them (Layer::qz); the matrices are views of it, stored once.
  void load_gdn_in(Layer& L, const std::string& p) {
    const GgufTensor &tq = f_.get(p + "attn_qkv.weight"), &tz = f_.get(p + "attn_gate.weight");
    const int E = hp.n_embd, C = (int)tq.dims[1], Di = (int)tz.dims[1], nab = 2 * hp.ssm_nv;
    static const bool no_qz = getenv("ARC_NO_QZ") != nullptr;  // (A/B: separate GEMVs)
    const bool quant = tq.type != GGML_F32 && tq.type != GGML_F16 && tq.type != GGML_BF16;
    if (no_qz || !quant || tq.type != tz.type || tq.dims[0] != E || tz.dims[0] != E || E % 256 || C % 64 || Di % 64 ||
        C != hp.conv_dim() || Di != hp.ssm_d_inner) {
      L.qkv = load_w(p + "attn_qkv.weight");
      L.z = load_w(p + "attn_gate.weight");
      L.ab = load_q8_concat(p + "ssm_alpha.weight", p + "ssm_beta.weight");
      return;
    }
    const uint32_t type = tq.type;
    std::vector<uint8_t> ab;
    {
      const GgufTensor &ta = f_.get(p + "ssm_alpha.weight"), &tb = f_.get(p + "ssm_beta.weight");
      std::vector<float> v((size_t)nab * E);
      dequant_row(ta.type, ta.data, (int64_t)nab / 2 * E, v.data());
      dequant_row(tb.type, tb.data, (int64_t)nab / 2 * E, v.data() + (size_t)nab / 2 * E);
      ab = quantize_rows_q8_0(v.data(), nab, E);
    }
    L.qz_ab = type == GGML_Q8_0 && nab % 64 == 0;
    std::vector<uint8_t> cat(tq.nbytes + tz.nbytes + (L.qz_ab ? ab.size() : 0));
    memcpy(cat.data(), tq.data, tq.nbytes);
    memcpy(cat.data() + tq.nbytes, tz.data, tz.nbytes);
    if (L.qz_ab) memcpy(cat.data() + tq.nbytes + tz.nbytes, ab.data(), ab.size());
    L.qz.type = type;
    L.qz.N = C + Di + (L.qz_ab ? nab : 0);
    L.qz.K = E;
    L.qz.role = kRoles - 1;
    L.qz.raw = upload_padded(type, cat.data(), cat.size());
    has_qz_ = true;
    const size_t sbr = (size_t)E / 256 * raw_padded_sb_bytes(type);
    auto view = [&](int row0, int N) -> cl_mem {
      if (!L.qz.raw) return nullptr;  // (dry load)
      cl_buffer_region reg{(size_t)row0 * sbr, (size_t)N * sbr + 256};  // (+256: the slack upload_padded leaves)
      cl_int err;
      cl_mem m = clCreateSubBuffer(L.qz.raw, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &reg, &err);
      CL_CHECK(err);
      mems_.push_back(m);
      return m;
    };
    const cl_mem vq = view(0, C), vz = view(C, Di);
    L.qkv = make_weight(type, tq.data, tq.nbytes, C, E, weight_role(p + "attn_qkv.weight"), &vq);
    L.z = make_weight(type, tz.data, tz.nbytes, Di, E, weight_role(p + "attn_gate.weight"), &vz);
    if (L.qz_ab) {
      const cl_mem va = view(C + Di, nab);
      L.ab = make_weight(GGML_Q8_0, ab.data(), ab.size(), nab, E, kRoles - 1, &va);
    } else {
      L.ab = make_weight(GGML_Q8_0, ab.data(), ab.size(), nab, E);
    }
  }
  // Two matrices with the same K stacked by rows, requantized to Q8_0 (the tiny DeltaNet gate projections).
  GpuWeight load_q8_concat(const std::string& a, const std::string& b) {
    const GgufTensor &ta = f_.get(a), &tb = f_.get(b);
    const int K = (int)ta.dims[0], Na = (int)ta.dims[1], Nb = (int)tb.dims[1];
    std::vector<float> v((size_t)(Na + Nb) * K);
    dequant_row(ta.type, ta.data, Na * K, v.data());
    dequant_row(tb.type, tb.data, Nb * K, v.data() + (size_t)Na * K);
    auto q8 = quantize_rows_q8_0(v.data(), Na + Nb, K);
    return make_weight(GGML_Q8_0, q8.data(), q8.size(), Na + Nb, K);
  }
  cl_mem load_f32(const std::string& name) {
    const GgufTensor& t = f_.get(name);
    std::vector<float> v(t.nelements());
    dequant_row(t.type, t.data, (int)v.size(), v.data());
    return buf(v.size() * 4, v.data());
  }
  // A matrix with its rows padded to Np rows of zeros and, for a type of small blocks (32 / 64: Q8_0, IQ4_NL, Q2_0)
  // whose K is not whole superblocks, its columns to padded_k(K) (zero blocks): Flash Next's 640-wide shared expert as 768 (the extra outputs are 0).
  GpuWeight load_w_pad(const std::string& name, int Np) {
    const GgufTensor& t = f_.get(name);
    const int K = (int)t.dims[0], N = (int)t.dims[1];
    if (K % 256 == 0 && N == Np) return load_w(name);
    int blk = 0, bytes = 0;
    ggml_block_info(t.type, blk, bytes);
    if (K % 256 && (blk >= 256 || K % blk)) throw std::runtime_error("unsupported row length: " + name);
    const int Kp = padded_k(K);
    const size_t prb = (size_t)Kp / blk * bytes;
    std::vector<uint8_t> v((size_t)Np * prb, 0);
    if (K % 256) pad_small_block_rows(t.type, t.data, N, K, v.data());
    else memcpy(v.data(), t.data, (size_t)N * prb);
    return make_weight(t.type, v.data(), v.size(), Np, Kp, weight_role(name));
  }
  // Flash Next mixer: the low-rank down rows [hc_lr][HCE] and the inject rows [hc_n][HCE] stacked (padded to 16 rows)
  // as one Q8_0 matrix: the GEMM gives both from the same normed input.
  GpuWeight load_hc_down(const std::string& down, const std::string& inject) {
    const int K = hp.hce(), LR = hp.hc_lr, NI = inject.empty() ? 0 : hp.hc_n, N = (LR + NI + 15) / 16 * 16;
    std::vector<float> v((size_t)N * K, 0.0f);
    const GgufTensor& td = f_.get(down);
    if ((int)td.dims[0] != K || (int)td.dims[1] != LR) throw std::runtime_error("unexpected tensor shape: " + down);
    dequant_row(td.type, td.data, (int64_t)LR * K, v.data());
    if (NI) {
      const GgufTensor& ti = f_.get(inject);
      if ((int)ti.dims[0] != K || (int)ti.dims[1] != NI) throw std::runtime_error("unexpected tensor shape: " + inject);
      dequant_row(ti.type, ti.data, (int64_t)NI * K, v.data() + (size_t)LR * K);
    }
    auto q8 = quantize_rows_q8_0(v.data(), N, K);
    return make_weight(GGML_Q8_0, q8.data(), q8.size(), N, K);
  }
  int hc_lr_pad() const { return padded_k(hp.hc_lr); }
  // Flash Next mixer up projection [HCE][hc_lr], its K padded with zeros to a multiple of 256
  GpuWeight load_hc_up(const std::string& name) {
    const GgufTensor& t = f_.get(name);
    const int K = hp.hc_lr, N = hp.hce(), KP = hc_lr_pad();
    if ((int)t.dims[0] != K || (int)t.dims[1] != N) throw std::runtime_error("unexpected tensor shape: " + name);
    std::vector<float> a((size_t)N * K), v((size_t)N * KP, 0.0f);
    dequant_row(t.type, t.data, (int64_t)N * K, a.data());
    for (int n = 0; n < N; ++n) memcpy(&v[(size_t)n * KP], &a[(size_t)n * K], (size_t)K * 4);
    auto q8 = quantize_rows_q8_0(v.data(), N, KP);
    return make_weight(GGML_Q8_0, q8.data(), q8.size(), N, KP);
  }

  void load_weights() {
    // The output head first: the GPU reads a large buffer allocated late in the load (after gigabytes of other
    // weights) at about half the speed (Gemma 4 12B's 826 MB head: 14 ms per token first, 21-28 ms last).
    load_head();
    if (hp.gemma) return load_weights_gemma4();
    layers_.resize(hp.n_layer);
    for (int il = 0; il < hp.n_layer; ++il) {
      Layer& L = layers_[il];
      L.il = il;
      const std::string p = "blk." + std::to_string(il) + ".";
      if (hp.hc) {
        for (int b = 0; b < 2; ++b) {
          const std::string hp_ = p + (b ? "hc_ffn" : "hc_attn");
          L.hc_norm[b] = load_f32(hp_ + "_norm.weight");
          L.hc_down[b] = load_hc_down(hp_ + "_down.weight", hp_ + "_inject.weight");
          L.hc_up[b] = load_hc_up(hp_ + "_up.weight");
        }
        if (il == hp.ple_layer) {
          L.ple_key = load_w(p + "ple_key.weight");
          L.ple_value = load_w(p + "ple_value.weight");
          L.ple_nk = load_f32(p + "ple_norm_key.weight");
          L.ple_nq = load_f32(p + "ple_norm_query.weight");
          L.ple_nc = load_f32(p + "ple_norm_conv.weight");
          L.ple_cw = load_f32(p + "ple_conv1d.weight");
          if (L.ple_key.N != hp.hce() || L.ple_key.K != hp.n_embd || L.ple_value.N != hp.n_embd)
            throw std::runtime_error("unexpected PLE block shape");
        }
      } else {
        L.attn_norm = load_f32(p + "attn_norm.weight");
        L.post_norm = load_f32(p + "post_attention_norm.weight");
      }
      if (hp.recurrent(il)) {
        load_gdn_in(L, p);
        L.ssm_out = load_w(p + "ssm_out.weight");
        L.dt_bias = load_f32(p + "ssm_dt.bias");
        L.ssm_a = load_f32(p + "ssm_a");
        L.conv_w = load_f32(p + "ssm_conv1d.weight");
        L.ssm_norm = load_f32(p + "ssm_norm.weight");
        L.conv_state = buf(conv_bytes());
        L.S = buf((size_t)hp.ssm_nv * hp.ssm_d_state * hp.ssm_d_state * 4);
        if (opt.mtp && !hp.gemma) {
          const size_t g0 = gpu_bytes, rows = 1 + kMaxDrafts;
          L.S_next = buf((size_t)hp.ssm_nv * hp.ssm_d_state * hp.ssm_d_state * 4);
          L.conv_pre = buf(conv_bytes());
          L.xin = buf(rows * hp.conv_dim() * 4);
          L.qkvin = buf(rows * hp.conv_dim() * 4);
          L.abin = buf(rows * 2 * hp.ssm_nv * 4);
          mem_mtp_ += gpu_bytes - g0;  // (the state of a verification pass, for rejected drafts)
        }
      } else {
        load_attn(L, p);
      }
      if (hp.moe) {
        load_moe(L, p);
      } else {
        L.gate = load_w(p + "ffn_gate.weight");
        L.up = load_w(p + "ffn_up.weight");
        L.down = load_w(p + "ffn_down.weight");
      }
      clFinish(cl_.queue);
    }
    // MTP (nextn) block: eh_proj + one gated-attention layer + FFN, sharing the output head
    const std::string mp = "blk." + std::to_string(hp.n_layer) + ".";
    const size_t g_mtp = gpu_bytes;
    struct MtpBytes { Engine& e; size_t g0; ~MtpBytes() { e.mem_mtp_ += e.gpu_bytes - g0; } } mtp_bytes{*this, g_mtp};
    if (opt.mtp && opt.mtp_block && f_.find(mp + "nextn.eh_proj.weight")) {
      has_mtp = mtp_block_ = true;
      mtp_.il = hp.n_layer;
      mtp_.attn_norm = load_f32(mp + "attn_norm.weight");
      mtp_.post_norm = load_f32(mp + "post_attention_norm.weight");
      load_attn(mtp_, mp);
      if (hp.moe) {
        load_moe(mtp_, mp);
      } else {
        mtp_.gate = load_w(mp + "ffn_gate.weight");
        mtp_.up = load_w(mp + "ffn_up.weight");
        mtp_.down = load_w(mp + "ffn_down.weight");
      }
      eh_proj_ = load_w(mp + "nextn.eh_proj.weight");
      enorm_ = load_f32(mp + "nextn.enorm.weight");
      hnorm_ = load_f32(mp + "nextn.hnorm.weight");
      shnorm_ = load_f32(mp + "nextn.shared_head_norm.weight");
      clFinish(cl_.queue);
    }
    // DFlash2 (with or without the MTP block: with both, mtp_draft picks one per step)
    if (!opt.dflash.empty()) load_dflash(opt.dflash);
  }

  // MoE FFN: the router and the shared-expert gate requantized together to Q8_0 (one GEMM gives both), the shared
  // expert's Q8_0 matrices, and the routed experts' 2-bit codes and bf16 scales as they are in the file.
  // ---------------------------------------------------------------- arcinferzero expert streaming
  void open_experts() {
    xc_ = std::make_unique<XCache>();
    std::vector<std::string> pre(hp.n_layer + 1);
    for (int il = 0; il <= hp.n_layer; ++il) {
      const std::string p = "blk." + std::to_string(il) + ".";
      if (f_.find(p + "ffn_gate_exps.weight") || f_.find(p + "ffn_gate_up_exps.weight")) pre[il] = p;
    }
    xc_->st.open_or_build(f_, path_, pre, hp.n_expert);
    // (staging records per half: ARC_XSTAGE, default 64 = 8 verification tokens x 8 experts)
    xc_->init(hp.n_layer + 1, hp.n_expert, getenv("ARC_XSTAGE") ? atoi(getenv("ARC_XSTAGE")) : 64);
    if (xc_->st.built)
      fprintf(stderr, "arcinferzero: expert file written in %.1f s\n", xc_->st.build_ms / 1000);
  }
  static uint64_t ram_total() {
    MEMORYSTATUSEX ms{sizeof ms};
    return GlobalMemoryStatusEx(&ms) ? ms.ullTotalPhys : 0;
  }
  void size_expert_cache() {
    const double GiB = 1073741824.0;
    const uint64_t all = xc_->all_bytes();
    uint64_t want;
    const char* env_gb = getenv("ARC_XCACHE_GB");
    if (env_gb || opt.expert_cache_gb > 0) {
      want = (uint64_t)((env_gb ? atof(env_gb) : opt.expert_cache_gb) * GiB);
    } else if (opt.expert_cache_pct > 0) {
      want = (uint64_t)(all * std::min(opt.expert_cache_pct, 100.0) / 100);
    } else {
      // the whole process (weights, buffers, transient area, cache) within RAM - reserve and within the GPU share
      // (minus room for the vision encoder and the driver)
      const uint64_t ram = ram_total(), reserve = (uint64_t)(opt.windows_reserve_gb * GiB);
      // (the GPU share less 1.4 GiB for the driver; the server's --ctx auto keeps 1.17. With 1.13 GiB left a kernel
      // launch failed with out of device memory, 35B at 262K)
      const uint64_t share = zrt::global_mem, slack = (uint64_t)(1.4 * GiB);
      uint64_t limit = std::min(ram > reserve ? ram - reserve : 0, share > slack ? share - slack : 0);
      // (+ kernels and host side, the server's state snapshots, and what the caller keeps for its encoders)
      const uint64_t used = gpu_bytes + xc_->stage_bytes + (uint64_t)(0.1 * GiB) + snapshot_bytes() * std::max(opt.slots, 1) + opt.stream_reserve;
      want = limit > used ? limit - used : 0;
      fprintf(stderr, "arcinferzero: memory limit %.2f GiB (RAM %.1f - Windows %.1f, GPU share %.2f), model + buffers %.2f GiB\n",
              limit / GiB, ram / GiB, reserve / GiB, share / GiB, used / GiB);
    }
    want = std::min(want, all);
    // at least one layer's worth of slots per pool (the experts of a few verification tokens must fit)
    want = std::max<uint64_t>(want, xc_->min_bytes());
    xc_->set_cache(want);
    gpu_bytes += xc_->cache_bytes;
    XCache::active() = xc_.get();
    zrt::oom_hook = [] { return XCache::active() && XCache::active()->shrink(); };
    // fill the cache before the first request: the experts the last run had cached (or the first of every layer)
    xc_->state_path = xc_->st.path.substr(0, xc_->st.path.size() - 8) + ".lru";
    if (!getenv("ARC_XWARM") || atoi(getenv("ARC_XWARM")) != 0) {
      xc_->warm();
      fprintf(stderr, "arcinferzero: cache filled from the SSD: %.2f GB in %.1f s\n", xc_->warm_bytes / 1e9, xc_->warm_ms / 1000);
    }
    zrt::print_memory();
    fprintf(stderr, "arcinferzero: expert cache %.2f GiB (%.0f %% of %.2f GiB experts, %d slots), staging %.0f MiB, file %s\n",
            xc_->cache_bytes / GiB, 100.0 * xc_->cache_bytes / all, all / GiB, xc_->slots(), xc_->stage_bytes / 1048576.0,
            xc_->st.path.c_str());
  }
  // After the routing of layer L: read the chosen experts and make them resident (see zero/xcache.h).
  // (tl1: one token's tiles when the caller has read them already)
  // (host time of streamed MoE, printed by expert_stats: the routing read (waits for the GPU), ensure (expert reads
  //  and copies while the GPU waits), and Flash Next's PLE rows)
  double hst_sync_ms_ = 0, hst_ensure_ms_ = 0, hst_ple_ms_ = 0;
  int seg_n_ = 0;  // tokens of the layer-major segment being processed (0: none)
  using hclk = std::chrono::steady_clock;
  static double hms(hclk::time_point a) { return std::chrono::duration<double, std::milli>(hclk::now() - a).count(); }
  void stream_layer(Layer& L, int T, bool gemm, const int* tl1 = nullptr) {
    const int NU = hp.n_expert_used;
    const auto h0 = hclk::now();
    static std::vector<int> ids;
    ids.resize((size_t)T * NU);
    std::vector<int> guess;
    if (T == 1) {
      std::vector<int> tl(1 + 3 * NU), tg;
      if (pf_next_) {  // (the guess of the next layer's routing, computed with this layer's)
        tg.resize(1 + 3 * NU);
        CL_CHECK(clEnqueueReadBuffer(cl_.queue, pf_tiles_, tl1 ? CL_TRUE : CL_FALSE, 0, tg.size() * 4, tg.data(), 0,
                                     nullptr, nullptr));
      }
      if (tl1)
        std::copy(tl1, tl1 + tl.size(), tl.begin());
      else
        CL_CHECK(clEnqueueReadBuffer(cl_.queue, mtiles_, CL_TRUE, 0, tl.size() * 4, tl.data(), 0, nullptr, nullptr));
      for (int k = 0; k < NU; ++k) ids[k] = tl[1 + 3 * k];
      if (pf_next_)
        for (int k = 0; k < NU; ++k) guess.push_back(tg[1 + 3 * k]);
    } else {
      CL_CHECK(clEnqueueReadBuffer(cl_.queue, mids_, CL_TRUE, 0, ids.size() * 4, ids.data(), 0, nullptr, nullptr));
    }
    const auto h1 = hclk::now();
    hst_sync_ms_ += std::chrono::duration<double, std::milli>(h1 - h0).count();
    xc_->ensure(L.il, ids.data(), ids.size(), T > 8, gemm);
    hst_ensure_ms_ += hms(h1);
    if (!guess.empty()) xc_->prefetch_ids(L.il + 1, guess.data(), (int)guess.size());
    pf_next_ = false;
    static const bool xcheck = getenv("ARC_XCHECK") != nullptr;  // (debug: every tile's expert address is in a pool)
    if (xcheck) {
      const int max_t = T * NU + hp.n_expert + 1;
      std::vector<int> tl(1 + 3 * (size_t)max_t);
      CL_CHECK(clEnqueueReadBuffer(cl_.queue, mtiles_, CL_TRUE, 0, tl.size() * 4, tl.data(), 0, nullptr, nullptr));
      const uint64_t* xa = xc_->xaddr + (size_t)L.il * hp.n_expert;
      auto valid = [&](uint64_t a) {
        for (auto& P : xc_->pools)
          for (size_t c = 0; c < P.chunks.size(); ++c) {
            const uint64_t b = (uint64_t)P.chunks[c];
            if (a >= b && a < b + (1ull << 30) + P.rec) return true;
          }
        return false;
      };
      if (tl[0] < 0 || tl[0] > max_t) fprintf(stderr, "XCHECK layer %d T %d: tile count %d\n", L.il, T, tl[0]);
      for (int t = 0; t < std::min(tl[0], max_t); ++t) {
        const int e = tl[1 + 3 * t];
        if (e < 0 || e >= hp.n_expert || !valid(xa[e])) {
          fprintf(stderr, "XCHECK layer %d T %d tile %d/%d: expert %d address %llx not in the cache\n", L.il, T, t, tl[0], e,
                  e >= 0 && e < hp.n_expert ? (unsigned long long)xa[e] : 0ull);
          fflush(stderr);
          _Exit(3);
        }
      }
      if (gemm)
        for (int e = 0; e < hp.n_expert; ++e)
          if (!valid(xa[e])) { fprintf(stderr, "XCHECK layer %d gemm: expert %d address invalid\n", L.il, e); _Exit(3); }
    }
    // prompt chunks use nearly every expert (128 tokens: ~98 %): read the next layer's missing ones while the GPU
    // works on this one (not across the chunk's end: the next chunk's layer 0 has this layer's parity at times)
    static const int pf_min = getenv("ARC_XPF_MIN") ? atoi(getenv("ARC_XPF_MIN")) : 128;
    const int next = L.il + 1;
    // (the tokens that go through the next layer before the layer after it: a layer-major segment's, else the chunk's)
    if (T >= pf_min && (next < hp.n_layer || (next == hp.n_layer && has_mtp))) xc_->prefetch(next, seg_n_ ? seg_n_ : T);
  }
 public:
  // expert cache statistics (empty without streaming)
  std::string expert_stats() const {  // (+ host times, see stream_layer)
    if (!xc_) return "";
    const XCache& x = *xc_;
    char b[512];
    snprintf(b, sizeof b,
             "experts: generation %llu hits / %llu misses (%.1f %% hit) | prompt %llu cached, %llu read (%llu prefetched) | "
             "SSD %.2f GB in %.0f ms (%.2f GB/s) | next-layer guesses %llu read, %llu used",
             (unsigned long long)x.hits, (unsigned long long)x.misses, 100.0 * x.hits / std::max<uint64_t>(1, x.hits + x.misses),
             (unsigned long long)x.p_hits, (unsigned long long)x.p_reads, (unsigned long long)x.pf_reads,
             x.st.bytes_read / 1e9, x.read_ms, x.st.bytes_read / 1e6 / std::max(1e-9, x.read_ms),
             (unsigned long long)x.g_pf, (unsigned long long)x.g_pf_used);
    std::string r = b;
    char h[160];
    snprintf(h, sizeof h, "\nhost: routing reads %.0f ms (waiting for the GPU), ensure %.0f ms, PLE rows %.0f ms",
             hst_sync_ms_, hst_ensure_ms_, hst_ple_ms_);
    return r + h;
  }
 private:
  // part offsets of an expert record (gate, up, down)
  int xoff(const Layer& L, int i) const {
    const auto& li = xc_->st.layers[L.il];
    return i == 0 ? 0 : i == 1 ? (int)li.part[0] : (int)(li.part[0] + li.part[1]);
  }
  // decode_w8 of experts [e0, e0 + G) of matrix i from their records (moe_gemm)
  void decode_w8_x(const Layer& L, int i, int e0, int G, int N, int K) {
    const uint32_t type = L.ex_type[i];
    const int nsb = K / 256;
    const bool sl = !getenv("ARC_NO_W8_SL") && !w8_requant(type) && nsb <= 16 && N % 16 == 0 &&
                    16 * nsb * raw_padded_sb_bytes(type) <= 32768 && type != GGML_IQ1_M;
    cl_mem tab = xc_->xtab[L.il];
    int NN = G * N, xo = xoff(L, i);
    if (sl) {
      cl_kernel k = kernel("decode", "-DXCACHE -DQT=" + std::to_string(type) + " -DSL_NSB=" + std::to_string(nsb), "decode_w8_sl");
      set_args(k, 0, tab, NN, K, tabs_, moe_wq_, moe_wh_, tab, e0, N, xo);
      size_t g1 = (size_t)NN * nsb, l1 = (size_t)16 * nsb;
      run("decode_w8", k, 1, &g1, &l1);
      return;
    }
    size_t lx = 256;
    while (NN % lx) lx /= 2;
    cl_kernel k = kernel("decode", "-DXCACHE -DQT=" + std::to_string(type), "decode_w8_sb");
    set_args(k, 0, tab, NN, K, tabs_, moe_wq_, moe_wh_, tab, e0, N, xo);
    size_t gws[2] = {(size_t)NN, (size_t)nsb}, lws[2] = {lx, 1};
    run("decode_w8", k, 2, gws, lws);
  }

  void load_moe(Layer& L, const std::string& p) {
    const int E = hp.n_embd, NR = hp.router_rows();
    std::vector<float> v((size_t)NR * E, 0.0f);
    const GgufTensor& r = f_.get(p + "ffn_gate_inp.weight");
    const GgufTensor& sg = f_.get(p + "ffn_gate_inp_shexp.weight");
    if (r.nelements() != (uint64_t)hp.n_expert * E || sg.nelements() != (uint64_t)E)
      throw std::runtime_error("unexpected MoE router shape");
    dequant_row(r.type, r.data, hp.n_expert * E, v.data());
    dequant_row(sg.type, sg.data, E, v.data() + (size_t)hp.n_expert * E);
    auto q8 = quantize_rows_q8_0(v.data(), NR, E);
    const GgufTensor& tg = f_.get(p + "ffn_gate_shexp.weight");
    const GgufTensor& tu = f_.get(p + "ffn_up_shexp.weight");
    const int Fs = hp.n_ff_shexp, NRP = rgu_rows();
    if (tg.type == GGML_Q8_0 && tu.type == GGML_Q8_0 && E % 256 == 0 && tg.nelements() == (uint64_t)Fs * E &&
        tu.nelements() == (uint64_t)Fs * E && !(w8_resident && !hp.moe)) {
      // router + shared-expert gate + up rows in one Q8_0 buffer: single tokens run one GEMV over all of them
      // (Layer::rgu); the three matrices are sub-buffers of it (stored once; their offsets are 512-byte aligned)
      const size_t rowb = (size_t)E / 32 * 34, sbr = (size_t)E / 256 * raw_padded_sb_bytes(GGML_Q8_0);
      std::vector<uint8_t> cat((size_t)(NRP + 2 * Fs) * rowb, 0);
      memcpy(cat.data(), q8.data(), q8.size());
      memcpy(cat.data() + (size_t)NRP * rowb, tg.data, tg.nbytes);
      memcpy(cat.data() + (size_t)(NRP + Fs) * rowb, tu.data, tu.nbytes);
      L.rgu.type = GGML_Q8_0;
      L.rgu.N = NRP + 2 * Fs;
      L.rgu.K = E;
      L.rgu.role = kRoles - 1;
      L.rgu.raw = upload_padded(GGML_Q8_0, cat.data(), cat.size());
      has_rgu_ = true;
      auto view = [&](int row0, int N) {
        GpuWeight w;
        w.type = GGML_Q8_0;
        w.gemm_qtype = (int)GGML_Q8_0;  // (prompt chunks: decoded just in time, as make_weight does for MoE models)
        w.N = N;
        w.K = E;
        w.role = kRoles - 1;
        stage_need_q_[w.role] = std::max(stage_need_q_[w.role], w8_bytes_q(N, E));
        stage_need_h_[w.role] = std::max(stage_need_h_[w.role], w8_bytes_h(N, E));
        if (L.rgu.raw) {  // (+256 bytes: the GEMV's block reads run a little past a row's end, as upload_padded allows)
          cl_buffer_region reg{(size_t)row0 * sbr, (size_t)N * sbr + 256};
          cl_int err;
          w.raw = clCreateSubBuffer(L.rgu.raw, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &reg, &err);
          CL_CHECK(err);
          mems_.push_back(w.raw);
        }
        return w;
      };
      L.router = view(0, NR);
      L.gate_sh = view(NRP, Fs);
      L.up_sh = view(NRP + Fs, Fs);
    } else {
      L.router = make_weight(GGML_Q8_0, q8.data(), q8.size(), NR, E);
      L.gate_sh = load_w_pad(p + "ffn_gate_shexp.weight", Fs);
      L.up_sh = load_w_pad(p + "ffn_up_shexp.weight", Fs);
    }
    L.down_sh = load_w_pad(p + "ffn_down_shexp.weight", E);
    if (L.gate_sh.N != Fs || L.down_sh.K != Fs) throw std::runtime_error("unexpected shared expert shape");
    const char* names[3] = {"gate", "up", "down"};
    const uint64_t per_expert = (uint64_t)hp.n_ff_exp * E;  // weights of one expert matrix
    if (hp.moe_std) {
      for (int i = 0; i < 3; ++i) {
        const GgufTensor& t = f_.get(p + "ffn_" + names[i] + "_exps.weight");
        const uint64_t K = t.dims[0];
        if (t.dims.size() != 3 || t.dims[2] != (uint64_t)hp.n_expert || t.nelements() != per_expert * hp.n_expert ||
            K != (i == 2 ? (uint64_t)hp.n_ff_exp : (uint64_t)E) || (K % 256 && !is_small_block_type(t.type)))
          throw std::runtime_error("unexpected expert tensor " + t.name);
        if (!is_w8_type(t.type))
          throw std::runtime_error("unsupported expert tensor " + t.name + " (" + ggml_type_name(t.type) + ")");
        if (K % 256 && !stream_) throw std::runtime_error("this model runs with expert streaming only");
        if (!stream_) L.ex_q[i] = upload_padded(t.type, t.data, t.nbytes);
        else stream_bytes_ += t.dims[1] * t.dims[2] * (padded_k((int)K) / 256) * raw_padded_sb_bytes(t.type);
        L.ex_type[i] = t.type;
      }
      if (L.ex_type[0] != L.ex_type[1]) throw std::runtime_error("MoE: gate and up experts of different types");
      return;
    }
    for (int i = 0; i < 3; ++i) {
      const GgufTensor& q = f_.get(p + "ffn_" + names[i] + "_exps.weight");
      const GgufTensor& s = f_.get(p + "ffn_" + names[i] + "_exps.scale");
      if (q.type != GGML_I32 || s.type != GGML_BF16 || q.nbytes != per_expert / 4 * hp.n_expert ||
          s.nbytes != per_expert / 64 * hp.n_expert)
        throw std::runtime_error("unexpected expert tensor " + q.name);
      L.ex_q[i] = buf(q.nbytes, q.data);
      L.ex_s[i] = buf(s.nbytes, s.data);
    }
  }

  // Gemma 4 26B-A4B experts of layer L (see layer_gemma_ffn): router rows padded to router_rows() (zeros past the
  // experts), the norms and scales, the experts (streamed: the expert file; else gate / up taken apart from
  // ffn_gate_up_exps, the down rows padded to whole superblocks)
  void load_moe_gemma(Layer& L, const std::string& p) {
    const int E = hp.n_embd, NR = hp.router_rows(), NE = hp.n_expert, F = hp.n_ff_exp;
    const GgufTensor& r = f_.get(p + "ffn_gate_inp.weight");
    if (r.nelements() != (uint64_t)NE * E) throw std::runtime_error("unexpected Gemma 4 router shape");
    // (f32 rows, read by gemv_f32 with f32 inputs; router_rows() of them, zeros past the experts)
    std::vector<float> v((size_t)NR * E, 0.0f);
    dequant_row(r.type, r.data, NE * E, v.data());
    L.router_f32 = buf(v.size() * 4, v.data());
    L.router.N = NR;
    L.router.K = E;
    {
      const GgufTensor& t = f_.get(p + "ffn_gate_inp.scale");
      std::vector<float> w(E);
      dequant_row(t.type, t.data, E, w.data());
      for (float& x : w) x /= std::sqrt((float)E);  // (llama.cpp gemma4: rms_norm(x) / sqrt(n_embd) * scale)
      L.rnorm = buf((size_t)E * 4, w.data());
    }
    {
      const GgufTensor& t = f_.get(p + "ffn_down_exps.scale");
      std::vector<float> w(NE);
      dequant_row(t.type, t.data, NE, w.data());
      L.escale = buf((size_t)NE * 4, w.data());
    }
    L.post_ffw1 = load_f32(p + "post_ffw_norm_1.weight");
    L.ffn_norm2 = load_f32(p + "pre_ffw_norm_2.weight");
    L.post_ffw2 = load_f32(p + "post_ffw_norm_2.weight");
    for (int i = 0; i < 3; ++i) {
      const ExpertStore::Src m = ExpertStore::src(f_, p, i);
      const GgufTensor& t = *m.t;
      const int K = (int)t.dims[0];
      if (t.dims.size() != 3 || t.dims[2] != (uint64_t)NE || m.rows != (uint64_t)(i == 2 ? E : F) || K != (i == 2 ? F : E) ||
          !is_w8_type(t.type))
        throw std::runtime_error("unexpected Gemma 4 expert tensor " + t.name);
      L.ex_type[i] = t.type;
      const size_t rows = (size_t)NE * m.rows, prow = (size_t)padded_k(K) / 256 * gguf_sb_bytes(t.type);
      if (stream_) {
        stream_bytes_ += rows * (padded_k(K) / 256) * raw_padded_sb_bytes(t.type);
        continue;
      }
      if (opt.dry) {
        gpu_bytes += rows * (padded_k(K) / 256) * raw_padded_sb_bytes(t.type) + kRawSlack;
        continue;
      }
      int blk = 0, bytes = 0;
      ggml_block_info(t.type, blk, bytes);
      const size_t rb = (size_t)K / blk * bytes;  // (GGUF row bytes)
      std::vector<uint8_t> all(rows * prow);
      for (int e = 0; e < NE; ++e) {
        const uint8_t* src = t.data + ((size_t)e * m.per + m.row0) * rb;
        uint8_t* dst = all.data() + (size_t)e * m.rows * prow;
        if (K % 256) pad_small_block_rows(t.type, src, (int)m.rows, K, dst);
        else memcpy(dst, src, m.rows * rb);
      }
      L.ex_q[i] = upload_padded(t.type, all.data(), all.size());
    }
    if (L.ex_type[0] != L.ex_type[1]) throw std::runtime_error("MoE: gate and up experts of different types");
  }

  void load_attn(Layer& L, const std::string& p) {
    {
      {
        L.wq = load_w(p + "attn_q.weight");
        L.wk = load_w(p + "attn_k.weight");
        L.wv = load_w(p + "attn_v.weight");
        L.wo = load_w(p + "attn_output.weight");
        L.q_norm = load_f32(p + "attn_q_norm.weight");
        L.k_norm = load_f32(p + "attn_k_norm.weight");
        const size_t kvb = (size_t)hp.n_head_kv * ctx * hp.head_dim * 2;
        // int8 attention: K/V int8 in 16-key blocks + per-row scales (Qwen: SB = 1, sb_opt); f32 attention: f16 K/V rows
        const size_t kvi = opt.attn_int8 ? kvb : 64, scb = opt.attn_int8 ? (size_t)hp.n_head_kv * ctx * 4 : 64;
        L.kc = buf(opt.attn_int8 ? 64 : kvb);
        L.vc = buf(opt.attn_int8 ? 64 : kvb);
        L.kq = buf(kvi / 2);
        L.vq = buf(kvi / 2);
        if (opt.v4 && opt.attn_int8) {  // (--v4: the 4-bit copy, half the bytes; a scale per 64 dims)
          L.vq4 = buf(kvi / 4);
          L.vsc4 = buf(scb * (hp.head_dim / 64));
          fill(L.vq4, kvi / 4);
          fill(L.vsc4, scb * (hp.head_dim / 64));
        }
        L.ksc = buf(scb);
        L.vsc = buf(scb);
        for (cl_mem m : {L.kc, L.vc}) fill(m, opt.attn_int8 ? 64 : kvb);
        for (cl_mem m : {L.kq, L.vq}) fill(m, kvi / 2);
        for (cl_mem m : {L.ksc, L.vsc}) fill(m, scb);
      }
    }
    if (hp.hc && L.il < (int)hp.qsa_ratio.size() && hp.qsa_ratio[L.il] > 0) {
      L.qr = hp.qsa_ratio[L.il];
      L.ix = load_q8_concat(p + "indexer.k_proj.weight", p + "indexer.q_proj.weight");
      L.ix_kn = load_f32(p + "indexer.k_norm.weight");
      L.ix_qn = load_f32(p + "indexer.q_norm.weight");
      if (L.ix.N != (1 + hp.qsa_heads) * hp.qsa_dim) throw std::runtime_error("unexpected QSA indexer shape");
      L.pk = buf((size_t)qsa_blocks(ctx) * hp.qsa_dim * 4 + 64);
      L.qsa_i = qsa_n_++;
      if (!opt.dry) {  // its tail raw keys inside the PLE layer's recurrent state (see conv_bytes)
        cl_buffer_region r{qsa_tail_off(L.qsa_i), (size_t)(L.qr - 1) * hp.qsa_dim * 4};
        cl_int err;
        L.qtail = clCreateSubBuffer(layers_[hp.ple_layer].conv_state, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &r, &err);
        CL_CHECK(err);
        mems_.push_back(L.qtail);
      }
    }
  }

  void load_weights_gemma4() {
    layers_.resize(hp.n_layer);
    cl_mem rope_ff = f_.find("rope_freqs.weight") ? load_f32("rope_freqs.weight") : nullptr;
    // sliding-window layers keep a ring of the last n_swa + chunk (+ room to rewind an agent's reply) positions.
    // Models with shared KV layers read another layer's ring later in the pass, so they keep full caches.
    swa_cap_ = hp.n_layer_kv < hp.n_layer ? ctx : std::min(ctx, (hp.n_swa + max_chunk + 4096 + 255) / 256 * 256);
    if (hp.n_ple) {
      ple_tok_embd_ = &f_.get("per_layer_token_embd.weight");
      ple_model_proj_ = load_w("per_layer_model_proj.weight");
      ple_proj_norm_ = load_f32("per_layer_proj_norm.weight");
    }
    for (int il = 0; il < hp.n_layer; ++il) {
      Layer& L = layers_[il];
      L.il = il;
      const std::string p = "blk." + std::to_string(il) + ".";
      L.swa = hp.swa[il] != 0;
      L.nkv = hp.kv_heads[il];
      L.hd = L.swa ? hp.hd_swa : hp.hd_full;
      L.cap = L.swa ? swa_cap_ : ctx;
      L.attn_norm = load_f32(p + "attn_norm.weight");
      L.post_attn = load_f32(p + "post_attention_norm.weight");
      L.ffn_norm = load_f32(p + "ffn_norm.weight");
      L.post_ffw = load_f32(p + "post_ffw_norm.weight");
      L.wq = load_w(p + "attn_q.weight");
      if (il >= hp.n_layer_kv) L.kv_src = hp.n_layer_kv - (L.swa ? 2 : 1);  // llama.cpp's KV reuse rule
      if (L.kv_src < 0) {
        L.wk = load_w(p + "attn_k.weight");
        L.has_v = f_.find(p + "attn_v.weight") != nullptr;
        if (L.has_v) L.wv = load_w(p + "attn_v.weight");
      }
      L.wo = load_w(p + "attn_output.weight");
      L.q_norm = load_f32(p + "attn_q_norm.weight");
      // (shared-KV layers compute no K: Google's GGUFs omit their attn_k_norm; attn_prep_g then reads only Q heads)
      L.k_norm = L.kv_src >= 0 && !f_.find(p + "attn_k_norm.weight") ? L.q_norm : load_f32(p + "attn_k_norm.weight");
      if (!L.swa) L.rope_ff = rope_ff;
      if (f_.find(p + "layer_output_scale.weight")) {
        const GgufTensor& t = f_.get(p + "layer_output_scale.weight");
        float v = 1.0f;
        dequant_row(t.type, t.data, 1, &v);
        L.out_scale = v;
      }
      if (L.kv_src >= 0) {
        const Layer& S = layers_[L.kv_src];
        if (S.hd != L.hd || S.nkv != L.nkv || S.swa != L.swa) throw std::runtime_error("Gemma 4: incompatible shared KV layer");
        L.kq = S.kq; L.vq = S.vq; L.ksc = S.ksc; L.vsc = S.vsc; L.cap = S.cap;
        L.kc = S.kc; L.vc = S.vc;
      } else {
        const size_t kvb = (size_t)L.nkv * L.cap * L.hd, scb = (size_t)L.nkv * L.cap * (L.hd / 64) * 4;
        L.kq = buf(kvb);
        L.vq = buf(kvb);
        L.ksc = buf(scb);
        L.vsc = buf(scb);
        for (cl_mem m : {L.kq, L.vq}) fill(m, kvb);
        for (cl_mem m : {L.ksc, L.vsc}) fill(m, scb);
        if (q2_) {  // (two-pass mode: f16 K / V rows for attn_fa_g; the int8 cache stays for the drafter)
          L.kc = buf(kvb * (q2_kf32_ ? 4 : 2));
          L.vc = buf(kvb * 2);
          fill(L.kc, kvb * (q2_kf32_ ? 4 : 2));
          fill(L.vc, kvb * 2);
        }
      }
      if (hp.n_ple) {
        L.ple_gate = load_w(p + "inp_gate.weight");
        L.ple_proj = load_w(p + "proj.weight");
        L.ple_post = load_f32(p + "post_norm.weight");
      }
      L.gate = load_w_pad(p + "ffn_gate.weight", hp.n_ff);
      L.up = load_w_pad(p + "ffn_up.weight", hp.n_ff);
      L.down = load_w_pad(p + "ffn_down.weight", hp.n_embd);
      if (hp.moe) load_moe_gemma(L, p);
      clFinish(cl_.queue);
    }
    if (opt.mtp) {
      std::string dpath = opt.mtp_model;
      if (dpath.empty()) dpath = find_gemma_drafter();
      if (!dpath.empty()) {
        const size_t g0 = gpu_bytes;
        load_gemma_drafter(dpath);
        mem_mtp_ += gpu_bytes - g0;
      }
    }
  }

  // An assistant drafter for this model next to it: mtp-*.gguf of arch gemma4-assistant with the same hidden size.
  std::string find_gemma_drafter() const {
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path dir = fs::path(path_).parent_path();
    for (const auto& e : fs::directory_iterator(dir.empty() ? fs::path(".") : dir, ec)) {
      const std::string n = e.path().filename().string();
      if (n.rfind("mtp-", 0) != 0 || e.path().extension() != ".gguf") continue;
      try {
        GgufFile g(e.path().string());
        const std::string a = g.get_str("general.architecture");
        if (a == "gemma4-assistant" && g.get_int(a + ".embedding_length_out", -1) == hp.n_embd) return e.path().string();
      } catch (...) {
      }
    }
    return "";
  }

  // GEMV-only matrix (raw rows; single tokens only), requantized to Q8_0 when the GEMV has no kernel for its type
  GpuWeight gemv_weight(const GgufFile& g, const std::string& name) {
    const GgufTensor& t = g.get(name);
    GpuWeight w;
    w.K = (int)t.dims[0];
    w.N = (int)t.dims[1];
    if (t.type == GGML_Q8_0 || is_w8_type(t.type)) {
      w.type = t.type;
      w.raw = upload_padded(t.type, t.data, t.nbytes);
      return w;
    }
    std::vector<uint8_t> q8;
    const size_t rb = t.nbytes / w.N;
    std::vector<float> v;
    for (int r0 = 0; r0 < w.N; r0 += 8192) {  // in row chunks (the drafter's lm head has 262144 rows)
      const int n = std::min(8192, w.N - r0);
      v.resize((size_t)n * w.K);
#pragma omp parallel for
      for (int r = 0; r < n; ++r) dequant_row(t.type, t.data + (size_t)(r0 + r) * rb, w.K, &v[(size_t)r * w.K]);
      auto q = quantize_rows_q8_0(v.data(), n, w.K);
      q8.insert(q8.end(), q.begin(), q.end());
    }
    w.type = GGML_Q8_0;
    w.raw = upload_padded(GGML_Q8_0, q8.data(), q8.size());
    return w;
  }
  cl_mem f32_from(const GgufFile& g, const std::string& name) {
    const GgufTensor& t = g.get(name);
    std::vector<float> v(t.nelements());
    dequant_row(t.type, t.data, (int)v.size(), v.data());
    return buf(v.size() * 4, v.data());
  }

  void load_gemma_drafter(const std::string& path) {
    df_ = std::make_unique<GgufFile>(path);
    const GgufFile& d = *df_;
    const std::string a = d.get_str("general.architecture");
    auto I = [&](const std::string& k) { return (int)d.get_int(a + "." + k, -1); };
    const int nl = I("block_count");
    d_E_ = I("embedding_length");
    d_F_ = I("feed_forward_length");
    d_nh_ = I("attention.head_count");
    d_swa_ = (int)d.get_int(a + ".attention.sliding_window", hp.n_swa);
    d_base_ = (float)d.get_float(a + ".rope.freq_base", 1e6);
    d_base_swa_ = (float)d.get_float(a + ".rope.freq_base_swa", 1e4);
    const auto* pat = d.get_ints(a + ".attention.sliding_window_pattern");
    const int Ls = hp.n_layer - 2, Lg = hp.n_layer - 1;  // main layers whose KV the drafter reads (llama.cpp rule)
    if (a != "gemma4-assistant" || I("embedding_length_out") != hp.n_embd || d_nh_ <= 0 ||
        d_nh_ % layers_[Ls].nkv || d_nh_ % layers_[Lg].nkv || d_swa_ > hp.n_swa || !pat || (int)pat->size() < nl ||
        d_E_ % 256 || d_F_ % 256 || !hp.swa[Ls] || hp.swa[Lg])
      throw std::runtime_error("unsupported Gemma 4 drafter " + path);
    d_pre_ = gemv_weight(d, "nextn.pre_projection.weight");
    d_post_ = gemv_weight(d, "nextn.post_projection.weight");
    d_lm_ = gemv_weight(d, "token_embd.weight");
    d_norm_ = f32_from(d, "output_norm.weight");
    cl_mem rope_ff = d.find("rope_freqs.weight") ? f32_from(d, "rope_freqs.weight") : nullptr;
    dl_.resize(nl);
    for (int il = 0; il < nl; ++il) {
      DraftLayer& D = dl_[il];
      const std::string p = "blk." + std::to_string(il) + ".";
      D.swa = (*pat)[il] != 0;
      D.hd = D.swa ? hp.hd_swa : hp.hd_full;
      D.wq = gemv_weight(d, p + "attn_q.weight");
      D.wo = gemv_weight(d, p + "attn_output.weight");
      D.gate = gemv_weight(d, p + "ffn_gate.weight");
      D.up = gemv_weight(d, p + "ffn_up.weight");
      D.down = gemv_weight(d, p + "ffn_down.weight");
      D.attn_norm = f32_from(d, p + "attn_norm.weight");
      D.q_norm = f32_from(d, p + "attn_q_norm.weight");
      D.post_attn = f32_from(d, p + "post_attention_norm.weight");
      D.ffn_norm = f32_from(d, p + "ffn_norm.weight");
      D.post_ffw = f32_from(d, p + "post_ffw_norm.weight");
      if (!D.swa) D.rope_ff = rope_ff;
      if (d.find(p + "layer_output_scale.weight")) {
        const GgufTensor& t = d.get(p + "layer_output_scale.weight");
        dequant_row(t.type, t.data, 1, &D.out_scale);
      }
      if (D.wq.N != d_nh_ * D.hd || D.hd != layers_[D.swa ? Ls : Lg].hd)
        throw std::runtime_error("Gemma 4 drafter: attention shape differs from the main model");
    }
    has_mtp = gemma_mtp_ = true;
    clFinish(cl_.queue);
  }

  void load_head() {
    if (hp.hc) {
      out_hc_norm_ = load_f32("output_hc_norm.weight");
      out_hc_down_ = load_hc_down("output_hc_down.weight", "");
      out_hc_up_ = load_hc_up("output_hc_up.weight");
      if (hp.ple_layer >= 0) {
        ple_table_ = &f_.get("per_layer_token_embd.weight");
        if ((int)ple_table_->dims[0] != hp.ple_dim) throw std::runtime_error("unexpected PLE table shape");
      }
    } else {
      out_norm_ = load_f32("output_norm.weight");
    }
    // lm_head is only ever a GEMV on the last token: keep just the raw rows.
    const GgufTensor& lt = f_.find("output.weight") ? f_.get("output.weight") : *tok_embd_;
    lm_head_.N = (int)lt.dims[1];
    lm_head_.K = (int)lt.dims[0];
    if (lt.type == GGML_F32 || lt.type == GGML_F16 || lt.type == GGML_BF16) {
      std::vector<float> v((size_t)lm_head_.N * lm_head_.K);
      dequant_row(lt.type, lt.data, (int)v.size(), v.data());
      auto q8 = quantize_rows_q8_0(v.data(), lm_head_.N, lm_head_.K);
      lm_head_.type = GGML_Q8_0;
      lm_head_.raw = upload_padded(GGML_Q8_0, q8.data(), q8.size());
    } else {
      lm_head_.type = lt.type;
      lm_head_.raw = upload_padded(lt.type, lt.data, lt.nbytes);
    }
  }

  void alloc_buffers() {
    const size_t M = max_chunk;
    const int maxK = std::max({hp.n_embd, hp.n_ff, hp.ssm_d_inner, hp.n_head * hp.head_dim, hp.hce(), hp.n_ff_exp_pad});
    const int ldq = hp.n_head * hp.head_dim * 2 + 2 * hp.n_head_kv * hp.head_dim;
    const int big_w = std::max({hp.conv_dim(), 2 * hp.n_ff, ldq});
    x_ = buf(M * hp.hce() * 4);
    aq_ = buf(M * maxK * (q2_ ? 2 : 1));
    if (hp.hc) {
      hc_xn_ = buf(M * hp.hce() * 4);
      hc_gate_ = buf(M * hp.hce() * 4);
      hc_lo_ = buf(M * (size_t)std::max(layers_[0].hc_down[0].N, out_hc_down_.N) * 4);
      hco_ = buf(M * hp.n_embd * 4);
      if (hp.n_qsa) {
        qsa_ixo_ = buf(M * (size_t)(1 + hp.qsa_heads) * hp.qsa_dim * 4);
        qsa_qx_ = buf(M * (size_t)hp.qsa_heads * hp.qsa_dim * 4);
        qsa_sc_ = buf((size_t)256 * qsa_blocks(ctx) * 4 + 64);
        qsa_mask_ = buf(M * (size_t)qsa_mask_words(ctx) * 4 + 64);
      }
      if (hp.ple_layer >= 0) {
        ple_emb_ = buf(M * hp.n_embd * 4);
        ple_host_.resize(M * hp.n_embd);
        if (!opt.dry) {  // the PLE conv history inside the PLE layer's recurrent state (see conv_bytes)
          cl_buffer_region r{ple_hist_off(), (size_t)ple_hist() * hp.hce() * 4};
          cl_int err;
          ple_hist_ = clCreateSubBuffer(layers_[hp.ple_layer].conv_state, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
                                        &r, &err);
          CL_CHECK(err);
          mems_.push_back(ple_hist_);
        }
      }
    }
    asc_ = buf(M * (maxK / 32) * 4 * (q2_ ? 2 : 1));
    absum_ = buf(M * (maxK / 16) * 2 * (q2_ ? 2 : 1));
    if (hp.gemma && q2_) {  // (two passes: the residual at the same offsets as in aq_ / asc_ / absum_)
      aqB_ = buf(M * maxK * 2);
      ascB_ = buf(M * (maxK / 32) * 4 * 2);
      absumB_ = buf(M * (maxK / 16) * 2 * 2);
    } else if (hp.gemma) {  // 16 padded rows, as single-token and verification passes use
      aqB_ = buf((size_t)16 * hp.n_ff);
      ascB_ = buf((size_t)16 * (hp.n_ff / 32) * 4);
      absumB_ = buf((size_t)16 * (hp.n_ff / 16) * 2);
    }
    big_ = buf(M * big_w * 4);
    // MoE: the expert activations / outputs (mh_ [slots][F], my_ [slots][E] f32) share memory with buffers that are
    // only used inside the attention / DeltaNet part of a layer (z_; conv_out_, o_, qrope_): never live at the same time
    // (Qwen3.6 35B, 2048-token chunks: 167 MB, ~18K tokens of context)
    const size_t z_bytes = std::max<size_t>(M * hp.ssm_d_inner * 4, 64);
    const size_t cv_bytes = (std::max<size_t>(M * hp.conv_dim() * 4, 64) + 4095) / 4096 * 4096,
                 o_bytes = (M * std::max(hp.ssm_d_inner, hp.n_head * hp.head_dim) * 4 + 4095) / 4096 * 4096,
                 qr_bytes = M * hp.n_head * hp.head_dim * 4;
    static const bool no_share = getenv("ARC_NO_MOE_SHARE") != nullptr;  // (A/B)
    const bool share = hp.moe && !no_share;
    auto sub = [&](cl_mem parent, size_t off, size_t bytes) -> cl_mem {
      if (opt.dry) return nullptr;
      cl_int err;
      cl_buffer_region r{off, bytes};
      cl_mem m = clCreateSubBuffer(parent, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &r, &err);
      CL_CHECK(err);
      mems_.push_back(m);
      return m;
    };
    if (share) {
      const size_t S = M * hp.n_expert_used, hb = S * hp.n_ff_exp_pad * 4, yb = S * hp.n_embd * 4;
      cl_mem a = buf(std::max(z_bytes, hb));
      z_ = sub(a, 0, z_bytes);
      mh_ = sub(a, 0, hb);
      cl_mem b = buf(std::max(cv_bytes + o_bytes + qr_bytes, yb));
      conv_out_ = sub(b, 0, cv_bytes);
      o_ = sub(b, cv_bytes, o_bytes);
      qrope_ = sub(b, cv_bytes + o_bytes, qr_bytes);
      my_ = sub(b, 0, yb);
    } else {
      z_ = buf(z_bytes);
    }
    if (has_qz_) {
      const size_t C = hp.conv_dim(), Di = hp.ssm_d_inner, nab = 2 * hp.ssm_nv;
      qz_out_ = buf((C + Di + nab + 64) * 4);
      if (!opt.dry) {
        cl_int err;
        cl_buffer_region rz{C * 4, Di * 4}, ra{(C + Di) * 4, (nab + 64) * 4};
        qz_z_ = clCreateSubBuffer(qz_out_, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &rz, &err);
        CL_CHECK(err);
        mems_.push_back(qz_z_);
        qz_ab_ = clCreateSubBuffer(qz_out_, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &ra, &err);
        CL_CHECK(err);
        mems_.push_back(qz_ab_);
      }
    }
    if (!share) {
      conv_out_ = buf(std::max<size_t>(M * hp.conv_dim() * 4, 64));
      o_ = buf(M * std::max(hp.ssm_d_inner, hp.n_head * hp.head_dim) * 4);
      qrope_ = buf(M * hp.n_head * hp.head_dim * 4);
    }
    gb_ = buf(std::max<size_t>(M * 2 * hp.ssm_nv * 4, 64));
    qi_ = buf(M * hp.n_head * hp.head_dim);
    if (q2_) qf_ = buf(M * hp.n_head * hp.head_dim * 4);  // (two-pass mode: the f32 queries of attn_fa_g)
    qsc_ = buf(M * hp.n_head * (hp.head_dim / 64) * 4);  // q scales per 64-dim block
    if (hp.gemma) {
      rope_tab_[0] = buf(M * (hp.hd_swa / 2) * 8);
      rope_tab_[1] = buf(M * (hp.hd_full / 2) * 8);
    }
    // decode key-split partials: up to 4 rows x splits (small splits only while splits * sub-groups < 512)
    apart_ = buf((size_t)4 * std::max(ctx / 256 + 1, 512) * hp.n_head * (hp.head_dim + 2) * 4);
    // rows of a speculative verification pass or a batched pass over the slots
    batch_rows_ = std::min(16, std::max(1 + max_drafts(), std::max(opt.slots, 1)));
    logits_ = buf((size_t)hp.n_vocab * 4 * batch_rows_);
    if (hp.moe) {
      const size_t S = M * hp.n_expert_used, F = hp.n_ff_exp_pad, NE = hp.n_expert;
      slots_ = (int)S;
      rl_ = buf(M * hp.router_rows() * 4);
      if (has_rgu_) {  // stacked router + shared gate / up output; the gate / up part as its own (aligned) buffer
        rgu_out_ = buf((size_t)(rgu_rows() + 2 * hp.n_ff_shexp) * 4);
        if (!opt.dry) {
          cl_buffer_region reg{(size_t)rgu_rows() * 4, (size_t)2 * hp.n_ff_shexp * 4};
          cl_int err;
          rgu_gu_ = clCreateSubBuffer(rgu_out_, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &reg, &err);
          CL_CHECK(err);
          mems_.push_back(rgu_gu_);
        }
      }
      mids_ = buf(S * 4);
      mwts_ = buf(S * 4);
      sgate_ = buf(M * 4);
      mcount_ = buf(NE * 4);
      moffs_ = buf((NE + 1) * 4);
      mcursor_ = buf(NE * 4);
      mtiles_ = buf((1 + 3 * (S / 16 + NE + S)) * 4);  // enough for 1-slot tiles too
      stok_ = buf(S * 4);
      swt_ = buf(S * 4);
      minv_ = buf(S * 4);
      if (!share) {
        mh_ = buf(S * F * 4);
        my_ = buf(S * hp.n_embd * 4);
      }
      mhq_ = buf(S * F * (q2_ ? 2 : 1));
      mhsc_ = buf(S * (F / 32) * 4 * (q2_ ? 2 : 1));
      if (q2_) {  // (two-pass mode: the down input's residual at these offsets; the up projection apart from the gate)
        q2m_hq_ = S * F;
        q2m_sc_ = S * (F / 32) * 4;
        mh2_ = buf(S * F * 4);
      }
      msh_ = buf(M * hp.n_embd * 4);
      if (hp.gemma) {  // (Gemma 4 experts: msh_ stays zero; the dense FFN's normed output; the experts' sum)
        fill(msh_, M * hp.n_embd * 4);
        gmlp_ = buf(M * hp.n_embd * 4);
        gmoe_ = buf(M * hp.n_embd * 4);
      }
      mtstart_ = buf((NE + 1) * 4);
      fill(mcount_, NE * 4);
      if (moe_gemm_ok()) {
        static const int g_env = getenv("ARC_MOE_G") ? atoi(getenv("ARC_MOE_G")) : 64;
        moe_group_ = std::clamp(g_env, 1, (int)NE);
        while (NE % moe_group_) --moe_group_;
        const size_t G = moe_group_, rows = std::max(F, (size_t)hp.n_embd);
        moe_wq_ = buf(G * F * hp.n_embd);
        moe_wh_ = buf(w8_bytes_h((int)(G * rows), (int)(F * hp.n_embd / rows)));
      }
    }
    if (hp.n_ple) {
      const size_t np = (size_t)hp.n_ple * hp.n_layer;
      ple_ = buf(M * np * 4);
      ple_tok_ = buf(M * np * 4);
      ple_g_ = buf(M * hp.n_ple * 4);
      ple_host_.resize(M * np);
    }
    if (gemma_mtp_) {
      d_x_ = buf((size_t)16 * d_E_ * 4);
      d_cat_ = buf((size_t)2 * hp.n_embd * 4);
      d_h_ = buf((size_t)hp.n_embd * 4);
      fill(d_x_, (size_t)16 * d_E_ * 4);
    }
    if (dfl_) dfl_buffers();
    if (has_mtp) {
      const size_t mb = mtp_block_ ? M : 1;  // (DFlash alone: no MTP block rows)
      mtp_x_ = buf(mb * hp.n_embd * 4);
      mtp_e_ = buf(mb * hp.n_embd * 4);
      hfin_ = buf((mb + 1) * hp.n_embd * 4);
      cl_buffer_region reg{(size_t)hp.n_embd * 4, mb * hp.n_embd * 4};
      if (!opt.dry) {
        cl_int err;
        hfin1_ = clCreateSubBuffer(hfin_, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &reg, &err);
        CL_CHECK(err);
        mems_.push_back(hfin1_);
      }
      hlast_ = buf((size_t)hp.n_embd * 4);
      hpend_ = buf((size_t)max_drafts() * hp.n_embd * 4);  // (hidden states of the accepted drafts)
      hver_ = buf((size_t)(1 + max_drafts()) * hp.n_embd * 4);
      mtp_h_ = buf((size_t)hp.n_embd * 4);
      fill(hlast_, (size_t)hp.n_embd * 4);
      mtp_logits_host_.resize(hp.n_vocab);
      // ARC_DRAFT_VOCAB=N: the drafts' output head covers only the first N token ids (BPE ids are roughly in order of
      // frequency) and the last 512 (control tokens). A draft is only a guess the main model verifies, so the output
      // stays the same; a token outside the list is never drafted. Off by default: 27B, 3 drafts, N = 65536: an
      // English code answer 9.0 -> 9.5 tok/s, but an Indonesian one 7.3 -> 6.1 (its common tokens lie beyond 65536:
      // 206 of 441 drafts accepted instead of 251 of 360).
      const int dv = getenv("ARC_DRAFT_VOCAB") ? atoi(getenv("ARC_DRAFT_VOCAB")) : 0;
      if (dv > 0 && dv + 512 < hp.n_vocab) {
        for (int i = 0; i < dv; ++i) draft_ids_.push_back(i);
        for (int i = hp.n_vocab - 512; i < hp.n_vocab; ++i) draft_ids_.push_back(i);
        draft_idx_ = buf(draft_ids_.size() * 4);
        if (!opt.dry)
          CL_CHECK(clEnqueueWriteBuffer(cl_.queue, draft_idx_, CL_TRUE, 0, draft_ids_.size() * 4, draft_ids_.data(), 0,
                                        nullptr, nullptr));
      }
    }
    for (int r = 0; r < kRoles; ++r)
      if (stage_need_q_[r]) {
        stage_q_[r] = buf(stage_need_q_[r]);
        stage_h_[r] = buf(stage_need_h_[r]);
      }
    if (!w8_resident || stream_) {
      // residual stream of a layer-major segment (+ one chunk of padding rows)
      lm_seg_ = lm_seg_for(ctx);
      const size_t rows = (size_t)lm_seg_ + max_chunk;
      x_all_ = buf(rows * hp.hce() * 4);
      cl_mem ple_all = hp.ple_layer >= 0 ? buf(rows * hp.n_embd * 4) : nullptr;
      for (size_t r0 = 0; r0 + max_chunk <= rows; r0 += max_chunk) {
        cl_buffer_region reg{r0 * hp.hce() * 4, (size_t)max_chunk * hp.hce() * 4};
        cl_int err;
        if (opt.dry) continue;
        x_views_.push_back(clCreateSubBuffer(x_all_, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &reg, &err));
        mems_.push_back(x_views_.back());
        CL_CHECK(err);
        if (ple_all) {
          cl_buffer_region rp{r0 * hp.n_embd * 4, (size_t)max_chunk * hp.n_embd * 4};
          ple_views_.push_back(clCreateSubBuffer(ple_all, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &rp, &err));
          mems_.push_back(ple_views_.back());
          CL_CHECK(err);
        }
      }
    }
    fill(x_, M * hp.hce() * 4);
    logits_host_.resize(hp.n_vocab);
    if (has_mtp) spec_logits_host_.resize((size_t)hp.n_vocab * (1 + max_drafts()));
    x_host_.resize(M * hp.hce());
  }
  // tokens per layer-major prompt segment (Flash Next: 8192, its residual rows are 4 streams wide)
  int lm_seg_for(int c) const { return std::max(max_chunk, std::min(c, hp.hc ? 8192 : 16384) / max_chunk * max_chunk); }

  // ---------------------------------------------------------------- dispatch helpers
  void run(const char* name, cl_kernel k, int dims, const size_t* gws, const size_t* lws) {
    cl_event e = nullptr;
    CL_CHECK(clEnqueueNDRangeKernel(cl_.queue, k, dims, nullptr, gws, lws, 0, nullptr, profile ? &e : nullptr));
    if (profile) events_.push_back({prof_group_ >= 0 ? std::string(name) + "@" + std::to_string(prof_group_) : std::string(name), e});
  }
  void collect_profile() {
    if (!profile) return;
    clFinish(cl_.queue);
    for (auto& [n, e] : events_) {
      prof_ms[n] += ClRt::event_ms(e);
      prof_n[n]++;
      clReleaseEvent(e);
    }
    events_.clear();
  }

  void decode_w8(const GpuWeight& w, cl_mem dq, cl_mem dh) { decode_w8(w.raw, w.type, w.N, w.K, dq, dh); }
  void decode_w8(cl_mem raw, uint32_t type, int N, int K, cl_mem dq, cl_mem dh) {
    // big-codebook types decode faster per superblock with the codebook in local memory; the rest per 32-block
    // one work-item per (row, superblock): the superblock is vector-loaded into registers and the codebook staged in
    // local memory (27B, 11K prompt: IQ3_S / IQ4_XS / IQ3_XXS 5.8 -> 2.2 s); ARC_W8_SB=0: the per-32-block kernel,
    // which reads global memory byte by byte (A/B tests; requantized types always go per superblock)
    static const bool per32 = getenv("ARC_W8_SB") && atoi(getenv("ARC_W8_SB")) == 0;
    static const bool by_type = getenv("ARC_PROF_W8TYPE") != nullptr;  // (profiling: decode time per weight type)
    const std::string nm = by_type ? std::string("decode_w8_") + ggml_type_name(type) : std::string("decode_w8");
    const bool per_sb = !per32 || w8_requant(type);
    // Q8_0 / Q4_0 with their vector block decoder (vblock, decode_w8_vb): 3x faster than decode_w8_sb. (For the
    // codebook types it was slower: each lane loads its own superblock, 16 cache lines per load instruction.)
    static const bool no_vb = getenv("ARC_NO_W8_VB") != nullptr;
    size_t lx = 256;
    while (N % lx) lx /= 2;
    const bool vb = !no_vb && per_sb && lx >= 64 && (type == GGML_Q8_0 || type == GGML_Q4_0);
    // whole rows of up to 16 superblocks, staged in local memory with coalesced loads (decode_w8_sl): types without
    // mins (Qwen3.6 35B prompt experts, 2048-token chunk: IQ3_XXS 1038 -> 518 ms, IQ4_XS 433 -> 303 ms)
    static const bool no_sl = getenv("ARC_NO_W8_SL") != nullptr;
    const int nsb = K / 256;
    const bool sl = !no_sl && per_sb && !w8_requant(type) && nsb <= 16 && N % 16 == 0 &&
                    16 * nsb * raw_padded_sb_bytes(type) <= 32768 && type != GGML_IQ1_M;
    if (sl) {
      cl_kernel k = kernel("decode", "-DQT=" + std::to_string(type) + " -DSL_NSB=" + std::to_string(nsb), "decode_w8_sl");
      set_args(k, 0, raw, N, K, tabs_, dq, dh);
      size_t g1 = (size_t)N * nsb, l1 = (size_t)16 * nsb;
      run(nm.c_str(), k, 1, &g1, &l1);
      return;
    }
    cl_kernel k = kernel("decode", "-DQT=" + std::to_string(type), vb ? "decode_w8_vb" : per_sb ? "decode_w8_sb" : "decode_w8");
    set_args(k, 0, raw, N, K, tabs_, dq, dh);
    if (per_sb) {
      size_t gws[2] = {(size_t)N, (size_t)K / 256}, lws[2] = {lx, 1};
      run(nm.c_str(), k, 2, gws, lws);
    } else {
      size_t gws[2] = {(size_t)N, (size_t)K / 32}, lws[2] = {16, 1};
      run(nm.c_str(), k, 2, gws, lws);
    }
  }

  // Gemma's final logit soft-capping happens in the output head's GEMV (heads only ever see <= 8 rows).
  std::string head_defs(const GpuWeight& w) const {
    if (&w != &lm_head_ || hp.softcap <= 0.0f || q2_nocap_) return "";
    char b[48];
    snprintf(b, sizeof b, " -DSOFTCAP=%.9ef", hp.softcap);
    return b;
  }
  // gemv_v / gemv_mt_v (zero/kernels/decode.cl: weights loaded straight into registers, bit-identical to gemv): for IQ4_XS,
  // whose codebook decoding made gemv compute-bound, and Q6_K output heads; other types are as fast or faster in gemv.
  static bool gemv_v_ok(const GpuWeight& w) {
    static const bool off = getenv("ARC_NO_GEMV_V") != nullptr;  // (A/B tests: the gemv / gemv_mt path)
    // (measured per type with bench/gemv_bench on the regrouped GPU forms; IQ3_S stays faster in gemv)
    return !off && (w.type == GGML_IQ4_XS || w.type == GGML_Q4_0 || w.type == GGML_IQ3_XXS || w.type == GGML_IQ2_XXS ||
                    w.type == GGML_IQ2_XS || w.type == GGML_IQ2_S || w.type == GGML_IQ1_M || w.type == GGML_Q6_K);
    // (Q6_K: every matrix, not only output heads: 27B MTP block FFN 63-67 -> 71-77 GB/s, attn_output 0.41 -> 0.34 ms,
    //  eh_proj 0.78 -> 0.61 ms; tiny matrices equal; 16 Oct)
  }
  static int gemv_v_rows(const GpuWeight& w) {
    static const int force = getenv("ARC_GVROWS") ? atoi(getenv("ARC_GVROWS")) : 0;  // (tuning)
    // (a per-type rule tuned on single matrices repeated in bench/gemv_bench, e.g. IQ4_XS one row, IQ3_XXS four,
    // was no faster in the model, where each matrix streams once: 4.68 / 4.67 / 4.79 vs 4.68 / 4.51 / 4.65 tok/s)
    if (force) return force;
    return (w.N > 4096 || w.K >= 8192) ? 2 : 1;
  }
  // (A/B: an env var set to 256 builds that kernel with the large register file; same results, other speed)
  static std::string grf_env(const char* var) {
    const char* e = getenv(var);
    return e && std::string(e) == "256" ? std::string("-cl-intel-256-GRF-per-thread ") : std::string();
  }
  // (A/B: ARC_MT_LAZY_MIN=N, the token count from which gemv_mt / gemv_mt_v load activations per token; default 4)
  static std::string mt_lazy_opt() {
    static const std::string o = getenv("ARC_MT_LAZY_MIN") ? std::string(" -DMT_LAZY_MIN=") + getenv("ARC_MT_LAZY_MIN") : std::string();
    return o;
  }
  // C[:, c_off : c_off+N] (+)= act * W^T for the current chunk. A single token uses the raw-row GEMV.
  // acc: 0 = store, 1 = accumulate into C, 2 = C = silu(C) * result (fused SwiGLU, GEMM path only)
  // ---- two int8 passes (ARC_Q2=1, Gemma 4; QUEUE F21.4): every activation row is stored twice by quant_store16,
  // the int8 values and, at fixed offsets in the same buffers, their rounding residual in int8; a matrix product
  // runs on both and adds them (the f32 reference simulating this: Gemma 4 26B KL 0.0985 -> 0.0053 together with
  // two-pass attention). The offsets come from alloc_buffers' sizes (max_chunk rows of the widest input).
  bool q2_ = false;
  cl_mem qf_ = nullptr, mh2_ = nullptr;
  size_t q2m_hq_ = 0, q2m_sc_ = 0;  // (the experts' down input: residual offsets in mhq_ / mhsc_, bytes)
  bool q2_off_ = false;
  bool q2_nocap_ = false;  // (the output head's two passes run without their softcap epilogue)
  bool q2_fuse_ = false, q2_fused_ = false;  // (matmul -> matmul1: run the fused two-pass GEMV; it ran)
  const bool q2_fuse_on_ = !getenv("ARC_Q2_FUSE") || atoi(getenv("ARC_Q2_FUSE")) != 0;  // (0: two launches, for A/B)
  bool q2_att_ = !getenv("ARC_Q2_ATT") || atoi(getenv("ARC_Q2_ATT")) != 0;  // (ARC_Q2_ATT=0: the int8 attention with two-pass matrices, for KL studies)
  bool q2_kf32_ = !getenv("ARC_Q2_KF32") || atoi(getenv("ARC_Q2_KF32")) != 0;  // K rows in f32 (0: f16)  // (single pass for now: the Gemma assistant drafter keeps its own arithmetic)
  size_t q2_aq_ = 0, q2_sc_ = 0, q2_ab_ = 0;  // bytes
  std::string q2_defs_;
  void q2_init() {
    if (getenv("ARC_Q2") && atoi(getenv("ARC_Q2")) != 0) opt.q2 = true;
    q2_ = hp.gemma && opt.q2;
    opt.q2 = q2_;
    if (!q2_) return;
    const size_t M = max_chunk;
    const int maxK = std::max({hp.n_embd, hp.n_ff, hp.ssm_d_inner, hp.n_head * hp.head_dim, hp.hce(), hp.n_ff_exp_pad});
    q2_aq_ = M * maxK;
    q2_sc_ = M * (maxK / 32) * 4;
    q2_ab_ = M * (maxK / 16) * 2;
    q2_defs_ = " -DQ2_AQ=" + std::to_string(q2_aq_) + " -DQ2_SC=" + std::to_string(q2_sc_ / 4) + " -DQ2_AB=" +
               std::to_string(q2_ab_ / 2);
    fprintf(stderr, "two-pass int8 activations (--q2): on\n");
  }
  void matmul(const char* name, const GpuWeight& w, int Mp, cl_mem C, int ldc, int c_off, int acc, bool act32) {
    // (the output head with a logit softcap: the cap applies to the sum of both passes, not to each; the head GEMVs
    //  cap in their epilogue, so both run without it and softcap_rows caps the sum)
    const bool cap = q2_ && !q2_off_ && &w == &lm_head_ && hp.softcap > 0.0f;
    q2_nocap_ = cap;
    // (one token: the GEMV runs both passes over one read of the weights, -DQ2F; it says so in q2_fused_)
    q2_fuse_ = q2_ && !q2_off_ && cur_T_ == 1 && q2_fuse_on_;
    q2_fused_ = false;
    matmul1(name, w, Mp, C, ldc, c_off, acc, act32);
    q2_fuse_ = false;
    if (!q2_ || q2_off_) return;
    if (acc > 1) throw std::logic_error(std::string("two-pass activations with a fused epilogue: ") + name);
    if (!q2_fused_) {
      const cl_mem a = aq_, s = asc_, b = absum_;
      aq_ = from(a, q2_aq_);
      asc_ = from(s, q2_sc_);
      absum_ = from(b, q2_ab_);
      matmul1(name, w, Mp, C, ldc, c_off, 1, act32);
      aq_ = a;
      asc_ = s;
      absum_ = b;
    }
    if (cap) {
      q2_nocap_ = false;
      if (cur_T_ > 8) throw std::logic_error("two-pass output head over more than 8 rows (no softcap there)");
      cl_kernel k = kernel("ops", "", "softcap_rows");
      int ld = cur_T_ == 1 ? 0 : ldc, off = c_off, N = w.N;
      float cp = hp.softcap;
      set_args(k, 0, C, ld, off, N, cp);
      size_t gws[2] = {((size_t)N + 255) / 256 * 256, (size_t)cur_T_};
      run(name, k, 2, gws, nullptr);
    }
  }
  void matmul1(const char* name, const GpuWeight& w, int Mp, cl_mem C, int ldc, int c_off, int acc, bool act32) {
    int a = acc;
    if (cur_T_ == 1) {
      // rows per sub-group: enough sub-groups for small N; very tall matrices (an lm_head of short rows) do better
      // with more sub-groups again
      const bool vk = gemv_v_ok(w);
      const int grows = vk ? gemv_v_rows(w) : w.N <= 4096 ? 1 : w.N <= 8192 ? 2 : w.N <= 65536 ? 4 : 2;
      // split-K for matrices with few rows (zero/kernels/decode.cl SPLITK; experiment, off: ARC_SPLITK=S / ARC_SPLITK_ROWS).
      // Qwen3.6 35B: 512-row matrices 10-15 % faster including the extra splitk_sum launch, larger ones slower, 18.6 vs
      // 18.8 tok/s overall: the small GEMVs lose time to the fixed cost of a launch, not to too few sub-groups
      static const int sk_env = getenv("ARC_SPLITK") ? atoi(getenv("ARC_SPLITK")) : -1;
      static const int sk_rows = getenv("ARC_SPLITK_ROWS") ? atoi(getenv("ARC_SPLITK_ROWS")) : 0;
      const int sgs = (w.N + grows - 1) / grows, pairs = (w.K / 256 + 1) / 2;
      int S = 1;
      if (&w != &lm_head_) S = sk_env >= 0 ? std::max(1, sk_env) : sgs <= sk_rows / 4 ? 4 : sgs <= sk_rows / 2 ? 2 : 1;
      S = std::min(S, pairs);
      int N = w.N, K = w.K, M = Mp;
      rd_bytes += raw_bytes(w);
      if (S > 1) {
        if (!splitk_) splitk_ = buf((size_t)4 * 65536 * 4);
        cl_kernel k = kernel("decode", "-DQT=" + std::to_string(w.type) + " -DGROWS=" + std::to_string(grows) +
                                           " -DSPLITK=" + std::to_string(S), vk ? "gemv_v" : "gemv");
        int z = 0;
        set_args(k, 0, w.raw, N, K, tabs_, aq_, asc_, absum_, M, splitk_, z, z);
        size_t g[2] = {((size_t)sgs * 16 + 255) / 256 * 256, (size_t)S}, l[2] = {256, 1};
        run(name, k, 2, g, l);
        cl_kernel k2 = kernel("decode", "-DQT=" + std::to_string(w.type), "splitk_sum");
        int SS = S;
        set_args(k2, 0, splitk_, SS, N, C, c_off, a);
        size_t g2 = ((size_t)N + 255) / 256 * 256, l2 = 256;
        run(name, k2, 1, &g2, &l2);
        return;
      }
      if (q2_fuse_ && a > 1) throw std::logic_error(std::string("fused two-pass GEMV with an epilogue: ") + name);
      cl_kernel k = kernel("decode", "-DQT=" + std::to_string(w.type) + " -DGROWS=" + std::to_string(grows) + head_defs(w) +
                                         (q2_fuse_ ? " -DQ2F" + q2_defs_ : std::string()),
                           vk ? "gemv_v" : "gemv");
      q2_fused_ = q2_fuse_;
      set_args(k, 0, w.raw, N, K, tabs_, aq_, asc_, absum_, M, C, c_off, a);
      size_t g1 = ((size_t)(N + grows - 1) / grows * 16 + 255) / 256 * 256, l1 = 256;
      run(name, k, 1, &g1, &l1);
      return;
    }
    if (cur_T_ > 1 && cur_T_ <= 8) {
      // small batch (e.g. speculative verification): read each weight once for all tokens
      // token-outer kernels (each token's activations loaded once for all of a sub-group's rows, no per-token
      // branches; bit-identical to the earlier row-outer ones), built for exactly T tokens with 256 registers.
      // 27B, 8 rows (bench/gemv_bench): IQ3_XXS gate 23.4 -> 42.9 GB/s with 4 rows per sub-group, IQ3_S down (now
      // gemv_mt_v) 22.6 -> 38.9 with 2; Q6_K one row (it spills with more). ARC_MT_OLD=1: the earlier choice
      static const bool old = getenv("ARC_MT_OLD") != nullptr;
      const int tb = old ? (cur_T_ <= 2 ? 2 : cur_T_ <= 4 ? 4 : 8) : cur_T_;
      const bool vk = gemv_v_ok(w) || (!old && w.type == GGML_IQ3_S);
      int gr = vk ? gemv_v_rows(w) : w.N <= 4096 ? 1 : w.N <= 8192 ? 2 : 4;
      // Below 6 tokens (MTP verification of prose: 2-3 rows) 4 rows per sub-group for the codebook types and for long
      // rows (K >= 8192, the FFN down projection), Q4_K / IQ4_XS short rows 2: 27B, bench/gemv_bench, 3 tokens: gate
      // IQ3_XXS 57.8 -> 64.7 GB/s, down Q4_K 57.5 -> 70.6; Indonesian prose with MTP 6.95 -> 7.2 tok/s. With 6-8
      // tokens the earlier rule stays: there 4 for the down projection and 2 for gate / up were faster only in
      // bench/gemv_bench (code with 7 drafts or DFlash2 2 % slower in the model). Bit-identical either way.
      if (!old) {
        const bool cb = vk && w.type != GGML_IQ4_XS;
        if (w.type == GGML_Q6_K) gr = 1;
        else if (cur_T_ < 6 && (vk || w.type == GGML_Q4_K)) gr = w.K >= 8192 || cb ? 4 : 2;
        else gr = w.type == GGML_IQ3_S ? 2 : vk ? (w.N >= 8192 ? 4 : 2) : std::min(gr, 2);
      }
      cl_kernel k = kernel("decode", std::string(old ? "" : "-cl-intel-256-GRF-per-thread ") + "-DQT=" +
                                         std::to_string(w.type) + " -DGROWS=" + std::to_string(gr) + " -DTB=" + std::to_string(tb) + head_defs(w) +
                                         mt_lazy_opt(),
                           vk ? "gemv_mt_v" : "gemv_mt");
      int N = w.N, K = w.K, M = Mp, T = cur_T_;
      rd_bytes += raw_bytes(w);
      set_args(k, 0, w.raw, N, K, tabs_, aq_, asc_, absum_, M, T, C, ldc, c_off, a);
      size_t g1 = ((size_t)(N + gr - 1) / gr * 16 + 255) / 256 * 256, l1 = 256;
      run(name, k, 1, &g1, &l1);
      return;
    }
    cl_mem wq = w.wq, wh = w.wh;
    if (!wq) {
      wq = stage_q_[w.role];
      wh = stage_h_[w.role];
      if (stage_owner_[w.role] != &w) {  // still decoded from the previous chunk otherwise
        decode_w8(w, wq, wh);
        stage_owner_[w.role] = &w;
      }
    }
    cl_kernel k = gemm_kernel(w.gemm_qtype, act32);
    int M = Mp, N = w.N, K = w.K;
    // Long reductions are split into ~17-superblock pieces (the activation slab of a token-row group then stays in
    // L3); later pieces accumulate into C. Token rows are swept together per weight slice (rgroup).
    static const int rg_env = getenv("ARC_GEMM_RG") ? std::max(1, atoi(getenv("ARC_GEMM_RG"))) : 4;
    static const int wgy_max = getenv("ARC_GEMM_WGY") ? atoi(getenv("ARC_GEMM_WGY")) : 8;
    const int tm = gemm_tm();
    // (the grid covers whole 16-column groups and tm-token tiles: other shapes would leave rows or columns out)
    if (Mp % tm || N % 16 || K % 256)
      throw std::runtime_error(std::string("gemm ") + name + ": shape " + std::to_string(Mp) + " x " + std::to_string(N) + " x " +
                               std::to_string(K) + " is not a multiple of " + std::to_string(tm) + " x 16 x 256");
    // (ARC_GEMM_SPLIT: superblocks per split-K piece, default 17; 0: no split-K)
    static const int sk_sb = getenv("ARC_GEMM_SPLIT") ? std::max(0, atoi(getenv("ARC_GEMM_SPLIT"))) : 17;
    const int nsb = K / 256, splits = K > 8192 && sk_sb > 0 ? (nsb + sk_sb - 1) / sk_sb : 1, rgroup = rg_env;
    int wgy = 1;
    while (wgy < wgy_max && (Mp / tm) % (wgy * 2) == 0) wgy *= 2;
    size_t gws[2] = {(size_t)N * (Mp / tm / wgy), (size_t)wgy}, lws[2] = {16, (size_t)wgy};
    for (int sp = 0; sp < splits; ++sp) {
      int s0 = nsb * sp / splits, s1 = nsb * (sp + 1) / splits, ac = sp == 0 ? a : 1;
      set_args(k, 0, wq, wh, aq_, asc_, absum_, C, M, N, K, ldc, c_off, ac, rgroup, s0, s1);
      run(name, k, 2, gws, lws);
    }
  }

  // ---- matrices of a model next to the language model (the vision encoders, src/vision_*.h)
  // A tensor of another GGUF (F16 / BF16 / F32) as a resident Q8_0 GEMM weight.
  GpuWeight weight_ext(const GgufTensor& t) {
    const int K = (int)t.dims[0], N = (int)t.dims[1];
    std::vector<float> v((size_t)N * K);
    dequant_row(t.type, t.data, (int64_t)N * K, v.data());
    auto q8 = quantize_rows_q8_0(v.data(), N, K);
    return make_weight(GGML_Q8_0, q8.data(), q8.size(), N, K);
  }
  // C[:, c_off : c_off + N] (+)= act * W^T for Mp rows, with the activations in the given buffers (quant_store16
  // layout, per-32 scales). acc: 0 = store, 1 = accumulate.
  void gemm_ext(const GpuWeight& w, cl_mem aq, cl_mem asc, cl_mem absum, int Mp, cl_mem C, int ldc, int c_off, int acc) {
    cl_kernel k = gemm_kernel(w.gemm_qtype, true);
    int M = Mp, N = w.N, K = w.K;
    if (Mp % 16 || N % 16 || K % 256)
      throw std::runtime_error("gemm_ext: shape " + std::to_string(Mp) + " x " + std::to_string(N) + " x " + std::to_string(K) +
                               " is not a multiple of 16 x 16 x 256");
    const int nsb = K / 256, splits = K > 8192 ? (nsb + 16) / 17 : 1, rgroup = 4;
    int wgy = 1;
    while (wgy < 8 && (Mp / 16) % (wgy * 2) == 0) wgy *= 2;
    size_t gws[2] = {(size_t)N * (Mp / 16 / wgy), (size_t)wgy}, lws[2] = {16, (size_t)wgy};
    for (int sp = 0; sp < splits; ++sp) {
      int s0 = nsb * sp / splits, s1 = nsb * (sp + 1) / splits, ac = sp == 0 ? acc : 1;
      set_args(k, 0, w.wq, w.wh, aq, asc, absum, C, M, N, K, ldc, c_off, ac, rgroup, s0, s1);
      run("vit_gemm", k, 2, gws, lws);
    }
  }

  // Activation layout for a GEMM input: per-32 scales for sensitive inputs, the single-token GEMV, or on request.
  int act32_for(bool sensitive) const { return (cur_T_ <= 8 || sensitive || opt.act32_all) ? 1 : 0; }

  void rmsnorm_quant(cl_mem x, int row_off, cl_mem w, int K, int Mp, int T, int act32) {
    float eps = hp.eps;
    int wh = 0;
    set_args(k_rmsnorm_, 0, x, row_off, w, K, Mp, T, eps, aq_, asc_, absum_, big_, wh, act32);
    size_t gws[2] = {256, (size_t)Mp}, lws[2] = {256, 1};
    run("rmsnorm_quant", k_rmsnorm_, 2, gws, lws);
  }

  // ---------------------------------------------------------------- forward pieces
  void embed(const int* tokens, int T, int Mp) {
    cur_T_ = T;
    attn_in_ready_ = ffn_in_ready_ = false;
    const size_t row_bytes = tok_embd_->nbytes / hp.n_vocab;
    if (hp.hc) {  // Flash Next: hc_n copies of the embedding per row
      const int W = hp.hce(), E = hp.n_embd;
      std::fill(x_host_.begin(), x_host_.begin() + (size_t)Mp * W, 0.0f);
#pragma omp parallel for if (T > 64)
      for (int t = 0; t < T; ++t) {
        if (tokens[t] < 0 || tokens[t] >= hp.n_vocab) continue;
        float* r = &x_host_[(size_t)t * W];
        dequant_row(tok_embd_->type, tok_embd_->data + (size_t)tokens[t] * row_bytes, E, r);
        for (int c = 1; c < hp.hc_n; ++c) memcpy(r + (size_t)c * E, r, (size_t)E * 4);
      }
      CL_CHECK(clEnqueueWriteBuffer(cl_.queue, x_, CL_TRUE, 0, (size_t)Mp * W * 4, x_host_.data(), 0, nullptr, nullptr));
      if (hp.ple_layer >= 0) ple_embed(tokens, T);
      return;
    }
    std::fill(x_host_.begin(), x_host_.begin() + (size_t)Mp * hp.n_embd, 0.0f);
#pragma omp parallel for if (T > 64)
    for (int t = 0; t < T; ++t) {
      if (tokens[t] < 0 || tokens[t] >= hp.n_vocab) continue;
      dequant_row(tok_embd_->type, tok_embd_->data + (size_t)tokens[t] * row_bytes, hp.n_embd,
                  &x_host_[(size_t)t * hp.n_embd]);
      if (hp.gemma) {  // Gemma scales the embeddings by sqrt(n_embd)
        const float sc = std::sqrt((float)hp.n_embd);
        for (int i = 0; i < hp.n_embd; ++i) x_host_[(size_t)t * hp.n_embd + i] *= sc;
      }
      if (hp.n_ple) {  // per-layer token embeddings, scaled by sqrt(n_ple)
        const size_t np = (size_t)hp.n_ple * hp.n_layer, rb = ple_tok_embd_->nbytes / hp.n_vocab;
        float* dst = &ple_host_[(size_t)t * np];
        dequant_row(ple_tok_embd_->type, ple_tok_embd_->data + (size_t)tokens[t] * rb, (int)np, dst);
        const float sc = std::sqrt((float)hp.n_ple);
        for (size_t i = 0; i < np; ++i) dst[i] *= sc;
      }
    }
    // blocking: x_host_ is refilled by the next chunk while this write may still be queued behind earlier work
    CL_CHECK(clEnqueueWriteBuffer(cl_.queue, x_, CL_TRUE, 0, (size_t)Mp * hp.n_embd * 4, x_host_.data(), 0, nullptr,
                                  nullptr));
    if (hp.n_ple) {
      CL_CHECK(clEnqueueWriteBuffer(cl_.queue, ple_tok_, CL_TRUE, 0, (size_t)T * hp.n_ple * hp.n_layer * 4,
                                    ple_host_.data(), 0, nullptr, nullptr));
      ple_inputs(T, Mp);
    }
  }

  int prof_group_ = -1;  // ARC_PROF_LAYERS=n: profile names carry the layer group (il / n)
  void run_layer(Layer& L, int T, int Mp) {
    static const int pg = getenv("ARC_PROF_LAYERS") ? atoi(getenv("ARC_PROF_LAYERS")) : 0;
    prof_group_ = pg > 0 ? L.il / pg : -1;
    struct Reset { int& g; ~Reset() { g = -1; } } reset{prof_group_};
    if (hp.gemma) {
      layer_gemma_attn(L, T, Mp);
      layer_gemma_ffn(L, T, Mp);
      return;
    }
    if (hp.hc) {  // Flash Next: each block reads the mixed streams and adds its output to every stream
      if (L.il == hp.ple_layer) ple_block(L, T, Mp);
      hc_mix(L.hc_norm[0], L.hc_down[0], L.hc_up[0], T, Mp, act32_for(false), 0);
      if (hp.recurrent(L.il)) layer_gdn(L, T, Mp);
      else layer_attn(L, T, Mp);
      hc_combine(L.hc_down[0].N, T);
      if (dbg_rows) dbg_row(T - 1);
      hc_mix(L.hc_norm[1], L.hc_down[1], L.hc_up[1], T, Mp, 1, 0);
      static const int dbg_mix = getenv("ARC_DBG_MOE") ? atoi(getenv("ARC_DBG_MOE")) : -1;
      if (L.il == dbg_mix && T > 1) {  // (debug: the FFN mixer's output and inject logits of the last row)
        const int E = hp.n_embd, LD = L.hc_down[1].N;
        std::vector<float> a(E), b(LD);
        CL_CHECK(clEnqueueReadBuffer(cl_.queue, hco_, CL_TRUE, (size_t)(T - 1) * E * 4, E * 4, a.data(), 0, nullptr, nullptr));
        CL_CHECK(clEnqueueReadBuffer(cl_.queue, hc_lo_, CL_TRUE, (size_t)(T - 1) * LD * 4, LD * 4, b.data(), 0, nullptr, nullptr));
        FILE* f = fopen("mix_dbg_gpu.bin", "wb");
        fwrite(a.data(), 4, E, f); fwrite(b.data() + hp.hc_lr, 4, hp.hc_n, f); fwrite(b.data(), 4, hp.hc_lr, f);
        fclose(f);
      }
      layer_moe(L, T, Mp);
      hc_combine(L.hc_down[1].N, T);
      if (dbg_rows) dbg_row(T - 1);
      return;
    }
    if (hp.recurrent(L.il)) layer_gdn(L, T, Mp);
    else layer_attn(L, T, Mp);
    if (dbg_rows) dbg_row(T - 1);
    if (hp.moe) layer_moe(L, T, Mp);
    else layer_ffn(L, T, Mp);
    if (dbg_rows) dbg_row(T - 1);
    if (dfl_) dfl_capture(L, T, Mp);
  }
  // (debug: the residual stream's row r after each half of every layer, to compare two decode paths)
 public:
  bool dbg_rows = false;
  std::vector<std::vector<float>> dbg_;
 private:
  void dbg_row(int r) {
    std::vector<float> v(hp.hce());
    CL_CHECK(clEnqueueReadBuffer(cl_.queue, x_, CL_TRUE, (size_t)r * hp.hce() * 4, v.size() * 4, v.data(), 0, nullptr,
                                 nullptr));
    dbg_.push_back(std::move(v));
  }

  // ---------------------------------------------------------------- Gemma 4 layers
  void layer_gemma_attn(Layer& L, int T, int Mp) {
    const int E = hp.n_embd, nh = hp.n_head, nkv = L.nkv, hd = L.hd, hpg = nh / nkv;
    const int ldq = (nh + 2 * nkv) * hd;
    const int a_in = act32_for(false);
    if (!attn_in_ready_) rmsnorm_quant(x_, 0, L.attn_norm, E, Mp, T, a_in);
    attn_in_ready_ = false;
    const bool own_kv = L.kv_src < 0;
    matmul("gemm_q", L.wq, Mp, big_, ldq, 0, 0, a_in);
    if (own_kv) {
      matmul("gemm_k", L.wk, Mp, big_, ldq, nh * hd, 0, a_in);
      if (L.has_v) matmul("gemm_v", L.wv, Mp, big_, ldq, (nh + nkv) * hd, 0, a_in);
    }
    const std::string hdo = "-DHD=" + std::to_string(hd) + sb_opt(hd);
    const int win = L.swa ? hp.n_swa : 0;
    // RoPE, the K / V writes into the ring (or cache) and the q rows for the attention: rows r0 .. r0 + Tq of the pass
    // at position p0 (a batched pass: per sequence, in its slot)
    auto prep = [&](int r0, int Tq, int p0) {
      // (the large register file: it spilled 8.8 KB per thread; E4B 8K prompt 676 -> 685 tok/s, decode 20.4 -> 20.7;
      //  ARC_GPREP_GRF=128: the default file)
      static const bool gprep_small = getenv("ARC_GPREP_GRF") && std::string(getenv("ARC_GPREP_GRF")) == "128";
      cl_kernel k = kernel("ops", (gprep_small ? std::string() : std::string("-cl-intel-256-GRF-per-thread ")) + hdo +
                                      (q2_kf32_ ? " -DKF32" : ""), "attn_prep_g");
      float eps = hp.eps, base = L.swa ? hp.rope_base_swa : hp.rope_base, qscale = 1.0f;  // Gemma 4: no 1/sqrt(d)
      int pp = p0, hv = L.has_v ? 1 : 0, uff = L.rope_ff ? 1 : 0, cap = L.cap, nn = nh, nk = nkv, ld = ldq, TT = Tq;
      cl_mem rt = rope_table(L.swa ? 0 : 1, hd, base, L.rope_ff, Tq);
      set_args(k, 0, from(big_, (size_t)r0 * ldq * 4), ld, nn, nk, hv, L.q_norm, L.k_norm, TT, pp, eps, base,
               L.rope_ff ? L.rope_ff : L.q_norm, uff, qscale, qi_, qsc_, L.kq, L.ksc, L.vq, L.vsc, cap, rt);
      int f16 = q2_ && q2_att_ ? 1 : 0;
      set_args(k, 22, f16, q2_ ? from(qf_, (size_t)r0 * nh * hd * 4) : qsc_, q2_ ? L.kc : qsc_, q2_ ? L.vc : qsc_);
      size_t gws[2] = {(size_t)(nh + (own_kv ? nkv : 0)) * 16, (size_t)Tq}, lws[2] = {16, 1};
      run("attn_prep", k, 2, gws, lws);
    };
    const std::string ao = attn_opts(hd, hpg);
    // (two-pass mode: float attention over the f16 rows for every pass; rows r0 .. r0 + n at position p0)
    auto fa_g = [&](int r0, int n, int p0) {
      const int gq = hpg <= 4 ? hpg : 4;
      cl_kernel k = kernel("ops", std::string(q2_kf32_ ? "-DKF32 " : "") + "-cl-intel-256-GRF-per-thread -DFA_G -DHD=" + std::to_string(hd) + " -DGQ=" +
                                      std::to_string(gq), "attn_fa_g");
      int cp = L.cap, nn = nh, nk = nkv, pp = p0, MM = n, w = win, nlo = L.swa ? nc_lo_ : -1, nhi = L.swa ? nc_hi_ : -1;
      set_args(k, 0, from(qf_, (size_t)r0 * nh * hd * 4), L.kc, L.vc, cp, nn, nk, pp, MM,
               from(o_, (size_t)r0 * nh * hd * 4), w, nlo, nhi);
      size_t gws[3] = {16, (size_t)(nh / gq), (size_t)n}, lws[3] = {16, 1, 1};
      run(L.swa ? "attn_swa" : "attn_global", k, 3, gws, lws);
    };
    if (q2_ && q2_att_) {
      if (!batch_.empty()) {
        for (const BatchSeq& q : batch_) {
          point_to(q.slot);
          pos = q.pos;
          prep(q.row, q.n, q.pos);  // (qi_ / o_ stay whole: the rows' offsets go to the kernels)
          fa_g(q.row, q.n, q.pos);
          int& wh = slot_host_[q.slot].written_hi;
          wh = std::max(wh, q.pos + q.n);
        }
        pos = 0;
      } else {
        prep(0, T, pos);
        fa_g(0, T, pos);
      }
    } else if (!batch_.empty()) {  // a batched pass (T <= 8): each sequence's rows with its slot's caches and position
      const cl_mem qi0 = qi_, qsc0 = qsc_, o0 = o_;
      for (const BatchSeq& q : batch_) {
        point_to(q.slot);
        pos = q.pos;  // (rope_table's position)
        qi_ = from(qi0, (size_t)q.row * nh * hd);
        qsc_ = from(qsc0, (size_t)q.row * nh * (hd / 64) * 4);
        o_ = from(o0, (size_t)q.row * nh * hd * 4);
        prep(q.row, q.n, q.pos);  // (its rows of big_ in; q rows into the views above)
        attn_decode_rows(L.swa ? "attn_swa" : "attn_global", L, L.cap, nkv, hpg, hd, q.pos, q.n, win);
        int& wh = slot_host_[q.slot].written_hi;
        wh = std::max(wh, q.pos + q.n);
      }
      qi_ = qi0;
      qsc_ = qsc0;
      o_ = o0;
      pos = 0;
    } else {
      prep(0, T, pos);
    }
    if (!batch_.empty() || (q2_ && q2_att_)) {
      // (done above)
    } else if (T <= 8) {
      attn_decode_rows(L.swa ? "attn_swa" : "attn_global", L, L.cap, nkv, hpg, hd, pos, T, win);
    } else if (!L.swa && hd == 512 && (hpg == 16 || hpg == 4) && !getenv("ARC_NO_PS")) {
      // Gemma 4 global layers: head dims split over 4 sub-groups of 4 q heads (attn_ps): attn_q8 spills there.
      // 12B (16 q heads per kv head): 1.25-1.4x faster at any prompt length (bench/attn_bench, ATT_PS=4,4,1
      // ATT_TPW=2); E4B (4 per kv head): TPW 8, 870 -> 619 ms per layer for a 13,921-token prompt (ATT_NH=8
      // ATT_NKV=2 ATT_HD=512 ATT_PS=4,4,1 ATT_TPW=8; same error against the CPU reference)
      const int tpw = hpg == 16 ? 2 : 8, lsz = 16 * (hpg / 4) * (tpw / 1) * 4;
      cl_kernel kp = kernel("ops", "-cl-intel-256-GRF-per-thread -DHD=512 -DHPG=" + std::to_string(hpg) + " -DHS=2 -DTPW=" +
                                       std::to_string(tpw) + " -DPNS=4 -DPHG=4 -DPTT=1" + sb_opt(hd), "attn_ps");
      int p0 = pos, cap = L.cap, nn = nh, TT = T, w = win;
      set_args(kp, 0, qi_, qsc_, L.kq, L.ksc, L.vq, L.vsc, cap, nn, p0, TT, o_, w);
      size_t gws[2] = {(size_t)(T + tpw - 1) / tpw * lsz, (size_t)nkv}, lws[2] = {(size_t)lsz, 1};
      run("attn_global", kp, 2, gws, lws);
    } else {
      cl_kernel ka = kernel("ops", ao, "attn_q8");
      // image tokens attend to each other non-causally in the sliding-window layers (Gemma 4 unified vision)
      int p0 = pos, cap = L.cap, nn = nh, TT = T, w = win, nlo = L.swa ? nc_lo_ : -1, nhi = L.swa ? nc_hi_ : -1;
      set_args(ka, 0, qi_, qsc_, L.kq, L.ksc, L.vq, L.vsc, cap, nn, p0, TT, o_, w, nlo, nhi);
      const int tpw = 32 / (hpg / attn_hs_for(hpg));
      size_t gws[2] = {(size_t)(T + tpw - 1) / tpw * 512, (size_t)nkv}, lws[2] = {512, 1};
      run(L.swa ? "attn_swa" : "attn_global", ka, 2, gws, lws);
    }
    if (batch_.empty() && pos + T > written_hi_) written_hi_ = pos + T;
    const int a_out = act32_for(true);
    {
      int K = nh * hd, MM = Mp, TT = T, fused = 1, gelu = 0;  // quantize the attention output for wo
      set_args(k_swiglu_, 0, o_, K, MM, TT, aq_, asc_, absum_, a_out, fused, gelu);
      run_rows_quant("attn_gate_quant", K, T, Mp);
    }
    matmul("gemm_o", L.wo, Mp, big_, E, 0, 0, a_out);
    if (T <= 8) {  // + the FFN input norm (layer_gemma_ffn skips its own)
      add_norm_quant(L.post_attn, T, Mp, 1.0f, L.ffn_norm, true, act32_for(false));
      ffn_in_ready_ = true;
    } else {
      add_rmsnorm(L.post_attn, T, 1.0f);
    }
  }

  void layer_gemma_ffn(Layer& L, int T, int Mp) {
    const int E = hp.n_embd, F = hp.n_ff;
    const int a_in = act32_for(false);
    if (!ffn_in_ready_) rmsnorm_quant(x_, 0, L.ffn_norm, E, Mp, T, a_in);
    ffn_in_ready_ = false;
    const int fused = cur_T_ > 8 && E <= 8192 && !q2_ ? 1 : 0;
    if (T <= 8 && (L.gate.type == GGML_IQ4_XS || L.gate.type == GGML_Q4_0) && L.up.type == L.gate.type &&
        F % 32 == 0 && Mp == 16 && !q2_ &&
        !getenv("ARC_NO_GLU")) {
      // E2B / E4B (IQ4_XS gate / up), T <= 8: gelu(gate) * up and its quantization in the GEMV itself (gemv_glu,
      // same values as gemm_gate + gemm_up + swiglu_quant); the down projection reads it from the B buffers
      const int tb = T == 1 ? 1 : T <= 2 ? 2 : T <= 4 ? 4 : 8;
      cl_kernel k = kernel("decode", "-DQT=" + std::to_string((int)L.gate.type) + " -DTB=" + std::to_string(tb), "gemv_glu");
      int N = F, K = E, MM = Mp, TT = T;
      set_args(k, 0, L.gate.raw, L.up.raw, N, K, tabs_, aq_, asc_, MM, TT, aqB_, ascB_, absumB_);
      size_t g1 = (size_t)F / 32 * 256, l1 = 256;
      rd_bytes += raw_bytes(L.gate) + raw_bytes(L.up);
      run("gemm_glu", k, 1, &g1, &l1);
      std::swap(aq_, aqB_);
      std::swap(asc_, ascB_);
      std::swap(absum_, absumB_);
      matmul("gemm_down", L.down, Mp, big_, E, 0, 0, act32_for(false));
      std::swap(aq_, aqB_);
      std::swap(asc_, ascB_);
      std::swap(absum_, absumB_);
    } else {
      if (fused) {
        matmul("gemm_gate", L.gate, Mp, big_, F, 0, 0, a_in);
        matmul("gemm_up", L.up, Mp, big_, F, 0, 3, a_in);  // gelu(gate) * up in the epilogue
      } else {
        matmul("gemm_gate", L.gate, Mp, big_, 2 * F, 0, 0, a_in);
        matmul("gemm_up", L.up, Mp, big_, 2 * F, F, 0, a_in);
      }
      const int a_dn = act32_for(false);
      {
        int gelu = 1, FF = F, MM = Mp, TT = T;
        set_args(k_swiglu_, 0, big_, FF, MM, TT, aq_, asc_, absum_, a_dn, fused, gelu);
        run_rows_quant("swiglu_quant", F, T, Mp);
      }
      matmul("gemm_down", L.down, Mp, big_, E, 0, 0, a_dn);
    }
    if (hp.moe) {
      // Gemma 4 26B-A4B (llama.cpp gemma4): ffn = rmsnorm(mlp) * post_ffw_norm_1 + rmsnorm(experts(rmsnorm(x) *
      // pre_ffw_norm_2)) * post_ffw_norm_2, the router on rmsnorm(x) * scale / sqrt(E); then the usual post_ffw norm
      float eps = hp.eps;
      int EE = E, MM = Mp, TT = T, r0 = 0, wh = 1, a32 = 1;
      size_t rgws[2] = {256, (size_t)Mp}, rlws[2] = {256, 1};
      set_args(k_rmsnorm_, 0, big_, r0, L.post_ffw1, EE, MM, TT, eps, aq_, asc_, absum_, gmlp_, wh, a32);
      run("rmsnorm_quant", k_rmsnorm_, 2, rgws, rlws);
      layer_moe(L, T, Mp);  // (the experts' sum into gmoe_)
      set_args(k_rmsnorm_, 0, gmoe_, r0, L.post_ffw2, EE, MM, TT, eps, aq_, asc_, absum_, big_, wh, a32);
      run("rmsnorm_quant", k_rmsnorm_, 2, rgws, rlws);
      cl_kernel k = kernel("ops", "", "dfl_add");
      set_args(k, 0, big_, gmlp_, EE);
      size_t gws[2] = {(size_t)E, (size_t)T};
      run("ffn_add", k, 2, gws, nullptr);
    }
    // single-token passes: the residual add + norm also prepares the next input (the next layer's attention norm,
    // or the per-layer-embedding gate's plain quantization)
    const Layer* next = L.il + 1 < (int)layers_.size() ? &layers_[L.il + 1] : nullptr;
    if (!hp.n_ple) {
      if (T <= 8 && next) {
        add_norm_quant(L.post_ffw, T, Mp, L.out_scale, next->attn_norm, true, act32_for(false));
        attn_in_ready_ = true;
      } else {
        add_rmsnorm(L.post_ffw, T, L.out_scale);
      }
      return;
    }
    // per-layer embedding block: x += rmsnorm(proj(gelu(inp_gate(x)) * ple[il])) * w, then the output scale
    const int a_p = act32_for(false);
    if (T <= 8) {
      add_norm_quant(L.post_ffw, T, Mp, 1.0f, L.post_ffw, false, a_p);
    } else {
      add_rmsnorm(L.post_ffw, T, 1.0f);
      quant_rows(x_, E, Mp, T, a_p);
    }
    matmul("gemm_ple_gate", L.ple_gate, Mp, ple_g_, hp.n_ple, 0, 0, a_p);
    {
      if (!k_ple_gate_) k_ple_gate_ = kernel("ops", "", "ple_gate_quant");
      int il = L.il, nl = hp.n_layer, MM = Mp, TT = T, a32 = a_p;
      set_args(k_ple_gate_, 0, ple_g_, ple_, il, nl, MM, TT, aq_, asc_, absum_, a32);
      size_t gws[2] = {16, (size_t)Mp}, lws[2] = {16, 1};
      run("ple", k_ple_gate_, 2, gws, lws);
    }
    matmul("gemm_ple_proj", L.ple_proj, Mp, big_, E, 0, 0, a_p);
    if (T <= 8 && next) {
      add_norm_quant(L.ple_post, T, Mp, L.out_scale, next->attn_norm, true, act32_for(false));
      attn_in_ready_ = true;
    } else {
      add_rmsnorm(L.ple_post, T, L.out_scale);
    }
  }

  // x_ = (x_ + rmsnorm(big_) * w) * s over the first T rows, then the next GEMV input from the new x_ (rmsnorm * w2,
  // or plain with norm2 = false) into aq_ / asc_ / absum_
  void add_norm_quant(cl_mem w, int T, int Mp, float s, cl_mem w2, bool norm2, int act32) {
    // (registers for the row's superblocks of each sub-group only: ANQ_MAX 4 spilled 3.7 KB per thread)
    if (!k_add_nq_) k_add_nq_ = kernel("ops", "-DANQ_MAX=" + std::to_string((hp.n_embd / 256 + 15) / 16), "add_norm_quant");
    int E = hp.n_embd, nrm = norm2 ? 1 : 0, MM = Mp, a32 = act32;
    float eps = hp.eps, sc = s;
    set_args(k_add_nq_, 0, x_, big_, w, E, eps, sc, w2, nrm, MM, aq_, asc_, absum_, a32);
    size_t gws[2] = {256, (size_t)T}, lws[2] = {256, 1};
    run("add_rmsnorm", k_add_nq_, 2, gws, lws);
  }
  // launches k_swiglu_ (arguments set) over K columns: a work-group per padded row for prompt chunks, one sub-group
  // per superblock of the T valid rows for single-token passes (the GEMVs read only those rows)
  void run_rows_quant(const char* name, int K, int T, int Mp) {
    if (T <= 8) {
      size_t gws[2] = {(size_t)16 * (K / 256), (size_t)T}, lws[2] = {16, 1};
      run(name, k_swiglu_, 2, gws, lws);
    } else {
      size_t gws[2] = {256, (size_t)Mp}, lws[2] = {256, 1};
      run(name, k_swiglu_, 2, gws, lws);
    }
  }
  // RoPE table of the current rows (pos .. pos + T - 1) for the sliding (0) or global (1) layers, computed once per
  // pass (T <= 8) or per chunk
  cl_mem rope_table(int type, int hd, float base, cl_mem ff, int T) {
    if (rt_pos_[type] == pos && rt_T_[type] == T) return rope_tab_[type];
    if (!k_rope_tab_) k_rope_tab_ = kernel("ops", "", "rope_table");
    int h = hd, p0 = pos, uff = ff ? 1 : 0;
    float b = base;
    set_args(k_rope_tab_, 0, rope_tab_[type], h, p0, b, ff ? ff : rope_tab_[type], uff);
    size_t gws[2] = {(size_t)hd / 2, (size_t)T};
    run("attn_prep", k_rope_tab_, 2, gws, nullptr);
    rt_pos_[type] = pos;
    rt_T_[type] = T;
    return rope_tab_[type];
  }

  // quantizes rows of an f32 matrix (K columns) for a GEMM / GEMV input, without normalization
  void quant_rows(cl_mem src, int K, int Mp, int T, int act32) {
    int KK = K, MM = Mp, TT = T, a32 = act32, fused = 1, gelu = 0;
    set_args(k_swiglu_, 0, src, KK, MM, TT, aq_, asc_, absum_, a32, fused, gelu);
    size_t gws[2] = {256, (size_t)Mp}, lws[2] = {256, 1};
    run("quant", k_swiglu_, 2, gws, lws);
  }

  // Gemma 4 E2B/E4B: per-layer inputs of the current chunk = combination of a per-layer token embedding and a
  // projection of the (scaled) token embedding. ple_tok_ was uploaded by embed().
  void ple_inputs(int T, int Mp) {
    const int E = hp.n_embd, NP = hp.n_ple * hp.n_layer, a32 = act32_for(false);
    quant_rows(x_, E, Mp, T, a32);
    matmul("gemm_ple_model", ple_model_proj_, Mp, ple_, NP, 0, 0, a32);
    if (!k_ple_comb_) k_ple_comb_ = kernel("ops", "", "ple_combine");
    int nl = hp.n_layer, TT = T;
    float pre = 1.0f / std::sqrt((float)E), eps = hp.eps, post = 1.0f / std::sqrt(2.0f);
    set_args(k_ple_comb_, 0, ple_, ple_tok_, ple_proj_norm_, nl, TT, pre, eps, post);
    size_t gws[2] = {(size_t)16 * hp.n_layer, (size_t)T}, lws[2] = {16, 1};
    run("ple", k_ple_comb_, 2, gws, lws);
  }

  // x_ = (x_ + rmsnorm(big_) * w) * s over the first T rows (Gemma sandwich norm + residual + layer scale)
  void add_rmsnorm(cl_mem w, int T, float s) {
    if (!k_add_norm_) k_add_norm_ = kernel("ops", "", "add_rmsnorm");
    int E = hp.n_embd;
    float eps = hp.eps, sc = s;
    set_args(k_add_norm_, 0, x_, big_, w, E, eps, sc);
    size_t gws[2] = {256, (size_t)T}, lws[2] = {256, 1};
    run("add_rmsnorm", k_add_norm_, 2, gws, lws);
  }

  // Flash-decoding of query rows 0..T-1 (positions pos0..pos0+T-1, keys up to each row's position, from
  // pos - win + 1 with a window) of one layer into o_. Keys are split so that at least ~512 sub-groups run. When a
  // work-group of 128+ items can cover the q heads of up to 4 rows (attn_split), each K/V tile goes through SLM once
  // for all of them; otherwise (few q heads per kv head) rows go one by one through attn_decode, whose head slices
  // share the reads through L1.
  void attn_decode_rows(const char* name, Layer& L, int cap, int nkv, int hpg, int hd, int pos0, int T, int win,
                        int n_q_heads = 0) {
    // the Gemma 4 assistant drafter's layers (n_q_heads given) keep the original kernels and split choices
    const bool drafter = n_q_heads != 0;
    // heads per sub-group: one at HD 512 (registers) and with per-64-dim scales (Gemma 4: the head slices of a split
    // then share its K/V through L1, 2x faster than two heads per sub-group); else as the prompt kernels
    const int hs = hd >= 512 || (hp.gemma && !drafter) ? 1 : attn_hs_for(hpg), nsp = hpg / hs;
    // (experiments: ARC_ATTN_DEC_GRF=128 builds the decode attention kernels with the default register file, twice
    //  the threads per EU; ARC_ATTN_DEC_SG sets the sub-groups the key split aims for, default 512)
    static const bool dec_small = [] {
      const char* e = getenv("ARC_ATTN_DEC_GRF");
      return e != nullptr && std::string(e) == "128";
    }();
    const std::string dec_grf = dec_small ? std::string() : std::string("-cl-intel-256-GRF-per-thread ");
    static const int dec_sg = getenv("ARC_ATTN_DEC_SG") ? atoi(getenv("ARC_ATTN_DEC_SG")) : 512;
    static const int dec_smax = getenv("ARC_ATTN_DEC_SMAX") ? std::max(1, atoi(getenv("ARC_ATTN_DEC_SMAX"))) : 256;  // (keys per split, most)
    auto opts = [&](int tpw) {
      return dec_grf + "-DHD=" + std::to_string(hd) + " -DHPG=" + std::to_string(hpg) +
             " -DHS=" + std::to_string(hs) + " -DTPW=" + std::to_string(tpw) + sb_opt(hd) + v4_opt();
    };
    // split length: the longest (fewest partials to combine, up to smax) that still gives `target` sub-groups
    auto pick_split = [&](int lo, int plast, int sg_per_split, int smax, int target, int& S, int& kstart, int& ns) {
      S = smax;
      while (S > 32 && ((plast - lo / S * S) / S + 1) * nkv * sg_per_split < target) S /= 2;
      kstart = lo / S * S;
      ns = (plast - kstart) / S + 1;
    };
    int nn = n_q_heads ? n_q_heads : hp.n_head, hdd = hd, cp = cap, w = win;
    for (int r0 = 0; r0 < T;) {
      // Gemma 4 (per-64-dim scales): attn_dsplit, head dims split over the sub-groups, every K/V byte read once for
      // all rows; up to 4 rows (pairs when a kv head has more than 4 q heads). 1.3-2.4x faster than the paths below
      // at long context (bench/attn_dec_bench). The Qwen models stay on those: there they gain less, and they keep
      // their verification rows bit-identical to single-token decoding (the MTP regression test).
      const int left = T - r0;
      // Qwen3.6 35B (8 q heads per kv head, 256-dim heads, one scale per row): attn_dsplit, 4 q heads per sub-group
      // and the dims split over 4 sub-groups, one row per launch as attn_decode before: bench/attn_dec_bench at 64K
      // 1.16 -> 0.94 ms per layer. The keys split in fixed pieces of 256 from position 0 and each row's P bound covers
      // only its own keys (-DROWVMT), so a verification row gets exactly the values of decoding its token alone.
      // 4B / 9B (4 per kv head, +7 %) and 27B (6, +4 %) keep attn_decode / attn_split, which read the K/V of up to 4
      // verification rows once. ARC_QWEN_DSPLIT=0: never, =1: every Qwen model (2 heads per sub-group, 2 rows).
      static const int qwen_ds = [] {
        const char* e = getenv("ARC_QWEN_DSPLIT");
        return e ? atoi(e) : -1;
      }();
      if (!drafter && !hp.gemma && hd == 256 && hpg % 2 == 0 && (qwen_ds == 1 || (qwen_ds < 0 && hpg >= 8))) {
        const int hg = hpg % 4 == 0 && qwen_ds != 1 ? 4 : 2, n = hg == 4 || opt.v4 ? 1 : std::min(left, 2), tr = n, nsub = 4;
        const int sgs = hpg / hg * nsub, lsz = 16 * sgs;
        const int lo = win ? std::max(0, pos0 + r0 - win + 1) : 0, plast = pos0 + r0 + n - 1;
        static const int qs_len = getenv("ARC_QWEN_DS_S") ? std::max(1, atoi(getenv("ARC_QWEN_DS_S"))) : 256;  // (experiment)
        const int S = qs_len, kstart = lo / S * S, ns = (plast - kstart) / S + 1;
        int p0 = pos0 + r0, nv = n, q0 = r0, SS = S, ks = kstart, nss = ns;
        cl_kernel k = kernel("ops", opts(1) + " -DHG=" + std::to_string(hg) + " -DNSUB=" + std::to_string(nsub) +
                                        " -DTR=" + std::to_string(tr) + " -DROWVMT", "attn_dsplit");
        set_args(k, 0, qi_, qsc_, L.kq, L.ksc, vq_dec(L), vsc_dec(L), cp, nn, p0, nv, apart_, w, ks, SS, nss, q0);
        v4_dec_args(k, 16, L, p0);
        size_t gws[2] = {(size_t)ns * lsz, (size_t)nkv}, lws[2] = {(size_t)lsz, 1};
        run(name, k, 2, gws, lws);
        for (int m = 0; m < n; ++m) {
          int orow = r0 + m, prow = m;
          set_args(k_attn_comb_, 0, apart_, nss, nn, o_, orow, hdd, prow);
          size_t g1 = (size_t)nn * hd, l1 = 64;
          run("attn_combine", k_attn_comb_, 1, &g1, &l1);
        }
        r0 += n;
        continue;
      }
      if (!drafter && hp.gemma) {
        // (one row per launch, as a single token at that position: the key split, the heads per sub-group and the
        //  partial sums are then those of decoding the row's token alone, so a verification row is bit-identical
        //  to it. Several rows per launch (ARC_GEMMA_VROWS=1, before 20 Oct) read each K/V byte once for all rows but
        //  split the keys by the last row's position: drafter logits up to 2.2 (26B) / 1.3 (12B) / 0.85 (E4B) off
        //  plain decoding, the 26B's greedy text changed after 7 tokens. Speed with the drafter, short / 11K prompt:
        //  12B 21.3 -> 23.2 / 14.8 -> 14.9, E4B 47.4 -> 48.9 / 30.9 -> 30.0, 26B 21.0 -> 21.0 at 11K)
        static const bool vrows = getenv("ARC_GEMMA_VROWS") && atoi(getenv("ARC_GEMMA_VROWS")) != 0;
        const bool one_row = !vrows;
        const int n = one_row ? 1 : std::min(left, left <= 2 ? left : hpg <= 4 ? 4 : 2), tr = n <= 1 ? 1 : n <= 2 ? 2 : 4;
        int hg = 1;  // q heads per sub-group: at most 4 (row, head) pairs (registers)
        for (int h : {4, 3, 2, 1})
          if (hpg % h == 0 && h * tr <= 4) { hg = h; break; }
        int nsub = hd / 64;  // dim groups: <= 512 work-items, <= 32 KB of SLM partial scores
        while (nsub > 1 && ((hpg / hg) * nsub > 32 || hpg * tr * nsub > 256)) nsub /= 2;
        const int sgs = hpg / hg * nsub, lsz = 16 * sgs, plast = pos0 + r0 + n - 1;
        const int lo = win ? std::max(0, pos0 + r0 - win + 1) : 0;
        int S, kstart, ns, p0 = pos0 + r0, nv = n, q0 = r0;
        pick_split(lo, plast, sgs, 512, 512, S, kstart, ns);
        cl_kernel k = kernel("ops", opts(1) + " -DHG=" + std::to_string(hg) + " -DNSUB=" + std::to_string(nsub) +
                                        " -DTR=" + std::to_string(tr), "attn_dsplit");
        set_args(k, 0, qi_, qsc_, L.kq, L.ksc, vq_dec(L), vsc_dec(L), cp, nn, p0, nv, apart_, w, kstart, S, ns, q0);
        size_t gws[2] = {(size_t)ns * lsz, (size_t)nkv}, lws[2] = {(size_t)lsz, 1};
        run(name, k, 2, gws, lws);
        for (int m = 0; m < n; ++m) {
          int orow = r0 + m, prow = m;
          set_args(k_attn_comb_, 0, apart_, ns, nn, o_, orow, hdd, prow);
          size_t g1 = (size_t)nn * hd, l1 = 64;
          run("attn_combine", k_attn_comb_, 1, &g1, &l1);
        }
        r0 += n;
        continue;
      }
      int tpw = 1;
      // (Qwen3.6 35B, 2 kv heads: a single token takes attn_decode (16 * nsp < 128), and attn_split over 2 rows split
      // the keys differently, so MTP output drifted from plain decoding after ~40 tokens of a 2K prompt; its
      // verification rows go through attn_decode one by one. The dense Qwen models pass the identity test as they are.)
      if ((!hp.moe || 16 * nsp >= 128 || drafter) && !opt.v4)  // (--v4: rows one by one, see v4_dec_args)
        while (tpw < T - r0 && tpw < 4 && 16 * nsp * tpw * 2 <= 512) tpw *= 2;  // apart_ holds 4 rows of partials
      const int n = std::min(tpw, T - r0);
      if (16 * nsp * tpw >= 128 && !opt.v4) {
        const int lsz = 16 * nsp * tpw, plast = pos0 + r0 + n - 1, lo = win ? std::max(0, pos0 + r0 - win + 1) : 0;
        int S, kstart, ns, p0 = pos0 + r0, nv = n, q0 = r0;
        pick_split(lo, plast, lsz / 16, dec_smax, dec_sg, S, kstart, ns);
        cl_kernel k = kernel("ops", opts(tpw), "attn_split");  // (int8 V only: not taken with --v4)
        set_args(k, 0, qi_, qsc_, L.kq, L.ksc, L.vq, L.vsc, cp, nn, p0, nv, apart_, w, kstart, S, ns, q0);
        size_t gws[2] = {(size_t)ns * lsz, (size_t)nkv}, lws[2] = {(size_t)lsz, 1};
        run(name, k, 2, gws, lws);
        for (int m = 0; m < n; ++m) {
          int orow = r0 + m, prow = m;
          set_args(k_attn_comb_, 0, apart_, ns, nn, o_, orow, hdd, prow);
          size_t g1 = (size_t)nn * hd, l1 = 64;
          run("attn_combine", k_attn_comb_, 1, &g1, &l1);
        }
      } else {
        cl_kernel kd = kernel("ops", opts(1), "attn_decode");
        for (int t = r0; t < r0 + n; ++t) {
          const int p = pos0 + t, lo = win ? std::max(0, p - win + 1) : 0;
          int S, kstart, ns, pp = p, row = t, prow = 0;
          pick_split(lo, p, nsp, dec_smax, dec_sg, S, kstart, ns);
          set_args(kd, 0, qi_, qsc_, L.kq, L.ksc, vq_dec(L), vsc_dec(L), cp, nn, pp, apart_, row, w, kstart, S);
          v4_dec_args(kd, 14, L, p);
          size_t gws[2] = {(size_t)ns * 16, (size_t)(nkv * nsp)}, lws[2] = {16, (size_t)nsp};
          run(name, kd, 2, gws, lws);
          set_args(k_attn_comb_, 0, apart_, ns, nn, o_, row, hdd, prow);
          size_t g1 = (size_t)nn * hd, l1 = 64;
          run("attn_combine", k_attn_comb_, 1, &g1, &l1);
        }
      }
      r0 += n;
    }
  }

  // ---------------------------------------------------------------- Flash Next hyper-connections and PLE
  // The mixer of rows r0 .. r0 + T - 1 of the residual streams x_: per-stream norm, low-rank down (+ inject logits,
  // into hc_lo_), silu(lo / hc_n), up to the stream gates, then the gated mean over the streams = the block input
  // (f32 in hco_, quantized in aq_ / asc_ / absum_ with Mp rows)
  void hc_mix(cl_mem normw, const GpuWeight& down, const GpuWeight& up, int T, int Mp, int act32, int r0) {
    const int E = hp.n_embd, HC = hp.hc_n, W = hp.hce(), LD = down.N, LR = hp.hc_lr, KP = hc_lr_pad();
    float eps = hp.eps;
    // single tokens: two fused kernels with f32 activations (norm + down + inject, then silu + up + gated mean) and
    // the block input's quantization; prompt chunks below: int8 GEMMs (ARC_HC_INT8=1: those for single tokens too)
    static const bool int8_1 = getenv("ARC_HC_INT8") != nullptr;
    if (T == 1 && !int8_1 && W <= 10240 && KP <= 1024) {
      if (!hc_rinv_) hc_rinv_ = buf(64 * 4);
      int rr = r0, e = E, hc = HC, n = down.N, lr = LR, kp = KP;
      cl_kernel k1 = kernel("ops", "", "hc_down1");
      set_args(k1, 0, x_, rr, normw, e, hc, eps, down.raw, n, hc_xn_, hc_rinv_);
      size_t g1[2] = {(size_t)256 * ((n + 15) / 16), 8}, l1[2] = {256, 1};  // (8 K slices: HC_KS)
      run("hc_down1", k1, 2, g1, l1);
      cl_kernel k2 = kernel("ops", "", "hc_up1");
      set_args(k2, 0, x_, rr, normw, hc_rinv_, hc_xn_, n, hc_lo_, lr, kp, e, hc, up.raw, hco_);
      size_t g2 = (size_t)256 * ((E + 15) / 16), l2 = 256;
      run("hc_up1", k2, 1, &g2, &l2);
      rd_bytes += raw_bytes(down) + raw_bytes(up);
      quant_rows(hco_, E, Mp, 1, act32);
      return;
    }
    {
      cl_kernel k = kernel("ops", "", "hc_norm_quant");
      int rr = r0, e = E, hc = HC, mm = Mp, tt = T, a32 = act32;
      set_args(k, 0, x_, rr, normw, e, hc, mm, tt, eps, hc_xn_, aq_, asc_, absum_, a32);
      size_t gws[2] = {256, (size_t)Mp}, lws[2] = {256, 1};
      run("hc_norm", k, 2, gws, lws);
    }
    // (ARC_HC_F32=1: the prompt chunk's mixers with f32 activations as single tokens have them, hc_gemm_f32; Flash Next
    //  is sensitive there: the f32 reference with int8 mixer inputs drifts by KL 0.06 on its own)
    // (1: both projections, 2: the down projection only, 3: the up projection only)
    // (4, the default since 20 Oct: the up projection's input as two int8 passes, the values then their rounding
    //  residual, accumulated: KL 0.0647 -> 0.0483 against the f32 reference, prompts 5-7 % slower; 0: one int8 pass)
    static const int f32_mode = [] {
      const int m = getenv("ARC_HC_F32") ? atoi(getenv("ARC_HC_F32")) : 4;
      if (m < 0 || m > 4) fprintf(stderr, "ARC_HC_F32=%d: not 0..4, the int8 mixers\n", m);
      return m < 0 || m > 4 ? 0 : m;
    }();
    static bool hc_logged = false;
    if (!hc_logged) {
      hc_logged = true;
      static const char* names[] = {"int8", "f32 down + up", "f32 down", "f32 up", "two-pass int8 up"};
      fprintf(stderr, "HC mixers of prompt chunks: %s (ARC_HC_F32=%d)\n", f32_mode >= 0 && f32_mode <= 4 ? names[f32_mode] : "int8", f32_mode);
    }
    const bool f32_down = f32_mode == 1 || f32_mode == 2, f32_up = f32_mode == 1 || f32_mode == 3;
    auto gemm_f32 = [&](cl_mem in, int ld_in, int K, int lo_in, const GpuWeight& w, int N, cl_mem C, int ldc) {
      cl_kernel k = kernel("ops", "", "hc_gemm_f32");
      int ldi = ld_in, kk = K, li = lo_in, lr = LR, hc = HC, nn = N, lc = ldc, mm = Mp;
      set_args(k, 0, in, ldi, kk, li, lr, hc, w.raw, nn, C, lc, mm);
      size_t gws[2] = {(size_t)256 * ((N + 255) / 256), (size_t)(Mp + 15) / 16}, lws[2] = {256, 1};
      run("hc_gemm_f32", k, 2, gws, lws);
    };
    if (f32_down)
      gemm_f32(hc_xn_, HC * E, HC * E, 0, down, LD, hc_lo_, LD);
    else
      matmul("hc_down", down, Mp, hc_lo_, LD, 0, 0, act32);
    if (f32_up) {
      gemm_f32(hc_lo_, LD, KP, 1, up, W, hc_gate_, W);
    } else {
      {
        cl_kernel k = kernel("ops", "", "hc_lo_quant");
        int lr = LR, ld = LD, kp = KP, hc = HC, mm = Mp, tt = T, a32 = act32;
        set_args(k, 0, hc_lo_, lr, ld, kp, hc, mm, tt, aq_, asc_, absum_, a32);
        size_t gws[2] = {(size_t)16 * KP / 256, (size_t)Mp}, lws[2] = {16, 1};
        run("hc_lo", k, 2, gws, lws);
      }
      matmul("hc_up", up, Mp, hc_gate_, W, 0, 0, act32);
      if (f32_mode == 4) {
        cl_kernel k = kernel("ops", "", "hc_lo_quant_res");
        int lr = LR, ld = LD, kp = KP, hc = HC, mm = Mp, tt = T, a32 = act32;
        set_args(k, 0, hc_lo_, lr, ld, kp, hc, mm, tt, aq_, asc_, absum_, a32);
        size_t gws[2] = {(size_t)16 * KP / 256, (size_t)Mp}, lws[2] = {16, 1};
        run("hc_lo", k, 2, gws, lws);
        matmul("hc_up", up, Mp, hc_gate_, W, 0, 1, act32);
      }
    }
    {
      cl_kernel k = kernel("ops", "", "hc_gate_mean_quant");
      int e = E, hc = HC, mm = Mp, tt = T, a32 = act32;
      set_args(k, 0, hc_xn_, hc_gate_, e, hc, mm, tt, hco_, aq_, asc_, absum_, a32);
      size_t gws[2] = {256, (size_t)Mp}, lws[2] = {256, 1};
      run("hc_mix", k, 2, gws, lws);
    }
  }
  // x_ streams += 2 sigmoid(inject / hc_n) * block output (hco_), inject = columns hc_lr.. of hc_lo_ (row stride ld)
  void hc_combine(int ld, int T) {
    cl_kernel k = kernel("ops", "", "hc_combine");
    int l = ld, off = hp.hc_lr, e = hp.n_embd, hc = hp.hc_n;
    set_args(k, 0, x_, hco_, hc_lo_, l, off, e, hc);
    size_t gws[2] = {(size_t)hp.hce(), (size_t)T};
    run("hc_combine", k, 2, gws, nullptr);
  }
  // the token ids before the current position, from the PLE layer's state (restored with snapshots / sessions)
  void ple_sync() {
    int32_t v[2] = {0, 0};
    CL_CHECK(clEnqueueReadBuffer(cl_.queue, layers_[hp.ple_layer].conv_state, CL_TRUE, ple_tok_off(), 8, v, 0, nullptr,
                                 nullptr));
    ple_prev_[0] = v[0] - 1;  // (stored + 1: a reset state of zeros reads as "no token")
    ple_prev_[1] = v[1] - 1;
  }
  // n-gram rows of the chunk's tokens (host hash, rows read from the table in the GGUF) into ple_emb_; advances the
  // token history (host and the PLE layer's state). llama.cpp qwen4exp: predecessors that are missing (sequence
  // start) or an EOS cut the window; everything at or before the cut reads as EOS.
  void ple_embed(const int* tokens, int T) {
    const auto h0 = hclk::now();
    struct Acc { double& t; hclk::time_point a; ~Acc() { t += hms(a); } } acc{hst_ple_ms_, h0};
    const int ng = hp.ple_ng, per = hp.ple_per, dim = hp.ple_dim, E = hp.n_embd;
    const int64_t eos = hp.ple_eos;
    const size_t prb = ple_table_->nbytes / ple_table_->dims[1];
    const int p0 = ple_prev_[0], p1 = ple_prev_[1];
#pragma omp parallel for if (T > 16)
    for (int t = 0; t < T; ++t) {
      int64_t ctxv[8];
      ctxv[0] = tokens[t];
      bool cut = false;
      for (int sft = 1; sft < ng; ++sft) {
        const int j = t - sft;
        const int64_t tk = cut ? -1 : j >= 0 ? tokens[j] : j == -1 ? p0 : j == -2 ? p1 : -1;
        cut = cut || tk < 0 || tk == eos;
        ctxv[sft] = cut ? eos : tk;
      }
      for (int n = 2; n <= ng; ++n) {
        uint64_t mixed = (uint64_t)ctxv[0] * (uint64_t)hp.ple_mult[0];
        for (int j = 1; j < n; ++j) mixed ^= (uint64_t)ctxv[j] * (uint64_t)hp.ple_mult[j];
        for (int g = 0; g < per; ++g) {
          const int h = (n - 2) * per + g;
          const uint64_t row = mixed % (uint64_t)hp.ple_voc[h] + (uint64_t)hp.ple_off[h];
          dequant_row(ple_table_->type, ple_table_->data + row * prb, dim, &ple_host_[(size_t)t * E + (size_t)h * dim]);
        }
      }
    }
    CL_CHECK(clEnqueueWriteBuffer(cl_.queue, ple_emb_, CL_TRUE, 0, (size_t)T * E * 4, ple_host_.data(), 0, nullptr, nullptr));
    ple_prev_[1] = T >= 2 ? tokens[T - 2] : ple_prev_[0];
    ple_prev_[0] = tokens[T - 1];
    const int32_t v[2] = {ple_prev_[0] + 1, ple_prev_[1] + 1};
    CL_CHECK(clEnqueueWriteBuffer(cl_.queue, layers_[hp.ple_layer].conv_state, CL_TRUE, ple_tok_off(), 8, v, 0, nullptr,
                                  nullptr));
  }
  // the PLE block on the residual streams (before the layer's first mixer): gates from the key / query match, then
  // the dilated causal conv over the normed gated values (history in the layer's state)
  void ple_block(Layer& L, int T, int Mp) {
    const int E = hp.n_embd, W = hp.hce(), a32 = act32_for(false);
    quant_rows(ple_emb_, E, Mp, T, a32);
    matmul("ple_key", L.ple_key, Mp, hc_gate_, W, 0, 0, a32);
    matmul("ple_value", L.ple_value, Mp, hco_, E, 0, 0, a32);
    float eps = hp.eps;
    {
      cl_kernel k = kernel("ops", "", "ple_gate");
      int e = E, hc = hp.hc_n;
      set_args(k, 0, hc_gate_, hco_, x_, L.ple_nk, L.ple_nq, L.ple_nc, e, hc, eps, hc_xn_, big_);
      size_t gws[2] = {256, (size_t)T}, lws[2] = {256, 1};
      run("ple", k, 2, gws, lws);
    }
    {
      cl_kernel k = kernel("ops", "", "ple_conv_add");
      int w = W, nt = hp.ple_taps, dil = hp.ple_ng, hist = ple_hist();
      set_args(k, 0, x_, hc_xn_, big_, ple_hist_, L.ple_cw, w, nt, dil, hist);
      size_t gws[2] = {(size_t)W, (size_t)T};
      run("ple", k, 2, gws, nullptr);
    }
    {
      cl_kernel k = kernel("ops", "", "ple_hist_update");
      int w = W, tt = T, hist = ple_hist();
      set_args(k, 0, ple_hist_, big_, w, tt, hist);
      size_t g1 = W;
      run("ple", k, 1, &g1, nullptr);
    }
  }

  void layer_gdn(Layer& L, int T, int Mp) {
    const int E = hp.n_embd, C = hp.conv_dim(), nv = hp.ssm_nv, nk = hp.ssm_nk, Di = hp.ssm_d_inner;
    float eps = hp.eps;
    const int a_in = act32_for(false);
    if (!hp.hc) rmsnorm_quant(x_, 0, L.attn_norm, E, Mp, T, a_in);
    // (one token with the stacked weights: one GEMV, the outputs are views of qz_out_)
    cl_mem bigc = big_, zc = z_, gbc = gb_;
    if (cur_T_ == 1 && L.qz.raw) {
      matmul("gemm_qkvz", L.qz, Mp, qz_out_, 0, 0, false, a_in);
      bigc = qz_out_;
      zc = qz_z_;
      if (L.qz_ab) gbc = qz_ab_;
      else matmul("gemm_ab", L.ab, Mp, gb_, 2 * nv, 0, false, a_in);
    } else {
      matmul("gemm_qkv", L.qkv, Mp, big_, C, 0, false, a_in);
      matmul("gemm_z", L.z, Mp, z_, Di, 0, false, a_in);
      matmul("gemm_ab", L.ab, Mp, gb_, 2 * nv, 0, false, a_in);
    }
    {
      int n = T * 2 * nv;
      set_args(k_gdn_gates_, 0, gbc, L.dt_bias, L.ssm_a, nv, n);
      size_t g1 = ((size_t)n + 255) / 256 * 256, l1 = 256;
      run("gdn_gates", k_gdn_gates_, 1, &g1, &l1);
    }
    // (a batched pass: per sequence, on its rows of the buffers and its slot's states)
    auto seq = [&](int r0, int Tq) {
      cl_mem bq = from(bigc, (size_t)r0 * C * 4), cq = from(conv_out_, (size_t)r0 * C * 4),
             gq = from(gbc, (size_t)r0 * 2 * nv * 4), oq = from(o_, (size_t)r0 * Di * 4);
      {
        set_args(k_gdn_conv_, 0, bq, L.conv_state, L.conv_w, C, nk, Tq, eps, cq);
        size_t gws[2] = {(size_t)C / 128 * 16, (size_t)(Tq + conv_ct_ - 1) / conv_ct_}, lws[2] = {64, 1};
        run("gdn_conv", k_gdn_conv_, 2, gws, lws);
        size_t g1 = ((size_t)C + 255) / 256 * 256, l1 = 256;  // (conv_state_update checks c < C)
        if (spec_) {  // (verification pass: what verify_done() needs to replay the accepted tokens)
          CL_CHECK(clEnqueueCopyBuffer(cl_.queue, L.conv_state, L.conv_pre, 0, 0, (size_t)3 * C * 4, 0, nullptr, nullptr));
          CL_CHECK(clEnqueueCopyBuffer(cl_.queue, bq, L.qkvin, 0, 0, (size_t)Tq * C * 4, 0, nullptr, nullptr));
          CL_CHECK(clEnqueueCopyBuffer(cl_.queue, cq, L.xin, 0, 0, (size_t)Tq * C * 4, 0, nullptr, nullptr));
          CL_CHECK(clEnqueueCopyBuffer(cl_.queue, gq, L.abin, 0, 0, (size_t)Tq * 2 * nv * 4, 0, nullptr, nullptr));
        }
        set_args(k_conv_upd_, 0, bq, L.conv_state, C, Tq);
        run("conv_state_update", k_conv_upd_, 1, &g1, &l1);
      }
      {
        // (ARC_GDN1=1: a T = 1 kernel with 4 lanes per state column and 128 GRF; bit-identical but slower: 35B 248-341
        //  vs 226-254 ms per 96 tokens for gdn_recur with vector state loads, which was 316)
        static const bool use1 = getenv("ARC_GDN1") && atoi(getenv("ARC_GDN1")) != 0;
        const bool one = Tq == 1 && use1;
        cl_kernel k = one ? k_gdn_recur1_ : k_gdn_recur_;
        set_args(k, 0, cq, C, gq, nk, nv, L.S, Tq, oq, spec_ ? L.S_next : L.S);
        size_t g1 = (size_t)nv * (one ? 512 : 128), l1 = 64;
        run(one ? "gdn_recur1" : "gdn_recur", k, 1, &g1, &l1);
      }
    };
    if (batch_.empty()) seq(0, T);
    else
      for (const BatchSeq& q : batch_) {
        point_to(q.slot);
        seq(q.row, q.n);
      }
    const int a_out = act32_for(true);
    {
      set_args(k_gnorm_, 0, o_, zc, L.ssm_norm, Di, Mp, T, eps, aq_, asc_, absum_, a_out);
      size_t gws[2] = {256, (size_t)Mp}, lws[2] = {256, 1};
      run("gated_norm_quant", k_gnorm_, 2, gws, lws);
    }
    matmul("gemm_ssm_out", L.ssm_out, Mp, hp.hc ? hco_ : x_, E, 0, hp.hc ? 0 : 1, a_out);
  }

  // ---- Flash Next QSA. The indexer rows of the pass (one GEMM over the block input), the pooled keys of the blocks it
  // completes, the tail state, the queries; when some row sees more blocks than it may select, its scores and mask.
  // Returns whether the attention must use the masks (else it is dense, as while every block is selected).
  bool qsa_prepare(Layer& L, int T, int Mp, int a_in) {
    const int QR = L.qr, D = hp.qsa_dim, H = hp.qsa_heads, ldx = (1 + H) * D, ntop = hp.qsa_topk / QR;
    const std::string defs = "-DQR=" + std::to_string(QR) + " -DQH=" + std::to_string(H);
    matmul("qsa_ix", L.ix, Mp, qsa_ixo_, ldx, 0, 0, a_in);
    float eps = hp.eps, ts = std::pow(hp.rope_base, -2.0f / hp.n_rot);
    int p0 = pos, ld = ldx, nr = hp.n_rot;
    const int b0 = pos / QR, b1 = (pos + T) / QR;
    if (b1 > b0) {
      cl_kernel k = kernel("ops", defs, "qsa_pool");
      int bb = b0;
      set_args(k, 0, qsa_ixo_, ld, p0, L.qtail, L.ix_kn, bb, eps, ts, nr, L.pk);
      size_t g1 = (size_t)16 * (b1 - b0), l1 = 16;
      run("qsa", k, 1, &g1, &l1);
    }
    {
      cl_kernel k = kernel("ops", defs, "qsa_tail");
      int tt = T;
      set_args(k, 0, qsa_ixo_, ld, p0, tt, L.qtail);
      size_t g1 = D, l1 = D;
      run("qsa", k, 1, &g1, &l1);
    }
    if (b1 <= ntop && !getenv("ARC_QSA_ALWAYS")) return false;  // every row sees at most ntop blocks: dense
    {
      cl_kernel k = kernel("ops", defs, "qsa_q");
      set_args(k, 0, qsa_ixo_, ld, p0, L.ix_qn, eps, ts, nr, qsa_qx_);
      size_t gws[2] = {(size_t)16 * H, (size_t)T}, lws[2] = {16, 1};
      run("qsa", k, 2, gws, lws);
    }
    int lds = b1, qmw = qsa_mask_words(ctx), nt = ntop;
    for (int r0 = 0; r0 < T; r0 += 256) {
      const int R = std::min(256, T - r0);
      int rr = r0;
      // (ARC_QSA_RT: query rows per qsa_score_rt work-group, each block's keys read once for them; 1 = qsa_score,
      //  one row per work-group. Flash Next 32K prompt: qsa_score 13.8 -> 2.4 s, 131-146 -> 153-154 tok/s, the same
      //  logits)
      static const int qrt = getenv("ARC_QSA_RT") ? std::max(1, atoi(getenv("ARC_QSA_RT"))) : 8;
      if (qrt > 1) {
        cl_kernel k = kernel("ops", defs + " -DQRT=" + std::to_string(qrt), "qsa_score_rt");
        int RR = R;
        set_args(k, 0, qsa_qx_, L.pk, rr, p0, RR, qsa_sc_, lds);
        size_t gws[2] = {(size_t)(lds + 63) / 64 * 64, (size_t)(R + qrt - 1) / qrt}, lws[2] = {64, 1};
        run("qsa_score", k, 2, gws, lws);
      } else {
        cl_kernel k = kernel("ops", defs, "qsa_score");
        set_args(k, 0, qsa_qx_, L.pk, rr, p0, qsa_sc_, lds);
        size_t gws[2] = {(size_t)(lds + 63) / 64 * 64, (size_t)R}, lws[2] = {64, 1};
        run("qsa_score", k, 2, gws, lws);
      }
      {
        cl_kernel k = kernel("ops", defs, "qsa_select");
        set_args(k, 0, qsa_sc_, lds, rr, p0, nt, qsa_mask_, qmw);
        size_t gws[2] = {256, (size_t)R}, lws[2] = {256, 1};
        run("qsa_select", k, 2, gws, lws);
      }
      static const int dbg_qsa = getenv("ARC_DBG_QSA") ? atoi(getenv("ARC_DBG_QSA")) : -1;
      if (L.il == dbg_qsa && T > 1 && r0 + R == T) {  // (debug: scores and mask of the last row)
        std::vector<float> sc(lds);
        std::vector<uint32_t> mk(qmw);
        CL_CHECK(clEnqueueReadBuffer(cl_.queue, qsa_sc_, CL_TRUE, (size_t)(R - 1) * lds * 4, (size_t)lds * 4, sc.data(), 0, nullptr, nullptr));
        CL_CHECK(clEnqueueReadBuffer(cl_.queue, qsa_mask_, CL_TRUE, (size_t)(T - 1) * qmw * 4, (size_t)qmw * 4, mk.data(), 0, nullptr, nullptr));
        FILE* f = fopen("qsa_dbg_gpu.bin", "wb");
        fwrite(sc.data(), 4, lds, f);
        for (int b = 0; b < lds; ++b) { const float v = (float)((mk[b / 32] >> (b % 32)) & 1); fwrite(&v, 4, 1, f); }
        fclose(f);
      }
    }
    return true;
  }
  // attention with the QSA masks: attn_q8 for prompt chunks, attn_decode row by row (key splits as attn_decode_rows)
  void qsa_attention(Layer& L, int T) {
    const int nh = hp.n_head, nkv = hp.n_head_kv, hd = hp.head_dim, hpg = nh / nkv, QR = L.qr;
    const std::string q = " -DQSA -DQR=" + std::to_string(QR);
    int qmw = qsa_mask_words(ctx);
    if (T > 8) {
      const std::string att = "-cl-intel-256-GRF-per-thread -DHPG=" + std::to_string(hpg) + " -DHS=" + std::to_string(attn_hs()) +
                              " -DTPW=" + std::to_string(attn_tpw()) + sb_opt(256) + q;
      cl_kernel k = kernel("ops", att, "attn_q8");
      int p0 = pos, cx = ctx, nn = nh, TT = T, win = 0, nlo = -1, nhi = -1;
      set_args(k, 0, qi_, qsc_, L.kq, L.ksc, L.vq, L.vsc, cx, nn, p0, TT, o_, win, nlo, nhi, qsa_mask_, qmw);
      const int tpw = attn_tpw();
      size_t gws[2] = {(size_t)(T + tpw - 1) / tpw * 512, (size_t)nkv}, lws[2] = {512, 1};
      run("attention", k, 2, gws, lws);
      return;
    }
    const int hs = attn_hs_for(hpg), nsp = hpg / hs;
    cl_kernel kd = kernel("ops", "-cl-intel-256-GRF-per-thread -DHD=" + std::to_string(hd) + " -DHPG=" + std::to_string(hpg) +
                                     " -DHS=" + std::to_string(hs) + " -DTPW=1" + sb_opt(hd) + v4_opt() + q, "attn_decode");
    int nn = nh, hdd = hd, cp = ctx, w = 0;
    for (int t = 0; t < T; ++t) {
      const int p = pos + t;
      int S = 256;
      while (S > 32 && (p / S + 1) * nkv * nsp < 512) S /= 2;
      int kstart = 0, ns = p / S + 1, pp = p, row = t, prow = 0;
      set_args(kd, 0, qi_, qsc_, L.kq, L.ksc, vq_dec(L), vsc_dec(L), cp, nn, pp, apart_, row, w, kstart, S, qsa_mask_, qmw);
      v4_dec_args(kd, 16, L, p);
      size_t gws[2] = {(size_t)ns * 16, (size_t)(nkv * nsp)}, lws[2] = {16, (size_t)nsp};
      run("attention", kd, 2, gws, lws);
      set_args(k_attn_comb_, 0, apart_, ns, nn, o_, row, hdd, prow);
      size_t g1 = (size_t)nn * hd, l1 = 64;
      run("attn_combine", k_attn_comb_, 1, &g1, &l1);
    }
  }

  void layer_attn(Layer& L, int T, int Mp) {
    const int E = hp.n_embd, nh = hp.n_head, nkv = hp.n_head_kv, hd = hp.head_dim;
    const int ldq = nh * hd * 2 + 2 * nkv * hd;
    const int a_in = act32_for(false);
    if (!hp.hc) rmsnorm_quant(x_, 0, L.attn_norm, E, Mp, T, a_in);
    matmul("gemm_q", L.wq, Mp, big_, ldq, 0, false, a_in);
    matmul("gemm_k", L.wk, Mp, big_, ldq, nh * hd * 2, false, a_in);
    matmul("gemm_v", L.wv, Mp, big_, ldq, nh * hd * 2 + nkv * hd, false, a_in);
    if (!batch_.empty()) {  // a batched pass: each sequence's rows with its slot's caches and position (T <= 8)
      const cl_mem qi0 = qi_, qsc0 = qsc_, o0 = o_;
      for (const BatchSeq& q : batch_) {
        point_to(q.slot);
        float eps = hp.eps, theta_scale = std::pow(hp.rope_base, -2.0f / hp.n_rot), kq_scale = 1.0f / std::sqrt((float)hd);
        int p0 = q.pos, nr = hp.n_rot, cx = ctx, f32a = 0, rs = q.rs, ur = 0, Tq = q.n;
        struct alignas(16) Int4 { int x, y, z, w; } sect = {hp.rope_sect[0], hp.rope_sect[1], hp.rope_sect[2], hp.rope_sect[3]};
        qi_ = from(qi0, (size_t)q.row * nh * hd);
        qsc_ = from(qsc0, (size_t)q.row * nh * 4);
        o_ = from(o0, (size_t)q.row * nh * hd * 4);
        set_args(k_attn_prep_, 0, from(big_, (size_t)q.row * ldq * 4), ldq, nh, nkv, L.q_norm, L.k_norm, Tq, p0, eps,
                 theta_scale, nr, kq_scale, qrope_, L.kc, L.vc, cx, qi_, qsc_, L.kq, L.ksc, L.vq, L.vsc, f32a, rs, qsc_, ur, sect);
        if (opt.v4) set_args(k_attn_prep_, 27, L.vq4, L.vsc4);
        size_t gws[2] = {(size_t)(nh + nkv) * 16, (size_t)Tq}, lws[2] = {16, 1};
        run("attn_prep", k_attn_prep_, 2, gws, lws);
        attn_decode_rows("attention", L, ctx, nkv, nh / nkv, hd, q.pos, Tq, 0);
      }
      qi_ = qi0;
      qsc_ = qsc0;
      o_ = o0;
    } else {
      float eps = hp.eps, theta_scale = std::pow(hp.rope_base, -2.0f / hp.n_rot), kq_scale = 1.0f / std::sqrt((float)hd);
      int p0 = pos, nr = hp.n_rot, cx = ctx, f32a = opt.attn_int8 ? 0 : 1, rs = rope_shift, ur = use_rpos_;
      struct alignas(16) Int4 { int x, y, z, w; } sect = {hp.rope_sect[0], hp.rope_sect[1], hp.rope_sect[2], hp.rope_sect[3]};
      set_args(k_attn_prep_, 0, big_, ldq, nh, nkv, L.q_norm, L.k_norm, T, p0, eps, theta_scale, nr, kq_scale, qrope_,
               L.kc, L.vc, cx, qi_, qsc_, L.kq, L.ksc, L.vq, L.vsc, f32a, rs, rpos_ ? rpos_ : qsc_, ur, sect);
      if (opt.v4) set_args(k_attn_prep_, 27, L.vq4, L.vsc4);  // (the 4-bit copy of V)
      size_t gws[2] = {(size_t)(nh + nkv) * 16, (size_t)T}, lws[2] = {16, 1};
      run("attn_prep", k_attn_prep_, 2, gws, lws);
    }
    const bool qsa = batch_.empty() && L.qr > 0 && qsa_prepare(L, T, Mp, a_in);
    if (!batch_.empty()) {
      // (done above, per sequence)
    } else if (qsa) {
      qsa_attention(L, T);
    } else if (opt.attn_int8 && T <= 8) {
      // flash-decoding, token by token (single tokens and speculative verification): split the keys over many
      // sub-groups, then merge
      attn_decode_rows("attention", L, ctx, nkv, nh / nkv, hd, pos, T, 0);
    } else if (opt.attn_int8 && qwen_ps(hd, nh / nkv).pns && pos + T > qwen_ps(hd, nh / nkv).from) {
      // long prompts: head dims split over sub-groups (attn_ps; one scale per row: bit-identical to attn_q8)
      const PsCfg c = qwen_ps(hd, nh / nkv);
      const int hpg = nh / nkv, lsz = 16 * (hpg / c.phg) * (c.tpw / c.ptt) * c.pns;
      cl_kernel kp = kernel("ops", "-cl-intel-256-GRF-per-thread -DHD=" + std::to_string(hd) + " -DHPG=" + std::to_string(hpg) +
                                       " -DHS=" + std::to_string(attn_hs()) + " -DTPW=" + std::to_string(c.tpw) +
                                       " -DPNS=" + std::to_string(c.pns) + " -DPHG=" + std::to_string(c.phg) +
                                       " -DPTT=" + std::to_string(c.ptt) + sb_opt(hd), "attn_ps");
      int p0 = pos, cx = ctx, nn = nh, TT = T, w = 0;
      set_args(kp, 0, qi_, qsc_, L.kq, L.ksc, L.vq, L.vsc, cx, nn, p0, TT, o_, w);
      size_t gws[2] = {(size_t)(T + c.tpw - 1) / c.tpw * lsz, (size_t)nkv}, lws[2] = {(size_t)lsz, 1};
      run("attention", kp, 2, gws, lws);
    } else if (opt.attn_int8) {
      int p0 = pos, cx = ctx;
      int win = 0, nlo = -1, nhi = -1;
      set_args(k_attn_q8_, 0, qi_, qsc_, L.kq, L.ksc, L.vq, L.vsc, cx, nh, p0, T, o_, win, nlo, nhi);
      const int tpw = attn_tpw();
      size_t gws[2] = {(size_t)(T + tpw - 1) / tpw * 512, (size_t)nkv}, lws[2] = {512, 1};
      run("attention", k_attn_q8_, 2, gws, lws);
    } else {
      int p0 = pos, cx = ctx;
      set_args(k_attn_, 0, qrope_, L.kc, L.vc, cx, nh, nkv, p0, T, o_);
      size_t gws[3] = {16, (size_t)(nh / gq()), (size_t)Mp}, lws[3] = {16, 1, 16};
      run("attention", k_attn_, 3, gws, lws);
    }
    const int a_out = act32_for(true);
    {
      int K = nh * hd;
      set_args(k_attn_gate_, 0, o_, big_, ldq, K, Mp, T, aq_, asc_, absum_, a_out);
      size_t gws[2] = {256, (size_t)Mp}, lws[2] = {256, 1};
      run("attn_gate_quant", k_attn_gate_, 2, gws, lws);
    }
    matmul("gemm_o", L.wo, Mp, hp.hc ? hco_ : x_, E, 0, hp.hc ? 0 : 1, a_out);
  }

  void layer_ffn(Layer& L, int T, int Mp) {
    const int E = hp.n_embd, F = hp.n_ff;
    const int a_in = act32_for(false);
    rmsnorm_quant(x_, 0, L.post_norm, E, Mp, T, a_in);
    // prompt chunks: the up GEMM applies silu(gate) * up in its epilogue (no split-K for K = n_embd <= 8192)
    const int fused = cur_T_ > 8 && E <= 8192 ? 1 : 0;
    if (fused) {
      matmul("gemm_gate", L.gate, Mp, big_, F, 0, 0, a_in);
      matmul("gemm_up", L.up, Mp, big_, F, 0, 2, a_in);
    } else {
      matmul("gemm_gate", L.gate, Mp, big_, 2 * F, 0, 0, a_in);
      matmul("gemm_up", L.up, Mp, big_, 2 * F, F, 0, a_in);
    }
    const int a_dn = act32_for(false);
    {
      int gelu = 0;
      set_args(k_swiglu_, 0, big_, F, Mp, T, aq_, asc_, absum_, a_dn, fused, gelu);
      size_t gws[2] = {256, (size_t)Mp}, lws[2] = {256, 1};
      run("swiglu_quant", k_swiglu_, 2, gws, lws);
    }
    matmul("gemm_down", L.down, Mp, x_, E, 0, true, a_dn);
  }

  // Mixture-of-experts FFN (qwen35moe): x += sigmoid(g . h) * shared(h) + sum_k w_k * expert_k(h), h = norm(x), over
  // the top n_expert_used experts of the router softmax (weights renormalized over the selection). The (token,
  // expert) pairs are grouped by expert into slots; tiles of up to `tm` slots of one expert are the GEMM rows.
  void layer_moe(Layer& L, int T, int Mp) {
    zrt::note_layer = L.il;
    zrt::note_T = T;
    const int E = hp.n_embd, F = hp.n_ff_exp, Fs = hp.n_ff_shexp, NR = hp.router_rows(), NU = hp.n_expert_used;
    const int Fp = hp.n_ff_exp_pad;  // (the experts' h rows and down input width: Flash Next 640 padded to 768)
    const bool g4 = hp.gemma;  // (Gemma 4: no shared expert; the router's input norm of its own, scaled experts)
    if (g4) {  // (the router in f32: its input normed into gmoe_, free until the experts' sum)
      {
        float eps = hp.eps;
        int r0 = 0, EE = E, MM = Mp, TT = T, wh = 1, a32 = 1;
        set_args(k_rmsnorm_, 0, x_, r0, L.rnorm, EE, MM, TT, eps, aq_, asc_, absum_, gmoe_, wh, a32);
        size_t gws[2] = {256, (size_t)Mp}, lws[2] = {256, 1};
        run("rmsnorm_quant", k_rmsnorm_, 2, gws, lws);
      }
      {
        cl_kernel k = kernel("ops", "", "gemv_f32");
        int K = E, N = NR, ld = NR;  // (all the padded rows: the logit past the experts stays 0, not stale)
        set_args(k, 0, gmoe_, K, L.router_f32, N, rl_, ld);
        size_t gws[2] = {(size_t)16 * NR, (size_t)T}, lws[2] = {16, 1};
        run("gemm_router", k, 2, gws, lws);
      }
      rmsnorm_quant(x_, 0, L.ffn_norm2, E, Mp, T, 1);
    } else if (!hp.hc) {
      rmsnorm_quant(x_, 0, L.post_norm, E, Mp, T, 1);
    }
    // router + shared expert gate/up (prompt chunks fuse silu(gate) * up into the up GEMM); one token: one GEMV
    // over the stacked rows when the layer has them (router logits at rgu_out_[0], gate / up in rgu_gu_)
    const int fused = cur_T_ > 8 ? 1 : 0;
    cl_mem rl = rl_, gu = big_;
    static const bool no_rgu = getenv("ARC_NO_RGU") != nullptr;  // (A/B: three GEMVs)
    if (g4) {
      // (done above)
    } else if (T == 1 && L.rgu.raw && !no_rgu) {
      matmul("gemm_router_sh", L.rgu, Mp, rgu_out_, 0, 0, 0, true);
      rl = rgu_out_;
      gu = rgu_gu_;
    } else if (fused) {
      matmul("gemm_router", L.router, Mp, rl_, NR, 0, 0, true);
      matmul("gemm_gate_sh", L.gate_sh, Mp, big_, Fs, 0, 0, true);
      matmul("gemm_up_sh", L.up_sh, Mp, big_, Fs, 0, 2, true);
    } else {
      matmul("gemm_router", L.router, Mp, rl_, NR, 0, 0, true);
      matmul("gemm_gate_sh", L.gate_sh, Mp, big_, 2 * Fs, 0, 0, true);
      matmul("gemm_up_sh", L.up_sh, Mp, big_, 2 * Fs, Fs, 0, true);
    }
    // routing, then the slots grouped by expert and their tiles
    static const int tm_std = getenv("ARC_MOE_TM") ? std::max(1, atoi(getenv("ARC_MOE_TM"))) : 16;  // (standard experts: slots per decoded weight; 2048-token prompt: TM 8 416, 16 473, 32 423, 64 187 tok/s)
    // (two-pass mode, Gemma 4: every pass through the T > 1 routing and moe_rg, whose two passes it runs)
    const bool q2g = q2_ && g4;
    const int tm = T == 1 && !q2g ? 1 : hp.moe_std ? tm_std : 16, n = T * NU;
    const std::string defs = "-cl-intel-256-GRF-per-thread -DNE=" + std::to_string(hp.n_expert) +
                             " -DNU=" + std::to_string(NU) + " -DTM=" + std::to_string(tm) +
                             (Fp != F ? " -DFV=" + std::to_string(F) : std::string()) + (g4 ? " -DESCALE" : "");
    const std::string hld = (Fp != F ? " -DHLD=" + std::to_string(Fp) : std::string()) + (g4 ? " -DMOE_GELU" : "");
    if (T == 1 && !q2g) {  // one token: its NU experts are the tiles
      cl_kernel k = kernel("moe", defs, "moe_route1");
      set_args(k, 0, rl, sgate_, mtiles_, stok_, swt_, minv_);
      if (g4) set_args(k, 6, L.escale);
      size_t g1 = 16, l1 = 16;
      run("moe_route", k, 1, &g1, &l1);
      // streamed experts: guess the next layer's experts with its router on this layer's FFN input (the residual
      // stream changes little from one layer to the next), so stream_layer can read them while the GPU works on
      // this layer. Only a prefetch: the next layer still routes on its own input. Off by default (ARC_XGUESS=1:
      // on): Flash Next 16 % cache, 37 % of the guesses used, 48 % of the misses caught, SSD reads +55 %, no faster
      static const bool guess = getenv("ARC_XGUESS") && atoi(getenv("ARC_XGUESS")) != 0;
      const int nx = L.il + 1;
      if (guess && xc_ && nx < hp.n_layer && layers_[nx].router.raw) {
        if (!pf_rl_) {
          pf_rl_ = buf((size_t)hp.router_rows() * 16 * 4);
          pf_tiles_ = buf((1 + 3 * (size_t)NU) * 4);
          pf_dummy_ = buf((size_t)64 * 4 * 4);
        }
        matmul("moe_guess", layers_[nx].router, Mp, pf_rl_, NR, 0, 0, true);
        set_args(k, 0, pf_rl_, pf_dummy_, pf_tiles_, pf_dummy_, pf_dummy_, pf_dummy_);
        run("moe_guess", k, 1, &g1, &l1);
        pf_next_ = true;
      }
    } else {
      {
        cl_kernel k = kernel("moe", defs, "moe_route");
        set_args(k, 0, rl_, NR, T, mids_, mwts_, sgate_, mcount_);
        if (g4) set_args(k, 7, L.escale);
        size_t gws[2] = {16, (size_t)T}, lws[2] = {16, 1};
        run("moe_route", k, 2, gws, lws);
      }
      {
        cl_kernel k = kernel("moe", defs, "moe_scan");
        set_args(k, 0, mcount_, moffs_, mcursor_, mtiles_, mtstart_);
        size_t g1 = hp.n_expert, l1 = hp.n_expert;
        run("moe_scan", k, 1, &g1, &l1);
      }
      {
        cl_kernel k = kernel("moe", defs, "moe_scatter");
        set_args(k, 0, mids_, mwts_, n, mcursor_, stok_, swt_, minv_);
        size_t g1 = ((size_t)n + 255) / 256 * 256, l1 = 256;
        run("moe_scatter", k, 1, &g1, &l1);
      }
    }
    // (ARC_DUMP_ROUTES=file: append the experts every token chose in this layer, for expert-cache studies:
    //  int32 layer, int32 tokens, then tokens x NU int32 expert ids)
    static FILE* routes = getenv("ARC_DUMP_ROUTES") ? fopen(getenv("ARC_DUMP_ROUTES"), "ab") : nullptr;
    if (routes) {
      std::vector<int> ids((size_t)T * NU);
      if (T == 1) {
        std::vector<int> tl(1 + 3 * NU);
        CL_CHECK(clEnqueueReadBuffer(cl_.queue, mtiles_, CL_TRUE, 0, tl.size() * 4, tl.data(), 0, nullptr, nullptr));
        for (int k = 0; k < NU; ++k) ids[k] = tl[1 + 3 * k];
      } else {
        CL_CHECK(clEnqueueReadBuffer(cl_.queue, mids_, CL_TRUE, 0, ids.size() * 4, ids.data(), 0, nullptr, nullptr));
      }
      const int hdr[2] = {L.il, T};
      fwrite(hdr, 4, 2, routes);
      fwrite(ids.data(), 4, ids.size(), routes);
      fflush(routes);
    }
    // the tile count is only known on the GPU: launch for the most tiles possible (extra work-groups exit)
    const size_t max_tiles = tm == 1 ? (size_t)n : (size_t)(n + tm - 1) / tm + std::min(n, hp.n_expert);
    if (hp.moe_std && T == 1)  // (weight bytes read: the token's NU experts, gate + up + down)
      for (int i = 0; i < 3; ++i)
        rd_bytes += (uint64_t)NU * (i == 2 ? E : F) * ((i == 2 ? F : E) / 256) * gguf_sb_bytes(L.ex_type[i]);
    static const int gr_gu = getenv("ARC_MOE_GR_GU") ? atoi(getenv("ARC_MOE_GR_GU")) : 1,
                     gr_dn = getenv("ARC_MOE_GR_DN") ? atoi(getenv("ARC_MOE_GR_DN")) : 4;  // (rows per sub-group;
    // the down projection's 512-wide rows: 4 per sub-group in generation, 5.1 -> 4.0 ms per token, 1 for prompts)
    // standard experts, chunks of >= gemm_min tokens: decode + int8 GEMM (moe_gemm). Decoding all experts costs ~1 s
    // per chunk, so shorter chunks keep the per-tile kernels (Qwen3.6 35B: 1024 tokens 499 vs 490 tok/s, 2048 tokens
    // 476 -> 642 tok/s; 512: 450 vs 310)
    static const int gemm_min = getenv("ARC_MOE_GEMM_MIN") ? atoi(getenv("ARC_MOE_GEMM_MIN")) : 1152;
    // prompt chunks of rg_min .. rg_max tokens: moe_rg, the int8 GEMM on the raw expert rows (the results of decode +
    // GEMM, bit for bit, so a prompt's logits no longer depend on its chunk size there, without the decode pass).
    // Qwen3.6 35B: 384 / 512 / 768 / 1024 / 1152 tokens 412 -> 424, 440 -> 459, 449 -> 511, 486 -> 520, 529 -> 548
    // tok/s; 256 tokens: 369 vs 359 with the per-tile kernels, 1536: 595 vs 565 with decode + GEMM (each expert's rows
    // are decoded once per 16-token tile here, once per chunk there). ARC_NO_MOE_RG=1: those two only.
    static const int rg_min = getenv("ARC_MOE_RG_MIN") ? atoi(getenv("ARC_MOE_RG_MIN")) : 320,
                     rg_max = getenv("ARC_MOE_RG_MAX") ? atoi(getenv("ARC_MOE_RG_MAX")) : 1280;
    static const bool no_rg = getenv("ARC_NO_MOE_RG") != nullptr;
    // Without decode + GEMM (Flash Next: F padded) only the down projection takes moe_rg, at every chunk size from
    // rg_min: Flash Next, 8K prompt in 2048-token chunks: down 13.8 -> 6.2 s GPU time, 145 -> 165 tok/s, and KL to the
    // CPU reference 0.0221 -> 0.0191 (337 tokens; max 0.96 -> 0.27); its gate / up stay per tile (15.4 vs 13.7 s, KL
    // 0.0230 with both).
    const bool gemm_ok = moe_group_ > 0 && Fp == F && moe_gemm_layer_ok(L);
    const bool rg_ok = !no_rg && hp.moe_std && tm == 16 && (T >= rg_min || q2g) && F % 64 == 0 && moe_rg_layer_ok(L);
    // (Gemma 4 26B-A4B, F 704 padded to 768: no decode + GEMM, so gate / up take moe_rg at every chunk size from
    //  rg_min as the down projection does; ARC_G4_RG_GU=0: per tile)
    static const bool g4_rg_gu = !getenv("ARC_G4_RG_GU") || atoi(getenv("ARC_G4_RG_GU")) != 0;
    const bool use_rg = rg_ok && ((gemm_ok && T < rg_max) || (g4 && !gemm_ok && g4_rg_gu)),
               rg_down = rg_ok && (T < rg_max || !gemm_ok);
    const bool use_gemm = !use_rg && gemm_ok && tm == 16 && T >= gemm_min;
    const std::string xdef = xc_ ? " -DXCACHE" : "";
    // standard expert types, per tile (zero/kernels/decode.cl moe_gateup_q / moe_down_q): the tiles in `tiles`
    auto gateup_q = [&](cl_mem tiles) {
      const int grows = gr_gu;
      cl_kernel k = kernel("decode", grf_env("ARC_MOE_GRF") + "-DMOE_Q -DQT=" + std::to_string(L.ex_type[0]) + " -DGROWS=" +
                                         std::to_string(grows) + " -DTM=" + std::to_string(tm) + xdef + hld, "moe_gateup_q");
      int M = Mp, K = E, FF = F;
      if (xc_) {
        set_args(k, 0, xc_->xtab[L.il], xc_->xtab[L.il], FF, K, tabs_, aq_, asc_, M, tiles, stok_, mh_, xc_->xtab[L.il],
                 xoff(L, 0), xoff(L, 1));
      } else {
        set_args(k, 0, L.ex_q[0], L.ex_q[1], FF, K, tabs_, aq_, asc_, M, tiles, stok_, mh_);
      }
      size_t gws[2] = {((size_t)(F + grows - 1) / grows * 16 + 255) / 256 * 256, max_tiles}, lws[2] = {256, 1};
      run("moe_gateup", k, 2, gws, lws);
    };
    auto quant_h = [&] {
      cl_kernel k = kernel("moe", defs + (q2_ ? " -DQ2M=" + std::to_string(q2m_hq_) + " -DQ2MS=" + std::to_string(q2m_sc_ / 4)
                                             : std::string()), "moe_quant");
      int FF = Fp;
      set_args(k, 0, mh_, FF, slots_, mhq_, mhsc_);
      size_t gws[2] = {(size_t)16 * Fp / 256, (size_t)n}, lws[2] = {16, 1};
      run("moe_quant", k, 2, gws, lws);
    };
    auto down_q = [&](cl_mem tiles) {
      const int grows = T == 1 ? gr_dn : 1;
      cl_kernel k = kernel("decode", "-DMOE_Q -DQT=" + std::to_string(L.ex_type[2]) + " -DGROWS=" + std::to_string(grows) +
                                         " -DTM=" + std::to_string(tm) + xdef +
                                         (Fp != F ? " -DFV=" + std::to_string(F) : std::string()), "moe_down_q");
      int NN = E, K = Fp, S = slots_;
      if (xc_)
        set_args(k, 0, xc_->xtab[L.il], NN, K, tabs_, mhq_, mhsc_, S, tiles, swt_, my_, xc_->xtab[L.il], xoff(L, 2));
      else
        set_args(k, 0, L.ex_q[2], NN, K, tabs_, mhq_, mhsc_, S, tiles, swt_, my_);
      size_t gws[2] = {((size_t)(E + grows - 1) / grows * 16 + 255) / 256 * 256, max_tiles}, lws[2] = {256, 1};
      run("moe_down", k, 2, gws, lws);
    };
    // One token whose experts are partly not cached: the cached ones' gate / up and down run while stream_layer reads
    // the others from the SSD, then the others' (the same kernels on per-slot rows: bit-identical). The victims of
    // the reads are never experts of this layer, so the running kernels' experts stay put. Flash Next (22 % cached,
    // 84.5 % hits): 6.6 -> 7.3 tok/s. ARC_XSPLIT=0: all after the reads.
    static const bool xsplit = !getenv("ARC_XSPLIT") || atoi(getenv("ARC_XSPLIT")) != 0;
    cl_mem tiles = mtiles_;
    std::vector<int> tl1;
    if (xc_ && T == 1 && !use_gemm && hp.moe_std && xsplit && !q2g) {
      tl1.resize(1 + 3 * (size_t)NU);
      CL_CHECK(clEnqueueReadBuffer(cl_.queue, mtiles_, CL_TRUE, 0, tl1.size() * 4, tl1.data(), 0, nullptr, nullptr));
      xt_hit_h_.assign(1, 0);
      xt_miss_h_.assign(1, 0);
      for (int t = 0; t < tl1[0] && t < NU; ++t) {
        auto& v = xc_->cached(L.il, tl1[1 + 3 * t]) ? xt_hit_h_ : xt_miss_h_;
        v.insert(v.end(), tl1.begin() + 1 + 3 * t, tl1.begin() + 4 + 3 * t);
        ++v[0];
      }
      if (xt_hit_h_[0] > 0 && xt_miss_h_[0] > 0) {
        if (!xt_hit_) {
          xt_hit_ = buf((1 + 3 * (size_t)NU) * 4);
          xt_miss_ = buf((1 + 3 * (size_t)NU) * 4);
        }
        CL_CHECK(clEnqueueWriteBuffer(cl_.queue, xt_hit_, CL_TRUE, 0, xt_hit_h_.size() * 4, xt_hit_h_.data(), 0, nullptr,
                                      nullptr));
        CL_CHECK(clEnqueueWriteBuffer(cl_.queue, xt_miss_, CL_TRUE, 0, xt_miss_h_.size() * 4, xt_miss_h_.data(), 0,
                                      nullptr, nullptr));
        gateup_q(xt_hit_);
        quant_h();
        down_q(xt_hit_);
        tiles = xt_miss_;
      }
    }
    if (xc_) stream_layer(L, T, use_gemm, tl1.empty() ? nullptr : tl1.data());
    if (use_rg) {
      moe_rg(L, 0, Mp, max_tiles);
    } else if (use_gemm) {
      moe_gemm(L, 0, Mp);
    } else if (hp.moe_std) {
      gateup_q(tiles);
    } else {
      cl_kernel k = kernel("moe", defs, "moe_gateup");
      int M = Mp, K = E;
      set_args(k, 0, L.ex_q[0], L.ex_s[0], L.ex_q[1], L.ex_s[1], aq_, asc_, M, K, F, mtiles_, stok_, mh_);
      size_t gws[2] = {(size_t)F, max_tiles}, lws[2] = {16, 1};
      run("moe_gateup", k, 2, gws, lws);
    }
    quant_h();
    // shared expert down: its quantized input replaces the FFN input in aq_ (the expert GEMM above has read it)
    if (!g4) {
      int a32 = 1, gelu = 0;
      set_args(k_swiglu_, 0, gu, Fs, Mp, T, aq_, asc_, absum_, a32, fused, gelu);
      size_t gws[2] = {256, (size_t)Mp}, lws[2] = {256, 1};
      run("swiglu_quant", k_swiglu_, 2, gws, lws);
      matmul("gemm_down_sh", L.down_sh, Mp, msh_, E, 0, 0, true);
    }
    if (rg_down) {
      moe_rg(L, 2, Mp, max_tiles);
    } else if (use_gemm) {
      moe_gemm(L, 2, Mp);
    } else if (hp.moe_std) {
      down_q(tiles);
    } else {
      cl_kernel k = kernel("moe", defs, "moe_down");
      set_args(k, 0, L.ex_q[2], L.ex_s[2], mhq_, mhsc_, slots_, F, E, mtiles_, swt_, my_);
      size_t gws[2] = {(size_t)E, max_tiles}, lws[2] = {16, 1};
      run("moe_down", k, 2, gws, lws);
    }
    {
      cl_kernel k = kernel("moe", defs, "moe_combine");
      if (hp.hc) fill(hco_, (size_t)T * E * 4);  // (Flash Next: the block output alone, combined into the streams after)
      if (g4) fill(gmoe_, (size_t)T * E * 4);    // (Gemma 4: the experts' sum alone; msh_ stays zero)
      set_args(k, 0, g4 ? gmoe_ : hp.hc ? hco_ : x_, msh_, sgate_, my_, minv_, E);
      size_t gws[2] = {((size_t)E + 255) / 256 * 256, (size_t)T}, lws[2] = {256, 1};  // (moe_combine checks c < N)
      run("moe_combine", k, 2, gws, lws);
    }
    // (debug, ARC_DBG_MOE=layer: that layer's shared-expert output, its gate and the FFN output of the last row)
    static const int dbg_moe = getenv("ARC_DBG_MOE") ? atoi(getenv("ARC_DBG_MOE")) : -1;
    if (L.il == dbg_moe && hp.hc && T > 1) {
      std::vector<float> a(E), b(E), g(1);
      CL_CHECK(clEnqueueReadBuffer(cl_.queue, msh_, CL_TRUE, (size_t)(T - 1) * E * 4, E * 4, a.data(), 0, nullptr, nullptr));
      CL_CHECK(clEnqueueReadBuffer(cl_.queue, sgate_, CL_TRUE, (size_t)(T - 1) * 4, 4, g.data(), 0, nullptr, nullptr));
      CL_CHECK(clEnqueueReadBuffer(cl_.queue, hco_, CL_TRUE, (size_t)(T - 1) * E * 4, E * 4, b.data(), 0, nullptr, nullptr));
      FILE* f = fopen("moe_dbg_gpu.bin", "wb");
      fwrite(a.data(), 4, E, f); fwrite(g.data(), 4, 1, f); fwrite(b.data(), 4, E, f);
      fclose(f);
    }
  }

  static bool moe_gemm_layer_ok(const Layer& L) {
    for (int i = 0; i < 3; ++i)
      if (!is_w8_type(L.ex_type[i]) || L.ex_type[i] == GGML_Q8_0 || L.ex_type[i] == GGML_Q4_0 || w8_variant(L.ex_type[i]) >= 2)
        return false;  // (the per-tile GEMM has no min term and no native Q8_0 form)
    return true;
  }
  bool moe_gemm_ok() const {
    if (!hp.moe_std || getenv("ARC_NO_MOE_GEMM")) return false;
    for (const Layer& L : layers_)
      if (!moe_gemm_layer_ok(L)) return false;
    return true;
  }
  // moe_rg's types: no per-block mins, not requantized by decode_w8 (IQ1_M: its own scale layout)
  static bool moe_rg_type(uint32_t t) {
    return t == GGML_IQ4_XS || t == GGML_IQ3_S || t == GGML_IQ3_XXS || t == GGML_IQ2_XXS || t == GGML_IQ2_XS ||
           t == GGML_IQ2_S || t == GGML_Q6_K || t == GGML_Q3_K || t == GGML_IQ4_NL || t == GGML_Q2_0 ||
           t == GGML_Q4_0 || t == GGML_Q8_0;
  }
  static bool moe_rg_layer_ok(const Layer& L) {
    for (int i = 0; i < 3; ++i)
      if (!moe_rg_type(L.ex_type[i])) return false;
    return true;
  }
  // Experts of a prompt chunk through zero/kernels/decode.cl moe_rg (the int8 GEMM of zero/kernels/gemm.cl on the raw rows,
  // decoded in registers): part 0 = gate, then up with the fused SwiGLU into mh_; part 2 = down, routing-weighted,
  // into my_. All the chunk's tiles in one launch per matrix (work-groups past the tile count exit at once).
  void moe_rg(Layer& L, int part, int Mp, size_t max_tiles) {
    const int E = hp.n_embd, F = hp.n_ff_exp, NE = hp.n_expert;
    static const int ny_env = getenv("ARC_MOE_RG_NY") ? atoi(getenv("ARC_MOE_RG_NY")) : 0;
    const std::string xd = xc_ ? " -DXCACHE" : "";
    for (int i = part; i < (part == 0 ? 2 : 3); ++i) {
      cl_kernel k = kernel("decode", "-cl-intel-256-GRF-per-thread -DMOE_RG -DQT=" + std::to_string(L.ex_type[i]) + xd,
                           "moe_rg");
      // (Flash Next: h rows of Fp = 768 for F = 640 values; the down projection's rows padded to Fp with zeros)
      const int Fp = hp.n_ff_exp_pad;
      int N = i == 2 ? E : F, K = i == 2 ? Fp : E, M = i == 2 ? slots_ : Mp, ldc = i == 2 ? E : Fp,
          acc = i == 1 ? (hp.gemma ? 3 : 2) : 0, ne = NE,
          um = i == 2 ? 0 : 1, us = i == 2 ? 1 : 0;
      int kv = i == 2 ? F : K;
      // (two-pass mode: the matrix on the values, then on their residual, added; the up projection into mh2_ and
      //  the GELU product after both)
      const int passes = q2_ ? 2 : 1;
      for (int ps = 0; ps < passes; ++ps) {
        cl_mem a = i == 2 ? (ps ? from(mhq_, q2m_hq_) : mhq_) : (ps ? from(aq_, q2_aq_) : aq_);
        cl_mem as = i == 2 ? (ps ? from(mhsc_, q2m_sc_) : mhsc_) : (ps ? from(asc_, q2_sc_) : asc_);
        cl_mem c = i == 2 ? my_ : (q2_ && i == 1 ? mh2_ : mh_);
        int ac = q2_ ? ps : acc;
        if (xc_)
          set_args(k, 0, xc_->xtab[L.il], N, K, tabs_, a, as, c, M, ldc, ac, mtiles_, mtstart_, ne, stok_, um, swt_, us, kv,
                   xc_->xtab[L.il], xoff(L, i));
        else
          set_args(k, 0, L.ex_q[i], N, K, tabs_, a, as, c, M, ldc, ac, mtiles_, mtstart_, ne, stok_, um, swt_, us, kv);
        const size_t ny = ny_env > 0 ? (size_t)ny_env : max_tiles;
        size_t gws[2] = {(size_t)N, ny}, lws[2] = {16, 1};
        run(i == 2 ? "moe_down" : "moe_gateup", k, 2, gws, lws);
      }
      if (q2_ && i == 1) {
        if (!hp.gemma) throw std::logic_error("two-pass experts: Gemma 4 only (the GELU product)");
        cl_kernel g = kernel("moe", "-DNE=" + std::to_string(NE) + " -DNU=" + std::to_string(hp.n_expert_used) + " -DTM=16",
                             "moe_gelu_mul");
        int ld = Fp, fv = F;
        set_args(g, 0, mh_, mh2_, ld, fv);
        size_t gws[2] = {(size_t)Fp, (size_t)cur_T_ * hp.n_expert_used};  // (the pass's slots: T x NU rows)
        run("moe_gateup", g, 2, gws, nullptr);
      }
    }
  }
  // Experts of a prompt chunk through the int8 GEMM: part 0 = gate + up (h = silu(x Wg^T) * x Wu^T into mh_),
  // part 2 = down (routing-weighted into my_). Per group of moe_group_ experts: decode to W8, then one GEMM over
  // the group's slot tiles (tiles per expert from moe_scan's tstart).
  void moe_gemm(Layer& L, int part, int Mp) {
    const int E = hp.n_embd, F = hp.n_ff_exp, NE = hp.n_expert, G = moe_group_;
    for (int i = part; i < (part == 0 ? 2 : 3); ++i) {
      const int N = i == 2 ? E : F, K = i == 2 ? F : E;
      const size_t eb = (size_t)N * (K / 256) * raw_padded_sb_bytes(L.ex_type[i]);  // raw bytes per expert
      if (L.ex_sub[i].empty() && !xc_) {
        const size_t total = NE * eb + kRawSlack;  // (upload_padded: the superblocks + the slack)
        for (int e0 = 0; e0 < NE; e0 += G) {
          cl_buffer_region reg{e0 * eb, std::min(total - e0 * eb, G * eb + kRawSlack)};
          cl_int err;
          L.ex_sub[i].push_back(clCreateSubBuffer(L.ex_q[i], CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &reg, &err));
          CL_CHECK(err);
          mems_.push_back(L.ex_sub[i].back());
        }
      }
      cl_kernel k = kernel("gemm", "-cl-intel-256-GRF-per-thread -DACT32 -DMOE_TILES -DQTYPE=" +
                                       std::to_string(100 + w8_variant(L.ex_type[i])), "gemm");
      // tile work-groups per launch: about the average tile count of a group (more loop over the tiles)
      const size_t ny = std::max<size_t>(64, (size_t)cur_T_ * hp.n_expert_used / 16 / (NE / G) + G);
      for (int e0 = 0; e0 < NE; e0 += G) {
        if (xc_)
          decode_w8_x(L, i, e0, G, N, K);
        else
          decode_w8(L.ex_sub[i][e0 / G], L.ex_type[i], G * N, K, moe_wq_, moe_wh_);
        int M = i == 2 ? slots_ : Mp, NN = N, KK = K, ldc = N, acc = i == 1 ? 2 : 0, e1 = e0 + G, um = i == 2 ? 0 : 1,
            us = i == 2 ? 1 : 0;
        set_args(k, 0, moe_wq_, moe_wh_, i == 2 ? mhq_ : aq_, i == 2 ? mhsc_ : asc_, absum_, i == 2 ? my_ : mh_, M, NN,
                 KK, ldc, acc, mtiles_, mtstart_, e0, e1, stok_, um, swt_, us);
        size_t gws[2] = {(size_t)N, ny}, lws[2] = {16, 1};
        run(i == 2 ? "moe_down" : "moe_gateup", k, 2, gws, lws);
      }
    }
  }

  // Final norm of rows [r0, r0 + n) of x_ (per-32 layout, as the GEMVs expect; the f32 rows are also written to
  // `hout` when given), then the lm_head: logits_ row i belongs to x_ row r0 + i.
  void head_rows(int r0, int n, cl_mem normw, cl_mem hout) {
    const int saved = cur_T_;
    cur_T_ = n;
    if (hp.hc) {  // Flash Next: the final mixer of the streams is the output norm
      hc_mix(out_hc_norm_, out_hc_down_, out_hc_up_, n, 16, 1, r0);
      matmul("lm_head", lm_head_, 16, logits_, hp.n_vocab, 0, 0, true);
      cur_T_ = saved;
      return;
    }
    float eps = hp.eps;
    int E = hp.n_embd, M = 16, nn = n, wh = hout ? 1 : 0, a32 = 1;
    set_args(k_rmsnorm_, 0, x_, r0, normw, E, M, nn, eps, aq_, asc_, absum_, hout ? hout : big_, wh, a32);
    size_t gws[2] = {256, (size_t)n}, lws[2] = {256, 1};
    run("rmsnorm_quant", k_rmsnorm_, 2, gws, lws);
    matmul("lm_head", lm_head_, 16, logits_, hp.n_vocab, 0, 0, true);
    cur_T_ = saved;
  }
  void head(int T) { head_rows(T - 1, 1, out_norm_, has_mtp ? hlast_ : nullptr); }

  // ---------------------------------------------------------------- Gemma 4 assistant drafter
  // One drafter step for token `tok` at position P (the same position for every chained draft, as in llama.cpp and
  // transformers) with main-model hidden state `hsrc` (after the output norm): logits -> logits_, the next hidden
  // state -> d_h_. The attention of each drafter layer reads the main model's cache up to position P - 1.
  void gemma_draft_step(int tok, cl_mem hsrc, int P) {
    const int Em = hp.n_embd, E = d_E_, saved_T = cur_T_;
    cur_T_ = 1;
    const size_t rb = tok_embd_->nbytes / hp.n_vocab;
    dequant_row(tok_embd_->type, tok_embd_->data + (size_t)tok * rb, Em, x_host_.data());
    const float sc = std::sqrt((float)Em);
    for (int i = 0; i < Em; ++i) x_host_[i] *= sc;
    CL_CHECK(clEnqueueWriteBuffer(cl_.queue, d_cat_, CL_TRUE, 0, (size_t)Em * 4, x_host_.data(), 0, nullptr, nullptr));
    CL_CHECK(clEnqueueCopyBuffer(cl_.queue, hsrc, d_cat_, 0, (size_t)Em * 4, (size_t)Em * 4, 0, nullptr, nullptr));
    quant_rows(d_cat_, 2 * Em, 16, 1, 1);
    matmul("draft", d_pre_, 16, d_x_, E, 0, 0, true);
    for (auto& D : dl_) gemma_draft_layer(D, P);
    rmsnorm_quant(d_x_, 0, d_norm_, E, 16, 1, 1);
    matmul("draft", d_lm_, 16, logits_, hp.n_vocab, 0, 0, true);
    matmul("draft", d_post_, 16, d_h_, Em, 0, 0, true);
    cur_T_ = saved_T;
  }
  void gemma_draft_layer(DraftLayer& D, int P) {
    const int E = d_E_, F = d_F_, nh = d_nh_, hd = D.hd;
    Layer& TL = layers_[D.swa ? hp.n_layer - 2 : hp.n_layer - 1];
    const int nkv = TL.nkv, hpg = nh / nkv;
    rmsnorm_quant(d_x_, 0, D.attn_norm, E, 16, 1, 1);
    matmul("draft", D.wq, 16, big_, nh * hd, 0, 0, true);
    {
      cl_kernel k = kernel("ops", "-DHD=" + std::to_string(hd) + sb_opt(hd), "attn_prep_g");
      float eps = hp.eps, base = D.swa ? d_base_swa_ : d_base_, qscale = 1.0f;
      int ld = nh * hd, nn = nh, nk = nkv, hv = 0, uff = D.rope_ff ? 1 : 0, TT = 1, p0 = P, cap = TL.cap;
      cl_mem no_table = nullptr;  // the angles are computed in the kernel
      set_args(k, 0, big_, ld, nn, nk, hv, D.q_norm, D.q_norm, TT, p0, eps, base, D.rope_ff ? D.rope_ff : D.q_norm, uff,
               qscale, qi_, qsc_, TL.kq, TL.ksc, TL.vq, TL.vsc, cap, no_table);
      int f16 = 0;
      set_args(k, 22, f16, qsc_, qsc_, qsc_);
      size_t gws[2] = {(size_t)nh * 16, 1}, lws[2] = {16, 1};  // q heads only: the cache is the main model's
      run("draft", k, 2, gws, lws);
    }
    // keys P - n_swa + 1 .. P - 1 (sliding) or 0 .. P - 1 (global): query row at P - 1 with a window one shorter
    attn_decode_rows("draft", TL, TL.cap, nkv, hpg, hd, P - 1, 1, D.swa ? d_swa_ - 1 : 0, nh);
    quant_rows(o_, nh * hd, 16, 1, 1);
    matmul("draft", D.wo, 16, big_, E, 0, 0, true);
    draft_add_norm(D.post_attn, 1.0f);
    rmsnorm_quant(d_x_, 0, D.ffn_norm, E, 16, 1, 1);
    matmul("draft", D.gate, 16, big_, 2 * F, 0, 0, true);
    matmul("draft", D.up, 16, big_, 2 * F, F, 0, true);
    {
      int gelu = 1, FF = F, MM = 16, TT = 1, a32 = 1, fused = 0;
      set_args(k_swiglu_, 0, big_, FF, MM, TT, aq_, asc_, absum_, a32, fused, gelu);
      size_t gws[2] = {256, 16}, lws[2] = {256, 1};
      run("draft", k_swiglu_, 2, gws, lws);
    }
    matmul("draft", D.down, 16, big_, E, 0, 0, true);
    draft_add_norm(D.post_ffw, D.out_scale);
  }
  // d_x_ = (d_x_ + rmsnorm(big_) * w) * s (one row)
  void draft_add_norm(cl_mem w, float s) {
    if (!k_add_norm_) k_add_norm_ = kernel("ops", "", "add_rmsnorm");
    int E = d_E_;
    float eps = hp.eps, sc = s;
    set_args(k_add_norm_, 0, d_x_, big_, w, E, eps, sc);
    size_t gws[2] = {256, 1}, lws[2] = {256, 1};
    run("draft", k_add_norm_, 2, gws, lws);
  }
  std::vector<int> gemma_draft(int tok, int max_drafts) {
    struct Q2Off { bool& f; bool was; ~Q2Off() { f = was; } } q2_keep{q2_off_, q2_off_};  // (the drafter: one pass)
    q2_off_ = true;
    std::vector<int> drafts;
    cl_mem h = hlast_;
    int t = tok;
    for (;;) {
      gemma_draft_step(t, h, pos);
      CL_CHECK(clEnqueueReadBuffer(cl_.queue, logits_, CL_TRUE, 0, (size_t)hp.n_vocab * 4, mtp_logits_host_.data(), 0,
                                   nullptr, nullptr));
      const auto mx = std::max_element(mtp_logits_host_.begin(), mtp_logits_host_.end());
      const int d = (int)(mx - mtp_logits_host_.begin());
      drafts.push_back(d);
      if ((int)drafts.size() >= max_drafts) break;
      double z = 0;
      for (float l : mtp_logits_host_) z += std::exp((double)l - *mx);
      if (1.0 / z < opt.mtp_min_p) break;  // not confident: verify fewer drafts
      t = d;
      h = d_h_;
    }
    collect_profile();
    return drafts;
  }


  // ---------------------------------------------------------------- DFlash2 block drafter (z-lab/dflash model.py)
  // Context: the target's hidden states after the layers dfl_tl_ (concatenated, fc, hidden_norm) give every
  // committed position a K / V entry per drafter layer (ring of dfl_R_ positions; attention looks back dfl_win_). A
  // draft is one pass over a block of dfl_B_ rows [token, mask...] at positions pos..: 5 layers (pre-norm attention and
  // MLP, each wrapped in a two-tap grouped dynamic causal conv, keys = context + block, non-causal), the target's
  // output head, then the candidate selector (top-k per position, a greedy path through bilinear codebook scores).
 public:
  bool dfl_ = false;
 private:
  struct DflLayer {
    GpuWeight q, k, v, o, gate, up, down, akp, mkp;
    cl_mem in_norm = nullptr, post_norm = nullptr, qn = nullptr, kn = nullptr, abase = nullptr, mbase = nullptr,
           rk = nullptr, rv = nullptr;
  };
  std::vector<DflLayer> dfl_l_;
  std::vector<GpuWeight> dfl_fc_;  // fc columns of each target layer (E x E)
  GpuWeight dfl_hproj_;
  cl_mem dfl_hnorm_ = nullptr, dfl_norm_ = nullptr;
  std::vector<int> dfl_tl_;
  int dfl_B_ = 8, dfl_K_ = 2, dfl_G_ = 16, dfl_mask_ = 0, dfl_topk_ = 16, dfl_rank_ = 256, dfl_win_ = 2048, dfl_R_ = 4096;
  int dfl_nh_ = 32, dfl_nkv_ = 8, dfl_F_ = 0, dfl_cap_lo_ = 0, dfl_cap_base_ = 0, dfl_cap_end_ = 0;
  float dfl_theta_ = 1e7f, dfl_eps_ = 1e-6f;
  std::vector<cl_mem> dfl_acc_;  // fc accumulators of the captured chunks
  cl_mem dfl_x_ = nullptr, dfl_h_ = nullptr, dfl_t_ = nullptr, dfl_dyn_ = nullptr, dfl_q_ = nullptr, dfl_k_ = nullptr,
         dfl_v_ = nullptr, dfl_o_ = nullptr, dfl_gu_ = nullptr, dfl_hp_ = nullptr, dfl_ck_ = nullptr, dfl_cv_ = nullptr;
  // the selector codebooks (bf16, host) from the mapped safetensors file
  HANDLE dfl_file_ = INVALID_HANDLE_VALUE, dfl_map_ = nullptr;
  const uint8_t* dfl_base_ = nullptr;
  const uint16_t *dfl_pred_ = nullptr, *dfl_succ_ = nullptr;
  std::vector<float> dfl_lg_, dfl_hph_;
  uint64_t dfl_steps_ = 0;

  struct StTensor { std::string dtype; std::vector<int64_t> shape; const uint8_t* data = nullptr; size_t n = 0; };
  std::map<std::string, StTensor> dfl_st_;
  void dfl_open(const std::string& file) {
    dfl_file_ = CreateFileA(file.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (dfl_file_ == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot open " + file);
    dfl_map_ = CreateFileMappingA(dfl_file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
    dfl_base_ = dfl_map_ ? (const uint8_t*)MapViewOfFile(dfl_map_, FILE_MAP_READ, 0, 0, 0) : nullptr;
    if (!dfl_base_) throw std::runtime_error("cannot map " + file);
    uint64_t hn = 0;
    memcpy(&hn, dfl_base_, 8);
    const Json h = Json::parse(std::string((const char*)dfl_base_ + 8, (size_t)hn));
    for (const auto& [name, t] : h.entries()) {
      if (name == "__metadata__") continue;
      StTensor st;
      st.dtype = t["dtype"].str();
      for (const auto& d : t["shape"].items()) st.shape.push_back(d.as_int());
      st.data = dfl_base_ + 8 + hn + (size_t)t["data_offsets"][0].as_int();
      st.n = (size_t)(t["data_offsets"][1].as_int() - t["data_offsets"][0].as_int());
      dfl_st_[name] = st;
    }
  }
  const StTensor& dfl_t(const std::string& n) {
    auto it = dfl_st_.find(n);
    if (it == dfl_st_.end() || it->second.dtype != "BF16") throw std::runtime_error("DFlash: missing bf16 tensor " + n);
    return it->second;
  }
  std::vector<float> dfl_f32(const StTensor& t, size_t off = 0, size_t n = 0) {
    if (!n) n = t.n / 2;
    std::vector<float> v(n);
    const uint16_t* b = (const uint16_t*)t.data + off;
#pragma omp parallel for if (n > 65536)
    for (int64_t i = 0; i < (int64_t)n; ++i) v[i] = bf16_to_f32(b[i]);
    return v;
  }
  cl_mem dfl_vec(const std::string& n) {
    const auto v = dfl_f32(dfl_t(n));
    return buf(v.size() * 4, v.data());
  }
  // a bf16 matrix [N][K] (columns k0 .. k0 + K of rows of Kfull) as a Q4_0 weight
  GpuWeight dfl_w(const std::string& n, int k0 = 0, int Kc = 0) {
    const StTensor& t = dfl_t(n);
    const int N = (int)t.shape[0], Kf = (int)t.shape[1], K = Kc ? Kc : Kf;
    std::vector<float> v((size_t)N * K);
    const uint16_t* b = (const uint16_t*)t.data;
#pragma omp parallel for
    for (int r = 0; r < N; ++r)
      for (int c = 0; c < K; ++c) v[(size_t)r * K + c] = bf16_to_f32(b[(size_t)r * Kf + k0 + c]);
    // Q4_0: the drafter's weights are read for every draft block; at half the bytes of Q8_0 its layers take ~45 %
    // less time (27B code: 539 -> 296 ms per answer, 16.9 -> 17.2 tok/s; 35B 160 -> 105 ms), the drafts are taken
    // as often (27B code 84.5 vs 85.4 %, 35B equal) and the drafter needs 0.84 GiB less (27B). ARC_DFL_Q8=1: Q8_0.
    static const bool q4 = getenv("ARC_DFL_Q8") == nullptr;
    if (q4) {
      auto q = quantize_rows_q4_0(v.data(), N, K);
      return make_weight(GGML_Q4_0, q.data(), q.size(), N, K);
    }
    auto q8 = quantize_rows_q8_0(v.data(), N, K);
    return make_weight(GGML_Q8_0, q8.data(), q8.size(), N, K);
  }
  void load_dflash(std::string path) {
    namespace fs = std::filesystem;
    if (fs::is_directory(path)) path = (fs::path(path) / "model.safetensors").string();
    const std::string cfgp = (fs::path(path).parent_path() / "config.json").string();
    const Json cfg = Json::parse(read_file(cfgp));
    const Json& dc = cfg["dflash_config"];
    const int E = hp.n_embd;
    if (cfg["hidden_size"].as_int() != E || cfg["head_dim"].as_int() != 128 || !dc.is_object())
      throw std::runtime_error("DFlash: drafter does not match this model");
    dfl_B_ = (int)dc["block_size"].as_int(8);
    dfl_K_ = (int)dc["conv_kernel_size"].as_int(2);
    dfl_G_ = (int)dc["conv_group_size"].as_int(16);
    dfl_mask_ = (int)dc["mask_token_id"].as_int(0);
    dfl_topk_ = (int)dc["selector_top_k"].as_int(16);
    dfl_rank_ = (int)dc["selector_rank"].as_int(256);
    for (const auto& x : dc["target_layer_ids"].items()) dfl_tl_.push_back((int)x.as_int());
    dfl_nh_ = (int)cfg["num_attention_heads"].as_int();
    dfl_nkv_ = (int)cfg["num_key_value_heads"].as_int();
    dfl_F_ = (int)cfg["intermediate_size"].as_int();
    dfl_win_ = (int)cfg["sliding_window"].as_int(2048);
    dfl_eps_ = (float)cfg["rms_norm_eps"].as_number(1e-6);
    dfl_theta_ = (float)cfg["rope_parameters"]["rope_theta"].as_number(1e7);
    if (dfl_B_ < 2 || dfl_B_ > 8 || dfl_K_ != 2 || E % dfl_G_ || dfl_F_ % 256 || dfl_win_ + 64 > dfl_R_ || dfl_tl_.empty() ||
        dfl_nh_ * 128 > 8192 || dfl_topk_ < 1 || dfl_topk_ > 64)
      throw std::runtime_error("DFlash: unsupported drafter configuration");
    for (int l : dfl_tl_)
      if (l < 0 || l >= hp.n_layer) throw std::runtime_error("DFlash: target layer out of range");
    dfl_open(path);
    const int nl = (int)cfg["num_hidden_layers"].as_int();
    for (size_t i = 0; i < dfl_tl_.size(); ++i) dfl_fc_.push_back(dfl_w("fc.weight", (int)i * E, E));
    dfl_hnorm_ = dfl_vec("hidden_norm.weight");
    dfl_norm_ = dfl_vec("norm.weight");
    dfl_hproj_ = dfl_w("candidate_selector.hidden_projection.weight");
    dfl_pred_ = (const uint16_t*)dfl_t("candidate_selector.predecessor_codebook").data;
    dfl_succ_ = (const uint16_t*)dfl_t("candidate_selector.successor_codebook").data;
    dfl_l_.resize(nl);
    for (int l = 0; l < nl; ++l) {
      DflLayer& D = dfl_l_[l];
      const std::string p = "layers." + std::to_string(l) + ".";
      D.q = dfl_w(p + "self_attn.q_proj.weight");
      D.k = dfl_w(p + "self_attn.k_proj.weight");
      D.v = dfl_w(p + "self_attn.v_proj.weight");
      D.o = dfl_w(p + "self_attn.o_proj.weight");
      D.gate = dfl_w(p + "mlp.gate_proj.weight");
      D.up = dfl_w(p + "mlp.up_proj.weight");
      D.down = dfl_w(p + "mlp.down_proj.weight");
      D.akp = dfl_w(p + "attention_conv.kernel_projection.weight");
      D.mkp = dfl_w(p + "mlp_conv.kernel_projection.weight");
      D.in_norm = dfl_vec(p + "input_layernorm.weight");
      D.post_norm = dfl_vec(p + "post_attention_layernorm.weight");
      D.qn = dfl_vec(p + "self_attn.q_norm.weight");
      D.kn = dfl_vec(p + "self_attn.k_norm.weight");
      D.abase = dfl_vec(p + "attention_conv.base_kernel");
      D.mbase = dfl_vec(p + "mlp_conv.base_kernel");
      const size_t rb = (size_t)dfl_R_ * dfl_nkv_ * 128 * 4;
      D.rk = buf(rb);
      D.rv = buf(rb);
      fill(D.rk, rb);
      fill(D.rv, rb);
      if (D.q.N != dfl_nh_ * 128 || D.k.N != dfl_nkv_ * 128 || D.akp.N != 2 * dfl_K_ * (E / dfl_G_) || D.gate.N != dfl_F_)
        throw std::runtime_error("DFlash: unexpected layer shape");
      clFinish(cl_.queue);
    }
    has_mtp = dfl_ = true;
    fprintf(stderr, "DFlash2 drafter: %d layers, block %d, %zu target layers, %s\n", nl, dfl_B_, dfl_tl_.size(), path.c_str());
  }
  void dfl_buffers() {
    const int E = hp.n_embd, M = max_chunk;
    const size_t R = 16;  // block rows (padded)
    const int nv = (dfl_win_ + M - 1) / M + 1;  // chunks overlapping the last dfl_win_ positions of a prompt
    for (int i = 0; i < nv; ++i) dfl_acc_.push_back(buf((size_t)M * E * 4));
    dfl_x_ = buf(R * E * 4);
    dfl_h_ = buf(R * E * 4);
    dfl_t_ = buf(R * E * 4);
    dfl_dyn_ = buf(R * 2 * dfl_K_ * (E / dfl_G_) * 4);
    dfl_q_ = buf(R * dfl_nh_ * 128 * 4);
    dfl_o_ = buf(R * dfl_nh_ * 128 * 4);
    dfl_k_ = buf(R * dfl_nkv_ * 128 * 4);
    dfl_v_ = buf(R * dfl_nkv_ * 128 * 4);
    dfl_gu_ = buf(R * 2 * dfl_F_ * 4);
    dfl_hp_ = buf(R * dfl_rank_ * 4);
    dfl_ck_ = buf((size_t)M * dfl_nkv_ * 128 * 4);
    dfl_cv_ = buf((size_t)M * dfl_nkv_ * 128 * 4);
    dfl_lg_.resize((size_t)(dfl_B_ - 1) * hp.n_vocab);
    dfl_hph_.resize((size_t)(dfl_B_ - 1) * dfl_rank_);
  }
  // the positions whose features this pass keeps: the last dfl_win_ of [p0, p0 + n)
  void dfl_window(int p0, int n) {
    dfl_cap_lo_ = std::max(p0, p0 + n - dfl_win_);
    dfl_cap_base_ = p0 + (dfl_cap_lo_ - p0) / max_chunk * max_chunk;
    dfl_cap_end_ = p0 + n;
  }
  // after a target layer: its rows' contribution to fc (the chunk at `pos`, when it overlaps the window)
  void dfl_capture(Layer& L, int T, int Mp) {
    const auto it = std::find(dfl_tl_.begin(), dfl_tl_.end(), L.il);
    if (it == dfl_tl_.end() || pos + T <= dfl_cap_lo_ || pos < dfl_cap_base_) return;
    const int v = (pos - dfl_cap_base_) / max_chunk, li = (int)(it - dfl_tl_.begin());
    if (v >= (int)dfl_acc_.size()) return;
    const int a32 = act32_for(false);
    quant_rows(x_, hp.n_embd, Mp, T, a32);
    matmul("dfl_fc", dfl_fc_[li], Mp, dfl_acc_[v], hp.n_embd, 0, li > 0 ? 1 : 0, a32);
  }
  // the context K / V of rows r0 .. r0 + n of accumulator v (positions p0 ..) into every drafter layer's ring
  void dfl_commit(int v, int r0, int n, int p0, cl_mem src = nullptr) {
    if (n <= 0) return;
    if (!src) src = dfl_acc_[v];
    const int saved = cur_T_, E = hp.n_embd, KW = dfl_nkv_ * 128;
    for (int c0 = 0; c0 < n; c0 += max_chunk) {
      const int T = std::min(max_chunk, n - c0), Mp = T >= 128 ? std::min((T + 127) / 128 * 128, max_chunk) : (T + 15) / 16 * 16;
      cur_T_ = T;
      const int a32 = act32_for(false);
      {
        float eps = dfl_eps_;
        int rr = r0 + c0, EE = E, MM = Mp, TT = T, wh = 0, aa = a32;
        set_args(k_rmsnorm_, 0, src, rr, dfl_hnorm_, EE, MM, TT, eps, aq_, asc_, absum_, big_, wh, aa);
        size_t gws[2] = {256, (size_t)Mp}, lws[2] = {256, 1};
        run("dfl_ctx", k_rmsnorm_, 2, gws, lws);
      }
      float eps = dfl_eps_, ts = std::pow(dfl_theta_, -2.0f / 128);
      int pp = p0 + c0, kw = KW, R = dfl_R_;
      for (auto& D : dfl_l_) {
        matmul("dfl_ctx", D.k, Mp, dfl_ck_, KW, 0, 0, a32);
        matmul("dfl_ctx", D.v, Mp, dfl_cv_, KW, 0, 0, a32);
        {
          cl_kernel k = kernel("ops", "", "dfl_norm_rope");
          set_args(k, 0, dfl_ck_, kw, D.kn, eps, pp, ts);
          size_t gws[2] = {(size_t)16 * dfl_nkv_, (size_t)T}, lws[2] = {16, 1};
          run("dfl_ctx", k, 2, gws, lws);
        }
        {
          cl_kernel k = kernel("ops", "", "dfl_kv_store");
          set_args(k, 0, dfl_ck_, dfl_cv_, kw, pp, R, D.rk, D.rv);
          size_t gws[2] = {(size_t)KW, (size_t)T};
          run("dfl_ctx", k, 2, gws, nullptr);
        }
      }
    }
    cur_T_ = saved;
  }
  // after a prompt: the context of its kept positions (the chunks of the window)
  void dfl_commit_prompt(int p0) {
    (void)p0;
    for (int v = 0; v < (int)dfl_acc_.size(); ++v) {
      const int c0 = dfl_cap_base_ + v * max_chunk, a = std::max(c0, dfl_cap_lo_), b = std::min(c0 + max_chunk, dfl_cap_end_);
      if (b > a) dfl_commit(v, a - c0, b - a, a);
    }
    dfl_cap_lo_ = dfl_cap_end_ = 0;
  }
  void dfl_conv(cl_mem x, cl_mem base, int half_, cl_mem out, int rows) {
    cl_kernel k = kernel("ops", "", "dfl_conv");
    int ldd = 2 * dfl_K_ * (hp.n_embd / dfl_G_), h = half_, E = hp.n_embd, G = dfl_G_, K = dfl_K_;
    set_args(k, 0, x, dfl_dyn_, ldd, h, base, E, G, K, out);
    size_t gws[2] = {(size_t)E, (size_t)rows};
    run("dfl", k, 2, gws, nullptr);
  }
  void dfl_norm(cl_mem x, int r0, cl_mem w, int rows, cl_mem hout) {
    float eps = dfl_eps_;
    int rr = r0, E = hp.n_embd, M = 16, TT = rows, wh = hout ? 1 : 0, a32 = 1;
    set_args(k_rmsnorm_, 0, x, rr, w, E, M, TT, eps, aq_, asc_, absum_, hout ? hout : big_, wh, a32);
    size_t gws[2] = {256, (size_t)rows}, lws[2] = {256, 1};
    run("dfl", k_rmsnorm_, 2, gws, lws);
  }
  void dfl_add(cl_mem x, cl_mem y, int rows) {
    cl_kernel k = kernel("ops", "", "dfl_add");
    int E = hp.n_embd;
    set_args(k, 0, x, y, E);
    size_t gws[2] = {(size_t)E, (size_t)rows};
    run("dfl", k, 2, gws, nullptr);
  }
  // With both drafters: DFlash2 while its drafts are taken (recent accepted drafts per pass >= ARC_DFL_MIN, default 2),
  // else the MTP block, with a DFlash2 pass after 8 steps to see whether that changed (code: DFlash2 takes ~6 of 7
  // drafts; Indonesian prose ~1, where the MTP block does better). Dense models: each probe that does not bring
  // DFlash2 back doubles the wait, up to ARC_DFL_GAPMAX (32) steps, as their 8-row DFlash2 pass costs more than an MTP
  // pass (27B, Indonesian prose: 6.2 -> 6.5 tok/s; runs of 8 / 32 / 64 / no probes 5.6-6.3 / 6.55 / 6.4-6.5 /
  // 6.5-6.6); a probe that does resets it to 8. MoE models probe every 8 steps (35B: 22.5 vs 21.4-21.9 with 32).
  bool dfl_used_ = false, dfl_probe_ = false;
  double dfl_ema_ = 4.0;
  int dfl_wait_ = 0, dfl_gap_ = 8;
  static double dfl_min_acc() {
    static const double m = getenv("ARC_DFL_MIN") ? atof(getenv("ARC_DFL_MIN")) : 2.0;
    return m;
  }
  bool dfl_pick() {
    static const int gap_env = getenv("ARC_DFL_GAPMAX") ? atoi(getenv("ARC_DFL_GAPMAX")) : 0;
    const int gap_max = gap_env > 0 ? gap_env : hp.moe ? 8 : 32;
    if (dfl_ema_ >= dfl_min_acc()) return true;
    if (++dfl_wait_ >= std::min(dfl_gap_, gap_max)) {
      dfl_wait_ = 0;
      dfl_probe_ = true;
      return true;
    }
    return false;
  }
 public:
  int drafter_dflash_steps() const { return (int)dfl_steps_; }
  bool last_draft_dflash() const { return dfl_used_; }  // the drafts of the last mtp_draft came from DFlash2
 private:
  // sum of exp(l[i] - mx) over n logits (mx their maximum): the drafts' confidence. float exp in blocks of 4096 (the
  // compiler vectorizes it), the blocks summed in double; was one double exp per vocabulary entry, 2-3 ms per draft
  static double softmax_den(const float* l, size_t n, float mx) {
    double z = 0;
    for (size_t i0 = 0; i0 < n; i0 += 4096) {
      const size_t i1 = std::min(n, i0 + 4096);
      float zs = 0.0f;
      for (size_t i = i0; i < i1; ++i) zs += std::exp(l[i] - mx);
      z += zs;
    }
    return z;
  }
  std::vector<int> dflash_draft(int tok, int max_drafts) {
    const int E = hp.n_embd, B = dfl_B_, saved = cur_T_, start = pos, V = hp.n_vocab, F = dfl_F_;
    const int QW = dfl_nh_ * 128, KW = dfl_nkv_ * 128, ldd = 2 * dfl_K_ * (E / dfl_G_);
    cur_T_ = B;
    {
      const size_t rb = tok_embd_->nbytes / hp.n_vocab;
      for (int r = 0; r < B; ++r) {
        const int t = r == 0 ? tok : dfl_mask_;
        dequant_row(tok_embd_->type, tok_embd_->data + (size_t)t * rb, E, &x_host_[(size_t)r * E]);
      }
      CL_CHECK(clEnqueueWriteBuffer(cl_.queue, dfl_x_, CL_TRUE, 0, (size_t)B * E * 4, x_host_.data(), 0, nullptr, nullptr));
    }
    float eps = dfl_eps_, ts = std::pow(dfl_theta_, -2.0f / 128);
    for (auto& D : dfl_l_) {
      // attention, wrapped in the conv: prepare (input), finish (output) with the second half of its dynamic kernel
      dfl_norm(dfl_x_, 0, D.in_norm, B, dfl_h_);
      matmul("dfl", D.akp, 16, dfl_dyn_, ldd, 0, 0, true);
      dfl_conv(dfl_h_, D.abase, 0, dfl_t_, B);
      quant_rows(dfl_t_, E, 16, B, 1);
      matmul("dfl", D.q, 16, dfl_q_, QW, 0, 0, true);
      matmul("dfl", D.k, 16, dfl_k_, KW, 0, 0, true);
      matmul("dfl", D.v, 16, dfl_v_, KW, 0, 0, true);
      {
        cl_kernel k = kernel("ops", "", "dfl_norm_rope");
        int ld = QW, p0 = start;
        set_args(k, 0, dfl_q_, ld, D.qn, eps, p0, ts);
        size_t gws[2] = {(size_t)16 * dfl_nh_, (size_t)B}, lws[2] = {16, 1};
        run("dfl", k, 2, gws, lws);
        int ldk = KW;
        set_args(k, 0, dfl_k_, ldk, D.kn, eps, p0, ts);
        size_t g2[2] = {(size_t)16 * dfl_nkv_, (size_t)B};
        run("dfl", k, 2, g2, lws);
      }
      {
        cl_kernel k = kernel("ops", "", "dfl_attn");
        int R = dfl_R_, st = start, rows = B, win = dfl_win_, nh = dfl_nh_, nkv = dfl_nkv_;
        if (win - 1 + rows > 4096 + 64)  // (dfl_attn keeps up to win - 1 context + rows scores in local memory)
          throw std::runtime_error("dfl_attn: window " + std::to_string(win) + " + " + std::to_string(rows) + " rows exceed its 4160 scores");
        set_args(k, 0, dfl_q_, dfl_k_, dfl_v_, D.rk, D.rv, R, st, rows, win, nh, nkv, dfl_o_);
        size_t gws[2] = {128, (size_t)B * dfl_nh_}, lws[2] = {128, 1};
        run("dfl", k, 2, gws, lws);
      }
      quant_rows(dfl_o_, QW, 16, B, 1);
      matmul("dfl", D.o, 16, dfl_t_, E, 0, 0, true);
      dfl_conv(dfl_t_, D.abase, 1, dfl_h_, B);
      dfl_add(dfl_x_, dfl_h_, B);
      // MLP, wrapped the same way
      dfl_norm(dfl_x_, 0, D.post_norm, B, dfl_h_);
      matmul("dfl", D.mkp, 16, dfl_dyn_, ldd, 0, 0, true);
      dfl_conv(dfl_h_, D.mbase, 0, dfl_t_, B);
      quant_rows(dfl_t_, E, 16, B, 1);
      matmul("dfl", D.gate, 16, dfl_gu_, 2 * F, 0, 0, true);
      matmul("dfl", D.up, 16, dfl_gu_, 2 * F, F, 0, true);
      {
        int FF = F, MM = 16, TT = B, a32 = 1, fused = 0, gelu = 0;
        set_args(k_swiglu_, 0, dfl_gu_, FF, MM, TT, aq_, asc_, absum_, a32, fused, gelu);
        run_rows_quant("dfl", F, B, 16);
      }
      matmul("dfl", D.down, 16, dfl_t_, E, 0, 0, true);
      dfl_conv(dfl_t_, D.mbase, 1, dfl_h_, B);
      dfl_add(dfl_x_, dfl_h_, B);
    }
    // the drafts' rows 1 .. B-1: final norm, the target's head, the selector's projection
    const int nd = B - 1;
    cur_T_ = nd;
    dfl_norm(dfl_x_, 1, dfl_norm_, nd, nullptr);
    matmul("dfl_head", lm_head_, 16, logits_, V, 0, 0, true);
    matmul("dfl", dfl_hproj_, 16, dfl_hp_, dfl_rank_, 0, 0, true);
    CL_CHECK(clEnqueueReadBuffer(cl_.queue, logits_, CL_FALSE, 0, (size_t)nd * V * 4, dfl_lg_.data(), 0, nullptr, nullptr));
    CL_CHECK(clEnqueueReadBuffer(cl_.queue, dfl_hp_, CL_TRUE, 0, (size_t)nd * dfl_rank_ * 4, dfl_hph_.data(), 0, nullptr, nullptr));
    cur_T_ = saved;
    // candidate selector: per position the top-k logits; score = logit + <pred[prev] * h, succ[cand]>, greedy path
    std::vector<int> out;
    int prev = tok;
    const int rk = dfl_rank_, K = std::min(dfl_topk_, V);
    std::vector<int> idx(V);
    // (ARC_DFL_SEL_OLD=1: nth_element over the whole vocabulary per position, ~2 ms each on the host: the one-pass
    //  top-K below made code answers 35B 48.4 -> 52.7, 27B 16.9 -> 18.3 tok/s with the same drafts)
    static const bool sel_old = getenv("ARC_DFL_SEL_OLD") != nullptr;
    // drafts are kept while the drafter's own probability of the chosen path stays >= ARC_DFL_P (each draft verified
    // costs more rows in the main model's pass: an unlikely tail is cheaper not to check)
    static const double min_p = getenv("ARC_DFL_P") ? atof(getenv("ARC_DFL_P")) : 0.3;  // (code 11.4 -> 11.6 tok/s, prose up)
    double path_p = 1.0;
    for (int t = 0; t < std::min(nd, max_drafts); ++t) {
      const float* l = &dfl_lg_[(size_t)t * V];
      if (sel_old) {
        for (int i = 0; i < V; ++i) idx[i] = i;
        std::nth_element(idx.begin(), idx.begin() + K, idx.end(), [&](int a, int b) { return l[a] > l[b]; });
      } else {
        // the K largest logits in one pass (kept sorted, largest first; almost every logit fails the first test)
        int n = 0;
        for (int i = 0; i < V; ++i) {
          const float v = l[i];
          if (n == K && v <= l[idx[K - 1]]) continue;
          int j = n < K ? n++ : K - 1;
          while (j > 0 && l[idx[j - 1]] < v) { idx[j] = idx[j - 1]; --j; }
          idx[j] = i;
        }
      }
      const float* h = &dfl_hph_[(size_t)t * rk];
      float ph[1024];
      for (int r = 0; r < rk; ++r) ph[r] = bf16_to_f32(dfl_pred_[(size_t)prev * rk + r]) * h[r];
      double best = -1e300;
      int bt = idx[0];
      for (int k = 0; k < K; ++k) {
        const int c = idx[k];
        double sc = l[c];
        const uint16_t* sv = dfl_succ_ + (size_t)c * rk;
        for (int r = 0; r < rk; ++r) sc += ph[r] * bf16_to_f32(sv[r]);
        if (sc > best) { best = sc; bt = c; }
      }
      if (min_p > 0) {
        float mx = l[idx[0]];
        for (int k = 1; k < K; ++k) mx = std::max(mx, l[idx[k]]);
        const double z = softmax_den(l, V, mx);
        path_p *= std::exp((double)l[bt] - mx) / z;
        if (t > 0 && path_p < min_p) break;  // (the first draft always goes)
      }
      out.push_back(bt);
      prev = bt;
    }
    ++dfl_steps_;
    collect_profile();
    return out;
  }

  // ---------------------------------------------------------------- MTP block
  // Runs the nextn block over n rows at positions p0..: inputs [enorm(embedding(toks[r])) | hnorm(h row r)] with h
  // rows from `hsrc` (main-model hidden states after the output norm, one position earlier), then eh_proj, one
  // gated-attention layer and the FFN (own KV cache). With `logits`, the shared head on the last row -> logits_.
  void mtp_rows(const int* toks, cl_mem hsrc, int n, int p0, bool logits, cl_mem hout = nullptr) {
    const int E = hp.n_embd;
    const int Mp = n >= 128 ? std::min((n + 127) / 128 * 128, max_chunk) : (n + 15) / 16 * 16;
    const int saved_T = cur_T_, saved_pos = pos;
    cl_mem saved_x = x_;
    cur_T_ = n;
    const size_t row_bytes = tok_embd_->nbytes / hp.n_vocab;
#pragma omp parallel for if (n > 64)
    for (int t = 0; t < n; ++t) {
      if (toks[t] < 0 || toks[t] >= hp.n_vocab) {
        std::fill_n(&x_host_[(size_t)t * E], E, 0.0f);
        continue;
      }
      dequant_row(tok_embd_->type, tok_embd_->data + (size_t)toks[t] * row_bytes, E, &x_host_[(size_t)t * E]);
    }
    CL_CHECK(clEnqueueWriteBuffer(cl_.queue, mtp_e_, CL_TRUE, 0, (size_t)n * E * 4, x_host_.data(), 0, nullptr, nullptr));
    {
      float eps = hp.eps;
      int EE = E, F2 = 2 * E, MM = Mp, TT = n, a32 = 1, fused = 1, gelu = 0;
      set_args(k_mtp_concat_, 0, mtp_e_, hsrc, enorm_, hnorm_, EE, eps, big_);
      size_t gws[2] = {256, (size_t)n}, lws[2] = {256, 1};
      run("mtp_concat", k_mtp_concat_, 2, gws, lws);
      set_args(k_swiglu_, 0, big_, F2, MM, TT, aq_, asc_, absum_, a32, fused, gelu);  // quantization only
      size_t g2[2] = {256, (size_t)Mp};
      run("mtp_concat", k_swiglu_, 2, g2, lws);
    }
    x_ = mtp_x_;
    pos = p0;
    matmul("mtp_eh_proj", eh_proj_, Mp, x_, E, 0, 0, true);
    layer_attn(mtp_, n, Mp);
    if (hp.moe) layer_moe(mtp_, n, Mp);
    else layer_ffn(mtp_, n, Mp);
    if (logits) {
      if (draft_ids_.empty()) {
        head_rows(n - 1, 1, shnorm_, hout);
      } else {  // (the draft head: the listed rows only, compact logits)
        const int saved = cur_T_;
        cur_T_ = 1;
        float eps = hp.eps;
        int E2 = hp.n_embd, M = 16, nn = 1, wh = hout ? 1 : 0, a32 = 1, r0 = n - 1;
        set_args(k_rmsnorm_, 0, x_, r0, shnorm_, E2, M, nn, eps, aq_, asc_, absum_, hout ? hout : big_, wh, a32);
        size_t gws[2] = {256, 1}, lws[2] = {256, 1};
        run("rmsnorm_quant", k_rmsnorm_, 2, gws, lws);
        const GpuWeight& w = lm_head_;
        const bool vk = gemv_v_ok(w);
        const int grows = vk ? gemv_v_rows(w) : 4;
        cl_kernel k = kernel("decode", "-DQT=" + std::to_string(w.type) + " -DGROWS=" + std::to_string(grows) + " -DROWIDX" +
                                           head_defs(w), vk ? "gemv_v" : "gemv");
        int NN = (int)draft_ids_.size(), K = w.K, MM = 16, z = 0;
        rd_bytes += raw_bytes(w) / w.N * NN;
        set_args(k, 0, w.raw, NN, K, tabs_, aq_, asc_, absum_, MM, logits_, z, z, draft_idx_);
        size_t g1 = ((size_t)(NN + grows - 1) / grows * 16 + 255) / 256 * 256, l1 = 256;
        run("lm_head_draft", k, 1, &g1, &l1);
        cur_T_ = saved;
      }
    }
    x_ = saved_x;
    pos = saved_pos;
    cur_T_ = saved_T;
  }
  // Keeps the MTP KV cache in step with a main-model chunk at positions p0..p0+T-1 (x_ holds its final residual
  // stream): row r pairs token r with the hidden state of position p0+r-1. Leaves hlast_ = hidden of the last row.
  void mtp_prefill(const int* toks, int T, int Mp, int p0) {
    const size_t E4 = (size_t)hp.n_embd * 4;
    CL_CHECK(clEnqueueCopyBuffer(cl_.queue, hlast_, hfin_, 0, 0, E4, 0, nullptr, nullptr));
    {
      float eps = hp.eps;
      int E = hp.n_embd, M = Mp, TT = T, r0 = 0, wh = 1, a32 = 1;
      set_args(k_rmsnorm_, 0, x_, r0, out_norm_, E, M, TT, eps, aq_, asc_, absum_, hfin1_, wh, a32);
      size_t gws[2] = {256, (size_t)T}, lws[2] = {256, 1};
      run("mtp_concat", k_rmsnorm_, 2, gws, lws);
    }
    CL_CHECK(clEnqueueCopyBuffer(cl_.queue, hfin_, hlast_, (size_t)T * E4, 0, E4, 0, nullptr, nullptr));
    mtp_rows(toks, hfin_, T, p0, false);
  }
  // The rows of accepted drafts (their MTP KV entries, now with the main model's hidden states) are processed
  // lazily, together with the next draft.
  void flush_mtp_pending() {
    if (pend_toks_.empty()) return;
    const std::vector<int> t = pend_toks_;
    pend_toks_.clear();
    mtp_rows(t.data(), hpend_, (int)t.size(), pos - (int)t.size(), false);
  }

 public:
  // ---------------------------------------------------------------- MTP speculative decoding (has_mtp)
  // Drafts per verification pass: Qwen's DeltaNet keeps state copies after tokens 0 and 1 (two drafts); Gemma has
  // no recurrent state, a rejected draft only moves the position back.
  // (Qwen: one DeltaNet state copy per draft, allocated for --mtp-drafts; Gemma's drafter keeps no state)
  static constexpr int kMaxDrafts = 7;  // (Qwen3.8's MTP block is trained for 7)
  // default MTP drafts per step (later ones chained while the block is confident, opt.mtp_min_p). With the
  // token-outer verification GEMVs (16 Oct night), code answers: 27B 3 -> 7 drafts 12.6 -> 14.7 tok/s, 35B 3 -> 5
  // 34.2 -> 38.6; Indonesian prose equal (27B 7.0, 35B 23.4 / 23.1). 9B 2, 4B 3 (not re-measured); Gemma's drafter 2.
  int mtp_block_drafts() const { return hp.gemma ? (hp.moe ? 3 : 2) : hp.moe ? 5 : hp.n_embd >= 5120 ? 7 : hp.n_embd <= 2560 ? 3 : 2; }
  int max_drafts() const { return gemma_mtp_ ? 3 : std::clamp(opt.mtp_drafts, 2, kMaxDrafts); }
  // `tok` is the next token (sampled, not yet processed; it goes to position pos). Returns up to `max_drafts` greedy
  // drafts of the MTP block for the tokens after it: the second one is chained on the block's own output and only
  // made when the first has probability >= opt.mtp_min_p.
  std::vector<int> mtp_draft(int tok, int max_drafts) {
    dfl_used_ = dfl_ && (!mtp_block_ || dfl_pick());
    if (dfl_used_) {
      if (mtp_block_) {  // the MTP block's KV row of `tok` too (and of the pending accepted drafts), as its own drafts
        // would write it: the block attends to every position once it drafts again (without the row: stale memory)
        const size_t E4 = (size_t)hp.n_embd * 4;
        std::vector<int> toks = pend_toks_;
        const int np = (int)toks.size();
        if (np) CL_CHECK(clEnqueueCopyBuffer(cl_.queue, hpend_, hfin_, 0, 0, np * E4, 0, nullptr, nullptr));
        CL_CHECK(clEnqueueCopyBuffer(cl_.queue, hlast_, hfin_, 0, np * E4, E4, 0, nullptr, nullptr));
        toks.push_back(tok);
        pend_toks_.clear();
        mtp_rows(toks.data(), hfin_, np + 1, pos - np, false);
      }
      max_drafts = std::clamp(max_drafts, 1, std::clamp(opt.mtp_drafts, 1, std::min(kMaxDrafts, dfl_B_ - 1)));
      if (pos + 1 + max_drafts > ctx) max_drafts = std::max(1, ctx - pos - 1);
      return dflash_draft(tok, max_drafts);
    }
    if (gemma_mtp_) {
      max_drafts = std::clamp(max_drafts, 1, std::clamp(opt.mtp_drafts, 1, this->max_drafts()));
      if (pos + 1 + max_drafts > ctx) max_drafts = std::max(1, ctx - pos - 1);
      return gemma_draft(tok, max_drafts);
    }
    const size_t E4 = (size_t)hp.n_embd * 4;
    std::vector<int> toks = pend_toks_;
    const int np = (int)toks.size();
    if (np) CL_CHECK(clEnqueueCopyBuffer(cl_.queue, hpend_, hfin_, 0, 0, np * E4, 0, nullptr, nullptr));
    CL_CHECK(clEnqueueCopyBuffer(cl_.queue, hlast_, hfin_, 0, np * E4, E4, 0, nullptr, nullptr));
    toks.push_back(tok);
    pend_toks_.clear();
    max_drafts = std::clamp(max_drafts, 1, std::clamp(opt.mtp_drafts, 1, kMaxDrafts));
    if (dfl_) max_drafts = std::min(max_drafts, mtp_block_drafts());  // (with DFlash2 the drafts setting is its 7)
    if (pos + 1 + max_drafts > ctx) max_drafts = std::max(1, ctx - pos - 1);
    std::vector<int> drafts;
    mtp_rows(toks.data(), hfin_, np + 1, pos - np, true, max_drafts > 1 ? mtp_h_ : nullptr);
    // (the draft head's logits: compact, row i is token draft_ids_[i]; the confidence is taken over those rows)
    const size_t nl = draft_ids_.empty() ? (size_t)hp.n_vocab : draft_ids_.size();
    for (int k = 0;; ++k) {
      CL_CHECK(clEnqueueReadBuffer(cl_.queue, logits_, CL_TRUE, 0, nl * 4, mtp_logits_host_.data(), 0, nullptr, nullptr));
      const auto first = mtp_logits_host_.begin(), last = first + nl;
      const auto mx = std::max_element(first, last);
      const int d = draft_ids_.empty() ? (int)(mx - first) : draft_ids_[mx - first];
      drafts.push_back(d);
      if ((int)drafts.size() >= max_drafts) break;
      const double z = softmax_den(&*first, nl, *mx);
      if (1.0 / z < opt.mtp_min_p) break;  // not confident: verify one draft only
      mtp_rows(&d, mtp_h_, 1, pos + 1 + k, true, mtp_h_);
    }
    collect_profile();
    return drafts;
  }
  // Main-model pass over [tok, drafts...] at positions pos, pos+1, ... Returns the logits of every row (row i
  // predicts the token after row i's token). The DeltaNet states after each draft-free prefix are kept so that
  // verify_done() can drop rejected drafts.
  const std::vector<float>& verify(int tok, const std::vector<int>& drafts) {
    const int T = 1 + (int)drafts.size();
    if (drafts.empty() || (int)drafts.size() > max_drafts()) throw std::runtime_error("verify: too many drafts");
    if (pos + T > ctx) throw std::runtime_error("context overflow");
    flush_mtp_pending();
    std::vector<int> toks{tok};
    toks.insert(toks.end(), drafts.begin(), drafts.end());
    last_drafts_ = drafts;
    vrows_ = false;
    embed(toks.data(), T, 16);
    if (dfl_) dfl_window(pos, T);
    spec_ = !gemma_mtp_;
    for (auto& L : layers_) run_layer(L, T, 16);
    fail_point();
    spec_ = false;
    pos += T;
    head_rows(0, T, out_norm_, hver_);
    CL_CHECK(clEnqueueReadBuffer(cl_.queue, logits_, CL_TRUE, 0, (size_t)hp.n_vocab * T * 4, spec_logits_host_.data(),
                                 0, nullptr, nullptr));
    if (pos > written_hi_) written_hi_ = pos;
    collect_profile();
    return spec_logits_host_;
  }
  // The first `accepted` drafts of the last verify() turned out right; the rest are dropped.
  void verify_done(int accepted) {
    const size_t E4 = (size_t)hp.n_embd * 4;
    const cl_mem dfl_src = vrows_ ? dfl_vrows_ : nullptr;  // (a batched pass: the slot's own copy of its rows)
    vrows_ = false;
    const int nd = (int)last_drafts_.size();
    accepted = std::clamp(accepted, 0, nd);
    if (!gemma_mtp_)
      for (auto& L : layers_) {
        if (!hp.recurrent(L.il)) continue;
        if (accepted < nd) {
          // the states after token `accepted`: the same recurrence over the pass's first accepted + 1 tokens, from the
          // states before it (same kernel, same inputs: the values the pass computed there)
          const int C = hp.conv_dim(), nk = hp.ssm_nk, nv = hp.ssm_nv, rows = accepted + 1;
          set_args(k_gdn_recur_, 0, L.xin, C, L.abin, nk, nv, L.S, rows, o_, L.S_next);
          size_t g1 = (size_t)nv * 128, l1 = 64;
          run("gdn_recur", k_gdn_recur_, 1, &g1, &l1);
          CL_CHECK(clEnqueueCopyBuffer(cl_.queue, L.conv_pre, L.conv_state, 0, 0, (size_t)3 * C * 4, 0, nullptr, nullptr));
          size_t g2 = ((size_t)C + 255) / 256 * 256, l2 = 256;
          set_args(k_conv_upd_, 0, L.qkvin, L.conv_state, C, rows);
          run("conv_state_update", k_conv_upd_, 1, &g2, &l2);
        }
        std::swap(L.S, L.S_next);
      }
    if (accepted < nd) pos -= nd - accepted;
    CL_CHECK(clEnqueueCopyBuffer(cl_.queue, hver_, hlast_, accepted * E4, 0, E4, 0, nullptr, nullptr));
    if (gemma_mtp_) {  // the drafter keeps no state of its own
      last_drafts_.clear();
      return;
    }
    if (dfl_) {  // DFlash: the context of the token and the accepted drafts (rows 0 .. accepted of the pass)
      dfl_commit(0, 0, accepted + 1, pos - accepted - 1, dfl_src);
      if (dfl_used_) dfl_ema_ = 0.75 * dfl_ema_ + 0.25 * accepted;
      if (dfl_used_ && dfl_probe_) {
        dfl_gap_ = dfl_ema_ >= dfl_min_acc() ? 8 : std::min(2 * dfl_gap_, 1 << 16);
        dfl_probe_ = false;
      }
      if (!mtp_block_) {
        last_drafts_.clear();
        return;
      }
    }
    // accepted draft i sits at position pos - accepted + i; its MTP row pairs it with the hidden state of the row before
    pend_toks_.assign(last_drafts_.begin(), last_drafts_.begin() + accepted);
    if (accepted) CL_CHECK(clEnqueueCopyBuffer(cl_.queue, hver_, hpend_, 0, 0, accepted * E4, 0, nullptr, nullptr));
    last_drafts_.clear();
  }

 private:
};
