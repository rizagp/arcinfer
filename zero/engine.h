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
  // automatic: what is left after the rest of the model within the GPU's memory share less kDriverSlack.
  bool stream_experts = true;
  double expert_cache_gb = 0, expert_cache_pct = 0;
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
    // (Gemma 4's two-pass mode: 2048 for streamed experts too. Its float K / V rings of the sliding layers hold the
    //  window + a chunk + 4096 rows, 4.1 GB at 8192 on the 26B, which left the expert cache at 39 %. 26B, ctx 16K,
    //  chunk 8192 / 4096 / 2048: expert cache 39 / 64 / 77 %, 4K prompt 216 / 184 / 217 tok/s, generation after a
    //  14K prompt 9.9 / 12.3 / 13.0 tok/s)
    max_chunk = o.chunk > 0 ? o.chunk : stream_ ? (hp.hc || hp.gemma ? 2048 : 8192) : hp.moe ? 2048 : (w8_resident ? 512 : 4096);
    max_chunk = (std::max(max_chunk, o.min_chunk) + 127) / 128 * 128;
    q2_init();
    if (!opt.dry) build_kernels();
    if (!opt.dry && (cl_.xmx || getenv("ARC_XMX_TRY"))) {  // (ARC_XMX_TRY=1: the check on any GPU, tests)
      xmx_ = xmx_check();
      if (xmx_) fprintf(stderr, "arcinferzero: XMX on for the prompt GEMMs (results checked against DP4a)\n");
    }
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
      const double left = (double)device_mem() - kDriverSlack - (double)gpu_bytes + (double)ctx_bytes(ctx) -
                          (double)snapshot_bytes() - (double)opt.stream_reserve;
      // (--slots N: 128K of context in all, each slot 128K / N; the floor of 3/4 only as far as it leaves every slot
      //  32K: with a plain floor Gemma 4 26B-A4B kept 8,192 tokens per slot, its sliding-window rings ~1.4 GB each.
      //  One slot: as before)
      const int ns = std::max(opt.slots, 1), per = (131072 / ns + 4095) / 4096 * 4096;
      auto ctx_all = [&](int c) { return (double)ctx_bytes(c) + (double)(ns - 1) * (double)slot_bytes(c); };
      // (Gemma 4 26B-A4B: a floor of 0.6, 128K instead of 49K: generation at 36K of depth 11.4 tok/s with 89 % of the
      //  experts cached (49K), 10.5 with 70 % (131K), 8.8 with 39 % (262K); the user ran 131K by hand. 2026-10-23)
      const double fl = hp.gemma ? 0.6 : 0.75;
      const double room = std::max(left - ctx_all(per), std::min(fl * left, left - ctx_all(kSlotCtxFloor)));
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
    if ((!w8_resident || stream_) && n > max_chunk && !hp.n_ple) {
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
  bool warm_up_once(int tok, bool prompt = true) {
    const std::string& dir = cl_.cache_dir;
    char hex[17];
    snprintf(hex, sizeof hex, "%016llx",
             (unsigned long long)ClRt::fnv1a(gemm_src_ + "\x01" + ops_src_ + "\x01" + dec_src_ + "\x01" + moe_src_ + "\x01" + f_.path +
                                             "|" + std::to_string(opt.mtp) + "|" + opt.dflash + (opt.mtp_block ? "" : "|nomtp") +
                                             "|" + std::to_string(opt.mtp_drafts) +
                                             "|" + std::to_string(max_chunk) + "|" + cl_.device_id));
    const std::string marker = dir.empty() ? std::string() : dir + "\\warm-" + hex;
    // (every start (prompt = true): a prompt long enough for the large-chunk kernels (moe_rg from 320 rows), then a token: their first
    //  use in a process cost seconds even from the kernel cache; 35B with drafters, a first 1,749-token request 8.6 s
    //  (203 tok/s), the next ones 3.5 s)
    const int nwarm = std::max(1, std::min({prompt ? 512 : 32, max_chunk, ctx - 12}));
    if (!marker.empty() && std::filesystem::exists(marker)) {
      if (prompt) {
        std::vector<int> w(nwarm, tok);
        forward(w.data(), nwarm);
      }
      forward(&tok, 1);
      reset();
      return false;
    }
    std::vector<int> t(nwarm, tok);
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
  // Qwen KV bytes per position and kv head: K and V int8 + their scales
  size_t kv_pos_bytes() const {
    if (!opt.attn_int8) return 4 * (size_t)hp.head_dim;  // (--attn-f32: f16 K and V rows, load_attn's kc / vc)
    return 2 * (size_t)hp.head_dim + 8;
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
      const int swa_cap = std::min(c, (hp.n_swa + max_chunk + 4096 + 255) / 256 * 256);
      for (const Layer& L : layers_)
        if (L.kv_src < 0) {
          const size_t cap = L.swa ? swa_cap : c;
          if (L.i8) n += 2 * (size_t)L.nkv * cap * L.hd + 2 * (size_t)L.nkv * cap * (L.hd / 64) * 4;
          if (q2_) n += 6 * (size_t)L.nkv * cap * L.hd;  // (two-pass mode: attn_fa_g's K / V rows)
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
  static constexpr int kSlotCtxFloor = 32768;
  static constexpr double kDriverSlack = 1.2 * 1073741824.0;  // (GPU share bytes left to the driver; the user's choice)  // (--ctx auto, slots of streamed MoE models: context kept per slot)
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
      for (cl_mem* f : {&L.kq, &L.vq, &L.ksc, &L.vsc, &L.conv_state, &L.S, &L.pk, &L.S_next, &L.conv_pre,
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
           vq = nullptr, vsc = nullptr;
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
    bool i8 = true;             // Gemma 4: keeps the int8 K / V copy (only the layers the drafter reads)
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
    cl_mem ixsv = nullptr;  // (a verification pass: its indexer rows, for verify_done)
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
        for (int t = 0; t < (k.L->i8 ? 4 : 0); ++t) {  // (Gemma layers without the int8 copy: the float rows only)
          cl_mem b = t == 0 ? k.L->kq : t == 1 ? k.L->vq : t == 2 ? k.L->ksc : k.L->vsc;
          const size_t unit = t < 2 ? (size_t)16 * k.hd : (size_t)k.sb * 16 * 4;
          const size_t base = (size_t)g * nbc;
          if (first > 0) v.push_back({b, (base + s0) * unit, (size_t)first * unit});
          if (n > first) v.push_back({b, base * unit, (size_t)(n - first) * unit});
        }
      if (q2_)  // (two-pass mode: the float K / V rows attn_fa_g reads, [kv head][cap][hd], the same 16-row blocks)
        for (int g = 0; g < k.nkv; ++g)
          for (int t = 0; t < 2; ++t) {
            const size_t unit = (size_t)16 * k.hd * (t == 0 ? 4 : 2), base = (size_t)g * nbc;
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
  int state_flags() const { return (hp.gemma ? 1 : 0) | (q2_ ? 4 | 8 : 0) | (q2_ && !q2mm_ ? 16 : 0); }  // (2: the removed
  //  --v4; 8: int8 copies of the drafter's layers only; 16: Gemma's matmuls in one pass, E2B / E4B)
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
            k_attn_comb_ = nullptr;
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
  // Gemma 4: the layer whose KV cache layer il uses (its own, or llama.cpp's KV reuse rule)
  int g4_kv_layer(int il) const { return il >= hp.n_layer_kv ? hp.n_layer_kv - (hp.swa[il] ? 2 : 1) : il; }
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
  // (a verification pass on Flash Next: the PLE / QSA state before it, the pass's PLE conv rows and the token ids
  //  before it, so that verify_done() rebuilds the state after the accepted tokens)
  cl_mem hc_bak_ = nullptr, ple_sv_ = nullptr;
  int spec_prev_[2] = {-1, -1}, spec_tok_ = -1;
  const GgufTensor* tok_embd_ = nullptr;
  // activations
  cl_mem x_ = nullptr, aq_ = nullptr, asc_ = nullptr, absum_ = nullptr, big_ = nullptr, z_ = nullptr,
         conv_out_ = nullptr, gb_ = nullptr, o_ = nullptr, qrope_ = nullptr, logits_ = nullptr, qi_ = nullptr,
         qsc_ = nullptr, apart_ = nullptr;
  // just-in-time W8 staging, one slot per matrix role (attn_qkv, ffn_up, ...): a layer's matrices stay decoded
  // while several chunks go through it (layer-major prefill)
  static constexpr int kRoles = 16;
  cl_mem stage_q_[kRoles] = {}, stage_h_[kRoles] = {};
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
    if (!hp.gemma && hpg % 4 == 0 && hpg / 4 <= 32 && ((hpg / 4) & (hpg / 4 - 1)) == 0) return 4;
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
    auto p = cl_.build(src, "-cl-std=CL3.0 -cl-mad-enable " + defs + (which == "ops" ? q2_defs_ : std::string()) +
                            (cl_.sg8 ? "" : " -DNO_SG8"));
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
  cl_kernel gemm_kernel(int qtype, bool act32, int xmx = -1) {
    return kernel("gemm", "-cl-intel-256-GRF-per-thread -DQTYPE=" + std::to_string(qtype) + (act32 ? " -DACT32" : "") +
                              (gemm_tm() != 16 ? " -DTM=" + std::to_string(gemm_tm()) : std::string()) + gemm_extra() +
                              ((xmx < 0 ? xmx_ : xmx != 0) ? " -DXMX -DTP=8" : ""),
                  "gemm");
  }
  // XMX for the prompt GEMMs (gemm.cl -DXMX): taken only when the GPU offers 16-lane DPAS (ClRt::xmx) and the XMX
  // kernel gives the DP4a kernel's results bit for bit on random data of every weight layout (with and without
  // per-32 activation scales); a GPU or driver that differs keeps DP4a. Written without an XMX GPU at hand (this
  // laptop's Xe-LPG has none): the check is what guards it on the GPUs that have one.
  bool xmx_ = false;
  bool xmx_check() {
    if (gemm_tm() % 8) return false;
    const int M = 32, N = 32, K = 512, nsb = K / 256, ng = N / 16;
    uint32_t seed = 12345;
    auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return seed; };
    std::vector<uint32_t> wq((size_t)ng * nsb * 4 * 16 * 16), wh((size_t)ng * nsb * 10 * 16);
    for (auto& v : wq) v = rnd();
    for (auto& v : wh) {  // (floats in [0.001, 0.011): headers and scales of every layout read sensible values)
      const float f = 0.001f + (float)(rnd() % 10000) * 1e-6f;
      memcpy(&v, &f, 4);
    }
    std::vector<int8_t> aq((size_t)M * K);
    for (auto& v : aq) v = (int8_t)(rnd() >> 24);
    std::vector<float> asc((size_t)(K / 32) * M);
    for (auto& v : asc) v = 0.01f + (float)(rnd() % 1000) * 1e-5f;
    std::vector<uint16_t> ab((size_t)(K / 16) * M);
    for (auto& v : ab) v = (uint16_t)(int16_t)((int)(rnd() % 4001) - 2000);
    cl_mem bq = buf(wq.size() * 4, wq.data()), bh = buf(wh.size() * 4, wh.data()), ba = buf(aq.size(), aq.data()),
           bs = buf(asc.size() * 4, asc.data()), bb = buf(ab.size() * 2, ab.data()), bc = buf((size_t)M * N * 4);
    bool same = true;
    std::string bad;
    struct Soft { bool& f; ~Soft() { f = false; } } soft{cl_.soft_fail};
    cl_.soft_fail = true;
    try {
      for (int q : {101, 100, 103, 102, 8, 12, 13, 14})
        for (int a32 = 1; a32 >= 0; --a32) {
          std::vector<float> out[2];
          for (int x = 0; x < 2; ++x) {
            cl_kernel k = gemm_kernel(q, a32 != 0, x);
            int MM = M, NN = N, KK = K, ldc = N, coff = 0, acc = 0, rg = 4, s0 = 0, s1 = nsb;
            set_args(k, 0, bq, bh, ba, bs, bb, bc, MM, NN, KK, ldc, coff, acc, rg, s0, s1);
            size_t gws[2] = {(size_t)N * (M / gemm_tm() / 2), 2}, lws[2] = {16, 2};
            run("xmx_check", k, 2, gws, lws);
            out[x].resize((size_t)M * N);
            CL_CHECK(clEnqueueReadBuffer(cl_.queue, bc, CL_TRUE, 0, out[x].size() * 4, out[x].data(), 0, nullptr, nullptr));
          }
          bool sane = false;  // (the reference itself: finite and not all zero, or the comparison proves nothing)
          for (float v : out[0]) {
            if (!std::isfinite(v)) { sane = false; break; }
            sane = sane || v != 0.0f;
          }
          if (!sane || memcmp(out[0].data(), out[1].data(), out[0].size() * 4)) {
            same = false;
            bad += " " + std::to_string(q) + (a32 ? "/act32" : "");
          }
        }
    } catch (const std::exception& e) {
      fprintf(stderr, "arcinferzero: XMX kernels do not build here (%s): DP4a\n", e.what());
      return false;
    }
    if (!same) fprintf(stderr, "arcinferzero: XMX results differ from DP4a (layouts%s): DP4a (please report)\n", bad.c_str());
    return same;
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
    k_attn_prep_ = kernel("ops", sb_opt(256), "attn_prep");
    k_attn_gate_ = p("attn_gate_quant");
    k_gdn_gates_ = p("gdn_gates");
    const std::string big = "-cl-intel-256-GRF-per-thread -DGQ=" + std::to_string(gq());
    k_attn_ = kernel("ops", big, "attn_fa");
    k_gdn_recur_ = kernel("ops", big, "gdn_recur");
    const std::string att = "-cl-intel-256-GRF-per-thread -DHPG=" + std::to_string(hp.n_head / hp.n_head_kv) +
                            " -DHS=" + std::to_string(attn_hs()) + " -DTPW=" + std::to_string(attn_tpw()) + sb_opt(256);
    k_attn_q8_ = kernel("ops", att, "attn_q8");
    {
      const int hpg = hp.n_head / hp.n_head_kv, hs = attn_hs_for(hpg);
      k_attn_dec_ = kernel("ops", "-cl-intel-256-GRF-per-thread -DHPG=" + std::to_string(hpg) + " -DHS=" + std::to_string(hs) +
                                      " -DTPW=" + std::to_string(32 / (hpg / hs)) + sb_opt(256), "attn_decode");
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
    const bool quant = tq.type != GGML_F32 && tq.type != GGML_F16 && tq.type != GGML_BF16;
    if (!quant || tq.type != tz.type || tq.dims[0] != E || tz.dims[0] != E || E % 256 || C % 64 || Di % 64 ||
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
    // DFlash2 (with or without the MTP block: with both, mtp_draft picks one per step; Flash Next: the features are the
    // means of the residual streams, the PLE / QSA state rolled back in verify_done)
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
      // the whole process (weights, buffers, transient area, cache) within the GPU share Windows gives the iGPU, less
      // kDriverSlack for the driver: the share is the user's setting, the rest of the RAM what they leave to Windows
      // (the user's rule since 2026-10-22; before: also RAM - a 14 GB Windows reserve, which held a 20.4 GiB share to
      // 17.7). (With 1.13 GiB left a kernel launch once failed with out of device memory, 35B at 262K.)
      const uint64_t share = zrt::global_mem, slack = (uint64_t)kDriverSlack;
      const uint64_t limit = share > slack ? share - slack : 0;
      // (+ kernels and host side, the server's state snapshots, and what the caller keeps for its encoders)
      const uint64_t used = gpu_bytes + xc_->stage_bytes + (uint64_t)(0.1 * GiB) + snapshot_bytes() * std::max(opt.slots, 1) + opt.stream_reserve;
      want = limit > used ? limit - used : 0;
      fprintf(stderr, "arcinferzero: memory limit %.2f GiB (GPU share %.2f - %.1f for the driver), model + buffers %.2f GiB\n",
              limit / GiB, share / GiB, kDriverSlack / GiB, used / GiB);
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
    if (T == 1) {
      std::vector<int> tl(1 + 3 * NU);
      if (tl1)
        std::copy(tl1, tl1 + tl.size(), tl.begin());
      else
        CL_CHECK(clEnqueueReadBuffer(cl_.queue, mtiles_, CL_TRUE, 0, tl.size() * 4, tl.data(), 0, nullptr, nullptr));
      for (int k = 0; k < NU; ++k) ids[k] = tl[1 + 3 * k];
    } else {
      CL_CHECK(clEnqueueReadBuffer(cl_.queue, mids_, CL_TRUE, 0, ids.size() * 4, ids.data(), 0, nullptr, nullptr));
    }
    const auto h1 = hclk::now();
    hst_sync_ms_ += std::chrono::duration<double, std::milli>(h1 - h0).count();
    xc_->ensure(L.il, ids.data(), ids.size(), T > 8, gemm);
    hst_ensure_ms_ += hms(h1);
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
    const bool sl = !w8_requant(type) && nsb <= 16 && N % 16 == 0 &&
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
      L.ixsv = buf((size_t)(1 + kMaxDrafts) * L.ix.N * 4);
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
    // sliding-window layers keep a ring of the last n_swa + chunk (+ room to rewind an agent's reply) positions. The
    // layers that share another layer's KV (E2B / E4B) read its ring later in the same pass over the same key range,
    // so the ring is enough for them too (they kept full caches before 2026-10-22: E4B 13,921-token prompt in a
    // 16K context, ring vs full, logits bit-identical with and without the drafter; ~115 KB less per token)
    swa_cap_ = std::min(ctx, (hp.n_swa + max_chunk + 4096 + 255) / 256 * 256);
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
        L.i8 = S.i8;
      } else {
        const size_t kvb = (size_t)L.nkv * L.cap * L.hd, scb = (size_t)L.nkv * L.cap * (L.hd / 64) * 4;
        // (the int8 copy only where the assistant drafter reads it: the sources of the last sliding and the last
        //  global layer, and only with the drafter; the two-pass attention reads the float rows. 26B: ~20 % less
        //  per token, the sliding rings ~0.7 GB less)
        L.i8 = !q2_ || (opt.mtp && (il == g4_kv_layer(hp.n_layer - 2) || il == g4_kv_layer(hp.n_layer - 1)));
        const size_t ib = L.i8 ? kvb : 64, is = L.i8 ? scb : 64;
        L.kq = buf(ib);
        L.vq = buf(ib);
        L.ksc = buf(is);
        L.vsc = buf(is);
        for (cl_mem m : {L.kq, L.vq}) fill(m, ib);
        for (cl_mem m : {L.ksc, L.vsc}) fill(m, is);
        if (q2_) {  // (two-pass mode: f32 K / f16 V rows for attn_fa_g; the int8 cache stays for the drafter)
          L.kc = buf(kvb * 4);
          L.vc = buf(kvb * 2);
          fill(L.kc, kvb * 4);
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
    aq_ = buf(M * maxK * (q2mm_ ? 2 : 1));
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
        hc_bak_ = buf(conv_bytes() - ple_hist_off());
        ple_sv_ = buf((size_t)(1 + kMaxDrafts) * hp.hce() * 4);
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
    asc_ = buf(M * (maxK / 32) * 4 * (q2mm_ ? 2 : 1));
    absum_ = buf(M * (maxK / 16) * 2 * (q2mm_ ? 2 : 1));
    big_ = buf(M * big_w * 4);
    // MoE: the expert activations / outputs (mh_ [slots][F], my_ [slots][E] f32) share memory with buffers that are
    // only used inside the attention / DeltaNet part of a layer (z_; conv_out_, o_, qrope_): never live at the same time
    // (Qwen3.6 35B, 2048-token chunks: 167 MB, ~18K tokens of context)
    const size_t z_bytes = std::max<size_t>(M * hp.ssm_d_inner * 4, 64);
    const size_t cv_bytes = (std::max<size_t>(M * hp.conv_dim() * 4, 64) + 4095) / 4096 * 4096,
                 o_bytes = (M * std::max(hp.ssm_d_inner, hp.n_head * hp.head_dim) * 4 + 4095) / 4096 * 4096,
                 qr_bytes = M * hp.n_head * hp.head_dim * 4;
    const bool share = hp.moe;
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
    // local memory (27B, 11K prompt: IQ3_S / IQ4_XS / IQ3_XXS 5.8 -> 2.2 s than a per-32-block kernel)
    static const bool by_type = getenv("ARC_PROF_W8TYPE") != nullptr;  // (profiling: decode time per weight type)
    const std::string nm = by_type ? std::string("decode_w8_") + ggml_type_name(type) : std::string("decode_w8");
    // Q8_0 / Q4_0 with their vector block decoder (vblock, decode_w8_vb): 3x faster than decode_w8_sb. (For the
    // codebook types it was slower: each lane loads its own superblock, 16 cache lines per load instruction.)
    size_t lx = 256;
    while (N % lx) lx /= 2;
    const bool vb = lx >= 64 && (type == GGML_Q8_0 || type == GGML_Q4_0);
    // whole rows of up to 16 superblocks, staged in local memory with coalesced loads (decode_w8_sl): types without
    // mins (Qwen3.6 35B prompt experts, 2048-token chunk: IQ3_XXS 1038 -> 518 ms, IQ4_XS 433 -> 303 ms)
    const int nsb = K / 256;
    const bool sl = !w8_requant(type) && nsb <= 16 && N % 16 == 0 &&
                    16 * nsb * raw_padded_sb_bytes(type) <= 32768 && type != GGML_IQ1_M;
    if (sl) {
      cl_kernel k = kernel("decode", "-DQT=" + std::to_string(type) + " -DSL_NSB=" + std::to_string(nsb), "decode_w8_sl");
      set_args(k, 0, raw, N, K, tabs_, dq, dh);
      size_t g1 = (size_t)N * nsb, l1 = (size_t)16 * nsb;
      run(nm.c_str(), k, 1, &g1, &l1);
      return;
    }
    cl_kernel k = kernel("decode", "-DQT=" + std::to_string(type), vb ? "decode_w8_vb" : "decode_w8_sb");
    set_args(k, 0, raw, N, K, tabs_, dq, dh);
    size_t gws[2] = {(size_t)N, (size_t)K / 256}, lws[2] = {lx, 1};
    run(nm.c_str(), k, 2, gws, lws);
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
    // (measured per type with bench/gemv_bench on the regrouped GPU forms; IQ3_S stays faster in gemv)
    return (w.type == GGML_IQ4_XS || w.type == GGML_Q4_0 || w.type == GGML_IQ3_XXS || w.type == GGML_IQ2_XXS ||
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
  // ---- two int8 passes (Gemma 4, always; QUEUE F21.4): every activation row is stored twice by quant_store16,
  // the int8 values and, at fixed offsets in the same buffers, their rounding residual in int8; a matrix product
  // runs on both and adds them (the f32 reference simulating this: Gemma 4 26B KL 0.0985 -> 0.0053 together with
  // two-pass attention). The offsets come from alloc_buffers' sizes (max_chunk rows of the widest input).
  bool q2_ = false;
  // (two-pass matmul inputs: Gemma 4's mode, and Flash Next's dense matmuls with ARC_FN_Q2=1, an F21.13 experiment:
  //  363-token KL against f32 0.0483 -> 0.0384, 4K prompt 191 -> 160 tok/s; its experts stay one pass)
  bool q2mm_ = false;
  cl_mem qf_ = nullptr, mh2_ = nullptr;
  size_t q2m_hq_ = 0, q2m_sc_ = 0;  // (the experts' down input: residual offsets in mhq_ / mhsc_, bytes)
  bool q2_off_ = false;
  bool q2_nocap_ = false;  // (the output head's two passes run without their softcap epilogue)
  cl_mem fa_part_ = nullptr;  // (attn_fa_g -DFA_SPLIT partials: 8 rows x heads x 64 splits x (head dim + 2) floats)
  bool q2_fuse_ = false, q2_fused_ = false;  // (matmul -> matmul1: run the fused two-pass GEMV; it ran)
  size_t q2_aq_ = 0, q2_sc_ = 0, q2_ab_ = 0;  // bytes
  std::string q2_defs_;
  void q2_init() {
    q2_ = hp.gemma;  // (Gemma 4's only mode since 2026-10-22: KL < 0.01 against f32, the one-pass mode removed)
    // (Gemma 4 E2B / E4B (per-layer embeddings): one int8 pass for the matmuls, the float attention kept; the
    //  user's limit is KL 0.015 and the one-pass E4B measured 0.0010 with int8 attention as well (2026-10-23).
    //  ARC_G4_Q2MM=1: two passes there too, as before)
    const bool g4_one = hp.gemma && hp.n_ple > 0 && !(getenv("ARC_G4_Q2MM") && atoi(getenv("ARC_G4_Q2MM")) != 0);
    q2mm_ = (q2_ && !g4_one) || (hp.hc && getenv("ARC_FN_Q2") && atoi(getenv("ARC_FN_Q2")) > 0);
    if (!q2mm_) return;
    const size_t M = max_chunk;
    const int maxK = std::max({hp.n_embd, hp.n_ff, hp.ssm_d_inner, hp.n_head * hp.head_dim, hp.hce(), hp.n_ff_exp_pad});
    q2_aq_ = M * maxK;
    q2_sc_ = M * (maxK / 32) * 4;
    q2_ab_ = M * (maxK / 16) * 2;
    q2_defs_ = " -DQ2_AQ=" + std::to_string(q2_aq_) + " -DQ2_SC=" + std::to_string(q2_sc_ / 4) + " -DQ2_AB=" +
               std::to_string(q2_ab_ / 2);

  }
  void matmul(const char* name, const GpuWeight& w, int Mp, cl_mem C, int ldc, int c_off, int acc, bool act32) {
    // (the output head with a logit softcap: the cap applies to the sum of both passes, not to each; the head GEMVs
    //  cap in their epilogue, so both run without it and softcap_rows caps the sum)
    const bool cap = q2mm_ && !q2_off_ && &w == &lm_head_ && hp.softcap > 0.0f;
    q2_nocap_ = cap;
    // (one token: the GEMV runs both passes over one read of the weights, -DQ2F; it says so in q2_fused_)
    q2_fuse_ = q2mm_ && !q2_off_ && cur_T_ <= 4;  // (2-4 tokens: gemv_mt_v only, see matmul1)
    q2_fused_ = false;
    matmul1(name, w, Mp, C, ldc, c_off, acc, act32);
    q2_fuse_ = false;
    if (!q2mm_ || q2_off_) return;
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
      int N = w.N, K = w.K, M = Mp;
      rd_bytes += raw_bytes(w);
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
      // gemv_mt_v) 22.6 -> 38.9 with 2; Q6_K one row (it spills with more)
      const int tb = cur_T_;
      const bool vk = gemv_v_ok(w) || w.type == GGML_IQ3_S;
      int gr = vk ? gemv_v_rows(w) : w.N <= 4096 ? 1 : w.N <= 8192 ? 2 : 4;
      // Below 6 tokens (MTP verification of prose: 2-3 rows) 4 rows per sub-group for the codebook types and for long
      // rows (K >= 8192, the FFN down projection), Q4_K / IQ4_XS short rows 2: 27B, bench/gemv_bench, 3 tokens: gate
      // IQ3_XXS 57.8 -> 64.7 GB/s, down Q4_K 57.5 -> 70.6; Indonesian prose with MTP 6.95 -> 7.2 tok/s. With 6-8
      // tokens the earlier rule stays: there 4 for the down projection and 2 for gate / up were faster only in
      // bench/gemv_bench (code with 7 drafts or DFlash2 2 % slower in the model). Bit-identical either way.
      {
        const bool cb = vk && w.type != GGML_IQ4_XS;
        if (w.type == GGML_Q6_K) gr = 1;
        else if (cur_T_ < 6 && (vk || w.type == GGML_Q4_K)) gr = w.K >= 8192 || cb ? 4 : 2;
        else gr = w.type == GGML_IQ3_S ? 2 : vk ? (w.N >= 8192 ? 4 : 2) : std::min(gr, 2);
      }
      // (two-pass mode, 2-4 tokens: gemv_mt_v with the residuals as TB further slots, -DQ2F, both passes in one launch)
      const bool fuse = q2_fuse_ && vk && a <= 1;
      cl_kernel k = kernel("decode", std::string("-cl-intel-256-GRF-per-thread ") + "-DQT=" +
                                         std::to_string(w.type) + " -DGROWS=" + std::to_string(gr) + " -DTB=" +
                                         std::to_string(fuse ? 2 * tb : tb) + head_defs(w) + mt_lazy_opt() +
                                         (fuse ? " -DQ2F" + q2_defs_ : std::string()),
                           vk ? "gemv_mt_v" : "gemv_mt");
      q2_fused_ = fuse;
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
      if (dfl_) dfl_capture(L, T, Mp);
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

#include "engine_gemma.h"  // (Gemma 4 layers)

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
    // heads per sub-group: one at HD 512 (registers); else as the prompt kernels. (Gemma 4's own layers run their
    // float attention in layer_gemma_attn; only its assistant drafter comes here)
    const int hs = hd >= 512 ? 1 : attn_hs_for(hpg), nsp = hpg / hs;
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
             " -DHS=" + std::to_string(hs) + " -DTPW=" + std::to_string(tpw) + sb_opt(hd);
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
        const int hg = hpg % 4 == 0 && qwen_ds != 1 ? 4 : 2, n = hg == 4 ? 1 : std::min(left, 2), tr = n, nsub = 4;
        const int sgs = hpg / hg * nsub, lsz = 16 * sgs;
        const int lo = win ? std::max(0, pos0 + r0 - win + 1) : 0, plast = pos0 + r0 + n - 1;
        static const int qs_len = getenv("ARC_QWEN_DS_S") ? std::max(1, atoi(getenv("ARC_QWEN_DS_S"))) : 256;  // (experiment)
        const int S = qs_len, kstart = lo / S * S, ns = (plast - kstart) / S + 1;
        int p0 = pos0 + r0, nv = n, q0 = r0, SS = S, ks = kstart, nss = ns;
        cl_kernel k = kernel("ops", opts(1) + " -DHG=" + std::to_string(hg) + " -DNSUB=" + std::to_string(nsub) +
                                        " -DTR=" + std::to_string(tr) + " -DROWVMT", "attn_dsplit");
        set_args(k, 0, qi_, qsc_, L.kq, L.ksc, L.vq, L.vsc, cp, nn, p0, nv, apart_, w, ks, SS, nss, q0);
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
      int tpw = 1;
      // (Qwen3.6 35B, 2 kv heads: a single token takes attn_decode (16 * nsp < 128), and attn_split over 2 rows split
      // the keys differently, so MTP output drifted from plain decoding after ~40 tokens of a 2K prompt; its
      // verification rows go through attn_decode one by one. The dense Qwen models pass the identity test as they are.)
      if (!hp.moe || 16 * nsp >= 128 || drafter)
        while (tpw < T - r0 && tpw < 4 && 16 * nsp * tpw * 2 <= 512) tpw *= 2;  // apart_ holds 4 rows of partials
      const int n = std::min(tpw, T - r0);
      if (16 * nsp * tpw >= 128) {
        const int lsz = 16 * nsp * tpw, plast = pos0 + r0 + n - 1, lo = win ? std::max(0, pos0 + r0 - win + 1) : 0;
        int S, kstart, ns, p0 = pos0 + r0, nv = n, q0 = r0;
        pick_split(lo, plast, lsz / 16, dec_smax, dec_sg, S, kstart, ns);
        cl_kernel k = kernel("ops", opts(tpw), "attn_split");
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
          set_args(kd, 0, qi_, qsc_, L.kq, L.ksc, L.vq, L.vsc, cp, nn, pp, apart_, row, w, kstart, S);
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

#include "engine_flashnext.h"  // (Qwen3.8 Flash Next)

#include "engine_qwen.h"  // (Qwen layers)

#include "engine_moe.h"  // (Mixture-of-experts FFN (qwen35moe, Flash Next, Gemma 4 26B))

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

#include "engine_drafters.h"  // (Speculative decoding)

 private:
};
