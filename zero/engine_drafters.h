// Speculative decoding: the Gemma 4 assistant drafter, the DFlash2 block drafter, the MTP block, verification.
// A part of class Engine (zero/engine.h), included inside the class body at the place it was cut from: not a
// header of its own. Split out of engine.h unchanged on 2026-10-23 (QUEUE F26.3).

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
    // as often (27B code 84.5 vs 85.4 %, 35B equal) and the drafter needs 0.84 GiB less (27B; Q8_0 before 17 Oct)
    auto q = quantize_rows_q4_0(v.data(), N, K);
    return make_weight(GGML_Q4_0, q.data(), q.size(), N, K);
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
    if (hp.hc) hc_mean(T);
    quant_rows(hp.hc ? hco_ : x_, hp.n_embd, Mp, T, a32);
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
    // (the top-K in one pass instead of nth_element over the whole vocabulary per position, ~2 ms each on the host:
    //  code answers 35B 48.4 -> 52.7, 27B 16.9 -> 18.3 tok/s with the same drafts)
    // drafts are kept while the drafter's own probability of the chosen path stays >= ARC_DFL_P (each draft verified
    // costs more rows in the main model's pass: an unlikely tail is cheaper not to check)
    static const double min_p = getenv("ARC_DFL_P") ? atof(getenv("ARC_DFL_P")) : 0.3;  // (code 11.4 -> 11.6 tok/s, prose up)
    double path_p = 1.0;
    for (int t = 0; t < std::min(nd, max_drafts); ++t) {
      const float* l = &dfl_lg_[(size_t)t * V];
      {
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
      head_rows(n - 1, 1, shnorm_, hout);
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
    const size_t nl = (size_t)hp.n_vocab;
    for (int k = 0;; ++k) {
      CL_CHECK(clEnqueueReadBuffer(cl_.queue, logits_, CL_TRUE, 0, nl * 4, mtp_logits_host_.data(), 0, nullptr, nullptr));
      const auto first = mtp_logits_host_.begin(), last = first + nl;
      const auto mx = std::max_element(first, last);
      const int d = (int)(mx - first);
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
    if (hp.hc && hp.ple_layer >= 0) hc_spec_save(tok);  // (before embed: it advances the token ids)
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
    if (hp.hc && hp.ple_layer >= 0 && accepted < nd) hc_rollback(accepted + 1);
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
