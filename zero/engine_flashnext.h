// Qwen3.8 Flash Next: hyper-connection mixers and the PLE n-gram embeddings.
// A part of class Engine (zero/engine.h), included inside the class body at the place it was cut from: not a
// header of its own. Split out of engine.h unchanged on 2026-10-23 (QUEUE F26.3).

  // ---------------------------------------------------------------- Flash Next hyper-connections and PLE
  // The mixer of rows r0 .. r0 + T - 1 of the residual streams x_: per-stream norm, low-rank down (+ inject logits,
  // into hc_lo_), silu(lo / hc_n), up to the stream gates, then the gated mean over the streams = the block input
  // (f32 in hco_, quantized in aq_ / asc_ / absum_ with Mp rows)
  void hc_mix(cl_mem normw, const GpuWeight& down, const GpuWeight& up, int T, int Mp, int act32, int r0) {
    const int E = hp.n_embd, HC = hp.hc_n, W = hp.hce(), LD = down.N, LR = hp.hc_lr, KP = hc_lr_pad();
    float eps = hp.eps;
    // single tokens: two fused kernels with f32 activations (norm + down + inject, then silu + up + gated mean) and
    // the block input's quantization; prompt chunks below: int8 GEMMs
    if (T == 1 && W <= 10240 && KP <= 1024) {
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
      if (f32_mode == 4 && !q2mm_) {  // (two-pass matmuls: hc_lo_quant stored the residual already)
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
