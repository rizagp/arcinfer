// Mixture-of-experts FFN (qwen35moe, Flash Next, Gemma 4 26B): routing, per-tile kernels, moe_rg, decode + GEMM.
// A part of class Engine (zero/engine.h), included inside the class body at the place it was cut from: not a
// header of its own. Split out of engine.h unchanged on 2026-10-23 (QUEUE F26.3).

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
    const int fused = cur_T_ > 8 && !q2mm_ ? 1 : 0;  // (two passes: no fused epilogue)
    cl_mem rl = rl_, gu = big_;
    if (g4) {
      // (done above)
    } else if (T == 1 && L.rgu.raw) {
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
    // (Gemma 4, two passes: prompt chunks through the T > 1 routing and moe_rg, whose two passes it runs; generation
    //  and verification (T <= 8) through the per-tile kernels with both passes fused, -DQ2F, as plain decoding)
    static const int rg_min0 = getenv("ARC_MOE_RG_MIN") ? atoi(getenv("ARC_MOE_RG_MIN")) : 320;
    const bool q2f = q2_ && g4 && hp.moe_std && T <= 8 && T < rg_min0;
    const bool q2g = q2_ && g4 && !q2f;
    const std::string q2fd = q2f ? " -DQ2F" + q2_defs_ + " -DQ2M=" + std::to_string(q2m_hq_) + " -DQ2MS=" +
                                       std::to_string(q2m_sc_ / 4)
                                 : std::string();
    // (fused two-pass tiles hold twice the sums: as many slots as the tokens (an expert gets at most T, here <= 8);
    //  TM 16 spilled 4 KB per thread in verification. Each slot's terms are the same at any TM: bit-identical)
    const int tm = T == 1 && !q2g ? 1 : q2f ? (T <= 4 ? 4 : 8) : hp.moe_std ? tm_std : 16, n = T * NU;
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
    // are decoded once per 16-token tile here, once per chunk there).
    static const int rg_min = getenv("ARC_MOE_RG_MIN") ? atoi(getenv("ARC_MOE_RG_MIN")) : 320,
                     rg_max = getenv("ARC_MOE_RG_MAX") ? atoi(getenv("ARC_MOE_RG_MAX")) : 1280;
    // Without decode + GEMM (Flash Next: F padded) only the down projection takes moe_rg, at every chunk size from
    // rg_min: Flash Next, 8K prompt in 2048-token chunks: down 13.8 -> 6.2 s GPU time, 145 -> 165 tok/s, and KL to the
    // CPU reference 0.0221 -> 0.0191 (337 tokens; max 0.96 -> 0.27); its gate / up stay per tile (15.4 vs 13.7 s, KL
    // 0.0230 with both).
    const bool gemm_ok = moe_group_ > 0 && Fp == F && moe_gemm_layer_ok(L);
    const bool rg_ok = hp.moe_std && tm == 16 && (T >= rg_min || q2g) && F % 64 == 0 && moe_rg_layer_ok(L);
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
      // (prompt tiles: the activation scales transposed first, a token's 80 scales then one coalesced read instead of
      //  16 lanes in 16 rows; Flash Next 4K prompt: gate / up 8.6 -> 7.0 s with the scales left out entirely)
      static const bool asct_on = !getenv("ARC_MOE_ASCT") || atoi(getenv("ARC_MOE_ASCT")) != 0;
      const bool asct = asct_on && tm > 1 && !q2f;
      cl_kernel k = kernel("decode", (q2f && tm > 1 ? std::string("-cl-intel-256-GRF-per-thread ") : grf_env("ARC_MOE_GRF")) +
                                         "-DMOE_Q -DQT=" + std::to_string(L.ex_type[0]) + " -DGROWS=" +
                                         std::to_string(grows) + " -DTM=" + std::to_string(tm) + xdef + hld + q2fd +
                                         (asct ? " -DASCT" : ""), "moe_gateup_q");
      int M = Mp, K = E, FF = F;
      cl_mem sc = asc_;
      if (asct) {
        cl_kernel t = kernel("ops", "", "asc_transpose");
        int MM = Mp, KB = E / 32;
        set_args(t, 0, asc_, MM, KB, asct_);
        size_t g2[2] = {((size_t)KB + 15) / 16 * 16, (size_t)Mp};
        run("moe_gateup", t, 2, g2, nullptr);
        sc = asct_;
      }
      if (xc_) {
        set_args(k, 0, xc_->xtab[L.il], xc_->xtab[L.il], FF, K, tabs_, aq_, sc, M, tiles, stok_, mh_, xc_->xtab[L.il],
                 xoff(L, 0), xoff(L, 1));
      } else {
        set_args(k, 0, L.ex_q[0], L.ex_q[1], FF, K, tabs_, aq_, sc, M, tiles, stok_, mh_);
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
                                         (Fp != F ? " -DFV=" + std::to_string(F) : std::string()) + q2fd, "moe_down_q");
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
    if (!hp.moe_std) return false;
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
          acc = i == 1 ? 2 : 0, ne = NE,
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
      cl_kernel k = moe_gemm_kernel(100 + w8_variant(L.ex_type[i]));
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
