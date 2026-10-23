// Qwen layers: Gated DeltaNet, Flash Next QSA, full attention, dense FFN.
// A part of class Engine (zero/engine.h), included inside the class body at the place it was cut from: not a
// header of its own. Split out of engine.h unchanged on 2026-10-23 (QUEUE F26.3).

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
        // (a T = 1 kernel with 4 lanes per state column and 128 GRF, bit-identical, was slower: 35B 248-341 vs 226-254
        //  ms per 96 tokens; removed 2026-10-22)
        cl_kernel k = k_gdn_recur_;
        set_args(k, 0, cq, C, gq, nk, nv, L.S, Tq, oq, spec_ ? L.S_next : L.S);
        size_t g1 = (size_t)nv * 128, l1 = 64;
        run("gdn_recur", k, 1, &g1, &l1);
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
                                     " -DHS=" + std::to_string(hs) + " -DTPW=1" + sb_opt(hd) + q, "attn_decode");
    int nn = nh, hdd = hd, cp = ctx, w = 0;
    for (int t = 0; t < T; ++t) {
      const int p = pos + t;
      int S = 256;
      while (S > 32 && (p / S + 1) * nkv * nsp < 512) S /= 2;
      int kstart = 0, ns = p / S + 1, pp = p, row = t, prow = 0;
      set_args(kd, 0, qi_, qsc_, L.kq, L.ksc, L.vq, L.vsc, cp, nn, pp, apart_, row, w, kstart, S, qsa_mask_, qmw);
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
