// Gemma 4 layers: attention (float, attn_fa_g / attn_fa_gh) and the dense / MoE FFN.
// A part of class Engine (zero/engine.h), included inside the class body at the place it was cut from: not a
// header of its own. Split out of engine.h unchanged on 2026-10-23 (QUEUE F26.3).

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
      cl_kernel k = kernel("ops", (gprep_small ? std::string() : std::string("-cl-intel-256-GRF-per-thread ")) + hdo,
                           "attn_prep_g");
      float eps = hp.eps, base = L.swa ? hp.rope_base_swa : hp.rope_base, qscale = 1.0f;  // Gemma 4: no 1/sqrt(d)
      int pp = p0, hv = L.has_v ? 1 : 0, uff = L.rope_ff ? 1 : 0, cap = L.cap, nn = nh, nk = nkv, ld = ldq, TT = Tq;
      cl_mem rt = rope_table(L.swa ? 0 : 1, hd, base, L.rope_ff, Tq);
      set_args(k, 0, from(big_, (size_t)r0 * ldq * 4), ld, nn, nk, hv, L.q_norm, L.k_norm, TT, pp, eps, base,
               L.rope_ff ? L.rope_ff : L.q_norm, uff, qscale, qi_, qsc_, L.kq, L.ksc, L.vq, L.vsc, cap, rt);
      int f16 = L.i8 ? 1 : 3;  // (the f32 query and the float K / V rows for attn_fa_g; 2: no int8 copy)
      set_args(k, 22, f16, from(qf_, (size_t)r0 * nh * hd * 4), L.kc, L.vc);
      size_t gws[2] = {(size_t)(nh + (own_kv ? nkv : 0)) * 16, (size_t)Tq}, lws[2] = {16, 1};
      run("attn_prep", k, 2, gws, lws);
    };
    // (two-pass mode: float attention over the f16 rows for every pass; rows r0 .. r0 + n at position p0)
    // (prompt chunks of the sliding layers (head 256): 32 rows per work-group sharing each key block in local memory;
    //  the global layers (head 512) do better without, but with 2 q heads per sub-group instead of 4 (11.4 KB spill
    //  -> less). 26B, 4K prompt: sliding 10.5 -> 4.5 s, global 6.2 -> 2.2 s. ARC_FA_ROWS / ARC_FA_GQ override them;
    //  every choice is bit-identical)
    static const int fa_rows = getenv("ARC_FA_ROWS") ? std::max(1, atoi(getenv("ARC_FA_ROWS"))) : 0;
    static const int fa_gq = getenv("ARC_FA_GQ") ? std::max(1, atoi(getenv("ARC_FA_GQ"))) : 0;
    // (prompt chunks of the head-512 layers: attn_fa_gh, each key row read once per row for all q heads of its kv head;
    //  ARC_FA_GH=0: attn_fa_g; ARC_FA_FS: sub-groups per work-group, default 8 (each holds 4 output dims per lane:
    //  26B, 13,921-token prompt 90.7 -> 77.7 s, bit-identical; 4 sub-groups: 8 dims). ARC_FA_GR: query rows per
    //  work-group, each key / value row read once for all of them; default as many as fit 64 output floats per lane
    //  (E4B, 4 q heads per kv head: 4 rows, 13,921 tokens 59.8 -> 47.2 s; 26B: 2, about the same; 12B: 1)
    static const bool fa_gh = !getenv("ARC_FA_GH") || atoi(getenv("ARC_FA_GH")) != 0;
    static const int fa_fs_env = getenv("ARC_FA_FS") ? std::max(1, atoi(getenv("ARC_FA_FS"))) : 0;
    static const int fa_gr_env = getenv("ARC_FA_GR") ? std::max(1, atoi(getenv("ARC_FA_GR"))) : 0;
    // (generation / verification, <= 8 rows: flash-decoding, the key range split over many sub-groups and combined;
    //  ARC_FA_SPLIT=0: one sub-group per (row, q-head pair) over every key, as the prompt kernel)
    static const bool fa_split = !getenv("ARC_FA_SPLIT") || atoi(getenv("ARC_FA_SPLIT")) != 0;
    auto fa_g = [&](int r0, int n, int p0) {
      if (fa_split && n <= 8) {
        const int gq = std::min(hpg, fa_gq ? fa_gq : hd >= 512 ? 2 : 4);
        const std::string fd = "-cl-intel-256-GRF-per-thread -DFA_G -DFA_SPLIT -DHD=" + std::to_string(hd) + " -DGQ=" +
                               std::to_string(gq);
        cl_kernel k = kernel("ops", fd, "attn_fa_g");
        // (splits of the launch: those of its widest row, as the kernel computes them)
        const int nlo = L.swa ? nc_lo_ : -1, nhi = L.swa ? nc_hi_ : -1, last = p0 + n - 1;
        int hi = last, lo0 = win > 0 ? std::max(0, p0 - win + 1) : 0;
        if (nhi >= p0 && nlo <= last) hi = std::max(hi, nhi);
        const int nb = hi / 16 - lo0 / 16 + 1, ns = std::min(64, (nb + 7) / 8);
        if (!fa_part_) fa_part_ = buf((size_t)8 * hp.n_head * 64 * (std::max({hp.head_dim, hp.hd_swa, hp.hd_full}) + 2) * 4);
        int cp = L.cap, nn = nh, nk = nkv, pp = p0, MM = n, w = win, sg = ns, nl = nlo, nh2 = nhi;
        set_args(k, 0, from(qf_, (size_t)r0 * nh * hd * 4), L.kc, L.vc, cp, nn, nk, pp, MM,
                 from(o_, (size_t)r0 * nh * hd * 4), w, nl, nh2, sg, fa_part_);
        size_t gws[3] = {16, (size_t)(nh / gq), (size_t)n * ns}, lws[3] = {16, 1, 1};
        run(L.swa ? "attn_swa" : "attn_global", k, 3, gws, lws);
        cl_kernel kc2 = kernel("ops", fd, "attn_fa_gc");
        set_args(kc2, 0, fa_part_, nn, pp, MM, from(o_, (size_t)r0 * nh * hd * 4), w, nl, nh2);
        size_t g2[3] = {(size_t)hd, (size_t)nh, (size_t)n};
        run(L.swa ? "attn_swa" : "attn_global", kc2, 3, g2, nullptr);
        return;
      }
      // (at most 64 output floats per lane, q heads x dims: with 4 sub-groups the 12B's 16 q heads held 128, spilled
      //  6.4 KB per thread and made its global attention slower, 3.4 -> 4.4-4.9 s for a 4K prompt; DeepSeek's audit)
      // (head 256, the sliding layers: 4 sub-groups and up to 4 rows; 26B, 4K prompt: sliding attention 4.17 -> 2.26 s
      //  instead of attn_fa_g with 32 rows)
      const int fa_fs = fa_fs_env ? fa_fs_env : hd >= 512 ? 8 : 4;
      if (fa_gh && n > 8 && hd >= 256 && hpg * (hd / fa_fs / 16) <= 64 && hd % (fa_fs * 16) == 0 &&
          (hd / fa_fs / 16 == 4 || hd / fa_fs / 16 == 8 || hd / fa_fs / 16 == 16)) {
        const int fa_gr = fa_gr_env ? fa_gr_env : std::clamp(64 / (hpg * (hd / fa_fs / 16)), 1, 4);
        cl_kernel k = kernel("ops", "-cl-intel-256-GRF-per-thread -DFA_GH -DHD=" + std::to_string(hd) + " -DHPG=" +
                                        std::to_string(hpg) + " -DFS=" + std::to_string(fa_fs) + " -DGR=" + std::to_string(fa_gr),
                             "attn_fa_gh");
        int cp = L.cap, nn = nh, nk = nkv, pp = p0, MM = n, w = win, nlo = L.swa ? nc_lo_ : -1, nhi = L.swa ? nc_hi_ : -1;
        set_args(k, 0, from(qf_, (size_t)r0 * nh * hd * 4), L.kc, L.vc, cp, nn, nk, pp, MM,
                 from(o_, (size_t)r0 * nh * hd * 4), w, nlo, nhi);
        size_t gws[3] = {(size_t)16 * fa_fs, (size_t)nkv, (size_t)(n + fa_gr - 1) / fa_gr}, lws[3] = {(size_t)16 * fa_fs, 1, 1};
        run(L.swa ? "attn_swa" : "attn_global", k, 3, gws, lws);
        return;
      }
      const int gq = std::min(hpg, fa_gq ? fa_gq : hd >= 512 ? 2 : 4);
      const int R = n <= 8 ? 1 : fa_rows ? fa_rows : hd >= 512 ? 1 : 32;  // (sliding: 32 rows 4.43 -> 4.16 s; 64: too large)
      cl_kernel k = kernel("ops", "-cl-intel-256-GRF-per-thread -DFA_G -DHD=" + std::to_string(hd) + " -DGQ=" +
                                      std::to_string(gq) + (R > 1 ? " -DFA_ROWS=" + std::to_string(R) : std::string()), "attn_fa_g");
      int cp = L.cap, nn = nh, nk = nkv, pp = p0, MM = n, w = win, nlo = L.swa ? nc_lo_ : -1, nhi = L.swa ? nc_hi_ : -1;
      set_args(k, 0, from(qf_, (size_t)r0 * nh * hd * 4), L.kc, L.vc, cp, nn, nk, pp, MM,
               from(o_, (size_t)r0 * nh * hd * 4), w, nlo, nhi);
      size_t gws[3] = {16, (size_t)(nh / gq), (size_t)(n + R - 1) / R * R}, lws[3] = {16, 1, (size_t)R};
      run(L.swa ? "attn_swa" : "attn_global", k, 3, gws, lws);
    };
    if (!batch_.empty()) {  // a batched pass (T <= 8): each sequence's rows with its slot's caches and position
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
    matmul("gemm_gate", L.gate, Mp, big_, 2 * F, 0, 0, a_in);
    matmul("gemm_up", L.up, Mp, big_, 2 * F, F, 0, a_in);
    const int a_dn = act32_for(false);
    {
      int gelu = 1, FF = F, MM = Mp, TT = T, fused = 0;  // (gelu(gate) * up, quantized for the down projection)
      set_args(k_swiglu_, 0, big_, FF, MM, TT, aq_, asc_, absum_, a_dn, fused, gelu);
      run_rows_quant("swiglu_quant", F, T, Mp);
    }
    matmul("gemm_down", L.down, Mp, big_, E, 0, 0, a_dn);
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
