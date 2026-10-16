// arcinferzero: the routed experts stay on the SSD (expert_store.h); the GPU keeps an LRU cache of them.
//
// Each MoE layer has a table xaddr[NE] (host USM, read by the expert kernels built with -DXCACHE): the address of
// every routed expert's record in the cache. After a layer's routing the engine reads the chosen expert ids and calls
// ensure(): a cached expert is a hit (moved to the front of its pool's LRU list); a missing one takes the slot at the
// back of the list, is read from the SSD into a host staging area and copied into the slot by the GPU.
//   - generation and MTP verification (few tokens): the new expert goes to the front of the list (plain LRU).
//   - prompt chunks (many tokens, nearly every expert of the layer): the new experts go to the back of the list, so
//     the next layer's misses reuse those slots first and a prompt does not sweep the cache (the back of the list
//     works as the prompt's transient area). While a thread reads the next layer's missing experts into the other
//     staging half (prefetch), the GPU works on this layer.
// The command list is in order and ensure() runs after a blocking read of the routing ids, so the GPU has finished
// everything before this layer; a slot reused within the layer is written by a copy queued after the kernels that
// read its previous expert.
#pragma once
#include "expert_store.h"
#include <algorithm>
#include <thread>

struct XCache {
  ExpertStore st;
  int NE = 0, NL = 0;
  struct Pool {
    uint64_t rec = 0;          // record bytes of the layers using this pool
    int n = 0, used = 0;       // slots, slots ever filled
    std::vector<char*> addr;   // slot -> device address
    std::vector<int> key, prev, next;  // slot -> layer * NE + expert; LRU list (head = most recent)
    int head = -1, tail = -1;
    std::vector<void*> chunks;
  };
  std::vector<Pool> pools;
  std::vector<int> pool_of;   // layer -> pool (-1: not a MoE layer)
  std::vector<int> where;     // layer * NE + expert -> slot, -1 when not cached
  uint64_t* xaddr = nullptr;  // [NL][NE] (host USM)
  std::vector<cl_mem> xtab;   // per layer: its xaddr row as a kernel argument
  // staging: two halves of S records (host USM); layer l reads into half l % 2, the prefetch of the next layer into
  // the other one. stage_e[h][i]: the expert in record i (stage_layer[h]: of which layer; prefetched records)
  int S = 64;
  char* stage[2] = {};
  uint64_t stage_rec = 0;
  std::vector<int> stage_e[2];
  int stage_layer[2] = {-1, -1};
  uint64_t cache_bytes = 0, stage_bytes = 0;
  // statistics (generation / verification lookups; prompt lookups separately)
  uint64_t hits = 0, misses = 0, p_hits = 0, p_reads = 0, pf_reads = 0, rounds = 0;
  double read_ms = 0, wait_ms = 0;

  ~XCache() {
    join();
    save_state();
    if (zrt::oom_hook && active() == this) { zrt::oom_hook = nullptr; active() = nullptr; }
  }
  static XCache*& active() { static XCache* a = nullptr; return a; }
  // the layer of the last ensure() and the experts it set: kernels of that layer may still be launched
  int last_l = -1;
  std::vector<uint8_t> last_seen;
  uint64_t shrunk_bytes = 0;

  // Gives the last piece (<= 1 GiB) of the largest pool back to the driver: its experts leave the cache, except the
  // ones the current layer uses, which move into the least recently used slots elsewhere. Every pool keeps at
  // least one layer's worth of slots. Returns false when nothing can go.
  bool shrink() {
    join();
    zrt::sync();
    Pool* P = nullptr;
    for (auto& Q : pools) {
      const int per = (int)std::max<uint64_t>(1, (1ull << 30) / Q.rec);
      const int s0 = (int)(Q.chunks.size() - 1) * per;
      if (Q.chunks.size() >= 2 && s0 >= NE && (!P || Q.n * Q.rec > P->n * P->rec)) P = &Q;
    }
    if (!P) return false;
    const int per = (int)std::max<uint64_t>(1, (1ull << 30) / P->rec);
    const int s0 = (int)(P->chunks.size() - 1) * per;
    char* base = (char*)P->chunks.back();
    const uint64_t bytes = (uint64_t)(P->n - s0) * P->rec + 65536;
    std::vector<int> rescue;
    for (int s = s0; s < P->n; ++s) {
      const int k = P->key[s];
      if (k < 0) continue;
      unlink(*P, s);
      where[k] = -1;
      P->key[s] = -1;
      if (k / NE == last_l && last_seen.size() == (size_t)NE && last_seen[k % NE]) rescue.push_back(k * 0 + s), rescue.push_back(k);
    }
    for (size_t i = 0; i < rescue.size(); i += 2) {
      const int from = rescue[i], k = rescue[i + 1];
      int t = P->tail;  // (a slot outside the piece that the current layer does not use)
      while (t >= 0 && P->key[t] >= 0 && P->key[t] / NE == last_l && last_seen[P->key[t] % NE]) t = P->prev[t];
      if (t < 0) return false;
      unlink(*P, t);
      if (P->key[t] >= 0) where[P->key[t]] = -1;
      P->key[t] = k;
      where[k] = t;
      push_front(*P, t);
      ZE_CHECK(zeCommandListAppendMemoryCopy(zrt::cl, P->addr[t], P->addr[from], P->rec, nullptr, 0, nullptr));
      xaddr[k] = (uint64_t)P->addr[t];
    }
    zrt::sync();
    P->n = s0;
    P->used = std::min(P->used, s0);
    P->addr.resize(s0);
    P->key.resize(s0);
    P->prev.resize(s0);
    P->next.resize(s0);
    P->chunks.pop_back();
    zeMemFree(zrt::ctx, base);
    zrt::device_bytes -= bytes;
    cache_bytes -= bytes - 65536;
    shrunk_bytes += bytes;
    // table entries into the freed piece: to a valid slot (as set_cache leaves them)
    const Pool* big = &pools[0];
    for (auto& Q : pools) if (Q.rec > big->rec) big = &Q;
    const uint64_t lo = (uint64_t)base, hi = lo + bytes;
    for (size_t i = 0; i < (size_t)(NL + 1) * NE; ++i)
      if (xaddr[i] >= lo && xaddr[i] < hi) xaddr[i] = (uint64_t)big->addr[0];
    fprintf(stderr, "arcinferzero: the GPU ran out of memory (other programs use the shared GPU memory too): expert "
                    "cache reduced by %.2f GiB to %.2f GiB\n", bytes / 1073741824.0, cache_bytes / 1073741824.0);
    return true;
  }

  // The tables and the staging area; the cache itself comes with set_cache once the rest of the model is loaded.
  void init(int n_layers, int n_expert, int stage_records) {
    NL = n_layers;
    NE = n_expert;
    S = std::max(16, stage_records);
    pool_of.assign(NL, -1);
    where.assign((size_t)NL * NE, -1);
    // (+ one row: an expert id of NE, which the routing kernel leaves when a row's logits are NaN, reads the next
    // row's entry; every entry always holds a valid address, see set_cache)
    xaddr = (uint64_t*)zrt::host_alloc((size_t)(NL + 1) * NE * 8);
    memset(xaddr, 0, (size_t)(NL + 1) * NE * 8);
    for (int l = 0; l < NL; ++l) xtab.push_back(zrt::wrap(xaddr + (size_t)l * NE, (size_t)NE * 8));
    for (int l = 0; l < NL; ++l) {
      const uint64_t r = l < (int)st.layers.size() ? st.layers[l].rec : 0;
      if (!r) continue;
      int p = -1;
      for (size_t i = 0; i < pools.size(); ++i) if (pools[i].rec == r) p = (int)i;
      if (p < 0) { pools.emplace_back(); pools.back().rec = r; p = (int)pools.size() - 1; }
      pool_of[l] = p;
      stage_rec = std::max(stage_rec, r);
    }
    // Many record sizes (Qwen3.8 Flash Next: 5 type mixes) would each need a pool of at least a layer's experts (see
    // min_bytes): more than the GPU memory. Then one pool of the largest record serves every layer (a smaller record
    // leaves the rest of its slot unused).
    if (pools.size() > 2) {
      pools.assign(1, Pool{});
      pools[0].rec = stage_rec;
      for (int l = 0; l < NL; ++l) if (pool_of[l] >= 0) pool_of[l] = 0;
    }
    stage_bytes = 2 * stage_rec * S;
    for (int h = 0; h < 2; ++h) {
      stage[h] = (char*)zrt::host_alloc(stage_rec * S);
      stage_e[h].clear();
    }
  }
  uint64_t all_bytes() const {  // every expert of every layer
    uint64_t s = 0;
    for (int l = 0; l < NL; ++l) if (pool_of[l] >= 0) s += pools[pool_of[l]].rec * NE;
    return s;
  }
  // Least bytes set_cache takes: every pool holds at least one layer's experts (a prompt chunk uses all of them).
  uint64_t min_bytes() const {
    uint64_t s = 0;
    for (auto& P : pools) s += P.rec * NE;
    return s;
  }
  // Splits `bytes` over the pools by their share of the experts (device USM, in pieces of <= 1 GiB).
  void set_cache(uint64_t bytes) {
    const uint64_t all = all_bytes();
    bytes = std::min(bytes, all);
    cache_bytes = 0;
    for (size_t p = 0; p < pools.size(); ++p) {
      Pool& P = pools[p];
      uint64_t mine = 0;
      for (int l = 0; l < NL; ++l) if (pool_of[l] == (int)p) mine += P.rec * NE;
      P.n = std::min((int)(mine / P.rec), std::max(NE, (int)((double)bytes * mine / all / P.rec)));
      const int per = (int)std::max<uint64_t>(1, (1ull << 30) / P.rec);
      for (int s0 = 0; s0 < P.n; s0 += per) {
        const int k = std::min(per, P.n - s0);
        // (+64 KB: the expert kernels' block reads run a little past the end of an expert's last row, as arcinfer's
        // upload_padded allows with 256 bytes; a record that is an exact multiple of 4 KB (35B: 1409024 bytes) has no
        // padding, so the last slot of a piece read past the allocation: "device lost" now and then)
        char* b = (char*)zrt::device_alloc((size_t)k * P.rec + 65536);
        P.chunks.push_back(b);
        for (int j = 0; j < k; ++j) P.addr.push_back(b + (size_t)j * P.rec);
      }
      P.key.assign(P.n, -1);
      P.prev.assign(P.n, -1);
      P.next.assign(P.n, -1);
      cache_bytes += (uint64_t)P.n * P.rec;
    }
    // every table entry points at a slot of the largest record size until ensure() sets it: an expert id the
    // kernels read without the host having seen it (out of range after NaN logits) reads valid memory, as the
    // OpenCL buffers of arcinfer do, instead of faulting
    const Pool* big = &pools[0];
    for (auto& P : pools) if (P.rec > big->rec) big = &P;
    for (size_t i = 0; i < (size_t)(NL + 1) * NE; ++i) xaddr[i] = (uint64_t)big->addr[0];
  }
  int slots() const { int n = 0; for (auto& P : pools) n += P.n; return n; }

  static void unlink(Pool& P, int s) {
    if (P.prev[s] >= 0) P.next[P.prev[s]] = P.next[s]; else P.head = P.next[s];
    if (P.next[s] >= 0) P.prev[P.next[s]] = P.prev[s]; else P.tail = P.prev[s];
    P.prev[s] = P.next[s] = -1;
  }
  static void push_front(Pool& P, int s) {
    P.prev[s] = -1;
    P.next[s] = P.head;
    if (P.head >= 0) P.prev[P.head] = s;
    P.head = s;
    if (P.tail < 0) P.tail = s;
  }
  static void push_back(Pool& P, int s) {
    P.next[s] = -1;
    P.prev[s] = P.tail;
    if (P.tail >= 0) P.next[P.tail] = s;
    P.tail = s;
    if (P.head < 0) P.head = s;
  }

  // ---- the cache's contents across runs: the keys of every pool in LRU order (most recent first), saved now and
  // then (state_path); a new process loads the same experts before its first request (warm). Without a saved state the
  // slots are filled with the first experts of every layer of the pool.
  std::string state_path;
  std::chrono::steady_clock::time_point saved_at = std::chrono::steady_clock::now();
  uint64_t warm_bytes = 0;
  double warm_ms = 0;
  void save_state() {
    if (state_path.empty()) return;
    FILE* f = fopen((state_path + ".tmp").c_str(), "wb");
    if (!f) return;
    const int hdr[3] = {NL, NE, (int)pools.size()};
    fwrite(hdr, 4, 3, f);
    for (auto& P : pools) {
      std::vector<int> keys;
      for (int s = P.head; s >= 0; s = P.next[s]) keys.push_back(P.key[s]);
      const int n = (int)keys.size();
      fwrite(&n, 4, 1, f);
      fwrite(keys.data(), 4, keys.size(), f);
    }
    fclose(f);
    std::error_code ec;
    std::filesystem::rename(state_path + ".tmp", state_path, ec);
    saved_at = std::chrono::steady_clock::now();
  }
  void warm() {
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::vector<int>> want(pools.size());
    if (FILE* f = state_path.empty() ? nullptr : fopen(state_path.c_str(), "rb")) {
      int hdr[3] = {};
      if (fread(hdr, 4, 3, f) == 3 && hdr[0] == NL && hdr[1] == NE && hdr[2] == (int)pools.size())
        for (auto& w : want) {
          int n = 0;
          if (fread(&n, 4, 1, f) != 1 || n < 0 || n > NL * NE) break;
          w.resize(n);
          if (fread(w.data(), 4, n, f) != (size_t)n) { w.clear(); break; }
        }
      fclose(f);
    }
    for (size_t p = 0; p < pools.size(); ++p) {
      Pool& P = pools[p];
      std::vector<int>& w = want[p];
      std::vector<uint8_t> have((size_t)NL * NE, 0);
      std::vector<int> keep;
      for (int k : w)  // (saved keys that still belong to this pool)
        if (k >= 0 && k < NL * NE && pool_of[k / NE] == (int)p && !have[k] && (int)keep.size() < P.n) keep.push_back(k), have[k] = 1;
      for (int e = 0; e < NE && (int)keep.size() < P.n; ++e)  // then the first experts of every layer
        for (int l = 0; l < NL && (int)keep.size() < P.n; ++l)
          if (pool_of[l] == (int)p && !have[(size_t)l * NE + e]) keep.push_back(l * NE + e), have[(size_t)l * NE + e] = 1;
      // slots in LRU order (keep[0] = most recent), records read in file order
      std::vector<int> order(keep.size());
      for (size_t i = 0; i < keep.size(); ++i) order[i] = (int)i;
      std::sort(order.begin(), order.end(), [&](int a, int b) { return keep[a] < keep[b]; });
      for (size_t i = 0; i < keep.size(); ++i) {
        const int s = P.used++;
        P.key[s] = keep[i];
        where[keep[i]] = s;
        push_back(P, s);
      }
      for (size_t b0 = 0; b0 < order.size(); b0 += S) {
        const int h = (int)((b0 / S) % 2);
        if (b0 >= 2 * (size_t)S) zrt::sync();  // (the half's previous records are copied out)
        std::vector<ExpertStore::Req> reqs;
        for (size_t j = b0; j < std::min(order.size(), b0 + S); ++j) {
          const int k = keep[order[j]];
          reqs.push_back({k / NE, k % NE, stage[h] + (j - b0) * stage_rec});
        }
        st.read(reqs);
        for (size_t j = b0; j < std::min(order.size(), b0 + S); ++j)
          ZE_CHECK(zeCommandListAppendMemoryCopy(zrt::cl, P.addr[order[j]], stage[h] + (j - b0) * stage_rec, P.rec,
                                                 nullptr, 0, nullptr));
        warm_bytes += reqs.size() * P.rec;
      }
      zrt::sync();
    }
    warm_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  }

  // ---- prefetch: up to S experts of layer X not in the cache, into stage[X % 2], on a thread
  std::thread pf_;
  std::vector<ExpertStore::Req> pf_reqs_;
  double pf_ms_ = 0;
  void join() {
    if (!pf_.joinable()) return;
    const auto t0 = std::chrono::steady_clock::now();
    pf_.join();
    wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    read_ms += pf_ms_;
  }
  // (the staging half of X is free: the copies out of it were queued before the sync of the current layer)
  void prefetch(int X) {
    join();
    if (X < 0 || X >= NL || pool_of[X] < 0) return;
    const int h = X % 2;
    if (stage_layer[h] == X) return;  // (layer-major prompts: already prefetched for an earlier chunk)
    stage_layer[h] = X;
    stage_e[h].clear();
    pf_reqs_.clear();
    for (int e = 0; e < NE && (int)stage_e[h].size() < S; ++e) {
      if (where[(size_t)X * NE + e] >= 0) continue;
      pf_reqs_.push_back({X, e, stage[h] + stage_e[h].size() * stage_rec});
      stage_e[h].push_back(e);
    }
    if (pf_reqs_.empty()) return;
    pf_reads += pf_reqs_.size();
    pf_ = std::thread([this] {
      const auto t0 = std::chrono::steady_clock::now();
      st.read(pf_reqs_);
      pf_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    });
  }

  // Makes the experts `ids` (any order, repeats allowed) of layer l readable by the kernels; prompt = a prompt chunk.
  // dummy_rest: also point the layer's other experts at valid memory (the prompt GEMM decodes whole expert groups).
  void ensure(int l, const int* ids, size_t n, bool prompt, bool dummy_rest) {
    join();
    Pool& P = pools[pool_of[l]];
    uint64_t* xa = xaddr + (size_t)l * NE;
    const int h = l % 2;
    static thread_local std::vector<uint8_t> seen;
    static thread_local std::vector<int> miss;
    seen.assign(NE, 0);
    miss.clear();
    // hits first (to the front: the victims below never include an expert this layer uses)
    for (size_t i = 0; i < n; ++i) {
      const int e = ids[i];
      if (e < 0 || e >= NE || seen[e]) continue;
      seen[e] = 1;
      const int s = where[(size_t)l * NE + e];
      if (s < 0) { miss.push_back(e); continue; }
      if (prompt) ++p_hits; else ++hits;
      unlink(P, s);
      push_front(P, s);
      xa[e] = (uint64_t)P.addr[s];
    }
    if (prompt) p_reads += miss.size(); else misses += miss.size();
    // slots for the misses: never-used ones, then from the back of the list
    std::vector<int> slot(miss.size());
    for (size_t i = 0; i < miss.size(); ++i) {
      int s;
      if (P.used < P.n) {
        s = P.used++;
      } else {
        s = P.tail;
        unlink(P, s);
        where[P.key[s]] = -1;
      }
      slot[i] = s;
    }
    for (size_t i = 0; i < miss.size(); ++i) {
      const int e = miss[i], s = slot[i];
      const size_t key = (size_t)l * NE + e;
      P.key[s] = (int)key;
      where[key] = s;
      if (prompt) push_back(P, s); else push_front(P, s);
      xa[e] = (uint64_t)P.addr[s];
    }
    // records: the prefetched ones are in stage[h] already; the rest are read in rounds of up to S records (a round
    // after the first waits for the GPU to finish copying the previous one out of the staging half)
    size_t i0 = 0;
    std::vector<int> pre(NE, -1);  // expert -> staging record (prefetched)
    if (stage_layer[h] == l)
      for (size_t r = 0; r < stage_e[h].size(); ++r) pre[stage_e[h][r]] = (int)r;
    bool first = true;
    while (i0 < miss.size()) {
      std::vector<ExpertStore::Req> reqs;
      std::vector<std::pair<int, int>> copies;  // (staging record, slot)
      std::vector<uint8_t> used_rec(S, 0);
      if (first) {
        for (size_t i = 0; i < miss.size(); ++i)
          if (pre[miss[i]] >= 0) used_rec[pre[miss[i]]] = 1;
      } else {
        zrt::sync();
        ++rounds;
      }
      int next_free = 0;
      size_t i = i0;
      for (; i < miss.size(); ++i) {
        const int e = miss[i];
        if (first && pre[e] >= 0) {
          copies.push_back({pre[e], slot[i]});
          continue;
        }
        while (next_free < S && used_rec[next_free]) ++next_free;
        if (next_free >= S) break;
        used_rec[next_free] = 1;
        reqs.push_back({l, e, stage[h] + (size_t)next_free * stage_rec});
        copies.push_back({next_free, slot[i]});
      }
      // (the prefetched records past the round's end: still copied in this round)
      if (first)
        for (size_t j = i; j < miss.size(); ++j)
          if (pre[miss[j]] >= 0) copies.push_back({pre[miss[j]], slot[j]}), pre[miss[j]] = -2;
      if (!reqs.empty()) {
        const auto t0 = std::chrono::steady_clock::now();
        st.read(reqs);
        read_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      }
      for (auto& c : copies)
        ZE_CHECK(zeCommandListAppendMemoryCopy(zrt::cl, P.addr[c.second], stage[h] + (size_t)c.first * stage_rec, P.rec,
                                               nullptr, 0, nullptr));
      // next round: the misses not done yet (skipping prefetched ones copied above)
      i0 = i;
      while (i0 < miss.size() && pre[miss[i0]] == -2) ++i0;
      if (first) {
        // compact: misses after i0 that were prefetched are done; drop them from the remaining list
        std::vector<int> m2, s2;
        for (size_t j = i0; j < miss.size(); ++j)
          if (pre[miss[j]] != -2) m2.push_back(miss[j]), s2.push_back(slot[j]);
        miss.assign(m2.begin(), m2.end());
        slot.assign(s2.begin(), s2.end());
        i0 = 0;
      }
      first = false;
    }
    stage_layer[h] = -1;  // (its records are queued for copying; the half is free after the next sync)
    if (dummy_rest) {
      const uint64_t any = (uint64_t)P.addr[0];
      for (int e = 0; e < NE; ++e) if (!seen[e]) xa[e] = any;
    }    last_l = l;
    last_seen = seen;
    if (std::chrono::steady_clock::now() - saved_at > std::chrono::seconds(60)) save_state();
  }
};
