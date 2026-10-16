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
#include <algorithm>
#include <cstdint>
#include <vector>

// ---- pools: the layers grouped by record size (pure host logic; tests/test_xcache_pools.cpp, no GPU)
// A pool's record is the largest record of its layers (a smaller record leaves the rest of its slot unused), and every
// pool holds at least one layer's experts (a prompt chunk uses all of them). With one or two record sizes every size
// has its own pool and the cache is split by the pools' share of the expert bytes (as before). With more (Qwen3.8
// Flash Next: 7 type mixes, Qwen3.6 35B: 4) one pool per size needs more memory than a small cache has, and one pool of
// the largest record wastes a fifth of it in padding: choose() tries every grouping of the sorted sizes into runs
// (2^(sizes-1)) whose pools fit the cache with one layer each, gives each the same fraction of its layers' experts,
// and keeps the one with the most slots (ties: fewer bytes, then fewer pools). One pool of the largest record is the
// fallback when nothing else fits.
namespace xpools {
struct Plan {
  std::vector<uint64_t> rec;  // pool -> record bytes
  std::vector<int> layers;    // pool -> MoE layers in it
  std::vector<int> pool_of;   // layer -> pool (-1: not a MoE layer)
  std::vector<int> n;         // pool -> slots
  uint64_t bytes = 0;         // sum of n * rec
  int slots = 0;
};
inline std::vector<uint64_t> sizes(const std::vector<uint64_t>& lrec) {  // distinct record sizes, ascending
  std::vector<uint64_t> d;
  for (uint64_t r : lrec) if (r) d.push_back(r);
  std::sort(d.begin(), d.end());
  d.erase(std::unique(d.begin(), d.end()), d.end());
  return d;
}
inline uint64_t real_bytes(const std::vector<uint64_t>& lrec, int NE) {  // every expert in a record of its own size
  uint64_t s = 0;
  for (uint64_t r : lrec) s += r * NE;
  return s;
}
// A layer goes to the pool of the smallest bound >= its record (bounds: ascending record sizes, the last = the
// largest); pools in the order of their first layer.
inline Plan group(const std::vector<uint64_t>& lrec, const std::vector<uint64_t>& bounds) {
  Plan P;
  P.pool_of.assign(lrec.size(), -1);
  std::vector<int> pb(bounds.size(), -1);
  for (size_t l = 0; l < lrec.size(); ++l) {
    if (!lrec[l]) continue;
    const size_t b = std::lower_bound(bounds.begin(), bounds.end(), lrec[l]) - bounds.begin();
    if (pb[b] < 0) { pb[b] = (int)P.rec.size(); P.rec.push_back(bounds[b]); P.layers.push_back(0); }
    P.pool_of[l] = pb[b];
    ++P.layers[pb[b]];
  }
  P.n.assign(P.rec.size(), 0);
  return P;
}
inline void tally(Plan& P) {
  P.bytes = 0;
  P.slots = 0;
  for (size_t g = 0; g < P.rec.size(); ++g) P.bytes += (uint64_t)P.n[g] * P.rec[g], P.slots += P.n[g];
}
// (one or two sizes) `bytes` split by the pools' share of the expert bytes, at least a layer per pool
inline void size_shares(Plan& P, int NE, uint64_t bytes) {
  uint64_t all = 0;
  for (size_t g = 0; g < P.rec.size(); ++g) all += P.rec[g] * NE * P.layers[g];
  bytes = std::min(bytes, all);
  for (size_t g = 0; g < P.rec.size(); ++g) {
    const uint64_t mine = P.rec[g] * NE * P.layers[g];
    P.n[g] = std::min((int)(mine / P.rec[g]), std::max(NE, (int)((double)bytes * mine / all / P.rec[g])));
  }
  tally(P);
}
// (more sizes) the same fraction of every pool's experts (at least a layer, at most all), the most that fits in
// `bytes`; the bytes left over as whole slots to the pools of the smallest records
inline void size_even(Plan& P, int NE, uint64_t bytes) {
  const size_t G = P.rec.size();
  auto fill = [&](double f) {
    uint64_t s = 0;
    for (size_t g = 0; g < G; ++g) {
      const int cap = NE * P.layers[g];
      P.n[g] = std::max(NE, std::min(cap, (int)(f * cap)));
      s += (uint64_t)P.n[g] * P.rec[g];
    }
    return s;
  };
  double lo = 0, hi = 1;
  if (fill(1.0) <= bytes) lo = 1;
  else for (int it = 0; it < 60; ++it) { const double m = (lo + hi) / 2; (fill(m) <= bytes ? lo : hi) = m; }
  uint64_t used = fill(lo);
  std::vector<int> order(G);
  for (size_t g = 0; g < G; ++g) order[g] = (int)g;
  std::sort(order.begin(), order.end(), [&](int a, int b) { return P.rec[a] < P.rec[b]; });
  for (int g : order) {
    if (used >= bytes) break;
    const uint64_t k = std::min<uint64_t>((uint64_t)NE * P.layers[g] - P.n[g], (bytes - used) / P.rec[g]);
    P.n[g] += (int)k;
    used += k * P.rec[g];
  }
  tally(P);
}
// Least bytes a cache takes: a layer for every pool of the exact grouping (<= 2 sizes), else for one pool of the
// largest record.
inline uint64_t min_bytes(const std::vector<uint64_t>& lrec, int NE) {
  const auto d = sizes(lrec);
  uint64_t s = 0;
  if (d.size() <= 2) for (uint64_t r : d) s += r * NE;
  else if (!d.empty()) s = d.back() * NE;
  return s;
}
inline Plan choose(const std::vector<uint64_t>& lrec, int NE, uint64_t bytes) {
  const auto d = sizes(lrec);
  if (d.size() <= 2) {
    Plan P = group(lrec, d);
    size_shares(P, NE, bytes);
    return P;
  }
  // (bit i of a mask: a run ends after size i; mask 0: one pool, always kept as the fallback)
  const int nm = d.size() > 20 ? 1 : 1 << (d.size() - 1);
  Plan best;
  for (int mask = 0; mask < nm; ++mask) {
    std::vector<uint64_t> bounds;
    for (size_t i = 0; i + 1 < d.size(); ++i) if (mask >> i & 1) bounds.push_back(d[i]);
    bounds.push_back(d.back());
    uint64_t need = 0;
    for (uint64_t b : bounds) need += b * NE;
    if (mask && need > bytes) continue;
    Plan P = group(lrec, bounds);
    size_even(P, NE, bytes);
    if (best.rec.empty() || P.slots > best.slots ||
        (P.slots == best.slots && (P.bytes < best.bytes || (P.bytes == best.bytes && P.rec.size() < best.rec.size()))))
      best = std::move(P);
  }
  return best;
}
}  // namespace xpools

#ifndef XCACHE_POOLS_ONLY
#include "expert_store.h"
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
    lrec.assign(NL, 0);
    for (int l = 0; l < NL; ++l) {
      lrec[l] = l < (int)st.layers.size() ? st.layers[l].rec : 0;
      stage_rec = std::max(stage_rec, lrec[l]);
    }
    // The pools until set_cache picks the grouping for the cache size (xpools::choose): one per record size with one
    // or two sizes (final), else one pool of the largest record (min_bytes: the least that always fits).
    const auto d = xpools::sizes(lrec);
    use(xpools::group(lrec, d.size() <= 2 ? d : std::vector<uint64_t>(1, stage_rec)));
    stage_bytes = 2 * stage_rec * S;
    for (int h = 0; h < 2; ++h) {
      stage[h] = (char*)zrt::host_alloc(stage_rec * S);
      stage_e[h].clear();
    }
  }
  std::vector<uint64_t> lrec;  // layer -> its record bytes (0: not a MoE layer)
  void use(const xpools::Plan& G) {  // (pools without slots yet)
    pools.assign(G.rec.size(), Pool{});
    for (size_t p = 0; p < pools.size(); ++p) pools[p].rec = G.rec[p];
    pool_of = G.pool_of;
  }
  // every expert of every layer, each in a record of its own size: the cache that holds them all (set_cache then
  // keeps one pool per size)
  uint64_t all_bytes() const { return xpools::real_bytes(lrec, NE); }
  // Least bytes set_cache takes: every pool holds at least one layer's experts (a prompt chunk uses all of them).
  uint64_t min_bytes() const {
    uint64_t s = 0;
    for (auto& P : pools) s += P.rec * NE;
    return s;
  }
  // The pools for `bytes` (xpools::choose: with one or two record sizes the pools of init split by their share of the
  // experts; with more the grouping with the most slots), allocated as device USM in pieces of <= 1 GiB.
  void set_cache(uint64_t bytes) {
    const xpools::Plan G = xpools::choose(lrec, NE, bytes);
    if (xpools::sizes(lrec).size() > 2) {
      use(G);
      const double GiB = 1073741824.0;
      uint64_t pad = 0;
      for (int l = 0; l < NL; ++l) if (lrec[l]) pad += (G.rec[G.pool_of[l]] - lrec[l]) * NE;
      fprintf(stderr, "arcinferzero: %zu expert record sizes in %zu pools (records", xpools::sizes(lrec).size(), G.rec.size());
      for (size_t p = 0; p < G.rec.size(); ++p) fprintf(stderr, " %.2f MB x %d layers", G.rec[p] / 1e6, G.layers[p]);
      fprintf(stderr, "), padding %.2f GiB of %.2f GiB experts\n", pad / GiB, all_bytes() / GiB);
    }
    cache_bytes = 0;
    for (size_t p = 0; p < pools.size(); ++p) {
      Pool& P = pools[p];
      P.n = G.n[p];
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
    // (the saved pools' key lists, any number of them: the grouping follows the cache size, so it may differ from
    // the saving run's; pool p takes the saved list p's keys first, then the other lists', each key into its pool)
    std::vector<std::vector<int>> want;
    if (FILE* f = state_path.empty() ? nullptr : fopen(state_path.c_str(), "rb")) {
      int hdr[3] = {};
      if (fread(hdr, 4, 3, f) == 3 && hdr[0] == NL && hdr[1] == NE && hdr[2] > 0 && hdr[2] <= 64) want.resize(hdr[2]);
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
      std::vector<uint8_t> have((size_t)NL * NE, 0);
      std::vector<int> keep;
      for (size_t q0 = 0; q0 < want.size(); ++q0) {
        const size_t q = q0 == 0 ? std::min(p, want.size() - 1) : q0 <= std::min(p, want.size() - 1) ? q0 - 1 : q0;
        for (int k : want[q])  // (saved keys that belong to this pool)
          if (k >= 0 && k < NL * NE && pool_of[k / NE] == (int)p && !have[k] && (int)keep.size() < P.n) keep.push_back(k), have[k] = 1;
      }
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

  // (generation) the experts `ids` of layer X that are not cached, into stage[X % 2] on a thread: a guess of the next
  // layer's routing, read while the GPU works; ensure() then copies the ones it needs from there
  uint64_t g_pf = 0, g_pf_used = 0;
  void prefetch_ids(int X, const int* ids, int n) {
    join();
    if (X < 0 || X >= NL || pool_of[X] < 0) return;
    const int h = X % 2;
    stage_layer[h] = X;
    stage_e[h].clear();
    pf_reqs_.clear();
    for (int i = 0; i < n && (int)stage_e[h].size() < S; ++i) {
      const int e = ids[i];
      if (e < 0 || e >= NE || where[(size_t)X * NE + e] >= 0) continue;
      if (std::find(stage_e[h].begin(), stage_e[h].end(), e) != stage_e[h].end()) continue;
      pf_reqs_.push_back({X, e, stage[h] + stage_e[h].size() * stage_rec});
      stage_e[h].push_back(e);
    }
    if (pf_reqs_.empty()) return;
    g_pf += pf_reqs_.size();
    pf_ = std::thread([this] {
      const auto t0 = std::chrono::steady_clock::now();
      st.read(pf_reqs_);
      pf_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    });
  }

  // Makes the experts `ids` (any order, repeats allowed) of layer l readable by the kernels; prompt = a prompt chunk.
  // dummy_rest: also point the layer's other experts at valid memory (the prompt GEMM decodes whole expert groups).
  // expert e of layer l is in the cache: its record's address goes to xaddr (where a prompt chunk's GEMM may have left
  // a dummy one) and it stays there through the next ensure() of this layer, which takes the hits first
  bool cached(int l, int e) {
    if (e < 0 || e >= NE) return false;
    const int s = where[(size_t)l * NE + e];
    if (s < 0) return false;
    xaddr[(size_t)l * NE + e] = (uint64_t)pools[pool_of[l]].addr[s];
    return true;
  }
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
    if (!prompt && stage_layer[h] == l)
      for (int e : miss) g_pf_used += pre[e] >= 0;
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
#endif  // XCACHE_POOLS_ONLY
