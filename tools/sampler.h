// Token sampling of tools/server.cpp (its own header so that tests/test_sampler.cpp can test it on the CPU alone):
// temperature, top_k, top_p, min_p, the OpenAI presence / frequency penalties and logit_bias, llama.cpp's
// repeat_penalty, and constrained output (sample_allowed: only tokens a predicate allows).
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

struct SamplingParams {
  float temperature = 1.0f, top_p = 0.95f, min_p = 0.0f, presence_penalty = 0.0f;
  // OpenAI frequency_penalty: subtracted once per earlier occurrence in the generated text; presence_penalty:
  // subtracted once from every token that occurs in it; llama.cpp repeat_penalty: a positive logit divided by it, a
  // negative one multiplied, for tokens among the last repeat_last_n generated (-1: all of them, 0: none)
  float frequency_penalty = 0.0f, repeat_penalty = 1.0f;
  int repeat_last_n = 64;
  int top_k = 20;
  uint64_t seed = 0;
  // OpenAI logit_bias: token id -> added to its logit; -100 or less bans the token (never drawn, greedy included)
  std::vector<std::pair<int, float>> logit_bias;
};

class Sampler {
 public:
  // the logit of a banned token (logit_bias <= -100), set after every penalty; sample() and sample_allowed() never
  // pick it while any other token is left
  static constexpr float kBanned = -1e30f;
  explicit Sampler(const SamplingParams& p) : p_(p), rng_(p.seed ? p.seed : std::random_device{}()) {
    for (const auto& [t, b] : p_.logit_bias) bans_ = bans_ || b <= -100.0f;
  }
  int sample(const std::vector<float>& raw, const std::vector<int>& recent) {
    const std::vector<float>& logits = adjusted(raw, recent);
    if (p_.temperature <= 0.0f) return (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
    const int V = (int)logits.size();
    int k = p_.top_k > 0 ? std::min<int>(p_.top_k, V) : 256;
    std::vector<std::pair<float, int>> c;
    // Only the top k logits (penalties included) are drawn from. Candidates: tokens within 40 temperatures of the
    // best logit. When at least k of them are found, they contain the exact top k: the result is the same as sorting
    // the whole vocabulary, without building a pair for each of its ~262K entries per token.
    const float best = *std::max_element(logits.begin(), logits.end()), thr = best - 40.0f * p_.temperature;
    for (int i = 0; i < V; ++i)
      if (logits[i] >= thr) c.push_back({logits[i], i});
    if ((int)c.size() < k) {  // not enough: the whole vocabulary
      c.clear();
      c.reserve(V);
      for (int i = 0; i < V; ++i) c.push_back({logits[i], i});
    }
    std::partial_sort(c.begin(), c.begin() + k, c.end(), [](auto& a, auto& b) { return a.first > b.first; });
    if (bans_)  // (banned tokens are in the top k only when few others are: never drawn)
      while (k > 1 && c[k - 1].first <= kBanned) --k;
    c.resize(k);
    const float mx = c[0].first;
    double z = 0;
    std::vector<double> pr(k);
    for (int i = 0; i < k; ++i) z += pr[i] = std::exp((c[i].first - mx) / p_.temperature);
    int n = k;
    double cum = 0;
    for (int i = 0; i < k; ++i) {
      pr[i] /= z;
      if (p_.min_p > 0 && pr[i] < p_.min_p * pr[0]) { n = std::max(1, i); break; }
      cum += pr[i];
      if (cum >= p_.top_p) { n = i + 1; break; }
    }
    double tot = 0;
    for (int i = 0; i < n; ++i) tot += pr[i];
    std::uniform_real_distribution<double> u(0.0, tot);
    double r = u(rng_);
    for (int i = 0; i < n; ++i) {
      r -= pr[i];
      if (r <= 0) return c[i].second;
    }
    return c[n - 1].second;
  }
  // A token the predicate allows (constrained output), drawn like sample() from the allowed ones: candidates are
  // tested from the most likely down until top_k (1 when greedy) are found. -1 when no token is allowed. Banned
  // tokens are never tested.
  int sample_allowed(const std::vector<float>& raw, const std::vector<int>& recent, const std::function<bool(int)>& ok) {
    const std::vector<float>& logits = adjusted(raw, recent);
    const int V = (int)logits.size();
    if (V == 0) return -1;
    const int need = p_.temperature <= 0.0f ? 1 : (p_.top_k > 0 ? std::min(p_.top_k, V) : 40);
    auto gt = [&](int a, int b) { return logits[a] > logits[b]; };
    // The `head` most likely tokens from one pass over the logits, without an index array of the whole vocabulary.
    // The order (ties included) and so the predicate's calls are those of std::partial_sort over such an array
    // (head, then the rest sorted when they hold too few allowed tokens), the earlier implementation: the same heap
    // steps as MSVC's partial_sort (make_heap on the first head, each later token that beats the top replaces it,
    // sort_heap), with the replaced tops recorded so that the rest can be rebuilt as it would have been.
    const int head = std::min(V, 512);
    heap_.resize(head);
    for (int i = 0; i < head; ++i) heap_[i] = i;
    std::make_heap(heap_.begin(), heap_.end(), gt);
    out_.clear();
    for (int i = head; i < V; ++i)
      if (gt(i, heap_[0])) {
        out_.push_back({i, heap_[0]});
        replace_top(i, gt);
      }
    std::sort_heap(heap_.begin(), heap_.end(), gt);
    std::vector<int> c;
    for (int r = 0; r < V && (int)c.size() < need; ++r) {
      if (r == head) {  // too few allowed among them: the rest of the vocabulary, sorted (scratch buffer)
        idx_.resize(V);
        for (int i = head; i < V; ++i) idx_[i] = i;
        for (const auto& [at, t] : out_) idx_[at] = t;
        std::sort(idx_.begin() + head, idx_.end(), gt);
      }
      const int t = r < head ? heap_[r] : idx_[r];
      if (bans_ && logits[t] <= kBanned) continue;
      if (ok(t)) c.push_back(t);
    }
    if (c.empty()) return -1;
    if (p_.temperature <= 0.0f || c.size() == 1) return c[0];
    const float mx = logits[c[0]];
    std::vector<double> pr(c.size());
    double z = 0;
    for (size_t i = 0; i < c.size(); ++i) z += pr[i] = std::exp((logits[c[i]] - mx) / p_.temperature);
    size_t n = c.size();
    double cum = 0;
    for (size_t i = 0; i < c.size(); ++i) {
      pr[i] /= z;
      if (p_.min_p > 0 && pr[i] < p_.min_p * pr[0]) { n = std::max<size_t>(1, i); break; }
      cum += pr[i];
      if (cum >= p_.top_p) { n = i + 1; break; }
    }
    double tot = 0;
    for (size_t i = 0; i < n; ++i) tot += pr[i];
    std::uniform_real_distribution<double> u(0.0, tot);
    double r = u(rng_);
    for (size_t i = 0; i < n; ++i) {
      r -= pr[i];
      if (r <= 0) return c[i];
    }
    return c[n - 1];
  }
  // The logits after the logit bias, the frequency / repeat / presence penalties and the bans (the input itself when
  // none is set)
  const std::vector<float>& adjusted(const std::vector<float>& logits, const std::vector<int>& recent) {
    const bool fp = p_.frequency_penalty != 0.0f, rp = p_.repeat_penalty != 1.0f && p_.repeat_penalty > 0.0f &&
                                                       p_.repeat_last_n != 0;
    const bool pp = p_.presence_penalty != 0.0f;
    if (!fp && !rp && !pp && p_.logit_bias.empty()) return logits;
    adj_ = logits;
    const int V = (int)adj_.size();
    for (const auto& [t, b] : p_.logit_bias)
      if (t >= 0 && t < V && b > -100.0f) adj_[t] += b;
    if (fp) {
      std::unordered_map<int, int> cnt;
      for (int t : recent) ++cnt[t];
      for (const auto& [t, n] : cnt)
        if (t >= 0 && t < V) adj_[t] -= p_.frequency_penalty * (float)n;
    }
    if (rp) {
      const size_t n = p_.repeat_last_n < 0 ? recent.size() : std::min(recent.size(), (size_t)p_.repeat_last_n);
      std::unordered_set<int> seen(recent.end() - n, recent.end());
      for (int t : seen)
        if (t >= 0 && t < V) adj_[t] = adj_[t] > 0.0f ? adj_[t] / p_.repeat_penalty : adj_[t] * p_.repeat_penalty;
    }
    if (pp) {  // once per token in the text frequency_penalty counts (here, so that greedy decoding has it too)
      std::unordered_set<int> seen(recent.begin(), recent.end());
      for (int t : seen)
        if (t >= 0 && t < V) adj_[t] -= p_.presence_penalty;
    }
    for (const auto& [t, b] : p_.logit_bias)  // (last: no penalty lifts a banned token)
      if (t >= 0 && t < V && b <= -100.0f) adj_[t] = kBanned;
    return adj_;
  }

 private:
  // MSVC's partial_sort step for a token at `t` that beats the heap's top (_Pop_heap_hole_unchecked): the hole at
  // the top moves down to the larger child down to a leaf, then t moves up from there to where it belongs
  template <class Gt>
  void replace_top(int t, const Gt& gt) {
    const int bottom = (int)heap_.size(), last_parent = (bottom - 1) >> 1;
    int hole = 0, i = 0;
    while (i < last_parent) {
      i = 2 * i + 2;
      if (gt(heap_[i], heap_[i - 1])) --i;
      heap_[hole] = heap_[i];
      hole = i;
    }
    if (i == last_parent && bottom % 2 == 0) {
      heap_[hole] = heap_[bottom - 1];
      hole = bottom - 1;
    }
    for (int p = (hole - 1) >> 1; 0 < hole && gt(heap_[p], t); p = (hole - 1) >> 1) {
      heap_[hole] = heap_[p];
      hole = p;
    }
    heap_[hole] = t;
  }
  SamplingParams p_;
  std::mt19937_64 rng_;
  bool bans_ = false;
  std::vector<float> adj_;
  std::vector<int> heap_, idx_;               // sample_allowed: the most likely tokens; the rest, when needed
  std::vector<std::pair<int, int>> out_;      // (position, token) of each top the heap replaced
};
