// CPU-only tests of the token sampler (tools/sampler.h): greedy and seeded sampling unchanged with the default
// parameters, the presence penalty in greedy decoding, logit_bias -100 as a ban, and sample_allowed (constrained
// output) choosing exactly as the implementation before it (kept below as OldSampler, the reference).
// Build: .\build.ps1 tests\test_sampler.cpp, run: .\bin\test_sampler.exe
#include "../tools/sampler.h"
#include <cstdio>
#include <set>

static int fails = 0;
static void expect(bool ok, const char* what) {
  printf("%s %s\n", ok ? "ok  " : "FAIL", what);
  if (!ok) ++fails;
}

// ---- the reference: class Sampler of tools/server.cpp before the change (commit 2112264), verbatim but its name
class OldSampler {
 public:
  explicit OldSampler(const SamplingParams& p) : p_(p), rng_(p.seed ? p.seed : std::random_device{}()) {}
  int sample(const std::vector<float>& raw, const std::vector<int>& recent) {
    const std::vector<float>& logits = adjusted(raw, recent);
    if (p_.temperature <= 0.0f) return (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
    const int V = (int)logits.size(), k = p_.top_k > 0 ? std::min<int>(p_.top_k, V) : 256;
    const float pp = p_.presence_penalty;
    std::vector<char> seen;
    if (pp != 0.0f) {
      seen.assign(V, 0);
      for (int t : recent)
        if (t >= 0 && t < V) seen[t] = 1;
    }
    std::vector<std::pair<float, int>> c;
    auto penalize = [&] {
      if (pp != 0.0f)
        for (auto& e : c)
          if (seen[e.second]) e.first -= pp;
    };
    // Only the top k (penalized) logits are drawn from. Candidates: tokens within 40 temperatures of the best logit
    // (widened by a negative penalty, a boost). Every other token ends below that threshold after its penalty, so
    // when at least k candidates stay at or above it, they contain the exact top k: the result is the same as
    // sorting the whole vocabulary, without building a pair for each of its ~262K entries per token.
    const float best = *std::max_element(logits.begin(), logits.end()), thr = best - 40.0f * p_.temperature;
    const float cut = thr - std::max(0.0f, -pp);
    for (int i = 0; i < V; ++i)
      if (logits[i] >= cut) c.push_back({logits[i], i});
    penalize();
    int above = 0;
    for (const auto& e : c) above += e.first >= thr;
    if (above < k) {  // not enough: the whole vocabulary
      c.clear();
      c.reserve(V);
      for (int i = 0; i < V; ++i) c.push_back({logits[i], i});
      penalize();
    }
    std::partial_sort(c.begin(), c.begin() + k, c.end(), [](auto& a, auto& b) { return a.first > b.first; });
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
  // tested from the most likely down until top_k (1 when greedy) are found. -1 when no token is allowed.
  int sample_allowed(const std::vector<float>& raw, const std::vector<int>& recent, const std::function<bool(int)>& ok) {
    const std::vector<float>& logits = adjusted(raw, recent);
    const int V = (int)logits.size();
    const int need = p_.temperature <= 0.0f ? 1 : (p_.top_k > 0 ? std::min(p_.top_k, V) : 40);
    std::vector<char> seen;
    if (p_.presence_penalty != 0.0f) {
      seen.assign(V, 0);
      for (int t : recent)
        if (t >= 0 && t < V) seen[t] = 1;
    }
    auto val = [&](int i) { return logits[i] - (seen.empty() || !seen[i] ? 0.0f : p_.presence_penalty); };
    std::vector<int> idx(V);
    for (int i = 0; i < V; ++i) idx[i] = i;
    const int head = std::min(V, 512);
    std::partial_sort(idx.begin(), idx.begin() + head, idx.end(), [&](int a, int b) { return val(a) > val(b); });
    std::vector<int> c;
    for (int r = 0; r < V && (int)c.size() < need; ++r) {
      if (r == head) std::sort(idx.begin() + head, idx.end(), [&](int a, int b) { return val(a) > val(b); });
      if (ok(idx[r])) c.push_back(idx[r]);
    }
    if (c.empty()) return -1;
    if (p_.temperature <= 0.0f || c.size() == 1) return c[0];
    const float mx = val(c[0]);
    std::vector<double> pr(c.size());
    double z = 0;
    for (size_t i = 0; i < c.size(); ++i) z += pr[i] = std::exp((val(c[i]) - mx) / p_.temperature);
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
  // The logits after the frequency / repeat penalties and the logit bias (the input itself when none is set; the
  // presence penalty is applied in sample() on the candidates only)
  const std::vector<float>& adjusted(const std::vector<float>& logits, const std::vector<int>& recent) {
    const bool fp = p_.frequency_penalty != 0.0f, rp = p_.repeat_penalty != 1.0f && p_.repeat_penalty > 0.0f &&
                                                       p_.repeat_last_n != 0;
    if (!fp && !rp && p_.logit_bias.empty()) return logits;
    adj_ = logits;
    const int V = (int)adj_.size();
    for (const auto& [t, b] : p_.logit_bias)
      if (t >= 0 && t < V) adj_[t] += b;
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
    return adj_;
  }

 private:
  SamplingParams p_;
  std::mt19937_64 rng_;
  std::vector<float> adj_;
};

// ---- random inputs
// mode 0: continuous logits; 1: a few levels (many exact ties); 2: mostly one value with some higher ones
static std::vector<float> rand_logits(std::mt19937_64& g, int V, int mode) {
  std::normal_distribution<float> nd(0.0f, 3.0f);
  std::uniform_int_distribution<int> lv(0, 7), pick(0, 99);
  std::vector<float> l(V);
  for (int i = 0; i < V; ++i) {
    if (mode == 0) l[i] = nd(g);
    else if (mode == 1) l[i] = 0.5f * (float)lv(g);
    else l[i] = pick(g) < 3 ? 1.0f + 0.25f * (float)lv(g) : -2.0f;
  }
  return l;
}
static std::vector<int> rand_recent(std::mt19937_64& g, int V) {
  std::uniform_int_distribution<int> n(0, 80), t(0, V - 1);
  std::vector<int> r(n(g));
  for (auto& x : r) x = t(g);
  return r;
}
// random sampling parameters; penalties / bias only when asked (default: none of them)
static SamplingParams rand_params(std::mt19937_64& g, int V, bool penalties) {
  SamplingParams p;
  const float temps[] = {0.0f, 0.3f, 0.7f, 1.0f, 1.5f};
  const int ks[] = {1, 5, 20, 40, 0};
  const float tps[] = {0.8f, 0.95f, 1.0f};
  const float mps[] = {0.0f, 0.0f, 0.05f};
  p.temperature = temps[g() % 5];
  p.top_k = ks[g() % 5];
  if (p.top_k == 0 && V < 256) p.top_k = 20;  // (top_k 0 draws from 256: needs that many tokens)
  p.top_p = tps[g() % 3];
  p.min_p = mps[g() % 3];
  p.seed = g() | 1;
  if (penalties) {
    if (g() % 2) p.frequency_penalty = 0.3f;
    if (g() % 2) p.repeat_penalty = 1.1f, p.repeat_last_n = (g() % 2) ? -1 : 64;
    if (g() % 2) p.presence_penalty = (g() % 2) ? 0.7f : -0.4f;
    if (g() % 2) p.logit_bias.push_back({(int)(g() % V), 2.5f});
    if (g() % 2) p.logit_bias.push_back({(int)(g() % V), -7.0f});
  }
  return p;
}

int main() {
  std::mt19937_64 g(12345);
  // ---- greedy, default parameters: the argmax, as before
  {
    bool same = true;
    for (int c = 0; c < 300; ++c) {
      const int V = 2 + (int)(g() % 5000);
      const auto l = rand_logits(g, V, c % 3);
      const auto rec = rand_recent(g, V);
      SamplingParams p;
      p.temperature = 0.0f;
      Sampler s(p);
      OldSampler o(p);
      const int a = s.sample(l, rec), b = o.sample(l, rec);
      same = same && a == b && a == (int)(std::max_element(l.begin(), l.end()) - l.begin());
    }
    expect(same, "greedy, default parameters: the same argmax as before (300 cases)");
  }
  // ---- greedy with presence_penalty
  {
    std::vector<float> l(100, 0.0f);
    l[0] = 5.0f, l[1] = 4.9f, l[2] = 4.0f;
    SamplingParams p;
    p.temperature = 0.0f;
    p.presence_penalty = 0.5f;
    Sampler s(p);
    OldSampler o(p);
    expect(o.sample(l, {0, 0, 7}) == 0, "greedy + presence_penalty: the old sampler ignored it (reference)");
    expect(s.sample(l, {0, 0, 7}) == 1, "greedy + presence_penalty: a token already generated loses the argmax");
    // (both penalized once: 4.5 against 4.4; per occurrence, 0 would end at 3.5)
    expect(s.sample(l, {0, 0, 0, 1}) == 0, "greedy + presence_penalty: once per token, not per occurrence");
    expect(s.sample(l, {3}) == 0, "greedy + presence_penalty: other tokens unaffected");
    p.presence_penalty = 0.05f;
    Sampler s2(p);
    expect(s2.sample(l, {0}) == 0, "greedy + small presence_penalty: argmax kept");
    p.presence_penalty = -2.0f;  // (negative: a boost)
    Sampler s3(p);
    expect(s3.sample(l, {2}) == 2, "greedy + negative presence_penalty: a generated token is boosted");
    p.presence_penalty = 0.5f;
    Sampler s4(p);
    expect(s4.sample_allowed(l, {0}, [](int) { return true; }) == 1, "greedy sample_allowed + presence_penalty");
  }
  // ---- logit_bias -100: banned
  {
    const int V = 1000;
    std::vector<float> l = rand_logits(g, V, 0);
    l[3] = 1000.0f;  // (far above every other: subtracting 100 alone would keep it the argmax)
    SamplingParams p;
    p.temperature = 0.0f;
    p.logit_bias.push_back({3, -100.0f});
    {
      OldSampler o(p);
      expect(o.sample(l, {}) == 3, "logit_bias -100: the old sampler still chose it (reference)");
      Sampler s(p);
      expect(s.sample(l, {}) != 3, "logit_bias -100: never the greedy choice");
      expect(s.sample_allowed(l, {}, [](int t) { return t == 3 || t == 7; }) == 7, "logit_bias -100: greedy sample_allowed skips it");
      expect(s.sample_allowed(l, {}, [](int t) { return t == 3; }) == -1, "logit_bias -100: the only allowed token banned: -1");
    }
    p.logit_bias[0].second = -250.0f;
    p.presence_penalty = -5000.0f;  // (a boost: still banned)
    {
      Sampler s(p);
      expect(s.sample(l, {3}) != 3, "logit_bias -250 with a presence boost: still banned");
    }
    p.presence_penalty = 0.0f;
    p.logit_bias[0].second = -100.0f;
    p.temperature = 1.0f;
    p.top_k = 20;
    p.seed = 99;
    {
      Sampler s(p);
      bool never = true;
      for (int i = 0; i < 2000; ++i) {
        never = never && s.sample(l, {}) != 3;
        never = never && s.sample_allowed(l, {}, [](int t) { return t < 30; }) != 3;
      }
      expect(never, "logit_bias -100: never drawn when sampling (2000 draws)");
    }
    {  // (most of the vocabulary banned: banned tokens reach the top k, still never drawn)
      SamplingParams q;
      q.temperature = 1.5f;
      q.top_k = 20;
      q.top_p = 1.0f;
      q.seed = 7;
      for (int t = 0; t < 290; ++t) q.logit_bias.push_back({t, -100.0f});
      const auto l2 = rand_logits(g, 300, 0);
      Sampler s(q);
      bool never = true;
      for (int i = 0; i < 3000; ++i) never = never && s.sample(l2, {}) >= 290;
      expect(never, "logit_bias -100 on 290 of 300 tokens: only the 10 others drawn");
      q.temperature = 0.0f;
      Sampler s2(q);
      expect(s2.sample(l2, {}) >= 290, "logit_bias -100 on 290 of 300 tokens: greedy");
    }
    {  // +100: a large positive bias, as before
      SamplingParams q;
      q.temperature = 0.0f;
      q.logit_bias.push_back({5, 100.0f});
      Sampler s(q);
      OldSampler o(q);
      expect(s.sample(l, {}) == o.sample(l, {}), "logit_bias +100: as before");
      std::vector<float> l3(50, 0.0f);
      l3[9] = 50.0f;
      expect(s.sample(l3, {}) == 5, "logit_bias +100: chosen over a logit 50 higher");
    }
  }
  // ---- seeded sampling: as before (default parameters, then with penalties and non-banning biases)
  for (int pen = 0; pen < 2; ++pen) {
    int bad = 0, n = 0;
    for (int c = 0; c < 400; ++c) {
      const int V = c % 50 == 0 ? 248320 : 64 + (int)(g() % 6000);
      // (with penalties: continuous logits only; with the presence penalty, the order among exactly equal logits
      //  may differ from before, see tools/sampler.h)
      const auto l = rand_logits(g, V, pen ? 0 : c % 3);
      SamplingParams p = rand_params(g, V, pen == 1);
      if (p.temperature <= 0.0f) p.presence_penalty = 0.0f;  // (greedy + presence_penalty: changed on purpose, above)
      Sampler s(p);
      OldSampler o(p);
      std::vector<int> rec = rand_recent(g, V);
      for (int d = 0; d < 6; ++d, ++n) {  // (several draws: the random generators stay in step)
        const int a = s.sample(l, rec), b = o.sample(l, rec);
        bad += a != b;
        rec.push_back(a);
      }
    }
    char what[160];
    snprintf(what, sizeof what, "sample(): %s: the same tokens as before for the same seed (%d draws, %d differ)",
             pen ? "with penalties / bias" : "default parameters", n, bad);
    expect(bad == 0, what);
  }
  // ---- sample_allowed: the same token and the same predicate calls as before
  for (int pen = 0; pen < 2; ++pen) {
    int bad = 0, badcalls = 0, n = 0, wide = 0, tails = 0;
    for (int c = 0; c < 1500; ++c) {
      const int V = c % 100 == 0 ? 248320 : c % 7 == 0 ? 1 + (int)(g() % 600) : 64 + (int)(g() % 20000);
      const auto l = rand_logits(g, V, c % 3);
      SamplingParams p = rand_params(g, V, pen == 1);
      const int dens[] = {1000, 500, 50, 5, 1, 0};  // allowed per 1000 tokens
      const int den = dens[g() % 6];
      const uint64_t salt = g();
      auto allowed = [&](int t) { return (std::mt19937_64(salt ^ (uint64_t)t)() % 1000) < (uint64_t)den; };
      std::vector<int> calls_new, calls_old;
      Sampler s(p);
      OldSampler o(p);
      std::vector<int> rec = rand_recent(g, V);
      for (int d = 0; d < 3; ++d, ++n) {
        calls_new.clear();
        calls_old.clear();
        const int a = s.sample_allowed(l, rec, [&](int t) { calls_new.push_back(t); return allowed(t); });
        const int b = o.sample_allowed(l, rec, [&](int t) { calls_old.push_back(t); return allowed(t); });
        bad += a != b;
        // (a negligible tail is no longer tested: then the calls stop after the first 512, a prefix of the old ones)
        const bool cut = calls_new.size() < calls_old.size() && calls_new.size() <= 512 &&
                         std::equal(calls_new.begin(), calls_new.end(), calls_old.begin());
        badcalls += calls_new != calls_old && !cut;
        tails += cut;
        wide += calls_old.size() > 512;
        if (a >= 0) rec.push_back(a);
      }
    }
    char what[260];
    snprintf(what, sizeof what, "sample_allowed(): %s: the same tokens and predicate calls as before (%d draws, %d "
             "past the first 512, %d of them a negligible tail skipped; %d tokens, %d call sequences differ)",
             pen ? "with penalties / bias" : "default parameters", n, wide, tails, bad, badcalls);
    expect(bad == 0 && badcalls == 0, what);
  }
  if (fails) {
    printf("%d FAILED\n", fails);
    return 1;
  }
  printf("ALL OK\n");
  return 0;
}
