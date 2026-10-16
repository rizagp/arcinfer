// arcinfer-server: OpenAI-compatible HTTP server (chat completions with streaming, reasoning and tool calls),
// prefix reuse across requests, and a built-in web UI.
// usage: arcinfer-server -m model.gguf [--host 127.0.0.1] [--port 8080] [--ctx 40960] [--alias name] [--webui dir]
//                       [--chunk N] [--budget GB] [--mtp] [--think-budget N] [--attn-f32] [--kernels dir]
//                       [--session-cache-gb N] [--session-dir dir]   (saved conversations; default 0 = off)
// ARC_ZERO (zero/serverz.cpp includes this file with it defined): the arcinferzero server, on zero/engine.h (Level
// Zero, MoE experts streamed from the SSD), with its own data folder (%LOCALAPPDATA%\arcinfer\zero: context
// cache, saved prompts and conversations) and the expert cache options.
#ifdef ARC_ZERO
#include "../zero/engine.h"
#include "../zero/gguf_pack.h"
#define ARC_DATA "/arcinfer/zero/"
#else
#include "../src/engine.h"
#define ARC_DATA "/arcinfer/"
#endif
#include "../src/tokenizer.h"
#include "../src/chat.h"
#include "../src/chat_gemma4.h"
#include "../src/chat_jinja.h"
#include "../src/image.h"
#include "../src/audio.h"
#include "../src/audio_gemma4.h"
#include "../src/vision_gemma4.h"
#ifdef ARC_ZERO
#include "../zero/vision_gemma4v.h"
#include "../zero/audio_gemma4a.h"
#include "../zero/vision_qwen3vl.h"
#else
#include "../src/vision_gemma4v.h"
#include "../src/audio_gemma4a.h"
#include "../src/vision_qwen3vl.h"
#endif
#include "../src/http.h"
#include "../src/full_speed.h"
#include "api_compat.h"
#include "json_constraint.h"
#include "tool_constraint.h"
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <shared_mutex>
#include <random>
#include <sstream>

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t) { return std::chrono::duration<double, std::milli>(clk::now() - t).count(); }

// What a model takes as input, in the fields clients look for: llama-server ("modalities", and "capabilities" with
// "multimodal" as in its Ollama-style list), Ollama ("vision"), OpenRouter ("architecture.input_modalities"),
// LM Studio ("type": "vlm"). Only models whose image encoder arcinfer runs are marked (Gemma 4 12B, gemma4uv).
// llama-server's own model list next to "data": a "models" array (Ollama style) whose "capabilities" hold
// "multimodal" for a model with an image encoder. Clients that treat arcinfer as a llama.cpp server read it there.
static Json llama_models(const Json& data) {
  Json out = Json::array();
  for (const Json& m : data.items()) {
    Json e = Json::object();
    const std::string id = m["id"].str();
    e.set("name", id);
    e.set("model", id);
    e.set("modified_at", "");
    e.set("size", "");
    e.set("digest", "");
    e.set("type", "model");
    e.set("description", "");
    Json tags = Json::array();
    tags.push("");
    e.set("tags", tags);
    Json caps = Json::array();
    caps.push("completion");
    if (m["modalities"]["vision"].is_bool() && m["modalities"]["vision"].as_bool()) caps.push("multimodal");
    e.set("capabilities", caps);
    e.set("parameters", "");
    Json det = Json::object();
    det.set("parent_model", "");
    det.set("format", "gguf");
    det.set("family", "");
    Json fam = Json::array();
    fam.push("");
    det.set("families", fam);
    det.set("parameter_size", "");
    det.set("quantization_level", "");
    e.set("details", det);
    out.push(e);
  }
  return out;
}

static void set_modalities(Json& j, bool vision) {
  Json caps = Json::array();
  caps.push("completion");
  caps.push("tools");
  if (vision) {
    caps.push("vision");
    caps.push("multimodal");
  }
  j.set("capabilities", caps);
  Json in = Json::array();
  in.push("text");
  if (vision) in.push("image");
  Json out = Json::array();
  out.push("text");
  Json arch = Json::object();
  arch.set("input_modalities", in);
  arch.set("output_modalities", out);
  j.set("architecture", arch);
  Json mod = Json::object();
  mod.set("vision", vision);
  mod.set("audio", false);
  j.set("modalities", mod);
  j.set("type", vision ? "vlm" : "llm");
}

struct SamplingParams {
  float temperature = 1.0f, top_p = 0.95f, min_p = 0.0f, presence_penalty = 0.0f;
  // OpenAI frequency_penalty: subtracted once per earlier occurrence in the generated text; llama.cpp
  // repeat_penalty: a positive logit divided by it, a negative one multiplied, for tokens among the last
  // repeat_last_n generated (-1: all of them, 0: none)
  float frequency_penalty = 0.0f, repeat_penalty = 1.0f;
  int repeat_last_n = 64;
  int top_k = 20;
  uint64_t seed = 0;
  std::vector<std::pair<int, float>> logit_bias;  // OpenAI logit_bias: token id -> added to its logit (-100: never)
};

class Sampler {
 public:
  explicit Sampler(const SamplingParams& p) : p_(p), rng_(p.seed ? p.seed : std::random_device{}()) {}
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

class Server {
 public:
  ~Server() {  // background writes (saved conversations, system prompt) finish first
    for (auto& s : saved_) written(s);
    prefix_written();
  }
  Server(Engine& e, Tokenizer& t, std::string id, std::string webui) : eng_(e), tok_(t), model_id_(std::move(id)), webui_(std::move(webui)) {
    const auto& g = e.gguf();
    q38_ = g.get_str("tokenizer.chat_template").find("reasoning_effort") != std::string::npos;
    can_preserve_ = !q38_ && g.get_str("tokenizer.chat_template").find("preserve_thinking") != std::string::npos;
    defaults_.temperature = (float)g.get_float("general.sampling.temp", 1.0);
    defaults_.top_p = (float)g.get_float("general.sampling.top_p", 0.95);
    defaults_.top_k = (int)g.get_int("general.sampling.top_k", 20);
    defaults_.min_p = (float)g.get_float("general.sampling.min_p", 0.0);
    gemma_ = e.hp.gemma;
    if (gemma_) {
      // Gemma 4: the GGUF template is rendered with minja; turns are <|turn>role ... <turn|>
      jinja_ = std::make_unique<JinjaChat>(g.get_str("tokenizer.chat_template"), "<bos>", "<eos>");
      im_start_ = tok_.token("<|turn>");
      for (const char* s : {"<turn|>", "<eos>", "<|tool_response>"}) {
        const int id = tok_.token(s);
        if (id >= 0) stop_ids_.push_back(id);
      }
      think_close_ = "<channel|>";
    } else {
      im_start_ = tok_.token("<|im_start|>");
      for (const char* s : {"<|im_end|>", "<|endoftext|>"}) {
        const int id = tok_.token(s);
        if (id >= 0) stop_ids_.push_back(id);
      }
    }
    if (tok_.eos >= 0) stop_ids_.push_back(tok_.eos);
  }

  bool has_vision() const { return vision_ != nullptr || qvision_ != nullptr; }
  bool has_audio() const { return audio_ != nullptr; }
  // The prompt tokens of a chat request (Anthropic's count_tokens): its rendered prompt, each image or audio part
  // counted as its placeholder token
  int count_tokens(const Json& body) {
    ChatOptions co;
    if (body["enable_thinking"].is_bool()) co.enable_thinking = body["enable_thinking"].as_bool();
    std::lock_guard<std::mutex> lock(mu_);
    const std::string prompt = gemma_ ? jinja_->render(body["messages"], body["tools"], true, co.enable_thinking)
                                      : chat::render(body["messages"], body["tools"], co, q38_);
    return (int)tok_.encode(prompt, true).size();
  }
  void handle(const HttpRequest& rq, HttpResponse& rs) {
    if (rq.method == "GET" && (rq.path == "/" || rq.path == "/index.html")) return serve_ui(rs);
    if (rq.path == "/health") return rs.send(200, "application/json", "{\"status\":\"ok\"}");
    if (rq.path == "/v1/models" || rq.path == "/models") {
      Json m = Json::object();
      m.set("id", model_id_);
      m.set("object", "model");
      m.set("owned_by", "arcinfer");
      m.set("context_length", eng_.ctx);
      set_modalities(m, has_vision());
      Json d = Json::array();
      d.push(m);
      Json r = Json::object();
      r.set("object", "list");
      r.set("models", llama_models(d));
      r.set("data", d);
      return rs.send(200, "application/json", r.dump());
    }
    if (rq.path == "/sessions" || (rq.path == "/sessions/save" && rq.method == "POST"))
      return sessions(rq, rs);  // GET: size of the session cache; DELETE: clear it; POST /sessions/save: keep the held one
    if (rq.path == "/props") {
      Json r = Json::object();
      r.set("model", model_id_);
      r.set("n_ctx", eng_.ctx);
      r.set("engine", "arcinfer");
      r.set("chunk", eng_.max_chunk);
      r.set("weights", eng_.w8_resident ? "w8-resident" : "w8-just-in-time");
      r.set("vision", has_vision());
      Json mod = Json::object();  // (llama-server's form)
      mod.set("vision", has_vision());
      mod.set("audio", audio_ != nullptr);
      r.set("modalities", mod);
      r.set("audio", audio_ != nullptr);
      r.set("mtp", eng_.has_mtp);
      r.set("mtp_drafts", eng_.opt.mtp_drafts);
      {  // GPU memory now (bytes): parts of the loaded model, encoders loaded since (on first use)
        const Engine::MemParts mp = eng_.mem_parts(eng_.ctx);
        Json mem = Json::object();
        mem.set("device", (double)eng_.device_mem());
        mem.set("total", (double)eng_.gpu_bytes);
        mem.set("weights", mp.weights);
        mem.set("mtp", mp.mtp);
        mem.set("buffers", mp.buffers);
        mem.set("context", mp.context);
        mem.set("encoders", (double)(eng_.gpu_bytes - eng_.loaded_bytes()));
        r.set("memory", mem);
      }
      {
        std::lock_guard<std::mutex> lk(stats_mu_);
        Json st = Json::object();
        st.set("requests", n_req_);
        st.set("busy", busy_.load());
        st.set("position", eng_.pos);
        if (!last_timings_.is_null()) st.set("last", last_timings_);
        r.set("stats", st);
      }
      return rs.send(200, "application/json", r.dump());
    }
    if (rq.path == "/tokenize" && rq.method == "POST") {
      const Json b = Json::parse(rq.body);
      Json ids = Json::array();
      for (int id : tok_.encode(b["content"].str(), true)) ids.push(id);
      Json r = Json::object();
      r.set("tokens", ids);
      return rs.send(200, "application/json", r.dump());
    }
    if (rq.path == "/v1/chat/completions" && rq.method == "POST") return chat(rq, rs);
  }

 private:
  Engine& eng_;
  Tokenizer& tok_;
  std::string model_id_, webui_;
  bool q38_ = false, gemma_ = false;

 public:
  void set_vision(std::unique_ptr<GemmaImageEncoder> v) {
    vision_ = std::move(v);
    img_tok_ = tok_.token("<|image|>");
    img_beg_ = tok_.token("<|image>");
    img_end_ = tok_.token("<image|>");
  }
  // Qwen3-VL (Qwen3.5 / 3.6 / 3.8 mmproj): the template already writes <|vision_start|> ... <|vision_end|> around
  // each <|image_pad|>, which becomes the image's embedding rows (causal, M-RoPE positions)
  void set_qvision(std::unique_ptr<VisionQwen3VL> v) {
    qvision_ = std::move(v);
    img_tok_ = tok_.token("<|image_pad|>");
    img_beg_ = img_end_ = -1;
  }
  void set_audio(std::unique_ptr<GemmaAudioEncoder> a) {
    audio_ = std::move(a);
    aud_tok_ = tok_.token("<|audio|>");
    aud_beg_ = tok_.token("<|audio>");
    aud_end_ = tok_.token("<audio|>");
  }

 private:
  std::unique_ptr<JinjaChat> jinja_;
  // vision (Gemma 4 mmproj): image token ids and an embedding cache keyed by the image bytes
  std::unique_ptr<GemmaImageEncoder> vision_;  // Gemma 4: 12B (gemma4uv, CPU) or E2B / E4B (gemma4v, GPU)
  std::unique_ptr<VisionQwen3VL> qvision_;
  int img_tok_ = -1, img_beg_ = -1, img_end_ = -1;
  // audio (Gemma 4 mmproj with the unified audio projector): token ids; embeddings share the cache below
  std::unique_ptr<GemmaAudioEncoder> audio_;  // Gemma 4: 12B (gemma4ua, CPU) or E2B / E4B (gemma4a conformer, GPU)
  int aud_tok_ = -1, aud_beg_ = -1, aud_end_ = -1;
  // monitoring (/props "stats"): requests served, one in progress, the last one's timings
  std::mutex stats_mu_;
  std::atomic<bool> busy_{false};
  int n_req_ = 0;
  Json last_timings_;
  struct MediaEmb {
    std::shared_ptr<std::vector<float>> e;
    int gw = 0, gh = 0;  // merged-token grid (Qwen M-RoPE); 0 for Gemma and audio
  };
  std::map<uint64_t, MediaEmb> img_cache_;  // media embeddings keyed by the file bytes
  std::string think_close_ = "</think>";
  SamplingParams defaults_;
  int im_start_ = -1;
  std::vector<int> stop_ids_;
  std::mutex mu_;
  std::vector<int> cached_;  // tokens whose state the engine currently holds (positions 0..pos)
  std::atomic<uint64_t> counter_{0};
 public:
  int default_think_budget_ = -1;  // -1 = unlimited
  bool can_preserve_ = false;      // the template has Qwen3.6's preserve_thinking option
  bool default_preserve_ = false;  // --preserve-thinking (requests can override it: chat_template_kwargs)
 private:

  void serve_ui(HttpResponse& rs) {
    std::ifstream f(webui_ + "/index.html", std::ios::binary);
    if (!f) return rs.send(404, "text/plain", "web UI not found: " + webui_ + "/index.html");
    std::stringstream ss;
    ss << f.rdbuf();
    rs.send(200, "text/html; charset=utf-8", ss.str());
  }

  // s with every byte that is not part of a valid UTF-8 sequence replaced by U+FFFD (JSON text must be valid UTF-8)
  static std::string utf8_safe(const std::string& s) {
    std::string o;
    for (size_t i = 0; i < s.size();) {
      const unsigned char c = (unsigned char)s[i];
      const size_t n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
      bool ok = n > 0 && i + n <= s.size();
      for (size_t k = 1; ok && k < n; ++k) ok = ((unsigned char)s[i + k] >> 6) == 2;
      if (ok) { o.append(s, i, n); i += n; }
      else { o += "\xEF\xBF\xBD"; ++i; }
    }
    return o;
  }
  static Json err(const std::string& msg) {
    Json e = Json::object();
    e.set("message", msg);
    e.set("type", "invalid_request_error");
    Json r = Json::object();
    r.set("error", e);
    return r;
  }

  static constexpr int kSnaps = 3;  // recurrent-state snapshot slots (see prepare_prefix)
  // Brings the engine to the state after `P[0..start)` reusing cached work; returns start.
  size_t prepare_prefix(const std::vector<int>& P) {
    size_t L = 0;
    while (L < P.size() && L < cached_.size() && P[L] == cached_[L]) ++L;
    if (L >= P.size()) L = P.size() - 1;  // at least one token must be evaluated to get logits
    while (L > 0 && L < P.size() && P[L] < 0 && P[L - 1] == P[L]) --L;  // back to the start of an image span
    if (L == cached_.size() && (int)cached_.size() == eng_.pos) return L;  // pure extension
    if (gemma_) {  // no recurrent state: move back to the common prefix if the sliding-window rings allow it
      if (eng_.rewind((int)L)) {
        cached_.resize(L);
        return L;
      }
      eng_.reset();
      cached_.clear();
      return 0;
    }
    // the DeltaNet states can only be restored where a snapshot was taken: the end of the last prompt (slot 1), its
    // last assistant turn start (slot 0), the turn start after the user's last message (slot 2: Qwen's template
    // drops the reasoning of earlier turns once the user writes again, so a new message diverges there), and (in
    // host memory) the end of the system prompt, which conversations of a client share
    int best = -1, bp = -1;
    for (int s = 0; s < kSnaps; ++s) {
      const int p = eng_.snapshot_pos(s);
      if (p >= 0 && (size_t)p <= L && p > bp) { best = s; bp = p; }
    }
    const int hpos = eng_.host_snapshot_pos();
    if (hpos > (int)L) eng_.drop_host_snapshot();  // the request diverges inside the system prompt
    if (hpos >= 0 && hpos <= (int)L && hpos > bp) { best = kSnaps; bp = hpos; }
    if (best >= 0) {
      if (best == kSnaps) eng_.restore_host();
      else eng_.restore(best);
      for (int s = 0; s < kSnaps; ++s) if (eng_.snapshot_pos(s) > bp) eng_.drop_snapshot(s);
      cached_.resize(bp);
      return bp;
    }
    eng_.reset();
    for (int s = 0; s < kSnaps; ++s) eng_.drop_snapshot(s);
    cached_.clear();
    return 0;
  }

  // ---- session cache: saved conversations on disk
  // When a request belongs to another conversation than the one the engine holds (it does not continue that one's
  // last prompt), the held conversation is written to disk first: its KV cache, DeltaNet states and snapshots
  // (Engine::save_state). A saved conversation that shares more with the request than the held one is loaded instead
  // of processing its prompt again. One file per conversation: [magic, version, n, tokens[n], gen cut, resume
  // floor, n resume points, points..., engine state]; the least recently used go beyond the size cap.
  struct SavedConv {
    std::string file;
    std::vector<int> tokens, resume;  // resume: positions with a recurrent-state snapshot (Qwen)
    int gen_cut = 0, floor = 0;       // floor (Gemma): the first position the sliding-window rings still allow
    uint64_t bytes = 0, used = 0;
    std::shared_ptr<std::thread> writer;          // the file is being written in the background
    std::shared_ptr<std::atomic<bool>> failed;    // ... and that write failed
  };
  // waits for a background write of s; false if it failed
  static bool written(SavedConv& s) {
    if (s.writer && s.writer->joinable()) s.writer->join();
    return !(s.failed && *s.failed);
  }
  void forget(size_t i) {
    written(saved_[i]);
    std::error_code ec;
    std::filesystem::remove(saved_[i].file, ec);
    saved_.erase(saved_.begin() + i);
  }
  std::string sess_dir_;
  uint64_t sess_cap_ = 0, use_clock_ = 0;
  std::vector<SavedConv> saved_;
  bool cur_saved_ = false;  // the engine holds exactly a saved conversation
  int last_gen_cut_ = 0;    // assistant turn start of the last prompt processed

  static size_t common_prefix(const std::vector<int>& a, const std::vector<int>& b) {
    size_t n = 0;
    while (n < a.size() && n < b.size() && a[n] == b[n]) ++n;
    return n;
  }
  // how much of P a state could keep: the largest resume point within the common prefix
  int usable(const std::vector<int>& P, const std::vector<int>& tokens, const std::vector<int>& resume, int floor) const {
    const int L = (int)std::min(common_prefix(P, tokens), P.size() - 1);
    if (gemma_) return L >= floor ? L : 0;
    int u = 0;
    for (int r : resume)
      if (r <= L) u = std::max(u, r);
    return u;
  }
  std::vector<int> current_resume() const {
    std::vector<int> r = {eng_.host_snapshot_pos()};
    for (int s = 0; s < kSnaps; ++s) r.push_back(eng_.snapshot_pos(s));
    if ((int)cached_.size() == eng_.pos) r.push_back(eng_.pos);
    return r;
  }
  int current_floor() const { return gemma_ ? eng_.rewind_floor() : 0; }

  // Saves the held conversation to the session cache and waits until it is on disk (arc-server stop / restart and a
  // model switch call it first, so that going back to the conversation afterwards loads it). False when there is
  // nothing worth saving (cache off, already saved, under 1,024 tokens).
 public:
  bool save_held() {
    std::lock_guard<std::mutex> lock(mu_);
    if (!sess_cap_ || cur_saved_ || cached_.size() < 1024) return false;
    save_current();
    return !saved_.empty() && written(saved_.back());
  }

 private:
  void sessions(const HttpRequest& rq, HttpResponse& rs) {
    if (rq.path == "/sessions/save") {
      Json r = Json::object();
      r.set("saved", save_held());
      return rs.send(200, "application/json", r.dump());
    }
    std::lock_guard<std::mutex> lock(mu_);
    uint64_t total = 0;
    for (auto& s : saved_) total += s.bytes;
    Json r = Json::object();
    if (rq.method == "DELETE") {
      r.set("deleted", (int)saved_.size());
      r.set("freed_bytes", (double)total);
      fprintf(stderr, "session cache: cleared %zu saved conversations (%.1f GiB)\n", saved_.size(), total / 1073741824.0);
      while (!saved_.empty()) forget(saved_.size() - 1);
      cur_saved_ = false;
      total = 0;
    }
    r.set("dir", sess_dir_);
    r.set("conversations", (int)saved_.size());
    r.set("bytes", (double)total);
    r.set("cap_bytes", (double)sess_cap_);
    rs.send(200, "application/json", r.dump());
  }

  // ---- system-prompt cache: the state after a client's system prompt, on disk
  // Saved when a system prompt is processed for the first time (positions [0, sys_cut): its KV cache and the
  // recurrent states there; 27B: ~0.3 GB for Pi's ~4K tokens) and loaded when a request starts with it while the
  // engine holds something else: after a server start, a model switch or another conversation. Kept per project
  // folder (Pi writes it into its system prompt): "<folder>-<folder hash>-<prompt hash>-<n>.arcp", the folder's 2
  // latest versions, so one project's files never push out another's; other clients' prompts
  // "<prompt hash>-<n>.arcp", the 8 used last. All models together stay under --prefix-cache-gb (least recently used
  // go first). Files: [magic, n, tokens[n], engine state].
  std::string prefix_dir_;
  uint64_t prefix_cap_ = 0;
  static constexpr int32_t kPrefixMagic = 0x50535241;  // "ARSP"
  static uint64_t fnv(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
    for (size_t i = 0; i < n; ++i) h = (h ^ ((const uint8_t*)p)[i]) * 1099511628211ull;
    return h;
  }
  // Pi's project folder: its system prompt carries a section "<cwd>\nC:/path/to/project\n</cwd>"
  static std::string prompt_folder(const Json& messages) {
    const Json& m = messages[0];
    if (m["role"].str() != "system" && m["role"].str() != "developer") return "";
    std::string text;
    const Json& c = m["content"];
    if (c.is_string()) text = c.as_string();
    else if (c.is_array())
      for (size_t i = 0; i < c.size(); ++i) text += c[i]["text"].str();
    const size_t a = text.find("<cwd>\n");
    if (a == std::string::npos) return "";
    const size_t b = text.find("\n</cwd>", a + 6);
    if (b == std::string::npos || b - (a + 6) > 1024) return "";
    return text.substr(a + 6, b - (a + 6));
  }
  // the file-name group of a folder: "<last path component>-<hash of the folder>-" ("" without a folder)
  static std::string folder_group(const std::string& folder) {
    if (folder.empty()) return "";
    std::string low = folder;
    for (auto& ch : low) ch = ch == '\\' ? '/' : (char)tolower((unsigned char)ch);  // (Windows paths: one spelling)
    while (low.size() > 1 && low.back() == '/') low.pop_back();
    std::string name = low.substr(low.find_last_of('/') + 1), slug;
    for (char ch : name)
      if (slug.size() < 32) slug += (isalnum((unsigned char)ch) || ch == '.' || ch == '_') ? ch : '_';
    char b[24];
    snprintf(b, sizeof b, "-%08x-", (unsigned)fnv(low.data(), low.size()));
    return (slug.empty() ? "folder" : slug) + b;
  }
  // group + "<prompt hash>-<n>.arcp" (files from before the per-folder names have no group)
  std::string prefix_file(const std::vector<int>& P, size_t n, const std::string& group) const {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) h = (h ^ (uint32_t)P[i]) * 1099511628211ull;
    char b[40];
    snprintf(b, sizeof b, "%016llx-%zu.arcp", (unsigned long long)h, n);
    return prefix_dir_ + "/" + group + b;
  }
  void load_prefix(const std::vector<int>& P, size_t n, const std::string& folder) {
    if (prefix_dir_.empty() || n < 256 || n >= P.size() || common_prefix(P, cached_) >= n) return;
    prefix_written();  // a write of it still going on
    const std::string group = folder_group(folder), file = prefix_file(P, n, group);
    std::error_code ec;
    if (!std::filesystem::exists(file, ec)) {  // saved before the per-folder names: adopt it
      const std::string old = prefix_file(P, n, "");
      if (group.empty() || !std::filesystem::exists(old, ec)) return;
      std::filesystem::rename(old, file, ec);
      if (ec) return;
    }
    FILE* f = fopen(file.c_str(), "rb");
    if (!f) return;
    const auto t0 = clk::now();
    int32_t h[2] = {0, 0};
    std::vector<int> toks;
    bool ok = fread(h, 4, 2, f) == 2 && h[0] == kPrefixMagic && h[1] == (int32_t)n;
    if (ok) {
      toks.resize(n);
      ok = fread(toks.data(), 4, n, f) == n;
    }
    if (ok && !std::equal(toks.begin(), toks.end(), P.begin())) {  // another prompt with the same hash: leave it
      fclose(f);
      return;
    }
    ok = ok && eng_.load_state(f);
    fclose(f);
    if (!ok) {  // another model's layout or a damaged file; the engine may have been reset
      std::filesystem::remove(file, ec);
      cached_.clear();
      last_gen_cut_ = 0;
      fprintf(stderr, "system-prompt cache: could not load %s\n", file.c_str());
      return;
    }
    std::filesystem::last_write_time(file, std::filesystem::file_time_type::clock::now(), ec);  // recently used
    if (!gemma_) eng_.snapshot_host();  // new conversations resume here again
    cached_ = toks;
    last_gen_cut_ = 0;
    cur_saved_ = false;
    fprintf(stderr, "system-prompt cache: loaded %zu tokens%s%s in %.2f s\n", n, folder.empty() ? "" : " for ",
            folder.c_str(), ms_since(t0) / 1000);
  }
  // after a write: the folder's 2 latest versions (or the 8 latest folder-less prompts), then all models' files
  // under the size cap, least recently used first (never the file just written)
  static void prune_prefix(const std::string& dir, const std::string& group, const std::string& keep, uint64_t cap) {
    namespace fs = std::filesystem;
    std::error_code e;
    auto is_plain = [](const std::string& name) {  // "<16 hex>-<n>.arcp"
      if (name.size() < 23 || name[16] != '-') return false;
      for (size_t i = 0; i < name.size() - 5; ++i)
        if (i != 16 && !(i < 16 ? isxdigit((unsigned char)name[i]) : isdigit((unsigned char)name[i]))) return false;
      return true;
    };
    std::vector<std::pair<fs::file_time_type, fs::path>> mine;
    for (const auto& de : fs::directory_iterator(dir, e)) {
      const std::string name = de.path().filename().string();
      if (de.path().extension() != ".arcp") continue;
      if (group.empty() ? is_plain(name) : name.rfind(group, 0) == 0) mine.push_back({de.last_write_time(e), de.path()});
    }
    std::sort(mine.begin(), mine.end(), [](auto& a, auto& b) { return a.first > b.first; });
    for (size_t i = group.empty() ? 8 : 2; i < mine.size(); ++i) fs::remove(mine[i].second, e);
    if (!cap) return;
    std::vector<std::tuple<fs::file_time_type, fs::path, uint64_t>> all;  // every model's folder
    uint64_t total = 0;
    for (const auto& de : fs::recursive_directory_iterator(fs::path(dir).parent_path(), e))
      if (de.path().extension() == ".arcp") {
        const uint64_t sz = de.file_size(e);
        total += sz;
        all.push_back({de.last_write_time(e), de.path(), sz});
      }
    std::sort(all.begin(), all.end(), [](auto& a, auto& b) { return std::get<0>(a) < std::get<0>(b); });
    for (auto& [t, p, sz] : all) {
      if (total <= cap) break;
      if (p == fs::path(keep)) continue;
      fs::remove(p, e);
      total -= sz;
    }
  }
  // The state is copied to host memory and written to disk in the background while the request goes on (with less
  // than 2 GiB of RAM left over: written to disk first).
  std::shared_ptr<std::thread> prefix_writer_;
  void prefix_written() {
    if (prefix_writer_ && prefix_writer_->joinable()) prefix_writer_->join();
  }
  void save_prefix(const std::vector<int>& P, size_t n, const std::string& folder) {
    if (prefix_dir_.empty() || n < 256 || eng_.pos != (int)n) return;
    prefix_written();
    const std::string group = folder_group(folder), file = prefix_file(P, n, group);
    std::error_code ec;
    if (std::filesystem::exists(file, ec)) return;
    const auto t0 = clk::now();
    const int32_t h[2] = {kPrefixMagic, (int32_t)n};
    auto finish = [dir = prefix_dir_, group, cap = prefix_cap_, folder, file, n, t0](bool ok) {
      std::error_code e;
      if (ok) std::filesystem::rename(file + ".tmp", file, e);
      if (!ok || e) {
        std::filesystem::remove(file + ".tmp", e);
        fprintf(stderr, "system-prompt cache: could not write %s\n", file.c_str());
        return;
      }
      prune_prefix(dir, group, file, cap);
      fprintf(stderr, "system-prompt cache: saved %zu tokens%s%s (%.0f MB) in %.2f s\n", n, folder.empty() ? "" : " for ",
              folder.c_str(), std::filesystem::file_size(file, e) / 1048576.0, ms_since(t0) / 1000);
    };
    // (the host snapshot taken at n equals the current state: written once, taken again after loading)
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    const uint64_t bytes = 8 + 4 * n + eng_.saved_size((int)n);
    if (GlobalMemoryStatusEx(&ms) && ms.ullAvailPhys > bytes + (2ull << 30)) {
      auto pieces = std::make_shared<std::vector<std::vector<uint8_t>>>();
      pieces->emplace_back((const uint8_t*)h, (const uint8_t*)(h + 2));
      pieces->emplace_back((const uint8_t*)P.data(), (const uint8_t*)(P.data() + n));
      const bool ok = eng_.save_state([&](const void* p, size_t k) {
        pieces->emplace_back((const uint8_t*)p, (const uint8_t*)p + k);
        return true;
      }, (int)n, false);
      if (!ok) return;
      prefix_writer_ = std::make_shared<std::thread>([pieces, file, finish] {
        FILE* f = fopen((file + ".tmp").c_str(), "wb");
        bool w = f != nullptr;
        for (const auto& pc : *pieces)
          if (w) w = fwrite(pc.data(), 1, pc.size(), f) == pc.size();
        if (f) w = fclose(f) == 0 && w;
        finish(w);
      });
      return;
    }
    FILE* f = fopen((file + ".tmp").c_str(), "wb");
    if (!f) return;
    bool ok = fwrite(h, 4, 2, f) == 2 && fwrite(P.data(), 4, n, f) == n && eng_.save_state(f, (int)n, false);
    ok = fclose(f) == 0 && ok;
    finish(ok);
  }

 public:
  void open_prefix_cache(const std::string& dir, double cap_gb) {
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    for (const auto& e : std::filesystem::directory_iterator(dir, ec))  // writes the server did not finish
      if (e.path().extension() == ".tmp") std::filesystem::remove(e.path(), ec);
    prefix_dir_ = dir;
    prefix_cap_ = (uint64_t)(cap_gb * 1073741824.0);
  }
  void open_sessions(const std::string& dir, double cap_gb) {
    sess_cap_ = (uint64_t)(cap_gb * 1073741824.0);
    if (!sess_cap_) return;
    sess_dir_ = dir;
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    for (const auto& de : std::filesystem::directory_iterator(dir, ec)) {
      if (de.path().extension() == ".tmp") {  // a write the server did not finish (it was stopped)
        std::error_code ec2;
        std::filesystem::remove(de.path(), ec2);
        continue;
      }
      if (de.path().extension() != ".arcs") continue;
      FILE* f = fopen(de.path().string().c_str(), "rb");
      if (!f) continue;
      SavedConv s;
      if (read_header(f, s)) {
        s.file = de.path().string();
        s.bytes = std::filesystem::file_size(de.path(), ec);
        s.used = (uint64_t)std::filesystem::last_write_time(de.path(), ec).time_since_epoch().count();
        saved_.push_back(std::move(s));
      }
      fclose(f);
    }
    std::sort(saved_.begin(), saved_.end(), [](const SavedConv& a, const SavedConv& b) { return a.used < b.used; });
    for (auto& s : saved_) s.used = ++use_clock_;
    uint64_t total = 0;
    for (auto& s : saved_) total += s.bytes;
    fprintf(stderr, "session cache: %s, %zu saved conversations (%.1f of %.1f GiB)\n", dir.c_str(), saved_.size(),
            total / 1073741824.0, cap_gb);
  }

 private:
  static constexpr int32_t kConvMagic = 0x56535241;  // "ARSV"
  static bool read_header(FILE* f, SavedConv& s) {
    int32_t h[3];
    if (fread(h, 4, 3, f) != 3 || h[0] != kConvMagic || h[1] != 1 || h[2] < 0 || h[2] > (1 << 24)) return false;
    s.tokens.resize(h[2]);
    int32_t g[3];
    if (fread(s.tokens.data(), 4, h[2], f) != (size_t)h[2] || fread(g, 4, 3, f) != 3 || g[2] < 0 || g[2] > 16) return false;
    s.gen_cut = g[0];
    s.floor = g[1];
    s.resume.resize(g[2]);
    return fread(s.resume.data(), 4, g[2], f) == (size_t)g[2];
  }
  // Saves the held conversation (replacing saved states it supersedes), then trims the cache. With enough free RAM
  // (2 GiB left over) the state is copied to host memory and written to disk in the background while the next
  // request is served; otherwise it is written to disk first.
  void save_current() {
    const auto t0 = clk::now();
    // the snapshot at the end of the last prompt is a few tokens after the one at its assistant turn start: not worth
    // its space (Qwen3.8 27B: 157 MB); the request being served diverges before both anyway
    eng_.drop_snapshot(1);
    const std::string file = sess_dir_ + "/" + std::to_string((uint64_t)time(nullptr)) + "-" + std::to_string(++use_clock_) + ".arcs";
    const std::vector<int> res = current_resume();
    // resumable from the last prompt's assistant turn on (Gemma keeps the sliding-window K/V from n_swa before it)
    const int floor = eng_.saved_floor(last_gen_cut_);
    std::vector<int32_t> head = {kConvMagic, 1, (int32_t)cached_.size()};
    head.insert(head.end(), cached_.begin(), cached_.end());
    head.insert(head.end(), {last_gen_cut_, floor, (int32_t)res.size()});
    head.insert(head.end(), res.begin(), res.end());
    const uint64_t bytes = head.size() * 4 + eng_.saved_size(last_gen_cut_);
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    const bool bg = GlobalMemoryStatusEx(&ms) && ms.ullAvailPhys > bytes + (2ull << 30);
    auto failed = std::make_shared<std::atomic<bool>>(false);
    std::shared_ptr<std::thread> writer;
    bool ok;
    std::error_code ec;
    if (bg) {
      auto pieces = std::make_shared<std::vector<std::vector<uint8_t>>>();
      pieces->emplace_back((const uint8_t*)head.data(), (const uint8_t*)(head.data() + head.size()));
      ok = eng_.save_state([&](const void* p, size_t n) {
        pieces->emplace_back((const uint8_t*)p, (const uint8_t*)p + n);
        return true;
      }, last_gen_cut_);
      if (ok)
        writer = std::make_shared<std::thread>([pieces, file, failed] {
          FILE* f = fopen((file + ".tmp").c_str(), "wb");
          bool w = f != nullptr;
          for (const auto& pc : *pieces)
            if (w) w = fwrite(pc.data(), 1, pc.size(), f) == pc.size();
          if (f) w = fclose(f) == 0 && w;
          std::error_code e;
          if (w) std::filesystem::rename(file + ".tmp", file, e);
          if (!w || e) {
            std::filesystem::remove(file + ".tmp", e);
            *failed = true;
            fprintf(stderr, "session cache: could not write %s\n", file.c_str());
          }
        });
    } else {
      FILE* f = fopen((file + ".tmp").c_str(), "wb");
      ok = f && fwrite(head.data(), 4, head.size(), f) == head.size() && eng_.save_state(f, last_gen_cut_);
      if (f) ok = fclose(f) == 0 && ok;
      if (ok) std::filesystem::rename(file + ".tmp", file, ec);
      ok = ok && !ec;
      if (!ok) std::filesystem::remove(file + ".tmp", ec);
    }
    if (!ok) {
      fprintf(stderr, "session cache: could not save the conversation (%zu tokens)\n", cached_.size());
      return;
    }
    // older states of the same conversation: everything they could resume, this one resumes too
    for (size_t i = 0; i < saved_.size();)
      if ((int)common_prefix(saved_[i].tokens, cached_) >= saved_[i].gen_cut) forget(i);
      else ++i;
    saved_.push_back(SavedConv{file, cached_, res, last_gen_cut_, floor, bytes, use_clock_, writer, failed});
    uint64_t total = 0;
    for (auto& c : saved_) total += c.bytes;
    while (total > sess_cap_ && saved_.size() > 1) {  // least recently used first (never the one just saved)
      auto lru = std::min_element(saved_.begin(), saved_.end() - 1, [](auto& a, auto& b) { return a.used < b.used; });
      total -= lru->bytes;
      forget(lru - saved_.begin());
    }
    cur_saved_ = true;
    fprintf(stderr, "session cache: saved a conversation of %zu tokens (%.0f MB) in %.2f s%s\n", cached_.size(),
            bytes / 1048576.0, ms_since(t0) / 1000, bg ? " (to memory; written to disk in the background)" : "");
  }
  // Before a request: keep the held conversation if the request does not continue it, and bring back a saved one
  // that shares more with the request.
  void switch_conversation(const std::vector<int>& P) {
    if (!sess_cap_ || P.empty()) return;
    const size_t lc = std::min(common_prefix(P, cached_), P.size() - 1);
    const bool other = (int)lc < last_gen_cut_;
    if (other && !cur_saved_ && cached_.size() >= 1024) save_current();
    const int u_cur = usable(P, cached_, current_resume(), current_floor());
    int bi = -1, bu = u_cur + 512;  // loading must save at least ~512 tokens of prompt processing
    for (size_t i = 0; i < saved_.size(); ++i) {
      const int u = usable(P, saved_[i].tokens, saved_[i].resume, saved_[i].floor);
      if (u > bu) { bi = (int)i; bu = u; }
    }
    if (bi < 0) return;
    SavedConv& s = saved_[bi];
    const auto t0 = clk::now();
    const bool done = written(s);  // a background write of this conversation finishes first
    FILE* f = done ? fopen(s.file.c_str(), "rb") : nullptr;
    SavedConv tmp;
    const bool ok = f && read_header(f, tmp) && eng_.load_state(f);
    if (f) fclose(f);
    if (!ok) {  // unreadable or from another layout: forget it; the engine was reset
      fprintf(stderr, "session cache: could not load %s\n", s.file.c_str());
      forget(bi);
      cached_.clear();
      last_gen_cut_ = 0;
      cur_saved_ = false;
      return;
    }
    cached_ = s.tokens;
    last_gen_cut_ = s.gen_cut;
    s.used = ++use_clock_;
    cur_saved_ = true;
    fprintf(stderr, "session cache: loaded a conversation of %zu tokens (%.0f MB) in %.2f s, %d prompt tokens reusable\n",
            s.tokens.size(), s.bytes / 1048576.0, ms_since(t0) / 1000, bu);
  }

  void chat(const HttpRequest& rq, HttpResponse& rs) {
    Json body;
    if (!Json::try_parse(rq.body, body)) return rs.send(400, "application/json", err("invalid JSON body").dump());
    const Json& messages = body["messages"];
    if (!messages.is_array() || messages.size() == 0) return rs.send(400, "application/json", err("messages required").dump());
    const std::string folder = prompt_folder(messages);  // Pi: the project folder (its system-prompt cache)
    Json tools = body["tools"];
    if (body["tool_choice"].str() == "none") tools = Json();
    // tool_choice "required" / a named function: the answer starts with a tool call (to that function)
    std::string force_tool;
    bool force_tool_on = false;
    {
      const Json& tc = body["tool_choice"];
      const std::string fname = tc["function"]["name"].str(tc["name"].str());
      if (tools.is_array() && tools.size() && (tc.str() == "required" || !fname.empty())) {
        force_tool_on = true;
        force_tool = gemma_ ? "<|tool_call>call:" + (fname.empty() ? std::string() : fname + "{")
                            : "<tool_call>\n<function=" + (fname.empty() ? std::string() : fname + ">\n");
      }
    }
    // response_format: the answer is JSON (an object; with json_schema one that fits the schema)
    std::unique_ptr<jcon::Checker> jc;
    {
      const Json& rf = body["response_format"];
      const std::string ty = rf["type"].str();
      if (ty == "json_object") jc = std::make_unique<jcon::Checker>(Json::object(), true);
      else if (ty == "json_schema") jc = std::make_unique<jcon::Checker>(rf["json_schema"]["schema"].is_object() ? rf["json_schema"]["schema"] : Json::object(), false);
      if (jc) tools = Json();  // (structured answers, no tool calls)
    }
    // tool calls (Gemma 4 and Qwen formats): while a call is open, only tokens that keep it well formed and fitting
    // its tool's schema (tools/tool_constraint.h; ARC_TOOL_CONSTRAINT=0 turns it off)
    std::unique_ptr<tcon::ToolCon> tcc;
    {
      const char* e = getenv("ARC_TOOL_CONSTRAINT");
      if (tools.is_array() && tools.size() && !(e && std::string(e) == "0")) {
        if (gemma_) tcc = std::make_unique<tcon::GemmaCall>(tools);
        else tcc = std::make_unique<tcon::QwenCall>(tools);
        if (tcc->empty()) tcc.reset();
      }
    }
    ChatOptions co;
    if (body["enable_thinking"].is_bool()) co.enable_thinking = body["enable_thinking"].as_bool();
    if (body["chat_template_kwargs"]["enable_thinking"].is_bool()) co.enable_thinking = body["chat_template_kwargs"]["enable_thinking"].as_bool();
    const std::string effort = body["reasoning_effort"].str(body["chat_template_kwargs"]["reasoning_effort"].str());
    if (effort == "none" || effort == "minimal" || effort == "off") co.enable_thinking = false;
    else if (effort == "low") co.reasoning_effort = "low";
    else if (effort == "medium") co.reasoning_effort = "medium";
    else if (!effort.empty()) co.reasoning_effort = "xhigh";
    // earlier turns keep their reasoning: only where the model's template offers it (Qwen3.6)
    co.preserve_thinking = can_preserve_ && body["chat_template_kwargs"]["preserve_thinking"].as_bool(default_preserve_);
    SamplingParams sp = defaults_;
    if (body["temperature"].is_number()) sp.temperature = (float)body["temperature"].as_number();
    if (body["top_p"].is_number()) sp.top_p = (float)body["top_p"].as_number();
    if (body["top_k"].is_number()) sp.top_k = (int)body["top_k"].as_int();
    if (body["min_p"].is_number()) sp.min_p = (float)body["min_p"].as_number();
    if (body["presence_penalty"].is_number()) sp.presence_penalty = (float)body["presence_penalty"].as_number();
    if (body["frequency_penalty"].is_number()) sp.frequency_penalty = (float)body["frequency_penalty"].as_number();
    if (body["repeat_penalty"].is_number()) sp.repeat_penalty = (float)body["repeat_penalty"].as_number();
    if (body["repeat_last_n"].is_number()) sp.repeat_last_n = (int)body["repeat_last_n"].as_int();
    for (const auto& [k, v] : body["logit_bias"].entries())  // {"token id": bias}
      if (v.is_number()) {
        char* end = nullptr;
        const long t = strtol(k.c_str(), &end, 10);
        if (end && *end == 0) sp.logit_bias.push_back({(int)t, (float)v.as_number()});
      }
    if (body["seed"].is_number()) sp.seed = (uint64_t)body["seed"].as_int();
    int max_tokens = (int)body["max_completion_tokens"].as_int(body["max_tokens"].as_int(-1));
    // reasoning budget: after this many reasoning tokens the server closes the <think> block itself
    int think_budget = (int)body["thinking_token_budget"].as_int(body["thinking_budget_tokens"].as_int(body["reasoning_budget"].as_int(default_think_budget_)));
    std::vector<std::string> stops;
    if (body["stop"].is_string()) stops.push_back(body["stop"].as_string());
    for (const Json& s : body["stop"].items()) if (s.is_string()) stops.push_back(s.as_string());
    const bool stream = body["stream"].as_bool(false);
    // logprobs: the log probability of every generated token (from the model's distribution, before penalties and
    // the logit bias) and of its top_logprobs most likely alternatives (0..20)
    const bool want_lp = body["logprobs"].as_bool(false);
    const int top_lp = std::clamp((int)body["top_logprobs"].as_int(0), 0, 20);
    const bool usage_in_stream = body["stream_options"]["include_usage"].as_bool(false);

    std::string prompt;
    const bool raw_prompt = body["__raw_prompt"].is_string();  // (api_compat: the prompt text as it is)
    if (raw_prompt) {
      tools = Json();
      co.enable_thinking = false;
    }
    try {
      prompt = raw_prompt ? body["__raw_prompt"].str()
               : gemma_   ? jinja_->render(messages, tools, true, co.enable_thinking)
                          : chat::render(messages, tools, co, q38_);
    } catch (const std::exception& e) {
      return rs.send(400, "application/json", err(e.what()).dump());
    }
    std::vector<int> P = tok_.encode(prompt, true);
    // media: the template writes one <|image|> per image part and one <|audio|> per input_audio part; each becomes
    // <|image> / <|audio> + n embedding rows + <image|> / <audio|>. Images are read non-causally, audio causally.
    struct ImgSpan {
      size_t start;
      int n;
      std::shared_ptr<std::vector<float>> embd;
      bool causal = false;
      int gw = 0, gh = 0;  // Qwen images: grid for the M-RoPE positions
    };
    std::vector<ImgSpan> spans;
    {
      std::vector<std::string> urls;
      std::vector<std::vector<uint8_t>> clips;
      try {
        for (const Json& m : messages.items())
          for (const Json& part : m["content"].items()) {
            const std::string ty = part["type"].str();
            if (ty == "image_url") urls.push_back(part["image_url"].is_string() ? part["image_url"].str() : part["image_url"]["url"].str());
            else if (ty == "image") urls.push_back(part["image"].is_string() ? part["image"].str() : part["image_url"]["url"].str());
            else if (ty == "input_audio" || ty == "audio") {
              // OpenAI: {"input_audio": {"data": <base64>, "format": "wav" | "mp3" ...}}; a data URL also works
              const Json& a = part[ty == "audio" ? "audio" : "input_audio"];
              const std::string d = a.is_string() ? a.str() : a["data"].str();
              clips.push_back(d.rfind("data:", 0) == 0 ? img::from_data_url(d) : img::base64_decode(d));
            }
          }
      } catch (const std::exception& e) {
        return rs.send(400, "application/json", err(std::string("media: ") + e.what()).dump());
      }
      if (!urls.empty() && (!has_vision() || img_tok_ < 0))
        return rs.send(400, "application/json", err("this model was loaded without an image encoder (mmproj)").dump());
      if (!clips.empty() && (!audio_ || aud_tok_ < 0))
        return rs.send(400, "application/json", err("this model was loaded without an audio encoder (mmproj with audio)").dump());
      if (!urls.empty() || !clips.empty()) {
        std::vector<int> P2;
        size_t ki = 0, ka = 0;
        auto fnv = [](const std::vector<uint8_t>& bytes) {
          uint64_t h = 1469598103934665603ull;  // FNV-1a of the file bytes
          for (uint8_t b : bytes) h = (h ^ b) * 1099511628211ull;
          return h;
        };
        try {
          for (int t : P) {
            const bool is_img = t == img_tok_ && ki < urls.size(), is_aud = t == aud_tok_ && ka < clips.size();
            if (!is_img && !is_aud) { P2.push_back(t); continue; }
            std::vector<uint8_t> bytes = is_img ? img::from_data_url(urls[ki++]) : std::move(clips[ka++]);
            const uint64_t h = fnv(bytes) ^ (is_img ? 0 : 0x5bd1e995ull);
            auto it = img_cache_.find(h);
            if (it == img_cache_.end()) {
              const auto t0 = clk::now();
              int n = 0, gw = 0, gh = 0;
              std::shared_ptr<std::vector<float>> e;
              if (is_img) {
                const ImageRGB im = img::decode(bytes);
                e = std::make_shared<std::vector<float>>(qvision_ ? qvision_->encode(im, n, gw, gh) : vision_->encode(im, n));
                fprintf(stderr, "image %dx%d -> %d tokens, encoded in %.0f ms\n", im.w, im.h, n, ms_since(t0));
              } else {
                const std::vector<float> s16 = audio::decode_16k(bytes);
                e = std::make_shared<std::vector<float>>(audio_->encode(s16, n));
                fprintf(stderr, "audio %.1f s -> %d tokens, decoded and encoded in %.0f ms\n", s16.size() / 16000.0, n, ms_since(t0));
              }
              if (img_cache_.size() > 64) img_cache_.clear();
              it = img_cache_.emplace(h, MediaEmb{e, gw, gh}).first;
            }
            const int width = is_img ? (qvision_ ? qvision_->n_out : vision_->out_dim()) : audio_->out_dim();
            const int n = (int)(it->second.e->size() / width);
            const int beg = is_img ? img_beg_ : aud_beg_, end = is_img ? img_end_ : aud_end_;
            if (beg >= 0) P2.push_back(beg);
            // causal: audio, Qwen images, and every token of E2B / E4B (llama.cpp: "always causal"); the 12B's images
            // attend to themselves both ways
            const bool causal = !is_img || qvision_ != nullptr || eng_.hp.n_ple != 0;
            spans.push_back({P2.size(), n, it->second.e, causal, it->second.gw, it->second.gh});
            // placeholder id: identical media (encoded to the same token count) match in the prefix cache
            const int ph = -1 - (int)((h ^ ((uint64_t)n * 0x9e3779b97f4a7c15ull)) & 0x3fffffff);
            P2.insert(P2.end(), n, ph);
            if (end >= 0) P2.push_back(end);
          }
        } catch (const std::exception& e) {
          return rs.send(400, "application/json", err(std::string("media: ") + e.what()).dump());
        }
        P.swap(P2);
      }
    }
    if ((int)P.size() + 64 > eng_.ctx) {
      // Same wording and error type as llama-server, so that clients recognize a context overflow (Pi then
      // compacts the conversation and retries). The hybrid DeltaNet state cannot drop old tokens, so there is no
      // server-side context shift.
      Json e = Json::object();
      e.set("code", 400);
      e.set("message", "the request exceeds the available context size (prompt " + std::to_string(P.size()) +
                           " tokens, context " + std::to_string(eng_.ctx) + " tokens), try increasing it");
      e.set("type", "exceed_context_size_error");
      e.set("n_prompt_tokens", (int)P.size());
      e.set("n_ctx", eng_.ctx);
      Json r = Json::object();
      r.set("error", e);
      return rs.send(400, "application/json", r.dump());
    }
    if (max_tokens <= 0 || (int)P.size() + max_tokens > eng_.ctx) max_tokens = eng_.ctx - (int)P.size();

    std::lock_guard<std::mutex> lock(mu_);
    busy_ = true;
    struct Idle { std::atomic<bool>& b; ~Idle() { b = false; } } idle{busy_};
    const std::string id = "chatcmpl-arc" + std::to_string(++counter_);
    const long long created = (long long)time(nullptr);
    if (stream) rs.sse_begin();

    // ---- prompt processing (with prefix reuse)
    const auto t_prompt = clk::now();
    // end of the system prompt (the start of the second turn; Gemma's first turn follows <bos>): every conversation
    // of a client (Pi: its system prompt and tool list, ~4K tokens) shares it
    size_t sys_cut = 0;
    {
      size_t i0 = 0;
      while (i0 < P.size() && i0 < 2 && P[i0] != im_start_) ++i0;
      if (i0 + 1 < P.size() && P[i0] == im_start_ && tok_.piece(P[i0 + 1]) == "system")
        for (size_t i = i0 + 1; i < P.size(); ++i)
          if (P[i] == im_start_) { sys_cut = i; break; }
    }
    switch_conversation(P);
    load_prefix(P, sys_cut, folder);
    const size_t start = prepare_prefix(P);
    cur_saved_ = false;
    // start of the last turn header "<|im_start|>assistant\n" (Gemma: "<|turn>model\n"). The next request of the
    // conversation diverges after it (the reply as the template renders it) or, after an aborted or failed reply that
    // the client drops (Pi does), right inside it: a snapshot here serves both.
    size_t gen_cut = 0;
    for (size_t i = P.size(); i-- > 0;)
      if (P[i] == im_start_) { gen_cut = i; break; }
    // Qwen: the start of the first assistant turn after the user's last message (tool results come as user messages
    // starting with <tool_response>, which do not count); a snapshot there survives the turn's tool calls
    size_t turn_cut = 0;
    if (!gemma_ && im_start_ >= 0)
      for (size_t i = gen_cut; i-- > 0;) {
        if (P[i] != im_start_ || i + 1 >= P.size() || tok_.piece(P[i + 1]) != "user") continue;
        std::string head;
        for (size_t k = i + 2; k < std::min(P.size(), i + 8); ++k) head += tok_.piece(P[k]);
        if (head.rfind("\n<tool_response>", 0) == 0) continue;
        for (size_t j = i + 1; j <= gen_cut; ++j)
          if (P[j] == im_start_) { turn_cut = j; break; }
        break;
      }
    std::vector<float> logits;
    size_t done = start;
    // The client going away (Pi: Esc) cancels the prompt processing within a layer (Engine::cancel): the engine keeps
    // the blocks it completed, so a request that continues the prompt goes on from there.
    bool gone = false;
    auto client_gone = [&] { return gone || (gone = rs.client_gone()); };
    auto report = [&](size_t d) {
      if (stream && !gone && !rs.sse_comment("prefill " + std::to_string(d) + "/" + std::to_string(P.size()))) gone = true;
    };
    struct Unhook {  // the engine's callbacks refer to this request
      Engine& e;
      ~Unhook() { e.cancel = nullptr; e.progress = nullptr; }
    } unhook{eng_};
    eng_.cancel = client_gone;
    // runs P[done, upto); false when cancelled (cached_ then matches the engine again)
    auto feed_tokens = [&](size_t upto) {
      try {
        if (!eng_.w8_resident && upto > done) {
          // just-in-time weights: one call, so the engine runs the prompt layer by layer (each matrix decoded once);
          // progress is reported per layer
          const size_t base = done;
          eng_.progress = [&](int d, int) { report(base + d); };
          logits = eng_.forward(P.data() + done, (int)(upto - done));
          eng_.progress = nullptr;
          cached_.insert(cached_.end(), P.begin() + done, P.begin() + upto);
          done = upto;
          report(done);
          return true;
        }
        while (done < upto) {
          if (client_gone()) return false;
          const size_t n = std::min<size_t>(upto - done, eng_.max_chunk);
          logits = eng_.forward(P.data() + done, (int)n);
          cached_.insert(cached_.end(), P.begin() + done, P.begin() + done + n);
          done += n;
          report(done);
        }
        return true;
      } catch (const Engine::Cancelled&) {
        eng_.progress = nullptr;
        if (eng_.pos < (int)done) cached_.clear();  // (Gemma: reset)
        else cached_.insert(cached_.end(), P.begin() + done, P.begin() + eng_.pos);
        done = eng_.pos;
        gone = true;
        return false;
      }
    };
    // text runs go through feed_tokens, media spans through the embedding path: an image as one non-causal chunk,
    // audio causally in pieces of up to max_chunk rows
    // Qwen M-RoPE: an image of gw x gh tokens advances the rotary position by max(gw, gh), so the text after it runs
    // at cache position + shift; the engine's rope_shift follows the position processing is at (also for generation)
    auto shift_at = [&](size_t p) {
      int sh = 0;
      for (const auto& s2 : spans)
        if (s2.gw > 0 && s2.start + s2.n <= p) sh += std::max(s2.gw, s2.gh) - s2.n;
      return sh;
    };
    eng_.rope_shift = shift_at(done);
    auto feed = [&](size_t upto) {
      while (done < upto) {
        const ImgSpan* sp = nullptr;
        for (const auto& s2 : spans)
          if (s2.start + s2.n > done && s2.start < upto) { sp = &s2; break; }
        eng_.rope_shift = shift_at(done);
        if (!sp) return feed_tokens(upto);
        if (done < sp->start && !feed_tokens(sp->start)) return false;
        if (client_gone()) return false;
        const int width = (int)(sp->embd->size() / sp->n);
        for (int o = 0; o < sp->n;) {
          const int k = sp->causal ? std::min(eng_.max_chunk, sp->n - o) : sp->n;
          std::vector<int> rp;
          if (sp->gw > 0) {  // (t, h, w) = start + shift + (0, row, column) over the merged-token grid
            const int t = (int)sp->start + shift_at(sp->start);
            rp.resize((size_t)k * 3);
            for (int i = 0; i < k; ++i) {
              const int r = o + i;
              rp[3 * i] = t;
              rp[3 * i + 1] = t + r / sp->gw;
              rp[3 * i + 2] = t + r % sp->gw;
            }
          }
          logits = eng_.forward_embd(sp->embd->data() + (size_t)o * width, k, !sp->causal, sp->start + o + k >= P.size(),
                                     rp.empty() ? nullptr : rp.data());
          o += k;
        }
        cached_.insert(cached_.end(), P.begin() + sp->start, P.begin() + sp->start + sp->n);
        done = sp->start + sp->n;
        report(done);
      }
      eng_.rope_shift = shift_at(done);
      return true;
    };
    bool ok = true;  // the prompt was processed up to where the stage aimed
    if (sys_cut > done && sys_cut < gen_cut) {
      ok = feed(sys_cut);
      if (ok && !gemma_) eng_.snapshot_host();  // (Gemma resumes anywhere in its cache: rewind())
      if (ok) save_prefix(P, sys_cut, folder);
    }
    if (ok && !client_gone() && turn_cut > done && turn_cut < gen_cut) {
      ok = feed(turn_cut);
      if (ok) eng_.snapshot(2);
    }
    if (ok && !client_gone() && gen_cut > done && gen_cut < P.size()) {
      ok = feed(gen_cut);
      if (ok) eng_.snapshot(0);
      if (ok && turn_cut == gen_cut) eng_.snapshot(2);
    }
    if (ok && !client_gone()) ok = feed(P.size());
    if (ok && done == P.size()) eng_.snapshot(1);
    eng_.cancel = nullptr;  // (generation: see Engine::cancel)
    last_gen_cut_ = (int)gen_cut;
    const double prompt_ms = ms_since(t_prompt);
    if (!ok || client_gone()) {
      fprintf(stderr, "[%s] cancelled by the client after %.1f s: prompt %zu tok (cached %zu), processed up to %zu\n",
              id.c_str(), prompt_ms / 1000, P.size(), start, done);
      return;
    }

    // ---- generation
    Sampler sampler(sp);
    std::unique_ptr<chat::OutputParserBase> parser_p;
    if (gemma_) {  // (the prompt may end with an opened thought block: Gemma 4 after a tool response, thinking on)
      static const std::string opened = "<|channel>thought\n";
      const bool in_thought = prompt.size() >= opened.size() && prompt.compare(prompt.size() - opened.size(), opened.size(), opened) == 0;
      parser_p = std::make_unique<gemma4::OutputParser>(in_thought, tools);
    }
    else parser_p = std::make_unique<chat::OutputParser>(co.enable_thinking, tools);
    chat::OutputParserBase& parser = *parser_p;
    std::vector<int> gen;
    std::string text, dr, dc;
    std::string finish = "length";
    const auto t_gen = clk::now();
    Json lp_all = Json::array(), lp_pending = Json::array();  // logprobs entries: all, and not yet streamed
    auto lp_json = [&](const Json& entries) {
      Json l = Json::object();
      l.set("content", entries);
      return l;
    };
    auto send_delta = [&](const std::string& r, const std::string& c) {
      if (r.empty() && c.empty()) return true;
      Json delta = Json::object();
      if (!r.empty()) delta.set("reasoning_content", r);
      if (!c.empty()) delta.set("content", c);
      Json ch = chunk(id, created, delta, Json());
      if (want_lp && lp_pending.size()) {
        Json cs = Json::array(), c0 = ch["choices"][0];
        c0.set("logprobs", lp_json(lp_pending));
        cs.push(c0);
        ch.set("choices", cs);
        lp_pending = Json::array();
      }
      return rs.sse_event(ch.dump());
    };
    // the token's entry: its log probability and the most likely alternatives, from the raw logits
    auto lp_entry = [&](int tok, const std::vector<float>& lg) {
      const float mx = *std::max_element(lg.begin(), lg.end());
      double z = 0;
      for (float v : lg) z += std::exp((double)v - mx);
      const double lse = mx + std::log(z);
      auto one = [&](int t) {
        Json e = Json::object();
        const std::string piece = tok_.piece(t);
        e.set("token", utf8_safe(piece));  // (a token can be part of a character: "bytes" has it exactly)
        e.set("logprob", (double)lg[t] - lse);
        Json bytes = Json::array();
        for (unsigned char b : piece) bytes.push((int)b);
        e.set("bytes", bytes);
        return e;
      };
      Json e = one(tok);
      Json tops = Json::array();
      if (top_lp > 0) {
        std::vector<int> idx(lg.size());
        for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int)i;
        std::partial_sort(idx.begin(), idx.begin() + top_lp, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
        for (int i = 0; i < top_lp; ++i) tops.push(one(idx[i]));
      }
      e.set("top_logprobs", tops);
      return e;
    };
    // Tool calls are streamed while they are generated, as OpenAI does: the first delta of a call carries its id and
    // name, the following ones pieces of its arguments JSON (clients concatenate them per index).
    auto call_id = [&](size_t k) { return "call_" + std::to_string(counter_.load()) + "_" + std::to_string(k); };
    std::vector<std::string> tc_sent;  // arguments sent so far, per call
    auto send_tool_deltas = [&](const std::string& t, bool fin) {
      const auto pcs = parser.partial_tool_calls(t, fin);
      for (size_t k = 0; k < pcs.size(); ++k) {
        const bool first = k >= tc_sent.size();
        if (first) tc_sent.emplace_back();
        const std::string& a = pcs[k].args;
        if (!first && a.size() <= tc_sent[k].size()) continue;
        if (a.compare(0, tc_sent[k].size(), tc_sent[k]) != 0) {  // cannot happen: the arguments only grow
          fprintf(stderr, "[%s] tool call %zu: streamed arguments diverged\n", id.c_str(), k);
          continue;
        }
        Json fn = Json::object();
        if (first) fn.set("name", pcs[k].name);
        fn.set("arguments", a.substr(tc_sent[k].size()));
        Json tc = Json::object();
        tc.set("index", (int)k);
        if (first) {
          tc.set("id", call_id(k));
          tc.set("type", "function");
        }
        tc.set("function", fn);
        tc_sent[k] = a;
        Json calls = Json::array();
        calls.push(tc);
        Json d = Json::object();
        d.set("tool_calls", calls);
        if (!rs.sse_event(chunk(id, created, d, Json()).dump())) return false;
      }
      return true;
    };
    // text up to a possible beginning of a stop string (it would be cut off, so it must not reach the tool calls)
    auto before_stops = [&](const std::string& t) {
      size_t cut = t.size();
      for (const auto& s : stops) {
        if (s.empty()) continue;
        for (size_t k = std::min(s.size() - 1, t.size()); k > 0; --k)
          if (t.compare(t.size() - k, k, s, 0, k) == 0) { cut = std::min(cut, t.size() - k); break; }
      }
      return t.substr(0, cut);
    };
    if (stream) {
      Json d0 = Json::object();
      d0.set("role", "assistant");
      d0.set("content", "");
      rs.sse_event(chunk(id, created, d0, Json()).dump());
    }
    const std::vector<int> close_think = tok_.encode(gemma_ ? "<channel|>" : "\n</think>\n\n", true);
    bool in_think = co.enable_thinking;
    int think_n = 0;
    std::vector<int> forced;  // tokens to emit instead of sampling (budget-forced end of reasoning)
    // Speculative decoding with the MTP block: after a token is chosen, the block drafts the following one and the
    // model processes both in one pass. The next token is still sampled from the model's own distribution; the
    // draft is kept only if it is exactly that token, so the output distribution is unchanged.
    const bool spec = eng_.has_mtp;
    const size_t V = eng_.hp.n_vocab;
    std::vector<int> drafts;  // drafts processed by the last verification, checked against the next sampled tokens
    size_t acc = 0;           // how many of them matched so far
    std::vector<float> vrows; // logits of the verification rows
    int n_drafts = 0, n_accepted = 0;
    size_t call_from = std::string::npos;  // where the open tool call's text starts (after <|tool_call>)
    int n_tool_tok = 0, n_tool_fixed = 0;   // tokens drawn inside tool calls; of them, a refused first choice
    for (int i = 0; i < max_tokens; ++i) {
      int t;
      if (!forced.empty()) {
        t = forced.front();
        forced.erase(forced.begin());
      } else {
        // the answer has begun: no reasoning, or the reasoning block is closed
        const size_t tclose = co.enable_thinking ? text.find(think_close_) : 0;
        const bool answering = tclose != std::string::npos;
        if (answering && force_tool_on) {
          force_tool_on = false;
          forced = tok_.encode(force_tool, true);
        }
        if (!forced.empty()) {
          t = forced.front();
          forced.erase(forced.begin());
        } else if (tcc && call_from != std::string::npos && (gemma_ || answering)) {  // (Qwen calls only after reasoning)
          // inside a tool call: the most likely tokens that keep the call valid (a refused first choice is counted)
          const std::string call = text.substr(call_from);
          bool first = true;
          t = sampler.sample_allowed(logits, gen, [&](int tok) {
            const std::string pc = tok_.piece(tok);
            bool ok = !pc.empty() && (!tok_.is_special(tok) || tcc->special_ok(pc)) && tcc->check(call + pc) != jcon::FAIL;
            if (first && !ok) ++n_tool_fixed;
            first = false;
            return ok;
          });
          ++n_tool_tok;
          if (t < 0) t = sampler.sample(logits, gen);
        } else if (answering && jc) {
          const std::string ans = text.substr(co.enable_thinking ? tclose + think_close_.size() : 0);
          if (jc->check(ans) == jcon::DONE) {
            t = stop_ids_.empty() ? tok_.eos : stop_ids_.front();  // the value is complete: end the answer
          } else {
            t = sampler.sample_allowed(logits, gen, [&](int tok) {
              if (tok_.is_special(tok)) return false;  // (control tokens are never part of the JSON text)
              const std::string pc = tok_.piece(tok);
              return !pc.empty() && jc->check(ans + pc) != jcon::FAIL;
            });
            if (t < 0) t = stop_ids_.empty() ? tok_.eos : stop_ids_.front();
          }
        } else {
          t = sampler.sample(logits, gen);
        }
      }
      if (want_lp && std::find(stop_ids_.begin(), stop_ids_.end(), t) == stop_ids_.end()) {
        Json e = lp_entry(t, logits);
        lp_all.push(e);
        lp_pending.push(e);
      }
      if (in_think) {
        if (text.find(think_close_) != std::string::npos) in_think = false;
        else if (think_budget >= 0 && ++think_n > think_budget && forced.empty()) {
          forced = close_think;  // budget exhausted: close the reasoning block and let the model answer
          t = forced.front();
          forced.erase(forced.begin());
          in_think = false;
        }
      }
      bool draft_ok = false;  // t is an accepted draft: already processed by the model
      if (!drafts.empty()) {
        if (acc < drafts.size() && t == drafts[acc]) {
          draft_ok = true;
          ++acc;
          ++n_accepted;
          cached_.push_back(t);
        }
        if (!draft_ok || acc == drafts.size()) {  // verification resolved
          eng_.verify_done((int)acc);
          drafts.clear();
        }
      }
      if (std::find(stop_ids_.begin(), stop_ids_.end(), t) != stop_ids_.end()) { finish = "stop"; break; }
      gen.push_back(t);
      const size_t before = text.size();
      text += tok_.piece(t);
      if (tcc) {  // the open tool call, followed piece by piece (the markers are single tokens; a little slack anyway)
        const size_t from = before > 16 ? before - 16 : 0;
        if (call_from == std::string::npos) {
          const size_t o = text.find(tcc->open(), from);
          if (o != std::string::npos) call_from = o + strlen(tcc->open());
        }
        if (call_from != std::string::npos && text.find(tcc->close(), std::max(from, call_from)) != std::string::npos)
          call_from = std::string::npos;
      }
      bool hit_stop = false;
      for (const auto& s : stops)
        if (!s.empty() && text.size() >= s.size() && text.find(s, text.size() > s.size() + 16 ? text.size() - s.size() - 16 : 0) != std::string::npos) {
          text = text.substr(0, text.find(s));
          hit_stop = true;
        }
      if (stream) {
        parser.update(text, false, dr, dc);
        if (!send_delta(dr, dc) || !send_tool_deltas(before_stops(text), false)) gone = true;
      }
      if (hit_stop) { finish = "stop"; break; }
      // the client went away (also when nothing was sent for this token): stop before the next pass; the drafts in
      // flight are settled below, so the engine holds exactly the tokens in cached_
      if (client_gone()) { finish = "cancelled"; break; }
      if ((int)cached_.size() + 1 >= eng_.ctx) break;
      if (draft_ok) {  // the logits after an accepted draft are already known
        logits.assign(vrows.begin() + acc * V, vrows.begin() + (acc + 1) * V);
        continue;
      }
      if (spec && (int)cached_.size() + 4 < eng_.ctx && i + 1 < max_tokens) {
        drafts = eng_.mtp_draft(t, std::min(eng_.opt.mtp_drafts, max_tokens - i - 1));
        const auto& lv = eng_.verify(t, drafts);
        cached_.push_back(t);
        vrows.assign(lv.begin(), lv.begin() + (1 + drafts.size()) * V);
        logits.assign(vrows.begin(), vrows.begin() + V);
        acc = 0;
        n_drafts += (int)drafts.size();
      } else {
        logits = eng_.forward(&t, 1);
        cached_.push_back(t);
      }
    }
    if (!drafts.empty()) eng_.verify_done((int)acc);  // the loop ended before all drafts were decided
    const double gen_ms = ms_since(t_gen);
    if (finish == "cancelled") {
      fprintf(stderr, "[%s] cancelled by the client after %zu generated tokens (%.1f s) | prompt %zu tok (cached %zu) %.1f s\n",
              id.c_str(), gen.size(), gen_ms / 1000, P.size(), start, prompt_ms / 1000);
      return;
    }
    if (stream) {
      parser.update(text, true, dr, dc);
      send_delta(dr, dc);
      send_tool_deltas(text, true);
    }
    const auto calls = parser.tool_calls(text);
    if (!calls.empty()) finish = "tool_calls";
    // ARC_LOG_TOOLCALLS=file: every answer that writes a tool call, raw text and parsed calls (to see what a model
    // wrote when a call fails), one JSON line each
    if (const char* lf = getenv("ARC_LOG_TOOLCALLS")) {
      if (text.find(gemma_ ? "<|tool_call>" : "<tool_call>") != std::string::npos) {
        Json j = Json::object(), pc = Json::array();
        for (const auto& c : calls) {
          Json o = Json::object();
          o.set("name", c.name);
          o.set("arguments", c.arguments);
          pc.push(o);
        }
        j.set("time", (double)time(nullptr));
        j.set("model", model_id_);
        j.set("id", id);
        j.set("raw", utf8_safe(text));
        j.set("calls", pc);
        j.set("constraint_fixed", n_tool_fixed);
        static std::mutex lm;
        std::lock_guard<std::mutex> g(lm);
        if (FILE* f = fopen(lf, "ab")) {
          const std::string line = j.dump() + "\n";
          fwrite(line.data(), 1, line.size(), f);
          fclose(f);
        }
      }
    }

    Json usage = Json::object();
    usage.set("prompt_tokens", P.size());
    usage.set("completion_tokens", gen.size());
    usage.set("total_tokens", P.size() + gen.size());
    Json ptd = Json::object();
    ptd.set("cached_tokens", start);
    usage.set("prompt_tokens_details", ptd);
    Json timings = Json::object();
    timings.set("cache_n", start);
    timings.set("prompt_n", P.size() - start);
    timings.set("prompt_ms", prompt_ms);
    timings.set("prompt_per_second", (P.size() - start) / std::max(prompt_ms, 1e-3) * 1000.0);
    timings.set("predicted_n", gen.size());
    timings.set("predicted_ms", gen_ms);
    timings.set("predicted_per_second", gen.size() / std::max(gen_ms, 1e-3) * 1000.0);
    fprintf(stderr, "[%s] prompt %zu tok (cached %zu) %.1f s (%.1f t/s) | gen %zu tok %.1f s (%.2f t/s) | %s", id.c_str(),
            P.size(), start, prompt_ms / 1000, (P.size() - start) / std::max(prompt_ms, 1e-3) * 1000, gen.size(), gen_ms / 1000,
            gen.size() / std::max(gen_ms, 1e-3) * 1000, finish.c_str());
    if (n_drafts) fprintf(stderr, " | MTP %d/%d drafts accepted", n_accepted, n_drafts);
    if (n_tool_fixed) fprintf(stderr, " | tool call: %d of %d tokens corrected", n_tool_fixed, n_tool_tok);
    fprintf(stderr, "\n");
    if (n_drafts) {
      timings.set("draft_n", n_drafts);
      timings.set("draft_n_accepted", n_accepted);
    }
    {
      std::lock_guard<std::mutex> lk(stats_mu_);
      ++n_req_;
      last_timings_ = timings;
      last_timings_.set("finish", finish);
      last_timings_.set("time", (double)time(nullptr));
    }

    if (stream) {
      Json fin = chunk(id, created, Json::object(), Json(finish));
      fin.set("timings", timings);
      rs.sse_event(fin.dump());
      if (usage_in_stream) {
        Json u = Json::object();
        u.set("id", id);
        u.set("object", "chat.completion.chunk");
        u.set("created", (double)created);
        u.set("model", model_id_);
        u.set("choices", Json::array());
        u.set("usage", usage);
        rs.sse_event(u.dump());
      }
      rs.sse_event("[DONE]");
      rs.sse_end();
      return;
    }
    Json tc_json = Json::array();
    for (size_t k = 0; k < calls.size(); ++k) {
      Json fn = Json::object();
      fn.set("name", calls[k].name);
      fn.set("arguments", calls[k].arguments.dump());
      Json tc = Json::object();
      tc.set("index", (int)k);
      tc.set("id", call_id(k));
      tc.set("type", "function");
      tc.set("function", fn);
      tc_json.push(tc);
    }
    Json msg = Json::object();
    msg.set("role", "assistant");
    const std::string content = parser.content_text(text), reasoning = parser.reasoning_text(text);
    msg.set("content", content.empty() && tc_json.size() ? Json() : Json(content));
    if (!reasoning.empty()) msg.set("reasoning_content", reasoning);
    if (tc_json.size()) msg.set("tool_calls", tc_json);
    Json choice = Json::object();
    choice.set("index", 0);
    choice.set("message", msg);
    if (want_lp) choice.set("logprobs", lp_json(lp_all));
    choice.set("finish_reason", finish);
    Json choices = Json::array();
    choices.push(choice);
    Json r = Json::object();
    r.set("id", id);
    r.set("object", "chat.completion");
    r.set("created", (double)created);
    r.set("model", model_id_);
    r.set("choices", choices);
    r.set("usage", usage);
    r.set("timings", timings);
    rs.send(200, "application/json", r.dump());
  }

  Json chunk(const std::string& id, long long created, const Json& delta, const Json& finish) {
    Json c = Json::object();
    c.set("index", 0);
    c.set("delta", delta);
    c.set("finish_reason", finish);
    Json choices = Json::array();
    choices.push(c);
    Json r = Json::object();
    r.set("id", id);
    r.set("object", "chat.completion.chunk");
    r.set("created", (double)created);
    r.set("model", model_id_);
    r.set("choices", choices);
    return r;
  }
};

// ---------------------------------------------------------------- model catalog
// The GGUF models in the Hugging Face cache and in arcinfer's model folder: name (org/repo), file, path, size,
// architecture, trained context, whether an image projector (mmproj-*) or a Gemma drafter (mtp-*) sits next to it,
// Qwen MTP layers, and the model id clients use (the alias; the six models arcinfer was tuned on keep their short
// ids). Projectors, drafters and the later parts of split files are not models.
struct ModelInfo {
  std::string name, file, path, arch, alias, shortcut;
  uint64_t bytes = 0;
  int64_t ctx_train = 0;
  // vision / mtp: what this entry loads. A file with an MTP block or drafter and / or an image projector is listed
  // once per combination (expand_variants): the full one under the file's id, the others with -mtp (MTP, no
  // vision), -vision (vision, no MTP) and -std (neither). base: the file's id; tags: "MTP + vision", ...
  bool supported = false, vision = false, mtp = false;
  std::string base, tags;
};
// Two spellings of one model file (separators, case, a path given on the command line vs one found by the scan).
static std::string lower_ascii(std::string s) {
  for (auto& c : s) c = (char)tolower((unsigned char)c);
  return s;
}
static bool same_file(const std::string& a, const std::string& b) {
  if (a == b) return true;
  std::error_code ec;
  const bool eq = std::filesystem::equivalent(a, b, ec);
  return !ec && eq;
}
// (defined below) the auto context of a model file with / without its MTP block and image encoders
static int auto_ctx_tokens(const std::string& model, bool mtp, bool vision);

static std::vector<ModelInfo> scan_models() {
  namespace fs = std::filesystem;
  std::string hub;
  if (const char* v = getenv("HF_HUB_CACHE")) hub = v;
  else if (const char* h = getenv("HF_HOME")) hub = std::string(h) + "/hub";
  else if (const char* u = getenv("USERPROFILE")) hub = std::string(u) + "/.cache/huggingface/hub";
  // (prio: 0 the Hugging Face cache and arcinfer's own folders, 1+ the extra model folders in their order, so a model
  //  kept in both places loads from the first)
  struct Cand { std::string name, file; fs::path path; fs::file_time_type t; int prio = 0; };
  std::vector<Cand> cands;
  auto lower = [](std::string s) { for (auto& c : s) c = (char)tolower((unsigned char)c); return s; };
  auto is_model = [&](const fs::path& p) {
    const std::string f = lower(p.filename().string());
    // (projectors: mmproj-*.gguf, or *-mmproj.gguf as Google names them)
    if (p.extension() != ".gguf" || f.find("mmproj") != std::string::npos || f.rfind("mtp-", 0) == 0) return false;
    const size_t k = f.find("-of-");  // split files: only part 1
    return k == std::string::npos || k < 6 || f.compare(k - 5, 5, "00001") == 0;
  };
  std::error_code ec;
  // a Hugging Face cache layout (models--org--repo\snapshots\<rev>\...); in extra folders also loose .gguf files
  auto scan_hub = [&](const fs::path& dir, int prio) {
    for (const auto& d : fs::directory_iterator(dir, ec)) {
      const std::string dn = d.path().filename().string();
      if (prio > 0 && is_model(d.path()) && d.is_regular_file(ec)) {
        cands.push_back({"local/" + d.path().stem().string(), dn, d.path(), d.last_write_time(ec), prio});
        continue;
      }
      if (!d.is_directory(ec) || dn.rfind("models--", 0) != 0) continue;
      std::string name = dn.substr(8);
      const size_t sep = name.find("--");
      if (sep != std::string::npos) name.replace(sep, 2, "/");
      for (const auto& e : fs::recursive_directory_iterator(d.path() / "snapshots", ec))
        if (is_model(e.path())) cands.push_back({name, e.path().filename().string(), e.path(), e.last_write_time(ec), prio});
    }
  };
  if (!hub.empty()) scan_hub(hub, 0);
  // extra model folders (e.g. a second drive): ARC_MODEL_DIRS (';'-separated) and %LOCALAPPDATA%\arcinfer\model-dirs.txt
  // (one folder per line, '#' comments); each in the Hugging Face cache layout or with .gguf files directly in it
  {
    std::vector<std::string> extra;
    auto add = [&](std::string d) {
      while (!d.empty() && (d.back() == '\r' || d.back() == ' ' || d.back() == '\t')) d.pop_back();
      while (!d.empty() && (d.front() == ' ' || d.front() == '\t')) d.erase(d.begin());
      if (!d.empty() && d[0] != '#') extra.push_back(d);
    };
    if (const char* v = getenv("ARC_MODEL_DIRS")) {
      std::string all = v;
      for (size_t a = 0, b; a <= all.size(); a = b + 1) {
        b = all.find(';', a);
        if (b == std::string::npos) b = all.size();
        add(all.substr(a, b - a));
      }
    }
    if (const char* la = getenv("LOCALAPPDATA")) {
      std::ifstream f(fs::path(la) / "arcinfer" / "model-dirs.txt");
      for (std::string line; std::getline(f, line);) add(line);
    }
    // (a folder on a drive that hangs, e.g. a USB stick after the laptop slept, would stall every model list and so
    //  every server start: each folder is first listed by a detached thread and skipped if that takes over 2 s)
    auto reachable = [](const fs::path& d) {
      auto st = std::make_shared<std::atomic<int>>(0);
      std::thread([d, st] {
        std::error_code e;
        const bool ok = fs::is_directory(d, e);
        if (ok) fs::directory_iterator it(d, e);
        st->store(ok && !e ? 1 : 2);
      }).detach();
      for (int i = 0; i < 100 && st->load() == 0; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(20));
      return st->load();
    };
    int prio = 1;
    for (const auto& d : extra) {
      const int r = reachable(fs::path(d));
      if (r == 1) scan_hub(fs::path(d), prio);
      else if (r == 0) fprintf(stderr, "model folder %s did not answer within 2 s (drive asleep or hung?): skipped\n", d.c_str());
      ++prio;
    }
  }
  if (const char* la = getenv("LOCALAPPDATA"))
    for (const auto& e : fs::directory_iterator(fs::path(la) / "arcinfer" / "models", ec))
      if (is_model(e.path())) cands.push_back({"arcinfer/" + e.path().stem().string(), e.path().filename().string(), e.path(), e.last_write_time(ec)});
#ifdef ARC_ZERO
  // arcinferzero packages (zero/gguf_pack.h): %LOCALAPPDATA%\arcinfer\zero\models\<name>\<name>.gguf
  for (const auto& d : fs::directory_iterator(zpack::packages_dir(), ec)) {
    const fs::path g = d.path() / (d.path().filename().string() + ".gguf");
    if (d.is_directory(ec) && fs::exists(g, ec))
      cands.push_back({"arcinferzero/" + d.path().filename().string(), g.filename().string(), g, fs::last_write_time(g, ec)});
  }
#endif
  // the same file in several snapshots (revisions) or folders: the first folder, then the newest
  std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) {
    return a.name != b.name ? a.name < b.name : a.file != b.file ? a.file < b.file : a.prio != b.prio ? a.prio < b.prio : a.t > b.t;
  });
  static const std::map<std::string, std::pair<std::string, std::string>> known = {
      {"Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf", {"qwen3.8-27b", "27b"}},
      {"Qwen3.6-35B-A3B-GSQ2-arc.gguf", {"qwen3.6-35b-a3b", "35b"}},
      // Unsloth's standard GGUFs: 35b = UD-Q3_K_M (IQ4_XS down experts), 35b-s = UD-Q3_K_S (smaller: longer context)
      {"Qwen3.6-35B-A3B-UD-Q3_K_M.gguf", {"qwen3.6-35b-a3b", "35b"}},
      {"Qwen3.6-35B-A3B-UD-Q3_K_S.gguf", {"qwen3.6-35b-a3b-s", "35b-s"}},
      {"Qwen3.5-9B-UD-Q4_K_XL.gguf", {"qwen3.5-9b", "9b"}},
      {"Qwen3.5-4B-UD-Q4_K_XL.gguf", {"qwen3.5-4b", "4b"}},
      {"gemma-4-12b-it-qat-q4_0.gguf", {"gemma4-12b", "g12"}},
      {"gemma-4-E4B_q4_0-it.gguf", {"gemma4-e4b", "e4b"}}};
  std::vector<ModelInfo> out;
  for (size_t i = 0; i < cands.size(); ++i) {
    const Cand& c = cands[i];
    if (i > 0 && cands[i - 1].name == c.name && cands[i - 1].file == c.file) continue;
    ModelInfo m;
    m.name = c.name;
    m.file = c.file;
    m.path = c.path.lexically_normal().string();  // (HF_HOME and the cache folder joined: one kind of separator)
    m.bytes = fs::file_size(c.path, ec);
    m.arch = "?";
    int64_t nextn = 0;
    bool gsq = false;  // qwen35moe: only GSQ 2-bit experts (tools/convert_gsq.py) have kernels
    try {
      GgufFile g(m.path);
      m.arch = g.get_str("general.architecture", "?");
      gsq = g.get_str("arcinfer.expert_format", "") == "gsq2-v1";
      m.ctx_train = g.get_int(m.arch + ".context_length", 0);
      nextn = g.get_int(m.arch + ".nextn_predict_layers", 0);
    } catch (const std::exception&) {
    }
    bool mmproj = false, drafter = false;
    for (const auto& s : fs::directory_iterator(c.path.parent_path(), ec)) {
      const std::string f = lower(s.path().filename().string());
      if (s.path().extension() != ".gguf") continue;
      if (f.find("mmproj") != std::string::npos && !mmproj) {  // images: a projector arcinfer reads
        try {
          const GgufFile g(s.path().string());
          const std::string pt = g.get_str("clip.vision.projector_type", "");
          mmproj = m.arch == "gemma4" ? pt == "gemma4uv" || pt == "gemma4v"
                                      : (m.arch == "qwen35" || m.arch == "qwen35moe") && VisionQwen3VL::usable(g);
        } catch (const std::exception&) {
        }
      }
      drafter = drafter || f.rfind("mtp-", 0) == 0;
    }
    m.supported = m.arch == "qwen35" || m.arch == "qwen35moe" || m.arch == "gemma4";  // (MoE: GSQ or standard experts)
#ifdef ARC_ZERO
    m.supported = m.supported || m.arch == "qwen4exp";  // (Qwen3.8 Flash Next: arcinferzero only, experts streamed)
#endif
    (void)gsq;
    m.vision = mmproj;  // (Gemma 4 12B: gemma4uv; E2B / E4B: gemma4v)
    m.mtp = (m.arch == "gemma4" && drafter) || nextn > 0;
    const auto k = known.find(m.file);
    if (k != known.end()) {
      m.alias = k->second.first;
      m.shortcut = k->second.second;
      // the same file name in a repository that keeps the MTP block (Unsloth's *-MTP-GGUF next to *-GGUF): its own id
      bool twin = false;  // (only then: Qwen3.5 4B / 9B come from *-MTP-GGUF repositories alone and keep their ids)
      for (const Cand& o : cands) twin = twin || (o.file == c.file && o.name != c.name);
      if (twin && nextn > 0 && m.alias.find("-mtp") == std::string::npos) {
        m.alias += "-mtp";
        if (!m.shortcut.empty()) m.shortcut += "-mtp";
      }
    } else {
      bool several = false;
      for (const Cand& o : cands) several = several || (o.name == c.name && o.file != c.file);
      std::string a = several ? fs::path(m.file).stem().string() : m.name.substr(m.name.rfind('/') + 1);
      a = lower(a);
      if (a.size() > 5 && a.compare(a.size() - 5, 5, "-gguf") == 0) a.resize(a.size() - 5);
      m.alias = a;
    }
    out.push_back(m);
  }
  // every downloaded file selectable on its own: a repeated id gets -2, -3, ..., a repeated shortcut is dropped
  for (size_t i = 0; i < out.size(); ++i) {
    const std::string base = out[i].alias;
    for (int n = 2;; ++n) {
      bool clash = false;
      for (size_t j = 0; j < i; ++j) clash = clash || out[j].alias == out[i].alias;
      if (!clash) break;
      out[i].alias = base + "-" + std::to_string(n);
    }
    for (size_t j = 0; j < i; ++j)
      if (!out[i].shortcut.empty() && out[j].shortcut == out[i].shortcut) out[i].shortcut.clear();
  }
  // the variants: the full one first (the file's id and shortcut, as before), then MTP / vision / neither
  std::vector<ModelInfo> all;
  for (const ModelInfo& m : out) {
    struct V { bool mtp, vision; const char* suffix; };
    // A projector that leaves the model little context (Qwen3.6 35B: 4,096 tokens with it, 40,960 without at 18.3 GiB)
    // is not loaded by default: the file's id is the variant without images, "-vision" the one with them.
    bool vision_default = true;
    if (m.supported && m.vision) {
      const int cv = auto_ctx_tokens(m.path, m.mtp, true), ct = auto_ctx_tokens(m.path, m.mtp, false);
      vision_default = !(cv > 0 && cv < 16384 && ct >= 2 * cv);
    }
    std::vector<V> vs;
    if (vision_default) {
      vs.push_back({m.mtp, m.vision, ""});
      if (m.supported && m.mtp && m.vision) vs.push_back({true, false, "-mtp"}), vs.push_back({false, true, "-vision"});
      if (m.supported && (m.mtp || m.vision)) vs.push_back({false, false, "-std"});
    } else {
      vs.push_back({m.mtp, false, ""});
      vs.push_back({false, true, "-vision"});
      if (m.mtp) vs.push_back({false, false, "-std"});
    }
    for (const V& v : vs) {
      ModelInfo e = m;
      e.base = m.alias;
      e.mtp = v.mtp;
      e.vision = v.vision;
      e.alias = m.alias + v.suffix;
      if (!m.shortcut.empty()) e.shortcut = m.shortcut + v.suffix;
      e.tags = v.mtp && v.vision ? "MTP + vision" : v.mtp ? "MTP" : v.vision ? "vision" : "standard";
      all.push_back(e);
    }
  }
  return all;
}

// ---------------------------------------------------------------- --ctx auto
// The largest context (a multiple of 4096, at most the trained one) for which the server's GPU memory, the ~0.22 GiB
// the driver adds and ~0.95 GiB for the desktop and other apps fit the device's global memory. A dry load (the
// engine adds up its allocations without making them: seconds, no GPU memory) gives what does not depend on the
// context; the result is remembered per model file, chunk size, MTP setting and device memory in %LOCALAPPDATA%\arcinfer\ctx-auto.txt.
static uint64_t device_global_mem() {
  static uint64_t dev = 0;
  if (!dev) {
    ClRt rt;
    if (rt.init(false)) clGetDeviceInfo(rt.device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(dev), &dev, nullptr);
    if (rt.queue) clReleaseCommandQueue(rt.queue);
    if (rt.ctx) clReleaseContext(rt.ctx);
  }
  return dev;
}
static std::string ctx_cache_file() {
  const char* la = getenv("LOCALAPPDATA");
  return std::string(la ? la : ".") + ARC_DATA "ctx-auto.txt";
}
static std::string ctx_key(const std::string& model, const EngineOptions& eo) {
  namespace fs = std::filesystem;
  std::error_code ec;
  std::string p = fs::absolute(model, ec).lexically_normal().make_preferred().string();  // one spelling per file
  for (auto& c : p) c = (char)tolower((unsigned char)c);
  return p + "|" + std::to_string(fs::file_size(model, ec)) + "|" + std::to_string(eo.chunk) +
#ifdef ARC_ZERO
         "|z" + std::to_string((int)(eo.windows_reserve_gb * 10)) + (eo.stream_experts ? "s" : "n") + "|" +
#endif
         (eo.min_chunk ? "/" + std::to_string(eo.min_chunk) : std::string()) + "|" + (eo.mtp ? "mtp" : "-") + "|" + (eo.attn_int8 ? "i8" : "f32") + "|" + std::to_string(device_global_mem()) +
         "|m9";  // engine memory layout version (m2: Q8_0 not stored twice, m4: MoE stacked router, m5: 3 MTP drafts
                 // for 27B, m6: MoE prompt GEMM staging, m7: MoE buffers shared, m8: MoE MTP 1 draft, Q3_K experts, m9: memory parts in the cache); bump on changes
}
// "ctx\tresident" lines (older lines without the second field are computed again)
// The largest context that fits, whether decoded weights stay resident, and how the GPU memory was counted (bytes):
// device total, kept free for Windows / other apps, the server's state snapshots, the encoders' reserve, the model's
// weights, MTP block / drafter, working buffers, the context (KV caches) at ctx, and the context per 1024 tokens.
struct AutoCtx {
  int ctx = 4096;
  bool resident = false;
  double dev = 0, keep = 0, snap = 0, reserve = 0, weights = 0, mtp = 0, buffers = 0, context = 0, per_1k = 0;
};
static bool cached_ctx(const std::string& key, AutoCtx& r) {
  std::ifstream f(ctx_cache_file());
  std::string line;
  while (std::getline(f, line)) {
    const size_t t = line.find('\t');
    if (t == std::string::npos || line.substr(0, t) != key) continue;
    std::vector<std::string> c;
    for (size_t a = t + 1;;) {
      const size_t b = line.find('\t', a);
      c.push_back(line.substr(a, b == std::string::npos ? std::string::npos : b - a));
      if (b == std::string::npos) break;
      a = b + 1;
    }
    if (c.size() < 2) continue;
    r.ctx = std::stoi(c[0]);
    r.resident = c[1] == "1";
    double* mem[] = {&r.dev, &r.keep, &r.snap, &r.reserve, &r.weights, &r.mtp, &r.buffers, &r.context, &r.per_1k};
    for (size_t i = 0; i < 9 && i + 2 < c.size(); ++i) *mem[i] = atof(c[i + 2].c_str());
    return true;
  }
  return false;
}
// The largest context that fits next to the weights, and whether the decoded W8 weights stay resident. They do
// when that still leaves at least min(128K, trained) tokens of context (a resident model processes a short prompt
// ~4x sooner: Qwen3.5 9B, 83 tokens: 1.05 -> 0.27 s; otherwise every request decodes every matrix again), else
// just in time with the largest context. An explicit --budget (budget_given) keeps the old rule.
// GPU bytes of the model's image / audio encoders (uploaded on the first image or clip, so the engine's own
// accounting misses them): --ctx auto keeps them free. 0 without a GPU encoder (Gemma 4 12B's run on the CPU).
static size_t encoder_reserve(const std::string& model, const std::string& mmproj_opt, int img_max) {
  if (mmproj_opt == "-") return 0;
  namespace fs = std::filesystem;
  std::string mp = mmproj_opt;
  std::error_code ec;
  if (mp.empty())
    for (const auto& de : fs::directory_iterator(fs::path(model).parent_path(), ec))
      if (lower_ascii(de.path().filename().string()).find("mmproj") != std::string::npos && de.path().extension() == ".gguf") {
        mp = de.path().string();
        break;
      }
  if (mp.empty()) return 0;
  try {
    const GgufFile m(model), g(mp);
    const std::string arch = m.get_str("general.architecture", "");
    size_t b = 0;
    if ((arch == "qwen35" || arch == "qwen35moe") && VisionQwen3VL::usable(g)) b += VisionQwen3VL::gpu_bytes(g, img_max > 0 ? img_max : 2048);
    if (arch == "gemma4") {
      if (g.get_str("clip.vision.projector_type", "") == "gemma4v") b += VisionGemma4V::gpu_bytes(g, img_max > 0 ? img_max : 1120);
      if (g.get_str("clip.audio.projector_type", "") == "gemma4a") b += AudioGemma4A::gpu_bytes(g);
    }
    return b ? b + ((size_t)64 << 20) : 0;  // (+ kernels and slack)
  } catch (const std::exception&) {
    return 0;
  }
}
static AutoCtx auto_ctx(const std::string& model, EngineOptions eo, bool budget_given = false, size_t reserve = 0) {
  namespace fs = std::filesystem;
  const std::string key = ctx_key(model, eo) + (budget_given ? "|b" + std::to_string(eo.w8_budget_gb) : "|auto") +
                          (reserve ? "|r" + std::to_string(reserve >> 20) : std::string());
  AutoCtx r;
  if (cached_ctx(key, r)) return r;
  int trained = 0;
  {
    GgufFile g(model);
    trained = (int)g.get_int(g.get_str("general.architecture") + ".context_length", 32768);
  }
  AutoCtx m1, m2;  // (memory counts of each probe)
  auto fit = [&](double budget_gb, bool& res, AutoCtx& mo) {
    EngineOptions e2 = eo;
    e2.ctx = std::min(16384, trained);
    e2.dry = true;
    e2.w8_budget_gb = budget_gb;
    Engine probe(model, e2);
    res = probe.w8_resident;
    const double gib = 1073741824.0,
                 budget = probe.device_mem() - 0.95 * gib - 0.22 * gib - probe.snapshot_bytes() - (double)reserve;
    const double base = (double)probe.gpu_bytes - (double)probe.ctx_bytes(e2.ctx);
    int ctx = 0;
    for (int c = 4096; c <= trained; c += 4096)
      if (base + probe.ctx_bytes(c) <= budget) ctx = c;
    if (trained % 4096 && base + probe.ctx_bytes(trained) <= budget) ctx = trained;
    {
      const int cc = std::max(ctx, 4096);
      const Engine::MemParts mp = probe.mem_parts(cc);
      mo.dev = (double)probe.device_mem();
      mo.keep = (0.95 + 0.22) * gib;
      mo.snap = (double)probe.snapshot_bytes();
      mo.reserve = (double)reserve;
      mo.weights = mp.weights;
      mo.mtp = mp.mtp;
      mo.buffers = mp.buffers;
      mo.context = mp.context;
      mo.per_1k = ((double)probe.ctx_bytes(cc + 4096) - (double)probe.ctx_bytes(cc)) / 4.0;
    }
    fprintf(stderr, "--ctx auto: %d tokens with %s weights (trained %d; %.2f GiB of GPU memory without the context, "
                    "%.2f GiB with it, device %.2f GiB)\n", ctx, res ? "resident" : "just-in-time", trained, base / gib,
            (base + probe.ctx_bytes(std::max(ctx, 4096))) / gib, probe.device_mem() / gib);
    return ctx;
  };
  bool res = false;
  if (budget_given) {
    const int c = std::max(4096, fit(eo.w8_budget_gb, res, m1));
    r = m1;
    r.ctx = c;
    r.resident = res;
  } else {
    const int c_res = fit(1e9, res, m1);  // everything resident
    if (res && c_res >= std::min(131072, trained)) {
      r = m1;
      r.ctx = c_res;
      r.resident = true;
    } else {
      bool r2 = false;
      const int c = std::max(4096, fit(eo.w8_budget_gb, r2, m2));
      r = m2;
      r.ctx = c;
      r.resident = r2;
    }
  }
  std::error_code ec;
  fs::create_directories(fs::path(ctx_cache_file()).parent_path(), ec);
  {
    std::ofstream o(ctx_cache_file(), std::ios::app);
    o << key << "\t" << r.ctx << "\t" << (r.resident ? 1 : 0);
    for (double v : {r.dev, r.keep, r.snap, r.reserve, r.weights, r.mtp, r.buffers, r.context, r.per_1k}) o << "\t" << (long long)v;
    o << "\n";
  }
  return r;
}

static int auto_ctx_tokens(const std::string& model, bool mtp, bool vision) {
  EngineOptions eo;
  eo.mtp = mtp;
  try {
    return auto_ctx(model, eo, false, vision ? encoder_reserve(model, "", 0) : 0).ctx;
  } catch (const std::exception&) {
    return 0;
  }
}

// --list-models: the catalog as JSON (scripts\arc-server.ps1 and pi-arc read it). ctx: the context --ctx auto chose
// for this GPU memory (with --mtp when the model has a drafter, default chunk), computed now if not known yet.
static int list_models() {
  Json out = Json::array();
  for (const ModelInfo& m : scan_models()) {
    Json j = Json::object();
    j.set("name", m.name);
    j.set("file", m.file);
    j.set("path", m.path);
    j.set("bytes", (double)m.bytes);
    j.set("arch", m.arch);
    j.set("supported", m.supported);
    j.set("ctx_train", (double)m.ctx_train);
    j.set("vision", m.vision);
    j.set("mtp", m.mtp);
    j.set("alias", m.alias);
    j.set("short", m.shortcut);
    j.set("base", m.base);
    j.set("tags", m.tags);
    EngineOptions eo;
    eo.mtp = m.mtp;
    AutoCtx a;  // computed now (dry load) if not known yet; per variant (MTP block, encoders' memory)
    a.ctx = 0;
    if (m.supported)
      try { a = auto_ctx(m.path, eo, false, m.vision ? encoder_reserve(m.path, "", 0) : 0); } catch (const std::exception&) {}
    j.set("ctx", a.ctx);
    if (m.supported) {  // the values the server picks for this variant when not given (the launcher fills them in)
      Json d = Json::object();
      const bool moe = m.arch == "qwen35moe" || m.arch == "qwen4exp", gemma = m.arch == "gemma4";
      int64_t n_embd = 0;
      std::string pt;  // image projector type
      try {
        const GgufFile g(m.path);
        n_embd = g.get_int(m.arch + ".embedding_length", 0);
      } catch (const std::exception&) {}
      if (m.vision) {
        std::error_code ec;
        for (const auto& de : std::filesystem::directory_iterator(std::filesystem::path(m.path).parent_path(), ec))
          if (lower_ascii(de.path().filename().string()).find("mmproj") != std::string::npos && de.path().extension() == ".gguf") {
            try { pt = GgufFile(de.path().string()).get_str("clip.vision.projector_type", ""); } catch (const std::exception&) {}
            break;
          }
      }
      int chunk = moe ? 2048 : a.resident ? 512 : 4096;
      if (gemma && m.vision) chunk = std::max(chunk, 2048);  // (an image is one chunk)
      d.set("chunk", chunk);
#ifdef ARC_ZERO
      d.set("mtp_drafts", !m.mtp ? 0 : gemma ? 2 : moe || n_embd >= 5120 || n_embd <= 2560 ? 3 : 2);
#else
      d.set("mtp_drafts", !m.mtp ? 0 : moe ? 1 : !gemma && n_embd >= 5120 ? 3 : 2);
#endif
      d.set("image_min", !m.vision ? 0 : gemma ? 1120 : 256);
      d.set("image_max", !m.vision ? 0 : pt == "gemma4v" ? 1120 : 2048);
      d.set("think_budget", 8192);
      d.set("prefix_cache_gb", 8);
      d.set("prompt_cache_gb", 0);
      d.set("resident_weights", a.resident);
      j.set("defaults", d);
    }
    if (a.dev > 0) {  // how --ctx auto counted the GPU memory (bytes)
      Json mem = Json::object();
      mem.set("device", a.dev);
      mem.set("keep_free", a.keep);
      mem.set("snapshots", a.snap);
      mem.set("encoders", a.reserve);
      mem.set("weights", a.weights);
      mem.set("mtp", a.mtp);
      mem.set("buffers", a.buffers);
      mem.set("context", a.context);
      mem.set("context_per_1k", a.per_1k);
      mem.set("resident", a.resident);
      j.set("memory", mem);
    }
    out.push(j);
  }
  printf("%s\n", out.dump().c_str());
  return 0;
}

// ---------------------------------------------------------------- loading a model
struct ServeOpts {
  EngineOptions eo;
  bool ctx_auto = false;
  std::string webui, mmproj, sess_dir;
  int think_budget = -1;
  bool preserve_thinking = false;
  double sess_gb = 0.0;
  bool prefix_cache = true;
  double prefix_gb = 8.0;  // all models' saved system prompts together (0 = no limit)
  int img_min = 0, img_max = 0;  // image tokens per image (0 = the encoder's default)
  bool budget_given = false;     // --budget: no automatic choice of resident weights
};
struct Loaded {  // destroyed in reverse order: the server before the tokenizer and engine it refers to
  std::unique_ptr<Engine> eng;
  std::unique_ptr<Tokenizer> tok;
  std::unique_ptr<Server> srv;
  std::string alias, path;
};
static std::unique_ptr<Loaded> load_model(const std::string& model, const std::string& alias, ServeOpts so) {
  auto L = std::make_unique<Loaded>();
  L->alias = alias;
  L->path = model;
  fprintf(stderr, "arcinfer-server: loading %s\n", model.c_str());
  EngineOptions eo = so.eo;
  // Gemma 4 with an image projector: an image's tokens attend to each other both ways, so one image is one forward
  // step; the steps must hold the largest image (resident E4B weights otherwise step through prompts in 512s)
  if (so.mmproj != "-") {
    bool gemma = false, proj = false;
    try {  // (E2B / E4B, with per-layer embeddings, process images causally in any steps)
      const GgufFile g(model);
      gemma = g.get_str("general.architecture", "") == "gemma4" && g.get_int("gemma4.embedding_length_per_layer_input", 0) == 0;
    } catch (const std::exception&) {
    }
    std::error_code ec;
    if (gemma && so.mmproj.empty())
      for (const auto& de : std::filesystem::directory_iterator(std::filesystem::path(model).parent_path(), ec))
        proj = proj || (lower_ascii(de.path().filename().string()).find("mmproj") != std::string::npos && de.path().extension() == ".gguf");
    if (gemma && (proj || !so.mmproj.empty())) eo.min_chunk = std::max(so.img_max, 2048);
  }
  if (so.ctx_auto) {
    const size_t enc = encoder_reserve(model, so.mmproj, so.img_max);
    if (enc) fprintf(stderr, "--ctx auto: %.2f GiB kept for the image / audio encoders\n", enc / 1073741824.0);
    const AutoCtx a = auto_ctx(model, eo, so.budget_given, enc);
    eo.ctx = a.ctx;
    if (!so.budget_given) eo.w8_budget_gb = a.resident ? 1e9 : eo.w8_budget_gb;
  }
#ifdef ARC_ZERO
  if (so.mmproj != "-") eo.stream_reserve = encoder_reserve(model, so.mmproj, so.img_max);  // (streamed experts' cache)
#endif
  L->eng = std::make_unique<Engine>(model, eo);
  Engine& eng = *L->eng;
  L->tok = std::make_unique<Tokenizer>(eng.gguf());
  fprintf(stderr, "loaded in %.1f s | %d layers | ctx %d | chunk %d | GPU %.2f GiB | weights %s\n", eng.load_ms / 1000,
          eng.hp.n_layer, eng.ctx, eng.max_chunk, eng.gpu_bytes / 1073741824.0, eng.w8_resident ? "W8 resident" : "W8 just-in-time");
  { int w = L->tok->encode("Hi", false)[0]; eng.forward(&w, 1); eng.reset(); }  // compile kernels before serving
  L->srv = std::make_unique<Server>(eng, *L->tok, alias, so.webui);
  Server& srv = *L->srv;
  std::string mmproj = so.mmproj;
  if (!eng.hp.gemma && mmproj != "-") {
    // Qwen vision (dense and MoE): --mmproj, or an mmproj-*.gguf next to the model with a Qwen3-VL projector arcinfer runs
    if (mmproj.empty())
      for (const auto& de : std::filesystem::directory_iterator(std::filesystem::path(model).parent_path()))
        if (lower_ascii(de.path().filename().string()).find("mmproj") != std::string::npos && de.path().extension() == ".gguf") { mmproj = de.path().string(); break; }
    if (!mmproj.empty()) {
      try {
        auto v = std::make_unique<VisionQwen3VL>(mmproj, eng);
        if (v->n_out != eng.hp.n_embd)  // (a projector of another model size)
          throw std::runtime_error("the projector outputs " + std::to_string(v->n_out) + " values per token, the model takes " +
                                   std::to_string(eng.hp.n_embd));
        if (so.img_min > 0) v->min_tokens = so.img_min;
        if (so.img_max > 0) v->max_tokens = so.img_max;
        v->min_tokens = std::min(v->min_tokens, v->max_tokens);
        fprintf(stderr, "vision: %s (Qwen3-VL, %d layers, %d..%d tokens per image, encoded on the GPU)\n", mmproj.c_str(),
                v->n_layer, v->min_tokens, v->max_tokens);
        srv.set_qvision(std::move(v));
      } catch (const std::exception& e) {
        fprintf(stderr, "vision disabled: %s\n", e.what());
      }
    }
  }
  if (eng.hp.gemma && mmproj != "-") {
    // vision: --mmproj, or an mmproj-*.gguf next to the model (Hugging Face snapshot folder)
    if (mmproj.empty())
      for (const auto& de : std::filesystem::directory_iterator(std::filesystem::path(model).parent_path()))
        if (lower_ascii(de.path().filename().string()).find("mmproj") != std::string::npos && de.path().extension() == ".gguf") { mmproj = de.path().string(); break; }
    if (!mmproj.empty()) {
      try {
        std::unique_ptr<GemmaImageEncoder> v;
        std::string what;
        if (GgufFile(mmproj).get_str("clip.vision.projector_type", "") == "gemma4v") {
          auto g = std::make_unique<VisionGemma4V>(mmproj, eng);
          what = std::to_string(g->n_layer) + "-layer ViT, 48-pixel tokens, encoded on the GPU";
          v = std::move(g);
        } else {
          auto g = std::make_unique<VisionGemma4>(mmproj);
          what = std::to_string(g->patch) + "-pixel patches";
          v = std::move(g);
        }
        if (so.img_min > 0) v->min_tokens = so.img_min;
        if (so.img_max > 0) v->max_tokens = so.img_max;
        v->min_tokens = std::min(v->min_tokens, v->max_tokens);
        fprintf(stderr, "vision: %s (%s, %d..%d tokens per image)\n", mmproj.c_str(), what.c_str(), v->min_tokens, v->max_tokens);
        srv.set_vision(std::move(v));
      } catch (const std::exception& e) {
        fprintf(stderr, "vision disabled: %s\n", e.what());
      }
      try {
        if (GgufFile(mmproj).get_str("clip.audio.projector_type", "") == "gemma4a") {
          auto a = std::make_unique<AudioGemma4A>(mmproj, eng);
          fprintf(stderr, "audio: %d-layer conformer on the GPU (25 tokens per second of audio)\n", a->n_layer);
          srv.set_audio(std::move(a));
        } else {
          auto a = std::make_unique<AudioGemma4>(mmproj);
          fprintf(stderr, "audio: %d-sample frames at %d Hz (%.0f tokens per second of audio)\n", a->frame, a->rate, (double)a->rate / a->frame);
          srv.set_audio(std::move(a));
        }
      } catch (const std::exception& e) {
        fprintf(stderr, "audio disabled: %s\n", e.what());
      }
    }
  }
  srv.default_think_budget_ = so.think_budget;
  srv.default_preserve_ = so.preserve_thinking;
  if (so.preserve_thinking && !srv.can_preserve_)
    fprintf(stderr, "--preserve-thinking: this model's chat template has no such option (Qwen3.6 has it; Qwen3.8 always keeps the reasoning)\n");
  if (so.prefix_cache && eo.attn_int8) {  // %LOCALAPPDATA%\arcinfer\prefix\<model>-<size>
    std::error_code ec;
    const std::filesystem::path mp(model);
    const char* la = getenv("LOCALAPPDATA");
    srv.open_prefix_cache(std::string(la ? la : ".") + ARC_DATA "prefix/" + mp.stem().string() + "-" +
                          std::to_string(std::filesystem::file_size(mp, ec)), so.prefix_gb);
  }
  if (so.sess_gb > 0 && eo.attn_int8) {
    // saved conversations of this model file: %LOCALAPPDATA%\arcinfer\sessions\<model>-<size>
    std::error_code ec;
    const std::filesystem::path mp(model);
    std::string dir = so.sess_dir;
    if (dir.empty()) {
      const char* la = getenv("LOCALAPPDATA");
      dir = std::string(la ? la : ".") + ARC_DATA "sessions";
    }
    srv.open_sessions(dir + "/" + mp.stem().string() + "-" + std::to_string(std::filesystem::file_size(mp, ec)), so.sess_gb);
  }
  fprintf(stderr, "serving %s as \"%s\"\n", model.c_str(), alias.c_str());
  return L;
}

int main(int argc, char** argv) {
  full_speed_process();
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]) == "--list-models") return list_models();
  std::string model, host = "127.0.0.1", alias;
  int port = 8080;
  bool swap = true, help = false;
  ServeOpts so;
  so.eo.ctx = 40960;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
    if (a == "-m" || a == "--model") model = next();
    else if (a == "--host") host = next();
    else if (a == "--port") port = std::stoi(next());
    else if (a == "--ctx" || a == "-c") {
      const std::string v = next();
      if (v == "auto") so.ctx_auto = true;
      else so.eo.ctx = std::stoi(v);
    }
    else if (a == "--chunk") so.eo.chunk = std::stoi(next());
#ifdef ARC_ZERO
    else if (a == "--expert-cache-gb") so.eo.expert_cache_gb = std::stod(next());
    else if (a == "--expert-cache-pct") so.eo.expert_cache_pct = std::stod(next());
    else if (a == "--windows-reserve-gb") so.eo.windows_reserve_gb = std::stod(next());
    else if (a == "--no-stream") so.eo.stream_experts = false;
    else if (a == "--pack") {  // make an arcinferzero package of a MoE GGUF, then exit
      const std::string src = next();
      try {
        fprintf(stderr, "arcinferzero: making a package of %s (reads the whole model once)\n", src.c_str());
        const zpack::Result r = zpack::make_package(src);
        fprintf(stderr, "arcinferzero: package ready in %.0f s\n  %s (%.2f GB)\n  %s (%.2f GB)\n"
                        "The original GGUF is no longer needed to run this model with arcinferzero.\n",
                r.seconds, r.gguf.c_str(), r.rest_bytes / 1e9, r.experts.c_str(), r.expert_bytes / 1e9);
        return 0;
      } catch (const std::exception& e) {
        fprintf(stderr, "arcinferzero: --pack failed: %s\n", e.what());
        return 1;
      }
    }
#endif
    else if (a == "--alias") alias = next();
    else if (a == "--webui") so.webui = next();
    else if (a == "--budget") { so.eo.w8_budget_gb = std::stod(next()); so.budget_given = true; }
    else if (a == "--attn-f32") so.eo.attn_int8 = false;
    else if (a == "--mtp") so.eo.mtp = true;
    else if (a == "--mmproj") so.mmproj = next();
    else if (a == "--no-mmproj") so.mmproj = "-";
    else if (a == "--image-min-tokens") so.img_min = atoi(next().c_str());
    else if (a == "--image-max-tokens") so.img_max = atoi(next().c_str());
    else if (a == "--mtp-drafts") so.eo.mtp_drafts = std::stoi(next());
    else if (a == "--think-budget") so.think_budget = std::stoi(next());
    else if (a == "--preserve-thinking") so.preserve_thinking = true;
    else if (a == "--kernels") so.eo.kernel_dir = next();
    else if (a == "--session-cache-gb") so.sess_gb = std::stod(next());
    else if (a == "--session-dir") so.sess_dir = next();
    else if (a == "--no-swap") swap = false;
    else if (a == "--no-prefix-cache") so.prefix_cache = false;
    else if (a == "--prefix-cache-gb") so.prefix_gb = std::stod(next());
    else if (a == "--help" || a == "-h") help = true;
  }
  if (model.empty() || help) {
    fprintf(stderr,
            "usage: arcinfer-server -m model.gguf [options]   OpenAI-compatible server (API at /v1, web UI at /)\n"
            "       arcinfer-server --list-models             the models it can load, as JSON (Hugging Face cache and\n"
            "                                                 %%LOCALAPPDATA%%\\arcinfer\\models) with each one's auto context\n"
            "  -m, --model FILE      model GGUF; an mmproj-* image projector and an mtp-* drafter next to it are found\n"
            "  --host H, --port P    listen address (default 127.0.0.1:8080)\n"
            "  --alias NAME          model id in the API (default: the file name)\n"
            "  -c, --ctx N|auto      context in tokens (default 40960); auto: the largest that fits the GPU memory with\n"
            "                        ~1 GiB left for other apps, at most the trained context (remembered in\n"
            "                        %%LOCALAPPDATA%%\\arcinfer\\ctx-auto.txt)\n"
            "  --chunk N             prompt tokens per step (default 4096; MoE 2048; 512 when decoded weights stay resident)\n"
            "  --mtp                 speculative decoding with the model's MTP block or Gemma assistant drafter\n"
            "  --mtp-drafts N        drafts per step, 1..3, arcinferzero Qwen 1..7 (default 3 for Qwen 27B-class, 1 for\n"
            "                        MoE, else 2)\n"
            "  --think-budget N      default reasoning budget in tokens (requests can override it)\n"
            "  --preserve-thinking   earlier turns keep their reasoning (Qwen3.6 template option; requests can override it\n"
            "                        with chat_template_kwargs.preserve_thinking; Qwen3.8 always keeps it)\n"
            "  --mmproj FILE         image projector (default: found next to the model); --no-mmproj: none\n"
            "  --image-min-tokens N  an image becomes at least N tokens (smaller images are scaled up; Qwen: 32x32 pixels\n"
            "                        per token, default 256; Gemma 4 12B: 48x48 pixels, default 1120)\n"
            "  --image-max-tokens N  at most N tokens per image (larger images are scaled down; Qwen default 2048, Gemma\n"
            "                        4 12B 2048, above the 1120 it was trained with)\n"
            "  --budget GB           keep decoded weights resident up to GB\n"
#ifdef ARC_ZERO
            "arcinferzero (MoE models: experts read from the SSD into a GPU cache; expert file in\n"
            "%%LOCALAPPDATA%%\\arcinfer\\zero, made on the first start, safe to delete):\n"
            "  --expert-cache-gb N   expert cache size in GiB (default: automatic, see --windows-reserve-gb)\n"
            "  --expert-cache-pct P  expert cache as a percentage of all experts\n"
            "  --windows-reserve-gb N  RAM left to Windows by the automatic cache size (default 14)\n"
            "  --pack FILE.gguf      make a package of a MoE model (its first part when split): the non-expert\n"
            "                        weights + an expert file under %%LOCALAPPDATA%%\\arcinfer\\zero\\models; afterwards\n"
            "                        the original GGUF can be moved away (e.g. to a USB disk)\n"
            "  --no-stream           experts on the GPU as in arcinfer (no SSD streaming)\n"
#endif
            "  --attn-f32            f32 attention (tests)\n"
            "  --kernels DIR         OpenCL kernel folder;  --webui DIR  web UI folder\n"
            "  --session-cache-gb N  disk space for saved conversations (default 0 = off): switching between\n"
            "                        conversations loads the other one's state instead of processing its prompt again\n"
            "  --session-dir DIR     where they are kept (default %%LOCALAPPDATA%%\\arcinfer\\sessions)\n"
            "  --no-prefix-cache     do not keep system prompts on disk (default: kept in %%LOCALAPPDATA%%\\arcinfer\\prefix,\n"
            "                        per project folder for Pi (its 2 latest versions), the 8 latest for other clients;\n"
            "                        a request that starts with one loads it instead of processing it)\n"
            "  --prefix-cache-gb N   disk space for them, all models together (default 8, 0 = no limit; least recently\n"
            "                        used go first)\n"
            "  --no-swap             keep the loaded model (default: a request whose \"model\" is another model of\n"
            "                        --list-models, by id, name or file, makes the server load that one, context auto)\n"
            "  A client that closes its connection (Pi: Esc) cancels its request: prompt processing stops within a\n"
            "  layer, generation before the next token.\n");
    return help ? 0 : 1;
  }
  // locate kernels/ and webui/ next to the executable's parent (bin/..) unless given
  std::filesystem::path exe_dir = std::filesystem::path(argv[0]).parent_path();
  auto find_dir = [&](const std::string& name) {
    for (auto p : {std::filesystem::current_path() / name, exe_dir / name, exe_dir.parent_path() / name})
      if (std::filesystem::exists(p)) return p.string();
    return name;
  };
  if (so.eo.kernel_dir == "kernels" || so.eo.kernel_dir == "zero/kernels") so.eo.kernel_dir = find_dir(so.eo.kernel_dir);
  if (so.webui.empty()) so.webui = find_dir("webui");
  if (alias.empty()) alias = std::filesystem::path(model).stem().string();

  std::unique_ptr<Loaded> cur = load_model(model, alias, so);
  std::vector<ModelInfo> catalog = swap ? scan_models() : std::vector<ModelInfo>();
  std::shared_mutex swap_mu;  // requests share it; switching models takes it exclusively
  // the catalog entry a request's "model" names (alias, org/repo name or file), rescanning once for new downloads
  auto find_model = [&](const std::string& want) -> const ModelInfo* {
    for (int pass = 0; pass < 2; ++pass) {
      for (const ModelInfo& m : catalog)
        if (m.supported && (m.alias == want || m.name == want || m.file == want)) return &m;
      if (pass == 0) catalog = scan_models();
    }
    return nullptr;
  };

  HttpServer http;
  std::function<void(const HttpRequest&, HttpResponse&)> dispatch;
  http.handler = [&](const HttpRequest& rq, HttpResponse& rs) {
    // other API formats (tools/api_compat.h): translated into a chat request; the answer is translated back
    static const std::map<std::string, std::string> kinds = {
        {"/v1/completions", "completions"}, {"/completions", "completions"}, {"/api/chat", "ollama_chat"},
        {"/api/generate", "ollama_generate"}, {"/v1/messages", "anthropic"}, {"/messages", "anthropic"},
        {"/v1/responses", "responses"}, {"/responses", "responses"}};
    const auto kit = kinds.find(rq.path);
    if (rq.method == "POST" && kit != kinds.end()) {
      Json in, chat_body;
      std::string e;
      if (!Json::try_parse(rq.body, in)) return rs.send(400, "application/json", "{\"error\":{\"message\":\"invalid JSON body\"}}");
      if (!apic::to_chat(kit->second, in, chat_body, e)) {
        Json er = Json::object(), r = Json::object();
        er.set("message", e);
        er.set("type", "invalid_request_error");
        r.set("error", er);
        return rs.send(400, "application/json", r.dump());
      }
      if (chat_body["model"].str().empty()) {
        std::shared_lock<std::shared_mutex> lk(swap_mu);
        chat_body.set("model", cur->alias);
      }
      HttpRequest rq2 = rq;
      rq2.path = "/v1/chat/completions";
      rq2.body = chat_body.dump();
      apic::Adapter ad(kit->second, in, rs);
      return dispatch(rq2, rs);
    }
    // Anthropic token count
    if (rq.method == "POST" && (rq.path == "/v1/messages/count_tokens" || rq.path == "/messages/count_tokens")) {
      Json in, chat_body;
      std::string e;
      if (!Json::try_parse(rq.body, in) || !apic::to_chat("anthropic", in, chat_body, e))
        return rs.send(400, "application/json", "{\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"message\":\"bad request\"}}");
      std::shared_lock<std::shared_mutex> lk(swap_mu);
      Json r = Json::object();
      r.set("input_tokens", cur->srv->count_tokens(chat_body));
      return rs.send(200, "application/json", r.dump());
    }
    // OpenAI audio transcription / translation (multipart upload): the clip goes to a model with an audio encoder
    if (rq.method == "POST" && (rq.path == "/v1/audio/transcriptions" || rq.path == "/v1/audio/translations")) {
      const auto form = apic::parse_multipart(rq.header("content-type"), rq.body);
      const auto fit = form.find("file");
      if (fit == form.end() || fit->second.data.empty())
        return rs.send(400, "application/json", "{\"error\":{\"message\":\"file required (multipart/form-data)\"}}");
      Json req = Json::object();
      auto field = [&](const char* k) { auto it = form.find(k); return it == form.end() ? std::string() : it->second.data; };
      const bool translate = rq.path == "/v1/audio/translations";
      req.set("model", apic::strip_tag(field("model")));
      req.set("response_format", field("response_format").empty() ? "json" : field("response_format"));
      req.set("language", field("language"));
      req.set("task", translate ? "translate" : "transcribe");
      std::string ext = fit->second.filename.substr(fit->second.filename.rfind('.') + 1);
      std::string instr = translate ? "Translate the speech in this audio into English. Reply with the translation only."
                                    : "Transcribe the speech in this audio exactly as spoken" +
                                          (field("language").empty() ? std::string() : " (language: " + field("language") + ")") +
                                          ". Reply with the transcription only.";
      if (!field("prompt").empty()) instr += " Context: " + field("prompt");
      Json audio = Json::object();
      audio.set("data", apic::base64_encode(fit->second.data));
      audio.set("format", ext);
      Json pa = Json::object();
      pa.set("type", "input_audio");
      pa.set("input_audio", audio);
      Json parts = Json::array();
      parts.push(pa);
      parts.push(apic::text_part(instr));
      Json msgs = Json::array();
      msgs.push(apic::msg("user", parts));
      Json chat_body = Json::object();
      chat_body.set("model", req["model"].str());
      if (chat_body["model"].str().empty()) {
        std::shared_lock<std::shared_mutex> lk(swap_mu);
        chat_body.set("model", cur->alias);
      }
      chat_body.set("messages", msgs);
      chat_body.set("enable_thinking", false);
      chat_body.set("temperature", field("temperature").empty() ? 0.0 : atof(field("temperature").c_str()));
      HttpRequest rq2 = rq;
      rq2.path = "/v1/chat/completions";
      rq2.body = chat_body.dump();
      apic::Adapter ad("transcription", req, rs);
      return dispatch(rq2, rs);
    }
    // n > 1 (several answers to one chat request): the request runs n times (the prompt is cached after the first;
    // a given seed changes per answer); not with streaming
    if (rq.method == "POST" && rq.path == "/v1/chat/completions") {
      Json in;
      if (Json::try_parse(rq.body, in) && in["n"].as_int(1) > 1) {
        if (in["stream"].as_bool(false))
          return rs.send(400, "application/json", "{\"error\":{\"message\":\"n > 1 is supported without streaming\",\"type\":\"invalid_request_error\"}}");
        const int n = (int)std::min<int64_t>(in["n"].as_int(1), 16);
        Json merged, choices = Json::array();
        int64_t prompt_tok = 0, compl_tok = 0;
        for (int k = 0; k < n; ++k) {
          Json one = in;
          one.set("n", 1);
          if (in["seed"].is_number()) one.set("seed", in["seed"].as_int() + k);
          HttpRequest rq2 = rq;
          rq2.body = one.dump();
          int status = 0;
          std::string body;
          rs.on_send = [&](int st, const std::string& b) { status = st; body = b; };
          dispatch(rq2, rs);
          rs.on_send = nullptr;
          if (status != 200) return rs.send(status ? status : 500, "application/json", body);
          Json r;
          Json::try_parse(body, r);
          if (merged.is_null()) merged = r;
          Json c = r["choices"][0];
          c.set("index", k);
          choices.push(c);
          prompt_tok = r["usage"]["prompt_tokens"].as_int();
          compl_tok += r["usage"]["completion_tokens"].as_int();
        }
        merged.set("choices", choices);
        Json u = Json::object();
        u.set("prompt_tokens", prompt_tok);
        u.set("completion_tokens", compl_tok);
        u.set("total_tokens", prompt_tok + compl_tok);
        merged.set("usage", u);
        return rs.send(200, "application/json", merged.dump());
      }
    }
    // Ollama: the model list, one model's details, the version, the loaded model
    if (rq.path == "/api/version") return rs.send(200, "application/json", "{\"version\":\"0.9.0\"}");
    if (rq.path == "/api/tags" || rq.path == "/api/ps" || (rq.path == "/api/show" && rq.method == "POST")) {
      std::shared_lock<std::shared_mutex> lk(swap_mu);
      auto entry = [&](const ModelInfo* m, const std::string& alias, bool vision, bool mtp) {
        Json d = Json::object();
        d.set("format", "gguf");
        d.set("family", m ? m->arch : std::string());
        Json fam = Json::array();
        fam.push(m ? m->arch : std::string());
        d.set("families", fam);
        d.set("parameter_size", "");
        d.set("quantization_level", "");
        Json j = Json::object();
        j.set("name", alias + ":latest");
        j.set("model", alias + ":latest");
        j.set("modified_at", apic::iso_now());
        j.set("size", (double)(m ? m->bytes : 0));
        j.set("digest", "");
        j.set("details", d);
        (void)vision;
        (void)mtp;
        return j;
      };
      if (rq.path == "/api/show") {
        Json in;
        Json::try_parse(rq.body, in);
        const std::string want = apic::strip_tag(in["model"].str(in["name"].str()));
        const ModelInfo* m = nullptr;
        for (const ModelInfo& c : catalog)
          if (c.supported && (c.alias == want || c.name == want)) m = &c;
        if (!m && want != cur->alias) return rs.send(404, "application/json", "{\"error\":\"model not found\"}");
        const bool loaded = want == cur->alias;
        Json r = Json::object();
        r.set("modelfile", "");
        r.set("parameters", "");
        r.set("template", "");
        Json e = entry(m, want, m ? m->vision : cur->srv->has_vision(), m ? m->mtp : false);
        r.set("details", e["details"]);
        Json mi = Json::object();
        const std::string arch = m ? m->arch : std::string("unknown");
        mi.set("general.architecture", arch);
        mi.set(arch + ".context_length", loaded ? (double)cur->eng->ctx : (double)(m ? m->ctx_train : 0));
        r.set("model_info", mi);
        Json caps = Json::array();
        caps.push("completion");
        caps.push("tools");
        caps.push("thinking");
        if (m ? m->vision : cur->srv->has_vision()) caps.push("vision");
        r.set("capabilities", caps);
        r.set("modified_at", apic::iso_now());
        return rs.send(200, "application/json", r.dump());
      }
      Json ms = Json::array();
      for (const ModelInfo& m : catalog)
        if (m.supported && (rq.path == "/api/tags" || m.alias == cur->alias)) ms.push(entry(&m, m.alias, m.vision, m.mtp));
      if ((rq.path == "/api/ps" || rq.path == "/api/tags") && !ms.size())  // (--no-swap: no catalog, the loaded model)
        ms.push(entry(nullptr, cur->alias, false, false));
      Json r = Json::object();
      r.set("models", ms);
      return rs.send(200, "application/json", r.dump());
    }
    dispatch(rq, rs);
  };
  dispatch = [&](const HttpRequest& rq, HttpResponse& rs) {
    if (swap && (rq.path == "/v1/models" || rq.path == "/models")) {  // every model a request can switch to
      std::shared_lock<std::shared_mutex> lk(swap_mu);
      Json d = Json::array();
      bool listed = false;
      for (const ModelInfo& m : catalog) {
        if (!m.supported) continue;
        const bool loaded = !listed && m.alias == cur->alias;
        listed = listed || loaded;
        Json j = Json::object();
        j.set("id", m.alias);
        j.set("object", "model");
        j.set("owned_by", "arcinfer");
        j.set("name", m.name);
        j.set("loaded", loaded);
        if (loaded) j.set("context_length", cur->eng->ctx);
        set_modalities(j, m.vision);
        // the loaded model first (clients that take the first entry get the running one)
        if (loaded) {
          Json d2 = Json::array();
          d2.push(j);
          for (const Json& e : d.items()) d2.push(e);
          d = d2;
        } else {
          d.push(j);
        }
      }
      if (!listed) {  // a model file outside the scanned folders: first, under the alias it was started with
        Json j = Json::object(), d2 = Json::array();
        j.set("id", cur->alias);
        j.set("object", "model");
        j.set("owned_by", "arcinfer");
        j.set("name", cur->alias);
        j.set("loaded", true);
        j.set("context_length", cur->eng->ctx);
        set_modalities(j, cur->srv->has_vision());
        d2.push(j);
        for (const Json& e : d.items()) d2.push(e);
        d = d2;
      }
      Json r = Json::object();
      r.set("object", "list");
      r.set("models", llama_models(d));
      r.set("data", d);
      return rs.send(200, "application/json", r.dump());
    }
    if (swap && rq.method == "POST" && rq.path == "/v1/chat/completions") {
      Json body;
      const std::string want = Json::try_parse(rq.body, body) ? body["model"].str() : "";
      bool other = false;
      {
        std::shared_lock<std::shared_mutex> lk(swap_mu);
        other = !want.empty() && want != cur->alias;
      }
      if (other) {
        std::unique_lock<std::shared_mutex> lk(swap_mu);
        const ModelInfo* m = want != cur->alias ? find_model(want) : nullptr;
        if (m && m->alias != cur->alias) {
          const std::string prev_path = cur->path, prev_alias = cur->alias;
          const auto t0 = clk::now();
          fprintf(stderr, "switching model: %s -> %s\n", prev_alias.c_str(), m->alias.c_str());
          if (cur->srv) cur->srv->save_held();  // (switching back later loads the conversation instead of redoing it)
          cur.reset();  // frees the GPU memory first
          ServeOpts s2 = so;
          s2.ctx_auto = true;  // the --ctx given at start was for the first model
          s2.eo.mtp = m->mtp;  // the variant's settings: MTP block / drafter, image projector
          s2.mmproj = m->vision ? "" : "-";
          try {
            cur = load_model(m->path, m->alias, s2);
            fprintf(stderr, "switched to %s in %.1f s\n", m->alias.c_str(), ms_since(t0) / 1000);
            if (const char* la = getenv("LOCALAPPDATA"))  // pi-arc starts with this model next time
              std::ofstream(std::string(la) + "/arcinfer/last-model.txt") << m->path << "\n";
          } catch (const std::exception& e) {
            fprintf(stderr, "could not load %s: %s; loading %s again\n", m->path.c_str(), e.what(), prev_alias.c_str());
            ServeOpts s3 = so;
            s3.ctx_auto = true;
            for (const ModelInfo& pm : catalog)  // (the previous variant's settings)
              if (pm.alias == prev_alias) {
                s3.eo.mtp = pm.mtp;
                s3.mmproj = pm.vision ? "" : "-";
              }
            cur = load_model(prev_path, prev_alias, s3);
            Json err = Json::object(), r = Json::object();
            err.set("message", std::string("could not load model ") + want + ": " + e.what());
            err.set("type", "server_error");
            r.set("error", err);
            return rs.send(500, "application/json", r.dump());
          }
        }
      }
    }
    std::shared_lock<std::shared_mutex> lk(swap_mu);
    cur->srv->handle(rq, rs);
  };
  if (!http.listen(host, port)) {
    fprintf(stderr, "cannot listen on %s:%d\n", host.c_str(), port);
    return 1;
  }
  fprintf(stderr, "listening on http://%s:%d  (OpenAI API at /v1, web UI at /)\n", host.c_str(), port);
  http.run();
  return 0;
}
