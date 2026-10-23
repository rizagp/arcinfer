// arcinfer-server: OpenAI-compatible HTTP server (chat completions with streaming, reasoning and tool calls),
// prefix reuse across requests, and a built-in web UI.
// usage: arcinfer-server -m model.gguf [--host 127.0.0.1] [--port 8080] [--ctx 40960] [--alias name] [--webui dir]
//                       [--chunk N] [--budget GB] [--mtp] [--think-budget N] [--attn-f32] [--kernels dir]
//                       [--session-cache-gb N] [--session-dir dir]   (saved conversations; default 0 = off)
// Built as bin\serverz.exe (zero/serverz.cpp includes this file): the arcinferzero engine, zero/engine.h (Level Zero,
// MoE experts streamed from the SSD), with its data folder %LOCALAPPDATA%\arcinfer\zero (context cache, saved
// prompts and conversations).
#include "../zero/engine.h"
#include "../zero/gguf_pack.h"
#define ARC_DATA "/arcinfer/zero/"
#define ARC_DATA_HELP "%%LOCALAPPDATA%%\\arcinfer\\zero\\"  // (the same folder, as --help prints it)
#include "../src/tokenizer.h"
#include "../src/chat.h"
#include "../src/chat_gemma4.h"
#include "../src/chat_jinja.h"
#include "../src/image.h"
#include "../src/audio.h"
#include "../src/audio_gemma4.h"
#include "../src/vision_gemma4.h"
#include "../zero/vision_gemma4v.h"
#include "../zero/audio_gemma4a.h"
#include "../zero/vision_qwen3vl.h"
#include "../src/http.h"
#include "../src/full_speed.h"
#include "api_compat.h"
#include "json_constraint.h"
#include "tool_constraint.h"
#include "sampler.h"
#include "web_tools.h"
#include <atomic>
#include <condition_variable>
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
// bytes: model id -> its file size ("size"; 0 when not known).
// The served context window in the fields clients read: llama-server's "meta" (n_ctx: the window served, n_ctx_train:
// the trained one; Unsloth Studio takes meta.n_ctx) and vLLM's "max_model_len". served 0: not loaded, not known yet.
static void set_window(Json& m, int64_t served, int64_t trained, int64_t n_vocab = 0, int64_t n_embd = 0) {
  Json meta = Json::object();
  if (served > 0) meta.set("n_ctx", (double)served);
  if (trained > 0) meta.set("n_ctx_train", (double)trained);
  if (n_vocab > 0) meta.set("n_vocab", (double)n_vocab);
  if (n_embd > 0) meta.set("n_embd", (double)n_embd);
  m.set("meta", meta);
  if (served > 0) m.set("max_model_len", (double)served);
}

static Json llama_models(const Json& data, const std::map<std::string, uint64_t>& bytes) {
  Json out = Json::array();
  for (const Json& m : data.items()) {
    Json e = Json::object();
    const std::string id = m["id"].str();
    e.set("name", id);
    e.set("model", id);
    e.set("modified_at", "");
    const auto b = bytes.find(id);
    e.set("size", (double)(b != bytes.end() ? b->second : 0));
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

// The bytes of a model file; a split GGUF (name-00001-of-0000N.gguf) counts every part that is present.
static uint64_t gguf_bytes(const std::string& path) {
  namespace fs = std::filesystem;
  std::error_code ec;
  const fs::path p(path);
  const std::string f = p.filename().string();
  const size_t k = f.find("-00001-of-");
  uint64_t n = fs::file_size(p, ec);
  if (ec) return 0;
  if (k == std::string::npos) return n;
  const int parts = atoi(f.c_str() + k + 10);
  for (int i = 2; i <= parts; ++i) {
    char num[16];
    snprintf(num, sizeof num, "%05d", i);
    const uint64_t b = fs::file_size(p.parent_path() / (f.substr(0, k + 1) + num + f.substr(k + 6)), ec);
    if (!ec) n += b;
  }
  return n;
}

// audio: a Gemma 4 audio encoder (mmproj audio projector gemma4ua or gemma4a)
static void set_modalities(Json& j, bool vision, bool audio) {
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
  if (audio) in.push("audio");
  Json out = Json::array();
  out.push("text");
  Json arch = Json::object();
  arch.set("input_modalities", in);
  arch.set("output_modalities", out);
  j.set("architecture", arch);
  Json mod = Json::object();
  mod.set("vision", vision);
  mod.set("audio", audio);
  j.set("modalities", mod);
  j.set("type", vision ? "vlm" : "llm");
}

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
    slots_.resize(e.slot_count());
  }

  bool has_vision() const { return vision_ != nullptr || qvision_ != nullptr; }
  bool has_audio() const { return audio_ != nullptr && aud_tok_ >= 0; }  // (what a chat request with audio needs)
  uint64_t model_bytes_ = 0;  // the model file's size (/v1/models "models": "size"), set by load_model
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
      set_window(m, eng_.ctx, eng_.gguf().get_int(eng_.gguf().get_str("general.architecture") + ".context_length", 0),
                 eng_.hp.n_vocab, eng_.hp.n_embd);
      set_modalities(m, has_vision(), has_audio());
      Json d = Json::array();
      d.push(m);
      Json r = Json::object();
      r.set("object", "list");
      r.set("models", llama_models(d, {{model_id_, model_bytes_}}));
      r.set("data", d);
      return rs.send(200, "application/json", r.dump());
    }
    if (rq.path == "/sessions" || (rq.path == "/sessions/save" && rq.method == "POST"))
      return sessions(rq, rs);  // GET: size of the session cache; DELETE: clear it; POST /sessions/save: keep the held one
    if (rq.path == "/props") {
      Json r = Json::object();
      r.set("model", model_id_);
      r.set("n_ctx", eng_.ctx);
      r.set("engine", "arcinferzero");
      r.set("chunk", eng_.max_chunk);
      r.set("weights", eng_.w8_resident ? "w8-resident" : "w8-just-in-time");
      r.set("vision", has_vision());
      Json mod = Json::object();  // (llama-server's form)
      mod.set("vision", has_vision());
      mod.set("audio", has_audio());
      r.set("modalities", mod);
      r.set("audio", has_audio());
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
        st.set("busy", busy_.load() > 0);
        st.set("position", eng_.pos);
        st.set("slots", (int)slots_.size());
        if (!last_timings_.is_null()) st.set("last", last_timings_);
        r.set("stats", st);
      }
      return rs.send(200, "application/json", r.dump());
    }
    if (rq.path == "/tokenize" && rq.method == "POST") {
      Json b;
      if (!Json::try_parse(rq.body, b)) return rs.send(400, "application/json", "{\"error\":{\"message\":\"invalid JSON body\"}}");
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
  std::atomic<int> busy_{0};  // requests running
  int n_req_ = 0;
  Json last_timings_;
  struct MediaEmb {
    std::shared_ptr<std::vector<float>> e;
    int gw = 0, gh = 0;  // merged-token grid (Qwen M-RoPE); 0 for Gemma and audio
  };
  std::map<uint64_t, MediaEmb> img_cache_;  // media embeddings keyed by the file bytes
  size_t img_cache_bytes_ = 0;               // (their floats: the cache is emptied past 512 MB or 64 entries)
  std::string think_close_ = "</think>";
  SamplingParams defaults_;
  int im_start_ = -1;
  std::vector<int> stop_ids_;
  std::mutex mu_;
  std::vector<int> cached_;  // tokens whose state the engine currently holds (positions 0..pos)
  // ---- conversation slots (--slots N, arcinferzero): the engine holds N conversations (Engine::bind). A request
  // takes a slot for its whole run and the engine lock for each step (a piece of its prompt, a generated token or
  // verification pass). Between two steps it hands the engine to a waiting request whose engine time is behind its
  // own (take_turns), so a request's tokens do not wait behind each 8K-token piece of another's long prompt: the
  // short steps run until they have used as much time. cached_ / cur_saved_ / last_gen_cut_ are the bound slot's.
  struct SlotSrv {
    std::vector<int> cached;
    bool cur_saved = false, busy = false;
    bool stale = false;  // a step of its conversation threw: its engine state is reset when it is next bound
    int last_gen_cut = 0;
    uint64_t used = 0;
    double vtime = 0;  // engine time of its request (ms; a new request starts kNewCredit behind the others)
    clk::time_point since;
  };
  static constexpr double kNewCredit = 10000;  // ms
  std::vector<SlotSrv> slots_;
  int cur_slot_ = 0;
  uint64_t slot_clock_ = 0, turns_ = 0;
  std::condition_variable slot_cv_, turn_cv_;
  std::atomic<int> waiting_{0};  // requests waiting for the engine lock (to start, or for their next step)
  void use_slot(int s) {
    if (s != cur_slot_ || eng_.bound_slot() != s) {  // (the engine can be on another slot after a failed step)
      eng_.bind(s);  // (first: if it throws, the server's fields still match the slot the engine holds)
      if (s != cur_slot_) {
        SlotSrv& o = slots_[cur_slot_];
        o.cached = std::move(cached_);
        o.cur_saved = cur_saved_;
        o.last_gen_cut = last_gen_cut_;
        SlotSrv& n = slots_[s];
        cached_ = std::move(n.cached);
        cur_saved_ = n.cur_saved;
        last_gen_cut_ = n.last_gen_cut;
        cur_slot_ = s;
      }
    }
    if (slots_[s].stale) {
      slots_[s].stale = false;
      wipe_bound();
    }
  }
  // the bound slot's engine state from scratch: no position, no recurrent-state snapshots
  void wipe_bound() {
    try {
      eng_.reset();
      for (int k = 0; k < kSnaps; ++k) eng_.drop_snapshot(k);
    } catch (...) {
    }
  }
  // the engine lock for a request (counted: the request holding it lets this one start)
  std::atomic<int> starting_{0};
  void acquire(std::unique_lock<std::mutex>& lk) {
    ++waiting_;
    ++starting_;
    lk.lock();
    --starting_;
    --waiting_;
    ++turns_;
    turn_cv_.notify_all();
  }
  // between two steps of a request in `slot`: a request starting, or one whose engine time is behind, goes first
  void take_turns(std::unique_lock<std::mutex>& lk, int slot) {
    if (slots_.size() < 2 || waiting_ == 0) return;
    SlotSrv& me = slots_[slot];
    const clk::time_point now = clk::now();
    me.vtime += std::chrono::duration<double, std::milli>(now - me.since).count();
    me.since = now;
    bool behind = starting_ > 0;
    for (int k = 0; k < (int)slots_.size(); ++k)
      behind = behind || (k != slot && slots_[k].busy && slots_[k].vtime < me.vtime);
    if (!behind) return;
    const uint64_t t = turns_;
    ++waiting_;
    turn_cv_.wait(lk, [&] { return turns_ != t; });
    --waiting_;
    ++turns_;
    turn_cv_.notify_all();
    me.since = clk::now();
    use_slot(slot);
  }
  // end of the system prompt (the start of the second turn; Gemma's first turn follows <bos>): every conversation of
  // a client (Pi: its system prompt and tool list, ~4K tokens) shares it
  size_t system_end(const std::vector<int>& P) const {
    size_t i0 = 0;
    while (i0 < P.size() && i0 < 2 && P[i0] != im_start_) ++i0;
    if (i0 + 1 < P.size() && P[i0] == im_start_ && tok_.piece(P[i0 + 1]) == "system")
      for (size_t i = i0 + 1; i < P.size(); ++i)
        if (P[i] == im_start_) return i;
    return 0;
  }
  // A free slot for the request (waits while every slot is in use): the one whose conversation it continues (shares
  // more than the system prompt), else an empty one, else the one used longest ago (its conversation goes to the
  // session cache when the request does not continue it).
  int claim_slot(std::unique_lock<std::mutex>& lk, const std::vector<int>& P) {
    if (slots_.size() < 2) return 0;
    slot_cv_.wait(lk, [&] {
      for (const auto& x : slots_)
        if (!x.busy) return true;
      return false;
    });
    const size_t sys = system_end(P);
    int cont = -1, empty = -1, lru = -1;
    size_t cl = 0;
    for (int k = 0; k < (int)slots_.size(); ++k) {
      if (slots_[k].busy) continue;
      const std::vector<int>& c = k == cur_slot_ ? cached_ : slots_[k].cached;
      const size_t l = common_prefix(P, c);
      if (l > sys + 8 && l > cl) { cont = k; cl = l; }
      if (c.empty() && empty < 0) empty = k;
      if (lru < 0 || slots_[k].used < slots_[lru].used) lru = k;
    }
    const int k = cont >= 0 ? cont : empty >= 0 ? empty : lru;  // (see below)
    double vmin = -1;
    for (const auto& x : slots_)
      if (x.busy && (vmin < 0 || x.vtime < vmin)) vmin = x.vtime;
    use_slot(k);  // (before busy: a slot left busy by a bind that threw would never be free again)
    slots_[k].vtime = std::max(0.0, vmin - kNewCredit);
    slots_[k].since = clk::now();
    slots_[k].busy = true;
    slots_[k].used = ++slot_clock_;
    if (cont < 0) eng_.reset_drafter_choice();  // (another conversation: its own DFlash2 / MTP choice)
    return k;
  }
  // ---- batched generation steps (--slots, Qwen models on arcinferzero: QUEUE 20.4). A request at a generation step
  // (its next token, with drafts or without) while other requests are generating waits for one of them to reach its
  // own step: that one runs both in one engine pass (Engine::forward_batch / verify_batch: every slot drafts in its
  // own state, one pass verifies all rows) and hands back the rows; each request then samples and settles its drafts
  // as before. The weights are read once for both: 27B two slots 1.45x (drafts) / 1.8x (none) the tokens per second.
  struct GenStep {
    int slot = 0, tok = 0, max_drafts = 0;  // max_drafts 0: one token without drafts
    bool done = false, alone = false;       // done: run by another request; alone: run it yourself
    bool dflash = false, batched = false;
    std::vector<int> drafts;
    std::vector<float> rows;  // logits of the token and of each draft
    std::string failed;       // the pass that ran it threw (its request fails too)
  };
  std::vector<GenStep*> gen_wait_;  // steps waiting for another request to run them
  int gen_active_ = 0;              // requests in their generation loop
  std::condition_variable gen_cv_;
  // one step of the bound slot's request: alone, or with waiting steps of other slots
  void run_steps(std::vector<GenStep*> st) {
    try {
      run_batch(st);
    } catch (const std::exception& e) {
      // every request of the step fails (the waiting ones would otherwise wait for good), and their slots start
      // over: the engine left their positions as they were, but part of their device state may be written
      try { eng_.bind(cur_slot_); } catch (...) {}
      for (GenStep* g : st) {
        g->failed = e.what();
        g->done = true;
        drop_conversation(g->slot);
      }
      gen_cv_.notify_all();
      throw;
    }
    for (GenStep* g : st) {
      g->done = true;
      g->batched = st.size() > 1;
    }
  }
  // the conversation of slot k is gone (a step that threw): it starts over. Its engine state is wiped now when the
  // engine holds k, else when k is next bound (use_slot)
  void drop_conversation(int k) {
    if (k == cur_slot_) {
      cached_.clear();
      cur_saved_ = false;
      last_gen_cut_ = 0;
    } else {
      slots_[k].cached.clear();
      slots_[k].cur_saved = false;
      slots_[k].last_gen_cut = 0;
    }
    if (eng_.bound_slot() == k) wipe_bound();
    else slots_[k].stale = true;
  }
  void run_batch(const std::vector<GenStep*>& st) {
    const int V = eng_.hp.n_vocab;
    if (st.size() == 1) {
      GenStep& g = *st[0];
      if (g.max_drafts > 0) {
        g.drafts = eng_.mtp_draft(g.tok, g.max_drafts);
        g.dflash = eng_.last_draft_dflash();
        const auto& lv = eng_.verify(g.tok, g.drafts);
        g.rows.assign(lv.begin(), lv.begin() + (1 + g.drafts.size()) * V);
      } else {
        g.rows = eng_.forward(&g.tok, 1);
      }
      return;
    }
    std::vector<int> sl;
    for (GenStep* g : st) sl.push_back(g->slot);
    if (st[0]->max_drafts > 0) {
      const int nd = eng_.batch_rows() / (int)st.size() - 1;  // (the rows of every slot within one pass)
      std::vector<std::vector<int>> rows;
      for (GenStep* g : st) {
        eng_.bind(g->slot);
        g->drafts = eng_.mtp_draft(g->tok, std::min(g->max_drafts, nd));
        g->dflash = eng_.last_draft_dflash();
        rows.push_back({g->tok});
        rows.back().insert(rows.back().end(), g->drafts.begin(), g->drafts.end());
      }
      eng_.bind(cur_slot_);
      const std::vector<float>& lv = eng_.verify_batch(sl, rows);
      size_t r0 = 0;
      for (size_t b = 0; b < st.size(); ++b) {
        st[b]->rows.assign(lv.begin() + r0 * V, lv.begin() + (r0 + rows[b].size()) * V);
        r0 += rows[b].size();
      }
    } else {
      std::vector<int> tk;
      for (GenStep* g : st) tk.push_back(g->tok);
      const std::vector<float>& lv = eng_.forward_batch(sl, tk);
      for (size_t b = 0; b < st.size(); ++b) st[b]->rows.assign(lv.begin() + b * V, lv.begin() + (b + 1) * V);
    }
  }
  // the step `me` of the request in slot `slot` (the engine lock held, the slot bound; bound again on return)
  void gen_step(std::unique_lock<std::mutex>& lk, GenStep& me, int slot) {
    bool can = slots_.size() > 1;
    static const bool off = getenv("ARC_NO_BATCH") && atoi(getenv("ARC_NO_BATCH")) != 0;  // (A/B: turns only)
    can = can && !off && (me.max_drafts == 0 ? eng_.batch_ok() : eng_.batch_verify_ok() && eng_.batch_rows() >= 4);
    for (;;) {
      if (!can) break;
      if (!gen_wait_.empty()) {  // lead: this step and the waiting ones of the same kind that fit one pass
        std::vector<GenStep*> st{&me};
        const size_t most = me.max_drafts > 0 ? (size_t)eng_.batch_rows() / 2 : (size_t)eng_.batch_rows();
        for (GenStep* g : gen_wait_) {
          if ((g->max_drafts > 0) == (me.max_drafts > 0) && st.size() < most) st.push_back(g);
          else g->alone = true;
        }
        gen_wait_.clear();
        run_steps(st);
        gen_cv_.notify_all();
        return;
      }
      if (gen_active_ - (int)gen_wait_.size() <= 1) break;  // nobody else will reach a step: alone
      gen_wait_.push_back(&me);
      gen_cv_.wait(lk, [&] { return me.done || me.alone || gen_active_ - (int)gen_wait_.size() <= 0; });
      gen_wait_.erase(std::remove(gen_wait_.begin(), gen_wait_.end(), &me), gen_wait_.end());
      use_slot(slot);
      eng_.cancel = nullptr;  // (a prompt step of another request may have set its own)
      if (!me.failed.empty()) {  // (this slot starts over, as the leader did with its own)
        drop_conversation(slot);
        throw std::runtime_error("batched step: " + me.failed);
      }
      if (me.done) return;
      if (me.alone) break;
      // (every generating request waits: the next one leads)
    }
    run_steps({&me});
  }
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
    if (P.empty()) return 0;  // (chat() refuses empty prompts; P.size() - 1 below would wrap)
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
#include "server_cache.h"  // (saved state: prefix cache, sessions)

#include "server_chat.h"  // (the chat endpoint)
};

#include "server_catalog.h"  // (model catalog, --ctx auto)

#include "server_load.h"  // (loading a model)

int main(int argc, char** argv) {
  full_speed_process();
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]) == "--list-models") return list_models();
  std::string model, host = "127.0.0.1", alias;
  int port = 8080;
  bool swap = true, help = false, web_tools = true;
  ServeOpts so;
  so.eo.ctx = 40960;
  // A bad option value is a message, not an uncaught std::stoi exception (which aborted the process).
  auto to_int = [](const std::string& opt, const std::string& s) {
    try {
      size_t used = 0;
      const int v = std::stoi(s, &used);
      if (used == s.size()) return v;
    } catch (...) {}
    throw std::runtime_error(opt + ": expected an integer (got \"" + s + "\")");
  };
  auto to_double = [](const std::string& opt, const std::string& s) {
    try {
      size_t used = 0;
      const double v = std::stod(s, &used);
      if (used == s.size() && std::isfinite(v)) return v;  // (nan / inf refused: they become sizes and limits)
    } catch (...) {}
    throw std::runtime_error(opt + ": expected a number (got \"" + s + "\")");
  };
  try {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
    if (a == "-m" || a == "--model") model = next();
    else if (a == "--host") host = next();
    else if (a == "--port") port = to_int(a, next());
    else if (a == "--ctx" || a == "-c") {
      const std::string v = next();
      if (v == "auto") so.ctx_auto = true;
      else so.eo.ctx = std::clamp(to_int(a, v), 0, 1 << 22);  // (4M tokens: past it the sizes overflow int)
    }
    else if (a == "--chunk") so.eo.chunk = std::clamp(to_int(a, next()), 0, 65536);
    else if (a == "--expert-cache-gb") so.eo.expert_cache_gb = to_double(a, next());
    else if (a == "--expert-cache-pct") so.eo.expert_cache_pct = to_double(a, next());
    else if (a == "--no-stream") so.eo.stream_experts = false;
    else if (a == "--pack") {  // make an arcinferzero package of a MoE GGUF, then exit
      const std::string src = next();
      try {
        fprintf(stderr, "arcinferzero: making a package of %s (reads the whole model once)\n", src.c_str());
        const zpack::Result r = zpack::make_package(src);
        fprintf(stderr, "arcinferzero: package ready in %.0f s\n  %s (%.2f GB)\n  %s (%.2f GB)\n"
                        "The original GGUF is no longer needed to run this model with arcinferzero: the model list now\n"
                        "picks the package. The engine's own expert file for the GGUF moved into the package, so loading\n"
                        "the GGUF itself (by its path) would write that file again; keep the GGUF only as a backup.\n",
                r.seconds, r.gguf.c_str(), r.rest_bytes / 1e9, r.experts.c_str(), r.expert_bytes / 1e9);
        return 0;
      } catch (const std::exception& e) {
        fprintf(stderr, "arcinferzero: --pack failed: %s\n", e.what());
        return 1;
      }
    }
    else if (a == "--alias") alias = next();
    else if (a == "--webui") so.webui = next();
    else if (a == "--budget") { so.eo.w8_budget_gb = to_double(a, next()); so.budget_given = true; }
    else if (a == "--attn-f32") so.eo.attn_int8 = false;
    else if (a == "--mtp") so.eo.mtp = true;
    else if (a == "--dflash") { so.eo.dflash = next(); so.eo.mtp = true; }
    else if (a == "--no-dflash") { so.eo.dflash.clear(); so.dflash_auto = false; }
    else if (a == "--draft") {  // (the drafter of the -nodf / -df variants)
      const std::string d = next();
      if (d == "mtp") so.eo.dflash.clear(), so.dflash_auto = false, so.eo.mtp_block = true;
      else if (d == "dflash") so.dflash_auto = true, so.eo.mtp_block = false;
      else if (d == "auto") so.dflash_auto = true, so.eo.mtp_block = true;
      else throw std::runtime_error("--draft: auto, mtp or dflash");
    }
    else if (a == "--mmproj") so.mmproj = next();
    else if (a == "--no-mmproj") so.mmproj = "-";
    else if (a == "--image-min-tokens") so.img_min = atoi(next().c_str());
    else if (a == "--image-max-tokens") so.img_max = atoi(next().c_str());
    else if (a == "--mtp-drafts") so.eo.mtp_drafts = to_int(a, next());
    else if (a == "--think-budget") so.think_budget = to_int(a, next());
    else if (a == "--preserve-thinking") so.preserve_thinking = true;
    else if (a == "--kernels") so.eo.kernel_dir = next();
    else if (a == "--session-cache-gb") so.sess_gb = std::clamp(to_double(a, next()), 0.0, 4096.0);
    else if (a == "--session-dir") so.sess_dir = next();
    else if (a == "--slots") so.eo.slots = std::clamp(to_int(a, next()), 1, 8);
    else if (a == "--no-swap") swap = false;
    else if (a == "--no-web-tools") web_tools = false;
    else if (a == "--no-prefix-cache") so.prefix_cache = false;
    else if (a == "--prefix-cache-gb") so.prefix_gb = std::clamp(to_double(a, next()), 0.0, 4096.0);
    else if (a == "--help" || a == "-h") help = true;
    else throw std::runtime_error("unknown option " + a + " (--help lists them; --q2 and --v4 were removed 2026-10-22)");
  }
  if (!help) {
    if (port < 1 || port > 65535) throw std::runtime_error("--port: expected 1..65535 (got " + std::to_string(port) + ")");
    if (host.empty()) throw std::runtime_error("--host: an address or name is required");
    if (!so.ctx_auto && so.eo.ctx < 1) throw std::runtime_error("--ctx: expected 1 or more tokens (or auto)");
    if (so.eo.chunk < 0) throw std::runtime_error("--chunk: expected 0 (auto) or more tokens");
    if (so.eo.mtp_drafts < 0) throw std::runtime_error("--mtp-drafts: expected 0 or more");
  }
  } catch (const std::exception& e) {
    fprintf(stderr, "arcinfer-server: %s\n", e.what());
    return 2;
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
            "                        1.2 GiB left for the driver, at most the trained context (remembered in\n"
            "                        " ARC_DATA_HELP "ctx-auto.txt)\n"
            "  --chunk N             prompt tokens per step (default 4096; MoE with experts on the SSD 8192, Flash\n"
            "                        Next 2048; MoE with --no-stream 2048; 512 when decoded weights stay resident;\n"
            "                        Gemma with images at least 2048; Gemma MoE 2048)\n"
            "  --mtp                 speculative decoding with the model's MTP block or Gemma assistant drafter\n"
            "  --mtp-drafts N        drafts per step: Qwen 1..7, Gemma 1..3 (default: 7 for Qwen 27B-class and with\n"
            "                        DFlash2, 5 for MoE, 3 for 4B and Gemma 4 26B, 2 otherwise)\n"
            "  --slots N             hold N conversations at once (default 1), each with its own KV cache and states\n"
            "                        of --ctx tokens (auto: the context that fits N of them; MoE with experts on the\n"
            "                        SSD: 128K in all, at least 32K each); requests in different slots take turns step\n"
            "                        by step (a prompt piece, a generated token), and requests generating at the same\n"
            "                        time run their next tokens and drafts in one pass (ARC_NO_BATCH=1: turns only)\n"
            "  --dflash DIR          speculative decoding with a DFlash2 block drafter (z-lab Qwen3.8 27B, incoai\n"
            "                        Qwen3.6 35B-A3B; 7 drafts per step); default with --mtp: a *DFlash2* drafter\n"
            "                        for the model in the Hugging Face cache or the extra model folders, if any\n"
            "                        (incoai Qwen3.8 Flash Next: with --dflash DIR only; no faster on an SSD-bound laptop)\n"
            "  --no-dflash           no DFlash2 drafter (--mtp: the model's MTP block only)\n"
            "  --draft auto|mtp|dflash  with --mtp and a DFlash2 drafter: both, picked per step (auto, default), the MTP\n"
            "                        block only (= --no-dflash), or DFlash2 only (the MTP block is not loaded: more\n"
            "                        expert cache); the catalog's -nodf / -df variants\n"
            "  --think-budget N      default reasoning budget in tokens (requests can override it)\n"
            "  --preserve-thinking   earlier turns keep their reasoning (Qwen3.6 template option; requests can override it\n"
            "                        with chat_template_kwargs.preserve_thinking; Qwen3.8 always keeps it)\n"
            "  --mmproj FILE         image projector (default: found next to the model); --no-mmproj: none\n"
            "  --image-min-tokens N  an image becomes at least N tokens (smaller images are scaled up; Qwen: 32x32 pixels\n"
            "                        per token, default 256; Gemma 4 12B: 48x48 pixels, default 1120)\n"
            "  --image-max-tokens N  at most N tokens per image (larger images are scaled down; Qwen default 2048, Gemma\n"
            "                        4 12B 2048, above the 1120 it was trained with)\n"
            "  --budget GB           keep decoded weights resident up to GB\n"
            "MoE models (experts read from the SSD into a GPU cache; expert file in\n"
            "%%LOCALAPPDATA%%\\arcinfer\\zero, made on the first start, safe to delete):\n"
            "  --expert-cache-gb N   expert cache size in GiB (default: automatic: the GPU share Windows reports, your\n"
            "                        setting, less 1.2 GB for the driver, less the rest of the model)\n"
            "  --expert-cache-pct P  expert cache as a percentage of all experts\n"
            "  --pack FILE.gguf      make a package of a MoE model (its first part when split): the non-expert\n"
            "                        weights + an expert file under %%LOCALAPPDATA%%\\arcinfer\\zero\\models; afterwards\n"
            "                        the original GGUF can be moved away (e.g. to a USB disk)\n"
            "  --no-stream           every expert on the GPU (no SSD streaming; MoE models then get less context)\n"
            "  --attn-f32            f32 attention (tests)\n"
            "  GPU: picked at start: Xe2 / Xe3 GPUs (no 8-lane sub-groups) get 16-lane kernels, and XMX for the prompt\n"
            "  GEMMs when their results match the DP4a kernels' on a test (ARC_NO_XMX=1: DP4a; ARC_NO_SG8=1: 16 lanes)\n"
            "  --kernels DIR         kernel folder (default zero\\kernels);  --webui DIR  web UI folder\n"
            "  --session-cache-gb N  disk space for saved conversations (default 0 = off): switching between\n"
            "                        conversations loads the other one's state instead of processing its prompt again\n"
            "  --session-dir DIR     where they are kept (default " ARC_DATA_HELP "sessions)\n"
            "  --no-prefix-cache     do not keep system prompts on disk (default: kept in " ARC_DATA_HELP "prefix,\n"
            "                        per project folder for Pi (its 2 latest versions), the 8 latest for other clients;\n"
            "                        a request that starts with one loads it instead of processing it)\n"
            "  --prefix-cache-gb N   disk space for them, all models together (default 8, 0 = no limit; least recently\n"
            "                        used go first)\n"
            "  --no-swap             keep the loaded model (default: a request whose \"model\" is another model of\n"
            "                        --list-models, by id, name or file, makes the server load that one, context auto)\n"
            "  --no-web-tools        no POST /v1/tools/web_search and /v1/tools/web_fetch (the web UI's search and page\n"
            "                        tools; default: on, public http(s) addresses only)\n"
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

  std::unique_ptr<Loaded> cur;
  try {  // (a model that cannot load: its reason, not a silent abort)
    cur = load_model(model, alias, so);
  } catch (const std::exception& e) {
    fprintf(stderr, "arcinfer-server: cannot load %s: %s\n", model.c_str(), e.what());
    return 1;
  }
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
  // (no model loaded: a model switch failed and loading the previous model again did too; caller holds swap_mu)
  auto no_model = [&](HttpResponse& rs) {
    Json err = Json::object(), r = Json::object();
    err.set("message", "no model is loaded (the last model switch failed); ask for a model or restart the server");
    err.set("type", "server_error");
    r.set("error", err);
    return rs.send(503, "application/json", r.dump());
  };
  http.handler = [&](const HttpRequest& rq, HttpResponse& rs) {
    // the web UI's tools (tools/web_tools.h): no model involved, so they answer while a model loads or generates
    if (rq.method == "POST" && (rq.path == "/v1/tools/web_search" || rq.path == "/v1/tools/web_fetch")) {
      auto fail = [&](int st, const std::string& m) {
        Json er = Json::object(), r = Json::object();
        er.set("message", m);
        r.set("error", er);
        return rs.send(st, "application/json", r.dump());
      };
      if (!web_tools) return fail(404, "web tools are off (--no-web-tools)");
      Json in;
      if (!Json::try_parse(rq.body, in) || !in.is_object()) return fail(400, "invalid JSON body");
      Json r = Json::object();
      if (rq.path == "/v1/tools/web_search") {
        const std::string q = in["query"].str();
        if (q.empty()) return fail(400, "query required");
        std::string used, err;
        const auto res = webt::search(q, (size_t)std::clamp<int64_t>(in["max_results"].as_int(8), 1, 20), in["engine"].str("auto"), &used, &err);
        if (res.empty()) return fail(502, "search failed: " + err);
        Json a = Json::array();
        for (const webt::Result& x : res) {
          Json o = Json::object();
          o.set("title", x.title);
          o.set("url", x.url);
          size_t n = std::min<size_t>(x.snippet.size(), 360);  // (some snippets run to whole paragraphs)
          while (n < x.snippet.size() && ((unsigned char)x.snippet[n] & 0xc0) == 0x80) --n;
          o.set("snippet", x.snippet.substr(0, n) + (n < x.snippet.size() ? "..." : ""));
          a.push(o);
        }
        r.set("query", q);
        r.set("engine", used);
        r.set("results", a);
        return rs.send(200, "application/json", r.dump());
      }
      const std::string url = in["url"].str();
      if (url.empty()) return fail(400, "url required");
      const webt::Page pg = webt::get(url);
      if (!pg.error.empty()) return fail(502, pg.error);
      if (pg.status != 200) return fail(502, "HTTP " + std::to_string(pg.status) + " from " + pg.url);
      const std::string ct = webt::lower(pg.content_type);
      std::string title, text;
      if (ct.find("html") != std::string::npos || ct.empty()) text = webt::html_to_text(pg.body, &title);
      else if (ct.rfind("text/", 0) == 0 || ct.find("json") != std::string::npos || ct.find("xml") != std::string::npos) text = pg.body;
      else return fail(415, "unsupported content type " + pg.content_type + " (text and HTML pages only)");
      // a window of the text: offset, max_chars (cut on a UTF-8 character boundary)
      const size_t total = text.size(), max_chars = (size_t)std::clamp<int64_t>(in["max_chars"].as_int(12000), 500, 100000);
      size_t off = (size_t)std::clamp<int64_t>(in["offset"].as_int(0), 0, (int64_t)total), end = std::min(total, off + max_chars);
      while (off < total && ((unsigned char)text[off] & 0xc0) == 0x80) ++off;
      while (end < total && ((unsigned char)text[end] & 0xc0) == 0x80) --end;
      r.set("url", pg.url);
      r.set("title", title);
      r.set("content", text.substr(off, end - off));
      r.set("offset", (double)off);
      r.set("total_chars", (double)total);
      if (end < total) r.set("next_offset", (double)end);
      return rs.send(200, "application/json", r.dump());
    }
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
        if (!cur) return no_model(rs);
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
      if (!cur) return no_model(rs);
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
        if (!cur) return no_model(rs);
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
      if (!cur) return no_model(rs);
      auto entry =[&](const ModelInfo* m, const std::string& alias, bool vision, bool mtp) {
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
    const bool chat_post = rq.method == "POST" && rq.path == "/v1/chat/completions";
    {  // (no model: a switch failed and so did loading the previous model again; a chat request naming one loads it)
      std::shared_lock<std::shared_mutex> lk(swap_mu);
      if (!cur && !(swap && chat_post)) {
        Json err = Json::object(), r = Json::object();
        err.set("message", "no model is loaded (the last model switch failed); ask for a model or restart the server");
        err.set("type", "server_error");
        r.set("error", err);
        return rs.send(503, "application/json", r.dump());
      }
    }
    if (swap && (rq.path == "/v1/models" || rq.path == "/models")) {  // every model a request can switch to
      std::shared_lock<std::shared_mutex> lk(swap_mu);
      Json d = Json::array();
      std::map<std::string, uint64_t> sizes;  // (llama_models "size")
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
        set_window(j, loaded ? cur->eng->ctx : 0, m.ctx_train, loaded ? cur->eng->hp.n_vocab : 0, loaded ? cur->eng->hp.n_embd : 0);
        // (the loaded model: the encoders that did load)
        set_modalities(j, loaded ? cur->srv->has_vision() : m.vision, loaded ? cur->srv->has_audio() : m.audio);
        sizes[m.alias] = m.bytes;
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
        set_window(j, cur->eng->ctx, 0, cur->eng->hp.n_vocab, cur->eng->hp.n_embd);
        set_modalities(j, cur->srv->has_vision(), cur->srv->has_audio());
        sizes[cur->alias] = cur->srv->model_bytes_;
        d2.push(j);
        for (const Json& e : d.items()) d2.push(e);
        d = d2;
      }
      Json r = Json::object();
      r.set("object", "list");
      r.set("models", llama_models(d, sizes));
      r.set("data", d);
      return rs.send(200, "application/json", r.dump());
    }
    if (swap && rq.method == "POST" && rq.path == "/v1/chat/completions") {
      Json body;
      const std::string want = Json::try_parse(rq.body, body) ? body["model"].str() : "";
      bool other = false;
      {
        std::shared_lock<std::shared_mutex> lk(swap_mu);
        other = !want.empty() && (!cur || want != cur->alias);
      }
      if (other) {
        std::unique_lock<std::shared_mutex> lk(swap_mu);
        const ModelInfo* m = !cur || want != cur->alias ? find_model(want) : nullptr;
        if (m && (!cur || m->alias != cur->alias)) {
          const std::string prev_path = cur ? cur->path : std::string(), prev_alias = cur ? cur->alias : std::string();
          const auto t0 = clk::now();
          fprintf(stderr, "switching model: %s -> %s\n", prev_alias.c_str(), m->alias.c_str());
          if (cur && cur->srv) cur->srv->save_held();  // (switching back later loads the conversation instead of redoing it)
          cur.reset();  // frees the GPU memory first
          ServeOpts s2 = so;
          s2.ctx_auto = true;  // the --ctx given at start was for the first model
          s2.eo.mtp = m->mtp;  // the variant's settings: MTP block / drafter, image projector
          s2.dflash_auto = m->draft != 1;  // (-nodf: the MTP block only; -df: DFlash2 only)
          s2.eo.mtp_block = m->draft != 2;
          if (m->draft == 1) s2.eo.dflash.clear();
          s2.mmproj = m->vision ? "" : "-";
          // (a --dflash drafter is for the first model, and only with MTP; other models find their own)
          if (!m->mtp || !same_file(m->path, model)) s2.eo.dflash.clear();
          try {
            s2.warm_prompt = false;  // (the waiting request warms it)
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
                s3.dflash_auto = pm.draft != 1;
                s3.eo.mtp_block = pm.draft != 2;
                if (pm.draft == 1) s3.eo.dflash.clear();
                s3.mmproj = pm.vision ? "" : "-";
              }
            if (!s3.eo.mtp || !same_file(prev_path, model)) s3.eo.dflash.clear();
            std::string also;
            try {  // (that failing too leaves no model: requests get 503 until one names a model that loads)
              if (!prev_path.empty()) cur = load_model(prev_path, prev_alias, s3);
            } catch (const std::exception& e2) {
              cur.reset();
              also = std::string("; loading ") + prev_alias + " again failed too: " + e2.what();
              fprintf(stderr, "could not load %s again: %s; no model loaded\n", prev_alias.c_str(), e2.what());
            }
            Json err = Json::object(), r = Json::object();
            err.set("message", std::string("could not load model ") + want + ": " + e.what() + also);
            err.set("type", "server_error");
            r.set("error", err);
            return rs.send(500, "application/json", r.dump());
          }
        }
      }
    }
    std::shared_lock<std::shared_mutex> lk(swap_mu);
    if (!cur) {  // (a chat request naming no model, or an unknown one, while none is loaded)
      Json err = Json::object(), r = Json::object();
      err.set("message", "no model is loaded (the last model switch failed); ask for a model or restart the server");
      err.set("type", "server_error");
      r.set("error", err);
      return rs.send(503, "application/json", r.dump());
    }
    cur->srv->handle(rq, rs);
  };
  if (!http.listen(host, port)) {
    fprintf(stderr, "cannot listen on %s:%d\n", host.c_str(), port);
    return 1;
  }
  fprintf(stderr, "listening on http://%s:%d  (OpenAI API at /v1, web UI at /)\n", host.c_str(), port);
  if (host != "127.0.0.1" && host != "localhost" && host != "::1")  // (review 6.2: there is no authentication)
    fprintf(stderr, "warning: listening on %s: anyone who can reach this address can use the model (no API key)\n",
            host.c_str());
  http.run();
  return 0;
}
