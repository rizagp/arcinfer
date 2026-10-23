// The chat endpoint (/v1/chat/completions): prompt, media, generation, streaming, tool calls.
// A part of class Server (tools/server.cpp), included inside the class body at the place it was cut from:
// not a header of its own. Split out of tools/server.cpp unchanged on 2026-10-24 (QUEUE item 11).

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
    // the engine lock, taken before the media are encoded: the media cache and the encoders (which share the GPU
    // queue with the engine) must not run alongside another request (one thread per connection)
    std::unique_lock<std::mutex> lock(mu_, std::defer_lock);
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
        acquire(lock);
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
              if (img_cache_.size() > 64 || img_cache_bytes_ + e->size() * 4 > (512u << 20)) {
                img_cache_.clear();
                img_cache_bytes_ = 0;
              }
              img_cache_bytes_ += e->size() * 4;
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
    if (P.empty()) return rs.send(400, "application/json", err("the prompt is empty (no tokens)").dump());
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

    if (!lock.owns_lock()) acquire(lock);
    const int my_slot = claim_slot(lock, P);
    struct SlotFree {  // (the lock is still held when this runs: declared after it)
      Server& s;
      int k;
      ~SlotFree() {
        s.slots_[k].busy = false;
        s.slot_cv_.notify_all();
      }
    } slot_free{*this, my_slot};
    ++busy_;
    struct Idle { std::atomic<int>& b; ~Idle() { --b; } } idle{busy_};
    const std::string id = "chatcmpl-arc" + std::to_string(++counter_);
    const long long created = (long long)time(nullptr);
    if (stream) rs.sse_begin();

    // ---- prompt processing (with prefix reuse)
    const auto t_prompt = clk::now();
    // end of the system prompt (the start of the second turn; Gemma's first turn follows <bos>): every conversation
    // of a client (Pi: its system prompt and tool list, ~4K tokens) shares it
    const size_t sys_cut = system_end(P);
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
    // --slots: before each piece of the prompt, a waiting request in another slot may take a step (then the engine is
    // bound to this slot and its callbacks set again)
    auto prompt_turn = [&] {
      take_turns(lock, my_slot);
      eng_.cancel = client_gone;
    };
    // runs P[done, upto); false when cancelled (cached_ then matches the engine again)
    auto feed_tokens = [&](size_t upto) {
      try {
        if (!eng_.w8_resident && upto > done) {
          // just-in-time weights: one call, so the engine runs the prompt layer by layer (each matrix decoded once);
          // progress is reported per layer (--slots: calls of up to 8192 tokens, other slots' steps in between)
          const size_t piece = slots_.size() > 1 ? std::max<size_t>(eng_.max_chunk, 8192) : upto - done;
          while (done < upto) {
            prompt_turn();
            const size_t base = done, to = std::min(upto, done + piece);
            eng_.progress = [&](int d, int) { report(base + d); };
            logits = eng_.forward(P.data() + done, (int)(to - done));
            eng_.progress = nullptr;
            cached_.insert(cached_.end(), P.begin() + done, P.begin() + to);
            done = to;
            report(done);
          }
          return true;
        }
        while (done < upto) {
          if (client_gone()) return false;
          prompt_turn();
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
    if (getenv("ARC_PROF_REQ")) eng_.profile = true;  // (the generation's kernels; printed after the request)
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
    // (per drafter for the log: [0] the MTP block / Gemma drafter, [1] DFlash2; steps, drafts, accepted)
    int dr_steps[2] = {0, 0}, dr_drafts[2] = {0, 0}, dr_acc[2] = {0, 0}, dr_cur = 0;
    size_t call_from = std::string::npos;  // where the open tool call's text starts (after <|tool_call>)
    int n_tool_tok = 0, n_tool_fixed = 0;   // tokens drawn inside tool calls; of them, a refused first choice
    int n_batched = 0;  // (--slots: steps run in one pass with other requests')
    ++gen_active_;
    struct GenLeave {  // (requests waiting for this one to lead their steps run them themselves)
      Server& s;
      ~GenLeave() {
        --s.gen_active_;
        s.gen_cv_.notify_all();
      }
    } gen_leave{*this};
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
          ++dr_acc[dr_cur];
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
      for (const auto& s : stops) {
        // (cut at the match in the recent tail, not at an earlier copy of the stop string in the text)
        const size_t at = s.empty() || text.size() < s.size()
                              ? std::string::npos
                              : text.find(s, text.size() > s.size() + 16 ? text.size() - s.size() - 16 : 0);
        if (at != std::string::npos) {
          text = text.substr(0, at);
          hit_stop = true;
        }
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
      take_turns(lock, my_slot);  // (--slots; the drafts are settled here)
      eng_.cancel = nullptr;      // (another request's prompt may have set its own)
      GenStep st;
      st.slot = my_slot;
      st.tok = t;
      st.max_drafts = spec && (int)cached_.size() + 4 < eng_.ctx && i + 1 < max_tokens ? std::min(eng_.opt.mtp_drafts, max_tokens - i - 1) : 0;
      gen_step(lock, st, my_slot);
      n_batched += st.batched ? 1 : 0;
      cached_.push_back(t);
      if (st.max_drafts > 0) {
        drafts = std::move(st.drafts);
        vrows = std::move(st.rows);
        logits.assign(vrows.begin(), vrows.begin() + V);
        acc = 0;
        n_drafts += (int)drafts.size();
        dr_cur = st.dflash ? 1 : 0;
        ++dr_steps[dr_cur];
        dr_drafts[dr_cur] += (int)drafts.size();
      } else {
        logits = std::move(st.rows);
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
    if (n_drafts && !dr_steps[1]) fprintf(stderr, " | MTP %d/%d drafts accepted", n_accepted, n_drafts);
    if (dr_steps[1])  // (DFlash2 drafted at least once: both drafters, each with its steps and accepted drafts)
      fprintf(stderr, " | DFlash2 %d steps, %d/%d drafts accepted | MTP %d steps, %d/%d", dr_steps[1], dr_acc[1],
              dr_drafts[1], dr_steps[0], dr_acc[0], dr_drafts[0]);
    if (n_tool_fixed) fprintf(stderr, " | tool call: %d of %d tokens corrected", n_tool_fixed, n_tool_tok);
    if (n_batched) fprintf(stderr, " | %d steps batched with other slots", n_batched);
    fprintf(stderr, "\n");
    // (ARC_PROF_REQ=1: the GPU time by kernel of each request, for diagnosis; every kernel is timed, which costs time)
    static const bool prof_req = getenv("ARC_PROF_REQ") != nullptr;
    if (prof_req) {
      double total = 0;
      std::vector<std::pair<double, std::string>> rows;
      for (auto& [n, ms] : eng_.prof_ms) total += ms, rows.push_back({ms, n});
      std::sort(rows.rbegin(), rows.rend());
      fprintf(stderr, "  GPU time %.1f ms:", total);
      for (size_t r = 0; r < rows.size() && r < 12; ++r)
        fprintf(stderr, " %s %.0f (%d)", rows[r].second.c_str(), rows[r].first, eng_.prof_n[rows[r].second]);
      fprintf(stderr, "\n");
      eng_.prof_ms.clear();
      eng_.prof_n.clear();
    }
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
      // (the last chunk carries usage and timings together, as llama-server's does: Unsloth Studio keeps the timings
      //  only from a chunk that has usage, and showed no token statistics)
      Json fin = chunk(id, created, Json::object(), Json(finish));
      fin.set("usage", usage);
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
        u.set("timings", timings);
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
