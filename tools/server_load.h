// Loading a model: engine options, tokenizer, encoders, the Server around them.
// Included by tools/server.cpp only, at the place it was cut from (after class Server); not a header of its
// own. Split out of tools/server.cpp unchanged on 2026-10-24 (QUEUE item 11).

// ---------------------------------------------------------------- loading a model
struct ServeOpts {
  EngineOptions eo;
  bool ctx_auto = false;
  bool warm_prompt = true;  // (the server's start: also a long warm-up prompt; a switch for a request skips it)
  std::string webui, mmproj, sess_dir;
  int think_budget = -1;
  bool preserve_thinking = false;
  double sess_gb = 0.0;
  bool prefix_cache = true;
  double prefix_gb = 8.0;  // all models' saved system prompts together (0 = no limit)
  int img_min = 0, img_max = 0;  // image tokens per image (0 = the encoder's default)
  bool budget_given = false;     // --budget: no automatic choice of resident weights
  bool dflash_auto = true;       // arcinferzero, with MTP: a DFlash2 drafter found for the model drafts too (--no-dflash: not)
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
  fprintf(stderr, "arcinferzero-server: loading %s\n", model.c_str());
  EngineOptions eo = so.eo;
  bool dfl_auto = false;
  if (eo.mtp && eo.dflash.empty() && so.dflash_auto) {  // (before --ctx auto: the drafter takes GPU memory)
    eo.dflash = find_dflash(model);
    dfl_auto = !eo.dflash.empty();
    if (!eo.dflash.empty()) fprintf(stderr, "DFlash2 drafter found: %s (--no-dflash: MTP block only)\n", eo.dflash.c_str());
  }
  if (eo.mtp && !eo.mtp_block && eo.dflash.empty()) {
    fprintf(stderr, "--draft dflash: no DFlash2 drafter for this model, the MTP block drafts\n");
    eo.mtp_block = true;
  }
  if (eo.mtp && !eo.dflash.empty())
    fprintf(stderr, "drafting: %s\n", eo.mtp_block ? "DFlash2 and the MTP block, picked per step" : "DFlash2 only (no MTP block)");
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
    AutoCtx a = auto_ctx(model, eo, so.budget_given, enc);
    if (dfl_auto && eo.mtp_block && a.ctx < kDflashMinCtx) {  // (a drafter that leaves little context: the MTP block only)
      fprintf(stderr, "DFlash2 drafter not used: it would leave %d tokens of context (--dflash DIR: use it anyway)\n", a.ctx);
      eo.dflash.clear();
      a = auto_ctx(model, eo, so.budget_given, enc);
    }
    eo.ctx = a.ctx;
    if (!so.budget_given) eo.w8_budget_gb = a.resident ? 1e9 : eo.w8_budget_gb;
  }
  if (so.mmproj != "-") eo.stream_reserve = encoder_reserve(model, so.mmproj, so.img_max);  // (streamed experts' cache)
  L->eng = std::make_unique<Engine>(model, eo);
  Engine& eng = *L->eng;
  L->tok = std::make_unique<Tokenizer>(eng.gguf());
  fprintf(stderr, "loaded in %.1f s | %d layers | ctx %d%s | chunk %d | GPU %.2f GiB | weights %s\n", eng.load_ms / 1000,
          eng.hp.n_layer, eng.ctx, eng.slot_count() > 1 ? (" in each of " + std::to_string(eng.slot_count()) + " slots").c_str() : "",
          eng.max_chunk, eng.gpu_bytes / 1073741824.0, eng.w8_resident ? "W8 resident" : "W8 just-in-time");
  {  // compile kernels before serving (all of the generation's after an update)
    const auto tw = std::chrono::steady_clock::now();
    if (eng.warm_up_once(L->tok->encode("Hi", false)[0], so.warm_prompt))
      fprintf(stderr, "kernels warmed up for this model in %.1f s (once after an update)\n",
              std::chrono::duration<double>(std::chrono::steady_clock::now() - tw).count());
  }
  if (const char* f = getenv("ARC_FAIL_PASS")) eng.fail_after = atol(f);  // (tests: a pass that throws, see Engine)
  L->srv = std::make_unique<Server>(eng, *L->tok, alias, so.webui);
  Server& srv = *L->srv;
  srv.model_bytes_ = gguf_bytes(model);
  std::string mmproj = so.mmproj;
  if (!eng.hp.gemma && mmproj != "-") {
    // Qwen vision (dense and MoE): --mmproj, or an mmproj-*.gguf next to the model with a Qwen3-VL projector arcinferzero runs
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
  // (saved states only load into the same layout: no MTP block (-std) and DFlash2 without the MTP block (-df) each
  // keep their own folders next to the plain one; auto and -nodf share one: DFlash2 saves no state of its own, its
  // context rings start empty after a load — give it a suffix once that changes)
  std::string layout;
  if (!eo.mtp) layout += "-std";
  if (eo.mtp && !eo.dflash.empty() && !eo.mtp_block) layout += "-nomtp";
  if (L->eng->hp.gemma) layout += "-q2";  // (Gemma 4: two-pass states; the one-pass ones of before 2026-10-22 stay unused)
  if (so.prefix_cache && eo.attn_int8) {  // %LOCALAPPDATA%\arcinferzero\prefix\<model>-<size>[<layout>]
    std::error_code ec;
    const std::filesystem::path mp(model);
    srv.open_prefix_cache(app_dir() + ARC_DATA "prefix/" + mp.stem().string() + "-" +
                          std::to_string(std::filesystem::file_size(mp, ec)) + layout, so.prefix_gb);
  }
  if (so.sess_gb > 0 && eo.attn_int8) {
    // saved conversations of this model file: %LOCALAPPDATA%\arcinferzero\sessions\<model>-<size>
    std::error_code ec;
    const std::filesystem::path mp(model);
    std::string dir = so.sess_dir;
    if (dir.empty()) dir = app_dir() + ARC_DATA "sessions";
    srv.open_sessions(dir + "/" + mp.stem().string() + "-" + std::to_string(std::filesystem::file_size(mp, ec)) + layout,
                      so.sess_gb);
  }
  fprintf(stderr, "serving %s as \"%s\"\n", model.c_str(), alias.c_str());
  return L;
}
