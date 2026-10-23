// The model catalog (the Hugging Face cache and extra folders), DFlash2 drafters, --ctx auto, --list-models.
// Included by tools/server.cpp only, at the place it was cut from (after class Server); not a header of its
// own. Split out of tools/server.cpp unchanged on 2026-10-24 (QUEUE item 11).

// ---------------------------------------------------------------- model catalog
// The GGUF models in the Hugging Face cache and in arcinferzero's model folder: name (org/repo), file, path, size (a
// split file: all its parts), architecture, trained context, whether an image projector (mmproj-*, with or without
// a Gemma 4 audio encoder) or a Gemma drafter (mtp-*) sits next to it, Qwen MTP layers, and the model id clients use
// (the alias; the six models arcinferzero was tuned on keep their short ids). Projectors, drafters and the later parts
// of split files are not models.
struct ModelInfo {
  std::string name, file, path, arch, alias, shortcut;
  uint64_t bytes = 0;
  int64_t ctx_train = 0;
  // vision / mtp: what this entry loads. A file with an MTP block or drafter and / or an image projector is listed
  // once per combination (expand_variants): the full one under the file's id, the others with -mtp (MTP, no
  // vision), -vision (vision, no MTP) and -std (neither). base: the file's id; tags: "MTP + vision", ...
  bool supported = false, vision = false, mtp = false;
  bool moe = false;  // routed experts (arcinferzero streams them from the SSD)
  bool audio = false;  // the image projector also holds a Gemma 4 audio encoder (loaded with it: vision variants only)
  // arcinferzero, a file with a DFlash2 drafter: 0 both drafters, picked per step (the file's id); 1 the MTP block
  // only (-nodf); 2 DFlash2 only, the MTP block not loaded (-df)
  int draft = 0;
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
static std::string find_dflash(const std::string& model);  // (defined below) a DFlash2 drafter for the file

// the Hugging Face cache (HF_HUB_CACHE, HF_HOME\hub or %USERPROFILE%\.cache\huggingface\hub)
static std::string hf_hub_dir() {
  if (const char* v = getenv("HF_HUB_CACHE")) return v;
  if (const char* h = getenv("HF_HOME")) return std::string(h) + "/hub";
  if (const char* u = getenv("USERPROFILE")) return std::string(u) + "/.cache/huggingface/hub";
  return "";
}
// extra model folders (e.g. a second drive): ARC_MODEL_DIRS (';'-separated) and %LOCALAPPDATA%\arcinferzero\model-dirs.txt
// (one folder per line, '#' comments); each in the Hugging Face cache layout or with .gguf files directly in it
static std::vector<std::string> extra_model_dirs() {
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
  {
    std::ifstream f(std::filesystem::path(app_dir()) / "model-dirs.txt");
    for (std::string line; std::getline(f, line);) add(line);
  }
  return extra;
}
// (a folder on a drive that hangs, e.g. a USB stick after the laptop slept, would stall every model list and so
//  every server start: each folder is first listed by a detached thread and skipped if that takes over 2 s)
// 1: listed, 2: not a readable folder, 0: no answer within 2 s
static int folder_reachable(const std::filesystem::path& d) {
  auto st = std::make_shared<std::atomic<int>>(0);
  std::thread([d, st] {
    std::error_code e;
    const bool ok = std::filesystem::is_directory(d, e);
    if (ok) std::filesystem::directory_iterator it(d, e);
    st->store(ok && !e ? 1 : 2);
  }).detach();
  for (int i = 0; i < 100 && st->load() == 0; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(20));
  return st->load();
}

static std::vector<ModelInfo> scan_models() {
  namespace fs = std::filesystem;
  const std::string hub = hf_hub_dir();
  // (prio: 0 the Hugging Face cache and arcinferzero's own folders, 1+ the extra model folders in their order, so a model
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
  {
    int prio = 1;
    for (const auto& d : extra_model_dirs()) {
      const int r = folder_reachable(fs::path(d));
      if (r == 1) scan_hub(fs::path(d), prio);
      else if (r == 0) fprintf(stderr, "model folder %s did not answer within 2 s (drive asleep or hung?): skipped\n", d.c_str());
      ++prio;
    }
  }
  for (const auto& e : fs::directory_iterator(fs::path(app_dir()) / "models", ec))
    if (is_model(e.path())) cands.push_back({"arcinferzero/" + e.path().stem().string(), e.path().filename().string(), e.path(), e.last_write_time(ec)});
  // arcinferzero packages (zero/gguf_pack.h): %LOCALAPPDATA%\arcinferzero\zero\models\<name>\<name>.gguf
  for (const auto& d : fs::directory_iterator(zpack::packages_dir(), ec)) {
    const fs::path g = d.path() / (d.path().filename().string() + ".gguf");
    if (d.is_directory(ec) && fs::exists(g, ec))
      cands.push_back({"arcinferzero/" + d.path().filename().string(), g.filename().string(), g, fs::last_write_time(g, ec)});
  }
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
      // UD-IQ4_XS from the MTP repository (the tested 35B file since 18 Oct; it keeps the id it had): 35b as well (with
      // a UD-Q3_K_M or GSQ file also downloaded, the first one found keeps the shortcut)
      {"Qwen3.6-35B-A3B-UD-IQ4_XS.gguf", {"qwen3.6-35b-a3b-mtp", "35b"}},
      {"Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf", {"qwen3.8-flash-next-gsq-rco", "flash"}},
      // (its arcinferzero package keeps the id and the shortcut; with both on disk the GGUF's entry comes first)
      {"Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S.gguf", {"qwen3.8-flash-next-gsq-rco", "flash"}},
      {"Qwen3.5-9B-UD-Q4_K_XL.gguf", {"qwen3.5-9b", "9b"}},
      {"Qwen3.5-4B-UD-Q4_K_XL.gguf", {"qwen3.5-4b", "4b"}},
      {"gemma-4-12b-it-qat-q4_0.gguf", {"gemma4-12b", "g12"}},
      {"gemma-4-E4B_q4_0-it.gguf", {"gemma4-e4b", "e4b"}},
      {"gemma-4-26B_q4_0-it.gguf", {"gemma4-26b", "g26"}}};
  std::vector<ModelInfo> out;
  for (size_t i = 0; i < cands.size(); ++i) {
    const Cand& c = cands[i];
    if (i > 0 && cands[i - 1].name == c.name && cands[i - 1].file == c.file) continue;
    ModelInfo m;
    m.name = c.name;
    m.file = c.file;
    m.path = c.path.lexically_normal().string();  // (HF_HOME and the cache folder joined: one kind of separator)
    m.bytes = gguf_bytes(c.path.string());  // (a split file: all its parts)
    m.arch = "?";
    int64_t nextn = 0;
    bool gsq = false;  // qwen35moe: only GSQ 2-bit experts (tools/convert_gsq.py) have kernels
    try {
      GgufFile g(m.path);
      m.arch = g.get_str("general.architecture", "?");
      gsq = g.get_str("arcinferzero.expert_format", g.get_str("arcinfer.expert_format", "")) == "gsq2-v1";  // (old name too)
      m.ctx_train = g.get_int(m.arch + ".context_length", 0);
      nextn = g.get_int(m.arch + ".nextn_predict_layers", 0);
      m.moe = g.get_int(m.arch + ".expert_count", 0) > 0;
    } catch (const std::exception&) {
    }
    bool mmproj = false, drafter = false, audio = false;
    for (const auto& s : fs::directory_iterator(c.path.parent_path(), ec)) {
      const std::string f = lower(s.path().filename().string());
      if (s.path().extension() != ".gguf") continue;
      if (f.find("mmproj") != std::string::npos && !mmproj) {  // images: a projector arcinferzero reads
        try {
          const GgufFile g(s.path().string());
          const std::string pt = g.get_str("clip.vision.projector_type", "");
          // (gemma4v: the head sizes the GPU encoder takes, 64 and 72 (26B-A4B))
          const int64_t ve = g.get_int("clip.vision.embedding_length", 0), vh = g.get_int("clip.vision.attention.head_count", 0);
          const bool g4v_ok = vh > 0 && ve % vh == 0 && VisionGemma4V::head_ok((int)(ve / vh));
          mmproj = m.arch == "gemma4" ? pt == "gemma4uv" || (pt == "gemma4v" && g4v_ok)
                                      : (m.arch == "qwen35" || m.arch == "qwen35moe") && VisionQwen3VL::usable(g);
          // (audio: the encoders load_model loads, AudioGemma4A for gemma4a, AudioGemma4 for gemma4ua)
          const std::string at = g.get_str("clip.audio.projector_type", "");
          audio = mmproj && m.arch == "gemma4" && (at == "gemma4a" || at == "gemma4ua");
        } catch (const std::exception&) {
        }
      }
      drafter = drafter || f.rfind("mtp-", 0) == 0;
    }
    m.supported = m.arch == "qwen35" || m.arch == "qwen35moe" || m.arch == "gemma4";  // (MoE: GSQ or standard experts)
    m.supported = m.supported || m.arch == "qwen4exp";  // (Qwen3.8 Flash Next: arcinferzero only, experts streamed)
    (void)gsq;
    m.vision = mmproj;  // (Gemma 4 12B: gemma4uv; E2B / E4B: gemma4v)
    m.audio = audio;
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
  // which copy of a model keeps its id and shortcut: a package first (--pack moved the GGUF's expert file into it, so
  // loading the GGUF it was made from would write that file again), then the Hugging Face cache, then the extra
  // folders (a backup on another drive); the others get -2, -3, ...
  {
    std::error_code e2;
    const std::string hub_c = lower_ascii(fs::weakly_canonical(hf_hub_dir(), e2).string()),
                      pk_c = lower_ascii(fs::weakly_canonical(zpack::packages_dir(), e2).string());
    std::vector<std::pair<int, ModelInfo>> ranked;  // (each path looked at once)
    for (ModelInfo& m : out) {
      std::error_code e3;
      const std::string p = lower_ascii(fs::weakly_canonical(m.path, e3).string());
      const int r = p.rfind(pk_c, 0) == 0 ? 0 : !hub_c.empty() && p.rfind(hub_c, 0) == 0 ? 1 : 2;
      ranked.push_back({r, std::move(m)});
    }
    std::stable_sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    for (size_t i = 0; i < out.size(); ++i) out[i] = std::move(ranked[i].second);
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
      e.audio = v.vision && m.audio;  // (the audio encoder comes with the projector)
      e.alias = m.alias + v.suffix;
      if (!m.shortcut.empty()) e.shortcut = m.shortcut + v.suffix;
      e.tags = v.mtp && v.vision ? "MTP + vision" : v.mtp ? "MTP" : v.vision ? "vision" : "standard";
      all.push_back(e);
    }
    // a file with a DFlash2 drafter: its main variant also with one drafter only (to pick in Pi's /model)
    if (m.supported && all.back().base == m.alias && all[all.size() - vs.size()].mtp && !find_dflash(m.path).empty()) {
      const ModelInfo main = all[all.size() - vs.size()];
      for (int d = 1; d <= 2; ++d) {
        ModelInfo e = main;
        e.draft = d;
        const char* suf = d == 1 ? "-nodf" : "-df";
        e.alias = main.alias + suf;
        if (!main.shortcut.empty()) e.shortcut = main.shortcut + suf;
        e.tags = main.tags + (d == 1 ? " (MTP block only)" : " (DFlash2 only)");
        all.push_back(e);
      }
    }
  }
  return all;
}

// A DFlash2 block drafter for a model file: a *DFlash2* repository (z-lab, incoai: config.json + model.safetensors)
// in the Hugging Face cache or an extra model folder whose config is a DFlash2DraftModel of the model's width
// (hidden_size = <arch>.embedding_length) and depth (num_target_layers = <arch>.block_count without the MTP layers);
// the newest snapshot of the first folder that has one. Empty if none.
// a DFlash2 drafter found next to a model is used only when the automatic context with it is at least this long (27B
// with images, MTP and the drafter: 8,192 tokens; without the drafter 45,056)
constexpr int kDflashMinCtx = 32768;
static std::string find_dflash(const std::string& model) {
  namespace fs = std::filesystem;
  int64_t n_embd = 0, n_layer = 0;
  try {
    const GgufFile g(model);
    const std::string arch = g.get_str("general.architecture", "");
    n_embd = g.get_int(arch + ".embedding_length", 0);
    n_layer = g.get_int(arch + ".block_count", 0) - g.get_int(arch + ".nextn_predict_layers", 0);
  } catch (const std::exception&) {
    return "";
  }
  if (n_embd <= 0 || n_layer <= 0) return "";
  std::vector<std::string> dirs;
  if (!hf_hub_dir().empty()) dirs.push_back(hf_hub_dir());
  for (const auto& d : extra_model_dirs())
    if (folder_reachable(fs::path(d)) == 1) dirs.push_back(d);
  std::error_code ec;
  for (const auto& dir : dirs) {
    fs::path best;
    fs::file_time_type best_t{};
    for (const auto& d : fs::directory_iterator(fs::path(dir), ec)) {
      const std::string dn = lower_ascii(d.path().filename().string());
      if (!d.is_directory(ec) || dn.rfind("models--", 0) != 0 || dn.find("dflash") == std::string::npos) continue;
      for (const auto& s : fs::directory_iterator(d.path() / "snapshots", ec)) {
        if (!fs::exists(s.path() / "model.safetensors", ec)) continue;
        std::ifstream f(s.path() / "config.json", std::ios::binary);
        Json cfg;
        if (!f || !Json::try_parse(std::string(std::istreambuf_iterator<char>(f), {}), cfg) || !cfg.is_object()) continue;
        bool arch = false;
        for (const auto& a : cfg["architectures"].items()) arch = arch || a.str() == "DFlash2DraftModel";
        if (!arch || cfg["hidden_size"].as_int() != n_embd || cfg["num_target_layers"].as_int() != n_layer ||
            !cfg["dflash_config"].is_object())
          continue;
        const fs::file_time_type t = fs::last_write_time(s.path(), ec);
        if (best.empty() || t > best_t) best = s.path(), best_t = t;
      }
    }
    if (!best.empty()) return best.lexically_normal().string();
  }
  return "";
}

// ---------------------------------------------------------------- --ctx auto
// The largest context (a multiple of 4096, at most the trained one) for which the server's GPU memory, the ~0.22 GiB
// the driver adds and ~0.95 GiB for the desktop and other apps fit the device's global memory. A dry load (the
// engine adds up its allocations without making them: seconds, no GPU memory) gives what does not depend on the
// context; the result is remembered per model file, chunk size, MTP setting and device memory in %LOCALAPPDATA%\arcinferzero\ctx-auto.txt.
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
  return app_dir() + ARC_DATA "ctx-auto.txt";
}
static std::string ctx_key(const std::string& model, const EngineOptions& eo) {
  namespace fs = std::filesystem;
  std::error_code ec;
  std::string p = fs::absolute(model, ec).lexically_normal().make_preferred().string();  // one spelling per file
  for (auto& c : p) c = (char)tolower((unsigned char)c);
  return p + "|" + std::to_string(fs::file_size(model, ec)) + "|" + std::to_string(eo.chunk) +
         "|z" + std::string(eo.stream_experts ? "s" : "n") +
         (eo.dflash.empty() ? std::string() : eo.mtp_block ? "|dflash" : "|dflash-only") + 
         (eo.slots > 1 ? "|slots" + std::to_string(eo.slots) + "/" + std::to_string(Engine::kSlotCtxFloor) : std::string()) +
         "|md" + std::to_string(eo.mtp_drafts) + "|" +  // (DFlash2 drafter: its weights and buffers; 4-bit V)
         (eo.min_chunk ? "/" + std::to_string(eo.min_chunk) : std::string()) + "|" + (eo.mtp ? "mtp" : "-") + "|" + (eo.attn_int8 ? "i8" : "f32") + "|" + std::to_string(device_global_mem()) +
         "|m20";  // engine memory layout version (m2: Q8_0 not stored twice, m4: MoE stacked router, m5: 3 MTP drafts
                 // for 27B, m6: MoE prompt GEMM staging, m7: MoE buffers shared, m8: MoE MTP 1 draft, Q3_K experts, m9: memory parts in the cache,
                 // m12: streamed MoE reserves 128K of context before the expert cache, m13: streamed MoE chunk 8192, m14: slot_bytes with every
                 // slot buffer, f16 K / V with --attn-f32, m15: --q2 Gemma MoE chunk 2048, m16: Gemma 4 always two-pass, m17: shared-KV Gemma
                 // sliding rings, m18: Gemma int8 KV copies only for the drafter's layers,
                 // m19: GPU share - 1.2 GB, no Windows reserve, m20: Gemma streamed experts' cache floor 0.6);
                 // bump on changes
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
    e2.stream_reserve = std::max(e2.stream_reserve, reserve);  // (streamed experts: the cache leaves room for the encoders)
    e2.w8_budget_gb = budget_gb;
    Engine probe(model, e2);
    res = probe.w8_resident;
    const double gib = 1073741824.0,
                 budget = probe.device_mem() - Engine::kDriverSlack - probe.snapshot_bytes() - (double)reserve;
    const double base = (double)probe.gpu_bytes - (double)probe.ctx_bytes(e2.ctx);
    // (--slots N: N - 1 more KV caches, states and snapshots of the same context)
    auto need = [&](int c) { return base + probe.ctx_bytes(c) + (double)(std::max(eo.slots, 1) - 1) * probe.slot_bytes(c); };
    int ctx = 0;
    for (int c = 4096; c <= trained; c += 4096)
      if (need(c) <= budget) ctx = c;
    if (trained % 4096 && need(trained) <= budget) ctx = trained;
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
  std::map<std::string, std::string> dflash;  // model file -> its DFlash2 drafter (one search per file, not per variant)
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
    j.set("moe", m.moe);
    j.set("alias", m.alias);
    j.set("short", m.shortcut);
    j.set("base", m.base);
    j.set("tags", m.tags);
    EngineOptions eo;
    eo.mtp = m.mtp;
    // dflash: a DFlash2 drafter for this model was found (dflash_path); variants with MTP draft with it as well (the
    // server picks it up with --mtp, see load_model), so their context and defaults count it
    std::string dfl;
    if (m.supported) {
      if (!dflash.count(m.path)) dflash[m.path] = find_dflash(m.path);
      dfl = dflash[m.path];
    }
    if (m.draft == 1) dfl.clear();
    j.set("dflash", !dfl.empty());
    j.set("dflash_path", dfl);
    j.set("draft", m.draft == 1 ? "mtp" : m.draft == 2 ? "dflash" : "auto");
    if (m.mtp) eo.dflash = dfl;
    eo.mtp_block = m.draft != 2;
    AutoCtx a;  // computed now (dry load) if not known yet; per variant (MTP block, encoders' memory)
    a.ctx = 0;
    if (m.supported)
      try {
        const size_t enc = m.vision ? encoder_reserve(m.path, "", 0) : 0;
        a = auto_ctx(m.path, eo, false, enc);
        // (the drafter only where it leaves kDflashMinCtx tokens of context: the server drops it there too)
        if (!eo.dflash.empty() && eo.mtp_block && a.ctx < kDflashMinCtx) {
          eo.dflash.clear();
          a = auto_ctx(m.path, eo, false, enc);
          j.set("dflash", false);
        }
      } catch (const std::exception&) {}
    j.set("ctx", a.ctx);
    if (m.supported) {  // the values the server picks for this variant when not given (the launcher fills them in)
      Json d = Json::object();
      bool moe = m.arch == "qwen35moe" || m.arch == "qwen4exp";
      const bool gemma = m.arch == "gemma4";
      int64_t n_embd = 0;
      std::string pt;  // image projector type
      try {
        const GgufFile g(m.path);
        n_embd = g.get_int(m.arch + ".embedding_length", 0);
        moe = moe || g.get_int(m.arch + ".expert_count", 0) > 0;  // (Gemma 4 26B-A4B)
      } catch (const std::exception&) {}
      if (m.vision) {
        std::error_code ec;
        for (const auto& de : std::filesystem::directory_iterator(std::filesystem::path(m.path).parent_path(), ec))
          if (lower_ascii(de.path().filename().string()).find("mmproj") != std::string::npos && de.path().extension() == ".gguf") {
            try { pt = GgufFile(de.path().string()).get_str("clip.vision.projector_type", ""); } catch (const std::exception&) {}
            break;
          }
      }
      // (the engine's default: streamed experts 8192, Flash Next 2048, resident decoded weights 512)
      int chunk = m.arch == "qwen4exp" || (gemma && moe) ? 2048 : moe ? 8192 : a.resident ? 512 : 4096;  // (as the engine)
      if (gemma && m.vision) chunk = std::max(chunk, 2048);  // (an image is one chunk)
      d.set("chunk", chunk);
      d.set("mtp_drafts", !m.mtp ? 0 : !eo.dflash.empty() ? 7 : gemma ? (moe ? 3 : 2) : moe ? 5 : n_embd >= 5120 ? 7 : n_embd <= 2560 ? 3 : 2);
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
