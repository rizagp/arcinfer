// The server's saved state: system-prompt prefix cache on disk and saved conversations (sessions).
// A part of class Server (tools/server.cpp), included inside the class body at the place it was cut from:
// not a header of its own. Split out of tools/server.cpp unchanged on 2026-10-24 (QUEUE item 11).

 public:
  bool save_held() {
    std::lock_guard<std::mutex> lock(mu_);
    bool any = false;
    struct Back {  // (the slot bound before: a request between two steps finds the engine as it left it)
      Server& s;
      int k;
      ~Back() {
        try { s.use_slot(k); } catch (const std::exception&) {}
      }
    } back{*this, cur_slot_};
    for (int k = 0; k < (int)slots_.size(); ++k) {  // (--slots: each slot's; a request takes its own back with use_slot)
      if (slots_.size() > 1 && slots_[k].busy) continue;  // (a running request's: mid-answer)
      use_slot(k);
      if (!sess_cap_ || cur_saved_ || cached_.size() < 1024) continue;
      save_current();
      any = (!saved_.empty() && written(saved_.back())) || any;
    }
    return any;
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
      for (auto& x : slots_) x.cur_saved = false;
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
