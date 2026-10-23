// The data folder: %LOCALAPPDATA%\arcinferzero (kernel cache, model packages and expert files, context / session /
// prefix caches, logs, model-dirs.txt). Until 2026-10-24 the project was called arcinfer and the folder
// %LOCALAPPDATA%\arcinfer: when only that one exists it is renamed once (the same drive: instant, the packages are not
// copied). If the rename fails (a file held open by an older process) the old folder is used as it is.
#pragma once
#include <cstdlib>
#include <filesystem>
#include <string>

inline std::string app_dir() {
  static const std::string dir = [] {
    namespace fs = std::filesystem;
    const char* la = getenv("LOCALAPPDATA");
    const fs::path base = la ? la : ".", now = base / "arcinferzero", old = base / "arcinfer";
    std::error_code ec;
    if (!fs::exists(now, ec) && fs::is_directory(old, ec)) {
      fs::rename(old, now, ec);
      if (ec) return old.string();
    }
    return now.string();
  }();
  return dir;
}
