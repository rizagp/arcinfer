// GPT-2 style byte-level BPE tokenizer from GGUF (Qwen2/Qwen3.5 pre-tokenizer), encode + decode.
#pragma once
#include "gguf.h"
#include "unicode_tables.h"
#include <algorithm>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace uni {
enum : uint16_t { NUMBER = 0x02, LETTER = 0x04, MARK = 0x10 };

inline uint16_t flags(uint32_t cp) {
  const size_t n = sizeof(kUniRangeStart) / sizeof(kUniRangeStart[0]);
  size_t lo = 0, hi = n;  // last range whose start <= cp
  while (hi - lo > 1) {
    const size_t mid = (lo + hi) / 2;
    if (kUniRangeStart[mid] <= cp) lo = mid;
    else hi = mid;
  }
  return kUniRangeFlags[lo];
}
inline bool is_ws(uint32_t cp) {
  for (uint32_t w : kUniWhitespace)
    if (w == cp) return true;
  return false;
}

// Decodes UTF-8 into codepoints, with the byte offset of each codepoint (invalid bytes map to U+FFFD).
inline void utf8_decode(const std::string& s, std::vector<uint32_t>& cps, std::vector<size_t>& offs) {
  cps.clear();
  offs.clear();
  for (size_t i = 0; i < s.size();) {
    const unsigned char c = (unsigned char)s[i];
    int len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
    if (i + len > s.size()) len = 1;
    uint32_t cp = len == 1 ? c : len == 2 ? (c & 0x1F) : len == 3 ? (c & 0x0F) : (c & 0x07);
    for (int k = 1; k < len; ++k) cp = (cp << 6) | ((unsigned char)s[i + k] & 0x3F);
    if (len == 1 && c >= 0x80) cp = 0xFFFD;
    cps.push_back(cp);
    offs.push_back(i);
    i += len;
  }
  offs.push_back(s.size());
}

inline void utf8_append(std::string& out, uint32_t cp) {
  if (cp < 0x80) out.push_back((char)cp);
  else if (cp < 0x800) { out.push_back((char)(0xC0 | (cp >> 6))); out.push_back((char)(0x80 | (cp & 0x3F))); }
  else if (cp < 0x10000) {
    out.push_back((char)(0xE0 | (cp >> 12))); out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back((char)(0x80 | (cp & 0x3F)));
  } else {
    out.push_back((char)(0xF0 | (cp >> 18))); out.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back((char)(0x80 | ((cp >> 6) & 0x3F))); out.push_back((char)(0x80 | (cp & 0x3F)));
  }
}
}  // namespace uni

class Tokenizer {
 public:
  std::vector<std::string> tokens;  // GGUF token strings (byte-level encoded)
  std::vector<int64_t> types;       // 1 normal, 3 control, 4 user-defined, ...
  int eos = -1, bos = -1;
  std::unordered_map<std::string, int> special;  // control / user-defined tokens by text

  explicit Tokenizer(const GgufFile& f) {
    auto t = f.get_strs("tokenizer.ggml.tokens");
    if (!t) throw std::runtime_error("GGUF has no tokenizer.ggml.tokens");
    tokens = *t;
    if (auto ty = f.get_ints("tokenizer.ggml.token_type")) types = *ty;
    eos = (int)f.get_int("tokenizer.ggml.eos_token_id", -1);
    bos = (int)f.get_int("tokenizer.ggml.bos_token_id", -1);
    // Gemma 4: SentencePiece-style BPE (spaces as U+2581, merges over UTF-8 characters, <0xXX> byte fallback)
    spm_ = f.get_str("tokenizer.ggml.model") == "gemma4";
    for (size_t i = 0; i < tokens.size(); ++i) {
      if (i < types.size() && (types[i] == 3 || types[i] == 4)) {
        special[tokens[i]] = (int)i;
        specials_by_len_.push_back(tokens[i]);
      } else {
        token_id_[tokens[i]] = (int)i;
      }
    }
    std::sort(specials_by_len_.begin(), specials_by_len_.end(),
              [](const std::string& a, const std::string& b) { return a.size() > b.size(); });
    if (auto m = f.get_strs("tokenizer.ggml.merges"))
      for (size_t r = 0; r < m->size(); ++r) merge_rank_[(*m)[r]] = (int)r;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
      const bool printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
      const uint32_t cp = printable ? b : 256 + n++;
      cp_to_byte_[cp] = (uint8_t)b;
      byte_to_str_[b].clear();
      uni::utf8_append(byte_to_str_[b], cp);
    }
  }

  int token(const std::string& s) const {
    auto it = special.find(s);
    if (it != special.end()) return it->second;
    auto jt = token_id_.find(s);
    return jt == token_id_.end() ? -1 : jt->second;
  }
  bool is_special(int id) const { return id >= 0 && id < (int)types.size() && (types[id] == 3 || types[id] == 4); }

  // Encodes text; special tokens written literally in the text are recognized when parse_special is set.
  std::vector<int> encode(const std::string& text, bool parse_special = true) const {
    std::vector<int> out;
    size_t start = 0;
    if (parse_special) {
      for (size_t i = 0; i < text.size();) {
        const std::string* hit = nullptr;
        if (text[i] == '<') {
          for (const auto& sp : specials_by_len_)
            if (text.compare(i, sp.size(), sp) == 0) { hit = &sp; break; }
        }
        if (hit) {
          encode_plain(text.substr(start, i - start), out);
          out.push_back(special.at(*hit));
          i += hit->size();
          start = i;
        } else {
          ++i;
        }
      }
    }
    encode_plain(text.substr(start), out);
    return out;
  }

  // Text for one token (special tokens are returned verbatim).
  std::string piece(int id) const {
    if (id < 0 || id >= (int)tokens.size()) return "";
    const std::string& s = tokens[id];
    if (is_special(id)) return s;
    if (spm_) {
      if (id < (int)types.size() && types[id] == 6 && s.size() == 6 && s.compare(0, 3, "<0x") == 0)
        return std::string(1, (char)strtol(s.substr(3, 2).c_str(), nullptr, 16));  // byte token
      std::string out;
      for (size_t i = 0; i < s.size();) {
        if (s.compare(i, 3, kSpace) == 0) { out.push_back(' '); i += 3; }
        else out.push_back(s[i++]);
      }
      return out;
    }
    std::vector<uint32_t> cps;
    std::vector<size_t> offs;
    uni::utf8_decode(s, cps, offs);
    std::string out;
    for (uint32_t cp : cps) {
      auto it = cp_to_byte_.find(cp);
      if (it != cp_to_byte_.end()) out.push_back((char)it->second);
    }
    return out;
  }
  std::string decode(const std::vector<int>& ids) const {
    std::string s;
    for (int id : ids) s += piece(id);
    return s;
  }

 private:
  bool spm_ = false;
  static constexpr const char* kSpace = "\xe2\x96\x81";  // U+2581, SentencePiece's visible space
  std::unordered_map<std::string, int> token_id_;
  std::unordered_map<std::string, int> merge_rank_;
  std::unordered_map<uint32_t, uint8_t> cp_to_byte_;
  std::string byte_to_str_[256];
  std::vector<std::string> specials_by_len_;

  // Qwen3.5 pre-tokenizer, equivalent to the regex
  //   (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}|
  //    ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
  static void pretokenize(const std::vector<uint32_t>& c, std::vector<std::pair<size_t, size_t>>& words) {
    const size_t n = c.size();
    auto F = [&](size_t i) -> uint16_t { return i < n ? uni::flags(c[i]) : 0; };
    auto LM = [&](size_t i) { return i < n && (F(i) & (uni::LETTER | uni::MARK)); };
    auto N_ = [&](size_t i) { return i < n && (F(i) & uni::NUMBER); };
    auto WS = [&](size_t i) { return i < n && uni::is_ws(c[i]); };
    auto NL = [&](size_t i) { return i < n && (c[i] == '\r' || c[i] == '\n'); };
    auto PUNCT = [&](size_t i) { return i < n && !WS(i) && !(F(i) & (uni::LETTER | uni::MARK | uni::NUMBER)); };
    auto lower = [&](size_t i) -> uint32_t { return i < n && c[i] < 128 ? (uint32_t)tolower((int)c[i]) : (i < n ? c[i] : 0); };
    size_t i = 0;
    while (i < n) {
      size_t e = 0;
      // 1. contractions
      if (c[i] == '\'') {
        const uint32_t a = lower(i + 1), b = lower(i + 2);
        if (a == 's' || a == 't' || a == 'm' || a == 'd') e = i + 2;
        else if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) e = i + 3;
      }
      // 2. optional non-letter/number prefix + letters/marks
      if (!e) {
        const bool prefix_ok = !NL(i) && !(F(i) & (uni::LETTER | uni::NUMBER));
        size_t j = 0;
        if (prefix_ok && LM(i + 1)) j = i + 1;
        else if (LM(i)) j = i;
        if (j || LM(i)) {
          if (!j) j = i;
          while (LM(j)) ++j;
          e = j;
        }
      }
      // 3. one number
      if (!e && N_(i)) e = i + 1;
      // 4. optional space + punctuation run + trailing newlines
      if (!e) {
        size_t j = 0;
        if (c[i] == ' ' && PUNCT(i + 1)) j = i + 1;
        else if (PUNCT(i)) j = i;
        if (j || PUNCT(i)) {
          if (!j) j = i;
          while (PUNCT(j)) ++j;
          while (NL(j)) ++j;
          e = j;
        }
      }
      if (!e && WS(i)) {
        size_t j = i;
        while (WS(j)) ++j;
        // 5. \s*[\r\n]+ : up to the last newline of the whitespace run
        size_t last_nl = 0;
        bool has_nl = false;
        for (size_t k = i; k < j; ++k)
          if (NL(k)) { last_nl = k; has_nl = true; }
        if (has_nl) e = last_nl + 1;
        // 6. \s+(?!\S)
        else if (j == n) e = j;
        else if (j - i >= 2) e = j - 1;
        // 7. \s+
        else e = j;
      }
      if (!e) e = i + 1;
      words.push_back({i, e});
      i = e;
    }
  }

  // Applies the BPE merges (lowest rank first) to a symbol sequence in place.
  void merge_symbols(std::vector<std::string>& sym) const {
    if (sym.size() > 1) {
      std::vector<int> prev(sym.size()), next(sym.size());
      for (size_t i = 0; i < sym.size(); ++i) { prev[i] = (int)i - 1; next[i] = i + 1 < sym.size() ? (int)i + 1 : -1; }
      using Item = std::tuple<int, int, int, size_t, size_t>;  // rank, left, right, left len, right len
      std::priority_queue<Item, std::vector<Item>, std::greater<Item>> pq;
      auto push = [&](int l) {
        const int r = next[l];
        if (l < 0 || r < 0) return;
        auto it = merge_rank_.find(sym[l] + " " + sym[r]);
        if (it != merge_rank_.end()) pq.push({it->second, l, r, sym[l].size(), sym[r].size()});
      };
      for (size_t i = 0; i + 1 < sym.size(); ++i) push((int)i);
      while (!pq.empty()) {
        auto [rank, l, r, ll, rl] = pq.top();
        pq.pop();
        if (next[l] != r || sym[l].size() != ll || sym[r].size() != rl) continue;  // stale
        sym[l] += sym[r];
        sym[r].clear();
        next[l] = next[r];
        if (next[r] >= 0) prev[next[r]] = l;
        if (prev[l] >= 0) push(prev[l]);
        push(l);
      }
      std::vector<std::string> merged;
      for (int i = 0; i >= 0; i = next[i]) merged.push_back(sym[i]);
      sym.swap(merged);
    }
  }

  void bpe_word(const std::string& word, std::vector<int>& out) const {
    // symbols = byte-level characters of the word
    std::vector<std::string> sym;
    for (unsigned char b : word) sym.push_back(byte_to_str_[b]);
    merge_symbols(sym);
    for (const auto& s : sym) {
      auto it = token_id_.find(s);
      if (it != token_id_.end()) {
        out.push_back(it->second);
      } else {
        // unknown symbol: fall back to its single-byte tokens
        std::vector<uint32_t> cps;
        std::vector<size_t> offs;
        uni::utf8_decode(s, cps, offs);
        for (uint32_t cp : cps) {
          std::string one;
          uni::utf8_append(one, cp);
          auto jt = token_id_.find(one);
          if (jt != token_id_.end()) out.push_back(jt->second);
        }
      }
    }
  }

  // Gemma 4: spaces -> U+2581; split into runs of non-newlines and of newlines (a newline run that is a token is
  // emitted as is); BPE over UTF-8 characters; symbols missing from the vocabulary fall back to <0xXX> bytes.
  void encode_spm(const std::string& text, std::vector<int>& out) const {
    std::string t;
    for (char c : text) {
      if (c == ' ') t += kSpace;
      else t.push_back(c);
    }
    for (size_t i = 0; i < t.size();) {
      const bool nl = t[i] == '\n';
      size_t j = i;
      while (j < t.size() && (t[j] == '\n') == nl) ++j;
      const std::string chunk = t.substr(i, j - i);
      i = j;
      if (nl) {
        auto it = token_id_.find(chunk);
        if (it != token_id_.end()) { out.push_back(it->second); continue; }
      }
      std::vector<uint32_t> cps;
      std::vector<size_t> offs;
      uni::utf8_decode(chunk, cps, offs);
      std::vector<std::string> sym;
      for (size_t k = 0; k < cps.size(); ++k) sym.push_back(chunk.substr(offs[k], offs[k + 1] - offs[k]));
      merge_symbols(sym);
      for (const auto& sy : sym) {
        auto it = token_id_.find(sy);
        if (it != token_id_.end()) { out.push_back(it->second); continue; }
        for (unsigned char b : sy) {
          char hex[8];
          snprintf(hex, sizeof(hex), "<0x%02X>", b);
          auto jt = token_id_.find(hex);
          if (jt != token_id_.end()) out.push_back(jt->second);
        }
      }
    }
  }

  void encode_plain(const std::string& text, std::vector<int>& out) const {
    if (text.empty()) return;
    if (spm_) return encode_spm(text, out);
    std::vector<uint32_t> cps;
    std::vector<size_t> offs;
    uni::utf8_decode(text, cps, offs);
    std::vector<std::pair<size_t, size_t>> words;
    pretokenize(cps, words);
    for (auto& [a, b] : words) bpe_word(text.substr(offs[a], offs[b] - offs[a]), out);
  }
};

using Vocab = Tokenizer;
