// Minimal JSON value with parser and serializer (objects keep insertion order).
#pragma once
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

class Json {
 public:
  enum Type { Null, Bool, Number, String, Array, Object };

  Json() = default;
  Json(std::nullptr_t) {}
  Json(bool b) : t_(Bool), b_(b) {}
  Json(int v) : t_(Number), n_(v) {}
  Json(int64_t v) : t_(Number), n_((double)v) {}
  Json(size_t v) : t_(Number), n_((double)v) {}
  Json(double v) : t_(Number), n_(v) {}
  Json(const char* s) : t_(String), s_(s) {}
  Json(std::string s) : t_(String), s_(std::move(s)) {}

  static Json array() { Json j; j.t_ = Array; return j; }
  static Json object() { Json j; j.t_ = Object; return j; }

  Type type() const { return t_; }
  bool is_null() const { return t_ == Null; }
  bool is_bool() const { return t_ == Bool; }
  bool is_number() const { return t_ == Number; }
  bool is_string() const { return t_ == String; }
  bool is_array() const { return t_ == Array; }
  bool is_object() const { return t_ == Object; }

  bool as_bool(bool def = false) const { return t_ == Bool ? b_ : def; }
  double as_number(double def = 0) const { return t_ == Number ? n_ : def; }
  int64_t as_int(int64_t def = 0) const {  // (out of range / not finite: def, never an undefined conversion)
    return t_ == Number && std::isfinite(n_) && std::fabs(n_) < 9.2e18 ? (int64_t)n_ : def;
  }
  const std::string& as_string() const { static const std::string empty; return t_ == String ? s_ : empty; }
  std::string str(const std::string& def = "") const { return t_ == String ? s_ : def; }

  // arrays
  size_t size() const { return t_ == Array ? a_.size() : t_ == Object ? o_.size() : 0; }
  const Json& operator[](size_t i) const { return i < a_.size() ? a_[i] : null_ref(); }
  const Json& operator[](int i) const { return i >= 0 ? (*this)[(size_t)i] : null_ref(); }
  Json& push(Json v) { if (t_ != Array) { t_ = Array; a_.clear(); } a_.push_back(std::move(v)); return a_.back(); }
  const std::vector<Json>& items() const { return a_; }

  // objects
  bool has(const std::string& k) const {
    for (auto& [kk, v] : o_) if (kk == k) return true;
    return false;
  }
  const Json& operator[](const std::string& k) const {
    for (auto& [kk, v] : o_) if (kk == k) return v;
    return null_ref();
  }
  const Json& operator[](const char* k) const { return (*this)[std::string(k)]; }
  Json& set(const std::string& k, Json v) {
    if (t_ != Object) { t_ = Object; o_.clear(); }
    for (auto& [kk, vv] : o_) if (kk == k) { vv = std::move(v); return vv; }
    o_.push_back({k, std::move(v)});
    return o_.back().second;
  }
  const std::vector<std::pair<std::string, Json>>& entries() const { return o_; }

  // serialization (pretty = Python json.dumps separators ", " / ": ", as Jinja's tojson does)
  std::string dump(bool spaced = false) const {
    std::string out;
    write(out, spaced);
    return out;
  }

  static Json parse(const std::string& s) {
    size_t i = 0;
    Json v = parse_value(s, i);
    skip_ws(s, i);
    if (i != s.size()) throw std::runtime_error("json: trailing characters");
    return v;
  }
  static bool try_parse(const std::string& s, Json& out) {
    try { out = parse(s); return true; } catch (...) { return false; }
  }

  static void escape(std::string& out, const std::string& s) {
    out.push_back('"');
    for (unsigned char c : s) {
      switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        case '\b': out += "\\b"; break;
        case '\f': out += "\\f"; break;
        default:
          if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); out += b; }
          else out.push_back((char)c);
      }
    }
    out.push_back('"');
  }

 private:
  Type t_ = Null;
  bool b_ = false;
  double n_ = 0;
  std::string s_;
  std::vector<Json> a_;
  std::vector<std::pair<std::string, Json>> o_;

  static const Json& null_ref() { static const Json n; return n; }

  void write(std::string& out, bool sp) const {
    switch (t_) {
      case Null: out += "null"; break;
      case Bool: out += b_ ? "true" : "false"; break;
      case Number: {
        char buf[32];
        if (!std::isfinite(n_)) snprintf(buf, sizeof buf, "null");  // (JSON has no nan / inf)
        else if (n_ == std::floor(n_) && std::fabs(n_) < 1e15) snprintf(buf, sizeof buf, "%lld", (long long)n_);
        else snprintf(buf, sizeof buf, "%.17g", n_);
        out += buf;
        break;
      }
      case String: escape(out, s_); break;
      case Array:
        out.push_back('[');
        for (size_t i = 0; i < a_.size(); ++i) {
          if (i) out += sp ? ", " : ",";
          a_[i].write(out, sp);
        }
        out.push_back(']');
        break;
      case Object:
        out.push_back('{');
        for (size_t i = 0; i < o_.size(); ++i) {
          if (i) out += sp ? ", " : ",";
          escape(out, o_[i].first);
          out += sp ? ": " : ":";
          o_[i].second.write(out, sp);
        }
        out.push_back('}');
        break;
    }
  }

  static void skip_ws(const std::string& s, size_t& i) {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) ++i;
  }
  static void put_utf8(std::string& o, uint32_t cp) {
    if (cp < 0x80) o.push_back((char)cp);
    else if (cp < 0x800) { o.push_back((char)(0xC0 | (cp >> 6))); o.push_back((char)(0x80 | (cp & 0x3F))); }
    else if (cp < 0x10000) { o.push_back((char)(0xE0 | (cp >> 12))); o.push_back((char)(0x80 | ((cp >> 6) & 0x3F))); o.push_back((char)(0x80 | (cp & 0x3F))); }
    else { o.push_back((char)(0xF0 | (cp >> 18))); o.push_back((char)(0x80 | ((cp >> 12) & 0x3F))); o.push_back((char)(0x80 | ((cp >> 6) & 0x3F))); o.push_back((char)(0x80 | (cp & 0x3F))); }
  }
  static std::string parse_string(const std::string& s, size_t& i) {
    if (s[i] != '"') throw std::runtime_error("json: expected string");
    ++i;
    std::string o;
    while (i < s.size() && s[i] != '"') {
      char c = s[i++];
      if (c != '\\') { o.push_back(c); continue; }
      if (i >= s.size()) break;
      char e = s[i++];
      switch (e) {
        case 'n': o.push_back('\n'); break;
        case 't': o.push_back('\t'); break;
        case 'r': o.push_back('\r'); break;
        case 'b': o.push_back('\b'); break;
        case 'f': o.push_back('\f'); break;
        case 'u': {
          auto hex4 = [&](size_t p) {  // (exactly four hex digits, or the string is invalid)
            if (p + 4 > s.size()) throw std::runtime_error("json: bad \\u escape");
            uint32_t v = 0;
            for (size_t q = p; q < p + 4; ++q) {
              const char h = s[q];
              const int d = h >= '0' && h <= '9' ? h - '0' : h >= 'a' && h <= 'f' ? h - 'a' + 10 : h >= 'A' && h <= 'F' ? h - 'A' + 10 : -1;
              if (d < 0) throw std::runtime_error("json: bad \\u escape");
              v = v * 16 + (uint32_t)d;
            }
            return v;
          };
          uint32_t cp = hex4(i);
          i += 4;
          if (cp >= 0xD800 && cp < 0xDC00 && i + 6 <= s.size() && s[i] == '\\' && s[i + 1] == 'u') {
            const uint32_t lo = hex4(i + 2);
            if (lo >= 0xDC00 && lo < 0xE000) { cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00); i += 6; }
          }
          put_utf8(o, cp);
          break;
        }
        default: o.push_back(e);
      }
    }
    if (i >= s.size()) throw std::runtime_error("json: unterminated string");
    ++i;
    return o;
  }
  // (nesting up to kMaxDepth: a body of 100,000 '[' overflowed the stack of the connection's thread, which no
  //  exception handler catches)
  static constexpr int kMaxDepth = 256;
  static Json parse_value(const std::string& s, size_t& i, int depth = 0) {
    if (depth > kMaxDepth) throw std::runtime_error("json: nesting too deep");
    skip_ws(s, i);
    if (i >= s.size()) throw std::runtime_error("json: unexpected end");
    const char c = s[i];
    if (c == '{') {
      Json o = object();
      std::unordered_map<std::string, size_t> at;  // (key -> entry: a repeated key replaces it in linear time)
      ++i;
      skip_ws(s, i);
      if (i < s.size() && s[i] == '}') { ++i; return o; }
      while (true) {
        skip_ws(s, i);
        std::string k = parse_string(s, i);
        skip_ws(s, i);
        if (i >= s.size() || s[i] != ':') throw std::runtime_error("json: expected ':'");
        ++i;
        Json v = parse_value(s, i, depth + 1);  // (a repeated key: the last value wins, as in JavaScript / Python)
        const auto [it, fresh] = at.emplace(k, o.o_.size());
        if (fresh) o.o_.push_back({std::move(k), std::move(v)});
        else o.o_[it->second].second = std::move(v);
        skip_ws(s, i);
        if (i < s.size() && s[i] == ',') { ++i; continue; }
        if (i < s.size() && s[i] == '}') { ++i; return o; }
        throw std::runtime_error("json: expected ',' or '}'");
      }
    }
    if (c == '[') {
      Json a = array();
      ++i;
      skip_ws(s, i);
      if (i < s.size() && s[i] == ']') { ++i; return a; }
      while (true) {
        a.a_.push_back(parse_value(s, i, depth + 1));
        skip_ws(s, i);
        if (i < s.size() && s[i] == ',') { ++i; continue; }
        if (i < s.size() && s[i] == ']') { ++i; return a; }
        throw std::runtime_error("json: expected ',' or ']'");
      }
    }
    if (c == '"') return Json(parse_string(s, i));
    if (s.compare(i, 4, "true") == 0) { i += 4; return Json(true); }
    if (s.compare(i, 5, "false") == 0) { i += 5; return Json(false); }
    if (s.compare(i, 4, "null") == 0) { i += 4; return Json(); }
    size_t j = i;
    while (j < s.size() && (isdigit((unsigned char)s[j]) || s[j] == '-' || s[j] == '+' || s[j] == '.' || s[j] == 'e' || s[j] == 'E')) ++j;
    if (j == i) throw std::runtime_error("json: unexpected character");
    // (the whole run must be one number: "1.0.0" or "1e" are errors, not 1)
    const std::string t = s.substr(i, j - i);
    size_t used = 0;
    double v = 0;
    try { v = std::stod(t, &used); } catch (const std::out_of_range&) { used = 0; } catch (const std::invalid_argument&) { used = 0; }
    if (used != t.size() || !std::isfinite(v)) throw std::runtime_error("json: bad number");
    i = j;
    return Json(v);
  }
};
