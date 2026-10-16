// Constrained tool calls: while the model writes a tool call, only tokens that keep the call well formed and fitting
// the tool's parameter schema are drawn. A long call can no longer lose a required argument (Pi's edit without its
// path), nest an argument in the wrong place, write code as a key, or end before its arguments are closed. What the
// arguments say is still the model's choice: only the tokens that would break the call's form are left out, so a call
// that was already well formed is sampled exactly as without the constraint.
//
// Gemma 4: after <|tool_call>: call:NAME{key:value,...}<tool_call|>; keys are bare words, strings are delimited by
// <|"|>, other values are numbers, true / false / null, [...] and {...}. Schema subset as jcon (type, properties,
// required, additionalProperties, items, minItems / maxItems, enum, const, anyOf / oneOf, $ref, {} = any value).
// Qwen 3.x: after <tool_call>: <function=NAME> then <parameter=KEY> VALUE </parameter> blocks, </function>
// </tool_call>; a string value is the raw text, other types are JSON (checked with jcon against the schema).
#pragma once
#include "../src/json.h"
#include "json_constraint.h"
#include <cstring>
#include <string>
#include <vector>

namespace tcon {

using jcon::Res;
using jcon::FAIL;
using jcon::PARTIAL;
using jcon::DONE;

// A tool-call format: its open / close markers and the check of a call's text (the text after the open marker).
class ToolCon {
 public:
  virtual ~ToolCon() = default;
  virtual const char* open() const = 0;
  virtual const char* close() const = 0;
  // a control token's text allowed inside a call (the call syntax uses some, e.g. Gemma's <|"|>)
  virtual bool special_ok(const std::string& piece) const { return piece == close(); }
  // FAIL, PARTIAL (it can still become a valid call) or DONE (the call is complete, the close marker included)
  virtual Res check(const std::string& s) const = 0;
  bool empty() const { return tools_.empty(); }

  // The text after the last open marker when that call is still open, else npos.
  size_t open_call(const std::string& text) const {
    const size_t o = text.rfind(open());
    if (o == std::string::npos || text.find(close(), o) != std::string::npos) return std::string::npos;
    return o + strlen(open());
  }

 protected:
  // tools: the request's "tools" (OpenAI form: [{type: function, function: {name, parameters}}])
  explicit ToolCon(const Json& tools) {
    for (const Json& t : tools.items()) {
      const Json& f = t.has("function") ? t["function"] : t;
      if (!f["name"].str().empty()) tools_.push_back({f["name"].str(), f["parameters"].is_object() ? f["parameters"] : Json::object()});
    }
  }
  // the parameters of tool `name` (nullptr if none); *prefix: some tool's name starts with `name`
  const Json* find(const std::string& name, bool* prefix) const {
    const Json* params = nullptr;
    *prefix = false;
    for (const auto& [n, p] : tools_) {
      if (n == name) params = &p;
      if (n.compare(0, name.size(), name) == 0) *prefix = true;
    }
    return params;
  }
  std::vector<std::pair<std::string, Json>> tools_;
};

class GemmaCall : public ToolCon {
 public:
  static constexpr const char* kOpen = "<|tool_call>";
  static constexpr const char* kClose = "<tool_call|>";
  static constexpr const char* kQ = "<|\"|>";

  explicit GemmaCall(const Json& tools) : ToolCon(tools) {}
  const char* open() const override { return kOpen; }
  const char* close() const override { return kClose; }
  bool special_ok(const std::string& piece) const override { return piece == kQ || piece == kClose; }

  // How `s` (the text after <|tool_call>) relates to a valid call: FAIL, PARTIAL (it can still become one) or
  // DONE (the call is complete, <tool_call|> included).
  Res check(const std::string& s) const override {
    size_t i = 0;
    Res r = lit(s, i, "call:");
    if (r != DONE) return r;
    const size_t n0 = i;
    while (i < s.size() && s[i] != '{') ++i;
    const std::string name = s.substr(n0, i - n0);
    bool prefix = false;
    const Json* params = find(name, &prefix);
    if (i >= s.size()) return prefix ? PARTIAL : FAIL;
    if (!params) return FAIL;
    root_ = params;
    r = value(*params, s, i, 0, true);
    if (r != DONE) return r;
    r = lit(s, i, kClose);  // then the call ends
    return r == DONE && i != s.size() ? FAIL : r;
  }

 private:
  mutable const Json* root_ = nullptr;

  // the literal t from i: DONE when all of it is there, PARTIAL when s ends inside it
  static Res lit(const std::string& s, size_t& i, const char* t) {
    const size_t n = strlen(t);
    for (size_t k = 0; k < n; ++k, ++i) {
      if (i >= s.size()) return PARTIAL;
      if (s[i] != t[k]) return FAIL;
    }
    return DONE;
  }
  const Json& resolve(const Json& sc) const {
    const std::string ref = sc["$ref"].str();
    if (ref.empty() || !root_) return sc;
    for (const char* pre : {"#/$defs/", "#/definitions/"}) {
      const std::string p = pre;
      if (ref.compare(0, p.size(), p) == 0) return (*root_)[p.substr(2, p.size() - 3)][ref.substr(p.size())];
    }
    return *root_;
  }
  static bool has_type(const Json& sc, const char* t) {
    const Json& ty = sc["type"];
    auto is = [&](const std::string& x) { return x == t || (std::string(t) == "number" && x == "integer"); };
    if (ty.is_string()) return is(ty.str());
    if (ty.is_array()) {
      for (const Json& x : ty.items())
        if (is(x.str())) return true;
      return false;
    }
    return true;
  }

  Res value(const Json& sc0, const std::string& s, size_t& i, int depth, bool top = false) const {
    if (depth > 32) return FAIL;
    const Json& sc = resolve(sc0);
    for (const char* k : {"anyOf", "oneOf"}) {
      if (!sc[k].is_array()) continue;
      Res best = FAIL;
      size_t bi = i;
      for (const Json& br : sc[k].items()) {
        size_t j = i;
        const Res r = value(br, s, j, depth + 1);
        if (r == DONE) { i = j; return DONE; }
        if (r == PARTIAL && best == FAIL) { best = PARTIAL; bi = j; }
      }
      i = bi;
      return best;
    }
    if (i >= s.size()) return PARTIAL;
    if (top) return s[i] == '{' ? object(sc, s, i, depth) : FAIL;  // (the arguments: always an object)
    // enum / const: one of the allowed values, written as the call syntax writes it
    if (sc["enum"].is_array() || sc.has("const")) {
      std::vector<std::string> opts;
      if (sc.has("const")) opts.push_back(text_of(sc["const"]));
      for (const Json& e : sc["enum"].items()) opts.push_back(text_of(e));
      const std::string rest = s.substr(i);
      bool partial = false;
      for (const std::string& o : opts) {
        if (rest.compare(0, o.size(), o) == 0) { i += o.size(); return DONE; }
        if (o.compare(0, rest.size(), rest) == 0) partial = true;
      }
      if (partial) { i = s.size(); return PARTIAL; }
      return FAIL;
    }
    const char c = s[i];
    if (c == '<') return has_type(sc, "string") ? string(s, i) : FAIL;
    if (c == '{') return has_type(sc, "object") ? object(sc, s, i, depth) : FAIL;
    if (c == '[') return has_type(sc, "array") ? array(sc, s, i, depth) : FAIL;
    if (c == '-' || (c >= '0' && c <= '9')) return has_type(sc, "number") ? number(s, i, int_only(sc)) : FAIL;
    if (c == 't') return has_type(sc, "boolean") ? lit(s, i, "true") : FAIL;
    if (c == 'f') return has_type(sc, "boolean") ? lit(s, i, "false") : FAIL;
    if (c == 'n') return has_type(sc, "null") ? lit(s, i, "null") : FAIL;
    return FAIL;
  }
  static std::string text_of(const Json& v) {
    if (v.is_string()) return std::string(kQ) + v.str() + kQ;
    return v.dump();
  }
  static bool int_only(const Json& sc) {
    const Json& ty = sc["type"];
    if (ty.is_string()) return ty.str() == "integer";
    if (!ty.is_array()) return false;
    bool num = false, in = false;
    for (const Json& x : ty.items()) { num = num || x.str() == "number"; in = in || x.str() == "integer"; }
    return in && !num;
  }
  // <|"|>text<|"|>: any text up to the closing delimiter
  Res string(const std::string& s, size_t& i) const {
    const Res r = lit(s, i, kQ);
    if (r != DONE) return r;
    const size_t e = s.find(kQ, i);
    if (e == std::string::npos) { i = s.size(); return PARTIAL; }
    i = e + strlen(kQ);
    return DONE;
  }
  static Res number(const std::string& s, size_t& i, bool integer) {
    if (s[i] == '-') ++i;
    if (i >= s.size()) return PARTIAL;
    if (s[i] == '0') ++i;
    else if (s[i] >= '1' && s[i] <= '9') { while (i < s.size() && isdigit((unsigned char)s[i])) ++i; }
    else return FAIL;
    if (i >= s.size()) return PARTIAL;
    if (s[i] == '.') {
      if (integer) return FAIL;
      ++i;
      if (i >= s.size()) return PARTIAL;
      if (!isdigit((unsigned char)s[i])) return FAIL;
      while (i < s.size() && isdigit((unsigned char)s[i])) ++i;
      if (i >= s.size()) return PARTIAL;
    }
    return DONE;  // (a number ends where ',', '}' or ']' follows: checked by the container)
  }
  Res object(const Json& sc, const std::string& s, size_t& i, int depth) const {
    ++i;  // '{'
    const Json& props = sc["properties"];
    const Json& addl = sc["additionalProperties"];
    const bool any_key = addl.is_object() || addl.as_bool(false) || (!props.is_object() && !addl.is_bool());
    std::vector<std::string> seen;
    auto required_seen = [&]() {
      for (const Json& r : sc["required"].items())
        if (std::find(seen.begin(), seen.end(), r.str()) == seen.end()) return false;
      return true;
    };
    for (bool first = true;; first = false) {
      if (i >= s.size()) return PARTIAL;
      if (s[i] == '}') {
        if (!required_seen()) return FAIL;
        ++i;
        return DONE;
      }
      if (!first) {
        if (s[i] != ',') return FAIL;
        if (!jcon::more_keys(props, any_key, seen)) return FAIL;  // (a comma only where another key can follow)
        if (++i >= s.size()) return PARTIAL;
      }
      // the key: a bare word, one of the listed properties not used yet (a prefix of one while being written)
      const size_t k0 = i;
      while (i < s.size() && (isalnum((unsigned char)s[i]) || s[i] == '_' || s[i] == '-' || s[i] == '.' || s[i] == '$')) ++i;
      const std::string key = s.substr(k0, i - k0);
      auto allowed = [&](bool prefix) {
        if (key.empty()) return prefix;
        if (std::find(seen.begin(), seen.end(), key) != seen.end() && !prefix) return false;
        if (any_key) return true;
        for (const auto& [name, v] : props.entries()) {
          (void)v;
          if (std::find(seen.begin(), seen.end(), name) != seen.end()) continue;
          if (prefix ? name.compare(0, key.size(), key) == 0 : name == key) return true;
        }
        return false;
      };
      if (i >= s.size()) return allowed(true) ? PARTIAL : FAIL;
      if (s[i] != ':' || !allowed(false)) return FAIL;
      seen.push_back(key);
      if (++i >= s.size()) return PARTIAL;
      static const Json any = Json::object();
      const Json& vs = props[key].is_null() ? (addl.is_object() ? addl : any) : props[key];
      const Res vr = value(vs, s, i, depth + 1);
      if (vr != DONE) return vr;
    }
  }
  Res array(const Json& sc, const std::string& s, size_t& i, int depth) const {
    ++i;  // '['
    static const Json any = Json::object();
    const Json& items = sc["items"].is_object() ? sc["items"] : any;
    const int64_t mn = sc["minItems"].as_int(0), mx = sc["maxItems"].as_int(1 << 30);
    for (int64_t n = 0;; ++n) {
      if (i >= s.size()) return PARTIAL;
      if (s[i] == ']') {
        if (n < mn) return FAIL;
        ++i;
        return DONE;
      }
      if (n > 0) {
        if (s[i] != ',') return FAIL;
        if (++i >= s.size()) return PARTIAL;
      }
      if (n >= mx) return FAIL;
      const Res r = value(items, s, i, depth + 1);
      if (r != DONE) return r;
    }
  }
};

// Qwen 3.x (chat.h render / parse_tool_calls): <tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n
// ... </function>\n</tool_call>. Between the parts up to two whitespace characters; a string value is any text without
// "</parameter>" or "<parameter=" (the parser would end it there); other types are JSON values of their schema.
class QwenCall : public ToolCon {
 public:
  explicit QwenCall(const Json& tools) : ToolCon(tools) {}
  const char* open() const override { return "<tool_call>"; }
  const char* close() const override { return "</tool_call>"; }

  Res check(const std::string& s) const override {
    size_t i = 0;
    Res r;
    if ((r = ws_lit(s, i, "<function=")) != DONE) return r;
    const size_t n0 = i;
    while (i < s.size() && s[i] != '>' && s[i] != '\n') ++i;
    const std::string name = s.substr(n0, i - n0);
    bool prefix = false;
    const Json* params = find(name, &prefix);
    if (i >= s.size()) return prefix ? PARTIAL : FAIL;
    if (s[i] != '>' || !params) return FAIL;
    ++i;
    const Json& props = (*params)["properties"];
    const Json& addl = (*params)["additionalProperties"];
    const bool any_key = addl.is_object() || addl.as_bool(false) || (!props.is_object() && !addl.is_bool());
    std::vector<std::string> seen;
    for (;;) {
      // the next part: a parameter, or the end of the function (only once every required parameter is there)
      size_t j = i;
      skip_ws(s, j);
      if (j >= s.size()) return PARTIAL;
      static const std::string kp = "<parameter=", kf = "</function>";
      const std::string rest = s.substr(j);
      auto starts = [](const std::string& a, const std::string& b) { return a.compare(0, b.size(), b) == 0; };
      const bool full_p = starts(rest, kp), pre_p = rest.size() < kp.size() && starts(kp, rest);
      const bool full_f = starts(rest, kf), pre_f = rest.size() < kf.size() && starts(kf, rest);
      if (!full_p && pre_p) return PARTIAL;  // ("<": a parameter or the end, "<p...": a parameter)
      if (!full_p && !full_f && !pre_f) return FAIL;
      if (!full_p) {  // the end of the function: only with every required parameter
        for (const Json& q : (*params)["required"].items())
          if (std::find(seen.begin(), seen.end(), q.str()) == seen.end()) return FAIL;
        i = j;
        if ((r = lit(s, i, "</function>")) != DONE) return r;
        if ((r = ws_lit(s, i, "</tool_call>")) != DONE) return r;
        return i == s.size() ? DONE : FAIL;
      }
      i = j;
      if ((r = lit(s, i, "<parameter=")) != DONE) return r;
      const size_t k0 = i;
      while (i < s.size() && s[i] != '>' && s[i] != '\n') ++i;
      const std::string key = s.substr(k0, i - k0);
      auto key_ok = [&](bool pre) {
        if (std::find(seen.begin(), seen.end(), key) != seen.end() && !pre) return false;
        if (any_key) return !key.empty() || pre;
        for (const auto& [name2, v] : props.entries()) {
          (void)v;
          if (std::find(seen.begin(), seen.end(), name2) != seen.end()) continue;
          if (pre ? name2.compare(0, key.size(), key) == 0 : name2 == key) return true;
        }
        return false;
      };
      if (i >= s.size()) return key_ok(true) ? PARTIAL : FAIL;
      if (s[i] != '>' || !key_ok(false)) return FAIL;
      seen.push_back(key);
      ++i;
      if (i < s.size() && s[i] == '\n') ++i;
      // the value, up to </parameter>
      static const std::string kend = "</parameter>";
      const size_t e = s.find(kend, i);
      const size_t bad = s.find("<parameter=", i);
      if (bad != std::string::npos && (e == std::string::npos || bad < e)) return FAIL;
      static const Json any = Json::object();
      const Json& vs = props[key].is_null() ? (addl.is_object() ? addl : any) : props[key];
      const bool is_str = vs["type"].is_string() && vs["type"].str() == "string";
      if (e == std::string::npos) {
        if (is_str || !vs.has("type")) return PARTIAL;
        // a JSON value so far, possibly followed by the start of "\n</parameter>"
        const std::string v = s.substr(i);
        for (size_t cut = v.size() + 1; cut-- > 0 && v.size() - cut <= kend.size() + 1;) {
          const std::string tail = v.substr(cut);
          const std::string closing = "\n" + kend;
          const bool tail_ok = closing.compare(0, tail.size(), tail) == 0 || kend.compare(0, tail.size(), tail) == 0;
          if (tail_ok && jcon::Checker(vs, false).check(v.substr(0, cut)) != FAIL) return PARTIAL;
        }
        return FAIL;
      }
      std::string v = s.substr(i, e - i);
      if (!v.empty() && v.back() == '\n') v.pop_back();
      if (!is_str && vs.has("type") && jcon::Checker(vs, false).check(v) != DONE) return FAIL;
      i = e + kend.size();
    }
  }

 private:
  static void skip_ws(const std::string& s, size_t& i) {
    for (int n = 0; n < 2 && i < s.size() && (s[i] == '\n' || s[i] == ' '); ++n) ++i;
  }
  static Res lit(const std::string& s, size_t& i, const char* t) {
    for (const char* p = t; *p; ++p, ++i) {
      if (i >= s.size()) return PARTIAL;
      if (s[i] != *p) return FAIL;
    }
    return DONE;
  }
  static Res ws_lit(const std::string& s, size_t& i, const char* t) {
    skip_ws(s, i);
    if (i >= s.size()) return PARTIAL;
    return lit(s, i, t);
  }
};

}  // namespace tcon
