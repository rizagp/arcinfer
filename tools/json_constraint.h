// Constrained JSON output (response_format json_object / json_schema): a prefix checker that tells whether a text is
// the beginning of a JSON value matching a schema. The sampler only draws tokens whose text keeps the answer such a
// prefix, and ends the answer once the value is complete, so the result always parses and fits the schema.
//
// Schema subset: type (one or a list), properties / required / additionalProperties, items / minItems / maxItems,
// enum, const, anyOf / oneOf (first branch that fits), $ref to #/$defs/... or #/definitions/..., {} (any value).
// With "properties" and no additionalProperties, other keys are not allowed (as OpenAI's strict mode requires).
#pragma once
#include "../src/json.h"
#include <algorithm>
#include <cctype>
#include <cstring>
#include <string>
#include <vector>

namespace jcon {

enum Res { FAIL, PARTIAL, DONE };

// whether an object with these properties (any_key: other keys allowed too) can take a key besides the `seen` ones
inline bool more_keys(const Json& props, bool any_key, const std::vector<std::string>& seen) {
  if (any_key) return true;
  for (const auto& [name, v] : props.entries()) {
    (void)v;
    if (std::find(seen.begin(), seen.end(), name) == seen.end()) return true;
  }
  return false;
}

class Checker {
 public:
  // schema: the JSON schema ({} = any value); object_only: the value must be an object (json_object mode)
  Checker(Json schema, bool object_only) : root_(std::move(schema)), object_only_(object_only) {}

  // How `s` (the answer so far) relates to the schema: FAIL (no continuation fits), PARTIAL (it can still grow
  // into a match), DONE (a complete match, with at most trailing whitespace).
  Res check(const std::string& s) const {
    size_t i = 0;
    ws(s, i);
    if (i == s.size()) return PARTIAL;
    if (object_only_ && s[i] != '{') return FAIL;
    const Res r = value(root_, s, i, 0);
    if (r != DONE) return r;
    size_t j = i;
    ws(s, j);
    return j == s.size() && j - i <= 2 ? DONE : FAIL;  // (at most a little whitespace after the value)
  }

 private:
  Json root_;
  bool object_only_;

  const Json& resolve(const Json& sc) const {
    const std::string ref = sc["$ref"].str();
    if (ref.empty()) return sc;
    for (const char* pre : {"#/$defs/", "#/definitions/"}) {
      const std::string p = pre;
      if (ref.compare(0, p.size(), p) == 0) {
        const Json& defs = root_[p.substr(2, p.size() - 3)];
        return defs[ref.substr(p.size())];
      }
    }
    return root_;  // ("#": the root)
  }
  // whitespace: at most 8 characters in a row (a model could otherwise pad forever)
  static bool ws(const std::string& s, size_t& i) {
    size_t n = 0;
    while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) { ++i; ++n; }
    return n <= 8;
  }
  static bool has_type(const Json& sc, const char* t) {
    const Json& ty = sc["type"];
    if (ty.is_string()) return ty.str() == t || (std::string(t) == "number" && ty.str() == "integer");
    if (ty.is_array()) {
      for (const Json& x : ty.items())
        if (x.str() == t || (std::string(t) == "number" && x.str() == "integer")) return true;
      return false;
    }
    return true;  // (no type: anything)
  }
  static bool integer_only(const Json& sc) {
    const Json& ty = sc["type"];
    if (ty.is_string()) return ty.str() == "integer";
    if (ty.is_array()) {
      bool num = false, integer = false;
      for (const Json& x : ty.items()) { num = num || x.str() == "number"; integer = integer || x.str() == "integer"; }
      return integer && !num;
    }
    return false;
  }

  Res value(const Json& sc0, const std::string& s, size_t& i, int depth) const {
    if (depth > 32) return FAIL;
    const Json& sc = resolve(sc0);
    // anyOf / oneOf: the first branch that is not FAIL from here
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
    const char c = s[i];
    // enum / const: the value's text must be (a prefix of) one of the allowed values' JSON text
    if (sc["enum"].is_array() || sc.has("const")) {
      std::vector<std::string> opts;
      if (sc.has("const")) opts.push_back(sc["const"].dump());
      for (const Json& e : sc["enum"].items()) opts.push_back(e.dump());
      const std::string rest = s.substr(i);
      bool partial = false;
      for (const std::string& o : opts) {
        if (rest.compare(0, o.size(), o) == 0) { i += o.size(); return DONE; }
        if (o.compare(0, rest.size(), rest) == 0) partial = true;
      }
      if (partial) { i = s.size(); return PARTIAL; }
      return FAIL;
    }
    if (c == '{') return has_type(sc, "object") ? object(sc, s, i, depth) : FAIL;
    if (c == '[') return has_type(sc, "array") ? array(sc, s, i, depth) : FAIL;
    if (c == '"') return has_type(sc, "string") ? string(s, i, nullptr) : FAIL;
    if (c == '-' || (c >= '0' && c <= '9')) return has_type(sc, "number") ? number(s, i, integer_only(sc)) : FAIL;
    if (c == 't') return has_type(sc, "boolean") ? literal(s, i, "true") : FAIL;
    if (c == 'f') return has_type(sc, "boolean") ? literal(s, i, "false") : FAIL;
    if (c == 'n') return has_type(sc, "null") ? literal(s, i, "null") : FAIL;
    return FAIL;
  }

  static Res literal(const std::string& s, size_t& i, const char* lit) {
    const std::string l = lit;
    for (size_t k = 0; k < l.size(); ++k, ++i) {
      if (i >= s.size()) return PARTIAL;
      if (s[i] != l[k]) return FAIL;
    }
    return DONE;
  }
  static Res number(const std::string& s, size_t& i, bool integer) {
    if (i < s.size() && s[i] == '-') ++i;
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
    if (s[i] == 'e' || s[i] == 'E') {
      if (integer) return FAIL;
      ++i;
      if (i < s.size() && (s[i] == '+' || s[i] == '-')) ++i;
      if (i >= s.size()) return PARTIAL;
      if (!isdigit((unsigned char)s[i])) return FAIL;
      while (i < s.size() && isdigit((unsigned char)s[i])) ++i;
      if (i >= s.size()) return PARTIAL;
    }
    return DONE;
  }
  // a string; its decoded text goes to `out` when given
  static Res string(const std::string& s, size_t& i, std::string* out) {
    if (s[i] != '"') return FAIL;
    ++i;
    while (i < s.size()) {
      const unsigned char c = (unsigned char)s[i];
      if (c == '"') { ++i; return DONE; }
      if (c < 0x20) return FAIL;
      if (c == '\\') {
        if (i + 1 >= s.size()) { i = s.size(); return PARTIAL; }
        const char e = s[i + 1];
        if (e == 'u') {
          for (size_t k = 2; k < 6; ++k) {
            if (i + k >= s.size()) { i = s.size(); return PARTIAL; }
            if (!isxdigit((unsigned char)s[i + k])) return FAIL;
          }
          if (out) *out += '?';
          i += 6;
          continue;
        }
        if (!strchr("\"\\/bfnrt", e)) return FAIL;
        if (out) *out += e == 'n' ? '\n' : e == 't' ? '\t' : e;
        i += 2;
        continue;
      }
      if (out) *out += (char)c;
      ++i;
    }
    return PARTIAL;
  }
  Res object(const Json& sc, const std::string& s, size_t& i, int depth) const {
    ++i;  // '{'
    const Json& props = sc["properties"];
    const Json& addl = sc["additionalProperties"];
    // other keys: allowed with additionalProperties true or a schema, or when no properties are listed
    const bool any_key = addl.is_object() || addl.as_bool(false) || (!props.is_object() && !addl.is_bool());
    std::vector<std::string> seen;
    bool first = true;
    while (true) {
      if (!ws(s, i)) return FAIL;
      if (i >= s.size()) return PARTIAL;
      if (s[i] == '}') {
        for (const Json& r : sc["required"].items())
          if (std::find(seen.begin(), seen.end(), r.str()) == seen.end()) return FAIL;
        ++i;
        return DONE;
      }
      if (!first) {
        if (s[i] != ',') return FAIL;
        if (!more_keys(props, any_key, seen)) return FAIL;  // (a comma only where another key can follow)
        ++i;
        if (!ws(s, i)) return FAIL;
        if (i >= s.size()) return PARTIAL;
      }
      first = false;
      if (s[i] != '"') return FAIL;
      std::string key;
      size_t k0 = i;
      const Res kr = string(s, i, &key);
      if (kr == FAIL) return FAIL;
      // the key: one of the listed properties not used yet (a prefix of one while it is still being written)
      auto allowed = [&](const std::string& k, bool prefix) {
        if (std::find(seen.begin(), seen.end(), k) != seen.end() && !prefix) return false;
        if (any_key) return true;
        for (const auto& [name, v] : props.entries()) {
          (void)v;
          if (std::find(seen.begin(), seen.end(), name) != seen.end()) continue;
          if (prefix ? name.compare(0, k.size(), k) == 0 : name == k) return true;
        }
        return false;
      };
      if (kr == PARTIAL) return allowed(key, true) ? PARTIAL : FAIL;
      (void)k0;
      if (!allowed(key, false)) return FAIL;
      seen.push_back(key);
      if (!ws(s, i)) return FAIL;
      if (i >= s.size()) return PARTIAL;
      if (s[i] != ':') return FAIL;
      ++i;
      if (!ws(s, i)) return FAIL;
      if (i >= s.size()) return PARTIAL;
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
    int64_t n = 0;
    while (true) {
      if (!ws(s, i)) return FAIL;
      if (i >= s.size()) return PARTIAL;
      if (s[i] == ']') {
        if (n < mn) return FAIL;
        ++i;
        return DONE;
      }
      if (n > 0) {
        if (s[i] != ',') return FAIL;
        ++i;
        if (!ws(s, i)) return FAIL;
        if (i >= s.size()) return PARTIAL;
      }
      if (n >= mx) return FAIL;
      const Res r = value(items, s, i, depth + 1);
      if (r != DONE) return r;
      ++n;
    }
  }
};

}  // namespace jcon
