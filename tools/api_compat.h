// Other API formats on top of the server's OpenAI chat handler: OpenAI completions (/v1/completions), Ollama
// (/api/chat, /api/generate), Anthropic Messages (/v1/messages) and OpenAI Responses (/v1/responses). Each request
// is translated into a /v1/chat/completions request; the handler's answer (a complete response, or its stream of
// chat.completion.chunk events) comes back through the HttpResponse taps and is translated into the client's format.
#pragma once
#include "../src/http.h"
#include "../src/json.h"
#include <chrono>
#include <cstdint>
#include <ctime>
#include <map>
#include <string>
#include <vector>

namespace apic {

inline std::string iso_now() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t t = std::chrono::system_clock::to_time_t(now);
  std::tm g{};
  gmtime_s(&g, &t);
  char b[40];
  strftime(b, sizeof b, "%Y-%m-%dT%H:%M:%S", &g);
  const long ms = (long)(std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000);
  char c[48];
  snprintf(c, sizeof c, "%s.%03ldZ", b, ms);
  return c;
}
inline long long unix_now() { return (long long)std::time(nullptr); }

inline std::string base64_encode(const std::string& in) {
  static const char* t = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  out.reserve((in.size() + 2) / 3 * 4);
  size_t i = 0;
  for (; i + 2 < in.size(); i += 3) {
    const uint32_t v = (uint8_t)in[i] << 16 | (uint8_t)in[i + 1] << 8 | (uint8_t)in[i + 2];
    out += t[v >> 18 & 63]; out += t[v >> 12 & 63]; out += t[v >> 6 & 63]; out += t[v & 63];
  }
  if (i < in.size()) {
    const uint32_t v = (uint8_t)in[i] << 16 | (i + 1 < in.size() ? (uint8_t)in[i + 1] << 8 : 0);
    out += t[v >> 18 & 63]; out += t[v >> 12 & 63];
    out += i + 1 < in.size() ? t[v >> 6 & 63] : '=';
    out += '=';
  }
  return out;
}
// multipart/form-data: field name -> (filename, content)
struct FormField { std::string filename, data; };
inline std::map<std::string, FormField> parse_multipart(const std::string& content_type, const std::string& body) {
  std::map<std::string, FormField> f;
  const size_t b = content_type.find("boundary=");
  if (b == std::string::npos) return f;
  std::string boundary = content_type.substr(b + 9);
  if (!boundary.empty() && boundary[0] == '"') boundary = boundary.substr(1, boundary.find('"', 1) - 1);
  const size_t semi = boundary.find(';');
  if (semi != std::string::npos) boundary.resize(semi);
  const std::string delim = "--" + boundary;
  size_t pos = body.find(delim);
  while (pos != std::string::npos) {
    pos += delim.size();
    if (body.compare(pos, 2, "--") == 0) break;
    if (body.compare(pos, 2, "\r\n") == 0) pos += 2;
    const size_t he = body.find("\r\n\r\n", pos);
    if (he == std::string::npos) break;
    const std::string head = body.substr(pos, he - pos);
    const size_t next = body.find("\r\n" + delim, he + 4);
    if (next == std::string::npos) break;
    FormField ff;
    ff.data = body.substr(he + 4, next - he - 4);
    std::string name;
    const size_t n = head.find("name=\"");
    if (n != std::string::npos) name = head.substr(n + 6, head.find('"', n + 6) - n - 6);
    const size_t fn = head.find("filename=\"");
    if (fn != std::string::npos) ff.filename = head.substr(fn + 10, head.find('"', fn + 10) - fn - 10);
    if (!name.empty()) f[name] = ff;
    pos = next + 2;
  }
  return f;
}

// an image given as base64 (Ollama "images", Anthropic base64 sources) as an OpenAI content part
inline Json image_part(const std::string& url_or_b64, const std::string& mime = "") {
  Json iu = Json::object();
  const bool url = url_or_b64.rfind("data:", 0) == 0 || url_or_b64.rfind("http", 0) == 0;
  iu.set("url", url ? url_or_b64 : "data:" + (mime.empty() ? std::string("image/png") : mime) + ";base64," + url_or_b64);
  Json p = Json::object();
  p.set("type", "image_url");
  p.set("image_url", iu);
  return p;
}
inline Json text_part(const std::string& t) {
  Json p = Json::object();
  p.set("type", "text");
  p.set("text", t);
  return p;
}
inline Json msg(const std::string& role, Json content) {
  Json m = Json::object();
  m.set("role", role);
  m.set("content", std::move(content));
  return m;
}
// a tool's result as text (string, or a list of text blocks)
inline std::string flatten_text(const Json& c) {
  if (c.is_string()) return c.str();
  std::string s;
  for (const Json& b : c.items())
    if (b["type"].str() == "text" || b["type"].str() == "output_text" || b["type"].str() == "input_text") s += b["text"].str();
  return c.is_null() ? "" : (s.empty() && !c.is_array() ? c.dump() : s);
}
inline Json openai_tool(const std::string& name, const std::string& desc, const Json& params) {
  Json fn = Json::object();
  fn.set("name", name);
  if (!desc.empty()) fn.set("description", desc);
  fn.set("parameters", params.is_null() ? Json::object() : params);
  Json t = Json::object();
  t.set("type", "function");
  t.set("function", fn);
  return t;
}
inline Json openai_call(const std::string& id, const std::string& name, const std::string& args) {
  Json fn = Json::object();
  fn.set("name", name);
  fn.set("arguments", args);
  Json tc = Json::object();
  tc.set("id", id);
  tc.set("type", "function");
  tc.set("function", fn);
  return tc;
}
inline Json parse_or_string(const std::string& s) {
  Json j;
  if (Json::try_parse(s, j)) return j;
  return Json(s);
}
// Ollama model names carry a tag ("qwen3.6-35b-a3b:latest"); arcinfer's ids do not
inline std::string strip_tag(const std::string& m) {
  const size_t c = m.rfind(':');
  return c == std::string::npos ? m : m.substr(0, c);
}

// Ollama "options" (and the top-level fields both APIs share) into OpenAI sampling fields
inline void ollama_options(const Json& in, Json& out) {
  const Json& o = in["options"];
  auto num = [&](const char* from, const char* to) { if (o[from].is_number()) out.set(to, o[from]); };
  num("temperature", "temperature");
  num("top_p", "top_p");
  num("top_k", "top_k");
  num("min_p", "min_p");
  num("seed", "seed");
  num("repeat_penalty", "repeat_penalty");
  num("repeat_last_n", "repeat_last_n");
  num("presence_penalty", "presence_penalty");
  num("frequency_penalty", "frequency_penalty");
  num("num_predict", "max_tokens");
  if (o["stop"].is_array() || o["stop"].is_string()) out.set("stop", o["stop"]);
  if (in["think"].is_bool()) out.set("enable_thinking", in["think"]);
  if (in["think"].is_string()) out.set("reasoning_effort", in["think"]);  // ("low" / "medium" / "high")
  if (in["format"].str() == "json") {
    Json rf = Json::object();
    rf.set("type", "json_object");
    out.set("response_format", rf);
  } else if (in["format"].is_object()) {
    Json js = Json::object(), rf = Json::object();
    js.set("name", "output");
    js.set("schema", in["format"]);
    rf.set("type", "json_schema");
    rf.set("json_schema", js);
    out.set("response_format", rf);
  }
}

// ---------------------------------------------------------------- request translation
// Returns false with `err` for a malformed request.
inline bool to_chat(const std::string& kind, const Json& in, Json& out, std::string& err) {
  out = Json::object();
  out.set("model", strip_tag(in["model"].str()));
  Json msgs = Json::array();
  if (kind == "completions") {
    std::string prompt = in["prompt"].is_array() ? in["prompt"][0].str() : in["prompt"].str();
    if (in["prompt"].is_null()) { err = "prompt required"; return false; }
    out.set("__raw_prompt", prompt);
    msgs.push(msg("user", Json(prompt)));
    for (const char* k : {"max_tokens", "temperature", "top_p", "top_k", "min_p", "seed", "stop", "presence_penalty",
                          "frequency_penalty", "repeat_penalty", "logit_bias", "logprobs", "stream", "n"})
      if (!in[k].is_null()) out.set(k, in[k]);
    if (in["logprobs"].is_number()) {  // (completions: logprobs = how many alternatives)
      out.set("logprobs", true);
      out.set("top_logprobs", in["logprobs"]);
    }
    if (in["max_tokens"].is_null()) out.set("max_tokens", 16);  // (the completions API's default)
  } else if (kind == "ollama_chat") {
    if (!in["messages"].is_array()) { err = "messages required"; return false; }
    int call_n = 0;
    std::vector<std::string> pending;  // ids of the last assistant tool calls, for the tool messages that follow
    for (const Json& m : in["messages"].items()) {
      const std::string role = m["role"].str();
      if (role == "tool") {
        Json t = msg("tool", Json(m["content"].str()));
        t.set("tool_call_id", pending.empty() ? "call_" + std::to_string(call_n) : pending.front());
        if (!pending.empty()) pending.erase(pending.begin());
        if (m["tool_name"].is_string()) t.set("name", m["tool_name"]);
        msgs.push(t);
        continue;
      }
      Json content = Json(m["content"].str());
      if (m["images"].is_array() && m["images"].size()) {
        content = Json::array();
        if (!m["content"].str().empty()) content.push(text_part(m["content"].str()));
        for (const Json& im : m["images"].items()) content.push(image_part(im.str()));
      }
      Json om = msg(role, content);
      if (role == "assistant" && m["thinking"].is_string()) om.set("reasoning_content", m["thinking"]);
      if (role == "assistant" && m["tool_calls"].is_array() && m["tool_calls"].size()) {
        Json tcs = Json::array();
        pending.clear();
        for (const Json& tc : m["tool_calls"].items()) {
          const std::string id = "call_" + std::to_string(call_n++);
          const Json& a = tc["function"]["arguments"];
          tcs.push(openai_call(id, tc["function"]["name"].str(), a.is_string() ? a.str() : a.dump()));
          pending.push_back(id);
        }
        om.set("tool_calls", tcs);
      }
      msgs.push(om);
    }
    if (in["tools"].is_array()) out.set("tools", in["tools"]);
    out.set("stream", in["stream"].as_bool(true));  // (Ollama streams unless told not to)
    ollama_options(in, out);
  } else if (kind == "ollama_generate") {
    if (in["raw"].as_bool(false)) {
      out.set("__raw_prompt", in["prompt"].str());
      msgs.push(msg("user", Json(in["prompt"].str())));
    } else {
      if (!in["system"].str().empty()) msgs.push(msg("system", Json(in["system"].str())));
      Json content = Json(in["prompt"].str());
      if (in["images"].is_array() && in["images"].size()) {
        content = Json::array();
        content.push(text_part(in["prompt"].str()));
        for (const Json& im : in["images"].items()) content.push(image_part(im.str()));
      }
      msgs.push(msg("user", content));
    }
    out.set("stream", in["stream"].as_bool(true));
    ollama_options(in, out);
  } else if (kind == "anthropic") {
    if (!in["messages"].is_array()) { err = "messages required"; return false; }
    const Json& sys = in["system"];
    if (sys.is_string() && !sys.str().empty()) msgs.push(msg("system", sys));
    else if (sys.is_array() && sys.size()) msgs.push(msg("system", Json(flatten_text(sys))));
    for (const Json& m : in["messages"].items()) {
      const std::string role = m["role"].str();
      const Json& c = m["content"];
      if (c.is_string()) { msgs.push(msg(role, c)); continue; }
      Json parts = Json::array(), tcs = Json::array();
      std::string text, thinking;
      for (const Json& b : c.items()) {
        const std::string ty = b["type"].str();
        if (ty == "text") {
          parts.push(text_part(b["text"].str()));
          text += b["text"].str();
        } else if (ty == "image") {
          const Json& s = b["source"];
          parts.push(s["type"].str() == "url" ? image_part(s["url"].str()) : image_part(s["data"].str(), s["media_type"].str()));
        } else if (ty == "tool_use") {
          tcs.push(openai_call(b["id"].str(), b["name"].str(), b["input"].dump()));
        } else if (ty == "tool_result") {
          Json t = msg("tool", Json(flatten_text(b["content"])));
          t.set("tool_call_id", b["tool_use_id"]);
          msgs.push(t);  // (the results first, then the rest of the user turn)
        } else if (ty == "thinking") {
          thinking += b["thinking"].str();
        }
      }
      if (role == "assistant") {
        Json om = msg("assistant", Json(text));
        if (!thinking.empty()) om.set("reasoning_content", thinking);
        if (tcs.size()) om.set("tool_calls", tcs);
        msgs.push(om);
      } else if (parts.size()) {
        msgs.push(msg(role, parts));
      }
    }
    if (in["tools"].is_array()) {
      Json tools = Json::array();
      for (const Json& t : in["tools"].items())
        if (t["name"].is_string()) tools.push(openai_tool(t["name"].str(), t["description"].str(), t["input_schema"]));
      out.set("tools", tools);
    }
    const Json& tc = in["tool_choice"];
    if (tc["type"].str() == "none") out.set("tool_choice", "none");
    else if (tc["type"].str() == "any") out.set("tool_choice", "required");
    else if (tc["type"].str() == "tool") {
      Json f = Json::object(), c2 = Json::object();
      f.set("name", tc["name"]);
      c2.set("type", "function");
      c2.set("function", f);
      out.set("tool_choice", c2);
    }
    // thinking only when asked for (Anthropic's default is off)
    const Json& th = in["thinking"];
    out.set("enable_thinking", th["type"].str() == "enabled");
    if (th["budget_tokens"].is_number()) out.set("thinking_token_budget", th["budget_tokens"]);
    for (const char* k : {"max_tokens", "temperature", "top_p", "top_k", "stream"})
      if (!in[k].is_null()) out.set(k, in[k]);
    if (in["stop_sequences"].is_array()) out.set("stop", in["stop_sequences"]);
  } else if (kind == "responses") {
    if (!in["instructions"].str().empty()) msgs.push(msg("system", Json(in["instructions"].str())));
    const Json& input = in["input"];
    if (input.is_string()) {
      msgs.push(msg("user", input));
    } else if (input.is_array()) {
      for (const Json& it : input.items()) {
        const std::string ty = it["type"].str();
        if (ty == "function_call") {  // (consecutive calls go into one assistant message)
          const Json call = openai_call(it["call_id"].str(), it["name"].str(), it["arguments"].str());
          if (msgs.size() && msgs[msgs.size() - 1]["role"].str() == "assistant" && msgs[msgs.size() - 1]["tool_calls"].is_array()) {
            Json last = msgs[msgs.size() - 1];
            Json tcs = last["tool_calls"];
            tcs.push(call);
            last.set("tool_calls", tcs);
            Json m2 = Json::array();
            for (size_t i = 0; i + 1 < msgs.size(); ++i) m2.push(msgs[i]);
            m2.push(last);
            msgs = m2;
          } else {
            Json om = msg("assistant", Json(""));
            Json tcs = Json::array();
            tcs.push(call);
            om.set("tool_calls", tcs);
            msgs.push(om);
          }
          continue;
        }
        if (ty == "function_call_output") {
          Json t = msg("tool", Json(it["output"].is_string() ? it["output"].str() : flatten_text(it["output"])));
          t.set("tool_call_id", it["call_id"]);
          msgs.push(t);
          continue;
        }
        if (ty == "reasoning") continue;
        const std::string role = it["role"].str("user");
        const Json& c = it["content"];
        if (c.is_string()) { msgs.push(msg(role == "developer" ? "system" : role, c)); continue; }
        Json parts = Json::array();
        std::string text;
        for (const Json& p : c.items()) {
          const std::string pt = p["type"].str();
          if (pt == "input_text" || pt == "output_text" || pt == "text") {
            parts.push(text_part(p["text"].str()));
            text += p["text"].str();
          } else if (pt == "input_image") {
            parts.push(image_part(p["image_url"].is_string() ? p["image_url"].str() : p["image_url"]["url"].str()));
          }
        }
        if (role == "assistant") msgs.push(msg("assistant", Json(text)));
        else msgs.push(msg(role == "developer" ? "system" : role, parts));
      }
    } else {
      err = "input required";
      return false;
    }
    if (in["tools"].is_array()) {
      Json tools = Json::array();
      for (const Json& t : in["tools"].items())
        if (t["type"].str() == "function") tools.push(openai_tool(t["name"].str(), t["description"].str(), t["parameters"]));
      if (tools.size()) out.set("tools", tools);
    }
    const Json& tc = in["tool_choice"];
    if (tc.is_string()) out.set("tool_choice", tc);
    else if (tc["type"].str() == "function") {
      Json f = Json::object(), c2 = Json::object();
      f.set("name", tc["name"]);
      c2.set("type", "function");
      c2.set("function", f);
      out.set("tool_choice", c2);
    }
    if (in["reasoning"]["effort"].is_string()) out.set("reasoning_effort", in["reasoning"]["effort"]);
    if (in["max_output_tokens"].is_number()) out.set("max_tokens", in["max_output_tokens"]);
    for (const char* k : {"temperature", "top_p", "stream"})
      if (!in[k].is_null()) out.set(k, in[k]);
    const Json& fmt = in["text"]["format"];
    if (fmt["type"].str() == "json_schema") {
      Json js = Json::object(), rf = Json::object();
      js.set("name", fmt["name"].str("output"));
      js.set("schema", fmt["schema"]);
      rf.set("type", "json_schema");
      rf.set("json_schema", js);
      out.set("response_format", rf);
    } else if (fmt["type"].str() == "json_object") {
      Json rf = Json::object();
      rf.set("type", "json_object");
      out.set("response_format", rf);
    }
  } else {
    err = "unknown API";
    return false;
  }
  if (!msgs.size()) { err = "no messages"; return false; }
  out.set("messages", msgs);
  Json so = Json::object();
  so.set("include_usage", true);  // (the final counts every format reports)
  out.set("stream_options", so);
  return true;
}

// ---------------------------------------------------------------- response translation
// One request's translation state: installed on the HttpResponse before the chat handler runs.
class Adapter {
 public:
  Adapter(std::string kind, const Json& req, HttpResponse& rs) : kind_(std::move(kind)), req_(req), rs_(rs) {
    model_ = req["model"].str();
    stream_ = kind_ == "transcription" ? false : kind_ == "ollama_chat" || kind_ == "ollama_generate" ? req["stream"].as_bool(true) : req["stream"].as_bool(false);
    id_ = std::to_string(unix_now()) + "_" + std::to_string((uintptr_t)this % 100000);
    rs.on_send = [this](int status, const std::string& body) { complete(status, body); };
    if (stream_) rs.on_event = [this](const std::string& data) { return event(data); };
  }

 private:
  std::string kind_, model_, id_;
  Json req_;
  HttpResponse& rs_;
  bool stream_ = false, started_ = false;
  // stream state: what was sent so far
  std::string text_, reasoning_;
  struct Call { std::string id, name, args; };
  std::vector<Call> calls_;
  int block_ = -1;          // Anthropic: index of the open content block
  std::string block_type_;  // its type: "thinking", "text", "tool_use"
  int seq_ = 0;             // Responses: sequence number of the events
  int out_index_ = -1;      // Responses: index of the open output item
  std::string item_type_, item_id_;
  Json usage_, timings_;

  // ---- errors in the client's format
  void error(int status, const std::string& message) {
    Json r = Json::object();
    if (kind_ == "anthropic") {
      Json e = Json::object();
      e.set("type", status == 400 ? "invalid_request_error" : "api_error");
      e.set("message", message);
      r.set("type", "error");
      r.set("error", e);
    } else if (kind_.rfind("ollama", 0) == 0) {
      r.set("error", message);
    } else {
      Json e = Json::object();
      e.set("message", message);
      e.set("type", status == 400 ? "invalid_request_error" : "server_error");
      r.set("error", e);
    }
    rs_.on_send = nullptr;
    rs_.send(status, "application/json", r.dump());
  }

  // ---- a complete (non-streamed) chat.completion
  void complete(int status, const std::string& body) {
    Json r;
    if (!Json::try_parse(body, r)) r = Json::object();
    if (status != 200) {
      const std::string m = r["error"]["message"].str(r["error"].str(body));
      if (stream_ && started_) return;  // (an error after the stream began: nothing more to send)
      return error(status, m);
    }
    const Json& ch = r["choices"][0];
    const Json& m = ch["message"];
    const std::string finish = ch["finish_reason"].str("stop");
    usage_ = r["usage"];
    timings_ = r["timings"];
    std::vector<Call> calls;
    for (const Json& tc : m["tool_calls"].items())
      calls.push_back({tc["id"].str(), tc["function"]["name"].str(), tc["function"]["arguments"].str()});
    const std::string content = m["content"].str(), reasoning = m["reasoning_content"].str();
    rs_.on_send = nullptr;
    if (kind_ == "transcription") {
      std::string text = content;
      while (!text.empty() && (text.back() == '\n' || text.back() == ' ')) text.pop_back();
      while (!text.empty() && (text.front() == '\n' || text.front() == ' ')) text.erase(0, 1);
      const std::string fmt = req_["response_format"].str("json");
      if (fmt == "text" || fmt == "srt" || fmt == "vtt") return rs_.send(200, "text/plain; charset=utf-8", text + "\n");
      Json r = Json::object();
      r.set("text", text);
      if (fmt == "verbose_json") {
        r.set("task", req_["task"].str("transcribe"));
        r.set("language", req_["language"].str(""));
      }
      return rs_.send(200, "application/json", r.dump());
    }
    rs_.send(200, "application/json", final_object(content, reasoning, calls, finish).dump());
  }

  Json final_object(const std::string& content, const std::string& reasoning, const std::vector<Call>& calls,
                    const std::string& finish) {
    const int in_tok = (int)usage_["prompt_tokens"].as_int(), out_tok = (int)usage_["completion_tokens"].as_int(),
              cached = (int)usage_["prompt_tokens_details"]["cached_tokens"].as_int();
    if (kind_ == "completions") {
      Json c = Json::object();
      c.set("text", req_["echo"].as_bool(false) ? req_["prompt"].str() + content : content);
      c.set("index", 0);
      c.set("logprobs", Json());
      c.set("finish_reason", finish == "tool_calls" ? "stop" : finish);
      Json cs = Json::array();
      cs.push(c);
      Json r = Json::object();
      r.set("id", "cmpl-" + id_);
      r.set("object", "text_completion");
      r.set("created", (double)unix_now());
      r.set("model", model_);
      r.set("choices", cs);
      r.set("usage", usage_);
      return r;
    }
    if (kind_ == "ollama_chat" || kind_ == "ollama_generate") {
      Json r = ollama_head(true);
      if (kind_ == "ollama_chat") {
        Json m = Json::object();
        m.set("role", "assistant");
        m.set("content", content);
        if (!reasoning.empty()) m.set("thinking", reasoning);
        if (!calls.empty()) m.set("tool_calls", ollama_calls(calls));
        r.set("message", m);
      } else {
        r.set("response", content);
        if (!reasoning.empty()) r.set("thinking", reasoning);
      }
      r.set("done", true);
      r.set("done_reason", finish == "length" ? "length" : "stop");
      ollama_stats(r);
      return r;
    }
    if (kind_ == "anthropic") {
      Json blocks = Json::array();
      if (!reasoning.empty()) {
        Json b = Json::object();
        b.set("type", "thinking");
        b.set("thinking", reasoning);
        b.set("signature", "");
        blocks.push(b);
      }
      if (!content.empty()) {
        Json b = Json::object();
        b.set("type", "text");
        b.set("text", content);
        blocks.push(b);
      }
      for (const Call& c : calls) {
        Json b = Json::object();
        b.set("type", "tool_use");
        b.set("id", c.id);
        b.set("name", c.name);
        b.set("input", parse_or_string(c.args));
        blocks.push(b);
      }
      Json r = Json::object();
      r.set("id", "msg_" + id_);
      r.set("type", "message");
      r.set("role", "assistant");
      r.set("model", model_);
      r.set("content", blocks);
      r.set("stop_reason", anthropic_stop(finish));
      r.set("stop_sequence", Json());
      Json u = Json::object();
      u.set("input_tokens", in_tok - cached);
      u.set("cache_read_input_tokens", cached);
      u.set("output_tokens", out_tok);
      r.set("usage", u);
      return r;
    }
    // responses
    Json output = Json::array();
    if (!reasoning.empty()) output.push(reasoning_item(reasoning, "completed"));
    if (!content.empty()) output.push(message_item(content, "completed"));
    for (const Call& c : calls) output.push(call_item(c, "completed"));
    Json r = response_object(finish == "length" ? "incomplete" : "completed");
    r.set("output", output);
    r.set("output_text", content);
    return r;
  }

  // ---- Ollama pieces
  Json ollama_head(bool) {
    Json r = Json::object();
    r.set("model", model_);
    r.set("created_at", iso_now());
    return r;
  }
  static Json ollama_calls(const std::vector<Call>& calls) {
    Json a = Json::array();
    for (const Call& c : calls) {
      Json fn = Json::object();
      fn.set("name", c.name);
      fn.set("arguments", parse_or_string(c.args));
      Json tc = Json::object();
      tc.set("function", fn);
      a.push(tc);
    }
    return a;
  }
  void ollama_stats(Json& r) {
    const double pms = timings_["prompt_ms"].as_number(), gms = timings_["predicted_ms"].as_number();
    r.set("total_duration", (double)(int64_t)((pms + gms) * 1e6));
    r.set("load_duration", 0);
    r.set("prompt_eval_count", usage_["prompt_tokens"].as_int());
    r.set("prompt_eval_duration", (double)(int64_t)(pms * 1e6));
    r.set("eval_count", usage_["completion_tokens"].as_int());
    r.set("eval_duration", (double)(int64_t)(gms * 1e6));
  }
  static std::string anthropic_stop(const std::string& f) {
    return f == "length" ? "max_tokens" : f == "tool_calls" ? "tool_use" : "end_turn";
  }

  // ---- Responses pieces
  Json response_object(const std::string& status) {
    Json r = Json::object();
    r.set("id", "resp_" + id_);
    r.set("object", "response");
    r.set("created_at", (double)unix_now());
    r.set("status", status);
    r.set("model", model_);
    r.set("output", Json::array());
    if (status == "incomplete") {
      Json d = Json::object();
      d.set("reason", "max_output_tokens");
      r.set("incomplete_details", d);
    }
    if (!usage_.is_null()) {
      Json u = Json::object();
      u.set("input_tokens", usage_["prompt_tokens"].as_int());
      u.set("output_tokens", usage_["completion_tokens"].as_int());
      u.set("total_tokens", usage_["total_tokens"].as_int());
      Json itd = Json::object(), otd = Json::object();
      itd.set("cached_tokens", usage_["prompt_tokens_details"]["cached_tokens"].as_int());
      otd.set("reasoning_tokens", 0);
      u.set("input_tokens_details", itd);
      u.set("output_tokens_details", otd);
      r.set("usage", u);
    }
    return r;
  }
  Json reasoning_item(const std::string& text, const std::string& status) {
    Json s = Json::object();
    s.set("type", "summary_text");
    s.set("text", text);
    Json sum = Json::array();
    if (!text.empty()) sum.push(s);
    Json it = Json::object();
    it.set("type", "reasoning");
    it.set("id", "rs_" + id_);
    it.set("summary", sum);
    it.set("status", status);
    return it;
  }
  Json message_item(const std::string& text, const std::string& status) {
    Json part = Json::object();
    part.set("type", "output_text");
    part.set("text", text);
    part.set("annotations", Json::array());
    Json content = Json::array();
    content.push(part);
    Json it = Json::object();
    it.set("type", "message");
    it.set("id", "msg_" + id_);
    it.set("status", status);
    it.set("role", "assistant");
    it.set("content", content);
    return it;
  }
  Json call_item(const Call& c, const std::string& status) {
    Json it = Json::object();
    it.set("type", "function_call");
    it.set("id", "fc_" + c.id);
    it.set("call_id", c.id);
    it.set("name", c.name);
    it.set("arguments", c.args);
    it.set("status", status);
    return it;
  }

  // ---- streaming
  bool sse(const std::string& event, const Json& data) {
    return rs_.write("event: " + event + "\ndata: " + data.dump() + "\n\n");
  }
  bool rsp(const std::string& type, Json data) {  // a Responses stream event
    data.set("type", type);
    data.set("sequence_number", seq_++);
    return sse(type, data);
  }
  bool line(const Json& j) { return rs_.write(j.dump() + "\n"); }  // an Ollama NDJSON line
  bool start() {
    if (started_) return true;
    started_ = true;
    if (kind_ == "ollama_chat" || kind_ == "ollama_generate") {
      rs_.begin_stream("application/x-ndjson");
      return true;
    }
    rs_.begin_stream("text/event-stream");
    if (kind_ == "anthropic") {
      Json m = Json::object();
      m.set("id", "msg_" + id_);
      m.set("type", "message");
      m.set("role", "assistant");
      m.set("model", model_);
      m.set("content", Json::array());
      m.set("stop_reason", Json());
      m.set("stop_sequence", Json());
      Json u = Json::object();
      u.set("input_tokens", 0);
      u.set("output_tokens", 0);
      m.set("usage", u);
      Json e = Json::object();
      e.set("type", "message_start");
      e.set("message", m);
      return sse("message_start", e);
    }
    if (kind_ == "responses") {
      Json e = Json::object();
      e.set("response", response_object("in_progress"));
      return rsp("response.created", e) && rsp("response.in_progress", e);
    }
    return true;
  }
  // Anthropic: open a block of this type (closing the open one)
  bool block(const std::string& type, const Call* c = nullptr) {
    if (block_type_ == type && !c) return true;
    if (block_ >= 0) {
      Json e = Json::object();
      e.set("type", "content_block_stop");
      e.set("index", block_);
      if (!sse("content_block_stop", e)) return false;
    }
    if (type.empty()) { block_type_.clear(); return true; }
    ++block_;
    block_type_ = type;
    Json cb = Json::object();
    cb.set("type", type);
    if (type == "thinking") cb.set("thinking", "");
    else if (type == "text") cb.set("text", "");
    else {
      cb.set("id", c->id);
      cb.set("name", c->name);
      cb.set("input", Json::object());
    }
    Json e = Json::object();
    e.set("type", "content_block_start");
    e.set("index", block_);
    e.set("content_block", cb);
    return sse("content_block_start", e);
  }
  bool block_delta(const std::string& dtype, const char* field, const std::string& v) {
    Json d = Json::object();
    d.set("type", dtype);
    d.set(field, v);
    Json e = Json::object();
    e.set("type", "content_block_delta");
    e.set("index", block_);
    e.set("delta", d);
    return sse("content_block_delta", e);
  }
  // Responses: open an output item of this type (closing the open one with its final content)
  bool item(const std::string& type, const Call* c = nullptr) {
    if (item_type_ == type && !c) return true;
    if (!close_item()) return false;
    if (type.empty()) return true;
    ++out_index_;
    item_type_ = type;
    Json it = type == "reasoning" ? reasoning_item("", "in_progress")
              : type == "message" ? message_item("", "in_progress")
                                  : call_item(*c, "in_progress");
    item_id_ = it["id"].str();
    if (type == "message") it.set("content", Json::array());
    Json e = Json::object();
    e.set("output_index", out_index_);
    e.set("item", it);
    if (!rsp("response.output_item.added", e)) return false;
    if (type == "message") {
      Json part = Json::object();
      part.set("type", "output_text");
      part.set("text", "");
      part.set("annotations", Json::array());
      Json e2 = Json::object();
      e2.set("item_id", item_id_);
      e2.set("output_index", out_index_);
      e2.set("content_index", 0);
      e2.set("part", part);
      return rsp("response.content_part.added", e2);
    }
    if (type == "reasoning") {
      Json part = Json::object();
      part.set("type", "summary_text");
      part.set("text", "");
      Json e2 = Json::object();
      e2.set("item_id", item_id_);
      e2.set("output_index", out_index_);
      e2.set("summary_index", 0);
      e2.set("part", part);
      return rsp("response.reasoning_summary_part.added", e2);
    }
    return true;
  }
  bool close_item() {
    if (item_type_.empty()) return true;
    Json done;
    Json e = Json::object();
    e.set("item_id", item_id_);
    e.set("output_index", out_index_);
    if (item_type_ == "message") {
      e.set("content_index", 0);
      e.set("text", text_);
      if (!rsp("response.output_text.done", e)) return false;
      Json part = Json::object();
      part.set("type", "output_text");
      part.set("text", text_);
      part.set("annotations", Json::array());
      Json e2 = Json::object();
      e2.set("item_id", item_id_);
      e2.set("output_index", out_index_);
      e2.set("content_index", 0);
      e2.set("part", part);
      if (!rsp("response.content_part.done", e2)) return false;
      done = message_item(text_, "completed");
    } else if (item_type_ == "reasoning") {
      e.set("summary_index", 0);
      e.set("text", reasoning_);
      if (!rsp("response.reasoning_summary_text.done", e)) return false;
      done = reasoning_item(reasoning_, "completed");
    } else {
      const Call& c = calls_.back();
      e.set("arguments", c.args);
      if (!rsp("response.function_call_arguments.done", e)) return false;
      done = call_item(c, "completed");
    }
    Json e3 = Json::object();
    e3.set("output_index", out_index_);
    e3.set("item", done);
    item_type_.clear();
    return rsp("response.output_item.done", e3);
  }

  // one chat.completion.chunk from the handler
  bool event(const std::string& data) {
    if (!start()) return false;
    if (data == "[DONE]") {
      if (kind_ == "completions") return rs_.write("data: [DONE]\n\n");
      return true;
    }
    Json c;
    if (!Json::try_parse(data, c)) return true;
    if (c["usage"].is_object()) usage_ = c["usage"];
    if (c["timings"].is_object()) timings_ = c["timings"];
    const Json& ch = c["choices"][0];
    const Json& d = ch["delta"];
    const std::string r = d["reasoning_content"].str(), t = d["content"].str();
    const std::string finish = ch["finish_reason"].str();
    // the pieces of this chunk, in the client's format
    if (kind_ == "completions") {
      if (!t.empty() || !finish.empty()) {
        Json cc = Json::object();
        cc.set("text", t);
        cc.set("index", 0);
        cc.set("logprobs", Json());
        cc.set("finish_reason", finish.empty() ? Json() : Json(finish == "tool_calls" ? "stop" : finish));
        Json cs = Json::array();
        cs.push(cc);
        Json o = Json::object();
        o.set("id", "cmpl-" + id_);
        o.set("object", "text_completion");
        o.set("created", (double)unix_now());
        o.set("model", model_);
        o.set("choices", cs);
        if (!rs_.write("data: " + o.dump() + "\n\n")) return false;
      }
      if (c["usage"].is_object() && !c["choices"].size()) return true;
      return true;
    }
    // tool call pieces (accumulated; Ollama gets them whole at the end)
    for (const Json& tc : d["tool_calls"].items()) {
      const size_t k = (size_t)tc["index"].as_int();
      const bool first = k >= calls_.size();
      if (first) calls_.push_back({tc["id"].str(), tc["function"]["name"].str(), ""});
      const std::string piece = tc["function"]["arguments"].str();
      calls_[k].args += piece;
      if (kind_ == "anthropic") {
        if (first && !block("tool_use", &calls_[k])) return false;
        if (!piece.empty() && !block_delta("input_json_delta", "partial_json", piece)) return false;
      } else if (kind_ == "responses") {
        if (first && !item("function_call", &calls_[k])) return false;
        if (!piece.empty()) {
          Json e = Json::object();
          e.set("item_id", item_id_);
          e.set("output_index", out_index_);
          e.set("delta", piece);
          if (!rsp("response.function_call_arguments.delta", e)) return false;
        }
      }
    }
    if (!r.empty()) {
      reasoning_ += r;
      if (kind_ == "anthropic") {
        if (!block("thinking") || !block_delta("thinking_delta", "thinking", r)) return false;
      } else if (kind_ == "responses") {
        Json e = Json::object();
        if (!item("reasoning")) return false;
        e.set("item_id", item_id_);
        e.set("output_index", out_index_);
        e.set("summary_index", 0);
        e.set("delta", r);
        if (!rsp("response.reasoning_summary_text.delta", e)) return false;
      } else {
        Json o = ollama_head(false);
        if (kind_ == "ollama_chat") {
          Json m = Json::object();
          m.set("role", "assistant");
          m.set("content", "");
          m.set("thinking", r);
          o.set("message", m);
        } else {
          o.set("response", "");
          o.set("thinking", r);
        }
        o.set("done", false);
        if (!line(o)) return false;
      }
    }
    if (!t.empty()) {
      text_ += t;
      if (kind_ == "anthropic") {
        if (!block("text") || !block_delta("text_delta", "text", t)) return false;
      } else if (kind_ == "responses") {
        Json e = Json::object();
        if (!item("message")) return false;
        e.set("item_id", item_id_);
        e.set("output_index", out_index_);
        e.set("content_index", 0);
        e.set("delta", t);
        if (!rsp("response.output_text.delta", e)) return false;
      } else {
        Json o = ollama_head(false);
        if (kind_ == "ollama_chat") {
          Json m = Json::object();
          m.set("role", "assistant");
          m.set("content", t);
          o.set("message", m);
        } else {
          o.set("response", t);
        }
        o.set("done", false);
        if (!line(o)) return false;
      }
    }
    if (!finish.empty()) finish_ = finish;
    // the end: the usage comes in the chunk after the finish reason (stream_options.include_usage)
    if (c["usage"].is_object()) return end();
    return true;
  }
  std::string finish_ = "stop";
  bool end() {
    if (kind_ == "anthropic") {
      if (!block("")) return false;
      Json d = Json::object();
      d.set("stop_reason", anthropic_stop(finish_));
      d.set("stop_sequence", Json());
      Json u = Json::object();
      u.set("output_tokens", usage_["completion_tokens"].as_int());
      Json e = Json::object();
      e.set("type", "message_delta");
      e.set("delta", d);
      e.set("usage", u);
      if (!sse("message_delta", e)) return false;
      Json e2 = Json::object();
      e2.set("type", "message_stop");
      return sse("message_stop", e2);
    }
    if (kind_ == "responses") {
      if (!close_item()) return false;
      Json output = Json::array();
      if (!reasoning_.empty()) output.push(reasoning_item(reasoning_, "completed"));
      if (!text_.empty()) output.push(message_item(text_, "completed"));
      for (const Call& c : calls_) output.push(call_item(c, "completed"));
      Json r = response_object(finish_ == "length" ? "incomplete" : "completed");
      r.set("output", output);
      r.set("output_text", text_);
      Json e = Json::object();
      e.set("response", r);
      return rsp(finish_ == "length" ? "response.incomplete" : "response.completed", e);
    }
    // Ollama: the tool calls whole, then the final line with the counts
    if (!calls_.empty()) {
      Json o = ollama_head(false);
      Json m = Json::object();
      m.set("role", "assistant");
      m.set("content", "");
      m.set("tool_calls", ollama_calls(calls_));
      o.set("message", m);
      o.set("done", false);
      if (!line(o)) return false;
    }
    Json o = ollama_head(true);
    if (kind_ == "ollama_chat") {
      Json m = Json::object();
      m.set("role", "assistant");
      m.set("content", "");
      o.set("message", m);
    } else {
      o.set("response", "");
    }
    o.set("done", true);
    o.set("done_reason", finish_ == "length" ? "length" : "stop");
    ollama_stats(o);
    return line(o);
  }
};

}  // namespace apic
