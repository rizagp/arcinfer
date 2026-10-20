// Minimal HTTP/1.1 server on Winsock2: one thread per connection, keep-alive, CORS, and server-sent events.
#pragma once
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>
#include <algorithm>
#include <atomic>
#include <charconv>
#include <functional>
#include <map>
#include <string>
#include <thread>
#pragma comment(lib, "ws2_32.lib")

struct HttpRequest {
  std::string method, path, query, body;
  std::map<std::string, std::string> headers;  // lower-case keys
  std::string header(const std::string& k) const {
    auto it = headers.find(k);
    return it == headers.end() ? "" : it->second;
  }
};

class HttpResponse {
 public:
  explicit HttpResponse(SOCKET s) : s_(s) {}
  // Taps: another API format served by the same handler (the server's Ollama / Anthropic / Responses endpoints run
  // its OpenAI chat handler): with on_send set a complete response goes there instead of the socket, with on_event
  // set the handler's server-sent events do (sse_begin then writes nothing; the tap writes its own stream).
  std::function<void(int status, const std::string& body)> on_send;
  std::function<bool(const std::string& data)> on_event;
  // A stream of another kind (application/x-ndjson, or named server-sent events): headers, then write().
  void begin_stream(const std::string& content_type) {
    std::string h = "HTTP/1.1 200 OK\r\nContent-Type: " + content_type + "\r\nCache-Control: no-cache\r\n";
    h += cors();
    h += "Connection: close\r\n\r\n";
    raw(h);
    streaming_ = true;
  }
  bool write(const std::string& d) { return raw(d); }
  // Complete response with a body.
  void send(int status, const std::string& content_type, const std::string& body) {
    if (on_send) {
      done_ = true;
      on_send(status, body);
      return;
    }
    std::string h = "HTTP/1.1 " + std::to_string(status) + " " + reason(status) + "\r\n";
    h += "Content-Type: " + content_type + "\r\n";
    h += "Content-Length: " + std::to_string(body.size()) + "\r\n";
    h += cors();
    h += "Connection: keep-alive\r\n\r\n";
    raw(h);
    raw(body);
    done_ = true;
  }
  // Server-sent events: begin, then write events, then end (the connection is closed afterwards).
  void sse_begin() {
    if (on_event) return;
    std::string h = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n";
    h += cors();
    h += "Connection: close\r\n\r\n";
    raw(h);
    streaming_ = true;
  }
  bool sse_event(const std::string& data) { return on_event ? on_event(data) : raw("data: " + data + "\n\n"); }
  bool sse_comment(const std::string& text) { return on_event ? ok_ : raw(": " + text + "\n\n"); }
  void sse_end() { done_ = true; }
  bool streaming() const { return streaming_; }
  bool done() const { return done_; }
  bool alive() const { return ok_; }
  // The client closed the connection (or it broke) while the response is being prepared, e.g. Pi's Esc: the request
  // was read completely, so the socket becomes readable only with the end of the stream (or a reset). Polled
  // without blocking or writing anything (microseconds).
  bool client_gone() {
    if (!ok_) return true;
    WSAPOLLFD p{s_, POLLRDNORM, 0};
    if (WSAPoll(&p, 1, 0) <= 0) return false;
    char c;
    if ((p.revents & (POLLERR | POLLHUP | POLLNVAL)) || ((p.revents & POLLRDNORM) && recv(s_, &c, 1, MSG_PEEK) <= 0))
      ok_ = false;
    return !ok_;
  }

 private:
  SOCKET s_;
  bool done_ = false, streaming_ = false, ok_ = true;
  static std::string cors() {
    return "Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\n";
  }
  static const char* reason(int s) {
    switch (s) {
      case 200: return "OK"; case 204: return "No Content"; case 400: return "Bad Request"; case 404: return "Not Found";
      case 413: return "Payload Too Large"; case 500: return "Internal Server Error"; case 503: return "Service Unavailable"; default: return "OK";
    }
  }
  bool raw(const std::string& d) {
    size_t off = 0;
    while (ok_ && off < d.size()) {
      const int n = ::send(s_, d.data() + off, (int)std::min<size_t>(d.size() - off, 1 << 20), 0);
      if (n <= 0) { ok_ = false; break; }
      off += n;
    }
    return ok_;
  }
};

class HttpServer {
 public:
  using Handler = std::function<void(const HttpRequest&, HttpResponse&)>;
  Handler handler;

  bool listen(const std::string& host, int port) {
    WSADATA w;
    if (WSAStartup(MAKEWORD(2, 2), &w) != 0) return false;
    ls_ = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (ls_ == INVALID_SOCKET) return false;
    BOOL yes = TRUE;
    setsockopt(ls_, SOL_SOCKET, SO_REUSEADDR, (const char*)&yes, sizeof yes);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons((u_short)port);
    inet_pton(AF_INET, host.c_str(), &a.sin_addr);
    if (bind(ls_, (sockaddr*)&a, sizeof a) != 0 || ::listen(ls_, 64) != 0) return false;
    return true;
  }

  void run() {
    while (true) {
      SOCKET c = accept(ls_, nullptr, nullptr);
      if (c == INVALID_SOCKET) continue;
      std::thread([this, c] { serve(c); }).detach();
    }
  }

 private:
  SOCKET ls_ = INVALID_SOCKET;
  static constexpr size_t kMaxBody = 256u << 20;  // (base64 images and audio; larger requests get 413)

  // A non-negative integer (decimal, or hex for chunk sizes, which may carry ";extensions"), else false.
  static bool parse_size(std::string v, int base, size_t& out) {
    if (base == 16) v = v.substr(0, v.find(';'));
    const size_t a = v.find_first_not_of(" \t"), b = v.find_last_not_of(" \t");
    if (a == std::string::npos) return false;
    const char *f = v.data() + a, *l = v.data() + b + 1;
    const auto r = std::from_chars(f, l, out, base);
    return r.ec == std::errc() && r.ptr == l;
  }
  static void reply_close(SOCKET c, int status, const char* msg) {
    HttpResponse rs(c);
    rs.send(status, "application/json", std::string("{\"error\":{\"message\":\"") + msg + "\"}}");
    closesocket(c);
  }

  // One connection: requests until it closes. Nothing may escape this thread (it is detached: an exception would
  // end the whole server), so malformed requests get 400 / 413 and the connection is closed.
  void serve(SOCKET c) {
    try {
      serve_requests(c);
    } catch (...) {
      closesocket(c);
    }
  }
  void serve_requests(SOCKET c) {
    DWORD tmo = 600000;
    setsockopt(c, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tmo, sizeof tmo);
    // a client that stops reading (half-open connection, stalled proxy) must not hold the engine: sends give up
    // after 60 s, the response is then dead and the generation stops at the next token
    DWORD stmo = 60000;
    setsockopt(c, SOL_SOCKET, SO_SNDTIMEO, (const char*)&stmo, sizeof stmo);
    BOOL nodelay = TRUE;  // (streamed tokens go out at once)
    setsockopt(c, IPPROTO_TCP, TCP_NODELAY, (const char*)&nodelay, sizeof nodelay);
    std::string buf;
    char tmp[65536];
    while (true) {
      // read headers
      size_t he;
      while ((he = buf.find("\r\n\r\n")) == std::string::npos) {
        const int n = recv(c, tmp, sizeof tmp, 0);
        if (n <= 0) { closesocket(c); return; }
        buf.append(tmp, n);
        if (buf.size() > (64u << 20)) { closesocket(c); return; }
      }
      HttpRequest rq;
      const std::string head = buf.substr(0, he);
      size_t p = head.find("\r\n");
      const std::string line = head.substr(0, p);
      const size_t s1 = line.find(' '), s2 = s1 == std::string::npos ? s1 : line.find(' ', s1 + 1);
      if (s2 == std::string::npos) return reply_close(c, 400, "malformed request line");
      rq.method = line.substr(0, s1);
      std::string target = line.substr(s1 + 1, s2 - s1 - 1);
      const size_t q = target.find('?');
      rq.path = target.substr(0, q);
      if (q != std::string::npos) rq.query = target.substr(q + 1);
      while (p != std::string::npos && p < head.size()) {
        const size_t e = head.find("\r\n", p + 2);
        const std::string hl = head.substr(p + 2, (e == std::string::npos ? head.size() : e) - p - 2);
        const size_t colon = hl.find(':');
        if (colon != std::string::npos) {
          std::string k = hl.substr(0, colon), v = hl.substr(colon + 1);
          std::transform(k.begin(), k.end(), k.begin(), ::tolower);
          v.erase(0, v.find_first_not_of(' '));
          rq.headers[k] = v;
        }
        p = e;
      }
      buf.erase(0, he + 4);
      // body
      if (rq.header("transfer-encoding").find("chunked") != std::string::npos) {
        while (true) {
          size_t le;
          while ((le = buf.find("\r\n")) == std::string::npos) {
            if (buf.size() > 4096) return reply_close(c, 400, "chunk size line too long");
            const int n = recv(c, tmp, sizeof tmp, 0);
            if (n <= 0) { closesocket(c); return; }
            buf.append(tmp, n);
          }
          size_t len = 0;
          if (!parse_size(buf.substr(0, le), 16, len)) return reply_close(c, 400, "malformed chunk size");
          if (len > kMaxBody || rq.body.size() + len > kMaxBody) return reply_close(c, 413, "request body too large");
          if (len == 0) {  // the last chunk: optional trailer lines (ignored), then an empty line
            size_t te;
            while ((te = buf.find("\r\n\r\n", le)) == std::string::npos) {
              if (buf.size() > le + 65536) return reply_close(c, 400, "chunk trailers too long");
              const int n = recv(c, tmp, sizeof tmp, 0);
              if (n <= 0) { closesocket(c); return; }
              buf.append(tmp, n);
            }
            buf.erase(0, te + 4);
            break;
          }
          while (buf.size() < le + 2 + len + 2) {
            const int n = recv(c, tmp, sizeof tmp, 0);
            if (n <= 0) { closesocket(c); return; }
            buf.append(tmp, n);
          }
          rq.body.append(buf, le + 2, len);
          buf.erase(0, le + 2 + len + 2);
        }
      } else {
        size_t len = 0;
        if (rq.headers.count("content-length") && !parse_size(rq.header("content-length"), 10, len))
          return reply_close(c, 400, "malformed Content-Length");
        if (len > kMaxBody) return reply_close(c, 413, "request body too large");
        while (buf.size() < len) {
          const int n = recv(c, tmp, sizeof tmp, 0);
          if (n <= 0) { closesocket(c); return; }
          buf.append(tmp, n);
        }
        rq.body = buf.substr(0, len);
        buf.erase(0, len);
      }
      HttpResponse rs(c);
      if (rq.method == "OPTIONS") rs.send(204, "text/plain", "");
      else {
        try {
          handler(rq, rs);
        } catch (const std::exception& e) {
          std::string m;  // (the message as a JSON string)
          for (const char* p = e.what(); *p; ++p) {
            const unsigned char ch = (unsigned char)*p;
            if (ch == '"' || ch == '\\') { m += '\\'; m += (char)ch; }
            else if (ch < 0x20) { char u[8]; snprintf(u, sizeof u, "\\u%04x", ch); m += u; }
            else m += (char)ch;
          }
          const std::string body = "{\"error\":{\"message\":\"" + m + "\",\"type\":\"server_error\"}}";
          if (!rs.done() && !rs.streaming()) rs.send(500, "application/json", body);
          else if (rs.streaming() && !rs.done()) {  // (a stream that broke off: an error event, as OpenAI sends one)
            try { rs.sse_event(body); } catch (...) {}
            rs.sse_end();
          }
        }
        if (!rs.done() && !rs.streaming()) rs.send(404, "application/json", "{\"error\":{\"message\":\"not found\"}}");
      }
      if (rs.streaming() || !rs.alive()) break;
    }
    shutdown(c, SD_BOTH);
    closesocket(c);
  }
};
