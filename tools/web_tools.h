// Web search and page fetch for the web UI's tools (server endpoints /v1/tools/web_search and /v1/tools/web_fetch):
// WinHTTP requests, result parsing for Brave Search (Bing when Brave gives nothing) and HTML to plain text. Only
// public http(s) addresses are fetched: every host, redirects included, is resolved first and loopback, private,
// link-local, multicast and reserved addresses are refused, and the address the connection went to is checked again
// before anything is read, so a model cannot reach the machine or the local network.
#pragma once
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <winhttp.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#pragma comment(lib, "winhttp.lib")
#pragma comment(lib, "ws2_32.lib")

namespace webt {

struct Page {
  int status = 0;
  std::string url, content_type, body, error;  // url: after redirects
};

struct Result {
  std::string title, url, snippet;
};

inline std::wstring widen(const std::string& s) {
  if (s.empty()) return {};
  const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), nullptr, 0);
  std::wstring w(n, L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), w.data(), n);
  return w;
}

inline std::string narrow(const wchar_t* p, size_t len, UINT cp = CP_UTF8) {
  if (!len) return {};
  const int n = WideCharToMultiByte(cp, 0, p, (int)len, nullptr, 0, nullptr, nullptr);
  std::string s(n, '\0');
  WideCharToMultiByte(cp, 0, p, (int)len, s.data(), n, nullptr, nullptr);
  return s;
}

inline std::string lower(std::string s) {
  for (char& c : s) c = (char)tolower((unsigned char)c);
  return s;
}

inline std::string url_encode(const std::string& s) {
  static const char* hex = "0123456789ABCDEF";
  std::string o;
  for (unsigned char c : s) {
    if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') o += (char)c;
    else if (c == ' ') o += '+';
    else { o += '%'; o += hex[c >> 4]; o += hex[c & 15]; }
  }
  return o;
}

// A public address: not loopback, private (RFC 1918, carrier-grade NAT, IPv6 unique local), link-local, multicast,
// unspecified, reserved or for documentation / benchmarks; an IPv6 address that carries an IPv4 one (mapped,
// compatible, NAT64, 6to4) is judged by that IPv4 address, Teredo is refused
inline bool public_v4(const uint8_t* b) {
  if (b[0] == 0 || b[0] == 10 || b[0] == 127 || b[0] >= 224) return false;  // (224+: multicast and 240/4 reserved)
  if (b[0] == 169 && b[1] == 254) return false;
  if (b[0] == 172 && (b[1] & 0xf0) == 16) return false;
  if (b[0] == 192 && b[1] == 168) return false;
  if (b[0] == 100 && (b[1] & 0xc0) == 64) return false;
  if (b[0] == 192 && b[1] == 0 && (b[2] == 0 || b[2] == 2)) return false;  // 192.0.0.0/24, 192.0.2.0/24
  if (b[0] == 198 && (b[1] & 0xfe) == 18) return false;                      // 198.18.0.0/15
  if (b[0] == 198 && b[1] == 51 && b[2] == 100) return false;                // 198.51.100.0/24
  if (b[0] == 203 && b[1] == 0 && b[2] == 113) return false;                 // 203.0.113.0/24
  return true;
}
inline bool public_addr(const sockaddr* sa) {
  if (sa->sa_family == AF_INET) return public_v4((const uint8_t*)&((const sockaddr_in*)sa)->sin_addr);
  if (sa->sa_family == AF_INET6) {
    const uint8_t* b = ((const sockaddr_in6*)sa)->sin6_addr.s6_addr;
    static const uint8_t v4map[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff};
    static const uint8_t v4compat[12] = {};
    static const uint8_t nat64[12] = {0, 0x64, 0xff, 0x9b, 0, 0, 0, 0, 0, 0, 0, 0};
    if (!memcmp(b, v4map, 12) || !memcmp(b, v4compat, 12) || !memcmp(b, nat64, 12)) return public_v4(b + 12);  // (:: and ::1 too)
    if (b[0] == 0x20 && b[1] == 0x02) return public_v4(b + 2);                 // 6to4 2002::/16
    if (b[0] == 0x20 && b[1] == 0x01 && b[2] == 0 && b[3] == 0) return false;  // Teredo 2001::/32
    if (b[0] == 0x20 && b[1] == 0x01 && b[2] == 0x0d && b[3] == 0xb8) return false;  // documentation 2001:db8::/32
    if (b[0] == 0x01 && b[1] == 0 && !b[2] && !b[3] && !b[4] && !b[5] && !b[6] && !b[7]) return false;  // discard 100::/64
    if ((b[0] & 0xfe) == 0xfc) return false;                   // fc00::/7
    if (b[0] == 0xfe && (b[1] & 0x80) == 0x80) return false;  // fe80::/10 link-local, fec0::/10 site-local
    if (b[0] == 0xff) return false;                            // multicast
    return true;
  }
  return false;
}

// Every address the host resolves to is public (empty error), else why not
inline std::string check_host(const std::string& host) {
  if (host.empty()) return "no host";
  const std::string h = lower(host);
  if (h == "localhost" || (h.size() > 10 && h.compare(h.size() - 10, 10, ".localhost") == 0)) return "local address refused";
  WSADATA wd;
  WSAStartup(MAKEWORD(2, 2), &wd);
  addrinfo hints{}, *res = nullptr;
  hints.ai_socktype = SOCK_STREAM;
  if (getaddrinfo(host.c_str(), nullptr, &hints, &res) != 0 || !res) return "cannot resolve " + host;
  std::string e;
  for (addrinfo* a = res; a && e.empty(); a = a->ai_next)
    if (!public_addr(a->ai_addr)) e = "local or private address refused (" + host + ")";
  freeaddrinfo(res);
  return e;
}

// A redirect target against the URL it came from (absolute, scheme-relative, absolute path or relative path)
inline std::string resolve_url(const std::string& base, const std::string& loc) {
  if (loc.rfind("http://", 0) == 0 || loc.rfind("https://", 0) == 0) return loc;
  const size_t se = base.find("://");
  if (se == std::string::npos) return loc;
  if (loc.rfind("//", 0) == 0) return base.substr(0, se + 1) + loc;
  const size_t he = base.find('/', se + 3);
  const std::string origin = he == std::string::npos ? base : base.substr(0, he);
  if (!loc.empty() && loc[0] == '/') return origin + loc;
  std::string dir = he == std::string::npos ? "/" : base.substr(he, base.find_first_of("?#", he) - he);
  dir = dir.substr(0, dir.rfind('/') + 1);
  return origin + dir + loc;
}

// GET a public http(s) URL: redirects followed by hand (each host checked), at most max_bytes of body, the whole
// request within timeout_ms
inline Page get(std::string url, size_t max_bytes = 4u << 20, int timeout_ms = 15000, const std::string& accept_lang = "en-US,en;q=0.9") {
  Page p;
  const auto t0 = std::chrono::steady_clock::now();
  HINTERNET ses = WinHttpOpen(L"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36",
                              WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
  if (!ses) { p.error = "WinHttpOpen failed"; return p; }
  DWORD dec = WINHTTP_DECOMPRESSION_FLAG_ALL;
  WinHttpSetOption(ses, WINHTTP_OPTION_DECOMPRESSION, &dec, sizeof dec);
  DWORD h2 = WINHTTP_PROTOCOL_FLAG_HTTP2;
  WinHttpSetOption(ses, WINHTTP_OPTION_ENABLE_HTTP_PROTOCOL, &h2, sizeof h2);
  WinHttpSetTimeouts(ses, timeout_ms, timeout_ms, timeout_ms, timeout_ms);
  for (int hop = 0; hop < 6; ++hop) {
    p = Page{};
    p.url = url;
    const std::wstring wu = widen(url);
    URL_COMPONENTS uc{};
    uc.dwStructSize = sizeof uc;
    wchar_t host[256] = {}, path[8192] = {}, extra[8192] = {};
    uc.lpszHostName = host, uc.dwHostNameLength = 256;
    uc.lpszUrlPath = path, uc.dwUrlPathLength = 8192;
    uc.lpszExtraInfo = extra, uc.dwExtraInfoLength = 8192;
    if (!WinHttpCrackUrl(wu.c_str(), (DWORD)wu.size(), 0, &uc) || (uc.nScheme != INTERNET_SCHEME_HTTP && uc.nScheme != INTERNET_SCHEME_HTTPS)) {
      p.error = "not an http(s) URL";
      break;
    }
    const std::string h8 = narrow(host, wcslen(host));
    if (!(p.error = check_host(h8)).empty()) break;
    HINTERNET con = WinHttpConnect(ses, host, uc.nPort, 0);
    const std::wstring target = std::wstring(path) + extra;
    HINTERNET rq = con ? WinHttpOpenRequest(con, L"GET", target.empty() ? L"/" : target.c_str(), nullptr, WINHTTP_NO_REFERER,
                                            WINHTTP_DEFAULT_ACCEPT_TYPES, uc.nScheme == INTERNET_SCHEME_HTTPS ? WINHTTP_FLAG_SECURE : 0)
                       : nullptr;
    if (!rq) {
      p.error = "cannot connect to " + h8;
      if (con) WinHttpCloseHandle(con);
      break;
    }
    DWORD noredir = WINHTTP_DISABLE_REDIRECTS;
    WinHttpSetOption(rq, WINHTTP_OPTION_DISABLE_FEATURE, &noredir, sizeof noredir);
    const std::wstring hdr = widen("Accept: text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8\r\nAccept-Language: " + accept_lang + "\r\n");
    bool ok = WinHttpSendRequest(rq, hdr.c_str(), (DWORD)hdr.size(), WINHTTP_NO_REQUEST_DATA, 0, 0, 0) && WinHttpReceiveResponse(rq, nullptr);
    if (ok) {
      // the address WinHTTP actually connected to (it resolves the host again: a name that answered with a public
      // address for check_host and a local one now, DNS rebinding, gets nothing back)
      WINHTTP_CONNECTION_INFO ci{};
      DWORD cn = sizeof ci;
      if (!WinHttpQueryOption(rq, WINHTTP_OPTION_CONNECTION_INFO, &ci, &cn) || !public_addr((const sockaddr*)&ci.RemoteAddress)) {
        p.error = "local or private address refused (" + h8 + ", at connection)";
        WinHttpCloseHandle(rq), WinHttpCloseHandle(con);
        break;
      }
      DWORD st = 0, sz = sizeof st;
      WinHttpQueryHeaders(rq, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_HEADER_NAME_BY_INDEX, &st, &sz, WINHTTP_NO_HEADER_INDEX);
      p.status = (int)st;
      wchar_t buf[4096];
      sz = sizeof buf;
      if (WinHttpQueryHeaders(rq, WINHTTP_QUERY_CONTENT_TYPE, WINHTTP_HEADER_NAME_BY_INDEX, buf, &sz, WINHTTP_NO_HEADER_INDEX))
        p.content_type = narrow(buf, sz / sizeof(wchar_t));
      if (st >= 300 && st < 400) {
        sz = sizeof buf;
        if (WinHttpQueryHeaders(rq, WINHTTP_QUERY_LOCATION, WINHTTP_HEADER_NAME_BY_INDEX, buf, &sz, WINHTTP_NO_HEADER_INDEX)) {
          url = resolve_url(url, narrow(buf, sz / sizeof(wchar_t)));
          WinHttpCloseHandle(rq), WinHttpCloseHandle(con);
          continue;
        }
      }
      for (;;) {
        DWORD avail = 0;
        if (!WinHttpQueryDataAvailable(rq, &avail) || !avail) break;
        const size_t old = p.body.size();
        avail = (DWORD)std::min<size_t>(avail, max_bytes - old);  // (old < max_bytes: the loop stops at max_bytes)
        p.body.resize(old + avail);
        DWORD got = 0;
        if (!WinHttpReadData(rq, p.body.data() + old, avail, &got)) { p.body.resize(old); break; }
        p.body.resize(old + got);
        if (p.body.size() >= max_bytes) { p.body.resize(max_bytes); break; }
        if (std::chrono::steady_clock::now() - t0 > std::chrono::milliseconds(timeout_ms)) break;
      }
    } else {
      p.error = "request to " + h8 + " failed (error " + std::to_string(GetLastError()) + ")";
    }
    WinHttpCloseHandle(rq), WinHttpCloseHandle(con);
    break;
  }
  if (p.error.empty() && p.status >= 300 && p.status < 400) p.error = "too many redirects";
  WinHttpCloseHandle(ses);
  // a body in Latin-1 / Windows-1252 (the content type or a meta tag says so) becomes UTF-8
  const std::string ct = lower(p.content_type), head = lower(p.body.substr(0, 2048));
  if (ct.find("iso-8859-1") != std::string::npos || ct.find("windows-1252") != std::string::npos ||
      (ct.find("charset") == std::string::npos && (head.find("charset=iso-8859-1") != std::string::npos || head.find("charset=windows-1252") != std::string::npos))) {
    const int n = MultiByteToWideChar(1252, 0, p.body.data(), (int)p.body.size(), nullptr, 0);
    std::wstring w(n, L'\0');
    MultiByteToWideChar(1252, 0, p.body.data(), (int)p.body.size(), w.data(), n);
    p.body = narrow(w.data(), w.size());
  }
  return p;
}

inline void put_utf8(std::string& o, uint32_t c) {
  if (c < 0x80) o += (char)c;
  else if (c < 0x800) o += (char)(0xc0 | c >> 6), o += (char)(0x80 | (c & 63));
  else if (c < 0x10000) o += (char)(0xe0 | c >> 12), o += (char)(0x80 | (c >> 6 & 63)), o += (char)(0x80 | (c & 63));
  else if (c < 0x110000) o += (char)(0xf0 | c >> 18), o += (char)(0x80 | (c >> 12 & 63)), o += (char)(0x80 | (c >> 6 & 63)), o += (char)(0x80 | (c & 63));
}

// HTML character references (numeric and the common named ones)
inline std::string decode_entities(const std::string& s) {
  static const std::pair<const char*, uint32_t> named[] = {
      {"amp", '&'}, {"lt", '<'}, {"gt", '>'}, {"quot", '"'}, {"apos", '\''}, {"nbsp", ' '}, {"ndash", 0x2013},
      {"mdash", 0x2014}, {"hellip", 0x2026}, {"lsquo", 0x2018}, {"rsquo", 0x2019}, {"ldquo", 0x201c}, {"rdquo", 0x201d},
      {"laquo", 0xab}, {"raquo", 0xbb}, {"copy", 0xa9}, {"reg", 0xae}, {"trade", 0x2122}, {"middot", 0xb7},
      {"bull", 0x2022}, {"times", 0xd7}, {"deg", 0xb0}, {"euro", 0x20ac}, {"pound", 0xa3}, {"rarr", 0x2192}};
  std::string o;
  o.reserve(s.size());
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] != '&') { o += s[i]; continue; }
    const size_t e = s.find(';', i + 1);
    if (e == std::string::npos || e - i > 10) { o += '&'; continue; }
    const std::string n = s.substr(i + 1, e - i - 1);
    uint32_t c = 0;
    if (n.size() > 1 && n[0] == '#') {
      c = (n[1] == 'x' || n[1] == 'X') ? (uint32_t)strtoul(n.c_str() + 2, nullptr, 16) : (uint32_t)strtoul(n.c_str() + 1, nullptr, 10);
    } else {
      for (auto& [k, v] : named) if (n == k) c = v;
    }
    if (!c) { o += '&'; continue; }
    put_utf8(o, c == 0xa0 ? ' ' : c);
    i = e;
  }
  return o;
}

// Collapse spaces within lines, trim lines, at most one blank line in a row
inline std::string tidy(const std::string& s) {
  std::string o;
  int nl = 2;
  bool sp = false;
  for (char c : s) {
    if (c == '\r') continue;
    if (c == '\n') {
      while (!o.empty() && o.back() == ' ') o.pop_back();
      if (nl < 2) o += '\n';
      ++nl, sp = false;
    } else if (c == ' ' || c == '\t' || c == '\f' || c == '\v') {
      sp = true;
    } else {
      if (sp && nl == 0) o += ' ';
      o += c, nl = 0, sp = false;
    }
  }
  while (!o.empty() && (o.back() == '\n' || o.back() == ' ')) o.pop_back();
  return o;
}

// The '>' that ends the tag starting at i (quoted attribute values may hold '>'), npos when none
inline size_t tag_end(const std::string& s, size_t i) {
  char q = 0;
  for (size_t j = i + 1; j < s.size(); ++j) {
    const char c = s[j];
    if (q) {
      if (c == q) q = 0;
    } else if ((c == '"' || c == '\'') && s[j - 1] == '=') {
      q = c;
    } else if (c == '>') {
      return j;
    }
  }
  return std::string::npos;
}

inline std::string strip_tags(const std::string& s) {
  std::string o;
  bool in = false;
  for (char c : s) {
    if (c == '<') in = true;
    else if (c == '>') in = false;
    else if (!in) o += c;
  }
  return tidy(decode_entities(o));
}

// HTML to readable text: scripts, styles, navigation, forms and the like dropped; headings as Markdown '#', list
// items as '- ', block elements on their own lines. The page's <main> (or first <article>) when it has enough text.
inline std::string html_to_text(const std::string& html, std::string* title = nullptr) {
  const std::string lo = lower(html);
  if (title) {
    const size_t a = lo.find("<title");
    const size_t b = a == std::string::npos ? a : lo.find('>', a);
    const size_t c = b == std::string::npos ? b : lo.find("</title", b);
    *title = c == std::string::npos ? std::string() : strip_tags(html.substr(b + 1, c - b - 1));
  }
  static const char* skip[] = {"script", "style", "noscript", "svg", "template", "head", "nav", "footer", "aside",
                               "form", "select", "button", "iframe", "object", "canvas"};
  static const char* block[] = {"p", "div", "br", "tr", "table", "section", "article", "main", "header", "ul", "ol",
                                "dl", "dt", "dd", "pre", "blockquote", "figure", "figcaption", "hr", "li", "h1", "h2",
                                "h3", "h4", "h5", "h6"};
  auto convert = [&](size_t from, size_t to) {
    std::string o;
    size_t i = from;
    while (i < to) {
      if (lo[i] != '<') {
        const size_t n = std::min(lo.find('<', i), to);
        o.append(html, i, n - i);
        i = n;
        continue;
      }
      if (lo.compare(i, 4, "<!--") == 0) {
        const size_t e = lo.find("-->", i + 4);
        i = e == std::string::npos ? to : e + 3;
        continue;
      }
      const size_t e = tag_end(lo, i);
      if (e == std::string::npos) break;
      const bool close = lo[i + 1] == '/';
      size_t ns = i + (close ? 2 : 1), ne = ns;
      while (ne < e && (isalnum((unsigned char)lo[ne]))) ++ne;
      const std::string name = lo.substr(ns, ne - ns);
      i = e + 1;
      if (!close && lo[e - 1] != '/') {
        bool sk = false;
        for (const char* t : skip) sk = sk || name == t;
        if (sk) {  // to the matching close tag (nesting of the same tag counted)
          int depth = 1;
          size_t j = i;
          while (depth > 0) {
            const size_t a = lo.find("<" + name, j), b = lo.find("</" + name, j);
            if (b == std::string::npos) { j = to; break; }
            if (a != std::string::npos && a < b) ++depth, j = a + 1;
            else --depth, j = b + 1;
          }
          const size_t c = lo.find('>', j);
          i = c == std::string::npos ? to : std::min(c + 1, to);
          o += '\n';
          continue;
        }
      }
      bool bl = false;
      for (const char* t : block) bl = bl || name == t;
      if (name.size() == 2 && name[0] == 'h' && name[1] >= '1' && name[1] <= '6') {
        o += close ? "\n" : "\n\n" + std::string(name[1] - '0', '#') + " ";
      } else if (name == "li") {
        if (!close) o += "\n- ";
      } else if (name == "p" || name == "pre" || name == "blockquote" || name == "table") {
        o += "\n\n";
      } else if (bl) {
        o += '\n';
      } else if (name == "td" || name == "th") {
        o += close ? "" : " ";
      }
    }
    return tidy(decode_entities(o));
  };
  for (const char* tag : {"<main", "<article"}) {
    const size_t a = lo.find(tag);
    if (a == std::string::npos) continue;
    const std::string closing = std::string("</") + (tag + 1);
    const size_t b = lo.rfind(closing);
    if (b == std::string::npos || b < a) continue;
    std::string t = convert(a, b);
    if (t.size() >= 400) return t;
  }
  return convert(0, lo.size());
}

inline std::string b64url_decode(const std::string& s) {
  std::string o;
  uint32_t acc = 0;
  int bits = 0;
  for (char c : s) {
    int v = c >= 'A' && c <= 'Z' ? c - 'A' : c >= 'a' && c <= 'z' ? c - 'a' + 26 : c >= '0' && c <= '9' ? c - '0' + 52
          : (c == '-' || c == '+') ? 62 : (c == '_' || c == '/') ? 63 : -1;
    if (v < 0) break;
    acc = acc << 6 | (uint32_t)v, bits += 6;
    if (bits >= 8) bits -= 8, o += (char)(acc >> bits & 255);
  }
  return o;
}

// The text between the first '>' after `from` and `end_tag`, tags stripped
inline std::string inner_text(const std::string& s, size_t from, const char* end_tag) {
  const size_t a = s.find('>', from);
  if (a == std::string::npos) return {};
  const size_t b = s.find(end_tag, a);
  return b == std::string::npos ? std::string() : strip_tags(s.substr(a + 1, b - a - 1));
}

inline std::vector<Result> parse_brave(const std::string& html, size_t n) {
  std::vector<Result> rs;
  const std::string mark = "data-type=\"web\"";
  size_t p = html.find(mark);
  while (p != std::string::npos && rs.size() < n) {
    const size_t q = html.find(mark, p + mark.size());
    const std::string seg = html.substr(p, (q == std::string::npos ? html.size() : q) - p);
    p = q;
    Result r;
    const size_t a = seg.find("<a href=\"");
    if (a == std::string::npos) continue;
    r.url = decode_entities(seg.substr(a + 9, seg.find('"', a + 9) - a - 9));
    const size_t t = seg.find("class=\"title");
    if (t != std::string::npos) r.title = inner_text(seg, t, "</div>");
    size_t s = seg.find("class=\"content");
    if (s == std::string::npos) s = seg.find("snippet-description");
    if (s != std::string::npos) r.snippet = inner_text(seg, s, "</div>");
    if (r.url.rfind("http", 0) == 0 && !r.title.empty()) rs.push_back(r);
  }
  return rs;
}

inline std::vector<Result> parse_bing(const std::string& html, size_t n) {
  std::vector<Result> rs;
  const std::string mark = "class=\"b_algo\"";
  size_t p = html.find(mark);
  while (p != std::string::npos && rs.size() < n) {
    const size_t q = html.find(mark, p + mark.size());
    const std::string seg = html.substr(p, (q == std::string::npos ? html.size() : q) - p);
    p = q;
    Result r;
    const size_t h = seg.find("<h2");
    const size_t a = h == std::string::npos ? h : seg.find("href=\"", h);
    if (a == std::string::npos) continue;
    std::string href = decode_entities(seg.substr(a + 6, seg.find('"', a + 6) - a - 6));
    const size_t u = href.find("&u=a1");  // bing.com/ck/a?...&u=a1<base64url of the target>&...
    if (href.find("bing.com/ck/a") != std::string::npos && u != std::string::npos)
      href = b64url_decode(href.substr(u + 5, href.find('&', u + 5) - u - 5));
    r.url = href;
    r.title = inner_text(seg, a, "</a>");
    const size_t s = seg.find("<p", a);
    if (s != std::string::npos) r.snippet = inner_text(seg, s, "</p>");
    if (r.url.rfind("http", 0) == 0 && !r.title.empty()) rs.push_back(r);
  }
  return rs;
}

// Search the web: Brave Search, then Bing when Brave answers with nothing usable (rate limit, captcha). engine:
// "auto", "brave" or "bing". *error: why an engine was skipped (with no results: why every one failed).
inline std::vector<Result> search(const std::string& query, size_t n, const std::string& engine, std::string* used, std::string* error) {
  std::vector<Result> rs;
  std::string err;
  if (engine != "bing") {
    const std::string u = "https://search.brave.com/search?q=" + url_encode(query) + "&source=web";
    Page pg = get(u, 4u << 20, 12000);
    if (pg.status == 429) Sleep(1500), pg = get(u, 4u << 20, 12000);  // (a short burst limit)
    if (pg.error.empty() && pg.status == 200) rs = parse_brave(pg.body, n);
    else err = "brave: " + (pg.error.empty() ? "HTTP " + std::to_string(pg.status) : pg.error);
    if (!rs.empty()) { *used = "brave"; return rs; }
    if (err.empty()) err = "brave: no results" + std::string(pg.body.find("captcha") != std::string::npos ? " (captcha)" : "");
  }
  if (engine != "brave") {
    const Page pg = get("https://www.bing.com/search?q=" + url_encode(query), 4u << 20, 12000);
    if (pg.error.empty() && pg.status == 200) rs = parse_bing(pg.body, n);
    if (!rs.empty()) { *used = "bing"; *error = err; return rs; }
    err += std::string(err.empty() ? "" : "; ") + "bing: " + (!pg.error.empty() ? pg.error : pg.status != 200 ? "HTTP " + std::to_string(pg.status) : "no results");
  }
  *error = err;
  return rs;
}

}  // namespace webt
