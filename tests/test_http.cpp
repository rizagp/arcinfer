// CPU-only check of the HTTP server (src/http.h) against malformed requests: each must get its 400 / 413 (or a
// closed connection) and the server must keep answering afterwards. usage: test_http [port] (default 8093)
#include "../src/http.h"
#include <cstdio>
#include <string>

static std::string exchange(int port, const std::string& req) {
  SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_port = htons((u_short)port);
  inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
  if (connect(s, (sockaddr*)&a, sizeof a) != 0) { closesocket(s); return "<no connection>"; }
  DWORD tmo = 5000;
  setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tmo, sizeof tmo);
  ::send(s, req.data(), (int)req.size(), 0);
  shutdown(s, SD_SEND);
  std::string out;
  char buf[4096];
  int n;
  while ((n = recv(s, buf, sizeof buf, 0)) > 0) out.append(buf, n);
  closesocket(s);
  return out;
}

int main(int argc, char** argv) {
  const int port = argc > 1 ? atoi(argv[1]) : 8093;
  HttpServer srv;
  srv.handler = [](const HttpRequest& rq, HttpResponse& rs) {
    if (rq.path == "/throw") throw 42;  // (not a std::exception)
    rs.send(200, "text/plain", "ok:" + std::to_string(rq.body.size()));
  };
  if (!srv.listen("127.0.0.1", port)) { fprintf(stderr, "cannot listen on %d\n", port); return 1; }
  std::thread([&] { srv.run(); }).detach();

  struct Case { const char* name; std::string req, expect; };
  const std::string big = std::to_string((size_t)1 << 40);
  const Case cases[] = {
      {"plain POST", "POST /x HTTP/1.1\r\nContent-Length: 3\r\n\r\nabc", "200 OK"},
      {"Content-Length abc", "POST /x HTTP/1.1\r\nContent-Length: abc\r\n\r\n", "400"},
      {"Content-Length -1", "POST /x HTTP/1.1\r\nContent-Length: -1\r\n\r\n", "400"},
      {"Content-Length overflow", "POST /x HTTP/1.1\r\nContent-Length: 99999999999999999999999\r\n\r\n", "400"},
      {"Content-Length 1 TB", "POST /x HTTP/1.1\r\nContent-Length: " + big + "\r\n\r\n", "413"},
      {"chunked", "POST /x HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n3\r\nabc\r\n2;ext=1\r\nde\r\n0\r\n\r\n", "ok:5"},
      {"chunk size zz", "POST /x HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\nzz\r\nabc\r\n0\r\n\r\n", "400"},
      {"chunk 1 TB", "POST /x HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n10000000000\r\n", "413"},
      {"request line without spaces", "GARBAGE\r\n\r\n", "400"},
      {"handler throws a non-exception", "GET /throw HTTP/1.1\r\n\r\n", ""},
      {"still alive", "GET /x HTTP/1.1\r\n\r\n", "200 OK"},
  };
  int fails = 0;
  for (const Case& c : cases) {
    const std::string r = exchange(port, c.req);
    const bool ok = c.expect.empty() ? r.find("200 OK") == std::string::npos : r.find(c.expect) != std::string::npos;
    printf("%-34s %s\n", c.name, ok ? "ok" : ("FAIL: " + r.substr(0, 60)).c_str());
    fails += !ok;
  }
  printf(fails ? "test_http: %d failed\n" : "test_http: all passed\n", fails);
  return fails ? 1 : 0;
}
