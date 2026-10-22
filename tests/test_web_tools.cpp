// CPU-only, offline checks of tools/web_tools.h: HTML to text, search result parsing (Brave and Bing page shapes),
// the address filter and redirect resolution. With --online it also runs one live search and one fetch.
#include "../tools/web_tools.h"
#include <cstdio>
#include <string>

static int fails = 0;
static void check(const char* name, bool ok, const std::string& got = "") {
  printf("%s %s%s%s\n", ok ? "ok  " : "FAIL", name, ok || got.empty() ? "" : "  got: ", ok ? "" : got.c_str());
  fails += !ok;
}

static bool addr_public(const char* ip) {
  sockaddr_storage ss{};
  if (strchr(ip, ':')) {
    auto* a = (sockaddr_in6*)&ss;
    a->sin6_family = AF_INET6;
    inet_pton(AF_INET6, ip, &a->sin6_addr);
  } else {
    auto* a = (sockaddr_in*)&ss;
    a->sin_family = AF_INET;
    inet_pton(AF_INET, ip, &a->sin_addr);
  }
  return webt::public_addr((const sockaddr*)&ss);
}

int main(int argc, char** argv) {
  using namespace webt;
  {
    std::string title;
    const std::string t = html_to_text(
        "<html><head><title>A &amp; B</title><style>p{x}</style></head><body><nav>menu</nav>"
        "<h1>Head</h1><p>One &lt;two&gt;&nbsp;three&#33; &#x263A;</p><script>var a='<p>';</script>"
        "<ul><li>x</li><li>y</li></ul><!-- c --><footer>foot</footer></body></html>", &title);
    check("title", title == "A & B", title);
    check("text", t == "# Head\n\nOne <two> three! \xe2\x98\xba\n\n- x\n- y", t);
  }
  {
    std::string body(500, 'w');
    const std::string t = html_to_text("<body><div>side</div><main><p>" + body + "</p></main></body>");
    check("main only", t == body, t.substr(0, 40));
  }
  {
    const std::string brave =
        "<div data-type=\"web\" x><a href=\"https://a.example/x?a=1&amp;b=2\" class=\"l1\"><div class=\"title t\" title=\"T\">"
        "Title <b>A</b></div></a><div class=\"generic-snippet\"><div class=\"content c\">Snip &amp; more</div></div></div>"
        "<div data-type=\"web\"><a href=\"https://b.example/\"><div class=\"title\">B</div></a></div>";
    const auto r = parse_brave(brave, 10);
    check("brave count", r.size() == 2, std::to_string(r.size()));
    check("brave first", r.size() == 2 && r[0].url == "https://a.example/x?a=1&b=2" && r[0].title == "Title A" && r[0].snippet == "Snip & more",
          r.empty() ? "" : r[0].url + " | " + r[0].title + " | " + r[0].snippet);
  }
  {
    // u=a1 + base64url("https://www.intel.com/x")
    const std::string bing =
        "<li class=\"b_algo\"><h2><a href=\"https://www.bing.com/ck/a?!&amp;&amp;p=1&amp;u=a1aHR0cHM6Ly93d3cuaW50ZWwuY29tL3g&amp;ntb=1\">"
        "Intel <strong>X</strong></a></h2><div class=\"b_caption\"><p class=\"b_lineclamp2\">The snippet.</p></div></li>"
        "<li class=\"b_algo\"><h2><a href=\"https://direct.example/\">D</a></h2></li>";
    const auto r = parse_bing(bing, 10);
    check("bing count", r.size() == 2, std::to_string(r.size()));
    check("bing redirect decoded", r.size() == 2 && r[0].url == "https://www.intel.com/x" && r[0].title == "Intel X" && r[0].snippet == "The snippet.",
          r.empty() ? "" : r[0].url + " | " + r[0].title + " | " + r[0].snippet);
    check("bing direct", r.size() == 2 && r[1].url == "https://direct.example/");
  }
  check("public 8.8.8.8", addr_public("8.8.8.8"));
  check("public 2606:4700::1111", addr_public("2606:4700::1111"));
  for (const char* ip : {"127.0.0.1", "10.1.2.3", "172.16.0.1", "172.31.255.255", "192.168.1.1", "169.254.1.1", "100.64.0.1",
                         "0.0.0.0", "224.0.0.1", "::1", "::", "fe80::1", "fd00::1", "::ffff:192.168.1.1", "ff02::1"})
    check((std::string("private ") + ip).c_str(), !addr_public(ip));
  check("public 172.32.0.1", addr_public("172.32.0.1"));
  check("localhost refused", !check_host("localhost").empty());
  check("127.0.0.1 refused", !check_host("127.0.0.1").empty());
  check("resolve absolute", resolve_url("https://a.com/x/y", "http://b.com/z") == "http://b.com/z");
  check("resolve scheme-relative", resolve_url("https://a.com/x/y", "//b.com/z") == "https://b.com/z");
  check("resolve path", resolve_url("https://a.com/x/y?q", "/z") == "https://a.com/z");
  check("resolve relative", resolve_url("https://a.com/x/y?q=/w", "z") == "https://a.com/x/z", resolve_url("https://a.com/x/y?q=/w", "z"));
  check("local fetch refused", !get("http://127.0.0.1:8080/").error.empty());
  check("non-http refused", !get("file:///C:/Windows/win.ini").error.empty());

  if (argc > 1 && std::string(argv[1]) == "--online") {
    std::string used, err;
    const auto r = search("intel arc level zero", 5, "auto", &used, &err);
    printf("search (%s): %zu results [%s]\n", used.c_str(), r.size(), err.c_str());
    for (auto& x : r) printf("  %s\n    %s\n    %.100s\n", x.title.c_str(), x.url.c_str(), x.snippet.c_str());
    const auto b = search("intel arc level zero", 5, "bing", &used, &err);
    printf("bing: %zu results %s\n", b.size(), err.c_str());
    const Page p = get("https://en.wikipedia.org/wiki/Intel_Arc");
    std::string title;
    const std::string t = html_to_text(p.body, &title);
    printf("fetch: HTTP %d, %zu bytes -> %zu chars, title '%s'\n%.600s\n", p.status, p.body.size(), t.size(), title.c_str(), t.c_str());
    fails += r.empty() || b.empty() || t.size() < 1000;
  }
  printf(fails ? "test_web_tools: %d failed\n" : "test_web_tools: all passed\n", fails);
  return fails ? 1 : 0;
}
