// CPU-only checks of src/json.h on input a client can send: numbers must be whole tokens, \u escapes four hex
// digits, a repeated key keeps its last value (in linear time), non-finite numbers serialize as null and as_int
// never converts an out-of-range double.
#include "../src/json.h"
#include <chrono>
#include <cstdio>
#include <string>

static int fails = 0;
static void check(const char* name, bool ok) {
  printf("%s %s\n", ok ? "ok  " : "FAIL", name);
  fails += !ok;
}
static bool parses(const std::string& s) {
  Json j;
  return Json::try_parse(s, j);
}

int main() {
  check("number", parses("1.5") && parses("-2e3") && parses("[0, 1, 2]"));
  check("1.0.0 refused", !parses("1.0.0"));
  check("1e refused", !parses("1e"));
  check("-- refused", !parses("--1"));
  check("1e999 refused (not finite)", !parses("1e999"));
  check("\\u0041 = A", [] { Json j; return Json::try_parse("\"\\u0041\"", j) && j.as_string() == "A"; }());
  check("\\u12 refused", !parses("\"\\u12\""));
  check("\\uZZZZ refused", !parses("\"\\uZZZZ\""));
  {
    Json j;
    const bool ok = Json::try_parse(R"({"a": 1, "b": 2, "a": 3})", j);
    check("repeated key: last wins, one entry", ok && j["a"].as_int() == 3 && j.size() == 2 && j.dump() == R"({"a":3,"b":2})");
  }
  {
    std::string big = "{";
    for (int i = 0; i < 200000; ++i) big += (i ? ",\"k" : "\"k") + std::to_string(i % 1000) + "\":" + std::to_string(i);
    big += "}";
    const auto t0 = std::chrono::steady_clock::now();
    Json j;
    const bool ok = Json::try_parse(big, j);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    check("200,000 keys (1,000 distinct) in under 2 s", ok && j.size() == 1000 && j["k5"].as_int() == 199005 && ms < 2000);
  }
  {
    Json o = Json::object();
    o.set("x", Json(std::nan("")));
    o.set("y", Json(1e300));
    check("nan serializes as null", o.dump().find("\"x\":null") != std::string::npos);
    check("as_int of 1e300 is the default", o["y"].as_int(-7) == -7);
  }
  printf(fails ? "test_json: %d failed\n" : "test_json: all passed\n", fails);
  return fails ? 1 : 0;
}
