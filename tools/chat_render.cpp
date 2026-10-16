// Renders a chat request with src/chat.h (Qwen, native) or with minja (--jinja template.jinja), for comparison
// against the GGUF Jinja template rendered by jinja2.
// usage: chat_render case.json [--jinja template.jinja]   (case: {messages, tools?, enable_thinking?, reasoning_effort?, q38})
#include "../src/chat.h"
#include "../src/chat_jinja.h"
#include <cstdio>
#include <fcntl.h>
#include <io.h>
#include <fstream>
#include <sstream>

int main(int argc, char** argv) {
  if (argc < 2) return 1;
  std::ifstream f(argv[1], std::ios::binary);
  std::stringstream ss;
  ss << f.rdbuf();
  const Json c = Json::parse(ss.str());
  ChatOptions o;
  if (c["enable_thinking"].is_bool()) o.enable_thinking = c["enable_thinking"].as_bool();
  o.reasoning_effort = c["reasoning_effort"].str();
  std::string out;
  if (argc > 3 && std::string(argv[2]) == "--jinja") {
    std::ifstream tf(argv[3], std::ios::binary);
    std::stringstream ts;
    ts << tf.rdbuf();
    try {
      JinjaChat j(ts.str(), "<bos>", "<eos>");
      out = j.render(c["messages"], c["tools"], true, o.enable_thinking, o.reasoning_effort);
    } catch (const std::exception& e) {
      fprintf(stderr, "template error: %s\n", e.what());
      return 2;
    }
  } else {
    out = chat::render(c["messages"], c["tools"], o, c["q38"].as_bool());
  }
  _setmode(_fileno(stdout), _O_BINARY);
  fwrite(out.data(), 1, out.size(), stdout);
  return 0;
}
