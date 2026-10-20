// arcinferzero: the Gemma 4 vision encoder (gemma4v) on one image, for tools/gemma4v_ref.py (the f32 reference):
// writes <image>.g4v.bin (n_tok x n_out f32 embeddings) and <image>.g4v.rgb (the resized pixels).
// usage: vision_testz model.gguf mmproj.gguf image.(png|jpg) [max_tokens]
#include "engine.h"
#include "vision_gemma4v.h"
#include <cstdio>

int main(int argc, char** argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: vision_testz model.gguf mmproj.gguf image.(png|jpg) [max_tokens]\n");
    return 1;
  }
  try {
    EngineOptions eo;
    eo.ctx = 4096;
    Engine eng(argv[1], eo);
    VisionGemma4V vis(argv[2], eng);
    if (argc > 4) vis.min_tokens = vis.max_tokens = atoi(argv[4]);
    std::vector<uint8_t> bytes;
    {
      FILE* f = fopen(argv[3], "rb");
      if (!f) throw std::runtime_error("cannot open image");
      fseek(f, 0, SEEK_END);
      bytes.resize(ftell(f));
      fseek(f, 0, SEEK_SET);
      if (fread(bytes.data(), 1, bytes.size(), f) != bytes.size()) throw std::runtime_error("cannot read image");
      fclose(f);
    }
    const ImageRGB im = img::decode(bytes);
    int n_tok, W, H;
    vis.plan(im.w, im.h, W, H);
    std::vector<float> gpu = vis.encode(im, n_tok);
    gpu = vis.encode(im, n_tok);  // (the second run: kernels built)
    printf("image %dx%d -> %dx%d, %d tokens, GPU encoder %.1f ms\n", im.w, im.h, W, H, n_tok, vis.encode_ms);
    const ImageRGB r = img::resize_bicubic(im, W, H);
    FILE* fo = fopen((std::string(argv[3]) + ".g4v.bin").c_str(), "wb");
    int hdr[3] = {n_tok, vis.n_out, 0};
    fwrite(hdr, 4, 3, fo);
    fwrite(gpu.data(), 4, gpu.size(), fo);
    fclose(fo);
    fo = fopen((std::string(argv[3]) + ".g4v.rgb").c_str(), "wb");
    int dim[2] = {W, H};
    fwrite(dim, 4, 2, fo);
    fwrite(r.px.data(), 1, r.px.size(), fo);
    fclose(fo);
  } catch (const std::exception& e) {
    fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
