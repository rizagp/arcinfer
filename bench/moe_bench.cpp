// Times the single-token MoE expert kernels (zero/kernels/decode.cl moe_gateup_q / moe_down_q, -DMOE_Q) on the real experts
// of a standard-type MoE GGUF: each step one token with NU random experts of a layer, layers in turn, so the weights
// come from memory as in token generation. Reports the effective weight bandwidth.
// usage: moe_bench model.gguf [first_layer=0] [layers=8] [steps=200] [gateup_defs] [down_defs] [kernel_file]
//   e.g. moe_bench 35b.gguf 0 8 200 "-DGROWS=1" "-DGROWS=4" zero/kernels/decode.cl
#include "../src/clrt.h"
#include "../src/quants.h"
#include <chrono>
#include <cmath>
#include <random>

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: moe_bench model.gguf [first] [layers] [steps] [gu_defs] [dn_defs] [file]\n"); return 1; }
  GgufFile g(argv[1]);
  const int l0 = argc > 2 ? atoi(argv[2]) : 0, nl = argc > 3 ? atoi(argv[3]) : 8, steps = argc > 4 ? atoi(argv[4]) : 200;
  const std::string gu_defs = argc > 5 ? argv[5] : "-DGROWS=1", dn_defs = argc > 6 ? argv[6] : "-DGROWS=4";
  const std::string file = argc > 7 ? argv[7] : "zero/kernels/decode.cl";
  const std::string arch = g.get_str("general.architecture");
  const int NE = (int)g.get_int(arch + ".expert_count"), NU = (int)g.get_int(arch + ".expert_used_count");
  struct L { uint32_t tg, td; cl_mem g, u, d; int F, K, Fp; };
  std::vector<L> ls;
  ClRt cl;
  if (!cl.init()) return 1;
  auto upload = [&](const GgufTensor& t, int N, int K0) {  // all experts, rows padded to whole superblocks
    const int K = padded_k(K0);
    int blk = 0, bytes = 0;
    ggml_block_info(t.type, blk, bytes);
    const size_t rows = (size_t)N * NE, rb0 = (size_t)K0 / blk * bytes, rbp = (size_t)K / blk * bytes;
    std::vector<uint8_t> src(rows * rbp);
    if (K != K0) pad_small_block_rows(t.type, t.data, (int)rows, K0, src.data());
    else memcpy(src.data(), t.data, rows * rb0);
    const size_t bb = gguf_sb_bytes(t.type), bp = raw_padded_sb_bytes(t.type), nsb = rows * (K / 256);
    std::vector<uint8_t> pad(nsb * bp + 65536, 0);
    for (size_t i = 0; i < nsb; ++i) gpu_raw_sb(t.type, src.data() + i * bb, &pad[i * bp]);
    return cl.buffer(CL_MEM_READ_ONLY, pad.size(), pad.data());
  };
  for (int l = l0; l < l0 + nl; ++l) {
    const std::string p = "blk." + std::to_string(l) + ".";
    const GgufTensor *tg = g.find(p + "ffn_gate_exps.weight"), *tu = g.find(p + "ffn_up_exps.weight"),
                     *td = g.find(p + "ffn_down_exps.weight");
    if (!tg || !tu || !td) continue;
    L x;
    x.tg = tg->type;
    x.td = td->type;
    x.F = (int)tg->dims[1];
    x.K = (int)tg->dims[0];
    x.Fp = padded_k(x.F);
    x.g = upload(*tg, x.F, x.K);
    x.u = upload(*tu, x.F, x.K);
    x.d = upload(*td, x.K, x.F);
    ls.push_back(x);
  }
  if (ls.empty()) { fprintf(stderr, "no expert layers\n"); return 1; }
  const int E = ls[0].K, F = ls[0].F, Fp = ls[0].Fp, M = 16;
  printf("%s: %zu layers, %d experts (%d used), F %d (%d), E %d; gate/up %s, down %s\n", arch.c_str(), ls.size(), NE, NU, F,
         Fp, E, ggml_type_name(ls[0].tg), ggml_type_name(ls[0].td));
  // activations of one token (int8 per 32 + scales), the expert slots' h (quantized per 32) and outputs
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<int8_t> aq((size_t)M * E);
  std::vector<float> asc((size_t)(E / 32) * M, 0.0f);
  for (auto& v : aq) v = (int8_t)std::clamp((int)std::lrint(nd(rng) * 40), -127, 127);
  for (int b = 0; b < E / 32; ++b) asc[(size_t)b * M] = 0.01f;
  std::vector<int8_t> hq((size_t)NU * Fp);
  for (auto& v : hq) v = (int8_t)std::clamp((int)std::lrint(nd(rng) * 40), -127, 127);
  std::vector<float> hsc((size_t)(Fp / 32) * NU, 0.01f), swt(NU, 0.1f);
  auto tabs = build_iq_tables();
  cl_mem d_tabs = cl.buffer(CL_MEM_READ_ONLY, tabs.size() * 4, tabs.data());
  cl_mem d_aq = cl.buffer(CL_MEM_READ_ONLY, aq.size(), aq.data()), d_asc = cl.buffer(CL_MEM_READ_ONLY, asc.size() * 4, asc.data());
  cl_mem d_hq = cl.buffer(CL_MEM_READ_ONLY, hq.size(), hq.data()), d_hsc = cl.buffer(CL_MEM_READ_ONLY, hsc.size() * 4, hsc.data());
  cl_mem d_swt = cl.buffer(CL_MEM_READ_ONLY, swt.size() * 4, swt.data());
  cl_mem d_h = cl.buffer(CL_MEM_READ_WRITE, (size_t)NU * Fp * 4), d_y = cl.buffer(CL_MEM_READ_WRITE, (size_t)NU * E * 4);
  std::vector<int> stok(NU, 0);
  cl_mem d_stok = cl.buffer(CL_MEM_READ_ONLY, NU * 4, stok.data());
  // tiles per step: NU one-slot tiles (expert, slot, 1), random experts
  const int nsteps = steps;
  std::vector<cl_mem> d_tiles(nsteps);
  for (int s = 0; s < nsteps; ++s) {
    std::vector<int> t(1 + 3 * NU);
    t[0] = NU;
    std::vector<int> ids(NE);
    for (int i = 0; i < NE; ++i) ids[i] = i;
    std::shuffle(ids.begin(), ids.end(), rng);
    for (int k = 0; k < NU; ++k) { t[1 + 3 * k] = ids[k]; t[2 + 3 * k] = k; t[3 + 3 * k] = 1; }
    d_tiles[s] = cl.buffer(CL_MEM_READ_ONLY, t.size() * 4, t.data());
  }
  const std::string src = read_file(file);
  auto build = [&](uint32_t type, const std::string& defs, const char* kn, bool gu) {
    std::string o = "-cl-std=CL3.0 -cl-mad-enable -DMOE_Q -DQT=" + std::to_string(type) + " -DTM=1 " + defs;
    if (gu && Fp != F) o += " -DHLD=" + std::to_string(Fp);
    if (!gu && Fp != F) o += " -DFV=" + std::to_string(F);
    return cl.kernel(cl.build(src, o), kn);
  };
  int grg = 1, grd = 4;
  if (auto q = gu_defs.find("-DGROWS="); q != std::string::npos) grg = atoi(gu_defs.c_str() + q + 8);
  if (auto q = dn_defs.find("-DGROWS="); q != std::string::npos) grd = atoi(dn_defs.c_str() + q + 8);
  std::vector<cl_kernel> kg(ls.size()), kd(ls.size());
  for (size_t i = 0; i < ls.size(); ++i) {
    kg[i] = build(ls[i].tg, gu_defs, "moe_gateup_q", true);
    kd[i] = build(ls[i].td, dn_defs, "moe_down_q", false);
  }
  double bytes_g = 0, bytes_d = 0;
  for (auto& x : ls) {
    bytes_g += 2.0 * NU * F * (E / 256) * gguf_sb_bytes(x.tg);
    bytes_d += 1.0 * NU * E * (Fp / 256) * gguf_sb_bytes(x.td);
  }
  bytes_g /= ls.size();
  bytes_d /= ls.size();
  for (int pass = 0; pass < 2; ++pass) {  // pass 0: warm-up (JIT, page-in)
    double tg = 0, td = 0;
    for (int s = 0; s < nsteps; ++s) {
      const size_t li = s % ls.size();
      L& x = ls[li];
      int FF = F, KK = E, MM = M;
      set_args(kg[li], 0, x.g, x.u, FF, KK, d_tabs, d_aq, d_asc, MM, d_tiles[s], d_stok, d_h);
      size_t gw[2] = {((size_t)(F + grg - 1) / grg * 16 + 255) / 256 * 256, (size_t)NU}, lw[2] = {256, 1};
      cl_event e1, e2;
      CL_CHECK(clEnqueueNDRangeKernel(cl.queue, kg[li], 2, nullptr, gw, lw, 0, nullptr, &e1));
      int NN = E, KD = Fp, S = NU;
      set_args(kd[li], 0, x.d, NN, KD, d_tabs, d_hq, d_hsc, S, d_tiles[s], d_swt, d_y);
      size_t gd[2] = {((size_t)(E + grd - 1) / grd * 16 + 255) / 256 * 256, (size_t)NU};
      CL_CHECK(clEnqueueNDRangeKernel(cl.queue, kd[li], 2, nullptr, gd, lw, 0, nullptr, &e2));
      clFinish(cl.queue);
      tg += ClRt::event_ms(e1);
      td += ClRt::event_ms(e2);
      clReleaseEvent(e1);
      clReleaseEvent(e2);
    }
    if (pass == 1)
      printf("gate/up %s: %.1f us per layer, %.1f GB/s | down %s: %.1f us, %.1f GB/s\n", gu_defs.c_str(),
             tg * 1000 / nsteps, bytes_g / (tg / nsteps) / 1e6, dn_defs.c_str(), td * 1000 / nsteps,
             bytes_d / (td / nsteps) / 1e6);
  }
  return 0;
}
