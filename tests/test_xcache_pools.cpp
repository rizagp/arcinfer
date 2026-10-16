// The expert cache's pool grouping (zero/xcache.h, xpools), host only: Qwen3.8 Flash Next's and Qwen3.6 35B's record
// sizes, the cache at a few sizes, one pool of the largest record (before) against xpools::choose (after).
// Build: .\build.ps1 tests\test_xcache_pools.cpp   Run: .\bin\test_xcache_pools.exe   (prints ALL OK)
#define XCACHE_POOLS_ONLY
#include "../zero/xcache.h"
#include <cstdio>
#include <string>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)
static const double GiB = 1073741824.0;

// one expert's record: gate + up (640 rows x 10 superblocks each) and down (2560 rows x 3 superblocks), GPU raw
// superblock bytes, rounded up to 4 KB
static uint64_t flash_rec(int gu_sb, int down_sb) {
  const uint64_t s = 2ull * 640 * 10 * gu_sb + 2560ull * 3 * down_sb;
  return (s + 4095) / 4096 * 4096;
}

// mix: what one pool of the largest record (before) and choose (after) make of a cache of `bytes`
static void report(const char* name, const std::vector<uint64_t>& lrec, int NE, uint64_t bytes) {
  const uint64_t real = xpools::real_bytes(lrec, NE);
  auto show = [&](const char* tag, const xpools::Plan& P) {
    // padding: slot bytes the cached records leave unused (each pool's slots filled in its layers' proportion)
    double pad = 0, padded_all = 0;
    for (size_t g = 0; g < P.rec.size(); ++g) {
      uint64_t sum = 0;
      for (size_t l = 0; l < lrec.size(); ++l) if (P.pool_of[l] == (int)g) sum += lrec[l];
      pad += (double)P.n[g] * (P.rec[g] - (double)sum / P.layers[g]);
      padded_all += (double)P.rec[g] * NE * P.layers[g];
    }
    printf("  %-6s %zu pools (", tag, P.rec.size());
    for (size_t g = 0; g < P.rec.size(); ++g) printf("%s%.3f MB x%d: %d", g ? ", " : "", P.rec[g] / 1e6, P.layers[g], P.n[g]);
    printf(")\n         %5d slots (%.1f %% of experts), %.2f GiB used, padding %.2f GiB (%.1f %%), all experts padded %.2f GiB\n",
           P.slots, 100.0 * P.slots / (NE * (double)(P.pool_of.size() - std::count(P.pool_of.begin(), P.pool_of.end(), -1))),
           P.bytes / GiB, pad / GiB, P.bytes ? 100.0 * pad / P.bytes : 0.0, padded_all / GiB);
  };
  printf("%s, cache %.2f GiB (experts %.2f GiB):\n", name, bytes / GiB, real / GiB);
  const auto d = xpools::sizes(lrec);
  xpools::Plan one = xpools::group(lrec, std::vector<uint64_t>(1, d.back()));
  xpools::size_shares(one, NE, bytes);  // (as set_cache sized the one pool before)
  show("before", one);
  const xpools::Plan P = xpools::choose(lrec, NE, bytes);
  show("after", P);
  // invariants: a layer's pool record holds its record, a layer per pool, at most all, within the cache
  CHECK(P.slots >= one.slots);
  CHECK(P.bytes <= std::max(bytes, xpools::min_bytes(lrec, NE)));
  for (size_t l = 0; l < lrec.size(); ++l) {
    CHECK((lrec[l] == 0) == (P.pool_of[l] < 0));
    if (lrec[l]) CHECK(P.rec[P.pool_of[l]] >= lrec[l]);
  }
  for (size_t g = 0; g < P.rec.size(); ++g) CHECK(P.n[g] >= NE && P.n[g] <= NE * P.layers[g]);
}

int main() {
  // Qwen3.8 Flash Next (512 experts, 48 MoE layers + the MTP layer without experts), the type mix of
  // Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S per layer: gate/up A,a IQ2_S (88 B), B,b IQ3_XXS (104), C,c IQ3_S (112),
  // D IQ4_XS (136); down upper case IQ4_NL (144), lower case Q2_0 (72)
  const std::string mix = "BaAAAAABaAaAbaBABCBAbcBAACACCCbBABACaBBACBBBCCBD-";
  std::vector<uint64_t> flash;
  for (char c : mix) {
    const int gu = c == 'A' || c == 'a' ? 88 : c == 'B' || c == 'b' ? 104 : c == 'C' || c == 'c' ? 112 : 136;
    flash.push_back(c == '-' ? 0 : flash_rec(gu, c >= 'a' ? 72 : 144));
  }
  const int NE = 512;
  CHECK(xpools::sizes(flash).size() == 7);
  CHECK(xpools::real_bytes(flash, NE) == 55983476736ull - 4096);  // (the expert file less its 4 KB header)
  for (double gb : {6.0, 7.94, 8.0, 10.0}) report("Flash Next", flash, NE, (uint64_t)(gb * GiB));
  // the whole cache: one pool per size, every expert
  {
    const xpools::Plan P = xpools::choose(flash, NE, xpools::real_bytes(flash, NE));
    CHECK(P.rec.size() == 7 && P.slots == 48 * NE && P.bytes == xpools::real_bytes(flash, NE));
  }
  // too small for anything but one pool (the engine never asks for less than min_bytes)
  {
    const xpools::Plan P = xpools::choose(flash, NE, xpools::min_bytes(flash, NE));
    CHECK(P.rec.size() == 1 && P.slots == NE);
  }

  // Qwen3.6 35B A3B UD-Q3_K_M (256 experts, 40 layers + MTP): 4 record sizes
  std::vector<uint64_t> q35(41, 1409024);
  q35[34] = q35[38] = 1736704;
  q35[39] = 1998848;
  q35[40] = 1507328;
  for (double gb : {3.0, 6.0, 10.0}) report("Qwen3.6 35B", q35, 256, (uint64_t)(gb * GiB));

  // one or two sizes: one pool per size, split by their share of the bytes (the sizing before, unchanged)
  {
    std::vector<uint64_t> two(41, 1409024);
    two[40] = 1507328;
    const uint64_t bytes = (uint64_t)(6 * GiB), all = xpools::real_bytes(two, 256);
    const xpools::Plan P = xpools::choose(two, 256, bytes);
    CHECK(P.rec.size() == 2 && P.rec[0] == 1409024 && P.rec[1] == 1507328 && P.pool_of[40] == 1 && P.pool_of[0] == 0);
    const uint64_t m0 = 1409024ull * 256 * 40, m1 = 1507328ull * 256;
    CHECK(P.n[0] == std::min((int)(m0 / 1409024), std::max(256, (int)((double)bytes * m0 / all / 1409024))));
    CHECK(P.n[1] == std::min((int)(m1 / 1507328), std::max(256, (int)((double)bytes * m1 / all / 1507328))));
    CHECK(xpools::min_bytes(two, 256) == (1409024ull + 1507328) * 256);
  }
  printf(fails ? "%d FAILED\n" : "ALL OK\n", fails);
  return fails ? 1 : 0;
}
