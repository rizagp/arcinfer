# Antrian optimisasi (mulai 2026-10-12)

Target: generasi token memakai > 90 % bandwidth terukur (~75 GB/dtk, jadi > 67,5 GB/dtk rata-rata per token) untuk
semua model chat di cache Hugging Face, dan vision / audio berfungsi di setiap model yang punya encoder-nya.
Status: ✅ selesai · 🔄 dikerjakan · ⏳ antri · ⚠️ catatan · [~] sebagian

| # | Tugas | Status | Catatan |
|---|---|---|---|
| 1 | Dukungan Q4_0 (Gemma 4 QAT dari Google): prefill, GEMV cepat, GLU gabungan | ✅ | E4B: prefill 788 tok/dtk, generasi 20,9 (drafter 26,8) tok/dtk |
| 2 | Alat ukur: bobot dibaca per token dan efisiensi bandwidth, per model | ✅ | `run.exe` mencetak `weights read: … GB/s (… % of ~75 GB/s)`. Pengukuran awal (unduhan 12B berjalan, angka berfluktuasi): 4B 60–67 %, 9B 72 %, E4B 68 %, 27B 36 % |
| 3 | Deteksi mmproj Google (`gemma-4-E4B-it-mmproj.gguf` tidak diawali `mmproj`) | ✅ | juga: 35B non-GSQ ditandai belum didukung |
| 4 | Gemma 4 12B (Google): uji teks, gambar, audio (gemma4ua) | ✅ | generasi 9,1 tok/dtk (85 % bandwidth), drafter 17–20; gambar dan audio jalan. Diperbaiki: template Google memakai string berdampingan (`"a" "b"`) yang tidak didukung minja → server crash saat start |
| 5 | Efisiensi bandwidth per model (ukur → optimasi yang paling jauh dari target) | 🔄 | tahap 1 selesai: bobot Q8_0 dan IQ2/IQ3 disusun ulang di GPU (blok rata) → GEMV Q8_0 37→77, IQ2_XXS 27→60, IQ3_XXS 41→62 GB/dtk. Sekarang: 4B 79 %, 9B 86 %, E4B 81 %, 12B 82 %, 27B / Swift 72 % (dulu 77/83/76/80/65 %). Sisa: kernel non-bobot (rekurensi DeltaNet, norm, attention), IQ1_M |
| 6 | Vision E4B di GPU (gemma4v: ViT 16 lapis, 768 lebar) | ✅ | error vs referensi f32 0,11–0,17 %; 1120 token 3,2 dtk (attention 2,8 TFLOPS, GEMM 2,2). Diperbaiki juga: input per-layer (PLE) untuk token gambar, token gambar E2B/E4B kausal (llama.cpp) |
| 7 | Audio E4B di GPU (gemma4a: conformer 12 lapis, 1024 lebar) | ✅ | transkripsi tepat kata per kata; 7,2 dtk audio di-encode 226 ms (request pertama ~23 dtk: kompilasi kernel + unggah bobot) |
| 8 | Vision Qwen3.5 9B (encoder 1152 lebar, head 72 dimensi) | ✅ | error 0,38 % vs f32; screenshot 0,75 dtk, gambar 2025 token 8,2 dtk (attention head 72 masih ~40 % di atas ideal: register spill) |
| 9 | Qwen3.6 35B-A3B UD-Q3_K_S: kernel MoE untuk tipe GGUF standar | ✅ | jalan: generasi 18,9 tok/dtk (58 %), prefill 465–480 tok/dtk, konteks otomatis 61K (17 GiB); hasil prefill = token-per-token, jawaban benar |
| 10 | Tes regresi: referensi baru untuk Gemma Google, run_tests.ps1 | ✅ | semua lulus (12 cek); Gemma 12B Google KL 0,0206 |

Tidak termasuk: Qwen3-ASR 1.7B (tidak diperlukan), bge-small (embedding), laya (bukan model chat).
⚠️ Pengukuran kecepatan yang final dilakukan setelah unduhan 12B selesai (CPU dan GPU berbagi daya dan panas).

---

# Fase 2: rencana optimisasi berikutnya (disusun 2026-10-13)

Dasar: profil per kernel tiap model. Ringkasan posisi sekarang:
- Generasi token model dense (4B, 9B, 12B, E4B, 27B) sudah dekat batas praktis: 72–86 % bandwidth; sisa waktu
  di kernel non-bobot. Pengungkit utamanya MTP / drafter (27B: 4,5 → 7,9 tok/dtk), dan itu sudah aktif.
- Prefill model dense hampir seluruhnya GEMM int8 di ~65 % puncak DP4a (batas praktis). DeltaNet hanya 4–9 %.
- Yang masih jauh dari batas: **kernel expert 35B** (prefill 0,04 TOPS, generasi ~41 GB/dtk) dan
  **attention di konteks panjang**.

Urutan dari dampak terbesar per usaha. Setiap tugas: ukur sebelum dan sesudah (best-of-two / bergantian A-B),
hasil harus identik (greedy sama, prefill = token-per-token), lalu run_tests.ps1.

| # | Tugas | Status | Perkiraan hasil | Biaya / risiko |
|---|---|---|---|---|
| F1 | **Prefill expert 35B dengan GEMM int8**: dekode expert ke int8 per 64 expert (`decode_w8`), lalu satu kernel GEMM berkelompok (lane = kolom, aktivasi dibagikan, DP4a) untuk semua tile (expert × token) | ✅ prefill 2048 token **476 → 642 tok/dtk** (+35 %); hasil per layer sama dengan kernel lama (selisih ~1e-7). Dekode expert ke int8 per 64 expert lewat kernel baru `decode_w8_sl` (baris disalin dulu ke memori lokal: IQ3_XXS 1038 → 518 ms, IQ4_XS 433 → 303 ms per chunk), lalu `gemm.cl` mode `MOE_TILES`. Dipakai untuk chunk ≥ 1152 token (dekode semua expert ~1 dtk per chunk; di 1024 token impas). Staging 75 MB | kerja expert 2,7 → ~1,1 dtk per chunk 2048; prefill 35B **476 → ~700–750 tok/dtk**; TTFT prompt panjang turun ~1/3 | beberapa jam; buffer staging ~70 MB → konteks otomatis −~7K token |
| F2 | **Attention prefill konteks panjang** (27B, 9B, 35B): ukur dulu porsinya di 16K / 32K token, lalu perbaiki kernel attention prefill (tile lebih besar, K/V lewat memori lokal, skor int8 + akumulasi f32) | ✅ (sebagian) diukur di prompt 21.590 token: attention 35B 22 % (9,4 dtk), 9B 14 %, 27B 12 % (22 dtk). Perbaikan: 4 head-q per sub-group untuk Qwen bila rasio GQA bisa dibagi 4 (attn_q8): 35B 9,4 → 8,4 dtk, 9B 8,2 → 7,0 dtk, hasil sama; 27B (6 per kv head) tetap. `attn_ps` (dim dibagi) dicoba: lebih lambat. Sisa terbesar 35B di prompt panjang: dekode expert per chunk (~0,85 dtk × 11 chunk); butuh dekode sekali per layer untuk semua chunk (pekerjaan berikutnya) | belum diketahui; 27B di 11K dulu ~6,4 dtk attention, porsinya naik terus seiring konteks | sedang; harus dicek akurasinya di konteks panjang (KL) |
| F3 | **Generasi 35B: kernel expert** (~41 GB/dtk → target ~60): matriks down barisnya pendek (512) → beberapa baris per sub-group dengan reduksi gabungan; gate/up: kurangi overhead per tile | ✅ (sebagian) kernel expert sudah di batas praktis: variasi baris per sub-group ±2 %, down + combine digabung 212 vs 211 ms (tidak dipakai). Gantinya: qkv + z (+ ab) DeltaNet ditumpuk jadi satu GEMV saat generasi (`load_gdn_in`, tanpa salinan ganda): 35B **19,2 → 19,8 tok/dtk** (+3 %), 27B 4,75 → 4,8–5,0; 9B sama (tipe qkv / z beda di banyak layer). Q8_0 lewat `gemv_v` dicoba: lebih lambat | generasi 35B **18,8 → ~20 tok/dtk** (+6–7 %) | sedang; tanpa efek ke memori |
| F4 | **MTP 3 tebakan per langkah** untuk Qwen (27B, 9B, 4B): engine sekarang dibatasi 2 | ✅ 27B kode 7,8 → 9,5 tok/dtk (+21 %), teks +2,5 %; 4B / 9B sama → default otomatis 3 untuk 27B-class, 2 lainnya; konteks 27B 86K → 82K | 27B kode: 7,9 → mungkin ~8,5 tok/dtk; teks biasa lebih kecil | kecil; hasil wajib identik dengan tanpa MTP |
| F5 | **Q8_0 satu salinan dalam susunan GEMM** (35B): GEMV baru yang membaca susunan selang-seling, sehingga dekode Q8_0 per chunk (0,22 dtk) hilang | ⏸️ tidak dikerjakan: setelah kernel dekode baru (`decode_w8_sl`) dekode Q8_0 per chunk 2048 tinggal 211 → 72 ms (~2 % prefill), jadi GEMV baru untuk susunan lain tidak sebanding | prefill 35B +~5 % | sedang; GEMV baru harus secepat yang sekarang |
| F6 | **Vision 35B**: repo Unsloth punya mmproj (encoder Qwen3-VL yang sama dengan 4B/9B); server belum memuat vision untuk model MoE | ✅ berfungsi (2026-10-14, mmproj-BF16 Unsloth): teks di gambar terbaca persis, encode 1.008 token ~3 dtk (pertama kali ~60 dtk: kompilasi kernel), prefill ~400 tok/dtk, generasi ~18 tok/dtk. Tapi encoder (~1,1–1,3 GB) hampir tidak muat bersama 35B di 18,3 GiB: konteks dengan gambar hanya 4.096. Karena itu `35b` tetap tanpa vision (40.960), gambar lewat `35b-vision` saja. Pilihan berikutnya: bobot encoder Q8_0 (hemat ~0,45 GB), encoder dimuat hanya saat ada gambar, atau jatah GPU dinaikkan | fitur baru: 35B bisa membaca gambar | kecil–sedang; memori encoder (~1 GB) mengurangi konteks |
| F7 | (opsional) **Attention vision 9B head 72**: satu baris query dibagi ke dua lane supaya register muat | ⏸️ tidak dikerjakan (opsional; kompiler tetap memakai 128 register untuk kernel ini, hasilnya tidak pasti) | attention vision 9B ~1,6 → ~1,1 dtk per gambar besar | hasil belum pasti (kompiler) |
| F8 | (perlu keputusan) **MTP untuk 35B**: butuh file versi MTP, dukungan Q3_K, expert Q2_K/Q4_K, dan MoE di blok MTP | ✅ `35b-mtp` (file repo MTP): hasil **identik bit** dengan tanpa MTP, generasi **19,7 → 22,3 tok/dtk** (+13 %, 83 % tebakan diterima), prefill −3,4 %, konteks otomatis 40K (varian `35b-mtp-std` 98K). Ditambah: Q3_K di kernel GPU, expert ber-min (Q4_K / Q2_K / Q5_K) di kernel expert, MoE di blok MTP, 1 tebakan default untuk MoE (1 dan 2 sama cepat; hemat satu salinan state). Diperbaiki: attention verifikasi 35B kini lewat kernel yang sama dengan token tunggal (sebelumnya teks MTP menyimpang setelah ~40 token) | generasi 35B ~1,5× | besar; konteks Q3_K_M turun ke ~40–50K |

Sudah dicoba / diukur dan **tidak** dilanjutkan:
- Level Zero (jeda antar kernel cuma ~1,6–2 %), antrian out-of-order + event.
- DeltaNet "chunked" (DeltaNet hanya 4–9 % prefill), state DeltaNet f16 (hemat ~1,5 %, risiko error menumpuk).
- Split-K GEMV untuk matriks kecil (tidak lebih cepat secara total), aturan GEMV per tipe hasil benchmark
  matriks tunggal (tidak lebih cepat di dalam model), gemv_glu untuk 27B (hanya 24/64 lapis cocok, ~0,3 %).

Saran urutan kerja (diperbarui 2026-10-13, file MTP 35B sedang diunduh): **F8 → F1 → F4 → F3 → F2 → F5 → F9 → F6**, lalu F7.

## F8 (prioritas 1): MTP untuk Qwen3.6 35B-A3B, rincian

Tujuan: generasi 35B 18,8 → ~27–30 tok/dtk dengan blok MTP, hasil identik dengan tanpa MTP, prefill tidak turun
lebih dari ~5 %.

| Langkah | Isi | Catatan |
|---|---|---|
| 8.1 | Periksa file yang diunduh: tensor `blk.40.*` (blok MTP, `nextn.*`), tipe expert-nya, ukuran; perkiraan konteks | file repo MTP: lapis utama sama dengan versi non-MTP, blok MTP ±0,5 GB |
| 8.2 | **Tipe Q3_K**: dekoder GPU (`decode32`), referensi CPU, dekode ke int8 untuk prefill, GEMV; tes kernel di `test_gemm` | belum didukung sama sekali; di file MTP dipakai expert blok MTP |
| 8.3 | **Expert bertipe dengan nilai minimum (Q2_K, Q4_K)** di kernel MoE: jumlah aktivasi per 16 (`absum`) untuk gate/up dan down (`moe_quant` ikut menulisnya) | sekarang kernel MoE menolak tipe ini |
| 8.4 | **Blok MTP versi MoE**: muat router, shared expert dan expert blok MTP (`load_moe`), lalu jalankan lapis MoE di jalur MTP (`mtp_rows`) — untuk prompt (batch) dan untuk menebak token | blok MTP 4B/9B/27B memakai FFN biasa |
| 8.5 | Verifikasi 2–3 token sekaligus di lapis MoE utama (tile kecil) + pembatalan state DeltaNet (sudah ada) | ukur: kecepatan verifikasi vs 1 token |
| 8.6 | **Uji**: greedy identik dengan/tanpa MTP; prefill dengan MTP ≤ 5 % lebih lambat (A/B bergantian); tingkat penerimaan dan tok/dtk di prompt kode dan teks biasa | syarat lulus |
| 8.7 | Alias: nama file sama dengan versi non-MTP (`Qwen3.6-35B-A3B-UD-Q3_K_M.gguf`), jadi dibedakan lewat nama repo → `35b-mtp` (id `qwen3.6-35b-a3b-mtp`); daftar Pi, CHANGELOG, README | `35b` / `35b-s` tetap |

Risiko: konteks Q3_K_M dengan MTP turun ke ~40–50K token (dari 94K); verifikasi beberapa token di lapis MoE
membaca lebih banyak expert sekaligus (token berbeda memilih expert berbeda), jadi percepatannya bisa di bawah 27B.

## Tugas kecil baru

| # | Tugas | Status | Perkiraan hasil |
|---|---|---|---|
| F9 | Daftar model Pi mengikuti setelan vision: `pi-arc --vision off` → model ditandai teks saja dan konteks yang ditampilkan lebih panjang | ✅ (lewat F10) | konsistensi /model; tanpa efek kecepatan |
| F11 | **Bobot encoder gambar int8, aktivasi tetap F32**: varian `vit_gemm` dengan bobot Q8_0 yang dibuka ke F16 di kernel; uji selisih vs referensi f32 dan teks kecil di gambar uji; dipakai bila tidak ada penurunan | ⏸️ ditunda (2026-10-14): arcinferzero (expert 35B di cache GPU + SSD) memberi ruang untuk encoder 0,9 GB, jadi vision 35B tidak lagi bergantung pada F11. Catatan: mmproj Q8_0 pun sekarang tetap dibuka ke F16 di GPU | encoder 35B ~0,9 → ~0,48 GB, `35b-vision` 4K → ~40K token; 4B / 9B sedikit lebih banyak konteks | ~1 hari; dulu int8 penuh (aktivasi juga) meleset 10–23 %, jadi aktivasi wajib tetap F32 |
| ✅ | Opsi vision on/off: `pi-arc --vision off` / `arc-server -Vision off` (dikerjakan 2026-10-13) | ✅ | Qwen3.5 9B: 143K → 217K token saat vision mati; ganti model lewat /model mengikuti setelan saat server dinyalakan |

## F10: semua model dan variannya otomatis muncul di Pi /model

Permintaan: saat Pi dibuka, /model menampilkan setiap model yang terunduh beserta semua variannya: standar, dengan
MTP, dengan vision, dengan keduanya. Tanpa mengedit apa pun saat ada model baru.

Sudah ada (dasar):
- Server memindai cache Hugging Face + `%LOCALAPPDATA%\arcinfer\models` di setiap start; `pi-arc` / `arc-server`
  menulis ulang daftar model Pi dari hasil pindaian itu.
- Setiap file punya id unik (2026-10-13): file dengan nama sama dari repo lain mendapat `-mtp` bila membawa blok
  MTP, sisanya `-2`, `-3`; id lama (`qwen3.5-4b`, `35b`, ...) tidak berubah.

Rancangan:

| Langkah | Isi |
|---|---|
| 10.1 | **Kemampuan per file** dari isinya, bukan dari tabel nama: blok MTP (`nextn_predict_layers` atau drafter `mtp-*.gguf` Gemma di folder yang sama), encoder gambar / audio (mmproj di folder yang sama yang bisa dijalankan engine), arsitektur didukung atau tidak |
| 10.2 | **Varian per file**: hanya kombinasi yang memang ada. Id = id dasar + akhiran: `<id>` (standar: tanpa MTP, tanpa vision), `<id>-mtp`, `<id>-vision`, `<id>-mtp-vision`. Contoh: Qwen3.5 9B → 4 varian; Qwen3.8 27B (MTP, tanpa mmproj) → 2; Qwen3.6 35B non-MTP (tanpa mmproj lokal) → 1 |
| 10.3 | **Konteks otomatis per varian**: dihitung sendiri-sendiri (vision mati → tanpa cadangan encoder; MTP mati → tanpa blok MTP), disimpan di cache ctx-auto seperti sekarang; ditampilkan di nama varian |
| 10.4 | **Nama di Pi**: mudah dibaca, mis. `Qwen3.5 9B · MTP · vision · 143K`; urut per model, varian terlengkap dulu; input `text`/`image` sesuai varian |
| 10.5 | **Server**: memilih varian lewat /model = memuat ulang file yang sama dengan setelan berbeda (mmproj on/off, MTP on/off); bila hanya varian yang berbeda dari yang sedang jalan, tetap dimuat ulang (10–70 dtk) |
| 10.6 | **`pi-arc` / `arc-server`**: menerima id varian (`pi-arc 9b-vision`, `pi-arc 35b-mtp`); singkatan lama = varian default model itu (default: MTP + vision bila ada, sama seperti sekarang) |
| 10.7 | **Uji**: setiap varian muncul di /model, berpindah varian lewat /model benar-benar mengganti setelan (cek `/props`: vision, MTP, n_ctx), id lama tetap jalan, model baru yang diunduh muncul tanpa edit kode |

Catatan:
- Jumlah entri: dengan 9 file sekarang sekitar 20 entri. Kalau terasa terlalu ramai, bisa diberi opsi untuk
  menyembunyikan varian "standar".
- Pindaian dan hitung konteks per varian butuh beberapa detik per varian baru (sekali saja, lalu disimpan di cache).
- Opsi `--vision off` / `--no-mtp` di `pi-arc` tetap ada, setara dengan memilih variannya.

**Status F10: ✅ selesai (2026-10-13).** 22 entri dari 9 file; id lama tetap (= varian terlengkap); varian lain
`-mtp`, `-vision`, `-std`; konteks per varian (mis. 27B: 86K dengan MTP, 123K tanpa; 9B: 143K penuh, 262K
standar); /model berpindah varian = memuat ulang dengan setelan varian (diuji: vision dan MTP mengikuti);
`pi-arc --vision off` / `--no-mtp` memilih varian yang sesuai.

Urutan bersama tugas lain: F10 kecil–sedang dan tidak bergantung pada F8. Usul: F10 dikerjakan selagi file MTP
35B diunduh, lalu F8 (yang otomatis menambah varian `35b-mtp`).

## arcinferzero: MoE dengan expert di cache GPU + SSD (Level Zero), dimulai 2026-10-14

- Fase 1 ✅ uji Level Zero (`zero/ze_probe.cpp`): biner kernel OpenCL bisa dimuat Level Zero; memori device 69,9 GB/s,
  host 60,5 GB/s; memori host di atas jatah GPU bisa dipakai tapi hanya 9–21 GB/s (Windows memindah-mindah memori),
  jadi cache tetap di dalam jatah GPU. SSD Kingston NVMe (`bench/ssd_bench.cpp`): ~3,5–4,2 GB/s untuk baca acak
  1–4 MB dengan 16 permintaan sekaligus.
- Rekaman rute expert 35B (`ARC_DUMP_ROUTES`, 3 sesi x 400 token) + simulasi (`zero/cache_sim.py`): LRU paling baik
  (pin expert populer lebih buruk); hit rate 99,8 % (cache 88 %), 98 % (70 %), 95 % (50 %), 87 % (30 %).
- Keputusan: **Windows minimal 14 GB**; ukuran cache expert **bisa diatur** — bawaan dihitung otomatis supaya total
  server <= RAM - 14 GB (35B di 262K: cache ~10,8 GiB ≈ 83 %, ~19,5 tok/s); `--expert-cache-gb` / `--expert-cache-pct`,
  `--windows-reserve-gb`; tidak ada cache di RAM; baca SSD langsung (FILE_FLAG_NO_BUFFERING).
- File expert di SSD (`%LOCALAPPDATA%\arcinfer\zero\<model>.experts`): setiap expert utuh, berurutan, rata 4 KB, dalam
  susunan siap-GPU (blok seperti arcinfer), sehingga satu expert = satu baca SSD langsung tanpa konversi. Tidak ada →
  dibuat otomatis saat model pertama dijalankan; ada dan cocok (ukuran, tanggal, sidik GGUF) → dipakai; tidak cocok →
  dibuat ulang. Boleh dihapus kapan saja (35B ~14 GB, Flash Next ~40–45 GB).
- Fase 2 hasil (2026-10-14), `zero/` (`bin/runz.exe`, `bin/serverz.exe`; arcinfer biasa tidak diubah):
  - Shim OpenCL→Level Zero (`zero/clrt.h`) identik bit dengan arcinfer dan lebih cepat (35B tanpa streaming: TTFT
    6,45 → 4,07 dtk, generasi 16,9 → 18,3 tok/s; 27B identik, prefill sama).
  - Streaming expert (`zero/expert_store.h`, `zero/xcache.h`, kernel `-DXCACHE` di `zero/kernels/decode.cl`): file
    expert 15,1 GB (41 layer termasuk MTP) dibuat sekali dalam 144 dtk; hasil **identik bit** dengan arcinfer (chunk
    2048, logit 48 token sama persis).
  - Cache: satu daftar LRU per ukuran record; prompt memakai ujung belakang daftar sebagai area sementara (tidak
    menyapu cache); staging host 2 x 64 record (244 MiB); prefetch layer berikutnya di thread selama GPU bekerja;
    isi cache disimpan (`.lru`) dan dimuat ulang saat start (10,6 GB dalam ~6 dtk).
  - 35B, konteks 262K, chunk 4096 (p1, 2216 token, generasi 256 token): **MTP 22,2 tok/s** (cache 79 %, hit 97,6 %),
    tanpa MTP 19,3 tok/s (hit 98,6 %), TTFT 3,3–3,5 dtk (~640–680 tok/s; arcinfer 4,1–6,5 dtk). Cache penuh: MTP
    23,9 tok/s.
  - Batas memori: jatah GPU − 1,4 GiB (1,13 GiB sisa → kernel gagal: out of device memory) dan RAM − 14 GiB.
  - Server zero: chat, tool call, gambar + MTP bersamaan (konteks 262K, cache ~70 %) jalan; **masih ada crash
    "device lost" sesekali pada gambar kedua** — ✅ diperbaiki 2026-10-14 (zeKernelSetIndirectAccess untuk kernel
    XCACHE; 5/5 urutan server lolos).
- Fase 2 ✅ (rencana awal; hasil di atas) engine MoE di Level Zero + cache LRU + prefetch dari router; uji 35B 262K, hasil harus identik. Juga
  **dengan MTP**: target generasi **~21 tok/s** di 262K (arcinfer sekarang 22,3 tok/s tapi hanya 40K); blok MTP ~0,46 GB
  ikut di memori GPU (cache expert ~80 %), verifikasi 2 token per langkah ikut diukur hit rate-nya.
- Prefill: **chunk 4096** (expert yang tidak di cache dibaca dari SSD sekali per potongan; per token separuh dari
  chunk 2048; buffer +~0,3 GB). Expert untuk prompt dialirkan lewat buffer sementara terpisah, tidak masuk cache LRU,
  supaya satu prompt panjang tidak menyapu cache yang dipakai generasi.
- Fase 3 ⏳ Qwen3.8 Flash Next (qwen4exp: QSA, tabel n-gram di SSD, 512 expert top-10).


## F12: API server lengkap (diminta 2026-10-15, semua tahap; tanpa --api-key)

Semua di sisi server (`tools/server.cpp`); `zero/serverz.cpp` dibuat otomatis dari situ saat build supaya kedua
server selalu sama. Setiap tahap diuji dengan klien sungguhan (paket Python openai / ollama / anthropic, curl) dan
fitur lama dicek tetap sama (Pi, tool call, gambar, MTP).

| Tahap | Isi | Status |
|---|---|---|
| 12.0 | serverz.cpp dibuat dari server.cpp saat build | ✅ (kode; tes malam) |
| 12.1 | `frequency_penalty`, `repeat_penalty`, `logit_bias`, `tool_choice` (none / auto / required / fungsi tertentu) | ✅ kode (tool_choice lewat pembuka tool call yang dipaksa); tes malam |
| 12.2 | `logprobs` + `top_logprobs`; `n` > 1 (snapshot state prompt) | ✅ kode (n > 1 tanpa streaming); tes malam |
| 12.3 | `/v1/completions` (prompt mentah, autocomplete kode) | ✅ kode; tes malam |
| 12.4 | `response_format`: `json_object`, `json_schema` (sampling dibatasi, JSON pasti valid) | ✅ kode + 26 tes unit CPU; tes model malam |
| 12.5 | API Ollama: `/api/chat`, `/api/generate`, `/api/tags`, `/api/show` (NDJSON) | ✅ kode (+ /api/tags, /api/show, /api/ps, /api/version); tes malam |
| 12.6 | `/v1/messages` (Anthropic: tool_use, streaming event) | ✅ kode (+ /v1/messages/count_tokens); tes malam |
| 12.7 | `/v1/responses` (OpenAI Responses API, Codex CLI) | ✅ kode; tes malam |
| 12.8 | `/v1/audio/transcriptions` (multipart upload, encoder audio Gemma) | ✅ kode (+ /v1/audio/translations); tes malam |
| 12.9 | `/v1/embeddings` dengan EmbeddingGemma 2 (`unsloth/embeddinggemma-2-GGUF`, BF16 558 MB; mmproj BF16 982 MB opsional untuk gambar/audio) + pilihan fitur tanya-jawab dokumen (RAG) di web UI | ⏳ besok (2026-10-16), unduh dengan izin; user sedang mengunduh Qwen3.8 Flash Next |
| 12.10 | Launcher: base URL + tombol copy; help; README / CARA_PAKAI | ✅ (launcher: alamat API per format + Copy; dokumen); tes tampilan malam |

## F13: arcinferzero jadi satu-satunya engine (keputusan 2026-10-15)

Kalau zero sudah lolos semua, OpenCL tidak dipakai lagi untuk menjalankan model (OpenCL.dll tetap dipakai hanya
untuk mengompilasi kernel).

| Langkah | Isi | Status |
|---|---|---|
| 13.1 | (Gemma + drafter ikut pindah: disetujui 2026-10-15, kode drafter tidak diubah) Validasi semua model di zero: 4B, 9B, 27B, 35B, Gemma 12B / E4B (identik bit vs arcinfer, kecepatan bergantian 3x, MTP, drafter Gemma, gambar, audio, Pi, tool call) | ✅ 2026-10-16 malam: semua lolos (identik, MTP, gambar/audio, Pi, tool call); lihat docs/arcinferzero-log.md |
| 13.2 | Zero jadi bawaan: `bin\server.exe` dibangun dari engine zero, `serverz.exe` dihapus (begitu juga `runz.exe` → `run.exe`); OpenCL sementara sebagai `server-opencl.exe` (`--opencl`); `--zero` tidak perlu | 🟡 2026-10-16: zero jadi bawaan di arc-server / pi-arc / launcher (`-OpenCL` / `--opencl` untuk engine lama); exe belum diganti nama |
| 13.3 | Dipakai sehari-hari 1–2 minggu | ⏳ |
| 13.4 | Hapus jalur OpenCL (src/engine.h OpenCL, server.exe / run.exe lama) | ⏳ |

## F14: drafting paralel dan DFlash di Level Zero (setelah F13)

Diminta 2026-10-15: Level Zero dimantapkan dulu (F13), lalu:
- Drafting paralel: drafter (MTP / drafter Gemma) jalan di antrian Level Zero terpisah, tumpang tindih dengan
  verifikasi model utama; ukur dulu seberapa banyak yang bisa tumpang tindih di iGPU (satu engine komputasi).
- DFlash (drafter blok, mis. z-lab/Qwen3.5-27B-DFlash): satu langkah menebak satu blok token; sasaran utama 27B
  (sekarang 5–7 tok/s). Unduh model hanya dengan izin.

F14 rincian (2026-10-15, dari config.json + README `z-lab/Qwen3.8-27B-DFlash2`, mirror `incoai/...`; bobot belum diunduh):
- Drafter DFlash2: 5 layer gaya Qwen3 (hidden 5120, 32 head / 8 KV, head 128, FFN 17408, sliding window 2048,
  non-kausal), masukan hidden state layer 5 / 19 / 33 / 47 / 61 model utama, blok 8 token (7 tebakan), top-16
  kandidat per posisi + selector (rank 256), konvolusi dua-tap (kernel 2, grup 16); 3,85 GB BF16 (~1,9B),
  Q8 ~2 GB / Q4 ~1,1 GB. Apache-2.0.
- Tabel resmi (H200): token diterima per langkah MTP-7 vs DFlash2: GSM8K 5,02 / 5,46; HumanEval 3,91 / 4,39;
  MT-Bench 3,74 / 4,10. Qwen3.8 punya MTP 7 tebakan; arcinfer sekarang maksimal 3.
- Urutan: (1) MTP 27B sampai 7 tebakan (verifikasi blok panjang + rollback state DeltaNet tanpa 7 salinan penuh),
  (2) DFlash2 di atasnya (unduh 3,85 GB dengan izin). Perkiraan kasar 27B: 10–12 tok/s bila ~4 token diterima.

## Paket arcinferzero (2026-10-15): model MoE tanpa GGUF asli

- `serverz --pack <gguf>` (part 1 bila dipecah): `%LOCALAPPDATA%\arcinfer\zero\models\<nama>\<nama>.gguf` (bobot
  non-expert + semua metadata + expert sebagai tensor "virtual") dan `<nama>.experts` (file expert). Sumber boleh
  dari flashdisk; setelah itu GGUF asli boleh dipindah. Paket muncul di daftar model (serverz) dan hanya jalan
  dengan streaming.
- GGUF reader: model berbagi part (`-00001-of-0000N`) dibaca sebagai satu; tipe metadata disimpan untuk ditulis ulang.
- Uji: salin ulang GGUF drafter Gemma identik (45 kunci, 49 tensor). ⏳ malam: paket 35B (buat, jalankan tanpa
  GGUF asli, hasil identik bit dengan zero biasa). Flash Next (JANGAN hapus GGUF-nya: user yang memutuskan kapan, setelah backup): unduh (IQ3_S 83,6 GB, 2 part) → backup ke flashdisk
  (1 TB, exFAT/NTFS) → hapus dari SSD → `--pack` dari flashdisk; arsitektur `qwen4exp` dikerjakan terpisah.

## F15: dua percakapan sekaligus dan perbaikan tool call Gemma (diminta 2026-10-15)

Target (2026-10-15): E4B "minimal jago Pi" (semua tool Pi: read, bash, edit, write, web, browser).
- [x] Tool call terbatasi skema untuk Gemma 4 (`tools/tool_constraint.h`, 12 unit test). Di server-new.exe.
- [x] Qwen: snapshot ke-3 di awal giliran asisten setelah pesan user terakhir (server-new.exe). Template Qwen membuang
      reasoning giliran lama begitu user menulis lagi, jadi tiap pesan baru di Pi memproses ulang semua dari akhir
      system prompt (siang ini: 8K -> 4K cached, 10 s; di 57K: ~3 menit). Dengan slot ini: hanya giliran terakhir.
- [x] Uji di GPU (malam 15 Okt, A1): Pi + Qwen 9B, dua pesan berturut-turut dengan tool call di antaranya: output
      identik dengan proses dari nol; pesan baru 5,2 dtk vs 16,8 dtk tanpa snapshot.
- [x] Uji di GPU (malam 15 Okt, A3): tes edit create_word.py (salinan di folder tiruan) dengan E4B,
      ARC_TOOL_CONSTRAINT=0 vs 1: tool call sah 2/10 → 9/10.
- [ ] Format XML Qwen (`<tool_call><function=...><parameter=...>`) dengan pembatas yang sama.
- [ ] Tes Pi otomatis: tugas tetap (read, run, edit kecil, edit besar, write, web_search/fetch, browser) di folder
      kosong, skor per model (E4B, 12B, 4B, 9B, 35B), sebelum dan sesudah pembatas.
- [ ] Kalau E4B masih kurang: fine-tune LoRA dengan percakapan Pi yang berhasil dari model besar (GPU cloud).

- [x] Web UI compaction: ringkas otomatis saat konteks hampir penuh / server menolak, tombol Compact (CHANGELOG).
- [x] Log teks mentah tool call: `ARC_LOG_TOOLCALLS=file.jsonl` (CARA_PAKAI).
- [ ] Tool call Gemma: `edits` ganda diratakan; `path` yang hilang diisi dari satu-satunya file yang dibaca
      (`read`) di percakapan itu (kalau lebih dari satu file: tidak ditebak); `&amp;` di perintah bash.
- [ ] Slot tahap 1: beberapa cache percakapan, tidak dibagi rata: satu kolam, percakapan aktif memakai sesuai
      kebutuhan; kalau penuh, slot yang paling lama tidak dipakai disimpan ke SSD (bukan dibuang) dan dimuat lagi saat
      kembali. Di Level Zero (F13).
- [ ] Slot tahap 2: decode batch (2 percakapan dalam satu langkah GPU; MTP/drafter per percakapan; prefill disela).
- [ ] Cek: web UI di browser headless Pi turun ke 4,9 tok/s (dugaan: GPU direbut render browser).

## Flash Next (qwen4exp): unduhan selesai 2026-10-15 13:02

File (HF cache, JANGAN dihapus): `~/.cache/huggingface/hub/models--ISTA-DASLab--Qwen3.8-Flash-Next-GSQ-RCO-GGUF/snapshots/ed59f920.../IQ3_S/`
part 1 54,8 GB (semua weight), part 2 28,8 GB (hanya `per_layer_token_embd` [160, 320.001.536] IQ4_NL, tabel n-gram).
Header terbaca (v3, 1224 tensor). Arsitektur yang perlu dibuat:
- 48 layer, hidden 2560, 3 DeltaNet : 1 attention penuh (sama pola Qwen3.6), MoE 512 expert, 10 aktif, ffn 640, + shared expert.
- Hyper-connections: 4 aliran residual (10240 = 4 x 2560), low rank 320 (`hc_attn_*`, `hc_ffn_*`, `output_hc_*`).
- Attention penuh: indexer sparse (4 head, key 128, top_k 2048) + `compress_ratios` 4 di layer attention.
- PLE n-gram: tabel hash 320 juta baris x 160 (part 2), 1 baris per token -> dibaca langsung dari SSD (tidak perlu di GPU).
- Tipe baru: 42 = Q2_0 (blok 64) di 9 `ffn_down_exps`; lainnya IQ3_S/IQ3_XXS/IQ2_S/IQ4_NL/IQ4_XS/Q4_K..Q6_K/BF16.
- Belum ada MTP di GGUF ini. mmproj BF16 0,91 GB tersedia (belum diunduh, perlu izin).
- Referensi: llama.cpp upstream mendukung (`-lm mmap --lazy-mode on`); GSQ/RCO di github IST-DASLab.
Disk: C: sisa ~67 GB. Paket expert (~50 GB) muat tapi mepet -> utamakan baca expert langsung dari GGUF.

## F16: optimasi kernel Level Zero semaksimal mungkin (diminta 2026-10-15, mulai malam itu)

"Tanpa jalan pintas." Hanya jalur zero (kernels/*.cl lewat zero/clrt.h). Baseline dulu (prefill/decode tok/s per model).
- [~] D1 Register spill (sebagian, lihat status di bawah): `ARC_ZE_KPROPS=1` mencetak private/spill per kernel; semua model + encoder mmproj; target 0 spill.
- [~] D2 Flag compile Intel (256 GRF diuji per kernel, lihat status; subgroup 32 belum): large GRF (256 per thread / auto), subgroup 16 vs 32, opsi buffer > 4 GB; uji per kernel.
- [ ] D3 Grouping/tiling: tile & bentuk work-group GEMM prefill, baris per subgroup GEMV decode, block read, SLM;
      autotune per bentuk matriks per model, hasil disimpan.
- [ ] D4 GEMM prefill tanpa XMX: GPU ini (Core Ultra 5 125H, Arc Xe-LPG 112 EU) TIDAK punya XMX
      (matrix_multiply_accumulate dan 2d_block_io tidak ada); maksimalkan DP4a (sudah dipakai) + register blocking + fp16.
- [~] D5 (decode 35B dsplit dan prefill Qwen selesai; sliding window Gemma / konteks pendek belum) Attention untuk konteks pendek (~1-4K), sedang (~16-32K), panjang (~128K-262K): decode split-K untuk konteks
      panjang, prefill ala flash-attention (SLM, fp16/DP4a; tanpa XMX), sliding window Gemma, MTP verify multi-baris;
      varian dipilih otomatis per panjang konteks; error diukur juga pada teks panjang.
- [~] D6 mmproj (gambar Qwen/Gemma, audio Gemma) di Level Zero: jalan dan identik (15 Okt); optimasi kernelnya belum.
- [~] E (MTP 1–7 draft identik selesai; draft paralel / pohon belum) Drafting paralel MTP di Level Zero (bagian F14): beberapa draft per langkah, verifikasi satu pass, output
      identik; jumlah draft terbaik per model.
Kebenaran ("pastikan error rate rendah"): urutan akumulasi baru = tidak bit-identik dengan OpenCL. Acuan = logit f32
pada teks tetap (~2K token) per model; baseline error kernel sekarang diukur dulu. Syarat setiap perubahan: KL rata-rata
<= baseline dan < 1e-3, top-1 >= baseline - 0,1 poin, perplexity naik < 0,1 %, greedy sama kecuali near-tie, tes tool
call lolos, jawaban gambar/audio sama. Tidak memenuhi = tidak dipakai (sakelar ke kernel lama).

### Status malam 15 Oktober (diperbarui terus)

- [x] A1–A5 lolos (snapshot giliran Qwen, preserve-thinking, pembatas tool call Gemma + Qwen, cache sesi,
      compacting web UI). Lihat docs/arcinferzero-log.md.
- [x] Baseline zero semua model (identik dgn OpenCL; 35B identik dgn chunk sama).
- [x] E: MTP 1–7 draft benar (identik), rollback replay (1 salinan state). 4–7 draft belum lebih cepat dari 3:
      tiap draft = satu lintasan blok MTP + kepala output penuh.
- [x] D1 GEMV TB=8 (4–7 draft): spill 9 KB → ≤ 2,9 KB, 3,8 → 9,2 tok/s (27B, 5 draft).
- [x] D1 attention decode: spill hilang dengan 256 GRF; 35B pakai attn_dsplit (1,16 → 0,94 ms/layer di 64K),
      verifikasi tetap identik bit. Koreksi: Level Zero MEMAKAI flag 256 GRF; kesimpulan sebelumnya salah karena
      sakelar env saya sendiri kosong (lihat log, bagian KOREKSI).
- [x] Default draft MTP zero: 35B 3, 4B 3, 9B 2, 27B 3, Gemma 2. Perbaikan: koma JSON setelah key terakhir,
      `/api/tags` dengan `--no-swap`, attn_prep_g Gemma 256 GRF (E4B +1–2 %).
- [x] Spill Gemma: attn_prep_g HD512 4,6 KB → 0, add_norm_quant 3,7 KB → 64 B (identik, E4B/12B sedikit lebih cepat).
- [ ] Sisa spill: gdn_conv 1 KB (256 GRF: 0 spill, tidak lebih cepat), moe_gateup_q 2 KB (256 GRF tidak lebih cepat).
- [x] D5 prefill attention Qwen: attn_q8 256 GRF (sudah dipakai engine) mengalahkan semua varian attn_ps
      (64K: 4B/9B 926 vs 1.117 ms, 35B 923 vs 1.138, 27B 1.516 vs 1.669). Tidak ada perubahan.
- [ ] Draft lebih murah: kepala draft ringkas (`ARC_DRAFT_VOCAB`) ada tapi off (rugi untuk Bahasa Indonesia);
      ide berikut: daftar token dinamis (token di konteks + yang sering), atau draft paralel (pohon) per lintasan.
- [ ] Server kadang lambat ±½ sepanjang umurnya (35B: 26 → 12 tok/s, log identik), terutama bila dijalankan
      tepat setelah server lain mati; jeda 5 dtk membantu. runz --bench-pp kini mencetak waktu per langkah (16 Okt:
      7 run E4B 32K stabil, lambat tidak muncul; dugaan: satu jeda besar, bukan GPU lambat terus).
