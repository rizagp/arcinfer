# Log pekerjaan arcinferzero (14 Oktober 17:00 – 15 Oktober pagi)

Tujuan: engine baru `arcinferzero` (folder `zero/`, arcinfer lama tidak diubah) untuk model MoE. Expert disimpan di
SSD, dan hanya expert yang sering dipakai yang ada di memori GPU (cache LRU). Target: Qwen3.6 35B dengan konteks
262K dan MTP sekitar 21 tok/s, hasil identik bit dengan arcinfer, Windows tetap kebagian RAM minimal 14 GB.

## Ringkasan hasil

| | arcinfer (lama) | arcinferzero |
|---|---|---|
| Konteks maksimum 35B (MTP) | 40.960 | 262.144 |
| Konteks 35B dengan gambar + MTP | 4.096 | 262.144 |
| Generasi 35B dengan MTP | 22,3 tok/s (konteks 40K) | 22,2 tok/s (konteks 262K) |
| Generasi 35B tanpa MTP | 16,9–18,3 tok/s | 19,3 tok/s |
| Time to first token (prompt 2.216 token) | 4,1–6,5 dtk | 3,3–3,5 dtk |
| Memori server 35B + gambar | ~34 GB (RAM bebas tinggal ~1 GB, load 426 dtk) | ~17,5 GB (Windows kebagian 14 GiB, load ~10 dtk) |
| Hasil (logit) | acuan | identik bit |

## Urutan pekerjaan

### 14 Oktober sore

1. **Uji Level Zero sebagai pengganti OpenCL** (`zero/clrt.h`). Kernel arcinfer dijalankan lewat Level Zero tanpa
   diubah. Hasil identik bit dan lebih cepat: 35B tanpa streaming TTFT 6,45 → 4,07 dtk, generasi 16,9 → 18,3 tok/s.
   27B juga identik, prefill sama (~110 tok/s).
2. **File expert di SSD** (`zero/expert_store.h`): `%LOCALAPPDATA%\arcinfer\zero\<model>-<id>.experts`. Satu expert
   = satu record siap-GPU, rata 4 KB, dibaca langsung tanpa cache Windows. 35B: 15,1 GB untuk 41 layer (termasuk
   MTP). Dibuat sekali (144 dtk), lalu dipakai terus. Dibuat ulang hanya kalau GGUF berubah. Aman dihapus.
3. **Cache expert di GPU** (`zero/xcache.h`) dan kernel expert versi `-DXCACHE` (alamat tiap expert diambil dari
   tabel). Uji pertama: identik bit dengan arcinfer, generasi 19,8 tok/s.
4. **Error #1: RAM habis saat file expert dibuat** (RAM bebas tinggal 3,9 GB, laptop melambat). Penyebab: file 15 GB
   lewat cache Windows dan halaman GGUF tetap menempel di RAM. Perbaikan: tulis tanpa cache Windows, halaman GGUF
   dilepas setiap selesai satu layer dan setelah load.
5. **Error #2: cache terpasang 100 %.** Jatah minimum per kelompok ukuran expert terlalu besar. Diperbaiki.
6. **Konteks 262K + chunk 4096.** Mula-mula chunk jatuh ke 512 (TTFT 5,7 dtk). Diperbaiki ke 4096, prompt panjang
   diproses per layer (layer-major) supaya expert dibaca sekali per layer, dan expert layer berikutnya dibaca dari
   SSD di thread terpisah selagi GPU bekerja (prefetch). TTFT turun ke ~3,4 dtk.
7. **Cache bentuk baru**: area sementara 976 MiB diganti staging 244 MiB. Prompt memakai ujung belakang daftar LRU,
   sehingga tidak menyapu cache. Ruang memori yang dihemat masuk ke cache. Hasil: MTP 22,3 tok/s di 262K (target 21
   tercapai). Batas atas dengan cache penuh: 23,9 tok/s.
8. **Error #3: GPU kehabisan memori** (cache 84 %, sisa jatah GPU 1,13 GiB). Diperbaiki dengan menyisakan 1,4 GiB
   untuk driver. Akibatnya cache ~79–82 % (tanpa gambar) atau ~70 % (dengan cadangan encoder gambar).
9. **Server zero** (`zero/serverz.cpp`): chat, tool call, gambar + MTP bersamaan jalan. Isi cache disimpan (`.lru`)
   dan dimuat ulang saat start (10,6 GB dalam ~6 dtk), sehingga prompt pertama cepat (3,4 → 0,6–1,1 dtk).
   Konteks otomatis zero memakai catatan sendiri (sebelumnya terbawa angka 4.096 milik arcinfer).

### 14 Oktober malam: mencari crash "device lost"

Server zero kadang crash di tengah (sekitar separuh dari rangkaian uji chat → tool call → beberapa gambar). arcinfer
biasa dengan urutan yang sama tidak crash. Ditelusuri dengan alat debug baru: sinkron per kernel
(`ARC_ZE_SYNC_EACH=1`), cek alamat expert (`ARC_XCHECK=1`), budget memori Windows, ukuran scratch kernel.
Temuan dan perbaikan:

| Temuan | Perbaikan |
|---|---|
| Kernel membaca sedikit melewati ujung expert terakhir. Record 35B pas kelipatan 4 KB, jadi tidak ada sisa di ujung blok 1 GiB | Setiap blok cache diberi cadangan 64 KB |
| Entri tabel alamat yang belum diisi bernilai 0 (ID expert di luar rentang kalau router NaN) | Semua entri selalu berisi alamat yang valid |
| Shim tidak memeriksa batas buffer seperti OpenCL. Server zero dan arcinfer berbagi folder cache system prompt | Batas dicek (transfer ditolak seperti di OpenCL), zero memakai folder `zero\prefix` dan `zero\sessions` sendiri |
| Jatah GPU dipakai bersama aplikasi lain (Chrome, desktop) | Kalau GPU kehabisan memori, cache dikecilkan 1 GiB lalu dicoba lagi. Diuji dengan simulasi 24 kali, hasil tetap identik |
| **Penyebab utama:** kernel expert mengakses memori cache lewat alamat di tabel, bukan argumen kernel. Level Zero mewajibkan izin "indirect access", kalau tidak memori itu tidak dijamin siap (hanya ~9,9 dari 15,5 GiB yang resident) | `zeKernelSetIndirectAccess` dipasang di semua kernel. Kecepatan tidak berubah (20,1 tok/s), tetap identik bit |

Setelah perbaikan terakhir: **5 dari 5** rangkaian uji server lolos (sebelumnya sekitar separuh crash). Rangkaian
lengkap berikutnya juga lolos: chat, tool call `get_weather({"city":"Jakarta"})`, tiga gambar (jawaban gambar sama
persis dengan arcinfer biasa).

### 15 Oktober pagi

- Laptop sempat hang. Penyebabnya, skrip tes Pi semalam gagal di tengah jalan sehingga server zero (~14 GB) tidak
  dimatikan dan tetap memakan RAM sampai pagi. Laptop restart sekitar 07:42. Sekarang tidak ada proses tes yang
  jalan. Ke depan, skrip tes selalu mematikan server di akhir, termasuk kalau tesnya gagal.
- (Mouse yang bergerak sendiri bukan dari pekerjaan ini: tidak ada alat kendali layar, mouse, atau browser yang
  dipakai. Tes Pi selalu dengan `--no-tools --no-mcp`.)
- Opsi diganti sesuai permintaan: **`--zero`** untuk arcinferzero (`pi-arc 35b --zero`, `arc-server start -zero`).
  Tanpa `--zero` selalu arcinfer biasa, tidak ada pilihan yang diingat. Launcher punya isian "Engine".
- Help diperbarui: `pi-arc --help`, `arc-server help`, `bin\serverz.exe --help` (opsi `--expert-cache-gb`,
  `--expert-cache-pct`, `--windows-reserve-gb`, `--no-stream`).
- Dokumen: CHANGELOG, QUEUE.md, CARA_PAKAI.md (bagian arcinferzero).

## Belum selesai

- Tes Pi otomatis dengan server zero (skripnya gagal semalam, bukan karena engine-nya). Percobaan langsung dengan
  `pi-arc 35b --zero` bisa menggantikannya.
- 27B dengan zero: dicoba langsung lewat Pi (15 Oktober) dan berjalan baik (konteks otomatis 81.920 dengan MTP, sama dengan arcinfer; permintaan pertama lambat karena kompilasi kernel sekali). Yang belum: perbandingan kecepatan yang rapi. 27B bukan MoE, jadi tidak ada streaming dan
  konteksnya sama dengan arcinfer. Yang diuji hanya identik bit dan kecepatannya.
- Opsional: menghapus otomatis file expert versi lama kalau GGUF diganti versi baru.

## Cara memakai

```
pi-arc 35b --zero          # arcinferzero
pi-arc 35b                 # arcinfer biasa
arc-server stop            # matikan server setelah selesai
```

File: `zero/` (engine, shim Level Zero, cache expert, kernel), `bin\runz.exe`, `bin\serverz.exe`.
Data: `%LOCALAPPDATA%\arcinfer\zero\` (file expert 15 GB, daftar cache `.lru`, konteks otomatis, prefix, sessions).

## Malam 15–16 Oktober: tes dan optimasi di zero

Server pengguna saat mulai: Qwen3.6 35B (OpenCL, konteks 262K), dihentikan pukul 17:00, dipulihkan pagi.
Semua tes di `bin\serverz.exe` (port 8091/8090), `--no-prefix-cache`, server tes selalu dihentikan.

### A1. Snapshot awal giliran (Qwen)

Percakapan seperti Pi: system prompt ±1.240 token, dua giliran user dengan tool call (hasil tool ±1.650 token), lalu
pesan baru. Pesan baru harus lanjut dari awal giliran sebelumnya.

| Model | Pesan baru: cached | Waktu prompt | Server dingin | Hasil vs server dingin |
|---|---|---|---|---|
| 9B | 4.679 / 6.376 | 5,2 dtk | 16,8 dtk | identik (teks + logprob) |
| 4B | 4.679 / 6.376 | 2,9 dtk | 9,7 dtk | identik |
| 35B | 4.679 / 6.376 | 3,4 dtk | 11,6 dtk | teks sama, logprob beda |

Tanpa snapshot ini pesan baru mulai dari 1.242 (akhir giliran pertama sebelumnya hilang).
35B: beda logprob bukan dari snapshot. Hasil MoE memang bergantung pada cara prompt dipotong: potongan ≤ 8 token
memakai attention decode + gate/up tidak digabung, ≥ 1.152 token GEMM expert, di antaranya kernel per-tile
(runz `--feed-step 7`: 0,1809 vs 0,1712 untuk token teratas; server dingin chunk 1024 vs 2048 juga beda kecil).
Server dingin dua kali: identik. Tanpa streaming: beda yang sama. Mana yang lebih dekat ke f32 diukur di baseline.

Bug yang ketemu dan diperbaiki: `logprobs` mengirim token potongan karakter UTF-8 mentah (JSON rusak). Sekarang
teksnya diganti U+FFFD, byte aslinya tetap di `bytes`.

### A2. `--preserve-thinking`

35B: pesan baru cached 4.662 / 4.720, prompt 0,6 dtk (tanpa opsi: 1.242 cached, 5,0 dtk). 9B: peringatan "this
model's chat template has no such option", perilaku tidak berubah.

### A3. Pembatas tool call Gemma (E4B, Pi sungguhan)

Tugas: baca skrip, jalankan, tambah bagian baru dengan tool edit, jalankan lagi (folder tes, read/bash/edit/write).

| Pembatas | Berhasil | Panggilan edit | Gagal | Waktu |
|---|---|---|---|---|
| mati | 3/3 | 10 | 7 | 87–226 dtk |
| hidup | 3/3 | 5 | 0 | 88–112 dtk |

Kegagalan saat mati semuanya soal isi (`oldText` tidak unik / tidak cocok), bukan format; pembatas tidak
mengubah satu token pun (0 koreksi), jadi tool call yang sudah benar tidak terganggu. Format di bawah tekanan: lihat A3b.

### A3b. Format tool call di bawah tekanan (E4B, suhu 1,3, edit besar dipaksa lewat tool_choice)

| Pembatas | Sah | Rusak |
|---|---|---|
| mati | 2/10 | 8 (isi edits salah 3, kunci asing 3, path hilang 1, tanpa call 1) |
| hidup | 9/10 | 1 (tidak selesai, batas 3.000 token) |

Pembatas mengoreksi 1–2 token dari ±1.100 per panggilan (log: "tool call: 2 of 1237 tokens corrected").

### A4. Cache sesi (9B, Pi + web UI bergantian)

Kembali ke percakapan Pi setelah web UI: dimuat dari disk 0,17 dtk, 4.679 token terpakai ulang, jawaban identik
dengan server dingin (15,9 dtk prompt). Percakapan web UI pendek (597 token) sengaja tidak disimpan (batas 1.024
token: memprosesnya ulang lebih murah).

### A5. Compacting web UI (4B, konteks 4.096)

Otomatis di 72 %: 10 pesan → ringkasan, konteks turun ke 27 %. Ringkasan pertama 4B kehilangan satu kata kode
(DELTA); instruksi ringkasan diperketat (daftar, setiap item disalin persis, suhu 0) → semua kata kode tersimpan,
juga setelah compacting kedua (8/8 benar). Stop saat meringkas: percakapan utuh, server membatalkan dalam 0,6 dtk.
Tombol Compact: jalan. Ringkasan 4B kadang panjang (±90 dtk) → instruksi "satu baris per item".

### B. Baseline zero sebelum optimasi kernel (runz --bench-pp; server dengan jawaban 400 token)

Prefill / decode dalam token per detik pada konteks pendek (2K), sedang (16K), panjang (131K):

| Model | Identik dgn OpenCL | Prefill 2K / 16K / 131K | Decode 2K / 16K / 131K | Server tanpa / dengan MTP |
|---|---|---|---|---|
| Qwen3.5 4B | ya | 862 / 697 / 281 | 20,2 / 18,7 / 9,5 | 20,6 / 29,2 |
| Qwen3.5 9B | ya | 399 / 388 / 222 | 12,1 / 11,5 / 8,8 | 12,2 / 17,7 |
| Gemma 4 E4B | ya | 733 / 571 / 209 | 21,0 / 19,6 / 9,5 | 21,6 / 39,6 (drafter) |
| Gemma 4 12B | ya | 271 / 218 / 86 | 9,5 / 9,2 / 7,4 | 9,8 / 21,1 (drafter) |
| Qwen3.6 35B | tidak (lihat bawah) | 660 / 679 / 254 | 20,1 / 18,8 / 13,8 | (menyusul) |

Di konteks panjang decode turun ke ±½ dan prefill ke ⅓–¼: attention. Perkiraan dari 4B: di 131K attention decode
memakai ±30 GB/s dari 77 GB/s, prefill attention ±28 % puncak DP4a. GEMM sudah ±70 % puncak (catatan review).
35B tidak identik run vs runz: diduga pemotongan chunk (runz streaming 4096, run MoE 2048; hasil MoE bergantung pada
potongan, lihat A1), dicek dengan `--chunk 2048`.

Lanjutan baseline:

| Model | Identik dgn OpenCL | Prefill 2K / 16K / panjang | Decode 2K / 16K / panjang | Server tanpa / dengan MTP |
|---|---|---|---|---|
| Qwen3.6 35B | ya, dengan chunk sama | 660 / 679 / 254 (131K), 146 (262K) | 20,1 / 18,8 / 13,8 (131K), 10,1 (262K) | 21,0 / 26,8 |
| Qwen3.8 27B | ya | 110 / 103 / 80 (65K) | 5,1 / 4,7 / 4,0 (65K) | 5,0 / 9,1 |

35B: run (chunk MoE 2048) vs runz (streaming: chunk 4096) beda; `--chunk 2048` di keduanya: identik bit.

### E. MTP sampai 7 draft (27B, server, suhu 0, dua jawaban 400 token)

Rollback DeltaNet diganti: pass verifikasi menulis state ke `S_next` dan menyimpan inputnya; draft ditolak →
rekurensi diputar ulang dari `S` untuk token yang diterima saja (satu salinan state, bukan 3 atau 7; tanpa salin
150 MB per langkah). Semua jumlah draft 1–7: jawaban identik dengan tanpa MTP.

| Draft | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| tok/s (kode) | 7,45 | 8,31 | 9,36 | 3,06 | 3,83 | 4,17 | 3,89 |
| draft diterima | 192/208 | 257/280 | 283/333 | 298/376 | 306/405 | 314/417 | 314/456 |

Tanpa MTP 5,07. Mulai 4 draft (5 baris verifikasi) engine memakai GEMV multi-token TB=8: aktivasi 8 token di
register (±128 GRF di GPU ini) → spill. Diperbaiki: lihat D1.

### D1. Register spill (ARC_ZE_KPROPS=1, 27B dengan 5 draft MTP)

| Kernel | Spill per thread | Catatan |
|---|---|---|
| `attn_split` (decode attention, baris verifikasi) | 11.232 B | akibat bug sakelar saya (lihat KOREKSI): 0 dengan flag 256 |
| `attn_decode` (decode attention, 1 token) | 7.456 B | idem: 0 dengan flag 256 |
| `gemv_mt` TB=8 | s/d 9.216 B → s/d 2.912 B | diperbaiki: aktivasi dimuat per token |
| `gemv_mt_v` TB=8 | s/d 5.024 B → s/d 1.248 B | idem |
| `gdn_conv` | 1.024 B | belum |
| `attn_prep` | 192 B | belum |

TB=8 (4–7 draft): 128 GRF lebih cepat dari 256 GRF (27B, 5 draft: 9,2 vs 7,6 tok/s; sebelum perbaikan 3,8).
Baris per sub-group 1/2/4: sama (9,2–9,45). Dengan perbaikan ini 5–7 draft = 3 draft (±9,2 tok/s): langkah
verifikasi berkurang 117 → 94 → 86 per 400 token, tapi tiap draft menambah satu lintasan blok MTP termasuk kepala
output 248K kosakata (±1 GB baca per draft di 27B).

Masalah sistem yang ketemu: server yang dijalankan tepat setelah server lain mati kadang berjalan setengah
kecepatan sepanjang umurnya (35B + MTP: 11,9 vs 26,5 tok/s, log identik). Jeda 5 detik: normal. `arc-server stop`
sekarang menunggu proses selesai + 5 detik; harness tes juga.

### KOREKSI (01:30): "Level Zero tidak memakai mode 256 register" SALAH, penyebabnya bug saya

Sakelar uji `ARC_ATTN_DEC_GRF` yang saya tambahkan sore itu (static `std::string` dari operator ternary) terisi kosong
saat runtime, sehingga sejak build 20:24 kernel attention decode zero dibangun TANPA `-cl-intel-256-GRF-per-thread`.
Bukti: `ARC_ZE_KPROPS` sekarang mencetak opsi build tiap program; `attn_decode` tercatat tanpa flag. Probe baru
(`zero/grf_probe.cpp`): kernel 192-GRF dengan flag 256 tidak spill dan sama cepat di OpenCL, Level Zero immediate dan
command list biasa (23 ms; tanpa flag 212–1.429 ms). Level Zero menghormati mode itu. Sakelar ditulis ulang eksplisit.
Yang terdampak: pengukuran decode attention zero setelah 20:24 (spill attn_decode/attn_split 27B di tabel D1,
"zero 13,1 vs OpenCL 15,0 tok/s di 64K", sapuan register-128, perbandingan dsplit). Baseline (binari 17:02) tidak.
Hasil KL tetap sah (spill tidak mengubah hitungan).

Pengukuran ulang dengan register benar: 4B decode 32K jalur lama 17,1–17,3 vs dsplit 17,1 (setara); 35B 32K 18,4 vs
18,5. Decode attention memang sudah dekat batas bandwidth. Bench 256 register di 64K: dsplit menang 23 % hanya di
bentuk 35B (8 head per kv head: 1,16 → 0,94 ms per layer), 4B/9B +7 %, 27B +4 %. Keputusan: dsplit (4 head per
sub-group, dimensi dibagi 4, 1 baris per peluncuran, potongan 256, skala P per baris) hanya untuk 35B; yang lain
tetap jalur lama (membaca K/V 4 baris verifikasi sekali). `ARC_QWEN_DSPLIT=0/1` mematikan / memaksa.

Error decode (KL rata-rata terhadap f32 CPU, runz --kl-decode; teks ±250 token; top-1 = token teratas sama):

| Model | Jalur lama | Jalur baru (dsplit) |
|---|---|---|
| 4B (2 teks) | 0,00064 / 0,00067; top-1 245+249 | 0,00062 / 0,00064; top-1 245+249 |
| 9B | 0,00069; top-1 246 | 0,00060; top-1 247 |
| 35B (2 teks) | 0,00774 / 0,00444; top-1 245+240 | 0,00811 / 0,00436; top-1 242+243 |
| 27B (3 teks) | 0,00030 / 0,00030 / 0,00034; top-1 702/709 | 0,00040 / 0,00034 / 0,00032; top-1 700/709 |

Derau pengukuran di tingkat ini: mengganti hanya pembulatan skor 27B (jumlah int eksak lawan float per 64 dimensi)
menggeser KL 0,000355 → 0,000414 dan top-1 700 → 699. Selisih lama/baru ada di dalam derau itu (arahnya juga
berganti antar teks dan model), jadi jalur baru dipakai untuk semua Qwen; `ARC_QWEN_DSPLIT=0` mengembalikan yang lama.

MTP dengan jalur decode baru (server, suhu 0, jawaban identik dengan tanpa MTP di semua kasus):
4B 3 draft 27,5 / 19,3 tok/s; 9B 3 draft 18,5 / 15,1; 35B 1 draft 26,3 / 22,9, 3 draft 30,7 / 23,4; 27B 3 draft
7,9 / 6,3 dan 5 draft 9,4 / 7,0 (angka 9B/27B ini terganggu acuan f32 yang dihitung bersamaan di CPU).

### D5 prefill attention Qwen, register default (attn_bench ATT_GRF128, chunk 4K terakhir di kedalaman 64K)

4B/9B: attn_q8 1.276 ms (spill 4.192 B) → attn_ps PNS 1, PHG 2, PTT 1, TPW 32: 1.166 ms (+9 %), bit-identik
(ATT_CMP). Varian PNS 2/4 lebih lambat di register default.

MTP diukur ulang dengan build benar (server, suhu 0, rata-rata jawaban kode + Indonesia, semua identik):
4B 2 / 3 / 4 draft: 22,9 / 24,8 / 25,2 tok/s; 9B 2 / 3 / 4: 15,8 / 15,6 / 15,9; 35B 1 / 3: 23,9 / 26,1;
27B 3 / 5 / 7: 8,3 / 8,2 / 8,2. Default zero: 4B 2 → 3, 35B 1 → 3 (9B 2, 27B 3, Gemma tetap).

### Zero jadi engine bawaan (16 Okt, ~02:30)

`arc-server`, `pi-arc` dan launcher memakai serverz secara bawaan; `-OpenCL` / `--opencl` untuk engine lama. Dasar:
semua tes malam ini lolos (identik dengan OpenCL, gambar/audio, MTP, Pi, tool call, format API).

### D5 prefill attention Qwen dengan 256 GRF (build benar; chunk 4K terakhir di kedalaman 64K)

| Bentuk | attn_q8 (256 GRF, = engine) | attn_ps terbaik | 
|---|---|---|
| 4B/9B (16/4) | 926 ms (spill 576 B) | 1.117 ms (PS 1,4,1 TPW 32) |
| 35B (16/2) | 923 ms (spill 576 B) | 1.138 ms (PS 1,4,1 TPW 16) |
| 27B (24/4) | 1.516 ms (spill 0) | 1.669 ms (PS 1,3,1 TPW 8) |

Dengan 256 GRF attn_q8 menang di semua bentuk (attn_ps hanya menang melawan attn_q8 128 GRF, 1.276 ms). Engine
sudah membangun attn_q8 dengan 256 GRF, jadi tidak ada perubahan; `ARC_QWEN_PS` tetap mati secara bawaan.

### Spill add_norm_quant (Gemma) 3.712 B → 0 (16 Okt, 03:00–04:50)

Pass kedua membaca ulang nilai yang baru ditulis work-item itu sendiri, bukan menahan seluruh baris di register.
Logit E4B dan 12B identik bit dengan OpenCL (dump 11K token). runz --bench-pp (vs baseline sore 15 Okt, yang juga
belum memakai attn_prep_g 256 GRF): E4B 2K 733 → 814 tok/s, 16K 571 → 620, 131K 209 → 197 (satu kali jalan);
12B 2K 271 → 286, 16K 218 → 224, 131K 86,1 → 86,5; generate sama atau sedikit naik.
A/B lama/baru bergantian, E4B: 2K identik (pp ~820, tg 21,4 keduanya); 64K pp sama (~330), 32K pp sama (~470).
Generate setelah prompt panjang sangat berderau: 32K lama 10,8 lalu 17,9 tok/s, baru 18,3 / 18,4; 64K lama
10,1 / 9,9, baru 8,9 / 8,9. Ini derau "kadang ½ kecepatan" yang sama (kernel ini tidak bergantung panjang konteks dan
di 2K tidak ada selisih), bukan regresi.

### gdn_conv (Qwen) spill 1 KB (16 Okt, 04:50–05:10)

256 GRF menghapus spill-nya, tapi 9B runz --bench-pp bergantian 3 putaran: 2K 435/430/417 (tanpa) vs 432/419/414
(256), 16K 390/366/365 vs 389/362/390: sama dalam derau. Membaca bobot per channel (vload4) malah 1.792 B. Bawaan
tetap; `ARC_CONV_GRF=256` untuk eksperimen.

### Koreksi add_norm_quant + spill attn_prep_g HD512 (16 Okt, 05:10–05:25)

Versi "baca ulang dari x" (03:00) dibatalkan walau identik: komentar kernel mencatat bahwa membaca x yang baru ditulis
bisa memberi nilai lama dari L1 (bisa muncul sesekali saja). Gantinya: nilai tetap di register, tapi `ANQ_MAX`
(superblock per sub-group) diberikan engine = ceil(n_embd / 256 / 16): E4B dan 12B 1 → spill 3.712 → 64 B.
attn_prep_g: bobot norm, pasangan RoPE dan tabel RoPE diproses per 8 elemen, dan K mentah (V untuk layer tanpa
proyeksi V) dibaca ulang dari qkv (hanya input, tidak ditulis kernel ini): HD 512 spill 4.608 → 0 (256 GRF).
E4B / 12B logit identik bit dengan OpenCL. runz --bench-pp: E4B 2K 827 tok/s (tg 21,5), 16K 625 (tg 20,1);
12B 2K 289 (9,55), 16K 228 (9,24).

### Generate kadang ½ kecepatan: pencatat per langkah (16 Okt, 05:30–05:45)

`runz --bench-pp` sekarang mencetak waktu per langkah generate (min / median / max, 8 langkah pertama) dan menerima
`--prof-gen`. E4B 32K, 7 kali (1 atau 3 run prompt, 128 token): 18,5–18,96 tok/s, median 52,4 ms, langkah pertama
91–231 ms; lambat tidak muncul. Run lambat sebelumnya (64 token, 10,8 vs 17,9 tok/s = ±2,4 dtk ekstra) lebih cocok
dengan satu jeda besar (mis. WDDM memindahkan memori) daripada GPU lambat sepanjang run; perlu dilihat lagi dengan
pencatat ini bila muncul.

### Ukuran halaman memori GPU (16 Okt, siang; `zero/page_probe.cpp`)

Dipicu dokumen "Intel GPU Level Zero Custom Kernel Blueprint" (halaman 4 KB, memori virtual read-only, KV berhalaman
lewat zePhysicalMemCreate). Hasil di driver ini:
- zeVirtualMemQueryPageSize: 64 KiB untuk alokasi kecil, 2 MiB untuk alokasi >= 2 MiB.
- zeMemAllocDevice: baca berurutan 67–77 GB/s; baca acak 256 B di area <= 1 GiB ±78 GB/s, di 1,5–2 GiB 22–24 GB/s
  (±3,5x lebih lambat; dugaan: cakupan TLB GPU ±1 GiB dengan halaman 2 MiB).
- zeVirtualMemReserve + zePhysicalMemCreate berhasil, tapi zeVirtualMemMap ditolak (0x7800000a, UNSUPPORTED_SIZE)
  untuk potongan 2, 64 dan 256 MiB: memori virtual Level Zero tidak bisa dipakai di Windows ini. KV berhalaman (slot
  tahap 1/2, draft pohon) harus lewat tabel blok di kernel, bukan pemetaan halaman fisik.
- arcinfer membaca bobot berurutan dan expert 35B per blok utuh (ratusan KB), jadi batas ±1 GiB untuk baca acak kecil
  hampir tidak berpengaruh; tidak ada perubahan.

Sapuan baca acak (`page_probe 3072 sweep`; GB/s, satu alokasi zeMemAllocDevice; satu alokasi 4 GiB ditolak driver):

| Area | 256 B | 4 KiB | 64 KiB |
|---|---|---|---|
| 4 MiB | 902 | 934 | 951 |
| 8 MiB | 156 | 158 | 266 |
| 16 MiB | 104 | 104 | 228 |
| 32 MiB | 89 | 89 | 99 |
| 64 MiB – 1 GiB | 77–83 | 78–83 | 77–89 |
| 1,5 GiB | 23,7 | 75,7 | 76,0 |
| 2 GiB | 21,7 | 75,3 | 76,6 |
| 3 GiB | 20,7 | 74,9 | 76,2 |

Cache GPU ±4 MiB (±900 GB/s); dari ±32 MiB kecepatan RAM (±78 GB/s); di atas ±1 GiB hanya baca acak 256 B yang jatuh
(±21 GB/s), potongan >= 4 KiB tetap ±75. Bobot, expert dan blok KV arcinfer (16 key x 256 dim = 4 KiB per head)
semuanya >= 4 KiB bersambung: tidak terpengaruh. Untuk KV berhalaman nanti: halaman >= 4 KiB.

### 27B: di mana waktunya (16 Okt, siang; runz --prof-gen, profil per kernel kini memakai timestamp Level Zero)

Sebelumnya `--prof-gen` di zero selalu 0 ms (clrt.h tidak punya timer); sekarang event dari pool timestamp kernel.
27B, generate (profil aktif: 4,29 tok/s, tanpa profil 4,8): waktu GPU 223 ms / langkah 233 ms -> jeda antar-kernel ±4 %
(±940 kernel per token). Spill: attn_prep 192 B (0,5 % waktu), gdn_conv 1 KB (0,6 %): tidak berarti.
GB/s efektif per peran (11,3 GB/token): attn_qkv 76, output 73,5, attn_gate 73; ffn_down 60, ffn_up 57, ffn_gate 56,
ssm_out 51. Per jenis kuantisasi (profil per layer, angka absolut lebih rendah karena beban profil): IQ4_XS / Q4_K
≈ 62, IQ3_S / IQ3_XXS ≈ 51–55, IQ2_* ≈ 47–52, IQ1_M ≈ 33: dekode codebook (IQ3/IQ2) belum tertutup waktu baca memori.
FFN 27B didominasi IQ3_S / IQ3_XXS (±5 GB). Perkiraan: dekode IQ3 setara IQ4_XS +5–10 %, gabung kernel kecil +2–4 %.
