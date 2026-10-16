# Cara pakai arcinfer

arcinfer adalah mesin inferensi LLM yang dibuat khusus untuk laptop ini (Intel Core Ultra 5 125H, GPU Intel Arc
terintegrasi). Fokusnya memproses prompt panjang dengan cepat, misalnya saat agen seperti Pi membaca banyak file
kode. Ia menjalankan model GGUF langsung dari cache Hugging Face dan menyediakan API yang kompatibel dengan OpenAI,
jadi bisa dipakai oleh Pi, aplikasi lain, atau lewat web UI bawaannya.

## 1. Setup (sekali saja)

Buka PowerShell atau Terminal di folder arcinfer, lalu jalankan:

```powershell
.\setup.cmd
```

Setup ini aman dijalankan ulang (langkah yang sudah beres dilewati). Ia:

- memeriksa kebutuhan: Windows 64-bit, GPU Intel Arc, Visual Studio Build Tools (C++), Python dan Pi, lalu
  memberi tahu cara memasang yang belum ada;
- membangun `bin\server.exe`, `run.exe` dan `clinfo.exe` bila belum ada atau lebih lama dari kode sumbernya, lalu
  memeriksa GPU (OpenCL 3.0, DP4a, memori GPU);
- menanyakan model mana yang diunduh ke cache Hugging Face (`.\setup.cmd -Models 27b,4b` tanpa bertanya; Gemma
  lengkap dengan drafter dan proyektor gambarnya; 35B diunduh lalu dikonversi). `huggingface_hub` / `numpy`
  dipasang lewat pip bila perlu (ditanya dulu);
- menambahkan provider `arcinfer` ke `~/.pi/agent/models.json` dengan semua model yang ditemukan beserta konteks
  masing-masing (provider lain tidak diubah, file lama disimpan sebagai `models.json.bak`), dan memasang perintah
  `pi-arc` dan `arc-server` ke `~/.pi/agent/bin` (sudah ada di PATH), sehingga bisa dipanggil dari folder mana pun.

`.\setup.cmd -Tests` juga menjalankan tes regresi.

Model yang tersedia (setup mengunduhnya ke `%USERPROFILE%\.cache\huggingface\hub`):

| Perintah | Model | File GGUF |
|---|---|---|
| `pi-arc 27b` | Qwen3.8 27B | `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF` → `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` |
| `pi-arc 35b` | Qwen3.6 35B-A3B (MoE) dengan MTP dan gambar, konteks 262K | `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` → `Qwen3.6-35B-A3B-UD-Q3_K_M.gguf` + `mmproj-BF16.gguf` (diunduh manual dengan `hf download`); kalau tidak ada: versi GSQ 2-bit `ISTA-DASLab/Qwen3.6-35B-A3B-2Bit-GSQ` yang dikonversi (lihat di bawah; tanpa MTP) |
| `pi-arc g12` | Gemma 4 12B (+ gambar) | `google/gemma-4-12B-it-qat-q4_0-gguf` → `gemma-4-12b-it-qat-q4_0.gguf` + `mmproj-gemma-4-12b-it-qat-q4_0.gguf`; drafter `mtp-gemma-4-12B-it.gguf` dari `unsloth/gemma-4-12B-it-qat-GGUF`, disalin ke folder snapshot model Google |
| `pi-arc e4b` | Gemma 4 E4B | `google/gemma-4-E4B-it-qat-q4_0-gguf` → `gemma-4-E4B_q4_0-it.gguf`; drafter `mtp-gemma-4-E4B-it.gguf` dari `unsloth/gemma-4-E4B-it-qat-GGUF`, disalin ke folder snapshot model Google |
| `pi-arc 9b` | Qwen3.5 9B | `unsloth/Qwen3.5-9B-MTP-GGUF` → `Qwen3.5-9B-UD-Q4_K_XL.gguf` |
| `pi-arc 4b` | Qwen3.5 4B | `unsloth/Qwen3.5-4B-MTP-GGUF` → `Qwen3.5-4B-UD-Q4_K_XL.gguf` |

Pilihan `35b` di `setup.cmd` mengunduh versi GSQ 2-bit dari ISTA-DASLab. Versi ini tidak tersedia dalam GGUF, jadi
dikonversi sekali (±2,5 menit, hasil 10,9 GB di `%LOCALAPPDATA%\arcinfer\models`):

```powershell
python tools\convert_gsq.py "<cache HF>\models--ISTA-DASLab--Qwen3.6-35B-A3B-2Bit-GSQ\snapshots\<rev>" `
  "$env:LOCALAPPDATA\arcinfer\models\Qwen3.6-35B-A3B-GSQ2-arc.gguf" --vocab-from "<GGUF Qwen3.5/3.6 apa saja>"
```

## 2. Memakai Pi

1. Buka terminal di folder proyek yang ingin dikerjakan. Di File Explorer: buka foldernya, klik kanan di area
   kosong, lalu pilih **Open in Terminal**.
2. Jalankan `pi-arc`. Pada pemakaian pertama, server dinyalakan dulu di latar belakang (5–40 detik sampai muncul
   "Ready"), lalu Pi terbuka.
3. Ketik permintaan seperti biasa di Pi.

| Perintah | Kegunaan |
|---|---|
| `pi-arc` | Model yang sedang berjalan atau yang terakhir dipakai (pertama kali: pilih dari daftar) |
| `pi-arc --pick` | Pilih model dari daftar semua model yang sudah diunduh, dengan nama lengkapnya |
| `pi-arc 27b` | Qwen3.8 27B: sangat pintar, paling lambat |
| `pi-arc 35b` | Qwen3.6 35B-A3B: pintar dan cepat (mixture of experts, hanya ±3B parameter aktif per token) |
| `pi-arc g12` | Gemma 4 12B: bisa membaca gambar (lampirkan gambar di Pi atau web UI) |
| `pi-arc e4b` | Gemma 4 E4B: kecil dan cepat |
| `pi-arc 9b` | Qwen3.5 9B: lebih cepat, kualitas menengah |
| `pi-arc 4b` | Qwen3.5 4B: paling cepat |
| `pi-arc --think off` | Tanpa tahap berpikir (reasoning): jawaban jauh lebih cepat |
| `pi-arc --think medium` | Berpikir lebih lama untuk tugas sulit (`off` / `minimal` / `low` / `medium` / `high`; bawaan `low`) |
| `pi-arc 9b --think off` | Kombinasi cepat untuk pertanyaan sehari-hari |
| `pi-arc -c` | Melanjutkan sesi terakhir di folder ini (`pi-arc -r`: memilih sesi) |
| `pi-arc -p "jelaskan app/auth.py"` | Argumen Pi lain tetap bisa dipakai, ditulis setelah opsi pi-arc |
| `pi-arc --ctx 32768` | Ukuran konteks lain (bawaan = maksimal yang muat, lihat bagian 5); `pi-arc --ctx auto` kembali ke otomatis |
| `pi-arc --no-mtp` | Tanpa speculative decoding (MTP / drafter) |
| `pi-arc --help` | Semua opsi pi-arc, server dan file yang dipakai |

Di dalam Pi, `/model` menampilkan semua model; memilih salah satu membuat server berganti ke model itu (5–40
detik), dengan konteksnya sendiri. `pi-arc` berikutnya memakai model yang terakhir dipilih.

Server tetap menyala setelah Pi ditutup. Sesi berikutnya langsung siap tanpa memuat model lagi, dan prompt yang
sama (misalnya prompt sistem Pi) diambil dari cache. Kalau `pi-arc` dipanggil dengan model lain, server berganti
model secara otomatis.

Prompt sistem Pi juga disimpan di disk (`%LOCALAPPDATA%\arcinfer\prefix`), jadi setelah server dinyalakan ulang
atau berganti model, sesi Pi baru tetap mulai dalam beberapa detik (27B: ~5 detik, bukan ~40 detik). Prompt sistem
Pi memuat folder proyek dan `AGENTS.md`-nya, jadi simpanannya dikelompokkan per folder proyek: setiap folder
menyimpan 2 versi terakhirnya dan tidak tergusur oleh folder lain (27B: ~290 MB per file). Semua model bersama-sama
dibatasi 8 GB (yang paling lama tidak dipakai dihapus duluan; ubah dengan `--prefix-cache-gb`).

Sesi baru selalu mulai kosong: Pi hanya mengirim prompt sistem dan pesan Anda, jadi model tidak melihat percakapan
sesi sebelumnya. Untuk melanjutkan percakapan, pakai `pi-arc -c` atau `pi-arc -r`. Yang diproses server saat itu:

| Situasi | Yang diproses (27B) |
|---|---|
| langsung setelah Pi ditutup | hanya pesan baru (server masih memegang percakapan itu) |
| setelah ada sesi atau proyek lain | hanya pesan baru: cache sesi (aktif secara default, 16 GB) menyimpan percakapan yang ditinggalkan ke disk dan memuatnya lagi dalam hitungan detik (`pi-arc --session-cache 0` mematikannya) |
| setelah `arc-server stop` / `restart` atau ganti model | hanya pesan baru: percakapan yang sedang dipegang disimpan dulu ke disk lalu dimuat lagi (hilang hanya kalau proses server dimatikan paksa) |

Menekan Esc di Pi langsung menghentikan server (saat membaca prompt: dalam ~1 detik; saat menulis jawaban: sebelum
token berikutnya), jadi pesan berikutnya tidak perlu menunggu. Pesan berikutnya juga tidak memproses ulang seluruh
percakapan: hanya pesan baru yang diproses.

## 3. Mengatur server

**Lewat jendela:** klik dua kali `arc-launcher.cmd` di folder proyek. Jendela ini menampilkan:
- semua model dan variannya (MTP, gambar, konteks otomatis, VRAM pada konteks itu, kompatibel atau tidak);
- untuk model yang dipilih: perhitungan konteks otomatis (memori GPU dikurangi jatah Windows, snapshot, encoder
  gambar, bobot, blok MTP dan buffer, lalu sisanya dibagi kebutuhan per 1.000 token);
- isian opsi server: yang utama tampil langsung (konteks, tebakan MTP, batas berpikir, prompt cache, akses dari
  komputer lain); sisanya (port, chunk, token gambar, bobot W8, folder dan ukuran cache, `--no-swap`, attention
  f32) muncul setelah centang "Show advanced options";
- pemantauan server yang berjalan: model, konteks terisi, VRAM per bagian, jumlah permintaan, kecepatan prompt dan
  jawaban terakhir, persentase tebakan MTP yang diterima.

Semua isian otomatis berisi nilai terbaik untuk model yang dipilih, jadi tinggal pilih model lalu Start. Isian yang
Anda ubah berwarna kuning dan hanya itu yang dikirim ke server. "Set as default" menyimpan isian yang diubah dan
menjadikan model terpilih sebagai model bawaan (juga untuk `pi-arc` tanpa nama model); "Reset options" kembali ke
nilai terbaik. Tombol Start / Stop memakai `arc-server`, jadi server tetap jalan walau jendela ditutup. (Teks di
jendela memakai bahasa Inggris: prompt cache = simpan percakapan, prefix cache = simpan system prompt.)

Batas berpikir di launcher hanya untuk klien yang tidak mengirim batasnya sendiri (web UI, aplikasi lain). Pi
mengirim batasnya sendiri sesuai level `/thinking`: minimal = berpikir mati, low 2.048, medium 8.192, high/xhigh
16.384 token.

**Lewat terminal:**

```powershell
arc-server status              # apakah server menyala, model apa yang dimuat, ukuran cache percakapan
arc-server stop                # matikan server dan bebaskan memori GPU (12–16 GB tergantung model)
arc-server clear-cache         # hapus semua percakapan tersimpan (cache sesi; aktif secara default, matikan dengan pi-arc --session-cache 0)
arc-server start -Model 9b     # nyalakan dengan model tertentu tanpa membuka Pi
arc-server start -Model 27b -Ctx 32768 -NoMtp
arc-server start -Model 27b -Ctx auto   # kembali ke konteks otomatis (me-restart server berkonteks tetap)
arc-server restart             # nyalakan ulang, misalnya setelah memori GPU diubah
arc-server start -Model 35b -MtpDrafts 2 -ThinkBudget 4096 -NoSwap   # opsi lain (lihat arc-server help)
arc-server help                # semua opsi
```

Log server ada di `%LOCALAPPDATA%\arcinfer\server.log`. Server memakai port 8090.

**arcinferzero (engine bawaan sejak 16 Okt 2026):** server berjalan lewat Level Zero (driver GPU Intel). Hasilnya
sama persis dengan engine OpenCL lama (sudah dites untuk semua model, gambar, audio, MTP, tool call, Pi). Untuk model
MoE (Qwen3.6 35B) expert disimpan di SSD dan hanya expert yang sering dipakai ada di memori GPU (cache): konteks bisa
sampai 262K bersama gambar dan MTP, dan Windows tetap kebagian RAM minimal 14 GB. Engine OpenCL lama masih ada: isian
"Engine" di launcher, atau:

```powershell
arc-server start -Model 35b          # arcinferzero (bawaan)
arc-server start -Model 35b -OpenCL  # engine OpenCL lama
pi-arc 35b                           # arcinferzero
pi-arc 35b --opencl                  # engine OpenCL lama (server yang jalan diganti)
pi-arc 35b --no-stream               # Level Zero tanpa cache SSD: semua expert di GPU (konteks sekecil OpenCL)
```

`-Zero` / `--zero` masih diterima (tidak mengubah apa-apa).

Pertama kali dipakai, arcinferzero membuat file expert sekali (35B: 15 GB, sekitar 2,5 menit) di
`%LOCALAPPDATA%\arcinfer\zero\`. File ini boleh dihapus kapan saja; kalau tidak ada, dibuat lagi otomatis. Ukuran
cache diatur otomatis (bisa diubah: opsi server `--expert-cache-gb`, `--expert-cache-pct`, `--windows-reserve-gb`).
Kalau memori GPU tiba-tiba kurang karena aplikasi lain (mis. Chrome), cache dikecilkan sendiri 1 GB per langkah,
bukan crash.

Catatan arcinferzero:
- Cache SSD hanya untuk model MoE (35B; expert layer MTP ikut di file yang sama). Model dense (27B, 9B, 4B, Gemma)
  hanya berganti engine ke Level Zero: konteks dan memori sama dengan engine OpenCL.
- File expert tidak pernah dihapus otomatis: start berikutnya langsung memakainya (35B sekitar 10 detik, termasuk
  mengisi cache GPU dari daftar terakhir). Dibuat ulang hanya kalau GGUF berubah; file versi lama perlu dihapus
  manual dari `%LOCALAPPDATA%\arcinfer\zero\`.
- Kernel GPU dikompilasi sekali lalu disimpan (`%LOCALAPPDATA%\arcinfer\kernels`): permintaan pertama sebuah model
  di zero, gambar pertama, atau setelah update driver Intel terasa lebih lambat, setelah itu normal.

Matikan server (`arc-server stop`) sebelum memakai laptop untuk pekerjaan GPU berat lain (game, edit video), atau
sebelum memindahkan/membangun ulang folder arcinfer.

## 4. Web UI dan aplikasi lain

Selama server menyala:

- **Web UI**: buka http://127.0.0.1:8090/ di browser untuk mengobrol langsung (bisa melampirkan file teks dengan
  drag-and-drop; juga gambar untuk Gemma 4 12B / E4B, Qwen3.5 4B / 9B dan Qwen3.6 35B: tombol **Image**,
  drag-and-drop, atau paste).
- **API OpenAI**: base URL `http://127.0.0.1:8090/v1`, API key bebas (misalnya `arcinfer`), model `qwen3.8-27b`
  (atau model yang sedang dimuat; model lain di daftar dimuat otomatis). Mendukung streaming, `reasoning_content`,
  tool calls, `usage`, penalti pengulangan (`frequency_penalty`, `repeat_penalty`), `logit_bias`, `logprobs`,
  beberapa jawaban sekaligus (`n`), jawaban JSON yang pasti valid (`response_format` dengan `json_schema`), dan
  `tool_choice` (memaksa tool call).
- **Format lain** (model dan fitur yang sama):
  - **Ollama**: aplikasi yang punya pilihan "Ollama" (mis. Open WebUI, banyak plugin editor) cukup diarahkan ke
    `http://127.0.0.1:8090` (`/api/chat`, `/api/generate`, `/api/tags`, ...).
  - **Anthropic**: `http://127.0.0.1:8090/v1/messages` (format Claude, termasuk tool_use dan thinking).
  - **OpenAI Responses**: `/v1/responses` (dipakai mis. Codex CLI); **Completions** lama: `/v1/completions`
    (prompt mentah, untuk autocomplete kode).
  - **Transkripsi audio**: `/v1/audio/transcriptions` (upload file audio, hasilnya teks; perlu Gemma 4 12B / E4B
    yang punya encoder audio).
- Embeddings (`/v1/embeddings`) belum ada: perlu model embedding dengan arsitektur yang belum didukung engine.
- **Web UI: ringkas otomatis.** Di kanan atas tampil pemakaian konteks ("context 12.3K / 40.9K"). Kalau pesan
  berikutnya membuat konteks hampir penuh, pesan-pesan lama diringkas dulu oleh model, lalu percakapan lanjut dengan
  ringkasan + tanya-jawab terakhir. Tombol **Compact** melakukannya kapan saja; **Stop** membatalkannya.
- **Tool call yang terjaga bentuknya** (Gemma 4 dan Qwen): selama model menulis tool call, server hanya memilih
  token yang membuat panggilannya sah menurut skema tool (argumen wajib seperti `path` tidak bisa terlewat, nama
  field hanya yang ada di skema). Isinya tetap pilihan model. Matikan dengan variabel `ARC_TOOL_CONSTRAINT=0`.
  `ARC_LOG_TOOLCALLS=file.jsonl` mencatat teks mentah setiap tool call (untuk menyelidiki tool call yang gagal).
- **Qwen3.6 35B: `pi-arc 35b --preserve-thinking`**: thinking giliran lama tetap disimpan di prompt (opsi
  resmi template Qwen3.6, Qwen3.8 27B selalu begitu). Pesan lanjutan tidak memproses ulang apa pun; konteks lebih
  cepat penuh.

Contoh dari PowerShell:

```powershell
$body = @{ model = "qwen3.8-27b"; reasoning_effort = "none"
           messages = @(@{ role = "user"; content = "Halo! Jelaskan arcinfer dalam satu kalimat." }) } | ConvertTo-Json -Depth 4
(Invoke-RestMethod http://127.0.0.1:8090/v1/chat/completions -Method Post -ContentType "application/json" -Body $body).choices[0].message.content
```

Dari Python (paket `openai`):

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8090/v1", api_key="arcinfer")
r = client.chat.completions.create(model="qwen3.8-27b", messages=[{"role": "user", "content": "Halo!"}],
                                   extra_body={"reasoning_effort": "none"})
print(r.choices[0].message.content)
```

## 5. Ukuran konteks, dan kalau percakapan sudah panjang

Setiap model memakai konteks terbesar yang muat di memori GPU yang sedang tersedia (diatur di Intel Graphics
Software, ±18–19 GiB di laptop ini; jumlah yang diberikan driver berubah-ubah tergantung aplikasi lain), dengan sisa
±1 GiB untuk Windows dan aplikasi lain, atau batas latih modelnya. Konteks menyesuaikan sendiri bila pengaturan itu
diubah (jalankan `arc-server restart` sesudahnya). Jendela launcher menampilkan angka saat ini. Contoh:

| Model | Konteks bawaan |
|---|---|
| Qwen3.8 27B | ±41K token dengan gambar + MTP (lebih besar tanpa gambar: `27b-mtp`, atau tanpa keduanya: `27b-std`) |
| Qwen3.6 35B-A3B | 262.144 token (penuh) dengan gambar + MTP (arcinferzero; engine OpenCL hanya ±40K dengan MTP) |
| Qwen3.5 9B / 4B | 262.144 token (penuh) |
| Gemma 4 12B | 262.144 token (penuh) |
| Gemma 4 E4B | 131.072 token (penuh) |

Jangan memasang memori GPU terlalu tinggi: pada 20 GB (dari 32 GB) model menyisakan terlalu sedikit memori untuk
Windows, sehingga laptop menjadi lambat atau tidak merespons.

Batas output (`maxTokens` di Pi) sama dengan ukuran konteks: Pi sendiri memotongnya ke ruang yang tersisa.
Prompt sangat panjang tetap butuh waktu (lihat bagian 6), jadi konteks besar terutama berguna untuk sesi panjang
yang memakai cache prefix.

Pi yang mengurus konteks penuh, bukan server. Saat percakapan melewati `ukuran konteks − 16384` token, Pi
meringkas bagian lama dan menyimpan ±20.000 token terbaru. Kalau satu permintaan tetap melebihi
konteks, server mengirim error yang sama dengan llama.cpp ("the request exceeds the available context size"). Pi
mengenalinya, lalu meringkas dan mencoba lagi secara otomatis. Anda juga bisa meringkas kapan saja dengan perintah
`/compact` di Pi.

Ringkasan ini butuh waktu (di 27B beberapa menit), karena model harus membaca bagian yang diringkas lalu memproses
ulang percakapan yang baru. Server tidak bisa sekadar membuang token lama: lapisan DeltaNet di Qwen3.5/3.8 menyimpan
state yang tidak bisa "melupakan" token tertentu.

## 6. Seberapa cepat

Diukur di laptop ini dengan arcinferzero, 15–16 Oktober 2026 (detail: `docs/performa-zero-2026-10-16.html`;
perbandingan dengan llama.cpp: `docs/review.html`). Kecepatan prompt = token per detik saat membaca prompt sepanjang
16K token; kecepatan menulis = jawaban 400 token lewat server (kode / bahasa Indonesia; MTP tidak mengubah isi
jawaban).

| Model | Membaca prompt (16K) | Menulis tanpa MTP | Menulis dengan MTP |
|---|---|---|---|
| Qwen3.6 35B-A3B | ±680 token/detik | ±20 token/detik | ±31 / 25 token/detik |
| Qwen3.5 4B | ±700 token/detik | ±20 token/detik | ±30 / 26 token/detik |
| Gemma 4 E4B | ±625 token/detik | ±22 token/detik | ±40 token/detik (kode) |
| Qwen3.5 9B | ±390 token/detik | ±12 token/detik | ±17 / 15 token/detik |
| Gemma 4 12B | ±228 token/detik | ±10 token/detik | ±21 token/detik (kode) |
| Qwen3.8 27B | ±103 token/detik | ±5 token/detik | ±9 / 7,4 token/detik |

Prompt yang sangat panjang lebih lambat dibaca (attention makin berat): di 131K token 35B ±250 token/detik, 4B ±280,
E4B ±200.

Tips:

- Bagian terlama biasanya tahap berpikir. `--think off` atau `--think minimal` mempercepat jawaban sederhana.
- Permintaan lanjutan dalam sesi yang sama hanya memproses token baru (cache prefix), jadi jauh lebih cepat dari
  permintaan pertama.
- Sesi baru di proyek yang sudah pernah dipakai cepat: prompt sistem Pi diambil dari cache (memori atau disk), jadi
  hanya pesan Anda yang diproses. Sesi pertama di folder proyek baru memproses prompt sistemnya sekali (27B: ±40 detik).
- Kembali ke sesi lama setelah ada sesi lain (atau web UI) memuat percakapannya dari disk dalam hitungan detik:
  cache sesi aktif secara default (16 GB di `%LOCALAPPDATA%\arcinfer\sessions`, hapus dengan `arc-server clear-cache`,
  matikan dengan `pi-arc --session-cache 0`). `arc-server stop` / `restart` dan pergantian model menyimpan dulu
  percakapan yang sedang dipegang server, jadi setelahnya percakapan itu juga dimuat, tidak diproses ulang.
- Pesan baru di sesi yang sama (Qwen 4B / 9B / 35B) hanya memproses ulang giliran terakhir: template Qwen membuang
  thinking giliran lama begitu Anda menulis lagi, dan server menyimpan titik awal giliran itu (arcinferzero).
- Matikan server lewat `arc-server stop` (atau tombol Stop di launcher), bukan dengan menutup prosesnya paksa: hanya
  dengan cara itu percakapan yang sedang dipegang disimpan dulu.
- Laptop yang panas melambat 5–10 %. Jangan menjalankan pekerjaan GPU lain bersamaan.

## 7. Memindahkan folder arcinfer

1. `arc-server stop`
2. Pindahkan folder `arcinfer` ke lokasi baru (File Explorer atau `mv`).
3. Jalankan ulang setup dari lokasi baru: `<folder-baru>\setup.cmd -Models none`

Hanya perintah pintas `pi-arc` / `arc-server` yang menyimpan lokasi folder; selebihnya memakai path relatif.

## 8. Mengatasi masalah

| Gejala | Penyebab dan solusi |
|---|---|
| `pi-arc` tidak dikenal | Jalankan `setup.cmd` (bagian 1), lalu buka terminal baru. |
| Setup gagal di langkah build | Visual Studio Build Tools (C++) belum terpasang; setup menunjukkan cara memasangnya. Keluaran compiler ada di `%LOCALAPPDATA%\arcinfer\build.log`. |
| "The server stopped while loading" | Lihat log yang ditampilkan. Biasanya memori GPU kurang: tutup aplikasi berat, atau pakai `--ctx 32768` / model lebih kecil. |
| Windows lambat atau tidak merespons saat model dimuat | Memori GPU diatur terlalu tinggi (bagian 5): turunkan di Intel Graphics Software, lalu `arc-server restart`. |
| `web_search` di Pi menjawab "No results found" atau tidak pernah selesai | Mesin pencari bawaannya DuckDuckGo dan Startpage; DuckDuckGo diblokir di Indonesia sehingga keduanya tidak memberi hasil. Tambahkan Yandex di `%USERPROFILE%\.pi\agent\settings.json`: `"webSearch": {"engines": ["yandex", "startpage", "duckduckgo"]}` (atau lewat perintah `/search-config` di Pi). |
| Sesi Pi baru butuh ±40 detik sebelum menjawab | Normal untuk sesi pertama di folder proyek baru (prompt sistem diproses sekali); sesi berikutnya di folder itu cepat. |
| "Model file ... not found" | Model belum diunduh ke cache Hugging Face (lihat tabel di bagian 1). |
| Pi berhenti dengan error konteks | Ketik `/compact` di Pi, atau mulai sesi baru. |
| Jawaban sangat lambat di awal | Normal untuk prompt besar pertama (lihat bagian 6); berikutnya memakai cache. |
| Ingin melihat apa yang terjadi | `arc-server status`, dan log di `%LOCALAPPDATA%\arcinfer\server.log` (waktu proses prompt, kecepatan, statistik MTP per permintaan). |

## 9. Membangun ulang dari kode sumber

Butuh Visual Studio 2026 Build Tools (MSVC). Dari folder arcinfer, `.\setup.cmd -Models none` membangun ulang yang
perlu saja (dan mematikan server dulu bila `server.exe` ikut dibangun). Atau secara manual:

```powershell
arc-server stop
.\build.ps1 zero\serverz.cpp zero\runz.cpp     # arcinferzero (bawaan)
.\build.ps1 tools\server.cpp tools\run.cpp     # engine OpenCL
```

Kernel GPU (`zero\kernels\*.cl` untuk arcinferzero, `kernels\*.cl` untuk engine OpenCL) dikompilasi oleh driver
saat server mulai, jadi perubahan kernel tidak perlu build.
Kernel yang sudah dikompilasi disimpan di `%LOCALAPPDATA%\arcinfer\kernels`, jadi hanya pemakaian pertama tiap varian
yang menunggu kompilasi.
Uji regresi: `.\tests\run_tests.ps1 -Model4B <path 4B gguf> -Model27B <path 27B gguf> -ModelGemma <path Gemma 12B> -Model35B <path 35B gguf>`.
