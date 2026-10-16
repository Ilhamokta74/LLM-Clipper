# Clipper offline (Node.js + Python)

Alur: yt-dlp -> whisper.cpp (transkrip) -> Ollama (cari dan nilai momen) -> ffmpeg (potong, crop 9:16, subtitle kata-per-kata).
Untuk layout `follow`, `facetrack.py` (Python + MediaPipe) mendeteksi siapa yang sedang bicara dan crop mengikutinya.

Semua program dipanggil lewat **PATH**, jadi setelah setup sekali, perintah cukup `node clipper.mjs "URL"`.

## Cara kerja pemilihan klip (otomatis)

Jumlah dan durasi klip **tidak ditentukan di awal**. Hasilnya menyesuaikan kualitas video:

1. Transkrip dibagi jadi potongan 15 menit yang saling bertumpuk, supaya klip panjang tidak terpotong di batas potongan.
2. LLM mengusulkan hanya bagian yang layak berdiri sendiri (boleh nol, boleh banyak). Durasi mengikuti isi.
3. Tiap kandidat dinilai ulang dengan rubrik: **hook**, **utuh**, **nilai isi**, **emosi**. Skor akhir = 70% rubrik + 30% skor usulan.
4. Semua kandidat dengan skor akhir **8 atau lebih** diproduksi, tanpa batas jumlah.

Batas ini ditulis langsung di kode (bagian atas `clipper.mjs`) dan tidak punya opsi CLI:

| Konstanta | Nilai |
|---|---|
| `MIN_SCORE` | 8 |
| `MIN_DUR` | 30 detik |
| `MAX_DUR` | 200 detik |

Video 1 jam bisa menghasilkan 3 klip atau 15 klip, tergantung isinya. Kalau tidak ada yang mencapai 8, tidak ada klip yang dibuat. Lihat `kandidat.json` untuk skor semua kandidat.

## Persiapan (sekali saja, Windows)

### 1. Node.js 18+
Cek: `node -v`

### 2. ffmpeg dan yt-dlp (masuk PATH otomatis lewat winget)
```
winget install Gyan.FFmpeg
winget install yt-dlp.yt-dlp
```
Setelah install, **tutup lalu buka ulang terminal**. Build ffmpeg dari Gyan sudah menyertakan libass yang dibutuhkan untuk subtitle.

### 3. whisper.cpp (build CUDA untuk GPU NVIDIA)
1. Download zip Windows yang namanya mengandung `cublas` dan `x64` dari https://github.com/ggml-org/whisper.cpp/releases
2. Ekstrak ke folder tetap, misalnya `C:\tools\whisper`.
3. Cari folder yang berisi `whisper-cli.exe` (biasanya subfolder `Release`), lalu tambahkan ke PATH (PowerShell):
   ```
   [Environment]::SetEnvironmentVariable("Path", [Environment]::GetEnvironmentVariable("Path","User") + ";C:\tools\whisper\Release", "User")
   ```
   Pakai perintah ini, bukan `setx Path`, karena `setx` bisa memotong PATH yang panjang.
   Lewat GUI: cari "Edit environment variables for your account" -> `Path` -> Edit -> New.
4. Download model `ggml-large-v3-turbo-q5_0.bin` (sekitar 570 MB) dari https://huggingface.co/ggerganov/whisper.cpp/tree/main

### 4. Lokasi model whisper (pilih salah satu)

**A. Variabel `WHISPER_MODEL`** (model bisa disimpan di mana saja):
```
setx WHISPER_MODEL "C:\tools\whisper\models\ggml-large-v3-turbo-q5_0.bin"
```

**B. Folder `models` di samping script** (tanpa variabel):
taruh file `.bin` di `models\` pada folder yang sama dengan `clipper.mjs`, dan jalankan script dari folder itu. Ini path bawaan script (`./models/ggml-large-v3-turbo-q5_0.bin`).

### 5. Ollama
Install dari https://ollama.com, lalu:
```
ollama pull qwen2.5:7b
```
Konteks yang dipakai 16384 token karena potongan transkrip panjang. Kalau VRAM mentok atau terlalu lambat:
- pakai model lebih kecil lewat `--model` (misalnya `qwen2.5:3b`; penilaian bisa kurang tajam), atau
- turunkan `--ctx 8192` **dan** `--chunk 450` bersamaan.

### 6. Python + MediaPipe (hanya untuk `--layout follow`)
Perlu Python 3.10-3.12 (MediaPipe kadang belum mendukung Python terbaru). Dari folder script:
```
pip install -r requirements.txt
```
Model wajah (`face_landmarker.task`, beberapa MB) diunduh otomatis ke folder `models\` saat pertama kali dipakai, jadi butuh internet sekali. Kalau gagal, unduh manual dari URL yang tampil di pesan error, simpan sebagai `models\face_landmarker.task`, atau tunjuk lokasinya dengan `--track-model`.

Kalau tidak memakai `follow`, langkah ini boleh dilewati.

### 7. Tutup semua terminal, buka yang baru, lalu cek
```
node -v
yt-dlp --version
ffmpeg -version
whisper-cli --help
ollama list
python --version
```
Semuanya harus jalan tanpa error "not recognized" (Python hanya perlu kalau memakai `follow`). Kalau memakai cara A di langkah 4, cek juga:
```
Test-Path $env:WHISPER_MODEL
```
Hasilnya harus `True`.

## Pemakaian

```
node clipper.mjs "https://www.youtube.com/watch?v=XXXX"
node clipper.mjs "URL" --lang id --layout follow --nvenc
node clipper.mjs "URL" --layout fit
node clipper.mjs "URL" --lang en --horizontal --no-subs
```

### Layout 9:16 (`--layout`)

| Nilai | Hasil |
|---|---|
| `crop` (default) | Potong bagian tengah frame |
| `fit` | Video utuh di tengah, latar belakang blur |
| `follow` | Crop mengikuti orang yang sedang bicara (butuh Python + MediaPipe) |

Kalau analisis `follow` gagal untuk sebuah klip (Python/MediaPipe belum terpasang, dsb.), klip itu otomatis memakai `crop` tengah dan ada peringatan di terminal. Proses tidak berhenti.

## Struktur folder

Folder dibuat otomatis di tempat kamu menjalankan perintah, memakai **judul video YouTube** sebagai nama foldernya:

```
video mentah/
  <judul video>/
    <judul video>.mp4     <- video asli hasil download
    audio.wav, transcript.json   <- file kerja transkrip
    kandidat.json         <- semua kandidat beserta skor akhir (termasuk yang tidak lolos)
    clip_XX_....srt/.ass  <- subtitle tiap klip
    clip_XX_....track.json <- jalur crop (hanya layout follow)
    sumber.txt            <- id dan URL video
output/
  <judul video>/
    clip_01_....mp4 ...   <- klip hasil produksi (jumlah menyesuaikan)
    clips.json            <- judul, alasan, skor tiap klip yang diproduksi
```

- Karakter yang tidak boleh dipakai di nama folder Windows (`\ / : * ? " < > |`) dan `%` dibuang dari judul. Nama folder dipotong maksimal 80 karakter.
- Kalau ada dua video berbeda dengan judul yang sama, folder kedua diberi tambahan ` [id-video]` supaya file tidak tercampur.
- Menjalankan ulang video yang sama tidak mengulang download dan transkrip. Pemilihan klip oleh LLM dan penilaian dijalankan ulang.
- Nomor klip berdasarkan urutan waktu di video, bukan peringkat skor.

## Opsi

| Opsi | Fungsi | Default |
|---|---|---|
| `--lang` | Bahasa video (`id`, `en`, `auto`) | `id` |
| `--model` | Model Ollama | `qwen2.5:7b` |
| `--ctx` | Konteks Ollama (`num_ctx`) | 16384 |
| `--chunk` | Panjang potongan transkrip untuk LLM (detik, saling bertumpuk) | 900 |
| `--layout` | `crop`, `fit`, atau `follow` | `crop` |
| `--horizontal` | Jangan crop ke 9:16 (menimpa `--layout`) | mati |
| `--nvenc` | Encode pakai GPU NVIDIA (lebih cepat) | mati |
| `--no-subs` | Tanpa subtitle | mati |
| `--font` | Font subtitle (harus terpasang di Windows) | `Arial` |
| `--sub-words` | Maksimal kata per layar subtitle | 3 |
| `--python` | Perintah Python untuk `follow` | `python` (Windows) / `python3` |
| `--track-model` | Path `face_landmarker.task` | `models\` di samping `facetrack.py` |
| `--whisper-bin` | Path `whisper-cli` (hanya kalau tidak ada di PATH) | `whisper-cli` |
| `--whisper-model` | Path model whisper (menimpa `WHISPER_MODEL`) | `./models/ggml-large-v3-turbo-q5_0.bin` |
| `--ollama` | URL server Ollama | `http://localhost:11434` |
| `--raw` | Folder video mentah dan file kerja | `./video mentah` |
| `--out` | Folder klip hasil produksi | `./output` |

Jumlah klip, skor minimal, dan rentang durasi tidak lagi berupa opsi. Ubah konstanta `MIN_SCORE`, `MIN_DUR`, `MAX_DUR` di bagian atas `clipper.mjs` kalau perlu.

Variabel lingkungan yang dibaca script: `WHISPER_BIN`, `WHISPER_MODEL`, `OLLAMA_MODEL`, `OLLAMA_URL`, `PYTHON`, `FACE_MODEL`.

## Menyetel `follow` (crop mengikuti pembicara)

Untuk melihat siapa yang dipilih sebagai pembicara pada sebuah klip, jalankan `facetrack.py` langsung dengan `--debug`:

```
python facetrack.py "video mentah\<judul>\<judul>.mp4" --start 125.3 --dur 48 --out track.json --debug
```

Terminal menampilkan timeline per shot, misalnya `0.0-7.8 -> wajah #0`. Cocokkan dengan videonya, lalu setel:

| Gejala | Setel |
|---|---|
| Terlalu sering pindah pembicara | naikkan `--switch-cost` (default 3) |
| Telat pindah, atau sela pendek tidak ditangkap | turunkan `--switch-cost` |
| Orang yang diam ikut dianggap bicara | naikkan `--speak-thr` (default 0.3) |
| Pembicara bersuara pelan tidak terdeteksi | turunkan `--speak-floor` atau `--speak-thr` |

Catatan: `clipper.mjs` belum meneruskan opsi ini, jadi setel dulu lewat `facetrack.py` langsung. Kalau sudah ketemu nilainya, ubah `DEFAULTS` di `facetrack.py`.

## Troubleshooting

| Pesan error | Penyebab | Solusi |
|---|---|---|
| `spawn yt-dlp ENOENT` (atau `ffmpeg`, `whisper-cli`) | Programnya belum terinstall, atau belum ada di PATH, atau terminal belum dibuka ulang | Install, pastikan folder program ada di PATH, lalu **buka terminal baru**. Cek dengan `<program> --version` |
| `whisper-cli berhenti dengan kode 3` | Model whisper gagal dimuat. Di output terminal biasanya tertulis `model file not found ...` | Periksa langkah 4. Cek `echo $env:WHISPER_MODEL` (tidak boleh kosong) dan `Test-Path $env:WHISPER_MODEL` (harus `True`). Variabel baru hanya terbaca di terminal baru. Ukuran file harus sekitar 570 MB; kalau jauh lebih kecil, download ulang |
| Transkrip jalan tapi sangat lambat | Build whisper.cpp bukan versi cuBLAS, jadi jalan di CPU | Pakai zip yang mengandung `cublas`, dan pastikan driver NVIDIA terpasang. Pantau GPU di Task Manager |
| `Ollama error` / koneksi ditolak | Ollama belum berjalan atau modelnya belum di-pull | Buka aplikasi Ollama, jalankan `ollama pull qwen2.5:7b` |
| Ollama error saat memuat model / kehabisan memori | Konteks 16384 terlalu besar untuk VRAM | Pakai `--ctx 8192 --chunk 450`, atau model yang lebih kecil |
| `Analisis pembicara gagal ... Pakai crop tengah` | Python atau MediaPipe belum terpasang, atau model wajah gagal diunduh | `pip install -r requirements.txt`, cek `python --version`, atau unduh model wajah manual (langkah 6). Bisa juga set `--python` ke Python yang benar |
| `Tidak ada klip yang lolos` | Tidak ada kandidat dengan skor akhir 8 ke atas | Buka `kandidat.json` untuk melihat skor tiap kandidat. Coba model lain yang lebih besar, cek `--lang`, atau turunkan `MIN_SCORE` di `clipper.mjs` |
| Subtitle tidak muncul / error filter `ass` | ffmpeg tanpa libass, atau font tidak terpasang | Pakai ffmpeg dari winget (Gyan), atau set `--font` ke font yang ada |

Kalau proses gagal di tengah, jalankan ulang perintah yang sama. Video, audio, dan transkrip yang sudah jadi tidak dibuat ulang.

## Catatan

- Video panjang butuh waktu: whisper membaca seluruh audio, lalu LLM dipanggil sekali per potongan 15 menit dan sekali lagi untuk **setiap kandidat**. Video 1 jam dengan 20-30 kandidat wajar memakan puluhan menit tergantung GPU.
- Subtitle gaya short-form: 2-3 kata per layar, kata yang sedang diucapkan disorot kuning.
- Durasi klip bisa sampai 200 detik. Instagram Reels dan YouTube Shorts membatasi sekitar 3 menit, jadi klip di atas itu hanya cocok untuk TikTok atau unggahan biasa.
- Skor dari LLM lokal tidak selalu konsisten. Cek `clips.json` dan tonton klipnya sebelum diunggah.
- Pastikan kamu punya izin untuk meng-clip videonya (program clipping resmi atau izin dari kreator).
