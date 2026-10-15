# Clipper offline (Node.js)

Alur: yt-dlp -> whisper.cpp (transkrip) -> Ollama (pilih momen) -> ffmpeg (potong, crop 9:16, subtitle).

Semua program dipanggil lewat **PATH**, jadi setelah setup sekali, perintah cukup `node clipper.mjs "URL"`.

## Persiapan (sekali saja, Windows)

### 1. Node.js 18+
Cek: `node -v`

### 2. ffmpeg dan yt-dlp (masuk PATH otomatis lewat winget)
```
winget install Gyan.FFmpeg
winget install yt-dlp.yt-dlp
```
Setelah install, **tutup lalu buka ulang terminal**.

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

### 4. Lokasi model (pilih salah satu)

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
Kalau VRAM mentok atau terlalu lambat, pakai model lebih kecil (misalnya `qwen2.5:3b`) lewat `--model`.

### 6. Tutup semua terminal, buka yang baru, lalu cek
```
node -v
yt-dlp --version
ffmpeg -version
whisper-cli --help
ollama list
```
Kelimanya harus jalan tanpa error "not recognized". Kalau memakai cara A di langkah 4, cek juga:
```
Test-Path $env:WHISPER_MODEL
```
Hasilnya harus `True`.

## Pemakaian

```
node clipper.mjs "https://www.youtube.com/watch?v=XXXX"
node clipper.mjs "URL" --lang id --clips 8 --min 30 --max 60 --nvenc
node clipper.mjs "URL" --lang en --horizontal --no-subs
```

## Struktur folder

Folder dibuat otomatis di tempat kamu menjalankan perintah, memakai **judul video YouTube** sebagai nama foldernya:

```
video mentah/
  <judul video>/
    <judul video>.mp4     <- video asli hasil download
    audio.wav, transcript.json, clip_XX_....srt   <- file kerja
    sumber.txt            <- id dan URL video
output/
  <judul video>/
    clip_01_....mp4 ... clip_05_....mp4           <- klip hasil produksi
    clips.json            <- judul, alasan, skor tiap klip
```

- Karakter yang tidak boleh dipakai di nama folder Windows (`\ / : * ? " < > |`) dan `%` dibuang dari judul. Nama folder dipotong maksimal 80 karakter.
- Kalau ada dua video berbeda dengan judul yang sama, folder kedua diberi tambahan ` [id-video]` supaya file tidak tercampur.
- Menjalankan ulang video yang sama tidak mengulang download dan transkrip.
- Nomor klip berdasarkan urutan waktu di video, bukan peringkat skor.

## Opsi

| Opsi | Fungsi | Default |
|---|---|---|
| `--lang` | Bahasa video (`id`, `en`, `auto`) | `id` |
| `--clips` | Jumlah klip maksimal | 5 |
| `--min` / `--max` | Rentang durasi klip (detik) | 30 / 60 |
| `--model` | Model Ollama | `qwen2.5:7b` |
| `--nvenc` | Encode pakai GPU NVIDIA (lebih cepat) | mati |
| `--horizontal` | Jangan crop ke 9:16 | mati |
| `--no-subs` | Tanpa subtitle | mati |
| `--chunk` | Panjang potongan transkrip untuk LLM (detik) | 360 |
| `--whisper-bin` | Path `whisper-cli` (hanya kalau tidak ada di PATH) | `whisper-cli` |
| `--whisper-model` | Path model whisper (menimpa `WHISPER_MODEL`) | `./models/ggml-large-v3-turbo-q5_0.bin` |
| `--ollama` | URL server Ollama | `http://localhost:11434` |
| `--raw` | Folder video mentah dan file kerja | `./video mentah` |
| `--out` | Folder klip hasil produksi | `./output` |

Variabel lingkungan yang dibaca script: `WHISPER_BIN`, `WHISPER_MODEL`, `OLLAMA_MODEL`, `OLLAMA_URL`.

## Troubleshooting

| Pesan error | Penyebab | Solusi |
|---|---|---|
| `spawn yt-dlp ENOENT` (atau `ffmpeg`, `whisper-cli`) | Programnya belum terinstall, atau belum ada di PATH, atau terminal belum dibuka ulang | Install, pastikan folder program ada di PATH, lalu **buka terminal baru**. Cek dengan `<program> --version` |
| `whisper-cli berhenti dengan kode 3` | Model whisper gagal dimuat. Di output terminal biasanya tertulis `model file not found ...` | Periksa langkah 4. Cek `echo $env:WHISPER_MODEL` (tidak boleh kosong) dan `Test-Path $env:WHISPER_MODEL` (harus `True`). Variabel baru hanya terbaca di terminal baru. Ukuran file harus sekitar 570 MB; kalau jauh lebih kecil, download ulang |
| Transkrip jalan tapi sangat lambat | Build whisper.cpp bukan versi cuBLAS, jadi jalan di CPU | Pakai zip yang mengandung `cublas`, dan pastikan driver NVIDIA terpasang. Pantau GPU di Task Manager |
| `Ollama error` / koneksi ditolak | Ollama belum berjalan atau modelnya belum di-pull | Buka aplikasi Ollama, jalankan `ollama pull qwen2.5:7b` |
| `Tidak ada klip yang lolos` | LLM tidak menemukan bagian yang cocok | Coba model lain, turunkan `--min`, atau cek `--lang` |

Kalau proses gagal di tengah, jalankan ulang perintah yang sama. Video, audio, dan transkrip yang sudah jadi tidak dibuat ulang.

## Catatan

- Crop 9:16 memakai tengah frame. Kalau pembicara tidak di tengah, tambahkan face tracking di langkah ffmpeg.
- Subtitle per segmen pendek (bukan efek kata-per-kata).
- Hasil LLM tidak selalu pas, jadi cek `clips.json` dan tonton klipnya sebelum diunggah.
- Pastikan kamu punya izin untuk meng-clip videonya (program clipping resmi atau izin dari kreator).
