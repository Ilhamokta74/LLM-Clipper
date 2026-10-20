#!/usr/bin/env node
// Thumbnail potret (1080x1920) dari klip jadi hasil clipper.mjs
// Alur: tiap file .mp4 di ./output/<folder>/ -> ambil beberapa frame -> pilih yang paling tajam -> blur area subtitle -> tempel teks hook
// Nama klip (nama file) dipakai sebagai judul; thumbnail disimpan di folder yang sama dengan nama yang sama (.jpg).
// Butuh: Node 18+, ffmpeg, sharp (npm i sharp), Ollama (opsional, untuk teks hook; tanpa Ollama dipakai nama klip)
//
// Pemakaian: node thumbnail.mjs                 -> semua folder di ./output
//            node thumbnail.mjs "<nama folder>"  -> satu folder saja
// Opsi: [--no-llm] [--accent none] [--force] [--keep-subs] [--text-y 0.55] [--font Arial] [--text-size 120]

import { spawn } from 'node:child_process';
import { parseArgs } from 'node:util';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

const { values: opt, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    model: { type: 'string', default: process.env.OLLAMA_MODEL || 'qwen2.5:7b' },
    ollama: { type: 'string', default: process.env.OLLAMA_URL || 'http://localhost:11434' },
    font: { type: 'string', default: 'Arial' }, // font teks (harus terinstall di sistem)
    'text-size': { type: 'string', default: '120' }, // ukuran font maksimal; otomatis mengecil kalau teks panjang
    'text-y': { type: 'string', default: '0.55' }, // posisi tengah teks dari atas (0-1); 0.55 = tepat di atas area subtitle
    accent: { type: 'string', default: '#FFD400' }, // warna baris terakhir teks; "none" = semua putih
    'no-llm': { type: 'boolean', default: false }, // pakai nama klip apa adanya sebagai teks
    'keep-subs': { type: 'boolean', default: false }, // jangan blur area subtitle (pakai kalau klip dibuat dengan --no-subs)
    force: { type: 'boolean', default: false }, // buat ulang thumbnail yang sudah ada (default: dilewati)
    out: { type: 'string', default: './output' }, // harus sama dengan clipper.mjs
  },
});

const arg = positionals[0]; // opsional: nama/path satu folder hasil. Kosong = proses semua folder di ./output

let sharp;
try {
  sharp = (await import('sharp')).default;
} catch {
  console.error('Modul "sharp" belum terinstall. Jalankan: npm i sharp');
  process.exit(1);
}

const W = 1080;
const H = 1920;
// Area subtitle (default clipper.mjs: kira-kira 68%-77% tinggi layar); di-blur supaya tidak bentrok dengan teks thumbnail
const SUB_BAND = { top: Math.round(H * 0.66), height: Math.round(H * 0.14) };

// ---------- Helper ----------
function run(cmd, args, options = {}) {
  return new Promise((resolve, reject) => {
    const p = spawn(cmd, args, { stdio: 'inherit', ...options });
    p.on('error', (e) =>
      reject(new Error(`Gagal menjalankan "${cmd}": ${e.message}. Pastikan sudah terinstall dan ada di PATH.`))
    );
    p.on('close', (code) => (code === 0 ? resolve() : reject(new Error(`${cmd} berhenti dengan kode ${code}`))));
  });
}

const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

// ---------- 1. Pilih frame terbaik ----------
// Ambil 1 frame tiap 1,5 detik dari awal klip (maks 20 detik), buang yang gelap/terang ekstrem, pilih yang paling tajam
async function extractFrames(video, dir) {
  await fs.rm(dir, { recursive: true, force: true });
  await fs.mkdir(dir, { recursive: true });
  await run('ffmpeg', [
    '-y', '-loglevel', 'error',
    '-ss', '0.5',
    '-t', '20',
    '-i', video,
    '-vf', 'fps=1/1.5',
    '-q:v', '2',
    path.join(dir, 'f_%02d.jpg'),
  ]);
  return (await fs.readdir(dir)).filter((f) => f.endsWith('.jpg')).sort().map((f) => path.join(dir, f));
}

async function frameScore(file) {
  // Nilai hanya bagian atas frame (62%) supaya teks subtitle tidak ikut mengerek skor ketajaman
  const { data, info } = await sharp(file).resize(480).greyscale().toBuffer({ resolveWithObject: true });
  const top = () => sharp(data).extract({ left: 0, top: 0, width: info.width, height: Math.floor(info.height * 0.62) });
  const { channels } = await top().stats();
  const mean = channels[0].mean;
  if (mean < 30 || mean > 235) return 0; // frame hitam/putih (fade, transisi)
  // Ketajaman = simpangan baku hasil filter Laplacian (makin blur, makin kecil)
  const lap = await top()
    .convolve({ width: 3, height: 3, kernel: [0, 1, 0, 1, -4, 1, 0, 1, 0], scale: 1, offset: 128 })
    .stats();
  return lap.channels[0].stdev;
}

async function bestFrame(files) {
  let best = null;
  for (const f of files) {
    const score = await frameScore(f);
    if (!best || score > best.score) best = { file: f, score };
  }
  return best?.file ?? files[0];
}

// ---------- 2. Teks hook ----------
async function makeHook(title) {
  if (opt['no-llm']) return title;
  const prompt = `Buat teks thumbnail untuk klip video short-form.
Judul klip: ${title}

Aturan:
- 2 sampai 5 kata, singkat, memancing rasa penasaran.
- Bahasa yang sama dengan judul. Tanpa emoji, tanpa tanda kutip.
- Jangan membocorkan seluruh isi/jawabannya.

Balas HANYA JSON: {"hook":""}`;
  try {
    const res = await fetch(`${opt.ollama}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: opt.model,
        stream: false,
        format: 'json',
        options: { temperature: 0.4 },
        messages: [{ role: 'user', content: prompt }],
      }),
    });
    if (!res.ok) throw new Error(`Ollama error ${res.status}`);
    const data = await res.json();
    const hook = String(JSON.parse(data.message?.content ?? '{}').hook ?? '')
      .replace(/["“”]/g, '')
      .replace(/\s+/g, ' ')
      .trim();
    const words = hook.split(' ').length;
    if (!hook || words > 7 || hook.length > 45) return title;
    return hook;
  } catch (err) {
    console.warn(`  ! Hook dari LLM gagal (${err.message}), pakai nama klip.`);
    return title;
  }
}

// ---------- 3. Gambar ----------
function wrapWords(text, maxChars) {
  const lines = [];
  let cur = '';
  for (const w of text.split(' ')) {
    if (cur && (cur + ' ' + w).length > maxChars) {
      lines.push(cur);
      cur = w;
    } else {
      cur = cur ? cur + ' ' + w : w;
    }
  }
  if (cur) lines.push(cur);
  return lines;
}

function overlaySvg(text) {
  const maxSize = Number(opt['text-size']) || 120;
  const boxW = 940; // lebar area teks
  const charW = 0.68; // perkiraan lebar rata-rata huruf kapital tebal (x ukuran font)
  const upper = text.toLocaleUpperCase('id-ID');

  // Bungkus teks maksimal 4 baris, lalu kecilkan font supaya baris terpanjang muat
  let maxChars = Math.max(6, Math.floor(boxW / (maxSize * charW)));
  let lines = wrapWords(upper, maxChars);
  while (lines.length > 4) {
    maxChars += 2;
    lines = wrapWords(upper, maxChars);
  }
  const longest = Math.max(...lines.map((l) => l.length));
  const fs = Math.floor(Math.min(maxSize, boxW / (longest * charW)));
  const lh = fs * 1.12;
  const frac = Math.min(0.9, Math.max(0.1, Number(opt['text-y']) || 0.55));
  const y0 = H * frac - (lines.length * lh) / 2 + fs * 0.85;
  const accent = opt.accent && opt.accent !== 'none' ? opt.accent : '#FFFFFF';
  const family = `${esc(opt.font)}, Arial, sans-serif`;

  const common = `x="${W / 2}" text-anchor="middle" font-family="${family}" font-weight="900" font-size="${fs}"`;
  const stroke = lines
    .map((l, i) => `<text ${common} y="${(y0 + i * lh).toFixed(1)}" fill="#000" stroke="#000" stroke-width="${(fs * 0.18).toFixed(1)}" stroke-linejoin="round">${esc(l)}</text>`)
    .join('');
  const fill = lines
    .map((l, i) => `<text ${common} y="${(y0 + i * lh).toFixed(1)}" fill="${i === lines.length - 1 ? accent : '#FFFFFF'}">${esc(l)}</text>`)
    .join('');

  return `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#000" stop-opacity="0"/>
      <stop offset="1" stop-color="#000" stop-opacity="0.75"/>
    </linearGradient>
  </defs>
  <rect x="0" y="${Math.round(H * 0.33)}" width="${W}" height="${H - Math.round(H * 0.33)}" fill="url(#g)"/>
  ${stroke}${fill}
</svg>`;
}

// Blur pita subtitle dengan tepi memudar (supaya tidak ada garis potongan)
async function blurredSubBand(base) {
  const { top, height } = SUB_BAND;
  const band = await sharp(base).extract({ left: 0, top, width: W, height }).blur(30).ensureAlpha().toBuffer();
  const mask = Buffer.from(`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${height}">
    <defs><linearGradient id="m" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#fff" stop-opacity="0"/>
      <stop offset="0.25" stop-color="#fff" stop-opacity="1"/>
      <stop offset="0.75" stop-color="#fff" stop-opacity="1"/>
      <stop offset="1" stop-color="#fff" stop-opacity="0"/>
    </linearGradient></defs>
    <rect width="${W}" height="${height}" fill="url(#m)"/></svg>`);
  const feathered = await sharp(band).composite([{ input: mask, blend: 'dest-in' }]).png().toBuffer();
  return { input: feathered, left: 0, top };
}

async function compose(frame, text, outFile) {
  // Klip sudah 9:16, jadi cukup dipastikan 1080x1920 (kalau ukurannya beda, crop ke area paling menarik)
  const base = await sharp(frame)
    .resize(W, H, { fit: 'cover', position: sharp.strategy.attention })
    .toBuffer();
  const layers = [];
  if (!opt['keep-subs']) layers.push(await blurredSubBand(base));
  layers.push({ input: Buffer.from(overlaySvg(text)) });
  await sharp(base).composite(layers).jpeg({ quality: 92 }).toFile(outFile);
}

// ---------- Main ----------
// Proses satu folder hasil (output/<judul video>): mengembalikan jumlah thumbnail yang dibuat
async function processFolder(dir, tmpRoot) {
  console.log(`\n=== ${path.basename(dir)} ===`);
  const videos = (await fs.readdir(dir)).filter((f) => /\.mp4$/i.test(f)).sort();
  if (!videos.length) {
    console.warn('  ! Tidak ada file .mp4, dilewati.');
    return 0;
  }
  let made = 0;
  for (let i = 0; i < videos.length; i++) {
    const title = path.parse(videos[i]).name; // nama klip = judul
    const outFile = path.join(dir, `${title}.jpg`);
    if (!opt.force && (await fs.stat(outFile).catch(() => null))) {
      console.log(`• ${i + 1}/${videos.length}: ${title} (sudah ada, dilewati; pakai --force untuk buat ulang)`);
      continue;
    }
    console.log(`• Thumbnail ${i + 1}/${videos.length}: ${title}`);
    try {
      const frames = await extractFrames(path.join(dir, videos[i]), tmpRoot);
      if (!frames.length) {
        console.warn('  ! Tidak ada frame yang bisa diambil, dilewati.');
        continue;
      }
      const frame = await bestFrame(frames);
      const hook = await makeHook(title);
      console.log(`  teks: ${hook}`);
      await compose(frame, hook, outFile);
      made++;
    } catch (err) {
      console.warn(`  ! Gagal: ${err.message}`);
    }
  }
  return made;
}

async function main() {
  let folders;
  if (arg) {
    // Satu folder saja: nama folder di ./output, atau path langsung
    const direct = path.resolve(arg);
    const isDir = (await fs.stat(direct).catch(() => null))?.isDirectory();
    folders = [isDir ? direct : path.resolve(opt.out, arg)];
  } else {
    // Tanpa argumen: semua subfolder di ./output
    const root = path.resolve(opt.out);
    const entries = await fs.readdir(root, { withFileTypes: true }).catch(() => {
      throw new Error(`Folder output tidak ditemukan: ${root}. Jalankan dari folder yang sama dengan clipper.mjs.`);
    });
    folders = entries.filter((e) => e.isDirectory()).map((e) => path.join(root, e.name));
    if (!folders.length) throw new Error(`Tidak ada folder di ${root}. Jalankan clipper.mjs dulu.`);
  }
  console.log(`• ${folders.length} folder akan diproses`);

  const tmpRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'thumb-'));
  let total = 0;
  try {
    for (const f of folders) {
      try {
        total += await processFolder(f, tmpRoot);
      } catch (err) {
        console.warn(`  ! ${path.basename(f)} gagal: ${err.message}`);
      }
    }
  } finally {
    await fs.rm(tmpRoot, { recursive: true, force: true });
  }
  console.log(`\nSelesai. ${total} thumbnail dibuat di ${folders.length} folder.`);
}

main().catch((e) => {
  console.error('\nERROR:', e.message);
  process.exit(1);
});
