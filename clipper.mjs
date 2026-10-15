#!/usr/bin/env node
// Clipper offline: YouTube -> transkrip (whisper.cpp) -> pilih momen (Ollama) -> potong (ffmpeg)
// Butuh: Node 18+, yt-dlp, ffmpeg, whisper.cpp (whisper-cli), Ollama

import { spawn, execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { parseArgs } from 'node:util';
import fs from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';

const execFileP = promisify(execFile);

// ---------- Konfigurasi ----------
const { values: opt, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    model: { type: 'string', default: process.env.OLLAMA_MODEL || 'qwen2.5:7b' },
    lang: { type: 'string', default: 'id' },
    clips: { type: 'string', default: '5' },
    min: { type: 'string', default: '30' },
    max: { type: 'string', default: '60' },
    'whisper-bin': { type: 'string', default: process.env.WHISPER_BIN || 'whisper-cli' },
    'whisper-model': { type: 'string', default: process.env.WHISPER_MODEL || './models/ggml-large-v3-turbo-q5_0.bin' },
    ollama: { type: 'string', default: process.env.OLLAMA_URL || 'http://localhost:11434' },
    chunk: { type: 'string', default: '360' }, // detik per potongan transkrip untuk LLM
    horizontal: { type: 'boolean', default: false }, // default: crop 9:16
    nvenc: { type: 'boolean', default: false }, // encode pakai GPU NVIDIA
    'no-subs': { type: 'boolean', default: false },
    raw: { type: 'string', default: './video mentah' }, // video mentah + file kerja (audio, transkrip, srt)
    out: { type: 'string', default: './output' }, // klip hasil produksi
  },
});

const url = positionals[0];
if (!url) {
  console.error('Pemakaian: node clipper.mjs <url-youtube> [--lang id] [--clips 5] [--min 30] [--max 60] [--model qwen2.5:7b] [--nvenc] [--horizontal] [--no-subs]');
  process.exit(1);
}

const N_CLIPS = Number(opt.clips);
const MIN_DUR = Number(opt.min);
const MAX_DUR = Number(opt.max);
const CHUNK_SEC = Number(opt.chunk);

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

const fmtSrt = (sec) => {
  const ms = Math.max(0, Math.round(sec * 1000));
  const h = String(Math.floor(ms / 3600000)).padStart(2, '0');
  const m = String(Math.floor((ms % 3600000) / 60000)).padStart(2, '0');
  const s = String(Math.floor((ms % 60000) / 1000)).padStart(2, '0');
  const r = String(ms % 1000).padStart(3, '0');
  return `${h}:${m}:${s},${r}`;
};

const safeName = (s) =>
  s.replace(/[^\p{L}\p{N}\s-]/gu, '').trim().replace(/\s+/g, '_').slice(0, 60) || 'clip';

// Nama folder dari judul video: spasi dipertahankan, karakter terlarang Windows dibuang
const folderName = (s) =>
  s
    .replace(/[\\/:*?"<>|%\u0000-\u001f]/g, '')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 80)
    .replace(/[. ]+$/, '') || 'video';

// ---------- 1. Download ----------
async function download(videoPath) {
  if (existsSync(videoPath)) {
    console.log('• Video sudah ada, lewati download.');
    return videoPath;
  }
  console.log('• Download video...');
  await run('yt-dlp', [
    '--no-playlist',
    '-f', 'bv*[height<=1080]+ba/b',
    '--merge-output-format', 'mp4',
    '-o', videoPath.replace(/\.mp4$/, '') + '.%(ext)s',
    url,
  ]);
  return videoPath;
}

// ---------- 2. Transkrip ----------
async function transcribe(workDir, videoPath) {
  const jsonPath = path.join(workDir, 'transcript.json');
  if (!existsSync(jsonPath)) {
    const wav = path.join(workDir, 'audio.wav');
    console.log('• Ekstrak audio...');
    await run('ffmpeg', ['-y', '-loglevel', 'error', '-i', videoPath, '-vn', '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', wav]);
    console.log('• Transkrip dengan whisper.cpp (bisa lama untuk video panjang)...');
    await run(opt['whisper-bin'], [
      '-m', opt['whisper-model'],
      '-f', wav,
      '-l', opt.lang,
      '-ml', '42', '-sow', // segmen pendek, pas untuk subtitle
      '-oj',
      '-of', path.join(workDir, 'transcript'),
    ]);
  } else {
    console.log('• Transkrip sudah ada, lewati.');
  }
  const data = JSON.parse(await fs.readFile(jsonPath, 'utf8'));
  return data.transcription
    .map((s) => ({ start: s.offsets.from / 1000, end: s.offsets.to / 1000, text: s.text.trim() }))
    .filter((s) => s.text && !/^\[.*\]$/.test(s.text)); // buang [MUSIC] dll
}

// ---------- 3. Pilih momen dengan LLM ----------
function chunkSegments(segs) {
  const chunks = [];
  let cur = [];
  for (const s of segs) {
    if (cur.length && s.start - cur[0].start >= CHUNK_SEC) {
      chunks.push(cur);
      cur = [];
    }
    cur.push(s);
  }
  if (cur.length) chunks.push(cur);
  return chunks;
}

function extractJson(text) {
  try {
    return JSON.parse(text);
  } catch {
    const m = text.match(/\{[\s\S]*\}/);
    if (m) {
      try { return JSON.parse(m[0]); } catch { /* lanjut */ }
    }
  }
  return null;
}

async function askLlm(chunk) {
  const transcript = chunk.map((s) => `[${Math.round(s.start)}] ${s.text}`).join('\n');
  const prompt = `Kamu editor video short-form (TikTok/Reels/Shorts). Di bawah ini transkrip sebagian video.
Setiap baris diawali detik awal dalam kurung siku, misalnya [125].

Pilih sampai 3 bagian terbaik untuk dijadikan klip pendek.
Aturan:
- Durasi tiap klip antara ${MIN_DUR} dan ${MAX_DUR} detik (end - start).
- Kalimat pertama harus menarik perhatian (hook): pertanyaan, klaim berani, cerita, atau fakta mengejutkan.
- Ide harus utuh dan jelas, jangan mulai atau berhenti di tengah kalimat.
- Jangan tumpang tindih antar klip.
- Kalau tidak ada bagian yang layak, kembalikan daftar kosong.
- "start" dan "end" dalam detik (angka), memakai angka dari transkrip.
- "title" singkat dan menarik, dalam bahasa yang sama dengan transkrip.
- "score" 1-10 untuk potensi viral.

Balas HANYA JSON dengan format:
{"clips":[{"start":0,"end":0,"title":"","reason":"","score":0}]}

TRANSKRIP:
${transcript}`;

  const res = await fetch(`${opt.ollama}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: opt.model,
      stream: false,
      format: 'json',
      options: { temperature: 0.3, num_ctx: 8192 },
      messages: [{ role: 'user', content: prompt }],
    }),
  });
  if (!res.ok) throw new Error(`Ollama error ${res.status}: ${await res.text()}`);
  const data = await res.json();
  const parsed = extractJson(data.message?.content ?? '');
  return Array.isArray(parsed?.clips) ? parsed.clips : [];
}

function snapToSegments(segs, start, end) {
  const a = segs.findLast((x) => x.start <= start) ?? segs[0];
  const b = segs.find((x) => x.end >= end) ?? segs.at(-1);
  return { start: a.start, end: b.end };
}

async function pickClips(segs) {
  const chunks = chunkSegments(segs);
  let candidates = [];
  for (let i = 0; i < chunks.length; i++) {
    console.log(`• LLM menganalisis bagian ${i + 1}/${chunks.length}...`);
    try {
      const found = await askLlm(chunks[i]);
      for (const c of found) {
        const s = Number(c.start), e = Number(c.end);
        if (!Number.isFinite(s) || !Number.isFinite(e) || e <= s) continue;
        const snapped = snapToSegments(segs, s, e);
        const dur = snapped.end - snapped.start;
        if (dur < MIN_DUR - 5 || dur > MAX_DUR + 15) continue;
        candidates.push({
          ...snapped,
          title: String(c.title || 'Klip'),
          reason: String(c.reason || ''),
          score: Number(c.score) || 0,
        });
      }
    } catch (err) {
      console.warn(`  ! Gagal di bagian ${i + 1}: ${err.message}`);
    }
  }
  // Urutkan dari skor tertinggi, buang yang tumpang tindih
  candidates.sort((a, b) => b.score - a.score);
  const chosen = [];
  for (const c of candidates) {
    if (chosen.some((x) => c.start < x.end && c.end > x.start)) continue;
    chosen.push(c);
    if (chosen.length >= N_CLIPS) break;
  }
  return chosen.sort((a, b) => a.start - b.start);
}

// ---------- 4. Potong + subtitle ----------
async function buildSrt(segs, clip, file) {
  const lines = [];
  let n = 1;
  for (const s of segs) {
    if (s.end <= clip.start || s.start >= clip.end) continue;
    const a = Math.max(s.start, clip.start) - clip.start;
    const b = Math.min(s.end, clip.end) - clip.start;
    lines.push(`${n++}\n${fmtSrt(a)} --> ${fmtSrt(b)}\n${s.text}\n`);
  }
  await fs.writeFile(file, lines.join('\n'), 'utf8');
}

async function cutClips(workDir, outDir, videoName, segs, clips) {
  await fs.mkdir(outDir, { recursive: true });
  for (let i = 0; i < clips.length; i++) {
    const c = clips[i];
    const base = `clip_${String(i + 1).padStart(2, '0')}_${safeName(c.title)}`;
    const srtName = `${base}.srt`;
    await buildSrt(segs, c, path.join(workDir, srtName));

    const filters = [];
    if (!opt.horizontal) {
      filters.push("crop='min(iw,ih*9/16)':'min(ih,iw*16/9)'", 'scale=1080:1920');
    }
    if (!opt['no-subs']) {
      // ffmpeg dijalankan dengan cwd=workDir supaya path .srt relatif (hindari masalah escape di Windows)
      filters.push(
        `subtitles=${srtName}:force_style='FontName=Arial,FontSize=11,Bold=1,Outline=2,Shadow=0,Alignment=2,MarginV=20'`
      );
    }

    const outFile = path.resolve(outDir, `${base}.mp4`);
    const venc = opt.nvenc
      ? ['-c:v', 'h264_nvenc', '-preset', 'p5', '-cq', '21']
      : ['-c:v', 'libx264', '-preset', 'fast', '-crf', '20'];

    console.log(`• Potong klip ${i + 1}/${clips.length}: ${c.title} (${Math.round(c.end - c.start)} dtk)`);
    await run(
      'ffmpeg',
      [
        '-y', '-loglevel', 'error', '-stats',
        '-ss', c.start.toFixed(2),
        '-i', videoName,
        '-t', (c.end - c.start).toFixed(2),
        ...(filters.length ? ['-vf', filters.join(',')] : []),
        ...venc,
        '-c:a', 'aac', '-b:a', '160k',
        '-movflags', '+faststart',
        outFile,
      ],
      { cwd: workDir }
    );
  }
}

// ---------- Main ----------
async function main() {
  const { stdout } = await execFileP('yt-dlp', [
    '--no-playlist', '--encoding', 'utf-8', '--print', 'id', '--print', 'title', url,
  ]).catch((e) => {
    throw new Error(`yt-dlp gagal: ${e.message}`);
  });
  const [id, ...titleParts] = stdout.trim().split(/\r?\n/);
  let name = folderName(titleParts.join(' ') || id);

  // Judul sama tapi video beda: tambahkan id supaya file kerja tidak tercampur
  try {
    const prev = await fs.readFile(path.resolve(opt.raw, name, 'sumber.txt'), 'utf8');
    if (prev.split(/\r?\n/)[0].trim() !== id) name = `${name} [${id}]`;
  } catch { /* folder baru */ }

  const rawDir = path.resolve(opt.raw, name); // video mentah + audio, transkrip, srt
  const outDir = path.resolve(opt.out, name); // klip hasil produksi
  await fs.mkdir(rawDir, { recursive: true });
  await fs.writeFile(path.join(rawDir, 'sumber.txt'), `${id}\n${url}\n`, 'utf8');
  console.log(`• Judul   : ${name}`);
  console.log(`• Mentah  : ${rawDir}`);
  console.log(`• Hasil   : ${outDir}`);

  const videoName = `${name}.mp4`;
  const videoPath = await download(path.join(rawDir, videoName));
  const segs = await transcribe(rawDir, videoPath);
  console.log(`• Transkrip: ${segs.length} segmen`);

  const clips = await pickClips(segs);
  if (!clips.length) {
    console.log('Tidak ada klip yang lolos. Coba model lain, turunkan --min, atau cek bahasa (--lang).');
    return;
  }
  await fs.mkdir(outDir, { recursive: true });
  await fs.writeFile(path.join(outDir, 'clips.json'), JSON.stringify(clips, null, 2));
  await cutClips(rawDir, outDir, videoName, segs, clips);
  console.log(`\nSelesai. ${clips.length} klip ada di: ${outDir}`);
}

main().catch((e) => {
  console.error('\nERROR:', e.message);
  process.exit(1);
});
