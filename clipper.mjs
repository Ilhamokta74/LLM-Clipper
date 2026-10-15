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
    font: { type: 'string', default: 'Arial' }, // font subtitle (harus terinstall di Windows)
    'sub-words': { type: 'string', default: '3' }, // maksimal kata per layar subtitle
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
const SUB_WORDS = Math.max(1, Number(opt['sub-words']) || 3);

// Ukuran subtitle mengikuti resolusi hasil: 1080x1920 (9:16) atau 1920x1080 (--horizontal)
const SUB = opt.horizontal
  ? { w: 1920, h: 1080, size: 58, outline: 4, marginV: 70, maxChars: 38 }
  : { w: 1080, h: 1920, size: 80, outline: 6, marginV: 300, maxChars: 20 }; // marginV = jarak dari tepi bawah (px); makin kecil makin ke bawah

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

// Gabungkan token whisper (potongan kata) menjadi kata utuh beserta waktu mulai/selesai
function tokensToWords(tokens) {
  const words = [];
  for (const t of tokens ?? []) {
    const raw = String(t.text ?? '');
    const piece = raw.trim();
    if (!piece || /^\[_.*\]$/.test(piece) || !t.offsets) continue; // lewati token spesial [_BEG_], [_TT_123]
    const from = t.offsets.from / 1000;
    const to = t.offsets.to / 1000;
    if (/^\s/.test(raw) || !words.length) {
      words.push({ start: from, end: to, text: piece });
    } else {
      const w = words.at(-1);
      w.text += piece;
      w.end = Math.max(w.end, to);
    }
  }
  return words;
}

// Pastikan timestamp per kata masuk akal; kalau tidak, kembalikan null (nanti dipakai perkiraan)
function validWords(words, seg) {
  if (!words.length) return null;
  for (let i = 0; i < words.length; i++) {
    const w = words[i];
    if (!(w.end >= w.start)) return null;
    if (w.start < seg.start - 0.5 || w.end > seg.end + 0.5) return null;
    if (i && w.start < words[i - 1].start - 0.05) return null;
  }
  return words;
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
      '-ojf', // JSON lengkap: ada timestamp per token, dipakai untuk subtitle per kata
      '-of', path.join(workDir, 'transcript'),
    ]);
  } else {
    console.log('• Transkrip sudah ada, lewati.');
  }
  const data = JSON.parse(await fs.readFile(jsonPath, 'utf8'));
  return data.transcription
    .map((s) => {
      const seg = { start: s.offsets.from / 1000, end: s.offsets.to / 1000, text: s.text.trim() };
      seg.words = validWords(tokensToWords(s.tokens), seg); // null kalau tidak ada timestamp per kata
      return seg;
    })
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

// ---------- Subtitle gaya short-form (ASS) ----------
// 2-3 kata per layar, besar dan tebal, kata yang sedang diucapkan disorot kuning
const ASS_WHITE = '&H00FFFFFF'; // format warna ASS: BGR
const ASS_BLACK = '&H00000000';
const ASS_SHADOW = '&H80000000';

const fmtAss = (sec) => {
  const cs = Math.max(0, Math.round(sec * 100));
  const h = Math.floor(cs / 360000);
  const m = String(Math.floor((cs % 360000) / 6000)).padStart(2, '0');
  const s = String(Math.floor((cs % 6000) / 100)).padStart(2, '0');
  const c = String(cs % 100).padStart(2, '0');
  return `${h}:${m}:${s}.${c}`;
};

function wordsOfSegment(seg) {
  if (seg.words) return seg.words;
  // Cadangan (transkrip lama tanpa timestamp per kata): bagi waktu menurut panjang kata
  const parts = seg.text.split(/\s+/).filter(Boolean);
  const total = parts.reduce((n, p) => n + p.length + 1, 0);
  let t = seg.start;
  return parts.map((p) => {
    const d = ((p.length + 1) / total) * (seg.end - seg.start);
    const w = { start: t, end: t + d, text: p };
    t += d;
    return w;
  });
}

function groupWords(words, maxWords, maxChars) {
  const groups = [];
  let cur = [];
  const chars = (g) => g.reduce((n, w) => n + w.text.length, 0) + Math.max(0, g.length - 1);
  words.forEach((w, i) => {
    if (cur.length && (cur.length >= maxWords || chars(cur) + 1 + w.text.length > maxChars)) {
      groups.push(cur);
      cur = [];
    }
    cur.push(w);
    const next = words[i + 1];
    // pisah grup di akhir kalimat atau saat ada jeda bicara
    if (/[.?!…]$/.test(w.raw) || (next && next.start - w.end > 0.5)) {
      groups.push(cur);
      cur = [];
    }
  });
  if (cur.length) groups.push(cur);
  return groups;
}

async function buildAss(segs, clip, file) {
  const dur = clip.end - clip.start;
  const clamp = (t) => Math.min(Math.max(t, 0), dur);

  const words = [];
  for (const s of segs) {
    if (s.end <= clip.start || s.start >= clip.end) continue;
    for (const w of wordsOfSegment(s)) {
      if (w.start < clip.start - 0.05 || w.start >= clip.end) continue;
      const raw = w.text.replace(/[{}\\]/g, '');
      const text = raw.replace(/[.,;:…]+$/, ''); // tanda baca di akhir kata dibuang, ? dan ! tetap
      if (!text) continue;
      const prev = words.at(-1);
      const start = Math.max(clamp(w.start - clip.start), prev ? prev.start : 0);
      words.push({ start, end: Math.max(clamp(w.end - clip.start), start), raw, text });
    }
  }

  const lines = [];
  const groups = groupWords(words, SUB_WORDS, SUB.maxChars);
  groups.forEach((g, gi) => {
    const nextGroupStart = groups[gi + 1]?.[0].start ?? dur;
    g.forEach((w, wi) => {
      const next = g[wi + 1];
      const rawEnd = next
        ? next.start
        : Math.min(Math.max(w.end, w.start + 0.15) + 0.25, nextGroupStart);
      const end = Math.min(Math.max(rawEnd, w.start + 0.06), dur);
      if (end <= w.start) return;
      const text = g
        .map((x, xi) => (xi === wi ? `{\\c&H00FFFF&}${x.text}{\\c&HFFFFFF&}` : x.text))
        .join(' ');
      lines.push(`Dialogue: 0,${fmtAss(w.start)},${fmtAss(end)},Default,,0,0,0,,${text}`);
    });
  });

  const header = [
    '[Script Info]',
    'ScriptType: v4.00+',
    `PlayResX: ${SUB.w}`,
    `PlayResY: ${SUB.h}`,
    'WrapStyle: 0',
    'ScaledBorderAndShadow: yes',
    '',
    '[V4+ Styles]',
    'Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding',
    `Style: Default,${opt.font},${SUB.size},${ASS_WHITE},${ASS_WHITE},${ASS_BLACK},${ASS_SHADOW},-1,0,0,0,100,100,0,0,1,${SUB.outline},2,2,60,60,${SUB.marginV},1`,
    '',
    '[Events]',
    'Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text',
  ];
  await fs.writeFile(file, [...header, ...lines, ''].join('\n'), 'utf8');
}

async function cutClips(workDir, outDir, videoName, segs, clips) {
  await fs.mkdir(outDir, { recursive: true });
  for (let i = 0; i < clips.length; i++) {
    const c = clips[i];
    const base = `clip_${String(i + 1).padStart(2, '0')}_${safeName(c.title)}`;
    const srtName = `${base}.srt`;
    await buildSrt(segs, c, path.join(workDir, srtName));
    const assName = `${base}.ass`;
    await buildAss(segs, c, path.join(workDir, assName));

    const filters = [];
    if (!opt.horizontal) {
      filters.push("crop='min(iw,ih*9/16)':'min(ih,iw*16/9)'", 'scale=1080:1920');
    }
    if (!opt['no-subs']) {
      // ffmpeg dijalankan dengan cwd=workDir supaya path .ass relatif (hindari masalah escape di Windows)
      filters.push(`ass=${assName}`);
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
