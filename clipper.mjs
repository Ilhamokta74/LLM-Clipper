#!/usr/bin/env node
// Clipper offline: YouTube -> transkrip (whisper.cpp) -> pilih momen (Ollama) -> potong (ffmpeg)
// Butuh: Node 18+, yt-dlp, ffmpeg, whisper.cpp (whisper-cli), Ollama

import { spawn, execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { parseArgs } from 'node:util';
import fs from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const execFileP = promisify(execFile);

// ---------- Konfigurasi ----------
const { values: opt, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    model: { type: 'string', default: process.env.OLLAMA_MODEL || 'qwen2.5:7b' },
    lang: { type: 'string', default: 'id' },
    ctx: { type: 'string', default: '16384' }, // num_ctx Ollama (potongan transkrip panjang butuh konteks besar)
    'whisper-bin': { type: 'string', default: process.env.WHISPER_BIN || 'whisper-cli' },
    'whisper-model': { type: 'string', default: process.env.WHISPER_MODEL || './models/ggml-large-v3-turbo-q5_0.bin' },
    ollama: { type: 'string', default: process.env.OLLAMA_URL || 'http://localhost:11434' },
    chunk: { type: 'string', default: '900' }, // detik per potongan transkrip untuk LLM (antar potongan dibuat bertumpuk)
    horizontal: { type: 'boolean', default: false }, // default: crop 9:16
    // layout 9:16: crop = potong tengah | fit = video utuh + background blur | follow = ikuti pembicara (butuh Python + mediapipe)
    layout: { type: 'string', default: 'crop' },
    python: { type: 'string', default: process.env.PYTHON || (process.platform === 'win32' ? 'python' : 'python3') },
    'track-model': { type: 'string', default: process.env.FACE_MODEL || '' }, // path face_landmarker.task (kosong = ./models di sebelah facetrack.py)
    'min-score': { type: 'string', default: '0' }, // skor akhir minimal (0 = tanpa batas skor)
    clips: { type: 'string', default: '0' }, // jumlah klip maksimal, diambil dari skor tertinggi (0 = semua yang lolos)
    nvenc: { type: 'boolean', default: false }, // encode pakai GPU NVIDIA
    'no-subs': { type: 'boolean', default: false },
    font: { type: 'string', default: 'Arial' }, // font subtitle (harus terinstall di Windows)
    'sub-line': { type: 'string', default: '4' }, // maksimal kata per baris subtitle (layar = 2 baris)
    'sub-size': { type: 'string', default: '13' }, // ukuran font subtitle
    'sub-min': { type: 'string', default: '1' }, // durasi tampil minimal per layar subtitle (detik); makin besar makin lambat gantinya
    'sub-margin': { type: 'string', default: '0' }, // jarak subtitle dari tepi bawah (0 = default: 90 vertikal / 22 horizontal); makin besar makin ke atas
    raw: { type: 'string', default: './video mentah' }, // video mentah + file kerja (audio, transkrip, srt)
    out: { type: 'string', default: './output' }, // klip hasil produksi
  },
});

const url = positionals[0];
if (!url) {
  console.error('Pemakaian: node clipper.mjs <url-youtube> [--lang id] [--model qwen2.5:7b] [--layout crop|fit|follow] [--nvenc] [--horizontal] [--no-subs]');
  process.exit(1);
}

if (!['crop', 'fit', 'follow'].includes(opt.layout)) {
  console.error(`--layout harus crop, fit, atau follow (bukan "${opt.layout}")`);
  process.exit(1);
}
const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));

// Skor minimal dan jumlah klip diatur lewat --min-score dan --clips (default: tanpa batas). Durasi dikunci 30-200 detik
const MIN_SCORE = Number(opt['min-score']) || 0;
const MAX_CLIPS = Math.floor(Number(opt.clips)) || 0;
const MIN_DUR = 30;
const MAX_DUR = 200;
const NUM_CTX = Number(opt.ctx) || 16384;
const CHUNK_SEC = Number(opt.chunk);
// Potongan transkrip saling menumpuk supaya klip panjang yang melewati batas potongan tetap utuh di salah satunya
const OVERLAP_SEC = Math.min(Math.max(120, MAX_DUR + 30), Math.floor(CHUNK_SEC / 2));
const LINE_WORDS = Math.max(1, Math.floor(Number(opt['sub-line'])) || 5);
const SUB_WORDS = LINE_WORDS * 2; // maksimal 2 baris per layar
const SUB_MIN = Math.max(0.1, Number(opt['sub-min']) || 1.8);

// Ukuran subtitle mengikuti resolusi hasil: 1080x1920 (9:16) atau 1920x1080 (--horizontal)
// Skala skrip ASS dibuat kecil (384 px sisi panjang) supaya font size 11 terlihat wajar, bukan raksasa/mikroskopis.
// Hasil akhir tetap 1080x1920 / 1920x1080; libass otomatis memperbesar sesuai rasio.
const SUB_SIZE = Number(opt['sub-size']) || 11;
const SUB = opt.horizontal
  ? { w: 384, h: 216, size: SUB_SIZE, outline: 0.8, marginV: Number(opt['sub-margin']) || 22, marginH: 12, maxChars: 999 }
  : { w: 216, h: 384, size: SUB_SIZE, outline: 0.8, marginV: Number(opt['sub-margin']) || 90, marginH: 12, maxChars: 999 }; // marginV = jarak dari tepi bawah; makin besar makin ke atas

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

// Nama file hasil (.mp4): murni judul klip, spasi dipertahankan (tanpa awalan clip_01_ dan tanpa underscore)
const prettyName = (s) =>
  s.replace(/[^\p{L}\p{N}\s-]/gu, '').replace(/\s+/g, ' ').trim().slice(0, 60).trim() || 'Klip';

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
// Tahap 1: LLM mengusulkan kandidat per potongan transkrip (potongan saling bertumpuk, jumlah dan durasi bebas).
// Tahap 2: tiap kandidat dinilai ulang dengan rubrik terpisah (hook, utuh, nilai, emosi).
// Tahap 3: produksi semua kandidat yang skor akhirnya >= MIN_SCORE (jumlah tidak dibatasi).
function chunkSegments(segs) {
  if (!segs.length) return [];
  const end = segs.at(-1).end;
  const stride = Math.max(60, CHUNK_SEC - OVERLAP_SEC);
  const chunks = [];
  for (let w = segs[0].start; w < end; w += stride) {
    const part = segs.filter((s) => s.start >= w && s.start < w + CHUNK_SEC);
    if (part.length) chunks.push(part);
    if (w + CHUNK_SEC >= end) break;
  }
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

async function llmJson(prompt) {
  const res = await fetch(`${opt.ollama}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: opt.model,
      stream: false,
      format: 'json',
      options: { temperature: 0.2, num_ctx: NUM_CTX },
      messages: [{ role: 'user', content: prompt }],
    }),
  });
  if (!res.ok) throw new Error(`Ollama error ${res.status}: ${await res.text()}`);
  const data = await res.json();
  return extractJson(data.message?.content ?? '');
}

async function askLlm(chunk) {
  const transcript = chunk.map((s) => `[${Math.round(s.start)}] ${s.text}`).join('\n');
  const prompt = `Kamu editor video short-form (TikTok/Reels/Shorts) yang sangat selektif. Di bawah ini transkrip sebagian video.
Setiap baris diawali detik awal dalam kurung siku, misalnya [125].

Pilih HANYA bagian yang benar-benar layak berdiri sendiri sebagai klip. Boleh nol, boleh satu, boleh beberapa. Jangan memaksakan jumlah; bagian yang biasa saja jangan dipilih.
Aturan:
- Durasi mengikuti isi: satu ide atau cerita utuh dari awal sampai punchline/kesimpulan. Minimal ${MIN_DUR} detik, maksimal ${MAX_DUR} detik. Jangan dipotong paksa dan jangan diulur-ulur.
- Kalimat pertama harus menarik perhatian (hook): pertanyaan, klaim berani, cerita, atau fakta mengejutkan.
- Jangan mulai atau berhenti di tengah kalimat. Jangan tumpang tindih antar klip.
- "start" dan "end" dalam detik (angka), memakai angka dari transkrip.
- "title" singkat dan menarik, dalam bahasa yang sama dengan transkrip.
- "score" 1-10 untuk potensi viral; beri 8 ke atas hanya untuk yang luar biasa.

Balas HANYA JSON dengan format:
{"clips":[{"start":0,"end":0,"title":"","reason":"","score":0}]}
Kalau tidak ada yang layak: {"clips":[]}

TRANSKRIP:
${transcript}`;
  const parsed = await llmJson(prompt);
  return Array.isArray(parsed?.clips) ? parsed.clips : [];
}

function snapToSegments(segs, start, end) {
  const a = segs.findLast((x) => x.start <= start) ?? segs[0];
  const b = segs.find((x) => x.end >= end) ?? segs.at(-1);
  return { start: a.start, end: b.end };
}

const textBetween = (segs, start, end) =>
  segs.filter((s) => s.end > start && s.start < end).map((s) => s.text).join(' ');

// Penilaian ulang per kandidat dengan rubrik; mengembalikan skor 1-10 atau null kalau LLM gagal
async function rateClip(segs, c) {
  const text = textBetween(segs, c.start, c.end).slice(0, 8000);
  const prompt = `Kamu penilai konten short-form yang ketat. Nilai transkrip klip berikut (durasi ${Math.round(c.end - c.start)} detik).
Beri nilai bulat 1-10 untuk tiap kriteria. Nilai 7 ke atas hanya untuk yang benar-benar bagus; klip yang biasa saja = 4-5.
- hook: apakah beberapa detik pertama langsung menarik perhatian orang yang belum tahu konteks
- utuh: apakah ide/ceritanya lengkap, bisa dipahami berdiri sendiri, dan berakhir rapi
- nilai: apakah isinya berharga (insight, fakta, cerita, humor, atau opini kuat)
- emosi: apakah memicu penasaran, tawa, kaget, atau keinginan untuk membagikan

Balas HANYA JSON: {"hook":0,"utuh":0,"nilai":0,"emosi":0}

TRANSKRIP:
${text}`;
  const j = await llmJson(prompt);
  if (!j) return null;
  const n = (k) => Math.min(10, Math.max(1, Number(j[k]) || 1));
  return 0.3 * n('hook') + 0.3 * n('utuh') + 0.25 * n('nilai') + 0.15 * n('emosi');
}

const clock = (sec) => {
  const s = Math.round(sec);
  const h = Math.floor(s / 3600);
  const m = String(Math.floor((s % 3600) / 60)).padStart(2, '0');
  const r = String(s % 60).padStart(2, '0');
  return h ? `${h}:${m}:${r}` : `${m}:${r}`;
};

function dropOverlaps(sorted) {
  const kept = [];
  for (const c of sorted) {
    if (kept.some((x) => c.start < x.end && c.end > x.start)) continue;
    kept.push(c);
  }
  return kept;
}

async function pickClips(segs, workDir) {
  const chunks = chunkSegments(segs);
  const cand = [];
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
        cand.push({
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

  // Potongan bertumpuk bisa menemukan momen yang sama dua kali: sisakan yang skor tahap 1-nya tertinggi
  const pool = dropOverlaps(cand.sort((a, b) => b.score - a.score));
  console.log(`• ${pool.length} kandidat, dinilai ulang satu per satu...`);
  for (let i = 0; i < pool.length; i++) {
    const c = pool[i];
    try {
      c.rubric = await rateClip(segs, c);
    } catch (err) {
      console.warn(`  ! Gagal menilai kandidat ${i + 1}: ${err.message}`);
      c.rubric = null;
    }
    // 70% penilaian rubrik, 30% skor usulan; kalau rubrik gagal pakai skor usulan saja
    c.final = Math.round((c.rubric == null ? c.score : 0.7 * c.rubric + 0.3 * c.score) * 10) / 10;
  }
  pool.sort((a, b) => b.final - a.final);

  // pool sudah urut skor tertinggi; terapkan batas skor (jika ada) lalu ambil N teratas (jika --clips diisi)
  let chosen = pool.filter((c) => c.final >= MIN_SCORE);
  if (MAX_CLIPS > 0) chosen = chosen.slice(0, MAX_CLIPS);
  const ids = new Set(chosen);
  for (const c of pool) {
    console.log(`  ${ids.has(c) ? '✔' : '✘'} ${c.final.toFixed(1).padStart(4)}  ${clock(c.start)}-${clock(c.end)} (${Math.round(c.end - c.start)} dtk)  ${c.title}`);
  }
  await fs.writeFile(path.join(workDir, 'kandidat.json'), JSON.stringify(pool, null, 2), 'utf8');
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
  // Satu baris subtitle per grup kata, tanpa sorotan warna
  groups.forEach((g, gi) => {
    const nextGroupStart = groups[gi + 1]?.[0].start ?? dur;
    const first = g[0];
    const last = g.at(-1);
    const start = first.start;
    // Tampil minimal SUB_MIN detik (supaya sempat terbaca), tapi tidak boleh menimpa grup berikutnya
    const wanted = Math.max(last.end + 0.3, start + SUB_MIN);
    const rawEnd = Math.min(wanted, nextGroupStart);
    const end = Math.min(Math.max(rawEnd, start + 0.1), dur);
    if (end <= start) return;
    // Maks 2 baris, tiap baris maks LINE_WORDS kata; kalau lebih dari satu baris, dibagi seimbang
    const ws = g.map((x) => x.text);
    const text = ws.length <= LINE_WORDS
      ? ws.join(' ')
      : ws.slice(0, Math.ceil(ws.length / 2)).join(' ') + '\\N' + ws.slice(Math.ceil(ws.length / 2)).join(' ');
    lines.push(`Dialogue: 0,${fmtAss(start)},${fmtAss(end)},Default,,0,0,0,,${text}`);
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
    `Style: Default,${opt.font},${SUB.size},${ASS_WHITE},${ASS_WHITE},${ASS_BLACK},${ASS_SHADOW},-1,0,0,0,100,100,0,0,1,${SUB.outline},0.5,2,${SUB.marginH},${SUB.marginH},${SUB.marginV},1`,
    '',
    '[Events]',
    'Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text',
  ];
  await fs.writeFile(file, [...header, ...lines, ''].join('\n'), 'utf8');
}

// ---------- Layout follow: crop mengikuti pembicara (facetrack.py) ----------
async function trackClip(workDir, videoName, c, trackFile) {
  const args = [
    path.join(SCRIPT_DIR, 'facetrack.py'), videoName,
    '--start', c.start.toFixed(2),
    '--dur', (c.end - c.start).toFixed(2),
    '--out', trackFile,
  ];
  if (opt['track-model']) args.push('--model', path.resolve(opt['track-model']));
  await run(opt.python, args, { cwd: workDir });
  return JSON.parse(await fs.readFile(path.join(workDir, trackFile), 'utf8'));
}

// Segmen [t0, t1, x0, x1] (detik, piksel) -> ekspresi x untuk filter crop ffmpeg (bentuk penjumlahan, tanpa nested if)
function followCropFilter(tr) {
  const f = (n) => Number(n).toFixed(3);
  const terms = tr.segments.map(([t0, t1, x0, x1], i) => {
    const last = i === tr.segments.length - 1;
    const cond = [`gte(t,${f(t0)})`, ...(last ? [] : [`lt(t,${f(t1)})`])].join('*');
    const val =
      Math.abs(x1 - x0) < 0.5 || t1 <= t0
        ? f(x0)
        : `(${f(x0)}+(${f(x1 - x0)})*(t-${f(t0)})/${f(t1 - t0)})`;
    return `${cond}*${val}`;
  });
  return `crop=${tr.cw}:ih:x='${terms.join('+')}':y=0`;
}

async function cutClips(workDir, outDir, videoName, segs, clips) {
  await fs.mkdir(outDir, { recursive: true });
  const usedNames = new Set(); // cegah dua klip berjudul sama saling menimpa
  for (let i = 0; i < clips.length; i++) {
    const c = clips[i];
    // base = nama file kerja (.srt/.ass/.track.json), tetap tanpa spasi supaya aman di filter ffmpeg
    const base = `clip_${String(i + 1).padStart(2, '0')}_${safeName(c.title)}`;
    // outName = nama file hasil: judul saja, pakai spasi
    const pretty = prettyName(c.title);
    let outName = pretty;
    for (let n = 2; usedNames.has(outName.toLowerCase()); n++) outName = `${pretty} ${n}`;
    usedNames.add(outName.toLowerCase());
    const srtName = `${base}.srt`;
    await buildSrt(segs, c, path.join(workDir, srtName));
    const assName = `${base}.ass`;
    await buildAss(segs, c, path.join(workDir, assName));

    const subFilter = opt['no-subs'] ? null : `ass=${assName}`; // cwd=workDir supaya path .ass relatif (hindari masalah escape di Windows)
    const centerCrop = ["crop='min(iw,ih*9/16)':'min(ih,iw*16/9)'", 'scale=1080:1920'];

    let layout = opt.horizontal ? 'horizontal' : opt.layout;
    let filters = []; // dipakai dengan -vf
    let complex = null; // dipakai dengan -filter_complex (layout fit)

    if (layout === 'follow') {
      try {
        console.log(`• Analisis wajah/pembicara klip ${i + 1}/${clips.length}...`);
        const tr = await trackClip(workDir, videoName, c, `${base}.track.json`);
        console.log(`  shot: ${tr.cuts.length + 1}, segmen crop: ${tr.segments.length}`);
        filters = [followCropFilter(tr), 'scale=1080:1920'];
      } catch (err) {
        console.warn(`  ! Analisis pembicara gagal (${err.message}). Pakai crop tengah untuk klip ini.`);
        layout = 'crop';
      }
    }
    if (layout === 'crop') filters = [...centerCrop];
    if (layout === 'fit') {
      complex =
        '[0:v]split=2[bg][fg];' +
        '[bg]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=40:6[b];' +
        '[fg]scale=1080:-2[f];' +
        `[b][f]overlay=(W-w)/2:(H-h)/2${subFilter ? '[v0];[v0]' + subFilter : ''}[v]`;
    } else if (subFilter) {
      filters.push(subFilter);
    }

    const outFile = path.resolve(outDir, `${outName}.mp4`);
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
        ...(complex ? ['-filter_complex', complex, '-map', '[v]', '-map', '0:a?'] : filters.length ? ['-vf', filters.join(',')] : []),
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

  const clips = await pickClips(segs, rawDir);
  if (!clips.length) {
    console.log('Tidak ada klip yang lolos. Lihat kandidat.json di folder mentah untuk skor tiap kandidat, coba model lain, atau cek bahasa (--lang).');
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
