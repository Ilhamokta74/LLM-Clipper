#!/usr/bin/env node
// Kompres klip hasil clipper.mjs: baca semua .mp4 di folder output, hasilnya ke folder terpisah.
// Butuh: Node 18+, ffmpeg
//
// Contoh:
//   node compress                      -> ./output  ->  ./output-compressed
//   node compress --crf 30             -> lebih kecil, kualitas lebih turun
//   node compress --nvenc              -> pakai GPU NVIDIA (lebih cepat)
//   node compress --replace            -> timpa file asli di ./output (hanya kalau hasil lebih kecil)

import { spawn } from 'node:child_process';
import { parseArgs } from 'node:util';
import fs from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';

const { values: opt, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    in: { type: 'string', default: './output' }, // folder sumber (dibaca rekursif)
    out: { type: 'string', default: './output-compressed' }, // folder hasil kompres
    crf: { type: 'string', default: '26' }, // 18 = nyaris asli, 23 = standar, 28 = kecil, 32 = sangat kecil
    preset: { type: 'string', default: 'medium' }, // lebih lambat = file lebih kecil (slow, slower)
    abr: { type: 'string', default: '96k' }, // bitrate audio
    maxrate: { type: 'string', default: '' }, // batas bitrate video, contoh: 3M (opsional)
    height: { type: 'string', default: '' }, // perkecil resolusi, contoh: 1280 untuk 720x1280 (opsional)
    hevc: { type: 'boolean', default: false }, // H.265: lebih kecil ~30%, tapi tidak semua platform/perangkat nyaman
    nvenc: { type: 'boolean', default: false }, // encode pakai GPU NVIDIA
    replace: { type: 'boolean', default: false }, // timpa file asli kalau hasil lebih kecil
    force: { type: 'boolean', default: false }, // kompres ulang walau hasilnya sudah ada
  },
});

const CRF = Number(opt.crf);
if (!Number.isFinite(CRF)) {
  console.error('--crf harus berupa angka.');
  process.exit(1);
}

// ---------- Helper ----------
function run(cmd, args) {
  return new Promise((resolve, reject) => {
    const p = spawn(cmd, args, { stdio: 'inherit' });
    p.on('error', (e) =>
      reject(new Error(`Gagal menjalankan "${cmd}": ${e.message}. Pastikan sudah terinstall dan ada di PATH.`))
    );
    p.on('close', (code) => (code === 0 ? resolve() : reject(new Error(`${cmd} berhenti dengan kode ${code}`))));
  });
}

const fmtSize = (bytes) => {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
};

async function walk(dir) {
  const found = [];
  for (const e of await fs.readdir(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) found.push(...(await walk(p)));
    else if (e.isFile() && path.extname(e.name).toLowerCase() === '.mp4') found.push(p);
  }
  return found.sort((a, b) => a.localeCompare(b));
}

function videoArgs() {
  if (opt.nvenc) {
    // NVENC pakai -cq (mirip CRF); angkanya digeser sedikit supaya ukuran setara libx264
    const codec = opt.hevc ? 'hevc_nvenc' : 'h264_nvenc';
    return ['-c:v', codec, '-preset', 'p6', '-rc', 'vbr', '-cq', String(CRF + 2), '-b:v', '0'];
  }
  if (opt.hevc) {
    return ['-c:v', 'libx265', '-preset', opt.preset, '-crf', String(CRF), '-tag:v', 'hvc1'];
  }
  return ['-c:v', 'libx264', '-preset', opt.preset, '-crf', String(CRF), '-pix_fmt', 'yuv420p'];
}

// ---------- Kompres satu file ----------
async function compressOne(src, dst) {
  const tmp = dst.replace(/\.mp4$/i, '') + '.tmp.mp4';
  await fs.mkdir(path.dirname(dst), { recursive: true });

  const args = ['-y', '-loglevel', 'error', '-stats', '-i', src];
  if (opt.height) args.push('-vf', `scale=-2:'min(ih,${Number(opt.height)})'`);
  args.push(...videoArgs());
  if (opt.maxrate) args.push('-maxrate', opt.maxrate, '-bufsize', opt.maxrate);
  args.push('-c:a', 'aac', '-b:a', opt.abr, '-movflags', '+faststart', tmp);

  try {
    await run('ffmpeg', args);
  } catch (e) {
    await fs.rm(tmp, { force: true });
    throw e;
  }

  const before = (await fs.stat(src)).size;
  const after = (await fs.stat(tmp)).size;

  if (after >= before) {
    // Hasil malah lebih besar: pakai file asli saja
    await fs.rm(tmp, { force: true });
    if (!opt.replace) await fs.copyFile(src, dst);
    return { before, after: before, kept: 'asli' };
  }

  if (opt.replace) {
    await fs.rename(tmp, src);
  } else {
    await fs.rename(tmp, dst);
  }
  return { before, after, kept: 'kompres' };
}

// ---------- Main ----------
async function main() {
  const inDir = path.resolve(positionals[0] || opt.in);
  const outDir = path.resolve(opt.out);

  if (!existsSync(inDir)) {
    console.error(`Folder tidak ditemukan: ${inDir}`);
    process.exit(1);
  }

  const files = await walk(inDir);
  if (!files.length) {
    console.log(`Tidak ada file .mp4 di ${inDir}.`);
    return;
  }

  console.log(`• Sumber : ${inDir}`);
  console.log(opt.replace ? '• Hasil   : menimpa file asli (hanya jika lebih kecil)' : `• Hasil   : ${outDir}`);
  console.log(`• Setelan: ${opt.hevc ? 'H.265' : 'H.264'}${opt.nvenc ? ' (NVENC)' : ''}, CRF ${CRF}, audio ${opt.abr}${opt.height ? `, tinggi maks ${opt.height}px` : ''}`);
  console.log(`• Ditemukan ${files.length} file.\n`);

  let totalBefore = 0;
  let totalAfter = 0;
  let done = 0;

  for (let i = 0; i < files.length; i++) {
    const src = files[i];
    const rel = path.relative(inDir, src);
    const dst = path.join(outDir, rel);

    try {
      if (!opt.replace && !opt.force && existsSync(dst)) {
        console.log(`[${i + 1}/${files.length}] Lewati (sudah ada): ${rel}`);
        continue;
      }
      console.log(`[${i + 1}/${files.length}] ${rel}`);
      const r = await compressOne(src, dst);
      totalBefore += r.before;
      totalAfter += r.after;
      done++;
      const pct = ((1 - r.after / r.before) * 100).toFixed(0);
      console.log(`  ${fmtSize(r.before)} -> ${fmtSize(r.after)} (hemat ${pct}%)${r.kept === 'asli' ? ' [hasil kompres lebih besar, pakai asli]' : ''}`);
    } catch (e) {
      console.error(`  ERROR: ${e.message}`);
      process.exitCode = 1;
    }
  }

  if (done) {
    const pct = ((1 - totalAfter / totalBefore) * 100).toFixed(0);
    console.log(`\nSelesai. ${done} file: ${fmtSize(totalBefore)} -> ${fmtSize(totalAfter)} (hemat ${pct}%)`);
  } else {
    console.log('\nTidak ada file yang diproses.');
  }
}

main().catch((e) => {
  console.error('\nERROR:', e.message);
  process.exit(1);
});
