#!/usr/bin/env python3
"""
facetrack.py - jalur crop 9:16 yang mengikuti pembicara (untuk clipper.mjs --layout follow)

Cara kerja:
  1. Baca video dari detik --start selama --dur (waktu dibaca dari timestamp frame, bukan hitung manual),
     deteksi pergantian kamera (cut) di setiap frame.
  2. Deteksi wajah + gerak rahang/bibir (MediaPipe FaceLandmarker, blendshape jawOpen) ~12x per detik.
  3. Per shot (satu kamera): lacak tiap wajah, hitung "keaktifan bicara" tiap wajah per waktu
     (jendela terpusat, bisa lihat ke depan karena ini offline), lalu pilih pembicara untuk SELURUH shot
     sekaligus dengan Viterbi (ada biaya pindah, jadi tidak lompat-lompat), kamera mulai bergeser sedikit
     SEBELUM pembicara baru mulai bicara.
  4. Tulis JSON: segmen posisi x crop (piksel) per waktu, relatif ke awal klip.

Pemakaian:
  python facetrack.py video.mp4 --start 125.3 --dur 48 --out track.json
  python facetrack.py video.mp4 --start 125.3 --dur 48 --out track.json --debug   # + file .debug.json & timeline pembicara
"""
import argparse
import json
import os
import subprocess
import sys
import urllib.request
from bisect import bisect_left, bisect_right

import cv2
import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# Nilai awal; perlu disetel kalau hasil di video nyata kurang pas (lihat --help)
DEFAULTS = dict(
    sample_fps=12.0,      # frekuensi deteksi wajah (bibir bicara ~4-5 Hz, 8 fps terlalu jarang)
    cut_mad=22.0,         # ambang selisih rata-rata piksel antar frame untuk dianggap cut
    track_dist=0.07,      # jarak maks (fraksi lebar frame) wajah dianggap orang yang sama dalam satu shot
    speak_win=0.7,        # detik: jendela (terpusat) untuk mengukur gerak mulut
    speak_floor=0.03,     # batas bawah skala gerak mulut (di bawah ini dianggap noise/diam)
    speak_thr=0.3,        # skor relatif minimal (0-1 dari skala bicara di shot itu) untuk dihitung bicara
    switch_cost=3.0,      # biaya pindah pembicara; makin besar makin "keras kepala" (tidak mudah pindah)
    lead=0.15,            # detik: kamera mulai pindah segini lebih awal dari keputusan pembicara
    dead_zone=0.04,       # geser kamera (tanpa ganti pembicara) hanya kalau target bergeser > ini (fraksi lebar frame)
    move_dur=0.3,         # detik durasi geser saat ganti pembicara
    drift_dur=0.5,        # detik durasi geser halus saat orangnya sendiri bergerak
    present_win=1.5,      # detik: wajah dianggap masih ada setelah terakhir terlihat
    min_track_frac=0.15,  # track yang terlihat < segini dari sampel shot dianggap wajah palsu (poster dsb)
    use_audio=1.0,        # 1 = pakai audio (gerbang suara + sinkron bibir-audio), 0 = visual saja
    audio_gate=0.25,      # bobot gerak mulut saat audio diam (0 = abaikan total, 1 = audio tidak berpengaruh)
    corr_w=1.0,           # kekuatan bonus/penalti korelasi bukaan mulut vs keras suara (0 = matikan)
    corr_win=1.5,         # detik: jendela (terpusat) untuk menghitung korelasi
    audio_cap=0.6,        # saat audio ada: batas atas skor gerak mulut (gerak besar tapi tidak sinkron tidak boleh menang)
)


# ---------------------------------------------------------------- detektor wajah
def ensure_model(path):
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    print(f"• Mengunduh model wajah ke {path} ...", file=sys.stderr)
    try:
        urllib.request.urlretrieve(MODEL_URL, path)
    except Exception as e:  # noqa: BLE001
        if os.path.exists(path):
            os.remove(path)
        raise RuntimeError(
            f"Gagal mengunduh model wajah ({e}). Unduh manual dari:\n  {MODEL_URL}\n"
            f"lalu simpan sebagai: {path}"
        ) from e
    return path


class FaceDetector:
    """MediaPipe FaceLandmarker (Tasks API). Mengembalikan cx (0-1), w (0-1), m (gerak mulut ~0-1)."""

    def __init__(self, model_path, max_faces=6):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self.mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=ensure_model(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=max_faces,
            min_face_detection_confidence=0.4,
            min_face_presence_confidence=0.4,
            min_tracking_confidence=0.4,
            output_face_blendshapes=True,  # jawOpen jauh lebih stabil daripada jarak bibir mentah (tahan miring kepala)
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect(self, bgr):
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(image)
        blends = getattr(result, "face_blendshapes", None) or []
        faces = []
        for i, pts in enumerate(result.face_landmarks):
            xs = [p.x for p in pts]
            ys = [p.y for p in pts]
            face_h = max(ys) - min(ys)
            if face_h <= 0:
                continue
            jaw = None
            if i < len(blends):
                for c in blends[i]:
                    if c.category_name == "jawOpen":
                        jaw = float(c.score)
                        break
            if jaw is None:  # cadangan: jarak bibir dalam atas (13) & bawah (14), diskalakan mendekati jawOpen
                jaw = 5.0 * abs(pts[14].y - pts[13].y) / face_h
            faces.append({"cx": (min(xs) + max(xs)) / 2, "w": max(xs) - min(xs), "m": jaw})
        return faces


# ---------------------------------------------------------------- audio
def load_audio_db(video, start, dur, ffmpeg="ffmpeg", sr=16000):
    """Keras suara (dB) tiap 10 ms, dibatasi ke pita suara manusia. None kalau ffmpeg gagal / tidak ada audio."""
    cmd = [ffmpeg, "-v", "error", "-ss", f"{max(start, 0.0):.3f}", "-t", f"{dur:.3f}", "-i", video,
           "-vn", "-ac", "1", "-ar", str(sr), "-af", "highpass=f=150,lowpass=f=3800", "-f", "f32le", "-"]
    try:
        raw = subprocess.run(cmd, capture_output=True, timeout=600).stdout
    except Exception:  # noqa: BLE001
        return None
    x = np.frombuffer(raw, dtype=np.float32)
    hop = sr // 100
    n = x.size // hop
    if n < 50:
        return None
    rms = np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1)) + 1e-8
    return 20.0 * np.log10(rms)


def attach_audio(samples, db, fps):
    """Isi s['a'] (keras suara 0-1) dan s['v'] (ada suara 0-1) di tiap sampel. Return False kalau audio tidak berguna."""
    if db is None or not samples:
        return False
    floor, peak = float(np.percentile(db, 10)), float(np.percentile(db, 95))
    if peak - floor < 6.0:  # audio datar / senyap: tidak ada informasi
        return False
    h = 0.5 / fps
    for s in samples:
        lo = max(int((s["t"] - h) * 100), 0)
        hi = min(max(int((s["t"] + h) * 100) + 1, lo + 1), len(db))
        seg = db[lo:hi] if lo < len(db) else db[-1:]
        a = float(np.clip((float(np.mean(seg)) - floor) / (peak - floor), 0.0, 1.0))
        s["a"] = a
        s["v"] = float(np.clip((a - 0.2) / 0.2, 0.0, 1.0))
    return True


def windowed_corr(x, y, half):
    """Korelasi Pearson di jendela geser; x boleh berisi NaN (wajah tidak terlihat)."""
    n = len(x)
    out = np.zeros(n)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        xs, ys = x[lo:hi], y[lo:hi]
        ok = ~np.isnan(xs)
        if ok.sum() < 6:
            continue
        xs, ys = xs[ok], ys[ok]
        if xs.std() < 1e-6 or ys.std() < 1e-6:
            continue
        out[i] = float(np.corrcoef(xs, ys)[0, 1])
    return out


# ---------------------------------------------------------------- analisis (murni, bisa diuji)
class Track:
    def __init__(self):
        self.t = []   # waktu terlihat
        self.x = []   # cx
        self.m = []   # gerak mulut
        self.w = []   # lebar wajah

    def add(self, t, face):
        self.t.append(t)
        self.x.append(face["cx"])
        self.m.append(face["m"])
        self.w.append(face["w"])

    @property
    def last_x(self):
        return self.x[-1]

    @property
    def last_w(self):
        return self.w[-1]

    def pos(self, t):
        """Posisi terakhir yang terlihat pada atau sebelum t (atau yang pertama kalau belum ada)."""
        i = bisect_right(self.t, t + 1e-6) - 1
        return self.x[max(i, 0)]

    def present(self, t, win):
        i = bisect_right(self.t, t + 1e-6) - 1
        return i >= 0 and t - self.t[i] <= win

    def activity(self, t, win, max_gap):
        """Rata-rata |selisih| gerak mulut antar sampel berurutan di jendela terpusat pada t."""
        lo = bisect_left(self.t, t - win / 2)
        hi = bisect_right(self.t, t + win / 2)
        if hi - lo < 3:
            return 0.0
        d = [
            abs(self.m[k + 1] - self.m[k])
            for k in range(lo, hi - 1)
            if self.t[k + 1] - self.t[k] <= max_gap
        ]
        return float(np.mean(d)) if len(d) >= 2 else 0.0

    def mean_w(self):
        return float(np.mean(self.w))


def build_tracks(samples, track_dist):
    """Gabungkan deteksi per sampel menjadi track per orang (pencocokan global terdekat, bukan greedy per ukuran)."""
    tracks = []
    for s in samples:
        faces = s["faces"]
        pairs = []
        for fi, f in enumerate(faces):
            for ti, tr in enumerate(tracks):
                d = abs(tr.last_x - f["cx"])
                lim = min(track_dist, 0.6 * max(f["w"], tr.last_w))  # dua orang berdekatan jangan tertukar
                if d < lim:
                    pairs.append((d, fi, ti))
        pairs.sort()
        used_f, used_t = set(), set()
        for _, fi, ti in pairs:
            if fi in used_f or ti in used_t:
                continue
            tracks[ti].add(s["t"], faces[fi])
            used_f.add(fi)
            used_t.add(ti)
        for fi, f in enumerate(faces):
            if fi not in used_f:
                tr = Track()
                tr.add(s["t"], f)
                tracks.append(tr)
    return tracks


def pick_speakers(samples, tracks, cfg):
    """Label pembicara per sampel (indeks track) lewat Viterbi dengan biaya pindah."""
    n, k = len(samples), len(tracks)
    gap = 2.5 / cfg["sample_fps"]
    S = np.zeros((n, k))
    P = np.zeros((n, k), bool)
    for i, s in enumerate(samples):
        for j, tr in enumerate(tracks):
            S[i, j] = tr.activity(s["t"], cfg["speak_win"], gap)
            P[i, j] = tr.present(s["t"], cfg["present_win"])
    S *= P
    pos = S[S > 0]
    ref = max(float(np.percentile(pos, 90)) if pos.size else 0.0, cfg["speak_floor"])
    E = np.clip(S / ref - cfg["speak_thr"], 0.0, 2.0)

    if cfg.get("use_audio", 0) >= 0.5 and n and all("a" in s for s in samples):
        A = np.array([s["a"] for s in samples])
        V = np.array([s["v"] for s in samples])
        # gerbang suara: gerak mulut saat tidak ada suara (senyum, tertawa tanpa suara, mengunyah) diredam
        w = max(1, int(round(cfg["speak_win"] * cfg["sample_fps"])))
        Vm = np.array([V[max(0, i - w // 2): i + w // 2 + 1].max() for i in range(n)])
        E = np.minimum(E, cfg["audio_cap"])
        E *= (cfg["audio_gate"] + (1.0 - cfg["audio_gate"]) * Vm)[:, None]
        # sinkron bibir-audio: bukaan mulut yang naik-turun bersama keras suara = kemungkinan besar pembicara
        if cfg["corr_w"] > 0:
            idx = {round(s["t"], 6): i for i, s in enumerate(samples)}
            half = max(3, int(round(cfg["corr_win"] * cfg["sample_fps"] / 2)))
            for j, tr in enumerate(tracks):
                m = np.full(n, np.nan)
                for t_, m_ in zip(tr.t, tr.m):
                    i = idx.get(round(t_, 6))
                    if i is not None:
                        m[i] = m_
                C = windowed_corr(m, A, half)
                E[:, j] *= np.clip(0.3 + 2.0 * cfg["corr_w"] * np.maximum(C, 0.0), 0.3, 2.0)
    E += 1e-3 * np.array([tr.mean_w() for tr in tracks])[None, :]  # seri -> condong ke wajah yang lebih besar
    E[~P] = -1.0  # wajah yang sedang tidak terlihat tidak layak dipilih

    cost = cfg["switch_cost"]
    best = np.zeros((n, k))
    back = np.zeros((n, k), dtype=int)
    best[0] = E[0]
    for i in range(1, n):
        prev = best[i - 1]
        src = int(np.argmax(prev))
        sw = prev[src] - cost
        for j in range(k):
            if prev[j] >= sw:
                back[i, j], best[i, j] = j, prev[j] + E[i, j]
            else:
                back[i, j], best[i, j] = src, sw + E[i, j]
    lab = [0] * n
    lab[-1] = int(np.argmax(best[-1]))
    for i in range(n - 1, 0, -1):
        lab[i - 1] = int(back[i, lab[i]])
    return lab


def plan_shot(samples, t0, t1, crop_norm, cfg, dbg=None):
    """Kembalikan segmen (t0, t1, x0, x1) pusat crop dalam fraksi lebar frame untuk satu shot."""
    if t1 <= t0:
        return []
    tracks = build_tracks(samples, cfg["track_dist"])
    n = len(samples)
    if tracks:
        keep = [tr for tr in tracks if len(tr.t) >= max(3, cfg["min_track_frac"] * n)]
        tracks = keep or [max(tracks, key=lambda tr: len(tr.t))]
    if not tracks or n == 0:
        return [(t0, t1, 0.5, 0.5)]

    lab = pick_speakers(samples, tracks, cfg)
    lead_n = int(round(cfg["lead"] * cfg["sample_fps"]))
    lab = [lab[min(i + lead_n, n - 1)] for i in range(n)]  # kamera mulai bergerak sedikit lebih awal

    if dbg is not None:
        runs, st = [], 0
        for i in range(1, n + 1):
            if i == n or lab[i] != lab[st]:
                runs.append({"from": round(samples[st]["t"], 2), "to": round(samples[i - 1]["t"], 2), "track": lab[st]})
                st = i
        dbg.append({
            "shot_from": round(t0, 2), "shot_to": round(t1, 2),
            "tracks": [{"x": round(float(np.mean(tr.x)), 3), "w": round(tr.mean_w(), 3), "n": len(tr.t)} for tr in tracks],
            "speaker_runs": runs,
        })

    def target(t, j):
        vis = [tr for tr in tracks if tr.present(t, cfg["present_win"])]
        if len(vis) > 1:
            lo = min(tr.pos(t) - tr.mean_w() / 2 for tr in vis)
            hi = max(tr.pos(t) + tr.mean_w() / 2 for tr in vis)
            if hi - lo <= crop_norm * 0.9:  # semua orang muat dalam satu crop -> jangan bolak-balik
                return (lo + hi) / 2
        return tracks[j].pos(t)

    segs = []
    cam = None
    seg_start = t0
    move_end = t0
    recent = []
    for i, s in enumerate(samples):
        t = s["t"]
        j = lab[i]
        changed = i > 0 and j != lab[i - 1]
        raw = target(t, j)
        recent = [raw] if changed else (recent + [raw])[-5:]
        tgt = float(np.median(recent))
        if cam is None:
            cam = tgt
            continue
        if changed:
            if abs(tgt - cam) <= 0.01:
                continue
            d = cfg["move_dur"]
        else:
            if t < move_end or abs(tgt - cam) <= cfg["dead_zone"]:
                continue
            d = cfg["drift_dur"]
        if t < move_end and segs:  # potong gerakan yang sedang berjalan, lanjut dari posisi sekarang
            ta, tb, xa, xb = segs[-1]
            cam = xa + (xb - xa) * (t - ta) / (tb - ta) if tb > ta else xb
            segs[-1] = (ta, t, xa, cam)
        else:
            segs.append((seg_start, t, cam, cam))
        end = min(t + d, t1)
        segs.append((t, end, cam, tgt))
        cam, seg_start, move_end = tgt, end, end
    segs.append((seg_start, t1, cam if cam is not None else 0.5, cam if cam is not None else 0.5))
    return [g for g in segs if g[1] > g[0] + 1e-6] or [(t0, t1, cam or 0.5, cam or 0.5)]


def plan_path(samples, shot_starts, dur, crop_norm, cfg, dbg=None):
    segs = []
    bounds = list(shot_starts) + [dur]
    for i in range(len(shot_starts)):
        ss = [s for s in samples if s["shot"] == i]
        segs.extend(plan_shot(ss, bounds[i], bounds[i + 1], crop_norm, cfg, dbg))
    return segs


def to_pixels(segs, width, cw):
    clamp = lambda cx: float(min(max(cx * width - cw / 2, 0), width - cw))  # noqa: E731
    out = []
    for t0, t1, a, b in segs:
        x0, x1 = clamp(a), clamp(b)
        if out and abs(out[-1][3] - x0) < 0.5 and abs(x0 - x1) < 0.5 and abs(out[-1][2] - out[-1][3]) < 0.5:
            out[-1][1] = t1  # gabung segmen diam yang berurutan
        else:
            out.append([round(t0, 3), round(t1, 3), round(x0, 1), round(x1, 1)])
    return out


# ---------------------------------------------------------------- baca video
def process(video, start, dur, detector, cfg, det_width=1920, debug=False, ffmpeg="ffmpeg"):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"Tidak bisa membuka video: {video}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Seek ke 1 detik sebelum start lalu buang frame sampai waktu aslinya tercapai.
    # Seek OpenCV sering mendarat di keyframe sebelumnya; kalau dipercaya mentah-mentah, seluruh jalur bergeser.
    pre = max(0.0, start - 1.0)
    cap.set(cv2.CAP_PROP_POS_MSEC, pre * 1000.0)

    shot_starts, samples = [0.0], []
    prev, mads = None, []
    next_sample = 0.0
    last_abs = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ms = cap.get(cv2.CAP_PROP_POS_MSEC)  # timestamp frame yang baru dibaca (benar juga untuk video VFR)
        if ms > 0:
            cur = ms / 1000.0
        elif last_abs is None:
            cur = pre
        else:
            cur = last_abs + 1.0 / fps
        if last_abs is not None and cur <= last_abs:
            cur = last_abs + 1.0 / fps
        last_abs = cur
        t = cur - start
        if t < -0.5 / fps:
            continue  # pra-gulung
        if t >= dur:
            break
        t = max(t, 0.0)

        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 90)).astype(np.int16)
        is_cut = False
        if prev is not None:
            mad = float(np.mean(np.abs(small - prev)))
            base = np.mean(mads[-10:]) if mads else 0.0
            if mad > cfg["cut_mad"] and mad > 3 * base:
                is_cut = True
                shot_starts.append(t)
                mads.clear()
            else:
                mads.append(mad)
        prev = small

        if t + 1e-6 >= next_sample or is_cut:
            f = frame
            if W > det_width:
                f = cv2.resize(frame, (det_width, int(H * det_width / W)))
            samples.append({"t": t, "shot": len(shot_starts) - 1, "faces": detector.detect(f)})
            next_sample = t + 1.0 / cfg["sample_fps"]
    cap.release()

    audio_ok = False
    if cfg.get("use_audio", 0) >= 0.5:
        audio_ok = attach_audio(samples, load_audio_db(video, start, dur, ffmpeg), cfg["sample_fps"])
        if not audio_ok:
            print("  ! Audio tidak terbaca / datar; pembicara ditentukan dari gerak mulut saja", file=sys.stderr)

    cw = min(W, int(H * 9 / 16) // 2 * 2)
    dbg = [] if debug else None
    segs = plan_path(samples, shot_starts, dur, cw / W, cfg, dbg)
    res = {
        "width": W, "height": H, "cw": cw, "ch": H, "fps": fps,
        "cuts": [round(c, 3) for c in shot_starts[1:]],
        "faces_seen": sum(1 for s in samples if s["faces"]),
        "samples": len(samples),
        "audio": audio_ok,
        "segments": to_pixels(segs, W, cw),
    }
    if debug:
        res["debug"] = dbg
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--dur", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "face_landmarker.task"))
    ap.add_argument("--max-faces", type=int, default=6)
    ap.add_argument("--det-width", type=int, default=1920, help="lebar maks frame untuk deteksi wajah (makin besar, bibir wajah kecil makin akurat)")
    ap.add_argument("--ffmpeg", default=os.environ.get("FFMPEG", "ffmpeg"), help="path ffmpeg (untuk membaca audio)")
    ap.add_argument("--debug", action="store_true", help="cetak timeline pembicara per shot dan simpan ke JSON")
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=float, default=v)
    a = ap.parse_args()
    cfg = {k: getattr(a, k) for k in DEFAULTS}
    try:
        det = FaceDetector(a.model, a.max_faces)
    except ImportError:
        print("mediapipe belum terpasang: pip install mediapipe opencv-python numpy", file=sys.stderr)
        return 2
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 3
    res = process(a.video, a.start, a.dur, det, cfg, a.det_width, a.debug, a.ffmpeg)
    with open(a.out, "w", encoding="utf8") as f:
        json.dump(res, f)
    print(f"  shot: {len(res['cuts']) + 1}, sampel dengan wajah: {res['faces_seen']}/{res['samples']}, audio: {'ya' if res['audio'] else 'tidak'}", file=sys.stderr)
    if a.debug:
        for sh in res["debug"]:
            print(f"  shot {sh['shot_from']}-{sh['shot_to']}s | wajah: " +
                  ", ".join(f"#{i}(x={t['x']},n={t['n']})" for i, t in enumerate(sh["tracks"])), file=sys.stderr)
            for r in sh["speaker_runs"]:
                print(f"     {r['from']:>6}-{r['to']:<6} -> wajah #{r['track']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
