#!/usr/bin/env python3
"""
blurproduct.py - blur produk/merek di video (dipakai terpisah, mis. pada klip hasil clipper.mjs)

Tiga cara menunjuk produk (boleh dikombinasikan):
  --ref foto.jpg      cocokkan dengan FOTO produk (fitur SIFT + homografi). Paling tepat untuk produk spesifik,
                      tahan zoom/rotasi/cut kamera. Pakai foto yang dipotong rapat, sisi label menghadap depan.
                      Boleh diulang untuk beberapa produk atau beberapa sudut: --ref a.jpg --ref a_samping.jpg
  --region x1,y1,x2,y2[@t0-t1]
                      area tetap, pecahan 0-1 dari frame (0,0 = kiri-atas). Opsional rentang detik.
                      Contoh: --region 0.6,0.5,0.95,0.9   atau   --region 0.6,0.5,0.95,0.9@5-12
  --detect "bottle,box"
                      deteksi per KATEGORI dengan YOLO-World (pip install ultralytics). Kurang spesifik daripada --ref:
                      semua benda sekategori ikut ter-blur. Nama kelas dalam bahasa Inggris.

Pemakaian (membaca semua .mp4 di folder, seperti compress.js):
  python blurproduct.py --ref produk.jpg                          # ./output  ->  ./output-blur
  python blurproduct.py ./output --ref produk.jpg -o ./hasil     # folder sumber & hasil sendiri
  python blurproduct.py klip.mp4 --ref produk.jpg                 # satu file juga boleh
  python blurproduct.py --ref produk.jpg --show-boxes             # cek dulu: kotak digambar, belum di-blur
  python blurproduct.py --ref produk.jpg --reuse                  # ulang tanpa deteksi ulang (setel gaya/padding saja)
  python blurproduct.py --ref produk.jpg --force                  # proses ulang walau hasilnya sudah ada

Catatan: --region berlaku untuk SEMUA video di folder. Kalau areanya beda per video, jalankan per file.

Urutan yang disarankan di pipeline clipper: blur dulu, subtitle belakangan. Kalau dijalankan pada klip yang sudah
bersubtitle, subtitle yang kebetulan menimpa area produk ikut ter-blur.
"""
import argparse
import json
import os
import subprocess
import sys
from bisect import bisect_right

import cv2
import numpy as np


# ---------------------------------------------------------------- util
def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def cut_between(cuts, a, b):
    """Ada cut di frame c dengan a < c <= b? (cut = frame pertama shot baru)"""
    return bisect_right(cuts, b) - bisect_right(cuts, a) > 0


def parse_region(s):
    t0, t1 = 0.0, float("inf")
    if "@" in s:
        s, rng = s.split("@", 1)
        a, b = rng.split("-", 1)
        t0 = float(a) if a.strip() else 0.0
        t1 = float(b) if b.strip() else float("inf")
    x1, y1, x2, y2 = [float(v) for v in s.split(",")]
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError(f"--region harus x1<x2 dan y1<y2 dalam 0-1: {s}")
    return {"box": (x1, y1, x2, y2), "t0": t0, "t1": t1}


# ---------------------------------------------------------------- detektor
class RefDetector:
    """Cocokkan foto produk ke frame: SIFT + FLANN + rasio Lowe + RANSAC homografi."""

    def __init__(self, paths, det_width=1280, ratio=0.75, min_inliers=12, max_inst=2):
        self.sift = cv2.SIFT_create(nfeatures=5000)
        self.flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=48))
        self.det_width, self.ratio, self.min_inliers, self.max_inst = det_width, ratio, min_inliers, max_inst
        self.refs = []
        for p in paths:
            img = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_GRAYSCALE) if os.path.exists(p) else None
            if img is None:
                raise RuntimeError(f"Foto referensi tidak bisa dibuka: {p}")
            h, w = img.shape
            s = 640.0 / max(h, w)  # samakan skala; terlalu kecil dinaikkan supaya fitur cukup
            if s < 1 or max(h, w) < 400:
                img = cv2.resize(img, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
            kp, des = self.sift.detectAndCompute(img, None)
            if des is None or len(kp) < 20:
                print(f"  ! {p}: fitur terlalu sedikit ({0 if des is None else len(kp)}); pakai foto yang lebih tajam/berdetail", file=sys.stderr)
                if des is None or len(kp) < 4:
                    continue
            hh, ww = img.shape
            self.refs.append({
                "name": os.path.basename(p), "pts": np.float32([k.pt for k in kp]), "des": des,
                "corners": np.float32([[0, 0], [ww, 0], [ww, hh], [0, hh]]).reshape(-1, 1, 2),
            })
        if not self.refs:
            raise RuntimeError("Tidak ada foto referensi yang bisa dipakai")

    def detect(self, bgr):
        h, w = bgr.shape[:2]
        sc = min(1.0, self.det_width / w)
        small = cv2.resize(bgr, (int(w * sc), int(h * sc))) if sc < 1 else bgr
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gh, gw = g.shape
        kp, des = self.sift.detectAndCompute(g, None)
        if des is None or len(kp) < self.min_inliers:
            return []
        pts = np.float32([k.pt for k in kp])
        out = []
        for ref in self.refs:
            alive = np.ones(len(kp), bool)
            for _ in range(self.max_inst):
                idx = np.flatnonzero(alive)
                if len(idx) < max(self.min_inliers, 4):
                    break
                knn = self.flann.knnMatch(ref["des"], des[idx], k=2)
                good = [(m.queryIdx, idx[m.trainIdx]) for pair in knn if len(pair) == 2
                        for m, n in [pair] if m.distance < self.ratio * n.distance]
                if len(good) < self.min_inliers:
                    break
                src = ref["pts"][[q for q, _ in good]].reshape(-1, 1, 2)
                dst = pts[[t for _, t in good]].reshape(-1, 1, 2)
                H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
                if H is None or int(mask.sum()) < self.min_inliers:
                    break
                quad = cv2.perspectiveTransform(ref["corners"], H).reshape(-1, 2)
                area = cv2.contourArea(quad.astype(np.float32))
                if not cv2.isContourConvex(quad.astype(np.float32).reshape(-1, 1, 2)) or not (0.0003 * gw * gh < area < 0.8 * gw * gh):
                    break
                m = np.zeros((gh, gw), np.uint8)
                cv2.fillConvexPoly(m, quad.astype(np.int32), 255)
                xi, yi = np.clip(pts[:, 0].astype(int), 0, gw - 1), np.clip(pts[:, 1].astype(int), 0, gh - 1)
                alive &= m[yi, xi] == 0  # fitur di dalam produk ini tidak dipakai lagi (cari instance berikutnya)
                x1, y1 = np.clip(quad.min(axis=0), 0, [gw, gh])
                x2, y2 = np.clip(quad.max(axis=0), 0, [gw, gh])
                if x2 - x1 > 4 and y2 - y1 > 4:
                    out.append((x1 / gw, y1 / gh, x2 / gw, y2 / gh))
        return out


class YoloWorldDetector:
    """Deteksi per kategori (open-vocabulary). Opsional: pip install ultralytics."""

    def __init__(self, classes, conf=0.25, weights="yolov8s-worldv2.pt"):
        try:
            from ultralytics import YOLOWorld
        except ImportError as e:
            raise RuntimeError("--detect butuh ultralytics: pip install ultralytics") from e
        self.model = YOLOWorld(weights)
        self.model.set_classes(classes)
        self.conf = conf

    def detect(self, bgr):
        h, w = bgr.shape[:2]
        r = self.model.predict(bgr, conf=self.conf, verbose=False)[0]
        return [(float(b[0]) / w, float(b[1]) / h, float(b[2]) / w, float(b[3]) / h)
                for b in r.boxes.xyxy.cpu().numpy()]


# ---------------------------------------------------------------- pass 1: deteksi
def scan(video, detectors, step, cut_mad):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"Tidak bisa membuka video: {video}")
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    info = {"fps": fps, "W": W, "H": H, "samples": [], "cuts": []}
    if not detectors:
        cap.release()
        return info
    prev, mads, f = None, [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 90)).astype(np.int16)
        is_cut = False
        if prev is not None:
            mad = float(np.mean(np.abs(small - prev)))
            base = float(np.mean(mads[-10:])) if mads else 0.0
            if mad > cut_mad and mad > 3 * base:
                is_cut = True
                info["cuts"].append(f)
                mads.clear()
            else:
                mads.append(mad)
        prev = small
        if f % step == 0 or is_cut:
            boxes = []
            for d in detectors:
                boxes += d.detect(frame)
            info["samples"].append((f, boxes))
        if f % 150 == 0:
            print(f"\r• Deteksi: frame {f}/{total or '?'}", end="", file=sys.stderr, flush=True)
        f += 1
    cap.release()
    print(f"\r• Deteksi selesai: {f} frame, {len(info['cuts'])} cut, "
          f"{sum(1 for _, b in info['samples'] if b)}/{len(info['samples'])} sampel berisi produk", file=sys.stderr)
    return info


# ---------------------------------------------------------------- jalur kotak
def make_tracks(samples, cuts, max_gap, iou_thr):
    """Sambung kotak antar sampel jadi track (satu produk = satu track); tidak menyeberang cut."""
    tracks, active = [], []
    for f, boxes in samples:
        active = [t for t in active if f - t["f"][-1] <= max_gap and not cut_between(cuts, t["f"][-1], f)]
        pairs = [(iou(b, t["b"][-1]), bi, ti) for bi, b in enumerate(boxes) for ti, t in enumerate(active)]
        pairs = sorted((p for p in pairs if p[0] >= iou_thr), reverse=True)
        used_b, used_t = set(), set()
        for _, bi, ti in pairs:
            if bi in used_b or ti in used_t:
                continue
            active[ti]["f"].append(f)
            active[ti]["b"].append(tuple(boxes[bi]))
            used_b.add(bi)
            used_t.add(ti)
        for bi, b in enumerate(boxes):
            if bi not in used_b:
                t = {"f": [f], "b": [tuple(b)]}
                tracks.append(t)
                active.append(t)
    return tracks


def track_box(t, f, hold, pre, cuts):
    fr = t["f"]
    if f < fr[0]:
        if fr[0] - f > pre or cut_between(cuts, f, fr[0]):
            return None
        return t["b"][0]
    if f > fr[-1]:
        if f - fr[-1] > hold or cut_between(cuts, fr[-1], f):
            return None
        return t["b"][-1]
    i = bisect_right(fr, f) - 1
    if i >= len(fr) - 1 or fr[i] == f:
        return t["b"][i]
    a = (f - fr[i]) / (fr[i + 1] - fr[i])
    return tuple(p + (q - p) * a for p, q in zip(t["b"][i], t["b"][i + 1]))


# ---------------------------------------------------------------- blur
def apply_blur(frame, box, style, pad):
    H, W = frame.shape[:2]
    bw, bh = (box[2] - box[0]) * W, (box[3] - box[1]) * H
    px, py = bw * pad + 4, bh * pad + 4
    x1, y1 = int(max(0, box[0] * W - px)), int(max(0, box[1] * H - py))
    x2, y2 = int(min(W, box[2] * W + px)), int(min(H, box[3] * H + py))
    w, h = x2 - x1, y2 - y1
    if w < 6 or h < 6:
        return
    roi = frame[y1:y2, x1:x2]
    s = max(2, int(max(w, h) / 40))  # hasil akhir ~40 px: tulisan/logo tidak terbaca
    small = cv2.resize(roi, (max(1, w // s), max(1, h // s)), interpolation=cv2.INTER_AREA)
    if style == "pixel":
        eff = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        small = cv2.GaussianBlur(small, (0, 0), 2.0)
        eff = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    f = max(2, int(min(w, h) * 0.08))  # tepi dilembutkan supaya tidak tampak seperti kotak
    mask = np.zeros((h, w), np.float32)
    mask[f:h - f, f:w - f] = 1.0
    mask = cv2.GaussianBlur(mask, (0, 0), f / 2.0)[..., None]
    roi[:] = (eff.astype(np.float32) * mask + roi.astype(np.float32) * (1.0 - mask)).astype(np.uint8)


def draw_box(frame, box, pad):
    H, W = frame.shape[:2]
    bw, bh = (box[2] - box[0]) * W, (box[3] - box[1]) * H
    px, py = bw * pad + 4, bh * pad + 4
    cv2.rectangle(frame, (int(max(0, box[0] * W - px)), int(max(0, box[1] * H - py))),
                  (int(min(W - 1, box[2] * W + px)), int(min(H - 1, box[3] * H + py))), (0, 0, 255), 3)


# ---------------------------------------------------------------- pass 2: tulis
def render(video, out, info, tracks, regions, a):
    W, H, fps, cuts = info["W"], info["H"], info["fps"], info["cuts"]
    hold, pre = int(round(a.hold * fps)), a.step
    enc = (["-c:v", "h264_nvenc", "-preset", "p5", "-cq", str(a.crf), "-b:v", "0"] if a.nvenc
           else ["-c:v", "libx264", "-preset", "medium", "-crf", str(a.crf)])
    cmd = [a.ffmpeg, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", f"{fps}",
           "-i", "-", "-i", video, "-map", "0:v:0", "-map", "1:a:0?",
           "-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2", *enc, "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest", out]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    except FileNotFoundError as e:
        raise RuntimeError(f"ffmpeg tidak ditemukan ({a.ffmpeg}); set --ffmpeg atau tambahkan ke PATH") from e
    cap = cv2.VideoCapture(video)
    f = blurred = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = f / fps
            boxes = [b for tr in tracks if (b := track_box(tr, f, hold, pre, cuts)) is not None]
            boxes += [r["box"] for r in regions if r["t0"] <= t < r["t1"]]
            for b in boxes:
                (draw_box(frame, b, a.pad) if a.show_boxes else apply_blur(frame, b, a.style, a.pad))
            blurred += bool(boxes)
            proc.stdin.write(frame.tobytes())
            f += 1
            if f % 150 == 0:
                print(f"\r• Tulis: frame {f}", end="", file=sys.stderr, flush=True)
    except BrokenPipeError:
        pass
    finally:
        cap.release()
        try:
            proc.stdin.close()
        except Exception:  # noqa: BLE001
            pass
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg berhenti dengan kode {rc}")
    print(f"\r• Selesai: {f} frame, {blurred} frame berisi area blur -> {out}", file=sys.stderr)


# ---------------------------------------------------------------- folder
def walk(root, exclude=None):
    """Cari semua .mp4 secara rekursif (folder hasil dilewati supaya tidak terbaca ulang)."""
    found = []
    for d, dirs, files in os.walk(root):
        if exclude:
            dirs[:] = [x for x in dirs if os.path.abspath(os.path.join(d, x)) != exclude]
        for n in files:
            low = n.lower()
            if low.endswith(".mp4") and not low.endswith(".tmp.mp4"):
                found.append(os.path.join(d, n))
    return sorted(found)


def process_one(src, dst, detectors, regions, a):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    cache = dst + ".det.json"
    if a.reuse and os.path.exists(cache):
        with open(cache, encoding="utf8") as fh:
            info = json.load(fh)
        print(f"  • Memakai deteksi tersimpan: {os.path.basename(cache)}", file=sys.stderr)
    elif detectors:
        info = scan(src, detectors, max(1, a.step), a.cut_mad)
        with open(cache, "w", encoding="utf8") as fh:
            json.dump(info, fh)
    else:
        info = scan(src, [], 1, a.cut_mad)
    tracks = make_tracks([(f, b) for f, b in info["samples"]], info["cuts"], int(round(a.max_gap * info["fps"])), 0.15)
    print(f"  • {len(tracks)} jalur produk", file=sys.stderr)
    tmp = os.path.splitext(dst)[0] + ".tmp.mp4"
    try:
        render(src, tmp, info, tracks, regions, a)
        os.replace(tmp, dst)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", nargs="?", default="./output", help="folder sumber (dibaca rekursif) atau satu file .mp4; default ./output")
    ap.add_argument("-o", "--out", default="./output-blur", help="folder hasil (struktur subfolder ikut disalin); default ./output-blur")
    ap.add_argument("--force", action="store_true", help="proses ulang walau hasilnya sudah ada")
    ap.add_argument("--ref", action="append", default=[], help="foto produk (boleh diulang)")
    ap.add_argument("--region", action="append", default=[], help="x1,y1,x2,y2[@t0-t1] (boleh diulang)")
    ap.add_argument("--detect", default="", help='kelas YOLO-World dipisah koma, mis. "bottle,box"')
    ap.add_argument("--detect-model", default="yolov8s-worldv2.pt")
    ap.add_argument("--conf", type=float, default=0.25, help="ambang keyakinan YOLO-World")
    ap.add_argument("--min-inliers", type=int, default=12, help="titik cocok minimal untuk --ref (naikkan kalau salah blur, turunkan kalau produk terlewat)")
    ap.add_argument("--ratio", type=float, default=0.75, help="ambang rasio Lowe untuk --ref (lebih kecil = lebih ketat)")
    ap.add_argument("--max-inst", type=int, default=2, help="maks jumlah kemunculan tiap foto referensi per frame")
    ap.add_argument("--det-width", type=int, default=1280, help="lebar maks frame saat deteksi (produk kecil butuh lebih besar)")
    ap.add_argument("--step", type=int, default=3, help="deteksi tiap N frame; di antaranya kotak diinterpolasi")
    ap.add_argument("--hold", type=float, default=0.5, help="detik: kotak dipertahankan setelah deteksi terakhir (menutup sampel yang terlewat)")
    ap.add_argument("--max-gap", type=float, default=1.0, help="detik: celah maks antar deteksi yang masih dianggap produk yang sama")
    ap.add_argument("--cut-mad", type=float, default=22.0)
    ap.add_argument("--style", choices=["blur", "pixel"], default="blur")
    ap.add_argument("--pad", type=float, default=0.12, help="pelebaran kotak (pecahan ukuran kotak)")
    ap.add_argument("--show-boxes", action="store_true", help="gambar kotak merah, jangan blur (untuk mengecek deteksi)")
    ap.add_argument("--reuse", action="store_true", help="pakai hasil deteksi tersimpan (<hasil>.det.json); otomatis memproses ulang")
    ap.add_argument("--nvenc", action="store_true")
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--ffmpeg", default=os.environ.get("FFMPEG", "ffmpeg"))
    a = ap.parse_args()

    if not (a.ref or a.region or a.detect):
        print("Isi minimal salah satu: --ref, --region, atau --detect", file=sys.stderr)
        return 1
    src_path = os.path.abspath(a.src)
    out_dir = os.path.abspath(a.out)
    if os.path.isfile(src_path):
        root, files = os.path.dirname(src_path), [src_path]
    elif os.path.isdir(src_path):
        root, files = src_path, walk(src_path, exclude=out_dir)
    else:
        print(f"Tidak ditemukan: {src_path}", file=sys.stderr)
        return 1
    if not files:
        print(f"Tidak ada file .mp4 di {src_path}.")
        return 0

    try:
        regions = [parse_region(r) for r in a.region]
        detectors = []
        if a.ref:
            detectors.append(RefDetector(a.ref, a.det_width, a.ratio, a.min_inliers, a.max_inst))
        if a.detect:
            detectors.append(YoloWorldDetector([c.strip() for c in a.detect.split(",") if c.strip()], a.conf, a.detect_model))
    except (RuntimeError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    redo = a.force or a.reuse or a.show_boxes
    modes = [x for x, on in (("ref", a.ref), ("region", a.region), ("detect", a.detect)) if on]
    print(f"• Sumber : {src_path}")
    print(f"• Hasil  : {out_dir}")
    print(f"• Mode   : {'+'.join(modes)}, gaya {a.style}{' (hanya kotak)' if a.show_boxes else ''}")
    print(f"• Ditemukan {len(files)} file.\n")

    done = skipped = failed = 0
    for i, src in enumerate(files, 1):
        rel = os.path.relpath(src, root)
        dst = os.path.join(out_dir, rel)
        if not redo and os.path.exists(dst):
            print(f"[{i}/{len(files)}] Lewati (sudah ada): {rel}")
            skipped += 1
            continue
        print(f"[{i}/{len(files)}] {rel}")
        try:
            process_one(src, dst, detectors, regions, a)
            done += 1
        except KeyboardInterrupt:
            print("\nDibatalkan.", file=sys.stderr)
            return 130
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR: {e}", file=sys.stderr)
            failed += 1

    print(f"\nSelesai. {done} diproses, {skipped} dilewati, {failed} gagal.")
    return 2 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
