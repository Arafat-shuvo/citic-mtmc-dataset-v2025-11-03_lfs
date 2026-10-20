import cv2, csv
from pathlib import Path

SRC = Path(r"C:\Users\Arafat\Desktop\citic-mtmc-dataset-v2025-11-03_lfs\GTEC-UDC-citic-mtmc-dataset-7b2f3a6")
DST = Path(r"C:\citic_frames_25fps")     # alada folder, purono 5fps chobir sathe mishbe na
STRIDE = 1                               # 1 = sob frame (25 FPS)
JPEG_QUALITY = 95
EXTS = {".mp4", ".avi", ".mkv", ".mov"}
CAMERAS = ["HallWide", "HallSeg", "PF", "PC", "Showroom"]

DST.mkdir(parents=True, exist_ok=True)
index_path = DST / "index.csv"
new_file = not index_path.exists()
csv_f = open(index_path, "a", newline="")
w = csv.writer(csv_f)
if new_file:
    w.writerow(["capture", "camera", "frame_idx", "time_sec", "path"])

def find_camera(p):
    t = str(p).lower()
    for c in CAMERAS:
        if c.lower() in t:
            return c
    return p.stem

def find_capture(p):
    for part in p.parts:
        if part.lower().startswith("capture"):
            return part
    return "capture_unknown"

videos = sorted(p for p in SRC.rglob("*") if p.suffix.lower() in EXTS)
print("Found", len(videos), "videos")

for n, v in enumerate(videos, 1):
    cap_name, cam = find_capture(v), find_camera(v)
    out_dir = DST / cap_name / cam
    done_flag = out_dir / ".done"
    if done_flag.exists():
        print(f"[{n}/{len(videos)}] skip (already done): {cap_name}/{cam}")
        continue
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(v))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx, saved = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % STRIDE == 0:
            fp = out_dir / f"{idx:06d}.jpg"
            cv2.imwrite(str(fp), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            w.writerow([cap_name, cam, idx, round(idx / fps, 3), str(fp)])
            saved += 1
        idx += 1
        if idx % 500 == 0:
            print(f"   {cap_name}/{cam}: {idx}/{total}", end="\r")
    cap.release()
    csv_f.flush()
    done_flag.touch()
    print(f"[{n}/{len(videos)}] {cap_name}/{cam}: {idx} frames read, {saved} saved")

csv_f.close()
print("Done.")