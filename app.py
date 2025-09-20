from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os, tempfile, json, shutil, zipfile, uuid, io, time
import cv2, numpy as np, pytesseract

from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from detect_effects import detect_effects  # your effects module

app = Flask(__name__)
CORS(app)

# Directory to persist uploaded videos (between /upload and /generate_capcut calls)
PERSIST_UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "video_edit_analyzer_uploads")
os.makedirs(PERSIST_UPLOAD_DIR, exist_ok=True)


# ---------- Video metadata ----------
def get_video_meta(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frames / fps if fps > 0 else 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    cap.release()
    return float(duration), frames, float(fps), (width, height)


# ---------- OCR ----------
def extract_visible_text(path):
    duration, total_frames, fps, _ = get_video_meta(path)
    cap = cv2.VideoCapture(path)
    results = []
    step = max(1, int(round(fps)))  # ~1 sample per second
    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]
        txt = pytesseract.image_to_string(gray)
        if txt and txt.strip():
            results.append({"time": round(i / fps, 2), "content": txt.strip(), "style": "plain"})
    cap.release()
    return results


# ---------- Scene detection ----------
def detect_scenes(path):
    video_manager = VideoManager([path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=30.0))
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scenes = scene_manager.get_scene_list()
    video_manager.release()

    structured = []
    for idx, (start, end) in enumerate(scenes, start=1):
        structured.append({
            "id": idx,
            "start": float(start.get_seconds()),
            "end": float(end.get_seconds()),
            "transition": "cut"
        })
    return structured


# ---------- Effects & beats (we assume detect_effects is available) ----------
def detect_beats_visual(path):
    # keep a small visual-beat detector if needed (copy from earlier code)
    duration, total_frames, fps, _ = get_video_meta(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened() or total_frames == 0 or fps == 0:
        return []

    sample_fps = min(10.0, fps)
    step = max(1, int(round(fps / sample_fps)))
    energies = []
    times = []
    idx = 0
    while idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        energies.append(float(np.var(gray)))
        times.append(idx / fps)
        idx += step
    cap.release()
    if len(energies) < 5:
        return []
    e = np.array(energies, dtype=np.float32)
    w = max(3, int(round(sample_fps * 0.5)))
    ma = np.convolve(e, np.ones(w) / w, mode='same')
    hp = e - ma
    peaks = []
    last_t = -999
    thr = max(0.0, float(np.mean(hp) + 0.75 * np.std(hp)))
    for i in range(1, len(hp) - 1):
        if hp[i] > thr and hp[i] > hp[i - 1] and hp[i] > hp[i + 1]:
            t = times[i]
            if t - last_t >= 0.2:
                peaks.append({"time": round(float(t), 2)})
                last_t = t
    return peaks


# ---------- /upload endpoint ----------
@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    f = request.files["video"]
    # Save to persistent uploads dir with UUID filename
    vid_id = str(uuid.uuid4())
    ext = ".mp4"
    saved_path = os.path.join(PERSIST_UPLOAD_DIR, vid_id + ext)
    f.save(saved_path)

    try:
        duration, _, fps, (w, h) = get_video_meta(saved_path)
        texts = extract_visible_text(saved_path)
        scenes = detect_scenes(saved_path)
        effects = detect_effects(saved_path) or {}
        beats = detect_beats_visual(saved_path)

        # Return video_id so generate_capcut can slice this saved file
        return jsonify({
            "video_id": vid_id,
            "video_path": saved_path,           # for debugging (you can remove later)
            "duration": round(duration, 2),
            "fps": round(fps, 2),
            "resolution": f"{w}x{h}",
            "timeline": {
                "scenes": scenes,
                "texts": texts,
                "effects": effects,
                "beats": beats,
                "color_filter": effects.get("color_filter", "none")
            }
        })
    except Exception as e:
        # on errors remove saved file
        if os.path.exists(saved_path):
            os.remove(saved_path)
        raise
        

# ---------- Helper: extract segment from video into new mp4 file ----------
def extract_segment(input_path, out_path, start_s, end_s, target_fps=None):
    """
    Extract frames from input_path between start_s and end_s (seconds),
    and write them to out_path as mp4 using cv2 VideoWriter.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video for slicing")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if target_fps is None:
        target_fps = int(round(fps))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360)

    start_frame = int(round(start_s * fps))
    end_frame = int(round(end_s * fps))
    end_frame = max(end_frame, start_frame + 1)

    # Prepare writer (use mp4v)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, target_fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame
    while frame_idx <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        frame_idx += 1

    writer.release()
    cap.release()


# ---------- /generate_capcut endpoint ----------
@app.route("/generate_capcut", methods=["POST"])
def generate_capcut():
    """
    Payload expects:
    {
      "video_id": "<id returned by /upload>",
      "duration": <project duration>,
      "timeline": { scenes, texts, effects, beats, ... }
    }
    """
    data = request.get_json()
    if not data or "timeline" not in data or "video_id" not in data:
        return jsonify({"error": "Invalid payload; need video_id and timeline"}), 400

    video_id = data["video_id"]
    timeline = data["timeline"]
    duration = float(data.get("duration", 0))

    saved_video_path = os.path.join(PERSIST_UPLOAD_DIR, video_id + ".mp4")
    if not os.path.exists(saved_video_path):
        return jsonify({"error": "video_id not found on server"}), 404

    # Create temp project folder
    project_id = str(uuid.uuid4())
    project_dir = tempfile.mkdtemp()
    medias_dir = os.path.join(project_dir, "medias")
    os.makedirs(medias_dir, exist_ok=True)

    # 1) create real scene clips by slicing saved_video_path
    for i, scene in enumerate(timeline.get("scenes", [])):
        start = float(scene.get("start", 0))
        end = float(scene.get("end", 0))
        if end <= start:
            end = min(start + 0.5, duration)  # fallback
        clip_filename = f"scene_{i+1}.mp4"
        clip_path = os.path.join(medias_dir, clip_filename)
        # Attempt to extract at original resolution/fps
        try:
            extract_segment(saved_video_path, clip_path, start, end)
        except Exception:
            # If extraction fails, fall back to a single-color placeholder of same duration
            create_colored_fallback = True
            # create a simple colored placeholder using numpy + VideoWriter
            w, h = 640, 360
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps_write = 25
            out = cv2.VideoWriter(clip_path, fourcc, fps_write, (w, h))
            frame = np.full((h, w, 3), (64 + i * 40 % 255, 64, 200), dtype=np.uint8)
            frames_count = int(max(1, round((end - start) * fps_write)))
            for _ in range(frames_count):
                out.write(frame)
            out.release()

    # 2) create effect clips (small slices around effect time; if within video)
    for i, eff in enumerate(timeline.get("effects", []) if timeline.get("effects") else []):
        t = float(eff.get("time", 0))
        small_start = max(0.0, t - 0.1)
        small_end = t + 0.2
        clip_filename = f"effect_{i+1}.mp4"
        clip_path = os.path.join(medias_dir, clip_filename)
        try:
            extract_segment(saved_video_path, clip_path, small_start, small_end)
        except Exception:
            # fallback colored small clip
            w, h = 640, 360
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps_write = 25
            out = cv2.VideoWriter(clip_path, fourcc, fps_write, (w, h))
            frame = np.full((h, w, 3), (255, 255, 255), dtype=np.uint8)  # white flash
            frames_count = int(max(1, round((small_end - small_start) * fps_write)))
            for _ in range(frames_count):
                out.write(frame)
            out.release()

    # 3) Build CapCut-style project.json (clips + text overlays + markers)
    project = {
        "project": {
            "name": "AI Generated Project",
            "duration": duration,
            "tracks": [],
            "markers": []
        }
    }

    # video tracks (scenes)
    for i, scene in enumerate(timeline.get("scenes", [])):
        project["project"]["tracks"].append({
            "id": f"scene_{i+1}",
            "type": "video",
            "start": scene.get("start", 0),
            "end": scene.get("end", 0),
            "media": f"medias/scene_{i+1}.mp4"
        })

    # text overlays
    for i, t in enumerate(timeline.get("texts", [])):
        project["project"]["tracks"].append({
            "id": f"text_{i+1}",
            "type": "text",
            "start": t.get("time", 0),
            "end": t.get("time", 0) + 2.0,
            "content": t.get("content", "")
        })

    # effects added as small clips
    for i, eff in enumerate(timeline.get("effects", []) if timeline.get("effects") else []):
        project["project"]["tracks"].append({
            "id": f"effect_{i+1}",
            "type": "video",
            "start": eff.get("time", 0),
            "end": eff.get("time", 0) + 0.3,
            "media": f"medias/effect_{i+1}.mp4"
        })

    # beat markers
    for i, b in enumerate(timeline.get("beats", []) if timeline.get("beats") else []):
        project["project"]["markers"].append({
            "id": f"beat_{i+1}",
            "time": b.get("time", 0)
        })

    # effects metadata included for completeness
    project["project"]["effects_metadata"] = timeline.get("effects", [])

    # draft_info.json
    draft_info = {
        "name": "AI Generated Project",
        "cover": "",
        "createTime": int(time.time() * 1000),
        "updateTime": int(time.time() * 1000),
        "platform": "pc",
        "version": "1.0.0",
        "draft_version": "3.0.0"
    }

    # 4) Package into .zip with .capcutproj root
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        folder = "AI_Project.capcutproj/"
        zf.writestr(folder + "project.json", json.dumps(project, indent=2))
        zf.writestr(folder + "draft_info.json", json.dumps(draft_info, indent=2))
        # add medias
        for root, _, files in os.walk(medias_dir):
            for file in files:
                full = os.path.join(root, file)
                rel = os.path.join(folder, "medias", file)
                zf.write(full, rel)

    zip_buffer.seek(0)

    # Cleanup the temp project_dir (keep original upload for later if wanted)
    shutil.rmtree(project_dir)

    return send_file(
        zip_buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"capcut_project_{video_id}.zip"
    )


if __name__ == "__main__":
    # dev: default 5000
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
