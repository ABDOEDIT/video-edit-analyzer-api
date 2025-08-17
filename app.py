from flask import Flask, request, jsonify
from flask_cors import CORS
import os, tempfile, traceback
import cv2
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

# Optional OCR: import if available; disable silently on servers without Tesseract binary
try:
    import pytesseract  # requires tesseract-ocr binary on the OS (not present on Render Native)
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

from detect_effects import detect_effects  # your effects module

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def index():
    return jsonify({"ok": True, "message": "Video Edit Analyzer API", "ocr_available": OCR_AVAILABLE})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

def extract_visible_text(video_path):
    """Attempt OCR ~1 fps; if OCR/FFmpeg unavailable, return empty list instead of crashing."""
    results = []
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("[WARN] OpenCV could not open video for OCR.")
            return results

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        step = max(1, int(cap.get(cv2.CAP_PROP_FPS()) or 30))  # ~1 fps
        if not OCR_AVAILABLE:
            print("[INFO] OCR disabled: pytesseract/tesseract not available.")
            cap.release()
            return results

        for i in range(0, frame_count, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            try:
                text = pytesseract.image_to_string(frame)
                if text and text.strip():
                    results.append({"frame": i, "text": text.strip()})
            except Exception as e:
                # If Tesseract is missing at runtime, fail gracefully
                print(f"[WARN] OCR failed at frame {i}: {e}")
                break
        cap.release()
    except Exception as e:
        print("[ERROR] extract_visible_text crashed:", e)
        print(traceback.format_exc())
    return results

def detect_scenes(video_path):
    """Shot boundary detection with SceneDetect. Fail gracefully on errors."""
    try:
        video_manager = VideoManager([video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=30.0))
        video_manager.set_downscale_factor()
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        scenes = scene_manager.get_scene_list()
        video_manager.release()
        return [{"start": str(start), "end": str(end)} for start, end in scenes]
    except Exception as e:
        print("[ERROR] detect_scenes crashed:", e)
        print(traceback.format_exc())
        return []

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file uploaded (field name must be 'video')"}), 400

    video = request.files['video']
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        video_path = tmp.name
        video.save(video_path)

    try:
        print(f"[INFO] Received file: {video.filename} -> {video_path}")
        visible_text = extract_visible_text(video_path)
        scene_changes = detect_scenes(video_path)

        effects = []
        color_filter = "unknown"
        try:
            detected = detect_effects(video_path)
            # Keep backward compatibility with your UI:
            # if detect_effects returns dict with lists + color_filter:
            if isinstance(detected, dict):
                effects_payload = {
                    "flash_shake": detected.get("flash_shake", []),
                    "zoom": detected.get("zoom", []),
                }
                color_filter = detected.get("color_filter", "unknown")
            else:
                # if it returns a flat list, map to a single list
                effects_payload = {"flash_shake": detected, "zoom": []}
            effects = effects_payload
        except Exception as e:
            print("[ERROR] detect_effects crashed:", e)
            print(traceback.format_exc())
            effects = {"flash_shake": [], "zoom": []}

        return jsonify({
            "visible_text": visible_text,
            "scene_changes": scene_changes,
            "effects": {**effects, "color_filter": color_filter},
            "ocr_available": OCR_AVAILABLE
        })

    except Exception as e:
        print("[ERROR] /upload crashed:", e)
        print(traceback.format_exc())
        return jsonify({"error": "Processing failed", "details": str(e)}), 500

    finally:
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
        except Exception:
            pass

if __name__ == '__main__':
    # Render provides PORT env var; bind on all interfaces.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
