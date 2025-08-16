from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
import pytesseract
import cv2
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from detect_effects import detect_effects  # our custom module

app = Flask(__name__)
CORS(app)


# -----------------------------
# OCR: Extract text from frames
# -----------------------------
def extract_visible_text(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    text_results = []

    for i in range(0, frame_count, 30):  # ~1 frame per second
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        text = pytesseract.image_to_string(frame)
        if text.strip():
            text_results.append({"frame": i, "text": text.strip()})

    cap.release()
    return text_results


# -----------------------------
# Scene Detection
# -----------------------------
def detect_scenes(video_path):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=30.0))
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scenes = scene_manager.get_scene_list()
    return [{"start": str(start), "end": str(end)} for start, end in scenes]


# -----------------------------
# Upload Endpoint
# -----------------------------
@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400

    video = request.files['video']
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        video_path = tmp.name
        video.save(video_path)

    try:
        visible_text = extract_visible_text(video_path)
        scene_changes = detect_scenes(video_path)
        effects = detect_effects(video_path)

        return jsonify({
            "visible_text": visible_text,
            "scene_changes": scene_changes,
            "effects": effects
        })

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)


# -----------------------------
# Run App
# -----------------------------
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
