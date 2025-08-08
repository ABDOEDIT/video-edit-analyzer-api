import cv2
import numpy as np

def detect_effects(video_path):
    flash_shake = detect_flash_shake(video_path)
    zoom = detect_zoom(video_path)
    color_filter = detect_color_filter(video_path)

    return {
        "flash_shake": flash_shake,
        "zoom": zoom,
        "color_filter": color_filter
    }

def detect_flash_shake(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_diffs = []
    prev_frame = None
    timestamps = []

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        if prev_frame is not None:
            diff = cv2.absdiff(blur, prev_frame)
            non_zero_count = np.count_nonzero(diff)
            frame_diffs.append(non_zero_count)

            if non_zero_count > 500000:  # High diff threshold for flash/shake
                timestamps.append({
                    "frame": frame_count,
                    "time": round(frame_count / fps, 2),
                    "effect": "flash/shake"
                })

        prev_frame = blur
        frame_count += 1

    cap.release()
    return timestamps


def detect_zoom(video_path):
    cap = cv2.VideoCapture(video_path)
    zoom_timestamps = []
    frame_count = 0
    prev_area = None
    fps = cap.get(cv2.CAP_PROP_FPS)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(frame_gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        area = sum([cv2.contourArea(c) for c in contours])
        if prev_area and abs(area - prev_area) > 100000:
            zoom_timestamps.append({
                "frame": frame_count,
                "time": round(frame_count / fps, 2),
                "effect": "zoom"
            })

        prev_area = area
        frame_count += 1

    cap.release()
    return zoom_timestamps


def detect_color_filter(video_path):
    cap = cv2.VideoCapture(video_path)
    hsv_scores = []
    frame_count = 0

    while frame_count < 60:  # Analyze ~2 seconds of video
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        avg_hue = hsv[:, :, 0].mean()
        avg_saturation = hsv[:, :, 1].mean()
        avg_value = hsv[:, :, 2].mean()
        hsv_scores.append((avg_hue, avg_saturation, avg_value))

        frame_count += 1

    cap.release()

    avg = np.mean(hsv_scores, axis=0)
    hue, sat, val = avg

    # Simple rule-based filter detection
    if sat < 50:
        return "Black and white or desaturated"
    elif hue > 100 and sat > 100:
        return "Warm or vivid filter"
    elif hue < 15 and sat > 100:
        return "Sepia or warm filter"
    else:
        return "No strong filter detected"
