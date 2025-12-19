import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from ultralytics import YOLO

############################################
# 0. 기본 설정
############################################
DATA_ROOT = r"D:\Abnormal"
SAVE_ROOT = r"D:\ProcessedDataset"
WINDOW_SEC = 5
TOTAL_PRE_SEC = 30
POS_WINDOWS = 2

############################################
# 1. 유틸
############################################
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

############################################
# 2. XML → 이상행동 시작 프레임
############################################
def parse_event_start_frame(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    fps = float(root.find("header/fps").text)

    start_time = root.find("event/starttime").text.strip()
    parts = start_time.split(":")

    if len(parts) == 3:
        h, m, s = parts
        start_sec = int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        start_sec = int(m) * 60 + float(s)
    else:
        raise ValueError(f"Unknown time format: {start_time}")

    start_frame = int(start_sec * fps)
    return fps, start_frame

############################################
# 3. N초 전 window 생성
############################################
def make_pre_event_windows(start_frame, fps):
    window_frames = int(WINDOW_SEC * fps)
    total_frames = int(TOTAL_PRE_SEC * fps)

    windows = []
    cur = start_frame
    idx = 0

    while cur - window_frames >= start_frame - total_frames:
        label = 1 if idx < POS_WINDOWS else 0
        windows.append({
            "start": cur - window_frames,
            "end": cur,
            "label": label
        })
        cur -= window_frames
        idx += 1

    return windows[::-1]

############################################
# 4. 프레임 구간 읽기
############################################
def read_video_frames(video_path, start, end):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    frames = []
    cur = start

    while cur < end:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        cur += 1

    cap.release()
    return frames

############################################
# 5. YOLO 사람 중심 좌표 추출
############################################
def extract_person_centers(frames, model):
    centers = []
    missing = 0  # ⭐ person 미검출 프레임 수

    for frame in frames:
        results = model(frame, verbose=False)[0]
        persons = []

        for box in results.boxes:
            if int(box.cls) == 0:  # person
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                area = (x2 - x1) * (y2 - y1)
                persons.append((area, x1, y1, x2, y2))

        if persons:
            _, x1, y1, x2, y2 = max(persons, key=lambda x: x[0])
            centers.append([(x1 + x2) / 2, (y1 + y2) / 2])
        else:
            centers.append([np.nan, np.nan])
            missing += 1

    return np.array(centers), missing

############################################
# 6. 이동 특징 생성
############################################
def make_motion_features(centers):
    feats = []
    prev_dx, prev_dy = 0, 0

    for i in range(1, len(centers)):
        x1, y1 = centers[i - 1]
        x2, y2 = centers[i]

        if np.any(np.isnan([x1, y1, x2, y2])):
            feats.append([0, 0, 0, 0, 0])
            continue

        dx = x2 - x1
        dy = y2 - y1
        speed = np.sqrt(dx**2 + dy**2)
        ax = dx - prev_dx
        ay = dy - prev_dy

        feats.append([dx, dy, speed, ax, ay])
        prev_dx, prev_dy = dx, dy

    return np.array(feats)

############################################
# 7. window 길이 고정
############################################
def pad_or_trim(features, target_len):
    T, F = features.shape
    if T > target_len:
        return features[:target_len]
    elif T < target_len:
        pad = np.zeros((target_len - T, F))
        return np.vstack([features, pad])
    return features

############################################
# 8. 전체 데이터셋 전처리 + 통계
############################################
def preprocess_dataset(class_name):
    print(f"\n[START] {class_name}")

    model = YOLO("yolov8n.pt")

    class_root = os.path.join(DATA_ROOT, class_name)
    save_dir = os.path.join(SAVE_ROOT, class_name)
    ensure_dir(save_dir)

    X_all, y_all, groups = [], [], []

    fps_list = []          # fps 통계
    total_frames = 0       # 전체 프레임 수
    missing_frames = 0     # person 미검출 프레임 수

    for scenario in sorted(os.listdir(class_root)):
        scenario_path = os.path.join(class_root, scenario)
        if not os.path.isdir(scenario_path):
            continue

        print(f"  ▶ Scenario {scenario}")

        for file in os.listdir(scenario_path):
            if not file.endswith(".xml"):
                continue

            xml_path = os.path.join(scenario_path, file)
            video_path = xml_path.replace(".xml", ".mp4")
            if not os.path.exists(video_path):
                continue

            fps, start_frame = parse_event_start_frame(xml_path)
            fps_list.append(fps)

            windows = make_pre_event_windows(start_frame, fps)

            for w in windows:
                frames = read_video_frames(video_path, w["start"], w["end"])
                centers, miss = extract_person_centers(frames, model)

                total_frames += len(centers)
                missing_frames += miss

                features = make_motion_features(centers)
                features = pad_or_trim(
                    features,
                    target_len=int(WINDOW_SEC * fps) - 1
                )

                X_all.append(features)
                y_all.append(w["label"])
                groups.append(scenario)

    X_all = np.array(X_all)
    y_all = np.array(y_all)
    groups = np.array(groups)

    np.save(os.path.join(save_dir, "X.npy"), X_all)
    np.save(os.path.join(save_dir, "y.npy"), y_all)
    np.save(os.path.join(save_dir, "groups.npy"), groups)

    print(f"[DONE] {class_name}")
    print(" X shape:", X_all.shape)
    print(" y shape:", y_all.shape)
    print(" groups shape:", groups.shape)

    # 통계 출력
    print(f" FPS mean: {np.mean(fps_list):.2f}, "
          f"min: {np.min(fps_list)}, max: {np.max(fps_list)}")

    print(f" Person missing ratio: "
          f"{missing_frames / total_frames * 100:.2f}%")

############################################
# 9. 실행
############################################
if __name__ == "__main__":
    ensure_dir(SAVE_ROOT)

    preprocess_dataset("assault")
    preprocess_dataset("fight")