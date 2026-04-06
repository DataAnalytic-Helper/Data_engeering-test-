import cv2
import psycopg2
import time
import numpy as np
import os
from ultralytics import YOLO
from datetime import datetime, timedelta
from minio import Minio
import io

# --- Настройки из окружения ---
DB_URL = os.getenv("DB_URL", "dbname=user43 user=user43 password=m5q3x8tpc7vn host=2.nntc.nnov.ru port=5402")
RTMP_URL = os.getenv("RTMP_URL", 'rtmp://2.nntc.nnov.ru:5566/stream')
MINIO_ENDPOINT = "2.nntc.nnov.ru:9000"  # Добавлен эндпоинт
MINIO_KEY = os.getenv("MINIO_KEY", "6f9tmxqp4zrb")
MINIO_SECRET = "6f9tmxqp4zrb"  # Секрет (обычно совпадает с ключом в задачах)
BUCKET_NAME = "traffic-snapshots"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Константы
ROI_PERCENT_POINTS = [(0, 0.60), (0.8, 0.20), (1, 1), (0, 1)]
REAL_DIST_M = 15.0
LINE_THRESHOLD = 25
PIXELS_PER_METER = 35
CRITICAL_DIST_M = 2
INCIDENT_COOLDOWN = 2
UPLOAD_INTERVAL_MIN = 30
LINE_START_P1, LINE_START_P2 = (720, 430), (1236, 817)
LINE_END_P1, LINE_END_P2 = (1070, 321), (1606, 474)


def get_pixel_coords(pts, w, h):
    return np.array([(int(p[0] * w), int(p[1] * h)) for p in pts], np.int32)


def point_line_distance(px, py, x1, y1, x2, y2):
    line_vec = np.array([x2 - x1, y2 - y1])
    p_vec = np.array([px - x1, py - y1])
    line_len = np.linalg.norm(line_vec)
    if line_len == 0: return np.linalg.norm(p_vec)
    line_unitvec = line_vec / line_len
    p_vec_scaled = p_vec / line_len
    t = np.clip(np.dot(line_unitvec, p_vec_scaled), 0, 1)
    nearest = line_vec * t
    return np.linalg.norm(p_vec - nearest)


def log_event(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def upload_snapshot(minio_client, frame):
    """Пытается отправить в MinIO, иначе сохраняет локально."""
    filename = f"traffic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"

    # 1. Пытаемся в MinIO
    if minio_client:
        try:
            success, buffer = cv2.imencode('.jpg', frame)
            if success:
                file_data = io.BytesIO(buffer)
                minio_client.put_object(
                    BUCKET_NAME, filename, file_data, len(buffer), content_type='image/jpeg'
                )
                log_event(f"☁️ Снимок отправлен в MinIO: {filename}")
                return
        except:
            log_event("⚠️ Ошибка MinIO, перехожу на локальное сохранение")

    # 2. Локальное сохранение (если MinIO упал или не настроен)
    local_path = os.path.join(BASE_DIR, "local_snapshots")
    if not os.path.exists(local_path):
        os.makedirs(local_path)

    cv2.imwrite(os.path.join(local_path, filename), frame)
    log_event(f"📁 Снимок сохранен локально: {filename}")


def run_detector():
    """Основная функция детекции. Вызывается как сервис оркестрации."""
    model_path = os.path.join(BASE_DIR, 'yolo12s.pt')
    model = YOLO(model_path)
    cap = cv2.VideoCapture(RTMP_URL)

    # Инициализация MinIO
    minio_client = None
    try:
        minio_client = Minio(MINIO_ENDPOINT, access_key=MINIO_KEY, secret_key=MINIO_SECRET, secure=False)
        if not minio_client.bucket_exists(BUCKET_NAME):
            minio_client.make_bucket(BUCKET_NAME)
        log_event("✅ MinIO подключен")
    except:
        log_event("⚠️ MinIO недоступен, будет использоваться локальная папка")

    db_conn = None
    db_active = False
    try:
        db_conn = psycopg2.connect(DB_URL, connect_timeout=5)
        db_conn.autocommit = True
        cursor = db_conn.cursor()
        db_active = True
        print("✅ Оркестратор: БД подключена")
    except Exception as e:
        print(f"⚠️ БД недоступна: {e}")

    total_cars_count = 0
    entry_times, processed_ids, last_incident_time = {}, set(), {}
    final_speeds = {}
    last_upload_time = datetime.now() - timedelta(minutes=UPLOAD_INTERVAL_MIN)

    try:
        while True:
            success, frame = cap.read()
            if not success:
                time.sleep(5)
                cap = cv2.VideoCapture(RTMP_URL)
                continue

            h, w = frame.shape[:2]
            now_ts, now_dt = time.time(), datetime.now()
            roi_pixels = get_pixel_coords(ROI_PERCENT_POINTS, w, h)
            display_frame = frame.copy()

            # Трекинг
            results = model.track(frame, persist=True, verbose=False, classes=[2, 3, 5, 7])
            class_names = model.names

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(float)
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                clss = results[0].boxes.cls.cpu().numpy().astype(int)
                active_objects = []

                for i in range(len(ids)):
                    x1, y1, x2, y2 = boxes[i]
                    t_id, cx, cy = int(ids[i]), (x1 + x2) / 2, (y1 + y2) / 2
                    obj_type = class_names[clss[i]]

                    if cv2.pointPolygonTest(roi_pixels, (int(cx), int(cy)), False) < 0:
                        continue

                    active_objects.append({'id': t_id, 'cx': cx, 'cy': cy, 'box': (x1, y1, x2, y2)})
                    cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                    # Логика скорости
                    d_start = point_line_distance(cx, cy, *LINE_START_P1, *LINE_START_P2)
                    d_end = point_line_distance(cx, cy, *LINE_END_P1, *LINE_END_P2)

                    if d_start < LINE_THRESHOLD and t_id not in entry_times:
                        entry_times[t_id] = now_ts

                    if d_end < LINE_THRESHOLD and t_id in entry_times and t_id not in processed_ids:
                        duration = now_ts - entry_times[t_id]
                        if duration > 0.3:
                            speed = (REAL_DIST_M / duration) * 3.6
                            processed_ids.add(t_id)
                            final_speeds[t_id] = speed
                            total_cars_count += 1
                            log_event(f"ID {t_id}: {speed:.1f} km/h")

                    # --- ЗАПИСЬ ДАННЫХ В БД НА КАЖДОМ КАДРЕ ---
                    if db_active:
                        try:
                            curr_speed = final_speeds.get(t_id, 0)
                            obj_w = (x2 - x1) / PIXELS_PER_METER
                            obj_l = (y2 - y1) / PIXELS_PER_METER
                            x_m = cx / PIXELS_PER_METER
                            y_m = cy / PIXELS_PER_METER

                            cursor.execute("""
                                INSERT INTO user43.full_tracking_data 
                                (video_id, track_id, object_type, width, length, detection_time, x_cord_m, y_cord_m, speed_km_h) 
                                VALUES (12, %s, %s, %s, %s, NOW(), %s, %s, %s)
                            """, (t_id, obj_type, float(obj_w), float(obj_l), float(x_m), float(y_m),
                                  float(curr_speed)))
                        except:
                            # Если БД отвалилась, не крашим основной цикл
                            pass

                    label = f"{obj_type} ID: {t_id}" + (
                        f" | {final_speeds[t_id]:.1f} km/h" if t_id in final_speeds else "")
                    cv2.putText(display_frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 255), 2)

                # Проверка опасных сближений (логика без изменений)
                for i in range(len(active_objects)):
                    for j in range(i + 1, len(active_objects)):
                        o1, o2 = active_objects[i], active_objects[j]
                        dist_m = (((o1['cx'] - o2['cx']) ** 2 + (o1['cy'] - o2['cy']) ** 2) ** 0.5) / PIXELS_PER_METER

                        if dist_m < CRITICAL_DIST_M:
                            pair = tuple(sorted((o1['id'], o2['id'])))
                            if pair not in last_incident_time or (
                                    now_ts - last_incident_time[pair]) > INCIDENT_COOLDOWN:
                                log_event(f"⚠️Опасное сближение: {o1['id']} & {o2['id']} ({dist_m:.2f}m)")
                                if db_active:
                                    try:
                                        dt_now = datetime.now()
                                        cursor.execute(
                                            "INSERT INTO user43.dangerous_incidents (time, track_id1, track_id2, distance) VALUES (%s, %s, %s, %s)",
            (dt_now, o1['id'], o2['id'], float(dist_m))
        )
                                    except:
                                        pass
                                last_incident_time[pair] = now_ts

            # Отрисовка интерфейса (без изменений)
            overlay = display_frame.copy()
            cv2.fillPoly(overlay, [roi_pixels], (255, 191, 0))
            cv2.addWeighted(overlay, 0.25, display_frame, 0.75, 0, display_frame)
            cv2.polylines(display_frame, [roi_pixels], True, (255, 191, 0), 2)
            cv2.line(display_frame, LINE_START_P1, LINE_START_P2, (0, 255, 0), 2)
            cv2.line(display_frame, LINE_END_P1, LINE_END_P2, (0, 0, 255), 2)
            cv2.putText(display_frame, f"TRAFFIC: {total_cars_count}", (25, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                        (255, 255, 255), 3)

            # Облако (теперь с поддержкой локального сохранения)
            if (now_dt - last_upload_time).total_seconds() >= UPLOAD_INTERVAL_MIN * 60:
                upload_snapshot(minio_client, display_frame)
                last_upload_time = now_dt

            cv2.imshow('Traffic Control', cv2.resize(display_frame, (1280, 720)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if db_conn: db_conn.close()


if __name__ == "__main__":
    run_detector()