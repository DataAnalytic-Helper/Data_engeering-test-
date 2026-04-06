import pandas as pd
import psycopg2
import os
import sys
from sklearn.ensemble import GradientBoostingRegressor
from datetime import datetime, timedelta

# Настройка путей
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_URL = os.getenv("DB_URL", "dbname=user43 user=user43 password=m5q3x8tpc7vn host=2.nntc.nnov.ru port=5402")
HORIZON = 60


def train_model():
    try:
        with psycopg2.connect(DB_URL) as conn:
            query = "SELECT detection_time, track_id FROM user43.v_ml_traffic_cleaned"
            df = pd.read_sql(query, conn)

        if df.empty or len(df) < 10:
            print("⚠️ Недостаточно данных для обучения")
            return None

        df['detection_time'] = pd.to_datetime(df['detection_time'])
        df_snaps = df.groupby(df['detection_time'].dt.floor('min'))['track_id'].nunique().reset_index(name='cars_now')
        df_snaps = df_snaps.sort_values('detection_time')

        # Обучаем на текущих данных
        model = GradientBoostingRegressor(n_estimators=50, random_state=42)
        # Для простоты: учим предсказывать то же самое число (в реальном кейсе нужны лаги)
        model.fit(df_snaps[['cars_now']], df_snaps['cars_now'])
        return model
    except Exception as e:
        print(f"❌ Ошибка обучения: {e}")
        return None


def predict_and_store(model):
    try:
        with psycopg2.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(DISTINCT track_id) FROM user43.full_tracking_data WHERE detection_time > NOW() - INTERVAL '5 minutes'")
                curr_cars = float(cur.fetchone()[0] or 0)

                pred_val = max(0, int(model.predict([[curr_cars]])[0]))
                target_time = datetime.now() + timedelta(minutes=HORIZON)

                cur.execute("""
                    INSERT INTO user43.traffic_predictions 
                    (prediction_made_at, target_time, horizon_minutes, predicted_intensity) 
                    VALUES (NOW(), %s, %s, %s)
                """, (target_time, HORIZON, pred_val))
                print(f"✅ Прогноз на {target_time}: {pred_val} ТС записан.")
    except Exception as e:
        print(f"❌ Ошибка предсказания: {e}")


if __name__ == "__main__":
    model = train_model()
    if model:
        predict_and_store(model)