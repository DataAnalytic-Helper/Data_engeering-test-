import pandas as pd
import psycopg2
import os
from sklearn.ensemble import GradientBoostingRegressor
from datetime import datetime, timedelta

# Параметры теперь можно брать из переменных окружения (стандарт оркестрации)
DB_URL = os.getenv("DB_URL", "dbname=user43 user=user43 password=m5q3x8tpc7vn host=2.nntc.nnov.ru port=5402")
HORIZON = 60


def train_model():
    """Функция обучения модели. Вызывается оркестратором раз в сутки или при запуске."""
    try:
        with psycopg2.connect(DB_URL) as conn:
            # Берем данные для обучения из витрины
            query = "SELECT detection_time, track_id FROM user43.v_ml_traffic_cleaned"
            df = pd.read_sql(query, conn)

        df['detection_time'] = pd.to_datetime(df['detection_time'])
        df['snap_time'] = df['detection_time'].dt.floor('S')

        df_snaps = df.groupby('snap_time')['track_id'].nunique().reset_index(name='cars_now')
        df_snaps = df_snaps.sort_values('snap_time')

        df_target = df_snaps.copy()
        df_target['snap_time'] = df_target['snap_time'] - timedelta(minutes=HORIZON)

        final_data = pd.merge_asof(
            df_snaps,
            df_target[['snap_time', 'cars_now']],
            on='snap_time',
            direction='forward',
            suffixes=('', '_future'),
            tolerance=pd.Timedelta('1min')
        )

        valid_data = final_data.dropna()
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(valid_data[['cars_now']], valid_data['cars_now_future'])
        return model
    except Exception as e:
        print(f"❌ Ошибка обучения: {e}")
        return None


def predict_and_store(model):
    """Функция одного шага прогноза. Вызывается Airflow каждые 5-10 минут."""
    if model is None:
        print("❌ Модель не обучена")
        return

    try:
        with psycopg2.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                # Получаем текущее количество машин
                cur.execute("""
                    SELECT COUNT(DISTINCT track_id) 
                    FROM user43.full_tracking_data 
                    WHERE detection_time >= (SELECT MAX(detection_time) FROM user43.full_tracking_data) - INTERVAL '1 second'
                """)
                curr_cars = float(cur.fetchone()[0] or 0)

                now = datetime.now()
                target_time = (now + timedelta(minutes=HORIZON)).replace(microsecond=0)

                # Прогноз
                pred_val = round(float(model.predict(pd.DataFrame([[curr_cars]], columns=['cars_now']))[0]))
                pred_val = max(0, pred_val)

                cur.execute("""
                    INSERT INTO user43.traffic_predictions 
                    (prediction_made_at, target_time, horizon_minutes, predicted_intensity) 
                    VALUES (NOW(), %s, %s, %s)
                """, (target_time, HORIZON, pred_val))

                print(f"[{now.strftime('%H:%M:%S')}] Прогноз записан оркестратором.")
    except Exception as e:
        print(f"❌ Ошибка предсказания: {e}")


# Если скрипт запущен вручную, он выполнит один цикл
if __name__ == "__main__":
    m = train_model()
    predict_and_store(m)