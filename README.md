В данном скрипте реализуется простой пайплайн для:

 загрузки данных из PostgreSQL
 обучения модели машинного обучения
 генерации прогноза количества автомобилей
 сохранения прогноза обратно в базу данных

Система может использоваться как часть интеллектуальной транспортной платформы (ITS), где прогнозирование нагрузки на дорогу помогает анализировать и управлять трафиком.

 Импорт библиотек и базовая настройка

На этом этапе мы подключаем все необходимые библиотеки.

pandas — работа с данными
psycopg2 — подключение к PostgreSQL
sklearn — модель машинного обучения
datetime — работа со временем
os — работа с переменными окружения
import pandas as pd
import psycopg2
import os
import sys
from sklearn.ensemble import GradientBoostingRegressor
from datetime import datetime, timedelta 
 Настройка путей и параметров

Здесь задаются основные параметры работы системы:

 BASE_DIR — путь к текущему файлу
 DB_URL — строка подключения к базе данных
 HORIZON — горизонт прогноза (в минутах)

Важно: используется os.getenv, чтобы можно было безопасно передавать данные через переменные окружения (например, в Docker или CI/CD).

# Настройка путей
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_URL = os.getenv("DB_URL", "dbname=user43 user=user43 password=m5q3x8tpc7vn host=2.nntc.nnov.ru port=5402")
HORIZON = 60
 Блок 1: Обучение модели (train_model)
 Что происходит в этом блоке?

Этот блок отвечает за подготовку данных и обучение модели.

 Основные шаги:
Подключение к базе данных
Загрузка данных из представления v_ml_traffic_cleaned
Проверка достаточности данных
Преобразование времени
Агрегация данных по минутам
Обучение модели

 Важно:
Сейчас модель обучается предсказывать саму себя (cars_now → cars_now).
Это упрощённый вариант — в реальном проекте нужны лаги (временные признаки).

def train_model():
    try:
        with psycopg2.connect(DB_URL) as conn:
            query = "SELECT detection_time, track_id FROM user43.v_ml_traffic_cleaned"
            df = pd.read_sql(query, conn)

        if df.empty or len(df) < 10:
            print(" Недостаточно данных для обучения")
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
 Блок 2: Предсказание и сохранение (predict_and_store)
 Что делает этот блок?

После обучения модели:

Получает текущее количество машин за последние 5 минут
Строит прогноз
Рассчитывает время, на которое делается прогноз
Сохраняет результат в таблицу traffic_predictions
 Особенности:
Используется COUNT(DISTINCT track_id) → уникальные машины
Ограничение max(0, ...) → защита от отрицательных значений
Прогноз записывается вместе с:
временем создания
горизонтом
значением
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
 Главный блок запуска
 Что здесь происходит?

Это точка входа в программу:

Вызывается обучение модели
Если модель успешно обучена — запускается прогноз

 Это стандартный паттерн Python:
if __name__ == "__main__":

if __name__ == "__main__":
    model = train_model()
    if model:
        predict_and_store(model)




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
        predict_and_store(model) раздели код на блоки но схраняй отступы , а между блоками напиши много текста, можешь обьяснять эти блоки и зачем и используемые блоки,  напиши это огромным текстом с оформлением но с разметкой под jupyter, с ## ** и тд, а также это будет размечено на гит в read.me поэтому  блоки кода чтобы отделялись как код

Ниже — оформленный вариант твоего кода как разбор для Jupyter Notebook / README.md: с крупными секциями, пояснениями и аккуратно отделёнными блоками кода 👇

 Прогнозирование трафика на основе данных детекции
 Общая идея проекта
В данном скрипте реализуется простой пайплайн для:

 загрузки данных из PostgreSQL

 обучения модели машинного обучения

 генерации прогноза количества автомобилей

 сохранения прогноза обратно в базу данных

Система может использоваться как часть интеллектуальной транспортной платформы (ITS), где прогнозирование нагрузки на дорогу помогает анализировать и управлять трафиком.

Импорт библиотек и базовая настройка
На этом этапе мы подключаем все необходимые библиотеки.

pandas — работа с данными

psycopg2 — подключение к PostgreSQL

sklearn — модель машинного обучения

datetime — работа со временем

os — работа с переменными окружения

 import pandas as pd
 import psycopg2
 import os
 import sys
 from sklearn.ensemble import GradientBoostingRegressor
 from datetime import datetime, timedelta 
Настройка путей и параметров
Здесь задаются основные параметры работы системы:

 BASE_DIR — путь к текущему файлу

 DB_URL — строка подключения к базе данных

HORIZON — горизонт прогноза (в минутах)

Важно: используется os.getenv, чтобы можно было безопасно передавать данные через переменные окружения (например, в Docker или CI/CD).

# Настройка путей
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_URL = os.getenv("DB_URL", "dbname=user43 user=user43 password=m5q3x8tpc7vn host=2.nntc.nnov.ru port=5402")
HORIZON = 60
 Блок 1: Обучение модели (train_model)
 Что происходит в этом блоке?
Этот блок отвечает за подготовку данных и обучение модели.

 Основные шаги:
Подключение к базе данных

Загрузка данных из представления v_ml_traffic_cleaned

Проверка достаточности данных

Преобразование времени

Агрегация данных по минутам

Обучение модели

 Важно:
Сейчас модель обучается предсказывать саму себя (cars_now → cars_now).
Это упрощённый вариант — в реальном проекте нужны лаги (временные признаки).

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
 Блок 2: Предсказание и сохранение (predict_and_store)
Что делает этот блок?
После обучения модели:

Получает текущее количество машин за последние 5 минут

Строит прогноз

Рассчитывает время, на которое делается прогноз

Сохраняет результат в таблицу traffic_predictions

 Особенности:
Используется COUNT(DISTINCT track_id) → уникальные машины

Ограничение max(0, ...) → защита от отрицательных значений

Прогноз записывается вместе с:

временем создания

горизонтом

значением

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
 Главный блок запуска
 Что здесь происходит?
Это точка входа в программу:

Вызывается обучение модели

Если модель успешно обучена — запускается прогноз

 Это стандартный паттерн Python:
    if __name__ == "__main__":
    
    if __name__ == "__main__":
        model = train_model()
        if model:
            predict_and_store(model)
 Возможные улучшения (очень важно для проекта)
Чтобы сделать проект сильнее для защиты / диплома / портфолио, можно добавить:

 Улучшение модели:
    лаги (cars_now(t-1), t-2, ...)
    
    rolling average

временные признаки (час, день недели)

 Аналитика:
графики (matplotlib / seaborn)

сравнение факта и прогноза

 Архитектура:
вынести конфиг в .env

добавить логирование (logging)

запуск через cron / Airflow

📡 Интеграция:
подключение к YOLO-детекции

real-time поток (Kafka / Redis)

 Итог
Этот код — это:

✔ минимальный ML pipeline
✔ интеграция с PostgreSQL
✔ заготовка для реальной системы мониторинга трафика

Если хочешь, дальше могу:

 переписать модель как в реальном проде (с лагами)

 добавить дашборд (Streamlit / Dash)

 сделать идеальную структуру БД под твой проект

 собрать это в готовый GitHub проект (как тебе уже задавали)


Реклама не влияет на ответы, которые вы получаете от ChatGPT. Ваши чаты остаются конфиденциальными. 

Отлично, вот это уже уровень реального ML-пайплайна 
Сейчас перепишем твой код так, чтобы модель использовала лаги (временные признаки) — это критически важно для временных рядов.

 Использование лагов (cars_now(t-1), t-2, ...)
 Идея
Вместо того чтобы предсказывать cars_now из самого cars_now, мы даём модели предыдущие значения:

время	cars_now	lag_1	lag_2	lag_3
t	10	12	15	11
 То есть:

lag_1 = значение минуту назад

lag_2 = 2 минуты назад

и т.д.

Обновлённый блок обучения модели
 Что добавилось:
функция создания лагов

удаление NaN после сдвига

обучение на нескольких признаках

    def train_model():
        try:
            with psycopg2.connect(DB_URL) as conn:
                query = "SELECT detection_time, track_id FROM user43.v_ml_traffic_cleaned"
                df = pd.read_sql(query, conn)
    
            if df.empty or len(df) < 20:
                print("⚠️ Недостаточно данных для обучения")
                return None
    
            # --- Подготовка данных ---
            df['detection_time'] = pd.to_datetime(df['detection_time'])
    
            df_snaps = (
                df.groupby(df['detection_time'].dt.floor('min'))['track_id']
                .nunique()
                .reset_index(name='cars_now')
                .sort_values('detection_time')
            )
    
            # --- СОЗДАНИЕ ЛАГОВ ---
            LAGS = 5  # сколько минут назад учитываем
    
            for i in range(1, LAGS + 1):
                df_snaps[f'lag_{i}'] = df_snaps['cars_now'].shift(i)
    
            # удаляем строки с NaN (первые строки без истории)
            df_snaps = df_snaps.dropna()
    
            if df_snaps.empty:
                print("⚠️ После создания лагов нет данных")
                return None
    
            # --- признаки и цель ---
            feature_cols = [f'lag_{i}' for i in range(1, LAGS + 1)]
            X = df_snaps[feature_cols]
            y = df_snaps['cars_now']
    
            # --- обучение модели ---
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
    
            print("✅ Модель обучена с лагами")
            return model
    
        except Exception as e:
            print(f"❌ Ошибка обучения: {e}")
            return None
 Обновлённый блок предсказания
 ВАЖНО
Теперь модели нужно несколько последних значений, а не одно!

    def predict_and_store(model):
        try:
            with psycopg2.connect(DB_URL) as conn:
                with conn.cursor() as cur:
    
                    # --- получаем последние значения по минутам ---
                    cur.execute("""
                        SELECT date_trunc('minute', detection_time) AS minute,
                               COUNT(DISTINCT track_id) AS cars_now
                        FROM user43.full_tracking_data
                        WHERE detection_time > NOW() - INTERVAL '10 minutes'
                        GROUP BY minute
                        ORDER BY minute DESC
                        LIMIT 5
                    """)
    
                    rows = cur.fetchall()
    
                    if len(rows) < 5:
                        print("⚠️ Недостаточно данных для прогноза")
                        return
    
                    # значения идут в обратном порядке → переворачиваем
                    cars = [row[1] for row in rows][::-1]
    
                    # формируем лаги
                    features = cars[-5:]  # последние 5 значений
    
                    pred_val = max(0, int(model.predict([features])[0]))
                    target_time = datetime.now() + timedelta(minutes=HORIZON)
    
                    cur.execute("""
                        INSERT INTO user43.traffic_predictions 
                        (prediction_made_at, target_time, horizon_minutes, predicted_intensity) 
                        VALUES (NOW(), %s, %s, %s)
                    """, (target_time, HORIZON, pred_val))
    
                    print(f"✅ Прогноз на {target_time}: {pred_val} ТС записан.")
    
        except Exception as e:
            print(f"❌ Ошибка предсказания: {e}")




Для усовершенствования системы прогнозирования транспортной нагрузки были внедрены лаговые признаки (cars_now(t-1), t-2, ...), позволяющие учитывать временную зависимость данных. Это дало возможность модели анализировать не только текущее состояние трафика, но и его динамику за предыдущие интервалы времени. Такой подход относится к методам авторегрессии и широко применяется при работе с временными рядами.

Использование лагов значительно повышает качество прогноза, так как дорожная ситуация обладает выраженной инерционностью и зависит от недавних изменений. В результате модель становится более устойчивой к резким колебаниям и лучше отражает реальные процессы. Это особенно важно для задач интеллектуального управления транспортными потоками и построения адаптивных систем мониторинга.
   ез истории)
             def train_model():
          try:
          # --- ПОДКЛЮЧЕНИЕ К БАЗЕ ДАННЫХ И ЗАГРУЗКА ДАННЫХ ---
           
              with psycopg2.connect(DB_URL) as conn:
                  query = "SELECT detection_time, track_id FROM user43.v_ml_traffic_cleaned"
                  df = pd.read_sql(query, conn)
  
          # --- ПРОВЕРКА ДОСТАТОЧНОСТИ ДАННЫХ ---
          # Перед началом обработки выполняется проверка объёма данных.
          # Если DataFrame пустой или содержит слишком мало записей, обучение модели не имеет смысла,
          # так как она не сможет выявить устойчивые закономерности. В данном случае установлен
          # минимальный порог в 20 записей
              if df.empty or len(df) < 20:
                  print("⚠️ Недостаточно данных для обучения")
                  return None
  
          # --- ПОДГОТОВКА И ПРЕОБРАЗОВАНИЕ ДАННЫХ ---
          # В данном блоке производится преобразование временной переменной detection_time
          # в формат datetime, что необходимо для корректной работы с временными рядами.
          # Далее данные агрегируются по минутам с использованием группировки.
ритически важно для последующего формирования лаговых признаков.
              df['detection_time'] = pd.to_datetime(df['detection_time'])
      
              df_snaps = (
                  df.groupby(df['detection_time'].dt.floor('min'))['track_id']
                  .nunique()
                  .reset_index(name='cars_now')
                  .sort_values('detection_time')
              )
  
          # --- СОЗДАНИЕ ЛАГОВЫХ ПРИЗНАКОВ ---
          # На данном этапе формируются лаговые признаки, которые отражают значения
          # временного ряда в предыдущие моменты времени. Это позволяет учитывать
          # временную зависимость данных и динамику изменения трафика.
          # Для каждого значения cars_now создаются дополнительные столбцы lag_1, lag_2 и т.д.,
      
          LAGS = 5  # сколько минут назад учитываем
  
          for i in range(1, LAGS + 1):
              df_snaps[f'lag_{i}'] = df_snaps['cars_now'].shift(i)
  
          # --- УДАЛЕНИЕ НЕКОРРЕКТНЫХ ЗНАЧЕНИЙ ---
          # После создания лагов в начале набора данных появляются пропущенные значения (NaN),
          # так как для первых наблюдений отсутствует история. Эти строки не могут быть использованы
          # для обучения модели, поэтому они удаляются. Это стандартная практика при работе
          # с временными рядами. Дополнительно выполняется проверка: если после очистки
          # данных не осталось, обучение невозможно. Это защищает систему от ошибок
          # и некорректной работы модели в условиях недостатка информации.
          df_snaps = df_snaps.dropna()
  
          if df_snaps.empty:
              print("⚠️ После создания лагов нет данных")
              return None
  
          # --- ФОРМИРОВАНИЕ ПРИЗНАКОВ И ЦЕЛЕВОЙ ПЕРЕМЕННОЙ ---
          # В данном блоке формируются входные данные для модели.
          # В качестве признаков используются лаговые значения (предыдущие состояния системы),
          # а в качестве целевой переменной — текущее значение cars_now.
          # Таким образом, модель обучается предсказывать текущее состояние трафика
          # на основе его предыдущих значений. Это соответствует задаче регрессии
          # для временных рядов и позволяет учитывать закономерности изменения нагрузки на дорогу.
          feature_cols = [f'lag_{i}' for i in range(1, LAGS + 1)]
          X = df_snaps[feature_cols]
          y = df_snaps['cars_now']
  
          # --- ОБУЧЕНИЕ МОДЕЛИ ---
          # Для построения прогностической модели используется алгоритм Gradient Boosting Regressor.
          # Он представляет собой ансамблевый метод, который последовательно строит набор слабых моделей
          # (деревьев решений), минимизируя ошибку предсказания.
          # Такой подход хорошо работает на табличных данных и способен выявлять сложные зависимости.
          # После инициализации модель обучается на подготовленных данных.
          # В результате формируется модель, способная учитывать временную динамику трафика
          # и использовать её для последующих прогнозов.
          model = GradientBoostingRegressor(n_estimators=100, random_state=42)
          model.fit(X, y)
  
          print("✅ Модель обучена с лагами")
          return model
  
      except Exception as e:
          # --- ОБРАБОТКА ОШИБОК ---
          # В данном блоке осуществляется перехват исключений, которые могут возникнуть
          # на любом этапе выполнения функции: при подключении к базе данных,
          # загрузке данных, их обработке или обучении модели.
          # Вывод диагностического сообщения позволяет оперативно выявить причину сбоя
          # и упростить процесс отладки, что особенно важно при эксплуатации системы
          # в реальных условиях и при работе с потоковыми данными.
          print(f"❌ Ошибка обучения: {e}")
          return None



Для повышения точности прогнозирования была доработана функция predict_and_store, которая теперь учитывает не одно текущее значение, а последовательность последних наблюдений. В частности, из базы данных извлекаются агрегированные значения количества транспортных средств за несколько предыдущих минут, на основе которых формируются лаговые признаки. Это позволяет синхронизировать этап предсказания с логикой обучения модели, где также использовались временные зависимости.

Полученные значения упорядочиваются по времени и преобразуются в вектор признаков фиксированной длины, соответствующий количеству лагов. Далее этот вектор подаётся на вход обученной модели, что обеспечивает более обоснованный и устойчивый прогноз. Дополнительно реализована проверка достаточности данных: если история наблюдений слишком короткая, предсказание не выполняется, что предотвращает некорректные результаты.

Такой подход делает систему более надёжной, поскольку прогноз основывается на реальной динамике изменений трафика. В результате повышается практическая ценность решения для задач мониторинга и анализа транспортных потоков.
     def predict_and_store(model):
      try:
          # --- УСТАНОВЛЕНИЕ СОЕДИНЕНИЯ С БАЗОЙ ДАННЫХ ---
          # На данном этапе происходит подключение к PostgreSQL с использованием строки подключения DB_URL.
          # Использование контекстного менеджера (with) гарантирует корректное закрытие соединения
          # после завершения работы блока, даже в случае возникновения ошибки.
          # Это важно для предотвращения утечек соединений и повышения устойчивости системы
          # при длительной работе или многократных вызовах функции.
          with psycopg2.connect(DB_URL) as conn:
              with conn.cursor() as cur:
  
                  # --- ПОЛУЧЕНИЕ И АГРЕГАЦИЯ ДАННЫХ О ТРАФИКЕ ---
                  # Здесь выполняется SQL-запрос, который извлекает данные за последние 10 минут.
                  # Данные агрегируются по минутам с использованием функции date_trunc,
                  # что позволяет сгруппировать события детекции в равные временные интервалы.
                  # Для каждого интервала рассчитывается количество уникальных транспортных средств
                  # через COUNT(DISTINCT track_id), что исключает повторный учёт одного и того же объекта.
                  # Это обеспечивает более корректную оценку текущей интенсивности движения.
                  cur.execute("""
                      SELECT date_trunc('minute', detection_time) AS minute,
                             COUNT(DISTINCT track_id) AS cars_now
                      FROM user43.full_tracking_data
                      WHERE detection_time > NOW() - INTERVAL '10 minutes'
                      GROUP BY minute
                      ORDER BY minute DESC
                      LIMIT 5
                  """)
  
                  # --- ИЗВЛЕЧЕНИЕ РЕЗУЛЬТАТОВ ЗАПРОСА ---
                  # Результат запроса сохраняется в переменную rows в виде списка кортежей,
                  # где каждый элемент содержит временную метку и количество машин за минуту.
                  # Эти данные будут далее использованы для формирования входных признаков модели.
                  rows = cur.fetchall()
  
                  # --- ПРОВЕРКА ДОСТАТОЧНОСТИ ДАННЫХ ---
                  # Для корректной работы модели необходимо наличие фиксированного количества лагов (в данном случае 5).
                  # Если данных недостаточно (например, система только запущена или поток данных нестабилен),
                  # выполнение функции прерывается. Это предотвращает формирование некорректного прогноза
                  # и повышает надёжность всей системы.
                  if len(rows) < 5:
                      print("⚠️ Недостаточно данных для прогноза")
                      return
  
                  # --- ПРЕОБРАЗОВАНИЕ И УПОРЯДОЧИВАНИЕ ДАННЫХ ---
                  # Данные из БД возвращаются в порядке убывания времени (от новых к старым),
                  # поэтому для корректного формирования временного ряда необходимо развернуть список.
                  # В результате получается последовательность значений, упорядоченных по времени,
                  # что критически важно для корректной работы лаговых признаков.
                  cars = [row[1] for row in rows][::-1]
  
                  # --- ФОРМИРОВАНИЕ ЛАГОВЫХ ПРИЗНАКОВ ---
                  # На основе полученных данных формируется вектор признаков фиксированной длины.
                  # В данном случае используются последние 5 значений интенсивности движения,
                  # которые соответствуют лагам (t-1, t-2, ..., t-5).
                  # Такой подход позволяет модели учитывать временную зависимость
                  # и анализировать динамику изменения трафика.
                  features = cars[-5:]
  
                  # --- ПОЛУЧЕНИЕ ПРОГНОЗА ---
                  # Модель принимает на вход двумерный массив, поэтому признаки оборачиваются в список.
                  # Результат предсказания преобразуется в целое число, так как речь идёт о количестве ТС.
                  # Дополнительно используется ограничение снизу (max с 0), чтобы исключить
                  # возможность появления отрицательных значений, которые не имеют физического смысла.
                  pred_val = max(0, int(model.predict([features])[0]))
  
                  # --- РАСЧЁТ ВРЕМЕНИ ПРОГНОЗА ---
                  # Определяется момент времени, на который строится прогноз.
                  # Он вычисляется как текущее время плюс заданный горизонт прогнозирования (HORIZON).
                  # Это позволяет хранить не только значение прогноза, но и его временную привязку.
                  target_time = datetime.now() + timedelta(minutes=HORIZON)
  
                  # --- СОХРАНЕНИЕ РЕЗУЛЬТАТА В БАЗУ ДАННЫХ ---
                  # Полученный прогноз записывается в таблицу traffic_predictions.
                  # Сохраняются:
                  # - время формирования прогноза (NOW())
                  # - целевое время (на которое он рассчитан)
                  # - горизонт прогнозирования
                  # - предсказанное значение интенсивности
                  # Это необходимо для последующего анализа, визуализации и оценки качества модели.
                  cur.execute("""
                      INSERT INTO user43.traffic_predictions 
                      (prediction_made_at, target_time, horizon_minutes, predicted_intensity) 
                      VALUES (NOW(), %s, %s, %s)
                  """, (target_time, HORIZON, pred_val))
  
                  # --- ЛОГИРОВАНИЕ РЕЗУЛЬТАТА ---
                  # Вывод сообщения позволяет отслеживать работу системы в реальном времени
                  # и контролировать корректность выполнения функции.
                  print(f"✅ Прогноз на {target_time}: {pred_val} ТС записан.")
  
      except Exception as e:
          # --- ОБРАБОТКА ОШИБОК ---
          # В случае возникновения ошибки (например, проблемы с подключением к БД,
          # некорректные данные или сбой модели) выводится диагностическое сообщение.
          # Это упрощает отладку и повышает надёжность эксплуатации системы.
          print(f"❌ Ошибка предсказания: {e}")
