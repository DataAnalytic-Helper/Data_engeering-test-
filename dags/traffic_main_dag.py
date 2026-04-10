from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import os

# Автоматически определяем корень проекта относительно папки DAGs
# Или просто укажите базовый путь один раз здесь:
# Вместо C:/Users/... пишем путь, который понимает Linux (через /mnt/c/)
PROJECT_ROOT = '/mnt/c/Users/muzhi/OneDrive/Документы/РАБОТЫ_КОЛЛЕДЖ/Чемпионат/traffic_final'
# Путь к питону внутри Ubuntu (создадим его на след. шаге)
PYTHON_EXE = '/usr/bin/python3'
SCRIPTS_DIR = f'{PROJECT_ROOT}/airflow_home/scripts'

default_args = {
    'owner': 'user',
    'depends_on_past': False,
    'start_date': datetime(2026, 4, 5),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'transport_system_pipeline',
    default_args=default_args,
    description='Оркестрация детекции и прогнозирования',
    schedule_interval='*/1 * * * *', # Раз в минуту
    catchup=False
) as dag:

    # Задача №1: Предиктор
    # Используем кавычки для путей, если в них есть пробелы или кириллица
    predict_task = BashOperator(
        task_id='predict_traffic_density',
        bash_command=f'"{PYTHON_EXE}" "{SCRIPTS_DIR}/predictor.py"'
    )

    # Задача №2: Очистка
    cleanup_task = BashOperator(
        task_id='cleanup_raw_data',
        bash_command='psql $DB_URL -c "DELETE FROM user43.full_tracking_data WHERE detection_time < NOW() - INTERVAL \'100 days\';"'
    )
    # Задача №0: Детектор (собирает свежие данные 10-20 секунд)
    detect_task = BashOperator(
        task_id='run_traffic_detector',
        bash_command=f'"{PYTHON_EXE}" "{SCRIPTS_DIR}/detector.py"'
    )

    # Выстраиваем цепочку: Сначала детектор -> потом прогноз -> потом очистка
    detect_task >> predict_task >> cleanup_task
