from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'start_date': datetime(2026, 4, 5),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG('transport_orchestration', default_args=default_args, schedule_interval='*/10 * * * *') as dag:
    # Задача 1: Запуск предиктора (раз в 10 минут)
    run_prediction = BashOperator(
        task_id='predict_traffic',
        bash_command='python3 /path/to/your/scripts/predictor.py'
    )

    # Задача 2: Очистка старых данных (опционально, для порядка в БД)
    clean_old_raw = BashOperator(
        task_id='db_cleanup',
        bash_command='psql $DB_URL -c "DELETE FROM full_tracking_data WHERE detection_time < NOW() - INTERVAL \'1 day\';"'
    )

    run_prediction >> clean_old_raw