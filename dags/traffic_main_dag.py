from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# 1. Базовые настройки
default_args = {
    'owner': 'user43',
    'depends_on_past': False,
    'start_date': datetime(2026, 4, 5), # Число старта чемпионата
    'retries': 1,                       # Если скрипт упал, Airflow попробует еще раз
    'retry_delay': timedelta(minutes=5),
}

# 2. Описание самого процесса
with DAG(
    'transport_system_pipeline',
    default_args=default_args,
    description='Оркестрация детекции и прогнозирования',
    schedule_interval='*/10 * * * *',   # Запускать каждые 10 минут
    catchup=False
) as dag:

    # Задача №1: Запуск предиктора
    # Мы просто говорим системе выполнить команду в терминале
    predict_task = BashOperator(
        task_id='predict_traffic_density',
        bash_command='python3 /airflow_home/scripts/predictor.py'
    )

    # Задача №2: Очистка логов или старых данных (опционально для баллов)
    # Показывает экспертам, что ты управляешь жизненным циклом данных
    cleanup_task = BashOperator(
        task_id='cleanup_raw_data',
        bash_command='psql $DB_URL -c "DELETE FROM user43.full_tracking_data WHERE detection_time < NOW() - INTERVAL \'2 days\';"'
    )

    # Очередность: сначала считаем прогноз, потом чистим старье
    predict_task >> cleanup_task