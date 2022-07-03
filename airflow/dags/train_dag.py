from datetime import datetime, timedelta

from airflow import DAG

from airflow.operators.python import PythonOperator
from train import train_model

with DAG(
    'movie-recommendation-train',
    # These args will get passed on to each operator
    # You can override them on a per-task basis during operator initialization
    default_args={
        'depends_on_past': False,
        'email_on_failure': False,
        'email_on_retry': False,
        'retries': 1,
        'retry_delay': timedelta(minutes=5)
    },
    description='Recommendation training',
    schedule_interval=timedelta(days=1), #you can change this cadence. 
    start_date=datetime(2021, 1, 1),
    catchup=False,
    tags=['staging'],
) as dag:
    
    PythonOperator(
        task_id='training_model',
        python_callable=train_model,
    )

