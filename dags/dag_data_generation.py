from datetime import datetime, timedelta
import os
from airflow import DAG
from airflow.providers.google.cloud.operators.cloud_run import CloudRunExecuteJobOperator
from dotenv import load_dotenv


load_dotenv()

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "data_generation_dag",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
) as dag_generation:

    execute1 = CloudRunExecuteJobOperator(
        task_id='Data-Generation',
        project_id=os.getenv('GOOGLE_CLOUD_PROJECT'),
        region=os.getenv('CLOUD_RUN_LOCATION'),
        job_name=os.getenv('DATA_GEN_JOB_NAME'),
        dag=dag_generation,
        deferrable=False,
    )