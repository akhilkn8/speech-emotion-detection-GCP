from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.google.cloud.operators.cloud_run import CloudRunExecuteJobOperator


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
        project_id='firm-site-417617',
        region='us-east4',
        job_name='data-gen-img',
        dag=dag_generation,
        deferrable=False,
    )