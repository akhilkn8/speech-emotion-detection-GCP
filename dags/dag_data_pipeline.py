from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.google.cloud.operators.cloud_run import CloudRunExecuteJobOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "data_pipeline_dag",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
) as dag_generation:

    data_generate = CloudRunExecuteJobOperator(
        task_id='Data-Generation',
        project_id='firm-site-417617',
        region='us-east4',
        job_name='data-gen-img',
        dag=dag_generation,
        deferrable=False,
    )

    data_transform_train = CloudRunExecuteJobOperator(
        task_id='Data-Transformation-Train',
        project_id='firm-site-417617',
        region='us-east4',
        job_name='data-trans-img',
        overrides={'container_overrides': [{'env':[{'name':'STAGE', 'value':'train'}]}]},
        dag=dag_generation,
        deferrable=False,
    )

    data_transform_test = CloudRunExecuteJobOperator(
        task_id='Data-Transformation-Test',
        project_id='firm-site-417617',
        region='us-east4',
        job_name='data-trans-img',
        overrides={'container_overrides': [{'env':[{'name':'STAGE', 'value':'test'}]}]},
        dag=dag_generation,
        deferrable=False,
    )

    trigger_training_dag = TriggerDagRunOperator(
        task_id="Trigger-Model-Training",
        trigger_dag_id="model_training_dag",
    )

    data_generate >> data_transform_train >> data_transform_test >> trigger_training_dag