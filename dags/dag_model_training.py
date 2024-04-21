from datetime import datetime, timedelta
from typing import Optional
import google.cloud.aiplatform as aiplatform
from airflow import DAG
from airflow.providers.google.cloud.operators.cloud_run import CloudRunExecuteJobOperator
from airflow.providers.google.cloud.operators.vertex_ai.custom_job import (
    CreateCustomContainerTrainingJobOperator,
    DeleteCustomTrainingJobOperator,
)


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "model_training_dag",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
) as dag_training:

    create_custom_container_training_job = CreateCustomContainerTrainingJobOperator(
        task_id="speech-model-training",
        staging_bucket=f"gs://artifacts-speech-emotion/model_training",
        display_name='model_train_img',
        container_uri='us-east1-docker.pkg.dev/firm-site-417617/model-training/model_train_img@sha256:e01127c1877665887164ca656dc25c11dcddf1ddb2f3026e1b523d01b20fa224',
        # model_serving_container_image_uri=MODEL_SERVING_CONTAINER_URI,
        # run params
        region="us-east1",
        project_id='firm-site-417617',
    )