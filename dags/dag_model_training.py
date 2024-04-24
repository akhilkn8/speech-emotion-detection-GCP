from datetime import datetime, timedelta
import os
from typing import Optional
import google.cloud.aiplatform as aiplatform
from airflow import DAG
from airflow.providers.google.cloud.operators.cloud_run import CloudRunExecuteJobOperator
from airflow.providers.google.cloud.operators.vertex_ai.custom_job import (
    CreateCustomContainerTrainingJobOperator,
    DeleteCustomTrainingJobOperator,
)
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
    "model_training_dag",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
) as dag_training:

    custom_container_training_job = CreateCustomContainerTrainingJobOperator(
        task_id="speech-model-training",
        staging_bucket=os.getenv('GCP_STAGING_BUCKET'),
        display_name='Model-Training',
        container_uri=os.getenv('VERTEXAI_TRAIN_CONTAINER_URI'),
        # model_serving_container_image_uri=MODEL_SERVING_CONTAINER_URI,
        # run params
        region=os.getenv('VERTEXAI_JOB_LOCATION'),
        project_id=os.getenv('GOOGLE_CLOUD_PROJECT'),
    )

    custom_container_evaluation_job = CreateCustomContainerTrainingJobOperator(
        task_id="speech-model-evaluation",
        staging_bucket=os.getenv('GCP_STAGING_BUCKET'),
        display_name='Model-Evaluation',
        container_uri=os.getenv('VERTEXAI_EVAL_CONTAINER_URI'),
        # model_serving_container_image_uri=MODEL_SERVING_CONTAINER_URI,
        # run params
        region=os.getenv('VERTEXAI_JOB_LOCATION'),
        project_id=os.getenv('GOOGLE_CLOUD_PROJECT'),
    )

    custom_container_training_job >> custom_container_evaluation_job


    