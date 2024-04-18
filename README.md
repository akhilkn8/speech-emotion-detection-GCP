# speech-emotion-detection-mlops


# Build and Push imagea to GCR
## Data Generation
```bash
docker buildx build -t data_gen_img --platform linux/amd64 .
docker tag data_gen_img us-east4-docker.pkg.dev/firm-site-417617/data-generation/data_gen_img:staging
docker push us-east4-docker.pkg.dev/firm-site-417617/data-generation/data_gen_img:staging
```

### Run a job on Gcloud
```bash
gcloud run jobs describe data-gen-img --format export > job.yaml
gcloud beta run jobs replace job.yaml  
gcloud beta run jobs execute data-gen-img
```

## Data Transformation
```bash
docker buildx build -t data_trans_img --platform linux/amd64 .
docker tag data_trans_img us-east4-docker.pkg.dev/firm-site-417617/data-transformation/data_trans_img:staging
docker push us-east4-docker.pkg.dev/firm-site-417617/data-transformation/data_trans_img:staging
```

## Model Training

```bash
gcloud artifacts repositories create model-training --repository-format=docker \
--location=us-east1 --description="Docker repository for model training"
```

```bash
docker buildx build -t model_training_img --platform linux/amd64 .
docker tag model_training_img us-east1-docker.pkg.dev/firm-site-417617/model-training/model_training_img:staging
docker push us-east1-docker.pkg.dev/firm-site-417617/model-training/model_training_img:staging
```

TODO:
1. train, test, val splits to be done
2. transform dag: try to pass stage as params in container_overides: https://airflow.apache.org/docs/apache-airflow-providers-google/stable/operators/cloud/cloud_run.html
3. training: using Vertex AI operator : https://airflow.apache.org/docs/apache-airflow-providers-google/stable/operators/cloud/vertex_ai.html#creating-an-endpoint-service

