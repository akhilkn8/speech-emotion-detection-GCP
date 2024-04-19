cd model_training

docker buildx build -t model_train_img --platform linux/amd64 .

docker tag model_train_img us-east4-docker.pkg.dev/firm-site-417617/model-training/model_train_img:staging

docker push us-east4-docker.pkg.dev/firm-site-417617/model-training/model_train_img:staging