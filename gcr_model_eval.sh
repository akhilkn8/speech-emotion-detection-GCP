cd model_evaluation

docker buildx build -t model_eval_img --platform linux/amd64 .

docker tag model_eval_img us-east4-docker.pkg.dev/firm-site-417617/model-evaluation/model_eval_img:staging

docker push us-east4-docker.pkg.dev/firm-site-417617/model-evaluation/model_eval_img:staging