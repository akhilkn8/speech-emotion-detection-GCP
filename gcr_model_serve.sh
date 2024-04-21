cd model_serving
 
docker buildx build -t model_serve_img --platform linux/amd64 .
 
docker tag model_serve_img us-east1-docker.pkg.dev/firm-site-417617/model-serving/model_serve_img:staging
 
docker push us-east1-docker.pkg.dev/firm-site-417617/model-serving/model_serve_img:staging