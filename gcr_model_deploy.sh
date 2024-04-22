cd model_deployment
 
docker buildx build -t model_deploy_img --platform linux/amd64 .
 
docker tag model_deploy_img us-east1-docker.pkg.dev/firm-site-417617/model-deployment/model_deploy_img:staging
 
docker push us-east1-docker.pkg.dev/firm-site-417617/model-deployment/model_deploy_img:staging