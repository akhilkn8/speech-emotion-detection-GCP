cd data_generation

docker buildx build -t data_gen_img --platform linux/amd64 .

docker tag data_gen_img us-east1-docker.pkg.dev/firm-site-417617/data-generation/data_gen_img:staging

docker push us-east1-docker.pkg.dev/firm-site-417617/data-generation/data_gen_img:staging