cd data_transformation

docker buildx build -t data_trans_img --platform linux/amd64 .

docker tag data_trans_img us-east4-docker.pkg.dev/firm-site-417617/data-transformation/data_trans_img:staging

docker push us-east4-docker.pkg.dev/firm-site-417617/data-transformation/data_trans_img:staging