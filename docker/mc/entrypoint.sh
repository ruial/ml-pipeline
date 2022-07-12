#!/bin/sh

while [[ "$(curl -s -o /dev/null -w ''%{http_code}'' $MINIO_HOST/minio/health/live)" != "200" ]];
do
  echo "Waiting for minio..."
  sleep 1
done

mc alias set minio $MINIO_HOST $MINIO_ROOT_USER $MINIO_ROOT_PASSWORD

mc mb minio/$BUCKET || true

mc cp --recursive /datasets minio/$BUCKET
