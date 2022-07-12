FROM python:3.8-slim

RUN pip install mlflow==1.27.0 && \   
    pip install psycopg2-binary==2.9.3 && \
    pip install boto3==1.24.26

# alternative backend: sqlite:///mlflow.db
#ENTRYPOINT ["sh", "-c", "mlflow server -h 0.0.0.0 --backend-store-uri $BACKEND --default-artifact-root $BUCKET_PATH"]

# to run as proxy mode
ENTRYPOINT ["sh", "-c", "mlflow server -h 0.0.0.0 --backend-store-uri $BACKEND --artifacts-destination $BUCKET_PATH --serve-artifacts"]
