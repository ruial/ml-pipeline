# ML Pipeline

A local [Metaflow](https://metaflow.org) and [MLflow](https://mlflow.org) Docker environment to experiment with machine learning pipelines.

A text classifier (Kaggle disaster tweets) training workflow and prediction API are provided as an example.

## Setup

```sh
# install poetry for package management and make sure it is added to your path
pip install poetry
export PATH="$HOME/.local/bin:$PATH"
# after cloning the repo
poetry install

# some examples of poetry commands
# poetry new ml-pipeline
# poetry shell
# poetry add metaflow scikit-learn pandas
# poetry add -D jupyter
# poetry show -v
# poetry run pytest
```

## Running

```sh
cp ./docker/metaflow-config.json ~/.metaflowconfig/config.json 
export MLFLOW_TRACKING_URI=http://localhost:5000
export AWS_ACCESS_KEY_ID=admin
export AWS_SECRET_ACCESS_KEY=s3secret
docker-compose up -d
# Training pipeline
python ml_pipeline/trainflow.py run --split_size=0.2
# API
uvicorn ml_pipeline.api:app
```

Then you can navigate to:

- Metaflow: http://localhost:4000
- MLflow: http://localhost:5000
- MinIO: http://localhost:9101

Some screenshots available in the [images folder](images/).

## Improvements

This is a small project to demonstrate some MLOps concepts. Other suggestions for a production workload include:

- Better code/project structure
- Pre-commit hooks for linting
- Github actions for testing and deployment
- Scheduling in AWS or Kubernetes
- Model monitoring with Prometheus

I recommend this [course](https://github.com/DataTalksClub/mlops-zoomcamp) if you are interested in the topic.
