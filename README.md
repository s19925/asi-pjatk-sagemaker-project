# asi-pjatk-sagemaker-project

## Sources
Dataset
[Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)


```
pip install -r src/requirements.txt
```

## Overview

Kedro project integreted with AWS SageMaker

## Prerequisites
- AWS CLI installed
- AWS SageMaker domain
- SageMaker Execution role ARN
- S3 Bucket
- Docker installed
- Wandb Account
- AWS ECR

## How to install dependencies

```
pip install -r src/requirements.txt
```

## How to run your Kedro pipeline

You can run your Kedro project with:

```
kedro run
```

## How to vizualize pipelines

```
kedro viz
```

## How to migrate project to AWS SageMaker
Create Docker container in ECR

```
kedro sagemaker init "{S3 Bucket Name}" "{arn role}" "{ECR URI}"
```

Run pipelines in AWS SageMaker

```
kedro sagemaker run --auto-build -y
```




