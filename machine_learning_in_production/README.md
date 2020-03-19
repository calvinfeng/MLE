# Machine Learning in Production

Majority of the notes and exercises are carried out via Jupyter notebooks. I export them as markdown
files and display them here.

## Lesson 1 Introduction to Deployment

> Deployment to production can simply be thought of as a method that integrates a machine learning
> model into an existing production environment so that the model can be used to make decisions or
> predictions based on upon data input into this model.

There are 3 primary steps to a Machine Learning Workflow. 

1. Explore & Process Data
2. Modeling
3. Deployment

### Explore & Process Data

I'd simply call this the ETL (Extract, Transform, and Load) step of the pipeline. Data need be
ingested from multiple sources. They need to be transformed and cleansed. The details of this
step will be explored in the future sections.

### Modeling

![Machine Learning Workflow Modeling](./ml_workflow_modeling.png)

#### Hyperparameter

In machine learning, a hyperparamter is a parameter whose value cannot be estimated from the data.
Specifically, a hyperparameter is _not directly learned_ through the estimators. For example, in
my old machine learning projects, I had to setup a grid search to find the best learning rate,
regularization rate, alpha or beta for Adams optimizer and etc... Often cloud platform machine
learning services do provide tools that allow for automatic hyperparameter tuning.

### Deployment

![Machine Learning Workflow Deployment](./ml_workflow_deployment.png)

#### Model Versioning

Besides saving the model version as a part of model's metadata in a database, the deployment
platform should allow one to indicate a deployed model's version. This is similar to a release
manager. It would be a model versioning control manager.

#### Model Monitoring

Once a model is deployed, you want to watch its performance and compare it to older versions. The
performance monitor is a crucial tool for deployment.

#### Model Updating and Routing

If a deployed model is failing to meet its performance metrics, you need to update the model or
revert it back to an older version. If there's a fundamental change in the data that's being input
into the model for predictions, you want to collect this input data to be used for updating the
model.

The deployment platform should support routing differing proportions of requests to the deployed
models; to allow comparison of perfromance between the deloyed model variants.

#### Model Predictions

There are two common types of predictions, **on-demand** and **batch**.

On-demand predictions are commonly used to provide users with real-time, online responses based
upon a deployed model. For example, making a prediction on estimated delivery time for DoorDash
drivers and customers.

Batch predictions are commonly used to help make business decisions. This computation would run on
a daily or weekly basis.

## Lesson 2 Building a Model Using SageMaker

XGBoost will be used a lot here, because our data siz is relatively small, not suitable for any
deep learning method. Also, XGBoost is computationally friendly, it will save me money and time.

### Boston House Market Prediction

Batch transform is essentially a batch job that AWS offers to run predictions on a batch of data.
This is not meant to be used as a real time on-demand service.

* [High Level API](./boston_housing_xgboost_batch_transform_high_level_api.md)
* [Low Level API](./boston_housing_xgboost_batch_transform_low_level_api.md)

### Mini Project: IMDB Sentimental Analysis

* [IMDB Sentimental Analysis](./imdb_sentiment_analysis_xgboost_batch_transform.md)

## Lesson 3 Deploying and Using a Model

