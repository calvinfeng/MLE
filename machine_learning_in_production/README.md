# Machine Learning in Production

Majority of the notes and exercises are carried out via Jupyter notebooks. I export them as markdown
files and display them here.

## Lesson 1 Introduction to Deployment

* [Introduction to Deployment](introduction_to_deployment.md)

## Lesson 2 Building a Model Using SageMaker

XGBoost will be used a lot here, because our data siz is relatively small, not suitable for any
deep learning method. Also, XGBoost is computationally friendly, it will save me money and time.

### Boston House Market XGBoost Batch Transform

Batch transform is essentially a batch job that AWS offers to run predictions on a batch of data.
This is not meant to be used as a real time on-demand service.

* [Batch Transform High Level API](boston_housing_xgboost_batch_transform_high_level_api.md)
* [Batch Transform Low Level API](boston_housing_xgboost_batch_transform_low_level_api.md)

### IMDB Movie Review Sentimental Analysis

This is a mini-project on natural language processing using XGBoost.

* [IMDB Sentimental Analysis](imdb_sentiment_analysis_xgboost_batch_transform.md)

## Lesson 3 Deploying and Using a Model

Instead of batch transform, now we are using actual deployment of a model. The high level API
looks pretty nice but it hides all the endpoints and request objects.

```python
container = get_image_uri(session.boto_region_name, 'xgboost', '0.90-1')

xgb = sagemaker.estimator.Estimator(container,
                                    role,
                                    train_instance_count=1,
                                    train_instance_type='ml.m4.xlarge',
                                    output_path='s3://{}/{}/output'.format(session.default_bucket(), prefix),
                                    sagemaker_session=session)

xgb.set_hyperparameters(max_depth=5,
                        eta=0.2,
                        gamma=4,
                        min_child_weight=6,
                        subsample=0.8,
                        objective='reg:linear',
                        early_stopping_rounds=10,
                        num_round=500)

xgb_predictor = xgb.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')
```

### Boston House Market XGBoost Deployment

* [Deployment High Level API](boston_housing_xgboost_deploy_high_level_api.md)
* [Deployment Low Level API](boston_housing_xgboost_deploy_low_level_api.md)

### IMDB Movie Review Sentimental Analysis Web Application

* [Web Application](imdb_sentiment_analysis_xgboost_web_app.md)

## Lesson 4 Updating a Model

### Boston House Market Model Update

This is an example of how to use two different models on one endpoint. The two models are load
balanced by AWS. Users may specify load distribution using weight on each model.

* [Update an Endpoint](boston_housing_updating_an_endpoint.md)
