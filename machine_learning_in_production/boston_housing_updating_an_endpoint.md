# Predicting Boston Housing Prices

## Updating a model using SageMaker

In this notebook, we will continue working with the [Boston HousingDataset](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html).
Our goal in this notebook will be to train two different models and to use SageMaker to switch a
deployed endpoint from using one model to the other. One of the benefits of using SageMaker to do
this is that we can make the change without interrupting service. What this means is that we can
continue sending data to the endpoint and at no point will that endpoint disappear.

## General Outline

Typically, when using a notebook instance with SageMaker, you will proceed through the following
steps. Of course, not every step will need to be done with each project. Also, there is quite a lot
of room for variation in many of the steps, as you will see throughout these lessons.

1. Download or otherwise retrieve the data.
2. Process / Prepare the data.
3. Upload the processed data to S3.
4. Train a chosen model.
5. Test the trained model (typically using a batch transform job).
6. Deploy the trained model.
7. Use the deployed model.

In this notebook we will be skipping step 5, testing the model. In addition, we will perform steps
4, 6 and 7 multiple times with different models.

## Step 0: Setting up the notebook

We begin by setting up all of the necessary bits required to run our notebook. To start that means
loading all of the Python modules we will need.

```python
%matplotlib inline

import os

import numpy as np
import pandas as pd

from pprint import pprint
import matplotlib.pyplot as plt
from time import gmtime, strftime

from sklearn.datasets import load_boston
import sklearn.model_selection
```

In addition to the modules above, we need to import the various bits of SageMaker that we will be using.

```python
import sagemaker
from sagemaker import get_execution_role
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.predictor import csv_serializer

# This is an object that represents the SageMaker session that we are currently operating in. This
# object contains some useful information that we will need to access later such as our region.
session = sagemaker.Session()

# This is an object that represents the IAM role that we are currently assigned. When we construct
# and launch the training job later we will need to tell it what IAM role it should have. Since our
# use case is relatively simple we will simply assign the training job the role we currently have.
role = get_execution_role()
```

## Step 1: Downloading the data

Fortunately, this dataset can be retrieved using sklearn and so this step is relatively straightforward.

```python
boston = load_boston()
```

## Step 2: Preparing and splitting the data

Given that this is clean tabular data, we don't need to do any processing. However, we do need to
split the rows in the dataset up into train, test and validation sets.

```python
# First we package up the input data and the target variable (the median value) as pandas dataframes. This
# will make saving the data to a file a little easier later on.

X_bos_pd = pd.DataFrame(boston.data, columns=boston.feature_names)
Y_bos_pd = pd.DataFrame(boston.target)

# We split the dataset into 2/3 training and 1/3 testing sets.
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X_bos_pd, Y_bos_pd, test_size=0.33)

# Then we split the training set further into 2/3 training and 1/3 validation sets.
X_train, X_val, Y_train, Y_val = sklearn.model_selection.train_test_split(X_train, Y_train, test_size=0.33)
```

## Step 3: Uploading the training and validation files to S3

When a training job is constructed using SageMaker, a container is executed which performs the
training operation. This container is given access to data that is stored in S3. This means that we
need to upload the data we want to use for training to S3. We can use the SageMaker API to do this
and hide some of the details.

### Save the data locally

First we need to create the train and validation csv files which we will then upload to S3.

```python
# This is our local data directory. We need to make sure that it exists.
data_dir = '../data/boston'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
```

```python
# We use pandas to save our train and validation data to csv files. Note that we make sure not to include header
# information or an index as this is required by the built in algorithms provided by Amazon. Also, it is assumed
# that the first entry in each row is the target variable.

pd.concat([Y_val, X_val], axis=1).to_csv(os.path.join(data_dir, 'validation.csv'), header=False, index=False)
pd.concat([Y_train, X_train], axis=1).to_csv(os.path.join(data_dir, 'train.csv'), header=False, index=False)
```

### Upload to S3

Since we are currently running inside of a SageMaker session, we can use the object which represents
this session to upload our data to the 'default' S3 bucket. Note that it is good practice to provide
a custom prefix (essentially an S3 folder) to make sure that you don't accidentally interfere with
data uploaded from some other notebook or project.

```python
prefix = 'boston-update-endpoints'

val_location = session.upload_data(os.path.join(data_dir, 'validation.csv'), key_prefix=prefix)
train_location = session.upload_data(os.path.join(data_dir, 'train.csv'), key_prefix=prefix)
```

## Step 4 (A): Train the XGBoost model

Now that we have the training and validation data uploaded to S3, we can construct our XGBoost model
and train it. We will be making use of the high level SageMaker API to do this which will make the
resulting code a little easier to read at the cost of some flexibility.

To construct an estimator, the object which we wish to train, we need to provide the location of a
container which contains the training code. Since we are using a built in algorithm this container
is provided by Amazon. However, the full name of the container is a bit lengthy and depends on the
region that we are operating in. Fortunately, SageMaker provides a useful utility method called
`get_image_uri` that constructs the image name for us.

To use the `get_image_uri` method we need to provide it with our current region, which can be
obtained from the session object, and the name of the algorithm we wish to use. In this notebook we
will be using XGBoost however you could try another algorithm if you wish. The list of built in
algorithms can be found in the list of [Common Parameters](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-algo-docker-registry-paths.html).

```python
# As stated above, we use this utility method to construct the image name for the training container.
xgb_container = get_image_uri(session.boto_region_name, 'xgboost', '0.90-1')

# Now that we know which container to use, we can construct the estimator object.
xgb = sagemaker.estimator.Estimator(xgb_container, # The name of the training container
                                    role,      # The IAM role to use (our current role in this case)
                                    train_instance_count=1, # The number of instances to use for training
                                    train_instance_type='ml.m4.xlarge', # The type of instance ot use for training
                                    output_path='s3://{}/{}/output'.format(session.default_bucket(), prefix),
                                                                        # Where to save the output (the model artifacts)
                                    sagemaker_session=session) # The current SageMaker session
```

Before asking SageMaker to begin the training job, we should probably set any model specific
hyperparameters. There are quite a few that can be set when using the XGBoost algorithm, below are
just a few of them. If you would like to change the hyperparameters below or modify additional ones
you can find additional information on the [XGBoost hyperparameter page](https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost_hyperparameters.html)

```python
xgb.set_hyperparameters(max_depth=5,
                        eta=0.2,
                        gamma=4,
                        min_child_weight=6,
                        subsample=0.8,
                        objective='reg:linear',
                        early_stopping_rounds=10,
                        num_round=200)
```

Now that we have our estimator object completely set up, it is time to train it. To do this we make
sure that SageMaker knows our input data is in csv format and then execute the `fit` method.

```python
# This is a wrapper around the location of our train and validation data, to make sure that SageMaker
# knows our data is in csv format.
s3_input_train = sagemaker.s3_input(s3_data=train_location, content_type='text/csv')
s3_input_validation = sagemaker.s3_input(s3_data=val_location, content_type='text/csv')

xgb.fit({'train': s3_input_train, 'validation': s3_input_validation})
```

```text
2020-03-22 20:01:04 Starting - Starting the training job...
2020-03-22 20:01:05 Starting - Launching requested ML instances...
2020-03-22 20:02:03 Starting - Preparing the instances for training......
2020-03-22 20:02:55 Downloading - Downloading input data...
2020-03-22 20:03:12 Training - Downloading the training image...
2020-03-22 20:03:56 Uploading - Uploading generated training model
2020-03-22 20:03:56 Completed - Training job completed
[34mINFO:sagemaker-containers:Imported framework sagemaker_xgboost_container.training[0m
[34mINFO:sagemaker-containers:Failed to parse hyperparameter objective value reg:linear to Json.[0m
[34mReturning the value itself[0m
[34mINFO:sagemaker-containers:No GPUs detected (normal if no gpus installed)[0m
[34mINFO:sagemaker_xgboost_container.training:Running XGBoost Sagemaker in algorithm mode[0m
[34mINFO:root:Determined delimiter of CSV input is ','[0m
[34mINFO:root:Determined delimiter of CSV input is ','[0m
[34mINFO:root:Determined delimiter of CSV input is ','[0m
[34m[20:03:46] 227x13 matrix with 2951 entries loaded from /opt/ml/input/data/train?format=csv&label_column=0&delimiter=,[0m
[34mINFO:root:Determined delimiter of CSV input is ','[0m
[34m[20:03:46] 112x13 matrix with 1456 entries loaded from /opt/ml/input/data/validation?format=csv&label_column=0&delimiter=,[0m
[34mINFO:root:Single node training.[0m
[34mINFO:root:Train matrix has 227 rows[0m
[34mINFO:root:Validation matrix has 112 rows[0m
[34m[20:03:46] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.[0m
[34m[0]#011train-rmse:18.7885#011validation-rmse:19.9369[0m
[34m[1]#011train-rmse:15.3861#011validation-rmse:16.6421[0m
[34m[2]#011train-rmse:12.7352#011validation-rmse:14.0599[0m
[34m[3]#011train-rmse:10.5695#011validation-rmse:11.8145[0m
[34m[4]#011train-rmse:8.8922#011validation-rmse:9.97116[0m
[34m[5]#011train-rmse:7.47445#011validation-rmse:8.52243[0m
[34m[6]#011train-rmse:6.35632#011validation-rmse:7.33662[0m
[34m[7]#011train-rmse:5.46761#011validation-rmse:6.4287[0m
[34m[8]#011train-rmse:4.79437#011validation-rmse:5.87523[0m
[34m[9]#011train-rmse:4.26802#011validation-rmse:5.27846[0m
[34m[10]#011train-rmse:3.87672#011validation-rmse:4.85555[0m
[34m[11]#011train-rmse:3.51002#011validation-rmse:4.54563[0m
[34m[12]#011train-rmse:3.22833#011validation-rmse:4.29897[0m
[34m[13]#011train-rmse:2.98659#011validation-rmse:4.06215[0m
[34m[14]#011train-rmse:2.74177#011validation-rmse:3.86221[0m
[34m[15]#011train-rmse:2.63335#011validation-rmse:3.77268[0m
[34m[16]#011train-rmse:2.53372#011validation-rmse:3.66919[0m
[34m[17]#011train-rmse:2.45825#011validation-rmse:3.63939[0m
[34m[18]#011train-rmse:2.38801#011validation-rmse:3.58045[0m
[34m[19]#011train-rmse:2.27832#011validation-rmse:3.50522[0m
[34m[20]#011train-rmse:2.19861#011validation-rmse:3.42862[0m
[34m[21]#011train-rmse:2.16046#011validation-rmse:3.40565[0m
[34m[22]#011train-rmse:2.09238#011validation-rmse:3.33703[0m
[34m[23]#011train-rmse:2.01778#011validation-rmse:3.30309[0m
[34m[24]#011train-rmse:1.99811#011validation-rmse:3.27207[0m
[34m[25]#011train-rmse:1.96687#011validation-rmse:3.2541[0m
[34m[26]#011train-rmse:1.89985#011validation-rmse:3.22794[0m
[34m[27]#011train-rmse:1.84902#011validation-rmse:3.22927[0m
[34m[28]#011train-rmse:1.76353#011validation-rmse:3.20553[0m
[34m[29]#011train-rmse:1.74001#011validation-rmse:3.18953[0m
[34m[30]#011train-rmse:1.69221#011validation-rmse:3.20908[0m
[34m[31]#011train-rmse:1.68776#011validation-rmse:3.19367[0m
[34m[32]#011train-rmse:1.6708#011validation-rmse:3.17418[0m
[34m[33]#011train-rmse:1.60134#011validation-rmse:3.15354[0m
[34m[34]#011train-rmse:1.55418#011validation-rmse:3.15062[0m
[34m[35]#011train-rmse:1.50699#011validation-rmse:3.12787[0m
[34m[36]#011train-rmse:1.45333#011validation-rmse:3.12888[0m
[34m[37]#011train-rmse:1.43869#011validation-rmse:3.12173[0m
[34m[38]#011train-rmse:1.4111#011validation-rmse:3.12407[0m
[34m[39]#011train-rmse:1.37853#011validation-rmse:3.16334[0m
[34m[40]#011train-rmse:1.33181#011validation-rmse:3.15474[0m
[34m[41]#011train-rmse:1.29799#011validation-rmse:3.13952[0m
[34m[42]#011train-rmse:1.28302#011validation-rmse:3.13033[0m
[34m[43]#011train-rmse:1.26264#011validation-rmse:3.13282[0m
[34m[44]#011train-rmse:1.22017#011validation-rmse:3.1258[0m
[34m[45]#011train-rmse:1.19069#011validation-rmse:3.12111[0m
[34m[46]#011train-rmse:1.18482#011validation-rmse:3.12008[0m
[34m[47]#011train-rmse:1.16051#011validation-rmse:3.12396[0m
[34m[48]#011train-rmse:1.14374#011validation-rmse:3.11859[0m
[34m[49]#011train-rmse:1.10153#011validation-rmse:3.12521[0m
[34m[50]#011train-rmse:1.09553#011validation-rmse:3.1377[0m
[34m[51]#011train-rmse:1.06733#011validation-rmse:3.14551[0m
[34m[52]#011train-rmse:1.04708#011validation-rmse:3.14401[0m
[34m[53]#011train-rmse:1.03895#011validation-rmse:3.15869[0m
[34m[54]#011train-rmse:1.02913#011validation-rmse:3.16086[0m
[34m[55]#011train-rmse:1.01808#011validation-rmse:3.14715[0m
[34m[56]#011train-rmse:1.00496#011validation-rmse:3.15294[0m
[34m[57]#011train-rmse:1.00153#011validation-rmse:3.15424[0m
[34m[58]#011train-rmse:0.996098#011validation-rmse:3.15584[0m
Training seconds: 61
Billable seconds: 61
```

## Step 5: Test the trained model

We will be skipping this step for now.

## Step 6 (A): Deploy the trained model

Even though we used the high level approach to construct and train the XGBoost model, we will be
using the lower level approach to deploy it. One of the reasons for this is so that we have
additional control over how the endpoint is constructed. This will be a little more clear later on
when construct more advanced endpoints.

### Build the model

Of course, before we can deploy the model, we need to first create it. The `fit` method that we used
earlier created some model artifacts and we can use these to construct a model object.

```python
# Remember that a model needs to have a unique name
xgb_model_name = "boston-update-xgboost-model" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

# We also need to tell SageMaker which container should be used for inference and where it should
# retrieve the model artifacts from. In our case, the xgboost container that we used for training
# can also be used for inference and the model artifacts come from the previous call to fit.
xgb_primary_container = {
    "Image": xgb_container,
    "ModelDataUrl": xgb.model_data
}

# And lastly we construct the SageMaker model
xgb_model_info = session.sagemaker_client.create_model(
                                ModelName = xgb_model_name,
                                ExecutionRoleArn = role,
                                PrimaryContainer = xgb_primary_container)
```

```python
pprint(xgb_model_info)
```

```json
{
    'ModelArn': 'arn:aws:sagemaker:us-west-2:171758673694:model/boston-update-xgboost-model2020-03-22-20-06-20',
    'ResponseMetadata': {
        'HTTPHeaders': {
            'content-length': '108',
            'content-type': 'application/x-amz-json-1.1',
            'date': 'Sun, 22 Mar 2020 20:06:20 GMT',
            'x-amzn-requestid': '672f1705-2cd9-43db-9757-41ed54feefec'
        },
        'HTTPStatusCode': 200,
        'RequestId': '672f1705-2cd9-43db-9757-41ed54feefec',
        'RetryAttempts': 0
    }
}
```

### Create the endpoint configuration

Once we have a model we can start putting together the endpoint. Recall that to do this we need to
first create an endpoint configuration, essentially the blueprint that SageMaker will use to build
the endpoint itself.

```python
# As before, we need to give our endpoint configuration a name which should be unique
xgb_endpoint_config_name = "boston-update-xgboost-endpoint-config-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

# And then we ask SageMaker to construct the endpoint configuration
xgb_endpoint_config_info = session.sagemaker_client.create_endpoint_config(
                            EndpointConfigName = xgb_endpoint_config_name,
                            ProductionVariants = [{
                                "InstanceType": "ml.m4.xlarge",
                                "InitialVariantWeight": 1,
                                "InitialInstanceCount": 1,
                                "ModelName": xgb_model_name,
                                "VariantName": "XGB-Model"
                            }])
```

### Deploy the endpoint

Now that the endpoint configuration has been created, we can ask SageMaker to build our endpoint.

**Note:** This is a friendly (repeated) reminder that you are about to deploy an endpoint. Make sure
that you shut it down once you've finished with it!

```python
# Again, we need a unique name for our endpoint
endpoint_name = "boston-update-endpoint-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

# And then we can deploy our endpoint
endpoint_info = session.sagemaker_client.create_endpoint(
                    EndpointName = endpoint_name,
                    EndpointConfigName = xgb_endpoint_config_name)
```

```python
endpoint_dec = session.wait_for_endpoint(endpoint_name)
```

## Step 7 (A): Use the model

Now that our model is trained and deployed we can send some test data to it and evaluate the results.

```python
response = session.sagemaker_runtime_client.invoke_endpoint(
                                                EndpointName = endpoint_name,
                                                ContentType = 'text/csv',
                                                Body = ','.join(map(str, X_test.values[0])))
```

```python
pprint(response)
```

```json
{
    'Body': <botocore.response.StreamingBody object at 0x7fdadc1153c8>,
    'ContentType': 'text/csv; charset=utf-8',
    'InvokedProductionVariant': 'XGB-Model',
    'ResponseMetadata': {
        'HTTPHeaders': {
            'content-length': '18',
            'content-type': 'text/csv; charset=utf-8',
            'date': 'Sun, 22 Mar 2020 20:37:15 GMT',
            'x-amzn-invoked-production-variant': 'XGB-Model',
            'x-amzn-requestid': '0ae2fffa-4765-43a8-801b-38c8adc1e23f'
        },
        'HTTPStatusCode': 200,
        'RequestId': '0ae2fffa-4765-43a8-801b-38c8adc1e23f',
        'RetryAttempts': 0
    }
}
```

```python
result = response['Body'].read().decode("utf-8")
```

```python
pprint(result)
```

```text
'10.521326065063477'
```

```python
Y_test.values[0]
```

```text
array([12.3])
```

## Shut down the endpoint

Now that we know that the XGBoost endpoint works, we can shut it down. We will make use of it again
later.

```python
session.sagemaker_client.delete_endpoint(EndpointName = endpoint_name)
```

```json
{
    'ResponseMetadata': {
        'RequestId': 'b06c5c22-4e1a-4804-99a1-bb11695ec552',
        'HTTPStatusCode': 200,
        'HTTPHeaders': {
            'x-amzn-requestid': 'b06c5c22-4e1a-4804-99a1-bb11695ec552',
            'content-type': 'application/x-amz-json-1.1',
            'content-length': '0',
            'date': 'Sun, 22 Mar 2020 20:38:40 GMT'
        },
        'RetryAttempts': 0
    }
}
```

## Step 4 (B): Train the Linear model

Suppose we are working in an environment where the XGBoost model that we trained earlier is becoming
too costly. Perhaps the number of calls to our endpoint has increased and the length of time it
takes to perform inference with the XGBoost model is becoming problematic.

A possible solution might be to train a simpler model to see if it performs nearly as well. In our
case, we will construct a linear model. The process of doing this is the same as for creating the
XGBoost model that we created earlier, although there are different hyperparameters that we need to
set.

```python
# Similar to the XGBoost model, we will use the utility method to construct the image name for the training container.
linear_container = get_image_uri(session.boto_region_name, 'linear-learner')

# Now that we know which container to use, we can construct the estimator object.
linear = sagemaker.estimator.Estimator(linear_container, # The name of the training container
                                        role,      # The IAM role to use (our current role in this case)
                                        train_instance_count=1, # The number of instances to use for training
                                        train_instance_type='ml.m4.xlarge', # The type of instance ot use for training
                                        output_path='s3://{}/{}/output'.format(session.default_bucket(), prefix),
                                                                            # Where to save the output (the model artifacts)
                                        sagemaker_session=session) # The current SageMaker session
```

Before asking SageMaker to train our model, we need to set some hyperparameters. In this case we
will be using a linear model so the number of hyperparameters we need to set is much fewer. For more
details see the [Linear model hyperparameter page](https://docs.aws.amazon.com/sagemaker/latest/dg/ll_hyperparameters.html)

```python
linear.set_hyperparameters(feature_dim=13, # Our data has 13 feature columns
                           predictor_type='regressor', # We wish to create a regression model
                           mini_batch_size=200) # Here we set how many samples to look at in each iteration
```

Now that the hyperparameters have been set, we can ask SageMaker to fit the linear model to our data.

```python
linear.fit({'train': s3_input_train, 'validation': s3_input_validation})
```

```text
2020-03-22 20:40:27 Starting - Starting the training job...
2020-03-22 20:40:28 Starting - Launching requested ML instances......
2020-03-22 20:41:30 Starting - Preparing the instances for training...
2020-03-22 20:42:27 Downloading - Downloading input data...
2020-03-22 20:42:58 Training - Downloading the training image...
2020-03-22 20:43:24 Uploading - Uploading generated training model[34mDocker entrypoint called with argument(s): train[0m
[34mRunning default environment configuration script[0m
[34m[03/22/2020 20:43:21 INFO 140403405219648] Reading default configuration from /opt/amazon/lib/python2.7/site-packages/algorithm/resources/default-input.json: {u'loss_insensitivity': u'0.01', u'epochs': u'15', u'feature_dim': u'auto', u'init_bias': u'0.0', u'lr_scheduler_factor': u'auto', u'num_calibration_samples': u'10000000', u'accuracy_top_k': u'3', u'_num_kv_servers': u'auto', u'use_bias': u'true', u'num_point_for_scaler': u'10000', u'_log_level': u'info', u'quantile': u'0.5', u'bias_lr_mult': u'auto', u'lr_scheduler_step': u'auto', u'init_method': u'uniform', u'init_sigma': u'0.01', u'lr_scheduler_minimum_lr': u'auto', u'target_recall': u'0.8', u'num_models': u'auto', u'early_stopping_patience': u'3', u'momentum': u'auto', u'unbias_label': u'auto', u'wd': u'auto', u'optimizer': u'auto', u'_tuning_objective_metric': u'', u'early_stopping_tolerance': u'0.001', u'learning_rate': u'auto', u'_kvstore': u'auto', u'normalize_data': u'true', u'binary_classifier_model_selection_criteria': u'accuracy', u'use_lr_scheduler': u'true', u'target_precision': u'0.8', u'unbias_data': u'auto', u'init_scale': u'0.07', u'bias_wd_mult': u'auto', u'f_beta': u'1.0', u'mini_batch_size': u'1000', u'huber_delta': u'1.0', u'num_classes': u'1', u'beta_1': u'auto', u'loss': u'auto', u'beta_2': u'auto', u'_enable_profiler': u'false', u'normalize_label': u'auto', u'_num_gpus': u'auto', u'balance_multiclass_weights': u'false', u'positive_example_weight_mult': u'1.0', u'l1': u'auto', u'margin': u'1.0'}[0m
[34m[03/22/2020 20:43:21 INFO 140403405219648] Reading provided configuration from /opt/ml/input/config/hyperparameters.json: {u'feature_dim': u'13', u'mini_batch_size': u'200', u'predictor_type': u'regressor'}[0m
[34m[03/22/2020 20:43:21 INFO 140403405219648] Final configuration: {u'loss_insensitivity': u'0.01', u'epochs': u'15', u'feature_dim': u'13', u'init_bias': u'0.0', u'lr_scheduler_factor': u'auto', u'num_calibration_samples': u'10000000', u'accuracy_top_k': u'3', u'_num_kv_servers': u'auto', u'use_bias': u'true', u'num_point_for_scaler': u'10000', u'_log_level': u'info', u'quantile': u'0.5', u'bias_lr_mult': u'auto', u'lr_scheduler_step': u'auto', u'init_method': u'uniform', u'init_sigma': u'0.01', u'lr_scheduler_minimum_lr': u'auto', u'target_recall': u'0.8', u'num_models': u'auto', u'early_stopping_patience': u'3', u'momentum': u'auto', u'unbias_label': u'auto', u'wd': u'auto', u'optimizer': u'auto', u'_tuning_objective_metric': u'', u'early_stopping_tolerance': u'0.001', u'learning_rate': u'auto', u'_kvstore': u'auto', u'normalize_data': u'true', u'binary_classifier_model_selection_criteria': u'accuracy', u'use_lr_scheduler': u'true', u'target_precision': u'0.8', u'unbias_data': u'auto', u'init_scale': u'0.07', u'bias_wd_mult': u'auto', u'f_beta': u'1.0', u'mini_batch_size': u'200', u'huber_delta': u'1.0', u'num_classes': u'1', u'predictor_type': u'regressor', u'beta_1': u'auto', u'loss': u'auto', u'beta_2': u'auto', u'_enable_profiler': u'false', u'normalize_label': u'auto', u'_num_gpus': u'auto', u'balance_multiclass_weights': u'false', u'positive_example_weight_mult': u'1.0', u'l1': u'auto', u'margin': u'1.0'}[0m
[34m[03/22/2020 20:43:21 WARNING 140403405219648] Loggers have already been setup.[0m
[34mProcess 1 is a worker.[0m
[34m[03/22/2020 20:43:21 INFO 140403405219648] Using default worker.[0m
[34m[03/22/2020 20:43:21 INFO 140403405219648] Create Store: local[0m
[34m[03/22/2020 20:43:21 INFO 140403405219648] Scaler algorithm parameters
    <algorithm.scaler.ScalerAlgorithmStable object at 0x7fb1e56edad0>[0m
[34m[03/22/2020 20:43:21 INFO 140403405219648] Scaling model computed with parameters:
    {'stdev_weight': [0m
[34m[9.3013363e+00 2.3006151e+01 6.6273022e+00 2.5514701e-01 1.1866623e-01
    6.7023408e-01 2.7099649e+01 2.0237646e+00 8.7710190e+00 1.6902739e+02
    2.1691098e+00 9.1123917e+01 7.1170626e+00][0m
[34m<NDArray 13 @cpu(0)>, 'stdev_label': [0m
[34m[8.358023][0m
[34m<NDArray 1 @cpu(0)>, 'mean_label': [0m
[34m[22.1625][0m
[34m<NDArray 1 @cpu(0)>, 'mean_weight': [0m
[34m[3.9689796e+00 1.0942500e+01 1.1581600e+01 7.0000000e-02 5.6570452e-01
    6.2404351e+00 6.9730003e+01 3.6385424e+00 1.0065000e+01 4.1854501e+02
    1.8508001e+01 3.5708319e+02 1.2576350e+01][0m
[34m<NDArray 13 @cpu(0)>}[0m
[34m[03/22/2020 20:43:21 INFO 140403405219648] nvidia-smi took: 0.0251889228821 secs to identify 0 gpus[0m
[34m[03/22/2020 20:43:21 INFO 140403405219648] Number of GPUs being used: 0[0m
[34m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 2, "sum": 2.0, "min": 2}, "Number of Batches Since Last Reset": {"count": 1, "max": 0, "sum": 0.0, "min": 0}, "Number of Records Since Last Reset": {"count": 1, "max": 0, "sum": 0.0, "min": 0}, "Total Batches Seen": {"count": 1, "max": 3, "sum": 3.0, "min": 3}, "Total Records Seen": {"count": 1, "max": 427, "sum": 427.0, "min": 427}, "Max Records Seen Between Resets": {"count": 1, "max": 227, "sum": 227.0, "min": 227}, "Reset Count": {"count": 1, "max": 2, "sum": 2.0, "min": 2}}, "EndTime": 1584909801.532301, "Dimensions": {"Host": "algo-1", "Meta": "init_train_data_iter", "Operation": "training", "Algorithm": "Linear Learner"}, "StartTime": 1584909801.532263}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9685323333740234, "sum": 0.9685323333740234, "min": 0.9685323333740234}}, "EndTime": 1584909801.571871, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.571812}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.8676612854003907, "sum": 0.8676612854003907, "min": 0.8676612854003907}}, "EndTime": 1584909801.571941, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.571927}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.325056915283203, "sum": 1.325056915283203, "min": 1.325056915283203}}, "EndTime": 1584909801.572006, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.571988}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.8580335998535156, "sum": 0.8580335998535156, "min": 0.8580335998535156}}, "EndTime": 1584909801.572116, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.572095}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.8695442962646485, "sum": 0.8695442962646485, "min": 0.8695442962646485}}, "EndTime": 1584909801.572182, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.572164}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.029219970703125, "sum": 1.029219970703125, "min": 1.029219970703125}}, "EndTime": 1584909801.57224, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.572224}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.115345458984375, "sum": 1.115345458984375, "min": 1.115345458984375}}, "EndTime": 1584909801.572302, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.572285}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.0999972534179687, "sum": 1.0999972534179687, "min": 1.0999972534179687}}, "EndTime": 1584909801.572371, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.572354}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9157221221923828, "sum": 0.9157221221923828, "min": 0.9157221221923828}}, "EndTime": 1584909801.572433, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.572422}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.4091554260253907, "sum": 1.4091554260253907, "min": 1.4091554260253907}}, "EndTime": 1584909801.572499, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.572482}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.8436393737792969, "sum": 0.8436393737792969, "min": 0.8436393737792969}}, "EndTime": 1584909801.572564, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.572547}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.093177261352539, "sum": 1.093177261352539, "min": 1.093177261352539}}, "EndTime": 1584909801.572629, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.572612}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.0018565368652343, "sum": 1.0018565368652343, "min": 1.0018565368652343}}, "EndTime": 1584909801.572694, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.572676}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.0144656372070313, "sum": 1.0144656372070313, "min": 1.0144656372070313}}, "EndTime": 1584909801.572763, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.572745}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.904182357788086, "sum": 0.904182357788086, "min": 0.904182357788086}}, "EndTime": 1584909801.572834, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.572816}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.045905227661133, "sum": 1.045905227661133, "min": 1.045905227661133}}, "EndTime": 1584909801.572908, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.57289}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.0723854827880859, "sum": 1.0723854827880859, "min": 1.0723854827880859}}, "EndTime": 1584909801.572978, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.57296}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.85283203125, "sum": 0.85283203125, "min": 0.85283203125}}, "EndTime": 1584909801.57305, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.573033}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.220689163208008, "sum": 1.220689163208008, "min": 1.220689163208008}}, "EndTime": 1584909801.573118, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.573101}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.0416190338134765, "sum": 1.0416190338134765, "min": 1.0416190338134765}}, "EndTime": 1584909801.573176, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.573162}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.8005082702636719, "sum": 0.8005082702636719, "min": 0.8005082702636719}}, "EndTime": 1584909801.573211, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.573203}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.1561756134033203, "sum": 1.1561756134033203, "min": 1.1561756134033203}}, "EndTime": 1584909801.573257, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.573242}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.7841773986816406, "sum": 0.7841773986816406, "min": 0.7841773986816406}}, "EndTime": 1584909801.573324, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.573307}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.8775112152099609, "sum": 0.8775112152099609, "min": 0.8775112152099609}}, "EndTime": 1584909801.573389, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.573371}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9612258911132813, "sum": 0.9612258911132813, "min": 0.9612258911132813}}, "EndTime": 1584909801.57345, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.573433}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9279397583007812, "sum": 0.9279397583007812, "min": 0.9279397583007812}}, "EndTime": 1584909801.57351, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.573493}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.2203185272216797, "sum": 1.2203185272216797, "min": 1.2203185272216797}}, "EndTime": 1584909801.573578, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.57356}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.1165598297119141, "sum": 1.1165598297119141, "min": 1.1165598297119141}}, "EndTime": 1584909801.573637, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.57362}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9937887573242188, "sum": 0.9937887573242188, "min": 0.9937887573242188}}, "EndTime": 1584909801.573704, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.573686}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9522576904296876, "sum": 0.9522576904296876, "min": 0.9522576904296876}}, "EndTime": 1584909801.573766, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.573749}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.1092294311523438, "sum": 1.1092294311523438, "min": 1.1092294311523438}}, "EndTime": 1584909801.573829, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.573811}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.1170328521728516, "sum": 1.1170328521728516, "min": 1.1170328521728516}}, "EndTime": 1584909801.573892, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.573874}
[0m
[34m[03/22/2020 20:43:21 INFO 140403405219648] #quality_metric: host=algo-1, epoch=0, train mse_objective <loss>=0.968532333374[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 88.19022042410714, "sum": 88.19022042410714, "min": 88.19022042410714}}, "EndTime": 1584909801.625215, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.625159}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 82.92397635323661, "sum": 82.92397635323661, "min": 82.92397635323661}}, "EndTime": 1584909801.625281, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.625267}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 117.10751778738839, "sum": 117.10751778738839, "min": 117.10751778738839}}, "EndTime": 1584909801.625342, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.625324}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 81.39910016741071, "sum": 81.39910016741071, "min": 81.39910016741071}}, "EndTime": 1584909801.625418, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.625398}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 59.16987827845982, "sum": 59.16987827845982, "min": 59.16987827845982}}, "EndTime": 1584909801.625484, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.625467}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 42.03749738420759, "sum": 42.03749738420759, "min": 42.03749738420759}}, "EndTime": 1584909801.625547, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.625532}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 54.70406232561384, "sum": 54.70406232561384, "min": 54.70406232561384}}, "EndTime": 1584909801.625588, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.625579}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 55.892381940569194, "sum": 55.892381940569194, "min": 55.892381940569194}}, "EndTime": 1584909801.625619, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.625611}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 86.16769845145089, "sum": 86.16769845145089, "min": 86.16769845145089}}, "EndTime": 1584909801.625668, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.625653}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 124.09327043805804, "sum": 124.09327043805804, "min": 124.09327043805804}}, "EndTime": 1584909801.625721, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.625708}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 81.56352887834821, "sum": 81.56352887834821, "min": 81.56352887834821}}, "EndTime": 1584909801.625778, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.625762}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 100.97731236049107, "sum": 100.97731236049107, "min": 100.97731236049107}}, "EndTime": 1584909801.625839, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.625823}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 55.56473650251116, "sum": 55.56473650251116, "min": 55.56473650251116}}, "EndTime": 1584909801.625899, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.625882}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 58.04821341378348, "sum": 58.04821341378348, "min": 58.04821341378348}}, "EndTime": 1584909801.625972, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.625954}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 63.436614990234375, "sum": 63.436614990234375, "min": 63.436614990234375}}, "EndTime": 1584909801.626045, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.626027}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 46.32187761579241, "sum": 46.32187761579241, "min": 46.32187761579241}}, "EndTime": 1584909801.626118, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.6261}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 97.45406668526786, "sum": 97.45406668526786, "min": 97.45406668526786}}, "EndTime": 1584909801.626181, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.626165}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 79.06388636997768, "sum": 79.06388636997768, "min": 79.06388636997768}}, "EndTime": 1584909801.626239, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.626224}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 114.63752092633929, "sum": 114.63752092633929, "min": 114.63752092633929}}, "EndTime": 1584909801.626306, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.626289}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 96.06624058314732, "sum": 96.06624058314732, "min": 96.06624058314732}}, "EndTime": 1584909801.626373, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.626355}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 60.65343802315848, "sum": 60.65343802315848, "min": 60.65343802315848}}, "EndTime": 1584909801.626437, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.62642}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 52.237191336495535, "sum": 52.237191336495535, "min": 52.237191336495535}}, "EndTime": 1584909801.626504, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.626487}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 50.366101946149556, "sum": 50.366101946149556, "min": 50.366101946149556}}, "EndTime": 1584909801.626569, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.626552}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 53.85080827985491, "sum": 53.85080827985491, "min": 53.85080827985491}}, "EndTime": 1584909801.626632, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.626615}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 93.67633056640625, "sum": 93.67633056640625, "min": 93.67633056640625}}, "EndTime": 1584909801.626701, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.626684}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 92.00265066964286, "sum": 92.00265066964286, "min": 92.00265066964286}}, "EndTime": 1584909801.626766, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.626748}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 112.89729527064732, "sum": 112.89729527064732, "min": 112.89729527064732}}, "EndTime": 1584909801.626839, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.62682}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 106.45497349330357, "sum": 106.45497349330357, "min": 106.45497349330357}}, "EndTime": 1584909801.626909, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.626892}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 100.33894566127232, "sum": 100.33894566127232, "min": 100.33894566127232}}, "EndTime": 1584909801.626979, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.626961}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 116.514892578125, "sum": 116.514892578125, "min": 116.514892578125}}, "EndTime": 1584909801.627045, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.627027}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 86.37225341796875, "sum": 86.37225341796875, "min": 86.37225341796875}}, "EndTime": 1584909801.627109, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.627092}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 86.88808768136161, "sum": 86.88808768136161, "min": 86.88808768136161}}, "EndTime": 1584909801.627172, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.627155}
[0m
[34m[03/22/2020 20:43:21 INFO 140403405219648] #quality_metric: host=algo-1, epoch=0, validation mse_objective <loss>=88.1902204241[0m
[34m[03/22/2020 20:43:21 INFO 140403405219648] #early_stopping_criteria_metric: host=algo-1, epoch=0, criteria=mse_objective, value=42.0374973842[0m
[34m[03/22/2020 20:43:21 INFO 140403405219648] Epoch 0: Loss improved. Updating best model[0m
[34m[03/22/2020 20:43:21 INFO 140403405219648] #progress_metric: host=algo-1, completed 6 % of epochs[0m
[34m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 2, "sum": 2.0, "min": 2}, "Number of Batches Since Last Reset": {"count": 1, "max": 2, "sum": 2.0, "min": 2}, "Number of Records Since Last Reset": {"count": 1, "max": 227, "sum": 227.0, "min": 227}, "Total Batches Seen": {"count": 1, "max": 5, "sum": 5.0, "min": 5}, "Total Records Seen": {"count": 1, "max": 654, "sum": 654.0, "min": 654}, "Max Records Seen Between Resets": {"count": 1, "max": 227, "sum": 227.0, "min": 227}, "Reset Count": {"count": 1, "max": 3, "sum": 3.0, "min": 3}}, "EndTime": 1584909801.628746, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 0}, "StartTime": 1584909801.532484}
[0m
[34m[03/22/2020 20:43:21 INFO 140403405219648] #throughput_metric: host=algo-1, train throughput=2355.5810308 records/second[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9173393249511719, "sum": 0.9173393249511719, "min": 0.9173393249511719}}, "EndTime": 1584909801.648946, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.648894}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.8245072174072265, "sum": 0.8245072174072265, "min": 0.8245072174072265}}, "EndTime": 1584909801.649009, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.648996}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.254846420288086, "sum": 1.254846420288086, "min": 1.254846420288086}}, "EndTime": 1584909801.649078, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.649059}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.8139730834960938, "sum": 0.8139730834960938, "min": 0.8139730834960938}}, "EndTime": 1584909801.649155, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.649135}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.7503659057617188, "sum": 0.7503659057617188, "min": 0.7503659057617188}}, "EndTime": 1584909801.649218, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.649202}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5242005157470703, "sum": 0.5242005157470703, "min": 0.5242005157470703}}, "EndTime": 1584909801.649278, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.649261}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6300767135620117, "sum": 0.6300767135620117, "min": 0.6300767135620117}}, "EndTime": 1584909801.649348, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.649331}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6428482055664062, "sum": 0.6428482055664062, "min": 0.6428482055664062}}, "EndTime": 1584909801.649417, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.649399}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.8722917175292969, "sum": 0.8722917175292969, "min": 0.8722917175292969}}, "EndTime": 1584909801.649483, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.649465}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.335128631591797, "sum": 1.335128631591797, "min": 1.335128631591797}}, "EndTime": 1584909801.649592, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.649571}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.8035623931884766, "sum": 0.8035623931884766, "min": 0.8035623931884766}}, "EndTime": 1584909801.64966, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.649642}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.0378865814208984, "sum": 1.0378865814208984, "min": 1.0378865814208984}}, "EndTime": 1584909801.649716, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.649699}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6713401794433593, "sum": 0.6713401794433593, "min": 0.6713401794433593}}, "EndTime": 1584909801.649779, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.649762}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6946302795410156, "sum": 0.6946302795410156, "min": 0.6946302795410156}}, "EndTime": 1584909801.649841, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.649824}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.8144718170166015, "sum": 0.8144718170166015, "min": 0.8144718170166015}}, "EndTime": 1584909801.6499, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.649884}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5740978622436523, "sum": 0.5740978622436523, "min": 0.5740978622436523}}, "EndTime": 1584909801.649967, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.64995}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.0177031707763673, "sum": 1.0177031707763673, "min": 1.0177031707763673}}, "EndTime": 1584909801.650025, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.650013}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.8149205780029297, "sum": 0.8149205780029297, "min": 0.8149205780029297}}, "EndTime": 1584909801.650086, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.650069}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.1618863677978515, "sum": 1.1618863677978515, "min": 1.1618863677978515}}, "EndTime": 1584909801.650147, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.65013}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9895668029785156, "sum": 0.9895668029785156, "min": 0.9895668029785156}}, "EndTime": 1584909801.650215, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.650198}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.8161609649658204, "sum": 0.8161609649658204, "min": 0.8161609649658204}}, "EndTime": 1584909801.650271, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.650255}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6041407012939453, "sum": 0.6041407012939453, "min": 0.6041407012939453}}, "EndTime": 1584909801.650331, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.650315}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.67644775390625, "sum": 0.67644775390625, "min": 0.67644775390625}}, "EndTime": 1584909801.650392, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.650374}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.684411392211914, "sum": 0.684411392211914, "min": 0.684411392211914}}, "EndTime": 1584909801.650452, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.650435}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9652757263183593, "sum": 0.9652757263183593, "min": 0.9652757263183593}}, "EndTime": 1584909801.65051, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.650494}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9407695770263672, "sum": 0.9407695770263672, "min": 0.9407695770263672}}, "EndTime": 1584909801.65057, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.650553}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.1776319885253905, "sum": 1.1776319885253905, "min": 1.1776319885253905}}, "EndTime": 1584909801.650635, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.650618}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.092220230102539, "sum": 1.092220230102539, "min": 1.092220230102539}}, "EndTime": 1584909801.6507, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.650683}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.0625088500976563, "sum": 1.0625088500976563, "min": 1.0625088500976563}}, "EndTime": 1584909801.650759, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.650742}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.2742840576171874, "sum": 1.2742840576171874, "min": 1.2742840576171874}}, "EndTime": 1584909801.650816, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.6508}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.8710610961914063, "sum": 0.8710610961914063, "min": 0.8710610961914063}}, "EndTime": 1584909801.650876, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.650859}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.8781427764892578, "sum": 0.8781427764892578, "min": 0.8781427764892578}}, "EndTime": 1584909801.650934, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.650917}
[0m
[34m[03/22/2020 20:43:21 INFO 140403405219648] #quality_metric: host=algo-1, epoch=1, train mse_objective <loss>=0.917339324951[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 83.68223353794643, "sum": 83.68223353794643, "min": 83.68223353794643}}, "EndTime": 1584909801.692382, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.692325}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 78.95027378627232, "sum": 78.95027378627232, "min": 78.95027378627232}}, "EndTime": 1584909801.692458, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.692438}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 111.04874965122768, "sum": 111.04874965122768, "min": 111.04874965122768}}, "EndTime": 1584909801.692527, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.692509}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 77.34737723214286, "sum": 77.34737723214286, "min": 77.34737723214286}}, "EndTime": 1584909801.692582, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.692565}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 79.02900041852679, "sum": 79.02900041852679, "min": 79.02900041852679}}, "EndTime": 1584909801.69264, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.692624}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 85.34110804966518, "sum": 85.34110804966518, "min": 85.34110804966518}}, "EndTime": 1584909801.6927, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.692684}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 96.77479771205357, "sum": 96.77479771205357, "min": 96.77479771205357}}, "EndTime": 1584909801.692766, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.692749}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 87.52188546316964, "sum": 87.52188546316964, "min": 87.52188546316964}}, "EndTime": 1584909801.692828, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.692811}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 82.23844691685268, "sum": 82.23844691685268, "min": 82.23844691685268}}, "EndTime": 1584909801.692888, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.692872}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 117.69803292410714, "sum": 117.69803292410714, "min": 117.69803292410714}}, "EndTime": 1584909801.692947, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.69293}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 77.91930280412946, "sum": 77.91930280412946, "min": 77.91930280412946}}, "EndTime": 1584909801.693006, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.69299}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 96.20928083147321, "sum": 96.20928083147321, "min": 96.20928083147321}}, "EndTime": 1584909801.693066, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.693054}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 89.22236851283482, "sum": 89.22236851283482, "min": 89.22236851283482}}, "EndTime": 1584909801.693119, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.693103}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 87.61398751395089, "sum": 87.61398751395089, "min": 87.61398751395089}}, "EndTime": 1584909801.693181, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.693165}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 76.618896484375, "sum": 76.618896484375, "min": 76.618896484375}}, "EndTime": 1584909801.693242, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.693225}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 95.12767682756696, "sum": 95.12767682756696, "min": 95.12767682756696}}, "EndTime": 1584909801.693315, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.693296}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 92.61039515904018, "sum": 92.61039515904018, "min": 92.61039515904018}}, "EndTime": 1584909801.693382, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.693364}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 75.74444580078125, "sum": 75.74444580078125, "min": 75.74444580078125}}, "EndTime": 1584909801.693454, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.693436}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 109.47392054966518, "sum": 109.47392054966518, "min": 109.47392054966518}}, "EndTime": 1584909801.693517, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.6935}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 91.43586077008929, "sum": 91.43586077008929, "min": 91.43586077008929}}, "EndTime": 1584909801.693578, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.693562}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 77.36906215122768, "sum": 77.36906215122768, "min": 77.36906215122768}}, "EndTime": 1584909801.693644, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.693625}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 82.25602504185268, "sum": 82.25602504185268, "min": 82.25602504185268}}, "EndTime": 1584909801.693705, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.693687}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 72.86502947126117, "sum": 72.86502947126117, "min": 72.86502947126117}}, "EndTime": 1584909801.693767, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.69375}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 67.73606218610492, "sum": 67.73606218610492, "min": 67.73606218610492}}, "EndTime": 1584909801.693837, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.693819}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 93.57673863002232, "sum": 93.57673863002232, "min": 93.57673863002232}}, "EndTime": 1584909801.693899, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.693882}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 92.40035574776786, "sum": 92.40035574776786, "min": 92.40035574776786}}, "EndTime": 1584909801.693961, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.693945}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 110.09360177176339, "sum": 110.09360177176339, "min": 110.09360177176339}}, "EndTime": 1584909801.694034, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.694016}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 105.16437639508929, "sum": 105.16437639508929, "min": 105.16437639508929}}, "EndTime": 1584909801.694098, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.694081}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 109.90206473214286, "sum": 109.90206473214286, "min": 109.90206473214286}}, "EndTime": 1584909801.694144, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.694134}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 106.9984130859375, "sum": 106.9984130859375, "min": 106.9984130859375}}, "EndTime": 1584909801.694197, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.69418}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 103.85379464285714, "sum": 103.85379464285714, "min": 103.85379464285714}}, "EndTime": 1584909801.694265, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.694249}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 99.44205147879464, "sum": 99.44205147879464, "min": 99.44205147879464}}, "EndTime": 1584909801.694331, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.694315}
[0m
[34m[03/22/2020 20:43:21 INFO 140403405219648] #quality_metric: host=algo-1, epoch=1, validation mse_objective <loss>=83.6822335379[0m
[34m[03/22/2020 20:43:21 INFO 140403405219648] #early_stopping_criteria_metric: host=algo-1, epoch=1, criteria=mse_objective, value=67.7360621861[0m
[34m[03/22/2020 20:43:21 INFO 140403405219648] #progress_metric: host=algo-1, completed 13 % of epochs[0m
[34m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 2, "sum": 2.0, "min": 2}, "Number of Batches Since Last Reset": {"count": 1, "max": 2, "sum": 2.0, "min": 2}, "Number of Records Since Last Reset": {"count": 1, "max": 227, "sum": 227.0, "min": 227}, "Total Batches Seen": {"count": 1, "max": 7, "sum": 7.0, "min": 7}, "Total Records Seen": {"count": 1, "max": 881, "sum": 881.0, "min": 881}, "Max Records Seen Between Resets": {"count": 1, "max": 227, "sum": 227.0, "min": 227}, "Reset Count": {"count": 1, "max": 4, "sum": 4.0, "min": 4}}, "EndTime": 1584909801.695153, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 1}, "StartTime": 1584909801.628992}
[0m
[34m[03/22/2020 20:43:21 INFO 140403405219648] #throughput_metric: host=algo-1, train throughput=3425.1164049 records/second[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.870918960571289, "sum": 0.870918960571289, "min": 0.870918960571289}}, "EndTime": 1584909801.717397, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.71731}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.7862915802001953, "sum": 0.7862915802001953, "min": 0.7862915802001953}}, "EndTime": 1584909801.71749, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.717469}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.1892984771728516, "sum": 1.1892984771728516, "min": 1.1892984771728516}}, "EndTime": 1584909801.717562, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.717542}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.7746347045898437, "sum": 0.7746347045898437, "min": 0.7746347045898437}}, "EndTime": 1584909801.717629, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.71761}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.023774642944336, "sum": 1.023774642944336, "min": 1.023774642944336}}, "EndTime": 1584909801.717695, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.717676}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.1580888366699218, "sum": 1.1580888366699218, "min": 1.1580888366699218}}, "EndTime": 1584909801.71776, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.717742}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.2425887298583984, "sum": 1.2425887298583984, "min": 1.2425887298583984}}, "EndTime": 1584909801.717834, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.717814}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.1144878387451171, "sum": 1.1144878387451171, "min": 1.1144878387451171}}, "EndTime": 1584909801.717903, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.717885}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.8330324554443359, "sum": 0.8330324554443359, "min": 0.8330324554443359}}, "EndTime": 1584909801.717969, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.717952}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.265610809326172, "sum": 1.265610809326172, "min": 1.265610809326172}}, "EndTime": 1584909801.718043, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.718025}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.7678083038330078, "sum": 0.7678083038330078, "min": 0.7678083038330078}}, "EndTime": 1584909801.718107, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.718089}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9872628021240234, "sum": 0.9872628021240234, "min": 0.9872628021240234}}, "EndTime": 1584909801.718165, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.718148}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.1648683166503906, "sum": 1.1648683166503906, "min": 1.1648683166503906}}, "EndTime": 1584909801.718229, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.718212}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.1315537261962891, "sum": 1.1315537261962891, "min": 1.1315537261962891}}, "EndTime": 1584909801.718293, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.718275}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9827345275878906, "sum": 0.9827345275878906, "min": 0.9827345275878906}}, "EndTime": 1584909801.718355, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.718338}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.2628499603271484, "sum": 1.2628499603271484, "min": 1.2628499603271484}}, "EndTime": 1584909801.71842, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.718403}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9673733520507812, "sum": 0.9673733520507812, "min": 0.9673733520507812}}, "EndTime": 1584909801.71848, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.718464}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.7815806579589843, "sum": 0.7815806579589843, "min": 0.7815806579589843}}, "EndTime": 1584909801.718544, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.718527}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.1077627563476562, "sum": 1.1077627563476562, "min": 1.1077627563476562}}, "EndTime": 1584909801.718605, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.718589}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9424713134765625, "sum": 0.9424713134765625, "min": 0.9424713134765625}}, "EndTime": 1584909801.718662, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.718645}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.979830093383789, "sum": 0.979830093383789, "min": 0.979830093383789}}, "EndTime": 1584909801.718725, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.718707}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.0476649475097657, "sum": 1.0476649475097657, "min": 1.0476649475097657}}, "EndTime": 1584909801.718787, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.71877}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9587369537353516, "sum": 0.9587369537353516, "min": 0.9587369537353516}}, "EndTime": 1584909801.718849, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.718833}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.8700780487060547, "sum": 0.8700780487060547, "min": 0.8700780487060547}}, "EndTime": 1584909801.71891, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.718893}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9596721649169921, "sum": 0.9596721649169921, "min": 0.9596721649169921}}, "EndTime": 1584909801.718971, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.718955}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9524058532714844, "sum": 0.9524058532714844, "min": 0.9524058532714844}}, "EndTime": 1584909801.719029, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.719014}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.1393058013916015, "sum": 1.1393058013916015, "min": 1.1393058013916015}}, "EndTime": 1584909801.719086, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.71907}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.0831605529785155, "sum": 1.0831605529785155, "min": 1.0831605529785155}}, "EndTime": 1584909801.71915, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.719133}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.2891990661621093, "sum": 1.2891990661621093, "min": 1.2891990661621093}}, "EndTime": 1584909801.71921, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.719194}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.2829238891601562, "sum": 1.2829238891601562, "min": 1.2829238891601562}}, "EndTime": 1584909801.719272, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.719256}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.2095419311523437, "sum": 1.2095419311523437, "min": 1.2095419311523437}}, "EndTime": 1584909801.719333, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.719317}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.1823320770263672, "sum": 1.1823320770263672, "min": 1.1823320770263672}}, "EndTime": 1584909801.719394, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.719378}
[0m
[34m[03/22/2020 20:43:21 INFO 140403405219648] #quality_metric: host=algo-1, epoch=2, train mse_objective <loss>=0.870918960571[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 79.71471296037946, "sum": 79.71471296037946, "min": 79.71471296037946}}, "EndTime": 1584909801.768873, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.768812}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 75.4705810546875, "sum": 75.4705810546875, "min": 75.4705810546875}}, "EndTime": 1584909801.768946, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.768932}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 105.90387834821429, "sum": 105.90387834821429, "min": 105.90387834821429}}, "EndTime": 1584909801.769005, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.768988}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 73.81806291852679, "sum": 73.81806291852679, "min": 73.81806291852679}}, "EndTime": 1584909801.769074, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.769057}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 31.85935538155692, "sum": 31.85935538155692, "min": 31.85935538155692}}, "EndTime": 1584909801.769139, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.769121}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 42.48286655970982, "sum": 42.48286655970982, "min": 42.48286655970982}}, "EndTime": 1584909801.769203, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.769184}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 46.41945539202009, "sum": 46.41945539202009, "min": 46.41945539202009}}, "EndTime": 1584909801.769266, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.769249}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 43.75364467075893, "sum": 43.75364467075893, "min": 43.75364467075893}}, "EndTime": 1584909801.769301, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.769293}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 78.941650390625, "sum": 78.941650390625, "min": 78.941650390625}}, "EndTime": 1584909801.769347, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.769332}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 112.36612374441964, "sum": 112.36612374441964, "min": 112.36612374441964}}, "EndTime": 1584909801.769415, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.769397}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 74.74310302734375, "sum": 74.74310302734375, "min": 74.74310302734375}}, "EndTime": 1584909801.769485, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.769468}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 92.03059605189732, "sum": 92.03059605189732, "min": 92.03059605189732}}, "EndTime": 1584909801.769549, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.769531}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 38.430773053850444, "sum": 38.430773053850444, "min": 38.430773053850444}}, "EndTime": 1584909801.769615, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.769597}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 38.91996111188616, "sum": 38.91996111188616, "min": 38.91996111188616}}, "EndTime": 1584909801.76968, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.769663}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 32.17027282714844, "sum": 32.17027282714844, "min": 32.17027282714844}}, "EndTime": 1584909801.769752, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.769733}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 40.634024483816965, "sum": 40.634024483816965, "min": 40.634024483816965}}, "EndTime": 1584909801.769817, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.769801}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 88.38020542689732, "sum": 88.38020542689732, "min": 88.38020542689732}}, "EndTime": 1584909801.769879, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.769862}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 72.76365007672992, "sum": 72.76365007672992, "min": 72.76365007672992}}, "EndTime": 1584909801.769942, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.769925}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 104.89763532366071, "sum": 104.89763532366071, "min": 104.89763532366071}}, "EndTime": 1584909801.770004, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.769987}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 87.51229422433036, "sum": 87.51229422433036, "min": 87.51229422433036}}, "EndTime": 1584909801.770069, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.770052}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 40.154183523995535, "sum": 40.154183523995535, "min": 40.154183523995535}}, "EndTime": 1584909801.770131, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.770114}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 40.924120221819194, "sum": 40.924120221819194, "min": 40.924120221819194}}, "EndTime": 1584909801.770204, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.770185}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 35.76114109584263, "sum": 35.76114109584263, "min": 35.76114109584263}}, "EndTime": 1584909801.770266, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.77025}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 34.38175746372768, "sum": 34.38175746372768, "min": 34.38175746372768}}, "EndTime": 1584909801.770331, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.770315}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 92.74232700892857, "sum": 92.74232700892857, "min": 92.74232700892857}}, "EndTime": 1584909801.770395, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.770379}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 93.10287039620536, "sum": 93.10287039620536, "min": 93.10287039620536}}, "EndTime": 1584909801.770456, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.770442}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 106.51630510602679, "sum": 106.51630510602679, "min": 106.51630510602679}}, "EndTime": 1584909801.770516, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.770502}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 104.33397565569196, "sum": 104.33397565569196, "min": 104.33397565569196}}, "EndTime": 1584909801.770582, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.770566}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 88.89585658482143, "sum": 88.89585658482143, "min": 88.89585658482143}}, "EndTime": 1584909801.770643, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.770626}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 85.88655308314732, "sum": 85.88655308314732, "min": 85.88655308314732}}, "EndTime": 1584909801.77071, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.770691}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 89.5050048828125, "sum": 89.5050048828125, "min": 89.5050048828125}}, "EndTime": 1584909801.770773, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.770756}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 90.97722516741071, "sum": 90.97722516741071, "min": 90.97722516741071}}, "EndTime": 1584909801.770838, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.770821}
[0m
[34m[03/22/2020 20:43:21 INFO 140403405219648] #quality_metric: host=algo-1, epoch=2, validation mse_objective <loss>=79.7147129604[0m
[34m[03/22/2020 20:43:21 INFO 140403405219648] #early_stopping_criteria_metric: host=algo-1, epoch=2, criteria=mse_objective, value=31.8593553816[0m
[34m[03/22/2020 20:43:21 INFO 140403405219648] Epoch 2: Loss improved. Updating best model[0m
[34m[03/22/2020 20:43:21 INFO 140403405219648] #progress_metric: host=algo-1, completed 20 % of epochs[0m
[34m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 2, "sum": 2.0, "min": 2}, "Number of Batches Since Last Reset": {"count": 1, "max": 2, "sum": 2.0, "min": 2}, "Number of Records Since Last Reset": {"count": 1, "max": 227, "sum": 227.0, "min": 227}, "Total Batches Seen": {"count": 1, "max": 9, "sum": 9.0, "min": 9}, "Total Records Seen": {"count": 1, "max": 1108, "sum": 1108.0, "min": 1108}, "Max Records Seen Between Resets": {"count": 1, "max": 227, "sum": 227.0, "min": 227}, "Reset Count": {"count": 1, "max": 5, "sum": 5.0, "min": 5}}, "EndTime": 1584909801.772801, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 2}, "StartTime": 1584909801.695435}
[0m
[34m[03/22/2020 20:43:21 INFO 140403405219648] #throughput_metric: host=algo-1, train throughput=2929.56002462 records/second[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.8259027099609375, "sum": 0.8259027099609375, "min": 0.8259027099609375}}, "EndTime": 1584909801.800583, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.800513}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.749178237915039, "sum": 0.749178237915039, "min": 0.749178237915039}}, "EndTime": 1584909801.800648, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.800635}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.1253377532958984, "sum": 1.1253377532958984, "min": 1.1253377532958984}}, "EndTime": 1584909801.800711, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.800693}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.736790771484375, "sum": 0.736790771484375, "min": 0.736790771484375}}, "EndTime": 1584909801.800752, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.800742}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.45150840759277344, "sum": 0.45150840759277344, "min": 0.45150840759277344}}, "EndTime": 1584909801.800803, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.800787}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6694000244140625, "sum": 0.6694000244140625, "min": 0.6694000244140625}}, "EndTime": 1584909801.800867, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.800851}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6786742401123047, "sum": 0.6786742401123047, "min": 0.6786742401123047}}, "EndTime": 1584909801.800908, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.800894}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6303299331665039, "sum": 0.6303299331665039, "min": 0.6303299331665039}}, "EndTime": 1584909801.800962, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.80095}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.7961368560791016, "sum": 0.7961368560791016, "min": 0.7961368560791016}}, "EndTime": 1584909801.801017, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.801001}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.1979327392578125, "sum": 1.1979327392578125, "min": 1.1979327392578125}}, "EndTime": 1584909801.801077, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.801061}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.7341841888427735, "sum": 0.7341841888427735, "min": 0.7341841888427735}}, "EndTime": 1584909801.801138, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.801122}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.938225326538086, "sum": 0.938225326538086, "min": 0.938225326538086}}, "EndTime": 1584909801.801182, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.801172}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5709184646606446, "sum": 0.5709184646606446, "min": 0.5709184646606446}}, "EndTime": 1584909801.801213, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.801205}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5595186996459961, "sum": 0.5595186996459961, "min": 0.5595186996459961}}, "EndTime": 1584909801.801243, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.801235}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.4524125289916992, "sum": 0.4524125289916992, "min": 0.4524125289916992}}, "EndTime": 1584909801.801291, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.801276}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6402424621582031, "sum": 0.6402424621582031, "min": 0.6402424621582031}}, "EndTime": 1584909801.801353, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.801336}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9192237854003906, "sum": 0.9192237854003906, "min": 0.9192237854003906}}, "EndTime": 1584909801.801426, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.801408}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.749781494140625, "sum": 0.749781494140625, "min": 0.749781494140625}}, "EndTime": 1584909801.801499, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.801482}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.0552902221679688, "sum": 1.0552902221679688, "min": 1.0552902221679688}}, "EndTime": 1584909801.801572, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.801554}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.8965641784667969, "sum": 0.8965641784667969, "min": 0.8965641784667969}}, "EndTime": 1584909801.801644, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.801626}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.48117549896240236, "sum": 0.48117549896240236, "min": 0.48117549896240236}}, "EndTime": 1584909801.801714, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.801697}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5534622955322266, "sum": 0.5534622955322266, "min": 0.5534622955322266}}, "EndTime": 1584909801.801787, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.801769}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.4394077301025391, "sum": 0.4394077301025391, "min": 0.4394077301025391}}, "EndTime": 1584909801.801849, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.801833}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.4314213943481445, "sum": 0.4314213943481445, "min": 0.4314213943481445}}, "EndTime": 1584909801.80192, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.801902}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9523495483398438, "sum": 0.9523495483398438, "min": 0.9523495483398438}}, "EndTime": 1584909801.80198, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.801964}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9592522430419922, "sum": 0.9592522430419922, "min": 0.9592522430419922}}, "EndTime": 1584909801.802032, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.802017}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.1009056854248047, "sum": 1.1009056854248047, "min": 1.1009056854248047}}, "EndTime": 1584909801.802091, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.802075}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.0718785858154296, "sum": 1.0718785858154296, "min": 1.0718785858154296}}, "EndTime": 1584909801.802152, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.802136}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9313733673095703, "sum": 0.9313733673095703, "min": 0.9313733673095703}}, "EndTime": 1584909801.802211, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.802196}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.8697193908691406, "sum": 0.8697193908691406, "min": 0.8697193908691406}}, "EndTime": 1584909801.80225, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.802241}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9301378631591797, "sum": 0.9301378631591797, "min": 0.9301378631591797}}, "EndTime": 1584909801.80228, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.802272}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9741603088378906, "sum": 0.9741603088378906, "min": 0.9741603088378906}}, "EndTime": 1584909801.802308, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.802301}
[0m
[34m[03/22/2020 20:43:21 INFO 140403405219648] #quality_metric: host=algo-1, epoch=3, train mse_objective <loss>=0.825902709961[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 76.08393205915179, "sum": 76.08393205915179, "min": 76.08393205915179}}, "EndTime": 1584909801.845819, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.845731}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 72.33786882672992, "sum": 72.33786882672992, "min": 72.33786882672992}}, "EndTime": 1584909801.84591, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.845889}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 101.08469063895089, "sum": 101.08469063895089, "min": 101.08469063895089}}, "EndTime": 1584909801.845976, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.845959}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 70.6239013671875, "sum": 70.6239013671875, "min": 70.6239013671875}}, "EndTime": 1584909801.846033, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.846019}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 42.788670131138396, "sum": 42.788670131138396, "min": 42.788670131138396}}, "EndTime": 1584909801.846085, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.846072}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 38.51235089983259, "sum": 38.51235089983259, "min": 38.51235089983259}}, "EndTime": 1584909801.84614, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.846125}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 42.54933820452009, "sum": 42.54933820452009, "min": 42.54933820452009}}, "EndTime": 1584909801.846198, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.846181}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 37.52103969029018, "sum": 37.52103969029018, "min": 37.52103969029018}}, "EndTime": 1584909801.846254, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.846239}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 75.96677071707589, "sum": 75.96677071707589, "min": 75.96677071707589}}, "EndTime": 1584909801.846307, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.846292}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 107.35875592912946, "sum": 107.35875592912946, "min": 107.35875592912946}}, "EndTime": 1584909801.846363, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.846348}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 71.88251604352679, "sum": 71.88251604352679, "min": 71.88251604352679}}, "EndTime": 1584909801.846419, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.846404}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 88.18191092354911, "sum": 88.18191092354911, "min": 88.18191092354911}}, "EndTime": 1584909801.84647, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.846458}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 41.08176531110491, "sum": 41.08176531110491, "min": 41.08176531110491}}, "EndTime": 1584909801.846519, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.846506}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 40.438267299107146, "sum": 40.438267299107146, "min": 40.438267299107146}}, "EndTime": 1584909801.84657, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.846555}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 41.019622802734375, "sum": 41.019622802734375, "min": 41.019622802734375}}, "EndTime": 1584909801.846625, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.846612}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 40.94205583844866, "sum": 40.94205583844866, "min": 40.94205583844866}}, "EndTime": 1584909801.846678, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.846664}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 84.47986711774554, "sum": 84.47986711774554, "min": 84.47986711774554}}, "EndTime": 1584909801.846728, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.846715}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 70.03952026367188, "sum": 70.03952026367188, "min": 70.03952026367188}}, "EndTime": 1584909801.846777, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.846764}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 100.65732247488839, "sum": 100.65732247488839, "min": 100.65732247488839}}, "EndTime": 1584909801.846827, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.846814}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 83.93590436662946, "sum": 83.93590436662946, "min": 83.93590436662946}}, "EndTime": 1584909801.846884, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.846868}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 66.9688241141183, "sum": 66.9688241141183, "min": 66.9688241141183}}, "EndTime": 1584909801.846938, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.846923}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 43.212493896484375, "sum": 43.212493896484375, "min": 43.212493896484375}}, "EndTime": 1584909801.846992, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.846977}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 63.008108956473215, "sum": 63.008108956473215, "min": 63.008108956473215}}, "EndTime": 1584909801.847048, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.847033}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 52.616245814732146, "sum": 52.616245814732146, "min": 52.616245814732146}}, "EndTime": 1584909801.847106, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.847091}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 92.42780412946429, "sum": 92.42780412946429, "min": 92.42780412946429}}, "EndTime": 1584909801.847166, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.847149}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 93.66891915457589, "sum": 93.66891915457589, "min": 93.66891915457589}}, "EndTime": 1584909801.847221, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.847206}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 103.25526646205357, "sum": 103.25526646205357, "min": 103.25526646205357}}, "EndTime": 1584909801.847276, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.847261}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 103.16184779575893, "sum": 103.16184779575893, "min": 103.16184779575893}}, "EndTime": 1584909801.847331, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.847316}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 104.89678955078125, "sum": 104.89678955078125, "min": 104.89678955078125}}, "EndTime": 1584909801.847384, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.84737}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 92.91693987165179, "sum": 92.91693987165179, "min": 92.91693987165179}}, "EndTime": 1584909801.84744, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.847424}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 113.21119907924107, "sum": 113.21119907924107, "min": 113.21119907924107}}, "EndTime": 1584909801.847524, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.847508}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 108.90509905133929, "sum": 108.90509905133929, "min": 108.90509905133929}}, "EndTime": 1584909801.84758, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.847566}
[0m
[34m[03/22/2020 20:43:21 INFO 140403405219648] #quality_metric: host=algo-1, epoch=3, validation mse_objective <loss>=76.0839320592[0m
[34m[03/22/2020 20:43:21 INFO 140403405219648] #early_stopping_criteria_metric: host=algo-1, epoch=3, criteria=mse_objective, value=37.5210396903[0m
[34m[03/22/2020 20:43:21 INFO 140403405219648] #progress_metric: host=algo-1, completed 26 % of epochs[0m
[34m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 2, "sum": 2.0, "min": 2}, "Number of Batches Since Last Reset": {"count": 1, "max": 2, "sum": 2.0, "min": 2}, "Number of Records Since Last Reset": {"count": 1, "max": 227, "sum": 227.0, "min": 227}, "Total Batches Seen": {"count": 1, "max": 11, "sum": 11.0, "min": 11}, "Total Records Seen": {"count": 1, "max": 1335, "sum": 1335.0, "min": 1335}, "Max Records Seen Between Resets": {"count": 1, "max": 227, "sum": 227.0, "min": 227}, "Reset Count": {"count": 1, "max": 6, "sum": 6.0, "min": 6}}, "EndTime": 1584909801.848553, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 3}, "StartTime": 1584909801.773037}
[0m
[34m[03/22/2020 20:43:21 INFO 140403405219648] #throughput_metric: host=algo-1, train throughput=3000.98343026 records/second[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.78623291015625, "sum": 0.78623291015625, "min": 0.78623291015625}}, "EndTime": 1584909801.874819, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.874768}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.7175883483886719, "sum": 0.7175883483886719, "min": 0.7175883483886719}}, "EndTime": 1584909801.874892, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.874872}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.0666070556640626, "sum": 1.0666070556640626, "min": 1.0666070556640626}}, "EndTime": 1584909801.874962, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.874944}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.7042116546630859, "sum": 0.7042116546630859, "min": 0.7042116546630859}}, "EndTime": 1584909801.875029, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.875011}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5904379653930664, "sum": 0.5904379653930664, "min": 0.5904379653930664}}, "EndTime": 1584909801.875094, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.875075}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6363718032836914, "sum": 0.6363718032836914, "min": 0.6363718032836914}}, "EndTime": 1584909801.875157, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.875139}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6725286102294922, "sum": 0.6725286102294922, "min": 0.6725286102294922}}, "EndTime": 1584909801.875219, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.875201}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5919160842895508, "sum": 0.5919160842895508, "min": 0.5919160842895508}}, "EndTime": 1584909801.875283, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.875265}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.7638082885742188, "sum": 0.7638082885742188, "min": 0.7638082885742188}}, "EndTime": 1584909801.875345, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.875329}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.1352942657470704, "sum": 1.1352942657470704, "min": 1.1352942657470704}}, "EndTime": 1584909801.875405, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.875388}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.7052783203125, "sum": 0.7052783203125, "min": 0.7052783203125}}, "EndTime": 1584909801.875487, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.875447}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.8944029235839843, "sum": 0.8944029235839843, "min": 0.8944029235839843}}, "EndTime": 1584909801.87555, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.875532}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6214497375488282, "sum": 0.6214497375488282, "min": 0.6214497375488282}}, "EndTime": 1584909801.875612, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.875595}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5929115295410157, "sum": 0.5929115295410157, "min": 0.5929115295410157}}, "EndTime": 1584909801.875677, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.875659}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5746974563598632, "sum": 0.5746974563598632, "min": 0.5746974563598632}}, "EndTime": 1584909801.875738, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.875721}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6838236236572266, "sum": 0.6838236236572266, "min": 0.6838236236572266}}, "EndTime": 1584909801.8758, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.875784}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.8759243011474609, "sum": 0.8759243011474609, "min": 0.8759243011474609}}, "EndTime": 1584909801.875861, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.875844}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.7217394256591797, "sum": 0.7217394256591797, "min": 0.7217394256591797}}, "EndTime": 1584909801.875919, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.875902}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.0080760192871094, "sum": 1.0080760192871094, "min": 1.0080760192871094}}, "EndTime": 1584909801.875977, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.875961}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.856258544921875, "sum": 0.856258544921875, "min": 0.856258544921875}}, "EndTime": 1584909801.876033, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.876018}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.7793168640136718, "sum": 0.7793168640136718, "min": 0.7793168640136718}}, "EndTime": 1584909801.876093, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.876075}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.572952880859375, "sum": 0.572952880859375, "min": 0.572952880859375}}, "EndTime": 1584909801.876153, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.876136}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.7260540008544922, "sum": 0.7260540008544922, "min": 0.7260540008544922}}, "EndTime": 1584909801.876217, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.8762}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.623348503112793, "sum": 0.623348503112793, "min": 0.623348503112793}}, "EndTime": 1584909801.876278, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.876261}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9511940002441406, "sum": 0.9511940002441406, "min": 0.9511940002441406}}, "EndTime": 1584909801.87634, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.876323}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9645113372802734, "sum": 0.9645113372802734, "min": 0.9645113372802734}}, "EndTime": 1584909801.876401, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.876384}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.0671381378173828, "sum": 1.0671381378173828, "min": 1.0671381378173828}}, "EndTime": 1584909801.876462, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.876445}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.056426773071289, "sum": 1.056426773071289, "min": 1.056426773071289}}, "EndTime": 1584909801.87652, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.876504}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.076737518310547, "sum": 1.076737518310547, "min": 1.076737518310547}}, "EndTime": 1584909801.876577, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.876561}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9277210998535156, "sum": 0.9277210998535156, "min": 0.9277210998535156}}, "EndTime": 1584909801.876635, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.87662}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.1787638092041015, "sum": 1.1787638092041015, "min": 1.1787638092041015}}, "EndTime": 1584909801.876692, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.876677}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.1589371490478515, "sum": 1.1589371490478515, "min": 1.1589371490478515}}, "EndTime": 1584909801.876744, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.87673}
[0m
[34m[03/22/2020 20:43:21 INFO 140403405219648] #quality_metric: host=algo-1, epoch=4, train mse_objective <loss>=0.786232910156[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 72.5796377999442, "sum": 72.5796377999442, "min": 72.5796377999442}}, "EndTime": 1584909801.92291, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.922852}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 69.33903285435268, "sum": 69.33903285435268, "min": 69.33903285435268}}, "EndTime": 1584909801.922983, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.922962}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 96.26747349330357, "sum": 96.26747349330357, "min": 96.26747349330357}}, "EndTime": 1584909801.923055, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.923036}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 67.56520298549107, "sum": 67.56520298549107, "min": 67.56520298549107}}, "EndTime": 1584909801.923123, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.923106}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 54.439117431640625, "sum": 54.439117431640625, "min": 54.439117431640625}}, "EndTime": 1584909801.923187, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.92317}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 45.258941650390625, "sum": 45.258941650390625, "min": 45.258941650390625}}, "EndTime": 1584909801.92325, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.923233}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 49.656712123325896, "sum": 49.656712123325896, "min": 49.656712123325896}}, "EndTime": 1584909801.923314, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.923296}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 44.044730050223215, "sum": 44.044730050223215, "min": 44.044730050223215}}, "EndTime": 1584909801.923377, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.92336}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 73.13272530691964, "sum": 73.13272530691964, "min": 73.13272530691964}}, "EndTime": 1584909801.923437, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.923421}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 102.31316266741071, "sum": 102.31316266741071, "min": 102.31316266741071}}, "EndTime": 1584909801.923524, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.923506}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 69.19494192940849, "sum": 69.19494192940849, "min": 69.19494192940849}}, "EndTime": 1584909801.923584, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.923568}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 84.44674246651786, "sum": 84.44674246651786, "min": 84.44674246651786}}, "EndTime": 1584909801.923641, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.923625}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 52.908316476004465, "sum": 52.908316476004465, "min": 52.908316476004465}}, "EndTime": 1584909801.923702, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.923684}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 51.18747820172991, "sum": 51.18747820172991, "min": 51.18747820172991}}, "EndTime": 1584909801.923765, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.923748}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 51.534149169921875, "sum": 51.534149169921875, "min": 51.534149169921875}}, "EndTime": 1584909801.923826, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.923811}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 53.201219831194194, "sum": 53.201219831194194, "min": 53.201219831194194}}, "EndTime": 1584909801.923889, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.923872}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 80.72545514787946, "sum": 80.72545514787946, "min": 80.72545514787946}}, "EndTime": 1584909801.92395, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.923934}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 67.45492117745536, "sum": 67.45492117745536, "min": 67.45492117745536}}, "EndTime": 1584909801.924009, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.923991}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 96.54197474888393, "sum": 96.54197474888393, "min": 96.54197474888393}}, "EndTime": 1584909801.924069, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.924052}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 80.43093436104911, "sum": 80.43093436104911, "min": 80.43093436104911}}, "EndTime": 1584909801.924124, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.924109}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 73.04622541155133, "sum": 73.04622541155133, "min": 73.04622541155133}}, "EndTime": 1584909801.924181, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.924164}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 58.08614240373884, "sum": 58.08614240373884, "min": 58.08614240373884}}, "EndTime": 1584909801.924242, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.924226}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 73.77780587332589, "sum": 73.77780587332589, "min": 73.77780587332589}}, "EndTime": 1584909801.924306, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.924288}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 63.87804739815848, "sum": 63.87804739815848, "min": 63.87804739815848}}, "EndTime": 1584909801.924366, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.92435}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 92.69971575055804, "sum": 92.69971575055804, "min": 92.69971575055804}}, "EndTime": 1584909801.924428, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.924411}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 93.80965750558036, "sum": 93.80965750558036, "min": 93.80965750558036}}, "EndTime": 1584909801.92449, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.924473}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 100.45240129743304, "sum": 100.45240129743304, "min": 100.45240129743304}}, "EndTime": 1584909801.924551, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.924535}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 101.37935965401786, "sum": 101.37935965401786, "min": 101.37935965401786}}, "EndTime": 1584909801.92461, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.924594}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 112.82016427176339, "sum": 112.82016427176339, "min": 112.82016427176339}}, "EndTime": 1584909801.924668, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.924651}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 101.57064383370536, "sum": 101.57064383370536, "min": 101.57064383370536}}, "EndTime": 1584909801.924726, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.924709}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 123.52996826171875, "sum": 123.52996826171875, "min": 123.52996826171875}}, "EndTime": 1584909801.924782, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.924767}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 115.73095703125, "sum": 115.73095703125, "min": 115.73095703125}}, "EndTime": 1584909801.924833, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.924819}
[0m
[34m[03/22/2020 20:43:21 INFO 140403405219648] #quality_metric: host=algo-1, epoch=4, validation mse_objective <loss>=72.5796377999[0m
[34m[03/22/2020 20:43:21 INFO 140403405219648] #early_stopping_criteria_metric: host=algo-1, epoch=4, criteria=mse_objective, value=44.0447300502[0m
[34m[03/22/2020 20:43:21 INFO 140403405219648] #progress_metric: host=algo-1, completed 33 % of epochs[0m
[34m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 2, "sum": 2.0, "min": 2}, "Number of Batches Since Last Reset": {"count": 1, "max": 2, "sum": 2.0, "min": 2}, "Number of Records Since Last Reset": {"count": 1, "max": 227, "sum": 227.0, "min": 227}, "Total Batches Seen": {"count": 1, "max": 13, "sum": 13.0, "min": 13}, "Total Records Seen": {"count": 1, "max": 1562, "sum": 1562.0, "min": 1562}, "Max Records Seen Between Resets": {"count": 1, "max": 227, "sum": 227.0, "min": 227}, "Reset Count": {"count": 1, "max": 7, "sum": 7.0, "min": 7}}, "EndTime": 1584909801.925869, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 4}, "StartTime": 1584909801.848808}
[0m
[34m[03/22/2020 20:43:21 INFO 140403405219648] #throughput_metric: host=algo-1, train throughput=2940.45327305 records/second[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.7500071716308594, "sum": 0.7500071716308594, "min": 0.7500071716308594}}, "EndTime": 1584909801.950316, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909801.950265}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6893840026855469, "sum": 0.6893840026855469, "min": 0.6893840026855469}}, "EndTime": 1584909801.950388, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909801.95037}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.0113765716552734, "sum": 1.0113765716552734, "min": 1.0113765716552734}}, "EndTime": 1584909801.95046, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909801.95044}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6750420379638672, "sum": 0.6750420379638672, "min": 0.6750420379638672}}, "EndTime": 1584909801.950516, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909801.9505}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.7188327789306641, "sum": 0.7188327789306641, "min": 0.7188327789306641}}, "EndTime": 1584909801.950589, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909801.95057}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6397027969360352, "sum": 0.6397027969360352, "min": 0.6397027969360352}}, "EndTime": 1584909801.950652, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909801.950635}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6982103729248047, "sum": 0.6982103729248047, "min": 0.6982103729248047}}, "EndTime": 1584909801.950724, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909801.950706}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6275647354125976, "sum": 0.6275647354125976, "min": 0.6275647354125976}}, "EndTime": 1584909801.950788, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909801.950771}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.735096206665039, "sum": 0.735096206665039, "min": 0.735096206665039}}, "EndTime": 1584909801.950856, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909801.950838}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.0762032318115233, "sum": 1.0762032318115233, "min": 1.0762032318115233}}, "EndTime": 1584909801.950897, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909801.950887}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6798036956787109, "sum": 0.6798036956787109, "min": 0.6798036956787109}}, "EndTime": 1584909801.95093, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909801.95092}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.8540394592285157, "sum": 0.8540394592285157, "min": 0.8540394592285157}}, "EndTime": 1584909801.950983, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909801.950968}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.7173468780517578, "sum": 0.7173468780517578, "min": 0.7173468780517578}}, "EndTime": 1584909801.951052, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909801.951033}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6817201995849609, "sum": 0.6817201995849609, "min": 0.6817201995849609}}, "EndTime": 1584909801.951112, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909801.951096}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.7020589447021485, "sum": 0.7020589447021485, "min": 0.7020589447021485}}, "EndTime": 1584909801.951173, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909801.951156}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.7678363037109375, "sum": 0.7678363037109375, "min": 0.7678363037109375}}, "EndTime": 1584909801.951235, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909801.951217}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.836290512084961, "sum": 0.836290512084961, "min": 0.836290512084961}}, "EndTime": 1584909801.951293, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909801.951277}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.696348876953125, "sum": 0.696348876953125, "min": 0.696348876953125}}, "EndTime": 1584909801.951344, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909801.951334}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9643987274169922, "sum": 0.9643987274169922, "min": 0.9643987274169922}}, "EndTime": 1584909801.951375, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909801.951367}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.8194747924804687, "sum": 0.8194747924804687, "min": 0.8194747924804687}}, "EndTime": 1584909801.951403, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909801.951396}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.8488652801513672, "sum": 0.8488652801513672, "min": 0.8488652801513672}}, "EndTime": 1584909801.951486, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909801.951446}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6864603424072265, "sum": 0.6864603424072265, "min": 0.6864603424072265}}, "EndTime": 1584909801.951558, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909801.95154}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.8429108428955078, "sum": 0.8429108428955078, "min": 0.8429108428955078}}, "EndTime": 1584909801.951622, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909801.951605}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.7442388916015625, "sum": 0.7442388916015625, "min": 0.7442388916015625}}, "EndTime": 1584909801.951694, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909801.951677}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9544205474853515, "sum": 0.9544205474853515, "min": 0.9544205474853515}}, "EndTime": 1584909801.951766, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909801.951748}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9672857666015625, "sum": 0.9672857666015625, "min": 0.9672857666015625}}, "EndTime": 1584909801.951829, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909801.951811}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.0365152740478516, "sum": 1.0365152740478516, "min": 1.0365152740478516}}, "EndTime": 1584909801.951892, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909801.951876}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.036024627685547, "sum": 1.036024627685547, "min": 1.036024627685547}}, "EndTime": 1584909801.951952, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909801.951935}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.1602764892578126, "sum": 1.1602764892578126, "min": 1.1602764892578126}}, "EndTime": 1584909801.951988, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909801.95198}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.0433261108398437, "sum": 1.0433261108398437, "min": 1.0433261108398437}}, "EndTime": 1584909801.952046, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909801.95203}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.3006307983398437, "sum": 1.3006307983398437, "min": 1.3006307983398437}}, "EndTime": 1584909801.952107, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909801.952091}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.2266719818115235, "sum": 1.2266719818115235, "min": 1.2266719818115235}}, "EndTime": 1584909801.952166, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909801.952151}
[0m
[34m[03/22/2020 20:43:21 INFO 140403405219648] #quality_metric: host=algo-1, epoch=5, train mse_objective <loss>=0.750007171631[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 69.25123814174107, "sum": 69.25123814174107, "min": 69.25123814174107}}, "EndTime": 1584909802.000655, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909802.00059}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 66.50734601702008, "sum": 66.50734601702008, "min": 66.50734601702008}}, "EndTime": 1584909802.000746, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909802.000725}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 91.548583984375, "sum": 91.548583984375, "min": 91.548583984375}}, "EndTime": 1584909802.000812, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909802.000795}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 64.68763950892857, "sum": 64.68763950892857, "min": 64.68763950892857}}, "EndTime": 1584909802.000876, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909802.000859}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 39.12682669503348, "sum": 39.12682669503348, "min": 39.12682669503348}}, "EndTime": 1584909802.000948, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909802.000929}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 42.505358014787944, "sum": 42.505358014787944, "min": 42.505358014787944}}, "EndTime": 1584909802.001008, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909802.000992}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 44.526903424944194, "sum": 44.526903424944194, "min": 44.526903424944194}}, "EndTime": 1584909802.001068, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909802.001052}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 40.99756295340402, "sum": 40.99756295340402, "min": 40.99756295340402}}, "EndTime": 1584909802.001129, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909802.001112}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 70.46681867327008, "sum": 70.46681867327008, "min": 70.46681867327008}}, "EndTime": 1584909802.001194, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909802.001174}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 97.33812604631696, "sum": 97.33812604631696, "min": 97.33812604631696}}, "EndTime": 1584909802.001255, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909802.001238}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 66.69845145089286, "sum": 66.69845145089286, "min": 66.69845145089286}}, "EndTime": 1584909802.001317, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909802.0013}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 80.88078090122768, "sum": 80.88078090122768, "min": 80.88078090122768}}, "EndTime": 1584909802.00138, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909802.001363}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 44.65081787109375, "sum": 44.65081787109375, "min": 44.65081787109375}}, "EndTime": 1584909802.001441, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909802.001423}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 43.011705671037944, "sum": 43.011705671037944, "min": 43.011705671037944}}, "EndTime": 1584909802.001502, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909802.001485}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 37.130445207868306, "sum": 37.130445207868306, "min": 37.130445207868306}}, "EndTime": 1584909802.001564, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909802.001547}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 48.250457763671875, "sum": 48.250457763671875, "min": 48.250457763671875}}, "EndTime": 1584909802.001626, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909802.001609}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 77.16443743024554, "sum": 77.16443743024554, "min": 77.16443743024554}}, "EndTime": 1584909802.001685, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909802.001669}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 65.01811872209821, "sum": 65.01811872209821, "min": 65.01811872209821}}, "EndTime": 1584909802.001745, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909802.001729}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 92.60847691127232, "sum": 92.60847691127232, "min": 92.60847691127232}}, "EndTime": 1584909802.001802, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909802.001786}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 77.05827113560268, "sum": 77.05827113560268, "min": 77.05827113560268}}, "EndTime": 1584909802.001861, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909802.001845}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 47.043627057756694, "sum": 47.043627057756694, "min": 47.043627057756694}}, "EndTime": 1584909802.001919, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909802.001903}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 57.811375209263396, "sum": 57.811375209263396, "min": 57.811375209263396}}, "EndTime": 1584909802.001973, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909802.001958}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 50.257969447544646, "sum": 50.257969447544646, "min": 50.257969447544646}}, "EndTime": 1584909802.002031, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909802.002015}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 45.87262398856027, "sum": 45.87262398856027, "min": 45.87262398856027}}, "EndTime": 1584909802.002091, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909802.002075}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 93.33924211774554, "sum": 93.33924211774554, "min": 93.33924211774554}}, "EndTime": 1584909802.002152, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909802.002137}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 93.6575927734375, "sum": 93.6575927734375, "min": 93.6575927734375}}, "EndTime": 1584909802.002212, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909802.002196}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 98.00827462332589, "sum": 98.00827462332589, "min": 98.00827462332589}}, "EndTime": 1584909802.002274, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909802.002257}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 99.27504185267857, "sum": 99.27504185267857, "min": 99.27504185267857}}, "EndTime": 1584909802.002335, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909802.002318}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 101.47578648158482, "sum": 101.47578648158482, "min": 101.47578648158482}}, "EndTime": 1584909802.002396, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909802.00238}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 99.34935651506696, "sum": 99.34935651506696, "min": 99.34935651506696}}, "EndTime": 1584909802.002455, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909802.002439}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 108.50881522042411, "sum": 108.50881522042411, "min": 108.50881522042411}}, "EndTime": 1584909802.002514, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909802.002497}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 102.30338832310268, "sum": 102.30338832310268, "min": 102.30338832310268}}, "EndTime": 1584909802.002572, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909802.002557}
[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] #quality_metric: host=algo-1, epoch=5, validation mse_objective <loss>=69.2512381417[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] #early_stopping_criteria_metric: host=algo-1, epoch=5, criteria=mse_objective, value=37.1304452079[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] #progress_metric: host=algo-1, completed 40 % of epochs[0m
[34m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 2, "sum": 2.0, "min": 2}, "Number of Batches Since Last Reset": {"count": 1, "max": 2, "sum": 2.0, "min": 2}, "Number of Records Since Last Reset": {"count": 1, "max": 227, "sum": 227.0, "min": 227}, "Total Batches Seen": {"count": 1, "max": 15, "sum": 15.0, "min": 15}, "Total Records Seen": {"count": 1, "max": 1789, "sum": 1789.0, "min": 1789}, "Max Records Seen Between Resets": {"count": 1, "max": 227, "sum": 227.0, "min": 227}, "Reset Count": {"count": 1, "max": 8, "sum": 8.0, "min": 8}}, "EndTime": 1584909802.003694, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 5}, "StartTime": 1584909801.926149}
[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] #throughput_metric: host=algo-1, train throughput=2922.09743731 records/second[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.7167576599121094, "sum": 0.7167576599121094, "min": 0.7167576599121094}}, "EndTime": 1584909802.029995, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.029925}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.663909912109375, "sum": 0.663909912109375, "min": 0.663909912109375}}, "EndTime": 1584909802.030076, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.030055}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9592703247070312, "sum": 0.9592703247070312, "min": 0.9592703247070312}}, "EndTime": 1584909802.030147, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.030127}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.648801498413086, "sum": 0.648801498413086, "min": 0.648801498413086}}, "EndTime": 1584909802.030213, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.030195}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5352636337280273, "sum": 0.5352636337280273, "min": 0.5352636337280273}}, "EndTime": 1584909802.030278, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.03026}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.534823989868164, "sum": 0.534823989868164, "min": 0.534823989868164}}, "EndTime": 1584909802.030339, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.030323}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5701558303833008, "sum": 0.5701558303833008, "min": 0.5701558303833008}}, "EndTime": 1584909802.030403, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.030386}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5468652725219727, "sum": 0.5468652725219727, "min": 0.5468652725219727}}, "EndTime": 1584909802.030465, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.030447}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.7094789123535157, "sum": 0.7094789123535157, "min": 0.7094789123535157}}, "EndTime": 1584909802.030527, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.03051}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.020320281982422, "sum": 1.020320281982422, "min": 1.020320281982422}}, "EndTime": 1584909802.030589, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.030573}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6571885681152344, "sum": 0.6571885681152344, "min": 0.6571885681152344}}, "EndTime": 1584909802.030648, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.030632}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.8167166900634766, "sum": 0.8167166900634766, "min": 0.8167166900634766}}, "EndTime": 1584909802.03071, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.030693}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5807131195068359, "sum": 0.5807131195068359, "min": 0.5807131195068359}}, "EndTime": 1584909802.03077, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.030753}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.555652847290039, "sum": 0.555652847290039, "min": 0.555652847290039}}, "EndTime": 1584909802.030834, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.030817}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.536109848022461, "sum": 0.536109848022461, "min": 0.536109848022461}}, "EndTime": 1584909802.030896, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.030879}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.632759895324707, "sum": 0.632759895324707, "min": 0.632759895324707}}, "EndTime": 1584909802.030954, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.030938}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.7999306488037109, "sum": 0.7999306488037109, "min": 0.7999306488037109}}, "EndTime": 1584909802.031009, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.030994}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6731254577636718, "sum": 0.6731254577636718, "min": 0.6731254577636718}}, "EndTime": 1584909802.031068, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.031051}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9238578033447266, "sum": 0.9238578033447266, "min": 0.9238578033447266}}, "EndTime": 1584909802.031133, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.031117}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.7856224060058594, "sum": 0.7856224060058594, "min": 0.7856224060058594}}, "EndTime": 1584909802.031194, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.031178}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5467159652709961, "sum": 0.5467159652709961, "min": 0.5467159652709961}}, "EndTime": 1584909802.031253, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.031238}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6405708312988281, "sum": 0.6405708312988281, "min": 0.6405708312988281}}, "EndTime": 1584909802.031313, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.031296}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5691900634765625, "sum": 0.5691900634765625, "min": 0.5691900634765625}}, "EndTime": 1584909802.03137, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.031353}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5316627883911133, "sum": 0.5316627883911133, "min": 0.5316627883911133}}, "EndTime": 1584909802.031436, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.031419}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.960267562866211, "sum": 0.960267562866211, "min": 0.960267562866211}}, "EndTime": 1584909802.031522, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.031504}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9683591461181641, "sum": 0.9683591461181641, "min": 0.9683591461181641}}, "EndTime": 1584909802.031585, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.031568}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.0085065460205078, "sum": 1.0085065460205078, "min": 1.0085065460205078}}, "EndTime": 1584909802.031648, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.031632}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.013434600830078, "sum": 1.013434600830078, "min": 1.013434600830078}}, "EndTime": 1584909802.031709, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.031692}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.0288938140869142, "sum": 1.0288938140869142, "min": 1.0288938140869142}}, "EndTime": 1584909802.031759, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.031742}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.013187713623047, "sum": 1.013187713623047, "min": 1.013187713623047}}, "EndTime": 1584909802.031812, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.031796}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.1232315063476563, "sum": 1.1232315063476563, "min": 1.1232315063476563}}, "EndTime": 1584909802.031877, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.03186}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.0450366973876952, "sum": 1.0450366973876952, "min": 1.0450366973876952}}, "EndTime": 1584909802.031931, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.031915}
[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] #quality_metric: host=algo-1, epoch=6, train mse_objective <loss>=0.716757659912[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 66.21990094866071, "sum": 66.21990094866071, "min": 66.21990094866071}}, "EndTime": 1584909802.076676, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.076619}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 63.955923897879465, "sum": 63.955923897879465, "min": 63.955923897879465}}, "EndTime": 1584909802.076749, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.076729}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 87.10568673270089, "sum": 87.10568673270089, "min": 87.10568673270089}}, "EndTime": 1584909802.076818, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.076799}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 62.10377720424107, "sum": 62.10377720424107, "min": 62.10377720424107}}, "EndTime": 1584909802.076886, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.076869}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 26.107572283063615, "sum": 26.107572283063615, "min": 26.107572283063615}}, "EndTime": 1584909802.076953, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.076935}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 39.513602120535715, "sum": 39.513602120535715, "min": 39.513602120535715}}, "EndTime": 1584909802.077027, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.077008}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 38.371852329799104, "sum": 38.371852329799104, "min": 38.371852329799104}}, "EndTime": 1584909802.077087, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.077073}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 37.072579520089285, "sum": 37.072579520089285, "min": 37.072579520089285}}, "EndTime": 1584909802.077139, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.077123}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 68.07570103236607, "sum": 68.07570103236607, "min": 68.07570103236607}}, "EndTime": 1584909802.07721, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.077192}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 92.63158307756696, "sum": 92.63158307756696, "min": 92.63158307756696}}, "EndTime": 1584909802.077281, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.077264}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 64.46640014648438, "sum": 64.46640014648438, "min": 64.46640014648438}}, "EndTime": 1584909802.077347, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.07733}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 77.60786655970982, "sum": 77.60786655970982, "min": 77.60786655970982}}, "EndTime": 1584909802.077409, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.077392}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 32.8871830531529, "sum": 32.8871830531529, "min": 32.8871830531529}}, "EndTime": 1584909802.077471, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.077454}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 32.39271763392857, "sum": 32.39271763392857, "min": 32.39271763392857}}, "EndTime": 1584909802.077531, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.077514}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 26.382441929408483, "sum": 26.382441929408483, "min": 26.382441929408483}}, "EndTime": 1584909802.077594, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.077577}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 38.37223161969866, "sum": 38.37223161969866, "min": 38.37223161969866}}, "EndTime": 1584909802.077658, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.077641}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 73.91422816685268, "sum": 73.91422816685268, "min": 73.91422816685268}}, "EndTime": 1584909802.077721, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.077705}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 62.80383736746652, "sum": 62.80383736746652, "min": 62.80383736746652}}, "EndTime": 1584909802.077789, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.077772}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 88.98114013671875, "sum": 88.98114013671875, "min": 88.98114013671875}}, "EndTime": 1584909802.077853, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.077835}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 73.96466064453125, "sum": 73.96466064453125, "min": 73.96466064453125}}, "EndTime": 1584909802.07792, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.077903}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 31.02677481515067, "sum": 31.02677481515067, "min": 31.02677481515067}}, "EndTime": 1584909802.07799, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.077972}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 51.682041713169646, "sum": 51.682041713169646, "min": 51.682041713169646}}, "EndTime": 1584909802.07806, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.078042}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 33.68668474469866, "sum": 33.68668474469866, "min": 33.68668474469866}}, "EndTime": 1584909802.078122, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.078105}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 33.35233415876116, "sum": 33.35233415876116, "min": 33.35233415876116}}, "EndTime": 1584909802.078184, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.078168}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 94.19819859095982, "sum": 94.19819859095982, "min": 94.19819859095982}}, "EndTime": 1584909802.078254, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.078237}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 93.42937360491071, "sum": 93.42937360491071, "min": 93.42937360491071}}, "EndTime": 1584909802.078319, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.078301}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 95.91932024274554, "sum": 95.91932024274554, "min": 95.91932024274554}}, "EndTime": 1584909802.078381, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.078365}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 97.21752057756696, "sum": 97.21752057756696, "min": 97.21752057756696}}, "EndTime": 1584909802.078442, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.078426}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 95.63436453683036, "sum": 95.63436453683036, "min": 95.63436453683036}}, "EndTime": 1584909802.078504, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.078486}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 101.92496163504464, "sum": 101.92496163504464, "min": 101.92496163504464}}, "EndTime": 1584909802.078575, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.078557}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 96.22028459821429, "sum": 96.22028459821429, "min": 96.22028459821429}}, "EndTime": 1584909802.078636, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.078619}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 92.54502650669643, "sum": 92.54502650669643, "min": 92.54502650669643}}, "EndTime": 1584909802.078696, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.07868}
[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] #quality_metric: host=algo-1, epoch=6, validation mse_objective <loss>=66.2199009487[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] #early_stopping_criteria_metric: host=algo-1, epoch=6, criteria=mse_objective, value=26.1075722831[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] Epoch 6: Loss improved. Updating best model[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] #progress_metric: host=algo-1, completed 46 % of epochs[0m
[34m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 2, "sum": 2.0, "min": 2}, "Number of Batches Since Last Reset": {"count": 1, "max": 2, "sum": 2.0, "min": 2}, "Number of Records Since Last Reset": {"count": 1, "max": 227, "sum": 227.0, "min": 227}, "Total Batches Seen": {"count": 1, "max": 17, "sum": 17.0, "min": 17}, "Total Records Seen": {"count": 1, "max": 2016, "sum": 2016.0, "min": 2016}, "Max Records Seen Between Resets": {"count": 1, "max": 227, "sum": 227.0, "min": 227}, "Reset Count": {"count": 1, "max": 9, "sum": 9.0, "min": 9}}, "EndTime": 1584909802.080395, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 6}, "StartTime": 1584909802.003974}
[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] #throughput_metric: host=algo-1, train throughput=2964.7539344 records/second[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6873529815673828, "sum": 0.6873529815673828, "min": 0.6873529815673828}}, "EndTime": 1584909802.105037, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.104984}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6419757080078125, "sum": 0.6419757080078125, "min": 0.6419757080078125}}, "EndTime": 1584909802.10511, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.10509}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9111833953857422, "sum": 0.9111833953857422, "min": 0.9111833953857422}}, "EndTime": 1584909802.105178, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.10516}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6262814331054688, "sum": 0.6262814331054688, "min": 0.6262814331054688}}, "EndTime": 1584909802.105245, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.105227}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.4133972930908203, "sum": 0.4133972930908203, "min": 0.4133972930908203}}, "EndTime": 1584909802.105318, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.1053}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5089419555664062, "sum": 0.5089419555664062, "min": 0.5089419555664062}}, "EndTime": 1584909802.10539, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.105372}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5032024765014649, "sum": 0.5032024765014649, "min": 0.5032024765014649}}, "EndTime": 1584909802.105457, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.105439}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5163076400756836, "sum": 0.5163076400756836, "min": 0.5163076400756836}}, "EndTime": 1584909802.105523, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.105505}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.687476577758789, "sum": 0.687476577758789, "min": 0.687476577758789}}, "EndTime": 1584909802.105587, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.10557}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9684847259521484, "sum": 0.9684847259521484, "min": 0.9684847259521484}}, "EndTime": 1584909802.105649, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.105631}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6378886032104493, "sum": 0.6378886032104493, "min": 0.6378886032104493}}, "EndTime": 1584909802.105712, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.105694}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.7832667541503906, "sum": 0.7832667541503906, "min": 0.7832667541503906}}, "EndTime": 1584909802.105784, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.105766}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.4622037124633789, "sum": 0.4622037124633789, "min": 0.4622037124633789}}, "EndTime": 1584909802.105849, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.105832}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.4564817428588867, "sum": 0.4564817428588867, "min": 0.4564817428588867}}, "EndTime": 1584909802.105905, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.10589}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.4399905014038086, "sum": 0.4399905014038086, "min": 0.4399905014038086}}, "EndTime": 1584909802.105972, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.105956}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5096223449707031, "sum": 0.5096223449707031, "min": 0.5096223449707031}}, "EndTime": 1584909802.106036, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.106019}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.7675386047363282, "sum": 0.7675386047363282, "min": 0.7675386047363282}}, "EndTime": 1584909802.106098, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.106081}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6526478576660156, "sum": 0.6526478576660156, "min": 0.6526478576660156}}, "EndTime": 1584909802.106165, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.106146}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.8872895812988282, "sum": 0.8872895812988282, "min": 0.8872895812988282}}, "EndTime": 1584909802.10623, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.106214}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.7555696105957032, "sum": 0.7555696105957032, "min": 0.7555696105957032}}, "EndTime": 1584909802.106293, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.106275}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.38698223114013675, "sum": 0.38698223114013675, "min": 0.38698223114013675}}, "EndTime": 1584909802.106347, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.106336}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5917935180664062, "sum": 0.5917935180664062, "min": 0.5917935180664062}}, "EndTime": 1584909802.106405, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.106389}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.40442256927490233, "sum": 0.40442256927490233, "min": 0.40442256927490233}}, "EndTime": 1584909802.106471, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.106454}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.40955406188964844, "sum": 0.40955406188964844, "min": 0.40955406188964844}}, "EndTime": 1584909802.106536, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.106519}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9682282257080078, "sum": 0.9682282257080078, "min": 0.9682282257080078}}, "EndTime": 1584909802.106601, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.106583}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9692342376708984, "sum": 0.9692342376708984, "min": 0.9692342376708984}}, "EndTime": 1584909802.106657, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.106645}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9841477966308594, "sum": 0.9841477966308594, "min": 0.9841477966308594}}, "EndTime": 1584909802.106698, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.106684}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9920592498779297, "sum": 0.9920592498779297, "min": 0.9920592498779297}}, "EndTime": 1584909802.106758, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.106742}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9996100616455078, "sum": 0.9996100616455078, "min": 0.9996100616455078}}, "EndTime": 1584909802.106822, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.106807}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.0855579376220703, "sum": 1.0855579376220703, "min": 1.0855579376220703}}, "EndTime": 1584909802.106884, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.106868}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.0075155639648437, "sum": 1.0075155639648437, "min": 1.0075155639648437}}, "EndTime": 1584909802.106941, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.106925}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9496178436279297, "sum": 0.9496178436279297, "min": 0.9496178436279297}}, "EndTime": 1584909802.107004, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.106987}
[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] #quality_metric: host=algo-1, epoch=7, train mse_objective <loss>=0.687352981567[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 63.52390398297991, "sum": 63.52390398297991, "min": 63.52390398297991}}, "EndTime": 1584909802.148666, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.148606}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 61.71821812220982, "sum": 61.71821812220982, "min": 61.71821812220982}}, "EndTime": 1584909802.148739, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.148725}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 83.02682059151786, "sum": 83.02682059151786, "min": 83.02682059151786}}, "EndTime": 1584909802.148796, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.148779}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 59.841238839285715, "sum": 59.841238839285715, "min": 59.841238839285715}}, "EndTime": 1584909802.148858, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.148842}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 29.311475481305802, "sum": 29.311475481305802, "min": 29.311475481305802}}, "EndTime": 1584909802.14892, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.148902}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 39.799551827566965, "sum": 39.799551827566965, "min": 39.799551827566965}}, "EndTime": 1584909802.148986, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.148969}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 38.12424577985491, "sum": 38.12424577985491, "min": 38.12424577985491}}, "EndTime": 1584909802.149045, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.149033}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 36.27350071498326, "sum": 36.27350071498326, "min": 36.27350071498326}}, "EndTime": 1584909802.14909, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.149075}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 65.98267037527901, "sum": 65.98267037527901, "min": 65.98267037527901}}, "EndTime": 1584909802.149143, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.149133}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 88.2998046875, "sum": 88.2998046875, "min": 88.2998046875}}, "EndTime": 1584909802.149179, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.149166}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 62.489270891462056, "sum": 62.489270891462056, "min": 62.489270891462056}}, "EndTime": 1584909802.149237, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.149221}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 74.66600690569196, "sum": 74.66600690569196, "min": 74.66600690569196}}, "EndTime": 1584909802.149305, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.149288}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 31.39300755092076, "sum": 31.39300755092076, "min": 31.39300755092076}}, "EndTime": 1584909802.149374, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.149355}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 31.664069039481028, "sum": 31.664069039481028, "min": 31.664069039481028}}, "EndTime": 1584909802.149444, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.149425}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 30.021619524274552, "sum": 30.021619524274552, "min": 30.021619524274552}}, "EndTime": 1584909802.149506, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.14949}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 35.83826773507254, "sum": 35.83826773507254, "min": 35.83826773507254}}, "EndTime": 1584909802.149568, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.149552}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 71.00674002511161, "sum": 71.00674002511161, "min": 71.00674002511161}}, "EndTime": 1584909802.149634, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.149617}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 60.81398228236607, "sum": 60.81398228236607, "min": 60.81398228236607}}, "EndTime": 1584909802.149696, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.149679}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 85.69838169642857, "sum": 85.69838169642857, "min": 85.69838169642857}}, "EndTime": 1584909802.149757, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.14974}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 71.21230207170758, "sum": 71.21230207170758, "min": 71.21230207170758}}, "EndTime": 1584909802.149827, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.149809}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 36.88196236746652, "sum": 36.88196236746652, "min": 36.88196236746652}}, "EndTime": 1584909802.149892, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.149876}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 48.22434343610491, "sum": 48.22434343610491, "min": 48.22434343610491}}, "EndTime": 1584909802.149958, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.149942}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 38.661795479910715, "sum": 38.661795479910715, "min": 38.661795479910715}}, "EndTime": 1584909802.150024, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.150007}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 37.86842564174107, "sum": 37.86842564174107, "min": 37.86842564174107}}, "EndTime": 1584909802.150087, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.150071}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 95.08626011439732, "sum": 95.08626011439732, "min": 95.08626011439732}}, "EndTime": 1584909802.150151, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.150135}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 93.24603271484375, "sum": 93.24603271484375, "min": 93.24603271484375}}, "EndTime": 1584909802.150216, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.150198}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 94.18504115513393, "sum": 94.18504115513393, "min": 94.18504115513393}}, "EndTime": 1584909802.150279, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.150261}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 95.45777239118304, "sum": 95.45777239118304, "min": 95.45777239118304}}, "EndTime": 1584909802.150343, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.150326}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 99.68144008091518, "sum": 99.68144008091518, "min": 99.68144008091518}}, "EndTime": 1584909802.150403, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.150387}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 108.10464913504464, "sum": 108.10464913504464, "min": 108.10464913504464}}, "EndTime": 1584909802.150464, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.150447}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 94.89310128348214, "sum": 94.89310128348214, "min": 94.89310128348214}}, "EndTime": 1584909802.150526, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.15051}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 92.7288818359375, "sum": 92.7288818359375, "min": 92.7288818359375}}, "EndTime": 1584909802.150587, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.150571}
[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] #quality_metric: host=algo-1, epoch=7, validation mse_objective <loss>=63.523903983[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] #early_stopping_criteria_metric: host=algo-1, epoch=7, criteria=mse_objective, value=29.3114754813[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] #progress_metric: host=algo-1, completed 53 % of epochs[0m
[34m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 2, "sum": 2.0, "min": 2}, "Number of Batches Since Last Reset": {"count": 1, "max": 2, "sum": 2.0, "min": 2}, "Number of Records Since Last Reset": {"count": 1, "max": 227, "sum": 227.0, "min": 227}, "Total Batches Seen": {"count": 1, "max": 19, "sum": 19.0, "min": 19}, "Total Records Seen": {"count": 1, "max": 2243, "sum": 2243.0, "min": 2243}, "Max Records Seen Between Resets": {"count": 1, "max": 227, "sum": 227.0, "min": 227}, "Reset Count": {"count": 1, "max": 10, "sum": 10.0, "min": 10}}, "EndTime": 1584909802.151438, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 7}, "StartTime": 1584909802.080685}
[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] #throughput_metric: host=algo-1, train throughput=3201.96336316 records/second[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6619586944580078, "sum": 0.6619586944580078, "min": 0.6619586944580078}}, "EndTime": 1584909802.172252, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.1722}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6237203216552735, "sum": 0.6237203216552735, "min": 0.6237203216552735}}, "EndTime": 1584909802.172323, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.172305}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.8673007202148437, "sum": 0.8673007202148437, "min": 0.8673007202148437}}, "EndTime": 1584909802.172393, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.172375}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6075299453735351, "sum": 0.6075299453735351, "min": 0.6075299453735351}}, "EndTime": 1584909802.172469, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.172449}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.4967330551147461, "sum": 0.4967330551147461, "min": 0.4967330551147461}}, "EndTime": 1584909802.17253, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.172518}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5642464447021485, "sum": 0.5642464447021485, "min": 0.5642464447021485}}, "EndTime": 1584909802.172593, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.172576}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5518309402465821, "sum": 0.5518309402465821, "min": 0.5518309402465821}}, "EndTime": 1584909802.172641, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.172625}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5478589630126953, "sum": 0.5478589630126953, "min": 0.5478589630126953}}, "EndTime": 1584909802.172703, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.172687}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6688837432861328, "sum": 0.6688837432861328, "min": 0.6688837432861328}}, "EndTime": 1584909802.172761, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.172745}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.920805435180664, "sum": 0.920805435180664, "min": 0.920805435180664}}, "EndTime": 1584909802.172821, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.172804}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6216037368774414, "sum": 0.6216037368774414, "min": 0.6216037368774414}}, "EndTime": 1584909802.172883, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.172865}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.75378662109375, "sum": 0.75378662109375, "min": 0.75378662109375}}, "EndTime": 1584909802.172944, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.172927}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.49758670806884764, "sum": 0.49758670806884764, "min": 0.49758670806884764}}, "EndTime": 1584909802.173013, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.172995}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.4991041946411133, "sum": 0.4991041946411133, "min": 0.4991041946411133}}, "EndTime": 1584909802.173076, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.173058}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5221139907836914, "sum": 0.5221139907836914, "min": 0.5221139907836914}}, "EndTime": 1584909802.17315, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.173131}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5252260208129883, "sum": 0.5252260208129883, "min": 0.5252260208129883}}, "EndTime": 1584909802.173214, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.173196}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.7390595245361328, "sum": 0.7390595245361328, "min": 0.7390595245361328}}, "EndTime": 1584909802.173274, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.173257}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6347591400146484, "sum": 0.6347591400146484, "min": 0.6347591400146484}}, "EndTime": 1584909802.173322, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.173311}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.8547866821289063, "sum": 0.8547866821289063, "min": 0.8547866821289063}}, "EndTime": 1584909802.173358, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.173345}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.7295132446289062, "sum": 0.7295132446289062, "min": 0.7295132446289062}}, "EndTime": 1584909802.173409, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.173394}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.4939820098876953, "sum": 0.4939820098876953, "min": 0.4939820098876953}}, "EndTime": 1584909802.17347, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.173453}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5970941162109376, "sum": 0.5970941162109376, "min": 0.5970941162109376}}, "EndTime": 1584909802.173531, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.173515}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.502131233215332, "sum": 0.502131233215332, "min": 0.502131233215332}}, "EndTime": 1584909802.173595, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.173577}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.4929453277587891, "sum": 0.4929453277587891, "min": 0.4929453277587891}}, "EndTime": 1584909802.173658, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.17364}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9769834899902343, "sum": 0.9769834899902343, "min": 0.9769834899902343}}, "EndTime": 1584909802.173727, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.17371}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9701864624023437, "sum": 0.9701864624023437, "min": 0.9701864624023437}}, "EndTime": 1584909802.173789, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.173772}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.96414794921875, "sum": 0.96414794921875, "min": 0.96414794921875}}, "EndTime": 1584909802.173849, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.173832}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.973936767578125, "sum": 0.973936767578125, "min": 0.973936767578125}}, "EndTime": 1584909802.173908, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.173891}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.0957870483398438, "sum": 1.0957870483398438, "min": 1.0957870483398438}}, "EndTime": 1584909802.173969, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.173951}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.2240897369384767, "sum": 1.2240897369384767, "min": 1.2240897369384767}}, "EndTime": 1584909802.174037, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.17402}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.0338111877441407, "sum": 1.0338111877441407, "min": 1.0338111877441407}}, "EndTime": 1584909802.1741, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.174082}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.999879379272461, "sum": 0.999879379272461, "min": 0.999879379272461}}, "EndTime": 1584909802.17416, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.174144}
[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] #quality_metric: host=algo-1, epoch=8, train mse_objective <loss>=0.661958694458[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 61.11305454799107, "sum": 61.11305454799107, "min": 61.11305454799107}}, "EndTime": 1584909802.213116, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.213061}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 59.736663818359375, "sum": 59.736663818359375, "min": 59.736663818359375}}, "EndTime": 1584909802.213181, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.213167}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 79.30253383091518, "sum": 79.30253383091518, "min": 79.30253383091518}}, "EndTime": 1584909802.21325, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.213231}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 57.84041050502232, "sum": 57.84041050502232, "min": 57.84041050502232}}, "EndTime": 1584909802.213344, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.213323}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 33.429918561662944, "sum": 33.429918561662944, "min": 33.429918561662944}}, "EndTime": 1584909802.213409, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.213391}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 36.043975830078125, "sum": 36.043975830078125, "min": 36.043975830078125}}, "EndTime": 1584909802.213448, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.213439}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 36.79362269810268, "sum": 36.79362269810268, "min": 36.79362269810268}}, "EndTime": 1584909802.213479, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.213471}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 31.81774684361049, "sum": 31.81774684361049, "min": 31.81774684361049}}, "EndTime": 1584909802.213555, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.213537}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 64.13298688616071, "sum": 64.13298688616071, "min": 64.13298688616071}}, "EndTime": 1584909802.213618, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.213601}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 84.3446044921875, "sum": 84.3446044921875, "min": 84.3446044921875}}, "EndTime": 1584909802.213682, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.213664}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 60.69565691266741, "sum": 60.69565691266741, "min": 60.69565691266741}}, "EndTime": 1584909802.21376, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.213726}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 72.00578090122768, "sum": 72.00578090122768, "min": 72.00578090122768}}, "EndTime": 1584909802.213821, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.213804}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 32.65591648646763, "sum": 32.65591648646763, "min": 32.65591648646763}}, "EndTime": 1584909802.21388, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.213869}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 32.54178946358817, "sum": 32.54178946358817, "min": 32.54178946358817}}, "EndTime": 1584909802.213927, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.213913}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 31.942786080496653, "sum": 31.942786080496653, "min": 31.942786080496653}}, "EndTime": 1584909802.213986, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.21397}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 36.578260149274556, "sum": 36.578260149274556, "min": 36.578260149274556}}, "EndTime": 1584909802.214053, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.214036}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 68.39553833007812, "sum": 68.39553833007812, "min": 68.39553833007812}}, "EndTime": 1584909802.214139, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.214122}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 58.991751534598215, "sum": 58.991751534598215, "min": 58.991751534598215}}, "EndTime": 1584909802.214196, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.21418}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 82.71187918526786, "sum": 82.71187918526786, "min": 82.71187918526786}}, "EndTime": 1584909802.214252, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.214236}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 68.7687508719308, "sum": 68.7687508719308, "min": 68.7687508719308}}, "EndTime": 1584909802.214313, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.214296}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 43.19414847237723, "sum": 43.19414847237723, "min": 43.19414847237723}}, "EndTime": 1584909802.214371, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.214355}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 42.236846923828125, "sum": 42.236846923828125, "min": 42.236846923828125}}, "EndTime": 1584909802.21443, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.214414}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 44.875135149274556, "sum": 44.875135149274556, "min": 44.875135149274556}}, "EndTime": 1584909802.214527, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.214509}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 41.90318080357143, "sum": 41.90318080357143, "min": 41.90318080357143}}, "EndTime": 1584909802.214581, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.214571}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 95.78766741071429, "sum": 95.78766741071429, "min": 95.78766741071429}}, "EndTime": 1584909802.214636, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.21462}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 93.13943917410714, "sum": 93.13943917410714, "min": 93.13943917410714}}, "EndTime": 1584909802.214686, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.214673}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 92.73975481305804, "sum": 92.73975481305804, "min": 92.73975481305804}}, "EndTime": 1584909802.214752, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.214735}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 94.10280936104911, "sum": 94.10280936104911, "min": 94.10280936104911}}, "EndTime": 1584909802.214816, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.214799}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 99.18069893973214, "sum": 99.18069893973214, "min": 99.18069893973214}}, "EndTime": 1584909802.214879, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.214862}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 104.65175955636161, "sum": 104.65175955636161, "min": 104.65175955636161}}, "EndTime": 1584909802.21494, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.214923}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 92.28945486886161, "sum": 92.28945486886161, "min": 92.28945486886161}}, "EndTime": 1584909802.215001, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.214985}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 92.15741838727679, "sum": 92.15741838727679, "min": 92.15741838727679}}, "EndTime": 1584909802.215064, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.215047}
[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] #quality_metric: host=algo-1, epoch=8, validation mse_objective <loss>=61.113054548[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] #early_stopping_criteria_metric: host=algo-1, epoch=8, criteria=mse_objective, value=31.8177468436[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] #progress_metric: host=algo-1, completed 60 % of epochs[0m
[34m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 2, "sum": 2.0, "min": 2}, "Number of Batches Since Last Reset": {"count": 1, "max": 2, "sum": 2.0, "min": 2}, "Number of Records Since Last Reset": {"count": 1, "max": 227, "sum": 227.0, "min": 227}, "Total Batches Seen": {"count": 1, "max": 21, "sum": 21.0, "min": 21}, "Total Records Seen": {"count": 1, "max": 2470, "sum": 2470.0, "min": 2470}, "Max Records Seen Between Resets": {"count": 1, "max": 227, "sum": 227.0, "min": 227}, "Reset Count": {"count": 1, "max": 11, "sum": 11.0, "min": 11}}, "EndTime": 1584909802.215905, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 8}, "StartTime": 1584909802.151698}
[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] #throughput_metric: host=algo-1, train throughput=3529.0670818 records/second[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6398678970336914, "sum": 0.6398678970336914, "min": 0.6398678970336914}}, "EndTime": 1584909802.23995, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.239899}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6083224868774414, "sum": 0.6083224868774414, "min": 0.6083224868774414}}, "EndTime": 1584909802.240023, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.240003}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.8270172119140625, "sum": 0.8270172119140625, "min": 0.8270172119140625}}, "EndTime": 1584909802.240092, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.240074}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5917279434204101, "sum": 0.5917279434204101, "min": 0.5917279434204101}}, "EndTime": 1584909802.24016, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.240143}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5603557968139649, "sum": 0.5603557968139649, "min": 0.5603557968139649}}, "EndTime": 1584909802.240217, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.240201}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5437210845947266, "sum": 0.5437210845947266, "min": 0.5437210845947266}}, "EndTime": 1584909802.240277, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.240261}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5674553680419921, "sum": 0.5674553680419921, "min": 0.5674553680419921}}, "EndTime": 1584909802.240338, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.240321}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5082304382324219, "sum": 0.5082304382324219, "min": 0.5082304382324219}}, "EndTime": 1584909802.240404, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.240388}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6528994750976562, "sum": 0.6528994750976562, "min": 0.6528994750976562}}, "EndTime": 1584909802.240467, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.24045}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.8766929626464843, "sum": 0.8766929626464843, "min": 0.8766929626464843}}, "EndTime": 1584909802.240525, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.240508}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6074574661254882, "sum": 0.6074574661254882, "min": 0.6074574661254882}}, "EndTime": 1584909802.240585, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.240569}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.7275556182861328, "sum": 0.7275556182861328, "min": 0.7275556182861328}}, "EndTime": 1584909802.240646, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.240629}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5432647323608398, "sum": 0.5432647323608398, "min": 0.5432647323608398}}, "EndTime": 1584909802.240711, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.240693}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5339703369140625, "sum": 0.5339703369140625, "min": 0.5339703369140625}}, "EndTime": 1584909802.240774, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.240756}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5545736312866211, "sum": 0.5545736312866211, "min": 0.5545736312866211}}, "EndTime": 1584909802.240834, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.240817}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5685578918457032, "sum": 0.5685578918457032, "min": 0.5685578918457032}}, "EndTime": 1584909802.240895, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.240877}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.7138057708740234, "sum": 0.7138057708740234, "min": 0.7138057708740234}}, "EndTime": 1584909802.240955, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.240938}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.618712158203125, "sum": 0.618712158203125, "min": 0.618712158203125}}, "EndTime": 1584909802.241013, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.240997}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.8256238555908203, "sum": 0.8256238555908203, "min": 0.8256238555908203}}, "EndTime": 1584909802.241067, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.241053}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.7066919708251953, "sum": 0.7066919708251953, "min": 0.7066919708251953}}, "EndTime": 1584909802.241119, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.241106}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.57400146484375, "sum": 0.57400146484375, "min": 0.57400146484375}}, "EndTime": 1584909802.241178, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.241161}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5423157501220703, "sum": 0.5423157501220703, "min": 0.5423157501220703}}, "EndTime": 1584909802.24123, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.241214}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5837326049804688, "sum": 0.5837326049804688, "min": 0.5837326049804688}}, "EndTime": 1584909802.241298, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.24128}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5400738143920898, "sum": 0.5400738143920898, "min": 0.5400738143920898}}, "EndTime": 1584909802.241359, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.241343}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9846402740478516, "sum": 0.9846402740478516, "min": 0.9846402740478516}}, "EndTime": 1584909802.241422, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.241404}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9706926727294922, "sum": 0.9706926727294922, "min": 0.9706926727294922}}, "EndTime": 1584909802.241485, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.241467}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9482324981689453, "sum": 0.9482324981689453, "min": 0.9482324981689453}}, "EndTime": 1584909802.241546, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.24153}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9597693634033203, "sum": 0.9597693634033203, "min": 0.9597693634033203}}, "EndTime": 1584909802.241653, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.241632}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.0963416290283203, "sum": 1.0963416290283203, "min": 1.0963416290283203}}, "EndTime": 1584909802.24172, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.241703}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.1926799774169923, "sum": 1.1926799774169923, "min": 1.1926799774169923}}, "EndTime": 1584909802.241782, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.241765}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.0104657745361327, "sum": 1.0104657745361327, "min": 1.0104657745361327}}, "EndTime": 1584909802.241843, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.241827}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.0075543212890625, "sum": 1.0075543212890625, "min": 1.0075543212890625}}, "EndTime": 1584909802.2419, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.241885}
[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] #quality_metric: host=algo-1, epoch=9, train mse_objective <loss>=0.639867897034[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 58.930877685546875, "sum": 58.930877685546875, "min": 58.930877685546875}}, "EndTime": 1584909802.287015, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.286958}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 57.93736049107143, "sum": 57.93736049107143, "min": 57.93736049107143}}, "EndTime": 1584909802.287089, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.287068}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 75.88851492745536, "sum": 75.88851492745536, "min": 75.88851492745536}}, "EndTime": 1584909802.287183, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.287162}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 56.04111589704241, "sum": 56.04111589704241, "min": 56.04111589704241}}, "EndTime": 1584909802.287252, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.287233}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 29.089013235909597, "sum": 29.089013235909597, "min": 29.089013235909597}}, "EndTime": 1584909802.287317, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.2873}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 28.08976309640067, "sum": 28.08976309640067, "min": 28.08976309640067}}, "EndTime": 1584909802.287385, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.287367}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 30.772761753627233, "sum": 30.772761753627233, "min": 30.772761753627233}}, "EndTime": 1584909802.287448, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.287431}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 25.404325212751115, "sum": 25.404325212751115, "min": 25.404325212751115}}, "EndTime": 1584909802.287529, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.287511}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 62.47886003766741, "sum": 62.47886003766741, "min": 62.47886003766741}}, "EndTime": 1584909802.287616, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.287597}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 80.72188023158482, "sum": 80.72188023158482, "min": 80.72188023158482}}, "EndTime": 1584909802.287674, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.287663}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 59.04706246512277, "sum": 59.04706246512277, "min": 59.04706246512277}}, "EndTime": 1584909802.287729, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.287712}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 69.57472011021206, "sum": 69.57472011021206, "min": 69.57472011021206}}, "EndTime": 1584909802.28779, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.287773}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 29.479862758091517, "sum": 29.479862758091517, "min": 29.479862758091517}}, "EndTime": 1584909802.28786, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.287842}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 29.280689784458705, "sum": 29.280689784458705, "min": 29.280689784458705}}, "EndTime": 1584909802.287931, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.287913}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 26.212642124720983, "sum": 26.212642124720983, "min": 26.212642124720983}}, "EndTime": 1584909802.287992, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.287976}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 33.008579799107146, "sum": 33.008579799107146, "min": 33.008579799107146}}, "EndTime": 1584909802.288055, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.288037}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 66.04335675920758, "sum": 66.04335675920758, "min": 66.04335675920758}}, "EndTime": 1584909802.288126, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.288108}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 57.30853271484375, "sum": 57.30853271484375, "min": 57.30853271484375}}, "EndTime": 1584909802.288188, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.288172}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 79.97156633649554, "sum": 79.97156633649554, "min": 79.97156633649554}}, "EndTime": 1584909802.288252, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.288234}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 66.56854684012276, "sum": 66.56854684012276, "min": 66.56854684012276}}, "EndTime": 1584909802.288316, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.288298}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 39.63690621512277, "sum": 39.63690621512277, "min": 39.63690621512277}}, "EndTime": 1584909802.288378, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.288361}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 34.26336669921875, "sum": 34.26336669921875, "min": 34.26336669921875}}, "EndTime": 1584909802.288438, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.288422}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 40.901293073381694, "sum": 40.901293073381694, "min": 40.901293073381694}}, "EndTime": 1584909802.288509, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.28849}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 39.38790021623884, "sum": 39.38790021623884, "min": 39.38790021623884}}, "EndTime": 1584909802.288576, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.288558}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 96.1993408203125, "sum": 96.1993408203125, "min": 96.1993408203125}}, "EndTime": 1584909802.288638, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.288621}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 93.10551234654018, "sum": 93.10551234654018, "min": 93.10551234654018}}, "EndTime": 1584909802.288699, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.288683}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 91.52680315290179, "sum": 91.52680315290179, "min": 91.52680315290179}}, "EndTime": 1584909802.288761, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.288744}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 93.15284075055804, "sum": 93.15284075055804, "min": 93.15284075055804}}, "EndTime": 1584909802.288825, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.288808}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 92.82329450334821, "sum": 92.82329450334821, "min": 92.82329450334821}}, "EndTime": 1584909802.288887, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.28887}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 94.290771484375, "sum": 94.290771484375, "min": 94.290771484375}}, "EndTime": 1584909802.288951, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.288935}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 86.55885532924107, "sum": 86.55885532924107, "min": 86.55885532924107}}, "EndTime": 1584909802.289012, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.288996}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 90.42560686383929, "sum": 90.42560686383929, "min": 90.42560686383929}}, "EndTime": 1584909802.289073, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.289056}
[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] #quality_metric: host=algo-1, epoch=9, validation mse_objective <loss>=58.9308776855[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] #early_stopping_criteria_metric: host=algo-1, epoch=9, criteria=mse_objective, value=25.4043252128[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] Epoch 9: Loss improved. Updating best model[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] #progress_metric: host=algo-1, completed 66 % of epochs[0m
[34m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 2, "sum": 2.0, "min": 2}, "Number of Batches Since Last Reset": {"count": 1, "max": 2, "sum": 2.0, "min": 2}, "Number of Records Since Last Reset": {"count": 1, "max": 227, "sum": 227.0, "min": 227}, "Total Batches Seen": {"count": 1, "max": 23, "sum": 23.0, "min": 23}, "Total Records Seen": {"count": 1, "max": 2697, "sum": 2697.0, "min": 2697}, "Max Records Seen Between Resets": {"count": 1, "max": 227, "sum": 227.0, "min": 227}, "Reset Count": {"count": 1, "max": 12, "sum": 12.0, "min": 12}}, "EndTime": 1584909802.29066, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 9}, "StartTime": 1584909802.216143}
[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] #throughput_metric: host=algo-1, train throughput=3041.63248303 records/second[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6205531311035156, "sum": 0.6205531311035156, "min": 0.6205531311035156}}, "EndTime": 1584909802.317496, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.317444}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5949491882324218, "sum": 0.5949491882324218, "min": 0.5949491882324218}}, "EndTime": 1584909802.317558, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.317545}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.7900350952148437, "sum": 0.7900350952148437, "min": 0.7900350952148437}}, "EndTime": 1584909802.317623, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.317605}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5782740020751953, "sum": 0.5782740020751953, "min": 0.5782740020751953}}, "EndTime": 1584909802.317692, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.317674}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.4840725326538086, "sum": 0.4840725326538086, "min": 0.4840725326538086}}, "EndTime": 1584909802.31776, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.317742}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.4449246215820313, "sum": 0.4449246215820313, "min": 0.4449246215820313}}, "EndTime": 1584909802.317824, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.317806}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.49679027557373046, "sum": 0.49679027557373046, "min": 0.49679027557373046}}, "EndTime": 1584909802.317897, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.31788}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.42694812774658203, "sum": 0.42694812774658203, "min": 0.42694812774658203}}, "EndTime": 1584909802.31796, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.317943}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6391589736938477, "sum": 0.6391589736938477, "min": 0.6391589736938477}}, "EndTime": 1584909802.318019, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.318003}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.8359712982177734, "sum": 0.8359712982177734, "min": 0.8359712982177734}}, "EndTime": 1584909802.318082, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.318065}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5950908279418945, "sum": 0.5950908279418945, "min": 0.5950908279418945}}, "EndTime": 1584909802.318144, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.318128}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.7041321563720703, "sum": 0.7041321563720703, "min": 0.7041321563720703}}, "EndTime": 1584909802.318201, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.318185}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5029395294189453, "sum": 0.5029395294189453, "min": 0.5029395294189453}}, "EndTime": 1584909802.318256, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.318242}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.4893633270263672, "sum": 0.4893633270263672, "min": 0.4893633270263672}}, "EndTime": 1584909802.318313, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.318298}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.46153812408447265, "sum": 0.46153812408447265, "min": 0.46153812408447265}}, "EndTime": 1584909802.318381, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.318364}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5262836837768554, "sum": 0.5262836837768554, "min": 0.5262836837768554}}, "EndTime": 1584909802.31844, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.318424}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.691524887084961, "sum": 0.691524887084961, "min": 0.691524887084961}}, "EndTime": 1584909802.318501, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.318484}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6042334365844727, "sum": 0.6042334365844727, "min": 0.6042334365844727}}, "EndTime": 1584909802.318558, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.318543}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.7993580627441407, "sum": 0.7993580627441407, "min": 0.7993580627441407}}, "EndTime": 1584909802.318615, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.318599}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6863380432128906, "sum": 0.6863380432128906, "min": 0.6863380432128906}}, "EndTime": 1584909802.318676, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.318659}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.4947404098510742, "sum": 0.4947404098510742, "min": 0.4947404098510742}}, "EndTime": 1584909802.318736, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.318719}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.4454799270629883, "sum": 0.4454799270629883, "min": 0.4454799270629883}}, "EndTime": 1584909802.318798, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.318781}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5063280868530273, "sum": 0.5063280868530273, "min": 0.5063280868530273}}, "EndTime": 1584909802.318861, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.318844}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.47918933868408203, "sum": 0.47918933868408203, "min": 0.47918933868408203}}, "EndTime": 1584909802.31892, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.318904}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9900666809082032, "sum": 0.9900666809082032, "min": 0.9900666809082032}}, "EndTime": 1584909802.318979, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.318963}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9706609344482422, "sum": 0.9706609344482422, "min": 0.9706609344482422}}, "EndTime": 1584909802.319042, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.319024}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9357696533203125, "sum": 0.9357696533203125, "min": 0.9357696533203125}}, "EndTime": 1584909802.319102, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.319086}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9497825622558593, "sum": 0.9497825622558593, "min": 0.9497825622558593}}, "EndTime": 1584909802.31916, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.319145}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9921808624267578, "sum": 0.9921808624267578, "min": 0.9921808624267578}}, "EndTime": 1584909802.319226, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.319209}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.0254523468017578, "sum": 1.0254523468017578, "min": 1.0254523468017578}}, "EndTime": 1584909802.319287, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.31927}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9157390594482422, "sum": 0.9157390594482422, "min": 0.9157390594482422}}, "EndTime": 1584909802.319345, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.319329}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9614736938476562, "sum": 0.9614736938476562, "min": 0.9614736938476562}}, "EndTime": 1584909802.319403, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.319388}
[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] #quality_metric: host=algo-1, epoch=10, train mse_objective <loss>=0.620553131104[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 56.981070382254465, "sum": 56.981070382254465, "min": 56.981070382254465}}, "EndTime": 1584909802.361153, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.361097}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 56.292140415736604, "sum": 56.292140415736604, "min": 56.292140415736604}}, "EndTime": 1584909802.361226, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.361205}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 72.79122924804688, "sum": 72.79122924804688, "min": 72.79122924804688}}, "EndTime": 1584909802.361295, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.361277}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 54.424918038504465, "sum": 54.424918038504465, "min": 54.424918038504465}}, "EndTime": 1584909802.361357, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.361345}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 24.843329293387278, "sum": 24.843329293387278, "min": 24.843329293387278}}, "EndTime": 1584909802.36142, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.361403}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 24.286117553710938, "sum": 24.286117553710938, "min": 24.286117553710938}}, "EndTime": 1584909802.36149, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.361471}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 26.357757568359375, "sum": 26.357757568359375, "min": 26.357757568359375}}, "EndTime": 1584909802.361561, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.361542}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 24.473181588309153, "sum": 24.473181588309153, "min": 24.473181588309153}}, "EndTime": 1584909802.36163, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.361612}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 61.009303501674104, "sum": 61.009303501674104, "min": 61.009303501674104}}, "EndTime": 1584909802.361694, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.361676}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 77.3984375, "sum": 77.3984375, "min": 77.3984375}}, "EndTime": 1584909802.361753, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.361737}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 57.555751255580354, "sum": 57.555751255580354, "min": 57.555751255580354}}, "EndTime": 1584909802.361812, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.361796}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 67.35762677873883, "sum": 67.35762677873883, "min": 67.35762677873883}}, "EndTime": 1584909802.361873, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.361856}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 25.770863124302455, "sum": 25.770863124302455, "min": 25.770863124302455}}, "EndTime": 1584909802.361934, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.361918}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 26.400360107421875, "sum": 26.400360107421875, "min": 26.400360107421875}}, "EndTime": 1584909802.362003, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.361985}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 22.942620413643972, "sum": 22.942620413643972, "min": 22.942620413643972}}, "EndTime": 1584909802.362067, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.36205}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 28.011361258370535, "sum": 28.011361258370535, "min": 28.011361258370535}}, "EndTime": 1584909802.362141, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.362121}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 63.95949445452009, "sum": 63.95949445452009, "min": 63.95949445452009}}, "EndTime": 1584909802.362213, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.362195}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 55.76385934012277, "sum": 55.76385934012277, "min": 55.76385934012277}}, "EndTime": 1584909802.362285, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.362267}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 77.48830740792411, "sum": 77.48830740792411, "min": 77.48830740792411}}, "EndTime": 1584909802.362346, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.362329}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 64.57713099888393, "sum": 64.57713099888393, "min": 64.57713099888393}}, "EndTime": 1584909802.362409, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.362394}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 37.56445748465402, "sum": 37.56445748465402, "min": 37.56445748465402}}, "EndTime": 1584909802.362469, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.362452}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 31.158782958984375, "sum": 31.158782958984375, "min": 31.158782958984375}}, "EndTime": 1584909802.362534, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.362517}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 37.771510532924104, "sum": 37.771510532924104, "min": 37.771510532924104}}, "EndTime": 1584909802.362596, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.362579}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 40.50040544782366, "sum": 40.50040544782366, "min": 40.50040544782366}}, "EndTime": 1584909802.362665, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.362648}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 96.34881591796875, "sum": 96.34881591796875, "min": 96.34881591796875}}, "EndTime": 1584909802.362726, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.362708}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 93.14202880859375, "sum": 93.14202880859375, "min": 93.14202880859375}}, "EndTime": 1584909802.362792, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.362774}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 90.55921282087054, "sum": 90.55921282087054, "min": 90.55921282087054}}, "EndTime": 1584909802.362861, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.362844}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 92.55637904575893, "sum": 92.55637904575893, "min": 92.55637904575893}}, "EndTime": 1584909802.362922, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.362904}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 90.56552559988839, "sum": 90.56552559988839, "min": 90.56552559988839}}, "EndTime": 1584909802.362983, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.362966}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 88.18352399553571, "sum": 88.18352399553571, "min": 88.18352399553571}}, "EndTime": 1584909802.363046, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.36303}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 86.59818812779018, "sum": 86.59818812779018, "min": 86.59818812779018}}, "EndTime": 1584909802.363105, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.363089}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 94.16473388671875, "sum": 94.16473388671875, "min": 94.16473388671875}}, "EndTime": 1584909802.363168, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.363151}
[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] #quality_metric: host=algo-1, epoch=10, validation mse_objective <loss>=56.9810703823[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] #early_stopping_criteria_metric: host=algo-1, epoch=10, criteria=mse_objective, value=22.9426204136[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] Epoch 10: Loss improved. Updating best model[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] #progress_metric: host=algo-1, completed 73 % of epochs[0m
[34m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 2, "sum": 2.0, "min": 2}, "Number of Batches Since Last Reset": {"count": 1, "max": 2, "sum": 2.0, "min": 2}, "Number of Records Since Last Reset": {"count": 1, "max": 227, "sum": 227.0, "min": 227}, "Total Batches Seen": {"count": 1, "max": 25, "sum": 25.0, "min": 25}, "Total Records Seen": {"count": 1, "max": 2924, "sum": 2924.0, "min": 2924}, "Max Records Seen Between Resets": {"count": 1, "max": 227, "sum": 227.0, "min": 227}, "Reset Count": {"count": 1, "max": 13, "sum": 13.0, "min": 13}}, "EndTime": 1584909802.364783, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 10}, "StartTime": 1584909802.290916}
[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] #throughput_metric: host=algo-1, train throughput=3067.80968829 records/second[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6041836166381835, "sum": 0.6041836166381835, "min": 0.6041836166381835}}, "EndTime": 1584909802.391643, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.391592}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5833802795410157, "sum": 0.5833802795410157, "min": 0.5833802795410157}}, "EndTime": 1584909802.391707, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.391693}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.756895523071289, "sum": 0.756895523071289, "min": 0.756895523071289}}, "EndTime": 1584909802.391778, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.39176}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5670868682861329, "sum": 0.5670868682861329, "min": 0.5670868682861329}}, "EndTime": 1584909802.39185, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.391832}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.38706031799316404, "sum": 0.38706031799316404, "min": 0.38706031799316404}}, "EndTime": 1584909802.391913, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.391896}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.4013888549804687, "sum": 0.4013888549804687, "min": 0.4013888549804687}}, "EndTime": 1584909802.391976, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.391958}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.4420859146118164, "sum": 0.4420859146118164, "min": 0.4420859146118164}}, "EndTime": 1584909802.392047, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.392028}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.41698905944824216, "sum": 0.41698905944824216, "min": 0.41698905944824216}}, "EndTime": 1584909802.392115, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.392097}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6276490783691406, "sum": 0.6276490783691406, "min": 0.6276490783691406}}, "EndTime": 1584909802.392179, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.392162}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.7988988494873047, "sum": 0.7988988494873047, "min": 0.7988988494873047}}, "EndTime": 1584909802.392231, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.39222}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5846038818359375, "sum": 0.5846038818359375, "min": 0.5846038818359375}}, "EndTime": 1584909802.392268, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.392255}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6835724639892579, "sum": 0.6835724639892579, "min": 0.6835724639892579}}, "EndTime": 1584909802.392336, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.392318}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.43949165344238283, "sum": 0.43949165344238283, "min": 0.43949165344238283}}, "EndTime": 1584909802.3924, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.392383}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.43723087310791015, "sum": 0.43723087310791015, "min": 0.43723087310791015}}, "EndTime": 1584909802.392466, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.392449}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.37810245513916013, "sum": 0.37810245513916013, "min": 0.37810245513916013}}, "EndTime": 1584909802.39253, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.392514}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.4534333419799805, "sum": 0.4534333419799805, "min": 0.4534333419799805}}, "EndTime": 1584909802.392585, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.392568}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6724369049072265, "sum": 0.6724369049072265, "min": 0.6724369049072265}}, "EndTime": 1584909802.392642, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.392627}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5913552093505859, "sum": 0.5913552093505859, "min": 0.5913552093505859}}, "EndTime": 1584909802.392705, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.392689}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.7762496948242188, "sum": 0.7762496948242188, "min": 0.7762496948242188}}, "EndTime": 1584909802.392772, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.392755}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6682780456542968, "sum": 0.6682780456542968, "min": 0.6682780456542968}}, "EndTime": 1584909802.392834, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.392817}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.4152138137817383, "sum": 0.4152138137817383, "min": 0.4152138137817383}}, "EndTime": 1584909802.392893, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.392877}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.41393264770507815, "sum": 0.41393264770507815, "min": 0.41393264770507815}}, "EndTime": 1584909802.392953, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.392936}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.4237185287475586, "sum": 0.4237185287475586, "min": 0.4237185287475586}}, "EndTime": 1584909802.393004, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.392989}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.45121639251708984, "sum": 0.45121639251708984, "min": 0.45121639251708984}}, "EndTime": 1584909802.393063, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.393046}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9932676696777344, "sum": 0.9932676696777344, "min": 0.9932676696777344}}, "EndTime": 1584909802.393129, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.393112}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9704633331298829, "sum": 0.9704633331298829, "min": 0.9704633331298829}}, "EndTime": 1584909802.393188, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.393171}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9266211700439453, "sum": 0.9266211700439453, "min": 0.9266211700439453}}, "EndTime": 1584909802.393242, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.393226}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9438592529296875, "sum": 0.9438592529296875, "min": 0.9438592529296875}}, "EndTime": 1584909802.393303, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.393286}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9352621459960937, "sum": 0.9352621459960937, "min": 0.9352621459960937}}, "EndTime": 1584909802.39336, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.393344}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9096284484863282, "sum": 0.9096284484863282, "min": 0.9096284484863282}}, "EndTime": 1584909802.393419, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.393403}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.8861485290527343, "sum": 0.8861485290527343, "min": 0.8861485290527343}}, "EndTime": 1584909802.39348, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.393464}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.97635009765625, "sum": 0.97635009765625, "min": 0.97635009765625}}, "EndTime": 1584909802.39354, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.393524}
[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] #quality_metric: host=algo-1, epoch=11, train mse_objective <loss>=0.604183616638[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 55.23768833705357, "sum": 55.23768833705357, "min": 55.23768833705357}}, "EndTime": 1584909802.439501, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.439389}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 54.79616001674107, "sum": 54.79616001674107, "min": 54.79616001674107}}, "EndTime": 1584909802.439596, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.439573}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 69.95530918666294, "sum": 69.95530918666294, "min": 69.95530918666294}}, "EndTime": 1584909802.439679, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.439659}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 52.983084542410715, "sum": 52.983084542410715, "min": 52.983084542410715}}, "EndTime": 1584909802.439758, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.439738}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 29.126534598214285, "sum": 29.126534598214285, "min": 29.126534598214285}}, "EndTime": 1584909802.439824, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.439806}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 27.42478506905692, "sum": 27.42478506905692, "min": 27.42478506905692}}, "EndTime": 1584909802.439898, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.43988}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 27.678490774972097, "sum": 27.678490774972097, "min": 27.678490774972097}}, "EndTime": 1584909802.439967, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.439949}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 29.186309814453125, "sum": 29.186309814453125, "min": 29.186309814453125}}, "EndTime": 1584909802.44003, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.440011}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 59.674734933035715, "sum": 59.674734933035715, "min": 59.674734933035715}}, "EndTime": 1584909802.440099, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.440081}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 74.33675711495536, "sum": 74.33675711495536, "min": 74.33675711495536}}, "EndTime": 1584909802.44016, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.440145}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 56.19694301060268, "sum": 56.19694301060268, "min": 56.19694301060268}}, "EndTime": 1584909802.440217, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.440201}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 65.34561157226562, "sum": 65.34561157226562, "min": 65.34561157226562}}, "EndTime": 1584909802.44028, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.440261}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 27.07244655064174, "sum": 27.07244655064174, "min": 27.07244655064174}}, "EndTime": 1584909802.440345, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.440327}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 28.49771990094866, "sum": 28.49771990094866, "min": 28.49771990094866}}, "EndTime": 1584909802.440418, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.440399}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 28.326533726283483, "sum": 28.326533726283483, "min": 28.326533726283483}}, "EndTime": 1584909802.44049, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.440472}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 27.097457885742188, "sum": 27.097457885742188, "min": 27.097457885742188}}, "EndTime": 1584909802.440559, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.440541}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 62.10130092075893, "sum": 62.10130092075893, "min": 62.10130092075893}}, "EndTime": 1584909802.44062, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.440604}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 54.346849714006694, "sum": 54.346849714006694, "min": 54.346849714006694}}, "EndTime": 1584909802.440683, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.440665}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 75.22535051618304, "sum": 75.22535051618304, "min": 75.22535051618304}}, "EndTime": 1584909802.440746, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.440729}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 62.77740914481027, "sum": 62.77740914481027, "min": 62.77740914481027}}, "EndTime": 1584909802.440807, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.44079}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 45.637991768973215, "sum": 45.637991768973215, "min": 45.637991768973215}}, "EndTime": 1584909802.440868, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.440853}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 34.59914071219308, "sum": 34.59914071219308, "min": 34.59914071219308}}, "EndTime": 1584909802.440929, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.440911}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 44.83282470703125, "sum": 44.83282470703125, "min": 44.83282470703125}}, "EndTime": 1584909802.440999, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.44098}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 49.511805943080354, "sum": 49.511805943080354, "min": 49.511805943080354}}, "EndTime": 1584909802.441059, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.441043}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 96.31849016462054, "sum": 96.31849016462054, "min": 96.31849016462054}}, "EndTime": 1584909802.44112, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.441101}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 93.22580392020089, "sum": 93.22580392020089, "min": 93.22580392020089}}, "EndTime": 1584909802.441191, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.441172}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 89.84182303292411, "sum": 89.84182303292411, "min": 89.84182303292411}}, "EndTime": 1584909802.441255, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.441237}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 92.20835658482143, "sum": 92.20835658482143, "min": 92.20835658482143}}, "EndTime": 1584909802.441315, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.441298}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 95.51175362723214, "sum": 95.51175362723214, "min": 95.51175362723214}}, "EndTime": 1584909802.441385, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.441366}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 88.33794294084821, "sum": 88.33794294084821, "min": 88.33794294084821}}, "EndTime": 1584909802.44145, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.441431}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 95.06767926897321, "sum": 95.06767926897321, "min": 95.06767926897321}}, "EndTime": 1584909802.441521, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.441502}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 101.7872314453125, "sum": 101.7872314453125, "min": 101.7872314453125}}, "EndTime": 1584909802.441593, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.441574}
[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] #quality_metric: host=algo-1, epoch=11, validation mse_objective <loss>=55.2376883371[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] #early_stopping_criteria_metric: host=algo-1, epoch=11, criteria=mse_objective, value=27.0724465506[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] #progress_metric: host=algo-1, completed 80 % of epochs[0m
[34m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 2, "sum": 2.0, "min": 2}, "Number of Batches Since Last Reset": {"count": 1, "max": 2, "sum": 2.0, "min": 2}, "Number of Records Since Last Reset": {"count": 1, "max": 227, "sum": 227.0, "min": 227}, "Total Batches Seen": {"count": 1, "max": 27, "sum": 27.0, "min": 27}, "Total Records Seen": {"count": 1, "max": 3151, "sum": 3151.0, "min": 3151}, "Max Records Seen Between Resets": {"count": 1, "max": 227, "sum": 227.0, "min": 227}, "Reset Count": {"count": 1, "max": 14, "sum": 14.0, "min": 14}}, "EndTime": 1584909802.44266, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 11}, "StartTime": 1584909802.365007}
[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] #throughput_metric: host=algo-1, train throughput=2918.07627215 records/second[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5905584716796874, "sum": 0.5905584716796874, "min": 0.5905584716796874}}, "EndTime": 1584909802.484304, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.48425}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5735742950439453, "sum": 0.5735742950439453, "min": 0.5735742950439453}}, "EndTime": 1584909802.484407, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.484389}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.727352523803711, "sum": 0.727352523803711, "min": 0.727352523803711}}, "EndTime": 1584909802.484472, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.484454}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5580436706542968, "sum": 0.5580436706542968, "min": 0.5580436706542968}}, "EndTime": 1584909802.484531, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.484514}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.3863252639770508, "sum": 0.3863252639770508, "min": 0.3863252639770508}}, "EndTime": 1584909802.484584, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.484569}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.449802131652832, "sum": 0.449802131652832, "min": 0.449802131652832}}, "EndTime": 1584909802.484646, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.484629}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.45942226409912107, "sum": 0.45942226409912107, "min": 0.45942226409912107}}, "EndTime": 1584909802.484711, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.484694}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.47144691467285155, "sum": 0.47144691467285155, "min": 0.47144691467285155}}, "EndTime": 1584909802.484767, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.484751}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6178178405761718, "sum": 0.6178178405761718, "min": 0.6178178405761718}}, "EndTime": 1584909802.484829, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.484812}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.765548095703125, "sum": 0.765548095703125, "min": 0.765548095703125}}, "EndTime": 1584909802.484905, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.48487}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.575626106262207, "sum": 0.575626106262207, "min": 0.575626106262207}}, "EndTime": 1584909802.484969, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.484952}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6658393859863281, "sum": 0.6658393859863281, "min": 0.6658393859863281}}, "EndTime": 1584909802.48503, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.485014}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.42561561584472657, "sum": 0.42561561584472657, "min": 0.42561561584472657}}, "EndTime": 1584909802.485089, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.485072}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.43662372589111326, "sum": 0.43662372589111326, "min": 0.43662372589111326}}, "EndTime": 1584909802.485159, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.485141}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.390457878112793, "sum": 0.390457878112793, "min": 0.390457878112793}}, "EndTime": 1584909802.485221, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.485205}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.42835708618164064, "sum": 0.42835708618164064, "min": 0.42835708618164064}}, "EndTime": 1584909802.485281, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.485264}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6560523986816407, "sum": 0.6560523986816407, "min": 0.6560523986816407}}, "EndTime": 1584909802.485342, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.485325}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5798511123657226, "sum": 0.5798511123657226, "min": 0.5798511123657226}}, "EndTime": 1584909802.4854, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.485383}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.7559560394287109, "sum": 0.7559560394287109, "min": 0.7559560394287109}}, "EndTime": 1584909802.48546, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.485443}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6524893951416015, "sum": 0.6524893951416015, "min": 0.6524893951416015}}, "EndTime": 1584909802.485519, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.485508}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.46635391235351564, "sum": 0.46635391235351564, "min": 0.46635391235351564}}, "EndTime": 1584909802.485594, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.48556}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.4599029159545898, "sum": 0.4599029159545898, "min": 0.4599029159545898}}, "EndTime": 1584909802.485666, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.485648}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.4710122299194336, "sum": 0.4710122299194336, "min": 0.4710122299194336}}, "EndTime": 1584909802.485727, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.485711}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.523547248840332, "sum": 0.523547248840332, "min": 0.523547248840332}}, "EndTime": 1584909802.485785, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.48577}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9947950744628906, "sum": 0.9947950744628906, "min": 0.9947950744628906}}, "EndTime": 1584909802.485846, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.48583}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9703578186035157, "sum": 0.9703578186035157, "min": 0.9703578186035157}}, "EndTime": 1584909802.485906, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.48589}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9205416107177734, "sum": 0.9205416107177734, "min": 0.9205416107177734}}, "EndTime": 1584909802.485973, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.485956}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9412248229980469, "sum": 0.9412248229980469, "min": 0.9412248229980469}}, "EndTime": 1584909802.486034, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.486016}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9771615600585938, "sum": 0.9771615600585938, "min": 0.9771615600585938}}, "EndTime": 1584909802.486097, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.486079}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.8943132019042969, "sum": 0.8943132019042969, "min": 0.8943132019042969}}, "EndTime": 1584909802.486156, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.48614}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9710855865478516, "sum": 0.9710855865478516, "min": 0.9710855865478516}}, "EndTime": 1584909802.486218, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.486202}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.0559449005126953, "sum": 1.0559449005126953, "min": 1.0559449005126953}}, "EndTime": 1584909802.486275, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.486259}
[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] #quality_metric: host=algo-1, epoch=12, train mse_objective <loss>=0.59055847168[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 53.68372017996652, "sum": 53.68372017996652, "min": 53.68372017996652}}, "EndTime": 1584909802.541404, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.54131}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 53.43624005998884, "sum": 53.43624005998884, "min": 53.43624005998884}}, "EndTime": 1584909802.541564, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.54154}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 67.347900390625, "sum": 67.347900390625, "min": 67.347900390625}}, "EndTime": 1584909802.541634, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.541616}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 51.692112513950896, "sum": 51.692112513950896, "min": 51.692112513950896}}, "EndTime": 1584909802.541694, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.541678}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 37.494864327566965, "sum": 37.494864327566965, "min": 37.494864327566965}}, "EndTime": 1584909802.541753, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.541737}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 31.379198346819198, "sum": 31.379198346819198, "min": 31.379198346819198}}, "EndTime": 1584909802.541812, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.541796}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 31.706697736467635, "sum": 31.706697736467635, "min": 31.706697736467635}}, "EndTime": 1584909802.541869, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.541854}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 32.399632045200896, "sum": 32.399632045200896, "min": 32.399632045200896}}, "EndTime": 1584909802.541924, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.541909}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 58.438201904296875, "sum": 58.438201904296875, "min": 58.438201904296875}}, "EndTime": 1584909802.54198, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.541965}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 71.49111502511161, "sum": 71.49111502511161, "min": 71.49111502511161}}, "EndTime": 1584909802.542034, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.54202}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 54.9476318359375, "sum": 54.9476318359375, "min": 54.9476318359375}}, "EndTime": 1584909802.54209, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.542075}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 63.514657156808035, "sum": 63.514657156808035, "min": 63.514657156808035}}, "EndTime": 1584909802.542145, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.54213}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 32.42124720982143, "sum": 32.42124720982143, "min": 32.42124720982143}}, "EndTime": 1584909802.542202, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.542187}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 32.94027927943638, "sum": 32.94027927943638, "min": 32.94027927943638}}, "EndTime": 1584909802.542256, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.542241}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 35.61633082798549, "sum": 35.61633082798549, "min": 35.61633082798549}}, "EndTime": 1584909802.542312, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.542297}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 31.224393572126115, "sum": 31.224393572126115, "min": 31.224393572126115}}, "EndTime": 1584909802.542367, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.542352}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 60.437709263392854, "sum": 60.437709263392854, "min": 60.437709263392854}}, "EndTime": 1584909802.542422, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.542407}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 53.03289358956473, "sum": 53.03289358956473, "min": 53.03289358956473}}, "EndTime": 1584909802.542479, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.542464}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 73.15806361607143, "sum": 73.15806361607143, "min": 73.15806361607143}}, "EndTime": 1584909802.542533, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.542519}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 61.142054966517854, "sum": 61.142054966517854, "min": 61.142054966517854}}, "EndTime": 1584909802.542588, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.542573}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 56.681056431361604, "sum": 56.681056431361604, "min": 56.681056431361604}}, "EndTime": 1584909802.542646, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.542631}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 39.41819109235491, "sum": 39.41819109235491, "min": 39.41819109235491}}, "EndTime": 1584909802.542701, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.542687}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 55.440281459263396, "sum": 55.440281459263396, "min": 55.440281459263396}}, "EndTime": 1584909802.542758, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.542742}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 56.82619367327009, "sum": 56.82619367327009, "min": 56.82619367327009}}, "EndTime": 1584909802.542813, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.542798}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 96.18596540178571, "sum": 96.18596540178571, "min": 96.18596540178571}}, "EndTime": 1584909802.542867, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.542853}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 93.30860246930804, "sum": 93.30860246930804, "min": 93.30860246930804}}, "EndTime": 1584909802.542924, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.542908}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 89.36885288783482, "sum": 89.36885288783482, "min": 89.36885288783482}}, "EndTime": 1584909802.542981, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.542966}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 91.98063441685268, "sum": 91.98063441685268, "min": 91.98063441685268}}, "EndTime": 1584909802.543036, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.543022}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 100.42381940569196, "sum": 100.42381940569196, "min": 100.42381940569196}}, "EndTime": 1584909802.54309, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.543076}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 89.06330217633929, "sum": 89.06330217633929, "min": 89.06330217633929}}, "EndTime": 1584909802.543145, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.543131}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 103.74072265625, "sum": 103.74072265625, "min": 103.74072265625}}, "EndTime": 1584909802.543202, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.543187}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 104.33241489955357, "sum": 104.33241489955357, "min": 104.33241489955357}}, "EndTime": 1584909802.543261, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.543245}
[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] #quality_metric: host=algo-1, epoch=12, validation mse_objective <loss>=53.68372018[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] #early_stopping_criteria_metric: host=algo-1, epoch=12, criteria=mse_objective, value=31.2243935721[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] #progress_metric: host=algo-1, completed 86 % of epochs[0m
[34m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 2, "sum": 2.0, "min": 2}, "Number of Batches Since Last Reset": {"count": 1, "max": 2, "sum": 2.0, "min": 2}, "Number of Records Since Last Reset": {"count": 1, "max": 227, "sum": 227.0, "min": 227}, "Total Batches Seen": {"count": 1, "max": 29, "sum": 29.0, "min": 29}, "Total Records Seen": {"count": 1, "max": 3378, "sum": 3378.0, "min": 3378}, "Max Records Seen Between Resets": {"count": 1, "max": 227, "sum": 227.0, "min": 227}, "Reset Count": {"count": 1, "max": 15, "sum": 15.0, "min": 15}}, "EndTime": 1584909802.544366, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 12}, "StartTime": 1584909802.442946}
[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] #throughput_metric: host=algo-1, train throughput=2235.30780861 records/second[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5793072891235351, "sum": 0.5793072891235351, "min": 0.5793072891235351}}, "EndTime": 1584909802.571016, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.570917}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5652274322509766, "sum": 0.5652274322509766, "min": 0.5652274322509766}}, "EndTime": 1584909802.571144, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.571096}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.7011515045166016, "sum": 0.7011515045166016, "min": 0.7011515045166016}}, "EndTime": 1584909802.57143, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.571384}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5506500244140625, "sum": 0.5506500244140625, "min": 0.5506500244140625}}, "EndTime": 1584909802.571555, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.571531}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.4404460525512695, "sum": 0.4404460525512695, "min": 0.4404460525512695}}, "EndTime": 1584909802.571654, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.571633}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.4903615951538086, "sum": 0.4903615951538086, "min": 0.4903615951538086}}, "EndTime": 1584909802.571716, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.571699}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.4948543930053711, "sum": 0.4948543930053711, "min": 0.4948543930053711}}, "EndTime": 1584909802.571809, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.571791}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.4818452453613281, "sum": 0.4818452453613281, "min": 0.4818452453613281}}, "EndTime": 1584909802.571867, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.571851}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6090438461303711, "sum": 0.6090438461303711, "min": 0.6090438461303711}}, "EndTime": 1584909802.571957, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.571937}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.7355956268310547, "sum": 0.7355956268310547, "min": 0.7355956268310547}}, "EndTime": 1584909802.572046, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.572027}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5676240921020508, "sum": 0.5676240921020508, "min": 0.5676240921020508}}, "EndTime": 1584909802.572104, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.572089}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.650494613647461, "sum": 0.650494613647461, "min": 0.650494613647461}}, "EndTime": 1584909802.572196, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.572177}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.4477265548706055, "sum": 0.4477265548706055, "min": 0.4477265548706055}}, "EndTime": 1584909802.572286, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.572267}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.45213623046875, "sum": 0.45213623046875, "min": 0.45213623046875}}, "EndTime": 1584909802.572348, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.572332}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.4291610336303711, "sum": 0.4291610336303711, "min": 0.4291610336303711}}, "EndTime": 1584909802.572436, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.572405}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.4545050048828125, "sum": 0.4545050048828125, "min": 0.4545050048828125}}, "EndTime": 1584909802.572527, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.572507}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6417829895019531, "sum": 0.6417829895019531, "min": 0.6417829895019531}}, "EndTime": 1584909802.572589, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.572572}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5692551422119141, "sum": 0.5692551422119141, "min": 0.5692551422119141}}, "EndTime": 1584909802.572674, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.57263}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.7380157470703125, "sum": 0.7380157470703125, "min": 0.7380157470703125}}, "EndTime": 1584909802.572764, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.572721}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6386724472045898, "sum": 0.6386724472045898, "min": 0.6386724472045898}}, "EndTime": 1584909802.572831, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.572813}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5782119750976562, "sum": 0.5782119750976562, "min": 0.5782119750976562}}, "EndTime": 1584909802.572888, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.572873}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.4972393035888672, "sum": 0.4972393035888672, "min": 0.4972393035888672}}, "EndTime": 1584909802.572978, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.572959}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.581148681640625, "sum": 0.581148681640625, "min": 0.581148681640625}}, "EndTime": 1584909802.573061, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.573041}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5910939788818359, "sum": 0.5910939788818359, "min": 0.5910939788818359}}, "EndTime": 1584909802.573124, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.573108}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9950400543212891, "sum": 0.9950400543212891, "min": 0.9950400543212891}}, "EndTime": 1584909802.573206, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.573165}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9702590179443359, "sum": 0.9702590179443359, "min": 0.9702590179443359}}, "EndTime": 1584909802.57327, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.573254}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9170980072021484, "sum": 0.9170980072021484, "min": 0.9170980072021484}}, "EndTime": 1584909802.573355, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.573337}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9406155395507813, "sum": 0.9406155395507813, "min": 0.9406155395507813}}, "EndTime": 1584909802.573413, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.573398}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.0257257843017578, "sum": 1.0257257843017578, "min": 1.0257257843017578}}, "EndTime": 1584909802.573496, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.573476}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.8948958587646484, "sum": 0.8948958587646484, "min": 0.8948958587646484}}, "EndTime": 1584909802.573585, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.573567}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.0686262512207032, "sum": 1.0686262512207032, "min": 1.0686262512207032}}, "EndTime": 1584909802.573645, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.57363}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.0856653594970702, "sum": 1.0856653594970702, "min": 1.0856653594970702}}, "EndTime": 1584909802.57375, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.573717}
[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] #quality_metric: host=algo-1, epoch=13, train mse_objective <loss>=0.579307289124[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 52.29193115234375, "sum": 52.29193115234375, "min": 52.29193115234375}}, "EndTime": 1584909802.633438, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.63334}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 52.184317452566965, "sum": 52.184317452566965, "min": 52.184317452566965}}, "EndTime": 1584909802.63355, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.633528}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 64.94333321707589, "sum": 64.94333321707589, "min": 64.94333321707589}}, "EndTime": 1584909802.633613, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.633598}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 50.520843505859375, "sum": 50.520843505859375, "min": 50.520843505859375}}, "EndTime": 1584909802.633675, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.63366}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 39.990997314453125, "sum": 39.990997314453125, "min": 39.990997314453125}}, "EndTime": 1584909802.633736, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.633719}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 30.308334350585938, "sum": 30.308334350585938, "min": 30.308334350585938}}, "EndTime": 1584909802.633796, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.63378}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 32.57765415736607, "sum": 32.57765415736607, "min": 32.57765415736607}}, "EndTime": 1584909802.633852, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.633838}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 30.79604012625558, "sum": 30.79604012625558, "min": 30.79604012625558}}, "EndTime": 1584909802.633907, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.633893}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 57.282374790736604, "sum": 57.282374790736604, "min": 57.282374790736604}}, "EndTime": 1584909802.633968, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.633951}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 68.84225899832589, "sum": 68.84225899832589, "min": 68.84225899832589}}, "EndTime": 1584909802.634026, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.634012}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 53.790130615234375, "sum": 53.790130615234375, "min": 53.790130615234375}}, "EndTime": 1584909802.63408, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.634066}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 61.837171282087056, "sum": 61.837171282087056, "min": 61.837171282087056}}, "EndTime": 1584909802.634133, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.634119}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 35.74285452706473, "sum": 35.74285452706473, "min": 35.74285452706473}}, "EndTime": 1584909802.634186, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.634172}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 34.23553248814174, "sum": 34.23553248814174, "min": 34.23553248814174}}, "EndTime": 1584909802.634238, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.634224}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 36.68561226981027, "sum": 36.68561226981027, "min": 36.68561226981027}}, "EndTime": 1584909802.63429, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.634276}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 35.543932233537944, "sum": 35.543932233537944, "min": 35.543932233537944}}, "EndTime": 1584909802.634344, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.63433}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 58.94756644112723, "sum": 58.94756644112723, "min": 58.94756644112723}}, "EndTime": 1584909802.634398, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.634384}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 51.809557233537944, "sum": 51.809557233537944, "min": 51.809557233537944}}, "EndTime": 1584909802.634453, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.634439}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 71.25745936802456, "sum": 71.25745936802456, "min": 71.25745936802456}}, "EndTime": 1584909802.634505, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.634492}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 59.632359095982146, "sum": 59.632359095982146, "min": 59.632359095982146}}, "EndTime": 1584909802.634558, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.634544}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 58.35848127092634, "sum": 58.35848127092634, "min": 58.35848127092634}}, "EndTime": 1584909802.634615, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.6346}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 41.96736798967634, "sum": 41.96736798967634, "min": 41.96736798967634}}, "EndTime": 1584909802.634671, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.634657}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 57.046962193080354, "sum": 57.046962193080354, "min": 57.046962193080354}}, "EndTime": 1584909802.634724, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.63471}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 54.53947230747768, "sum": 54.53947230747768, "min": 54.53947230747768}}, "EndTime": 1584909802.634776, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.634762}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 96.00078473772321, "sum": 96.00078473772321, "min": 96.00078473772321}}, "EndTime": 1584909802.634828, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.634815}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 93.36281912667411, "sum": 93.36281912667411, "min": 93.36281912667411}}, "EndTime": 1584909802.634879, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.634865}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 89.10237339564732, "sum": 89.10237339564732, "min": 89.10237339564732}}, "EndTime": 1584909802.634931, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.634917}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 91.79480852399554, "sum": 91.79480852399554, "min": 91.79480852399554}}, "EndTime": 1584909802.634982, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.634969}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 99.79109409877232, "sum": 99.79109409877232, "min": 99.79109409877232}}, "EndTime": 1584909802.635033, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.635019}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 89.49817766462054, "sum": 89.49817766462054, "min": 89.49817766462054}}, "EndTime": 1584909802.635085, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.635071}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 105.10137939453125, "sum": 105.10137939453125, "min": 105.10137939453125}}, "EndTime": 1584909802.635135, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.635122}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 99.64419119698661, "sum": 99.64419119698661, "min": 99.64419119698661}}, "EndTime": 1584909802.635191, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.635176}
[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] #quality_metric: host=algo-1, epoch=13, validation mse_objective <loss>=52.2919311523[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] #early_stopping_criteria_metric: host=algo-1, epoch=13, criteria=mse_objective, value=30.3083343506[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] #progress_metric: host=algo-1, completed 93 % of epochs[0m
[34m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 2, "sum": 2.0, "min": 2}, "Number of Batches Since Last Reset": {"count": 1, "max": 2, "sum": 2.0, "min": 2}, "Number of Records Since Last Reset": {"count": 1, "max": 227, "sum": 227.0, "min": 227}, "Total Batches Seen": {"count": 1, "max": 31, "sum": 31.0, "min": 31}, "Total Records Seen": {"count": 1, "max": 3605, "sum": 3605.0, "min": 3605}, "Max Records Seen Between Resets": {"count": 1, "max": 227, "sum": 227.0, "min": 227}, "Reset Count": {"count": 1, "max": 16, "sum": 16.0, "min": 16}}, "EndTime": 1584909802.636325, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 13}, "StartTime": 1584909802.544626}
[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] #throughput_metric: host=algo-1, train throughput=2471.71478786 records/second[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5698220443725586, "sum": 0.5698220443725586, "min": 0.5698220443725586}}, "EndTime": 1584909802.662589, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.662535}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.557751121520996, "sum": 0.557751121520996, "min": 0.557751121520996}}, "EndTime": 1584909802.662663, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.662643}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6778724670410157, "sum": 0.6778724670410157, "min": 0.6778724670410157}}, "EndTime": 1584909802.662738, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.662719}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5442504119873047, "sum": 0.5442504119873047, "min": 0.5442504119873047}}, "EndTime": 1584909802.662794, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.662777}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.4423760986328125, "sum": 0.4423760986328125, "min": 0.4423760986328125}}, "EndTime": 1584909802.662865, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.662847}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.4413953018188477, "sum": 0.4413953018188477, "min": 0.4413953018188477}}, "EndTime": 1584909802.662933, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.662915}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.46802677154541017, "sum": 0.46802677154541017, "min": 0.46802677154541017}}, "EndTime": 1584909802.662993, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.662975}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.4147364807128906, "sum": 0.4147364807128906, "min": 0.4147364807128906}}, "EndTime": 1584909802.663048, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.663032}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6009479904174805, "sum": 0.6009479904174805, "min": 0.6009479904174805}}, "EndTime": 1584909802.663109, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.663093}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.7086914825439453, "sum": 0.7086914825439453, "min": 0.7086914825439453}}, "EndTime": 1584909802.663164, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.663147}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5602148818969727, "sum": 0.5602148818969727, "min": 0.5602148818969727}}, "EndTime": 1584909802.663218, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.663202}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6369557189941406, "sum": 0.6369557189941406, "min": 0.6369557189941406}}, "EndTime": 1584909802.663273, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.663258}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.44056026458740233, "sum": 0.44056026458740233, "min": 0.44056026458740233}}, "EndTime": 1584909802.663333, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.663316}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.4277658462524414, "sum": 0.4277658462524414, "min": 0.4277658462524414}}, "EndTime": 1584909802.663383, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.663368}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.4127415466308594, "sum": 0.4127415466308594, "min": 0.4127415466308594}}, "EndTime": 1584909802.663438, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.663423}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.4653852081298828, "sum": 0.4653852081298828, "min": 0.4653852081298828}}, "EndTime": 1584909802.663533, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.663515}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6291641616821289, "sum": 0.6291641616821289, "min": 0.6291641616821289}}, "EndTime": 1584909802.663596, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.663579}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5593299865722656, "sum": 0.5593299865722656, "min": 0.5593299865722656}}, "EndTime": 1584909802.663652, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.663635}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.7218338775634766, "sum": 0.7218338775634766, "min": 0.7218338775634766}}, "EndTime": 1584909802.663706, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.66369}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6262467193603516, "sum": 0.6262467193603516, "min": 0.6262467193603516}}, "EndTime": 1584909802.663767, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.663751}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.6020691299438476, "sum": 0.6020691299438476, "min": 0.6020691299438476}}, "EndTime": 1584909802.663822, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.663806}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.48386669158935547, "sum": 0.48386669158935547, "min": 0.48386669158935547}}, "EndTime": 1584909802.663873, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.663857}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.603116683959961, "sum": 0.603116683959961, "min": 0.603116683959961}}, "EndTime": 1584909802.663925, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.663909}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.5658351516723633, "sum": 0.5658351516723633, "min": 0.5658351516723633}}, "EndTime": 1584909802.663985, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.663968}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.994081802368164, "sum": 0.994081802368164, "min": 0.994081802368164}}, "EndTime": 1584909802.664042, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.664025}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9700664520263672, "sum": 0.9700664520263672, "min": 0.9700664520263672}}, "EndTime": 1584909802.664094, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.664077}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9155027770996094, "sum": 0.9155027770996094, "min": 0.9155027770996094}}, "EndTime": 1584909802.664145, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.664129}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.9410218811035156, "sum": 0.9410218811035156, "min": 0.9410218811035156}}, "EndTime": 1584909802.664206, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.664189}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.013902130126953, "sum": 1.013902130126953, "min": 1.013902130126953}}, "EndTime": 1584909802.664264, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.664248}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 0.8917839050292968, "sum": 0.8917839050292968, "min": 0.8917839050292968}}, "EndTime": 1584909802.664315, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.6643}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.08326904296875, "sum": 1.08326904296875, "min": 1.08326904296875}}, "EndTime": 1584909802.664367, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.664351}
[0m
[34m#metrics {"Metrics": {"train_mse_objective": {"count": 1, "max": 1.029201126098633, "sum": 1.029201126098633, "min": 1.029201126098633}}, "EndTime": 1584909802.664424, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.664408}
[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] #quality_metric: host=algo-1, epoch=14, train mse_objective <loss>=0.569822044373[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 51.04401070731027, "sum": 51.04401070731027, "min": 51.04401070731027}}, "EndTime": 1584909802.713681, "Dimensions": {"model": 0, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.713624}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 51.01598685128348, "sum": 51.01598685128348, "min": 51.01598685128348}}, "EndTime": 1584909802.713785, "Dimensions": {"model": 1, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.713738}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 62.743299211774556, "sum": 62.743299211774556, "min": 62.743299211774556}}, "EndTime": 1584909802.713885, "Dimensions": {"model": 2, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.713861}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 49.450531005859375, "sum": 49.450531005859375, "min": 49.450531005859375}}, "EndTime": 1584909802.714027, "Dimensions": {"model": 3, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.713985}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 35.14858136858259, "sum": 35.14858136858259, "min": 35.14858136858259}}, "EndTime": 1584909802.714146, "Dimensions": {"model": 4, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.714081}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 26.698645455496653, "sum": 26.698645455496653, "min": 26.698645455496653}}, "EndTime": 1584909802.714246, "Dimensions": {"model": 5, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.714201}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 29.93497794015067, "sum": 29.93497794015067, "min": 29.93497794015067}}, "EndTime": 1584909802.714391, "Dimensions": {"model": 6, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.714335}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 28.748674665178573, "sum": 28.748674665178573, "min": 28.748674665178573}}, "EndTime": 1584909802.714462, "Dimensions": {"model": 7, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.714443}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 56.20296369280134, "sum": 56.20296369280134, "min": 56.20296369280134}}, "EndTime": 1584909802.714585, "Dimensions": {"model": 8, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.714551}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 66.4066641671317, "sum": 66.4066641671317, "min": 66.4066641671317}}, "EndTime": 1584909802.714654, "Dimensions": {"model": 9, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.714636}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 52.716884068080354, "sum": 52.716884068080354, "min": 52.716884068080354}}, "EndTime": 1584909802.714787, "Dimensions": {"model": 10, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.714748}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 60.300262451171875, "sum": 60.300262451171875, "min": 60.300262451171875}}, "EndTime": 1584909802.714892, "Dimensions": {"model": 11, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.714839}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 34.582345145089285, "sum": 34.582345145089285, "min": 34.582345145089285}}, "EndTime": 1584909802.714987, "Dimensions": {"model": 12, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.714948}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 31.995956420898438, "sum": 31.995956420898438, "min": 31.995956420898438}}, "EndTime": 1584909802.715138, "Dimensions": {"model": 13, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.715075}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 32.77268327985491, "sum": 32.77268327985491, "min": 32.77268327985491}}, "EndTime": 1584909802.715219, "Dimensions": {"model": 14, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.7152}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 36.05118887765067, "sum": 36.05118887765067, "min": 36.05118887765067}}, "EndTime": 1584909802.715348, "Dimensions": {"model": 15, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.715309}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 57.62144252232143, "sum": 57.62144252232143, "min": 57.62144252232143}}, "EndTime": 1584909802.715417, "Dimensions": {"model": 16, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.715398}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 50.67798287527902, "sum": 50.67798287527902, "min": 50.67798287527902}}, "EndTime": 1584909802.715529, "Dimensions": {"model": 17, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.715509}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 69.50888933454242, "sum": 69.50888933454242, "min": 69.50888933454242}}, "EndTime": 1584909802.715592, "Dimensions": {"model": 18, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.715575}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 58.222242082868306, "sum": 58.222242082868306, "min": 58.222242082868306}}, "EndTime": 1584909802.715719, "Dimensions": {"model": 19, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.715697}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 50.224413190569194, "sum": 50.224413190569194, "min": 50.224413190569194}}, "EndTime": 1584909802.715818, "Dimensions": {"model": 20, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.715765}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 44.490814208984375, "sum": 44.490814208984375, "min": 44.490814208984375}}, "EndTime": 1584909802.715929, "Dimensions": {"model": 21, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.715906}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 48.980503627232146, "sum": 48.980503627232146, "min": 48.980503627232146}}, "EndTime": 1584909802.716002, "Dimensions": {"model": 22, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.715983}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 46.71553257533482, "sum": 46.71553257533482, "min": 46.71553257533482}}, "EndTime": 1584909802.716107, "Dimensions": {"model": 23, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.716087}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 95.77655901227679, "sum": 95.77655901227679, "min": 95.77655901227679}}, "EndTime": 1584909802.716218, "Dimensions": {"model": 24, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.716196}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 93.40262276785714, "sum": 93.40262276785714, "min": 93.40262276785714}}, "EndTime": 1584909802.716284, "Dimensions": {"model": 25, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.716267}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 88.99286760602679, "sum": 88.99286760602679, "min": 88.99286760602679}}, "EndTime": 1584909802.716425, "Dimensions": {"model": 26, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.716369}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 91.63579450334821, "sum": 91.63579450334821, "min": 91.63579450334821}}, "EndTime": 1584909802.716501, "Dimensions": {"model": 27, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.71648}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 96.43183244977679, "sum": 96.43183244977679, "min": 96.43183244977679}}, "EndTime": 1584909802.716603, "Dimensions": {"model": 28, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.716581}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 93.19574846540179, "sum": 93.19574846540179, "min": 93.19574846540179}}, "EndTime": 1584909802.716707, "Dimensions": {"model": 29, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.71665}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 101.06034633091518, "sum": 101.06034633091518, "min": 101.06034633091518}}, "EndTime": 1584909802.716779, "Dimensions": {"model": 30, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.71676}
[0m
[34m#metrics {"Metrics": {"validation_mse_objective": {"count": 1, "max": 93.74844796316964, "sum": 93.74844796316964, "min": 93.74844796316964}}, "EndTime": 1584909802.716882, "Dimensions": {"model": 31, "Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.716859}
[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] #quality_metric: host=algo-1, epoch=14, validation mse_objective <loss>=51.0440107073[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] #early_stopping_criteria_metric: host=algo-1, epoch=14, criteria=mse_objective, value=26.6986454555[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] Epoch 14: Loss has not improved for 0 epochs.[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] #progress_metric: host=algo-1, completed 100 % of epochs[0m
[34m#metrics {"Metrics": {"Max Batches Seen Between Resets": {"count": 1, "max": 2, "sum": 2.0, "min": 2}, "Number of Batches Since Last Reset": {"count": 1, "max": 2, "sum": 2.0, "min": 2}, "Number of Records Since Last Reset": {"count": 1, "max": 227, "sum": 227.0, "min": 227}, "Total Batches Seen": {"count": 1, "max": 33, "sum": 33.0, "min": 33}, "Total Records Seen": {"count": 1, "max": 3832, "sum": 3832.0, "min": 3832}, "Max Records Seen Between Resets": {"count": 1, "max": 227, "sum": 227.0, "min": 227}, "Reset Count": {"count": 1, "max": 17, "sum": 17.0, "min": 17}}, "EndTime": 1584909802.718294, "Dimensions": {"Host": "algo-1", "Meta": "training_data_iter", "Operation": "training", "Algorithm": "Linear Learner", "epoch": 14}, "StartTime": 1584909802.636588}
[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] #throughput_metric: host=algo-1, train throughput=2772.79070404 records/second[0m
[34m[03/22/2020 20:43:22 WARNING 140403405219648] wait_for_all_workers will not sync workers since the kv store is not running distributed[0m
[34m[03/22/2020 20:43:22 WARNING 140403405219648] wait_for_all_workers will not sync workers since the kv store is not running distributed[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] #early_stopping_criteria_metric: host=algo-1, epoch=14, criteria=mse_objective, value=26.6986454555[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] #validation_score (algo-1) : ('mse_objective', 22.942620413643972)[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] #validation_score (algo-1) : ('mse', 22.942620413643972)[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] #validation_score (algo-1) : ('absolute_loss', 3.571606227329799)[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] #quality_metric: host=algo-1, validation mse_objective <loss>=22.9426204136[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] #quality_metric: host=algo-1, validation mse <loss>=22.9426204136[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] #quality_metric: host=algo-1, validation absolute_loss <loss>=3.57160622733[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] Best model found for hyperparameters: {"lr_scheduler_step": 10, "wd": 0.01, "optimizer": "adam", "lr_scheduler_factor": 0.99, "l1": 0.0, "learning_rate": 0.1, "lr_scheduler_minimum_lr": 0.0001}[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] Saved checkpoint to "/tmp/tmpMiNH72/mx-mod-0000.params"[0m
[34m[03/22/2020 20:43:22 INFO 140403405219648] Test data is not provided.[0m
[34m#metrics {"Metrics": {"totaltime": {"count": 1, "max": 1498.0640411376953, "sum": 1498.0640411376953, "min": 1498.0640411376953}, "finalize.time": {"count": 1, "max": 66.49613380432129, "sum": 66.49613380432129, "min": 66.49613380432129}, "initialize.time": {"count": 1, "max": 153.78594398498535, "sum": 153.78594398498535, "min": 153.78594398498535}, "check_early_stopping.time": {"count": 16, "max": 1.1320114135742188, "sum": 7.656097412109375, "min": 0.19693374633789062}, "setuptime": {"count": 1, "max": 27.779102325439453, "sum": 27.779102325439453, "min": 27.779102325439453}, "update.time": {"count": 15, "max": 101.21989250183105, "sum": 1179.3622970581055, "min": 64.0571117401123}, "epochs": {"count": 1, "max": 15, "sum": 15.0, "min": 15}}, "EndTime": 1584909802.792606, "Dimensions": {"Host": "algo-1", "Operation": "training", "Algorithm": "Linear Learner"}, "StartTime": 1584909801.37827}
[0m

2020-03-22 20:43:31 Completed - Training job completed
Training seconds: 64
Billable seconds: 64
```

## Step 6 (B): Deploy the trained model

Similar to the XGBoost model, now that we've fit the model we need to deploy it. Also like the
XGBoost model, we will use the lower level approach so that we have more control over the endpoint
that gets created.

### Build the model

Of course, before we can deploy the model, we need to first create it. The `fit` method that we used
earlier created some model artifacts and we can use these to construct a model object.

```python
# First, we create a unique model name
linear_model_name = "boston-update-linear-model" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

# We also need to tell SageMaker which container should be used for inference and where it should
# retrieve the model artifacts from. In our case, the linear-learner container that we used for training
# can also be used for inference.
linear_primary_container = {
    "Image": linear_container,
    "ModelDataUrl": linear.model_data
}

# And lastly we construct the SageMaker model
linear_model_info = session.sagemaker_client.create_model(
                                ModelName = linear_model_name,
                                ExecutionRoleArn = role,
                                PrimaryContainer = linear_primary_container)
```

```python
pprint(linear_model_info)
```

```json
{
    'ModelArn': 'arn:aws:sagemaker:us-west-2:171758673694:model/boston-update-linear-model2020-03-22-21-54-30',
    'ResponseMetadata': {
        'HTTPHeaders': {
            'content-length': '107',
            'content-type': 'application/x-amz-json-1.1',
            'date': 'Sun, 22 Mar 2020 21:54:30 GMT',
            'x-amzn-requestid': 'a817fa72-ae06-49b5-9adf-945952a38809'
        },
        'HTTPStatusCode': 200,
        'RequestId': 'a817fa72-ae06-49b5-9adf-945952a38809',
        'RetryAttempts': 0
    }
}
```

### Create the endpoint configuration

Once we have the model we can start putting together the endpoint by creating an endpoint
configuration.

```python
# As before, we need to give our endpoint configuration a name which should be unique
linear_endpoint_config_name = "boston-linear-endpoint-config-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

# And then we ask SageMaker to construct the endpoint configuration
linear_endpoint_config_info = session.sagemaker_client.create_endpoint_config(
                            EndpointConfigName = linear_endpoint_config_name,
                            ProductionVariants = [{
                                "InstanceType": "ml.m4.xlarge",
                                "InitialVariantWeight": 1,
                                "InitialInstanceCount": 1,
                                "ModelName": linear_model_name,
                                "VariantName": "Linear-Model"
                            }])
```

### Deploy the endpoint

Now that the endpoint configuration has been created, we can ask SageMaker to build our endpoint.

**Note:** This is a friendly (repeated) reminder that you are about to deploy an endpoint. Make sure
that you shut it down once you've finished with it!

```python
# Again, we need a unique name for our endpoint
endpoint_name = "boston-update-endpoint-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

# And then we can deploy our endpoint
endpoint_info = session.sagemaker_client.create_endpoint(
                    EndpointName = endpoint_name,
                    EndpointConfigName = linear_endpoint_config_name)
```

```python
endpoint_dec = session.wait_for_endpoint(endpoint_name)
```

## Step 7 (B): Use the model

Just like with the XGBoost model, we will send some data to our endpoint to make sure that it is
working properly. An important note is that the output format for the linear model is different from
the XGBoost model.

```python
response = session.sagemaker_runtime_client.invoke_endpoint(
                                                EndpointName = endpoint_name,
                                                ContentType = 'text/csv',
                                                Body = ','.join(map(str, X_test.values[0])))
```

```python
pprint(response)
```

```json
{
    'Body': <botocore.response.StreamingBody object at 0x7fdadc171a58>,
    'ContentType': 'application/json',
    'InvokedProductionVariant': 'Linear-Model',
    'ResponseMetadata': {
        'HTTPHeaders': {
            'content-length': '48',
            'content-type': 'application/json',
            'date': 'Sun, 22 Mar 2020 22:03:52 GMT',
            'x-amzn-invoked-production-variant': 'Linear-Model',
            'x-amzn-requestid': '96a14434-f726-4f57-b472-d721c53d0d15'
        },
        'HTTPStatusCode': 200,
        'RequestId': '96a14434-f726-4f57-b472-d721c53d0d15',
        'RetryAttempts': 0
    }
}
```

```python
result = response['Body'].read().decode("utf-8")
```

```python
pprint(result) # Linear model performs much better than XGBoost model, it's quite simple. The data is indeed linear.
```

```json
{
    "predictions": [
        {"score": 11.719074249267578}
    ]
}
```

```python
Y_test.values[0]
```

```text
array([12.3])
```

## Shut down the endpoint

Now that we know that the Linear model's endpoint works, we can shut it down.

```python
session.sagemaker_client.delete_endpoint(EndpointName = endpoint_name)
```

```text
boston-update-endpoint-2020-03-22-22-10-15
```

## Step 6 (C): Deploy a combined model

So far we've constructed two separate models which we could deploy and use. Before we talk about how
we can change a deployed endpoint from one configuration to another, let's consider a slightly
different situation. Suppose that before we switch from using only the XGBoost model to only the
Linear model, we first want to do something like an A-B test, where we send some of the incoming
data to the XGBoost model and some of the data to the Linear model.

Fortunately, SageMaker provides this functionality. And to actually get SageMaker to do this for us
is not too different from deploying a model in the way that we've already done. The only difference
is that we need to list more than one model in the production variants parameter of the endpoint
configuration.

A reasonable question to ask is, how much data is sent to each of the models that I list in the
production variants parameter? The answer is that it depends on the weight set for each model.

Suppose that we have $k$ models listed in the production variants and that each model $i$ is
assigned the weight $w_i$. Then each model $i$ will receive $w_i / W$ of the traffic where
$W = \sum_{i} w_i$.

In our case, since we have two models, the linear model and the XGBoost model, and each model has
weight 1, we see that each model will get 1 / (1 + 1) = 1/2 of the data sent to the endpoint.

```python
# As before, we need to give our endpoint configuration a name which should be unique
combined_endpoint_config_name = "boston-combined-endpoint-config-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

# And then we ask SageMaker to construct the endpoint configuration
combined_endpoint_config_info = session.sagemaker_client.create_endpoint_config(
                            EndpointConfigName = combined_endpoint_config_name,
                            ProductionVariants = [
                                { # First we include the linear model
                                    "InstanceType": "ml.m4.xlarge",
                                    "InitialVariantWeight": 1,
                                    "InitialInstanceCount": 1,
                                    "ModelName": linear_model_name,
                                    "VariantName": "Linear-Model"
                                }, { # And next we include the xgb model
                                    "InstanceType": "ml.m4.xlarge",
                                    "InitialVariantWeight": 1,
                                    "InitialInstanceCount": 1,
                                    "ModelName": xgb_model_name,
                                    "VariantName": "XGB-Model"
                                }])
```

Now that we've created the endpoint configuration, we can ask SageMaker to construct the endpoint.

**Note:** This is a friendly (repeated) reminder that you are about to deploy an endpoint. Make sure
that you shut it down once you've finished with it!

```python
# Again, we need a unique name for our endpoint
endpoint_name = "boston-update-endpoint-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

# And then we can deploy our endpoint
endpoint_info = session.sagemaker_client.create_endpoint(
                    EndpointName = endpoint_name,
                    EndpointConfigName = combined_endpoint_config_name)
```

```python
endpoint_dec = session.wait_for_endpoint(endpoint_name)
```

## Step 7 (C): Use the model

Now that we've constructed an endpoint which sends data to both the XGBoost model and the linear
model we can send some data to the endpoint and see what sort of results we get back.

```python
response = session.sagemaker_runtime_client.invoke_endpoint(
                                                EndpointName = endpoint_name,
                                                ContentType = 'text/csv',
                                                Body = ','.join(map(str, X_test.values[0])))
pprint(response)
```

```json
{
    'Body': <botocore.response.StreamingBody object at 0x7fdadca87518>,
    'ContentType': 'text/csv; charset=utf-8',
    'InvokedProductionVariant': 'XGB-Model',
    'ResponseMetadata': {
        'HTTPHeaders': {
            'content-length': '18',
            'content-type': 'text/csv; charset=utf-8',
            'date': 'Sun, 22 Mar 2020 22:17:09 GMT',
            'x-amzn-invoked-production-variant': 'XGB-Model',
            'x-amzn-requestid': '9151e286-9440-4131-b8a2-a6738d5c5419'
        },
        'HTTPStatusCode': 200,
        'RequestId': '9151e286-9440-4131-b8a2-a6738d5c5419',
        'RetryAttempts': 0
    }
}
```

Since looking at a single response doesn't give us a clear look at what is happening, we can instead
take a look at a few different responses to our endpoint

```python
for rec in range(10):
    response = session.sagemaker_runtime_client.invoke_endpoint(
                                                EndpointName = endpoint_name,
                                                ContentType = 'text/csv',
                                                Body = ','.join(map(str, X_test.values[rec])))
    pprint(response)
    result = response['Body'].read().decode("utf-8")
    print("Model prediction: {}".format(result))
    print("Actual: {}".format(Y_test.values[rec]))
```

```json
{'Body': <botocore.response.StreamingBody object at 0x7fdadcb3f438>,
    'ContentType': 'application/json',
    'InvokedProductionVariant': 'Linear-Model',
    'ResponseMetadata': {'HTTPHeaders': {'content-length': '48',
                                        'content-type': 'application/json',
                                        'date': 'Sun, 22 Mar 2020 22:18:09 GMT',
                                        'x-amzn-invoked-production-variant': 'Linear-Model',
                                        'x-amzn-requestid': '95e3e62a-a60f-4dfb-8579-391b579672f8'},
                        'HTTPStatusCode': 200,
                        'RequestId': '95e3e62a-a60f-4dfb-8579-391b579672f8',
                        'RetryAttempts': 0}}
Model prediction: {"predictions": [{"score": 11.719074249267578}]}
Actual: [12.3]
{'Body': <botocore.response.StreamingBody object at 0x7fdadcb1eb38>,
    'ContentType': 'text/csv; charset=utf-8',
    'InvokedProductionVariant': 'XGB-Model',
    'ResponseMetadata': {'HTTPHeaders': {'content-length': '18',
                                        'content-type': 'text/csv; charset=utf-8',
                                        'date': 'Sun, 22 Mar 2020 22:18:09 GMT',
                                        'x-amzn-invoked-production-variant': 'XGB-Model',
                                        'x-amzn-requestid': 'd5c4e964-8b7e-4a9f-bc4f-5b52d99943a7'},
                        'HTTPStatusCode': 200,
                        'RequestId': 'd5c4e964-8b7e-4a9f-bc4f-5b52d99943a7',
                        'RetryAttempts': 0}}
Model prediction: 20.114362716674805
Actual: [16.2]
{'Body': <botocore.response.StreamingBody object at 0x7fdadcb1ee48>,
    'ContentType': 'application/json',
    'InvokedProductionVariant': 'Linear-Model',
    'ResponseMetadata': {'HTTPHeaders': {'content-length': '48',
                                        'content-type': 'application/json',
                                        'date': 'Sun, 22 Mar 2020 22:18:09 GMT',
                                        'x-amzn-invoked-production-variant': 'Linear-Model',
                                        'x-amzn-requestid': 'bc81c4d4-130d-4c1b-91d1-82f80dd37fbd'},
                        'HTTPStatusCode': 200,
                        'RequestId': 'bc81c4d4-130d-4c1b-91d1-82f80dd37fbd',
                        'RetryAttempts': 0}}
Model prediction: {"predictions": [{"score": 28.711286544799805}]}
Actual: [23.6]
{'Body': <botocore.response.StreamingBody object at 0x7fdadcb1e908>,
    'ContentType': 'application/json',
    'InvokedProductionVariant': 'Linear-Model',
    'ResponseMetadata': {'HTTPHeaders': {'content-length': '48',
                                        'content-type': 'application/json',
                                        'date': 'Sun, 22 Mar 2020 22:18:09 GMT',
                                        'x-amzn-invoked-production-variant': 'Linear-Model',
                                        'x-amzn-requestid': '71fa6dbd-acb2-4f73-bc7a-c63410defe8b'},
                        'HTTPStatusCode': 200,
                        'RequestId': '71fa6dbd-acb2-4f73-bc7a-c63410defe8b',
                        'RetryAttempts': 0}}
Model prediction: {"predictions": [{"score": 11.144815444946289}]}
Actual: [13.6]
{'Body': <botocore.response.StreamingBody object at 0x7fdadcb1e198>,
    'ContentType': 'application/json',
    'InvokedProductionVariant': 'Linear-Model',
    'ResponseMetadata': {'HTTPHeaders': {'content-length': '47',
                                        'content-type': 'application/json',
                                        'date': 'Sun, 22 Mar 2020 22:18:09 GMT',
                                        'x-amzn-invoked-production-variant': 'Linear-Model',
                                        'x-amzn-requestid': 'fec8558e-179e-452c-bc46-a71492cf1386'},
                        'HTTPStatusCode': 200,
                        'RequestId': 'fec8558e-179e-452c-bc46-a71492cf1386',
                        'RetryAttempts': 0}}
Model prediction: {"predictions": [{"score": 26.18394660949707}]}
Actual: [22.1]
{'Body': <botocore.response.StreamingBody object at 0x7fdadc1a9c88>,
    'ContentType': 'text/csv; charset=utf-8',
    'InvokedProductionVariant': 'XGB-Model',
    'ResponseMetadata': {'HTTPHeaders': {'content-length': '17',
                                        'content-type': 'text/csv; charset=utf-8',
                                        'date': 'Sun, 22 Mar 2020 22:18:09 GMT',
                                        'x-amzn-invoked-production-variant': 'XGB-Model',
                                        'x-amzn-requestid': '47cf4b1c-0909-4a01-b5e7-59b8d87901c0'},
                        'HTTPStatusCode': 200,
                        'RequestId': '47cf4b1c-0909-4a01-b5e7-59b8d87901c0',
                        'RetryAttempts': 0}}
Model prediction: 35.28992462158203
Actual: [36.5]
{'Body': <botocore.response.StreamingBody object at 0x7fdadc1a9da0>,
    'ContentType': 'application/json',
    'InvokedProductionVariant': 'Linear-Model',
    'ResponseMetadata': {'HTTPHeaders': {'content-length': '48',
                                        'content-type': 'application/json',
                                        'date': 'Sun, 22 Mar 2020 22:18:09 GMT',
                                        'x-amzn-invoked-production-variant': 'Linear-Model',
                                        'x-amzn-requestid': '10d48a4c-2d34-460d-9265-8ac653c00309'},
                        'HTTPStatusCode': 200,
                        'RequestId': '10d48a4c-2d34-460d-9265-8ac653c00309',
                        'RetryAttempts': 0}}
Model prediction: {"predictions": [{"score": 17.303707122802734}]}
Actual: [14.9]
{'Body': <botocore.response.StreamingBody object at 0x7fdadc1a9a58>,
    'ContentType': 'application/json',
    'InvokedProductionVariant': 'Linear-Model',
    'ResponseMetadata': {'HTTPHeaders': {'content-length': '48',
                                        'content-type': 'application/json',
                                        'date': 'Sun, 22 Mar 2020 22:18:09 GMT',
                                        'x-amzn-invoked-production-variant': 'Linear-Model',
                                        'x-amzn-requestid': 'acd3d2db-2b42-42b7-9946-640f91dd8f8a'},
                        'HTTPStatusCode': 200,
                        'RequestId': 'acd3d2db-2b42-42b7-9946-640f91dd8f8a',
                        'RetryAttempts': 0}}
Model prediction: {"predictions": [{"score": 19.875295639038086}]}
Actual: [22.6]
{'Body': <botocore.response.StreamingBody object at 0x7fdadc1a96d8>,
    'ContentType': 'text/csv; charset=utf-8',
    'InvokedProductionVariant': 'XGB-Model',
    'ResponseMetadata': {'HTTPHeaders': {'content-length': '18',
                                        'content-type': 'text/csv; charset=utf-8',
                                        'date': 'Sun, 22 Mar 2020 22:18:09 GMT',
                                        'x-amzn-invoked-production-variant': 'XGB-Model',
                                        'x-amzn-requestid': '355d8c53-97b6-410e-9eba-518f84eb1a7e'},
                        'HTTPStatusCode': 200,
                        'RequestId': '355d8c53-97b6-410e-9eba-518f84eb1a7e',
                        'RetryAttempts': 0}}
Model prediction: 14.214831352233887
Actual: [14.2]
{'Body': <botocore.response.StreamingBody object at 0x7fdadc1a9860>,
    'ContentType': 'application/json',
    'InvokedProductionVariant': 'Linear-Model',
    'ResponseMetadata': {'HTTPHeaders': {'content-length': '48',
                                        'content-type': 'application/json',
                                        'date': 'Sun, 22 Mar 2020 22:18:09 GMT',
                                        'x-amzn-invoked-production-variant': 'Linear-Model',
                                        'x-amzn-requestid': '95dd8c9e-9655-4c0e-b669-bf209fd49a47'},
                        'HTTPStatusCode': 200,
                        'RequestId': '95dd8c9e-9655-4c0e-b669-bf209fd49a47',
                        'RetryAttempts': 0}}
Model prediction: {"predictions": [{"score": 28.779239654541016}]}
Actual: [23.7]
```

If at some point we aren't sure about the properties of a deployed endpoint, we can use the
`describe_endpoint` function to get SageMaker to return a description of the deployed endpoint.

```python
pprint(session.sagemaker_client.describe_endpoint(EndpointName=endpoint_name))
```

```json
{
    'CreationTime': datetime.datetime(2020, 3, 22, 22, 10, 15, 958000, tzinfo=tzlocal()),
    'EndpointArn': 'arn:aws:sagemaker:us-west-2:171758673694:endpoint/boston-update-endpoint-2020-03-22-22-10-15',
    'EndpointConfigName': 'boston-combined-endpoint-config-2020-03-22-22-06-47',
    'EndpointName': 'boston-update-endpoint-2020-03-22-22-10-15',
    'EndpointStatus': 'InService',
    'LastModifiedTime': datetime.datetime(2020, 3, 22, 22, 16, 23, 934000, tzinfo=tzlocal()),
    'ProductionVariants': [
        {
            'CurrentInstanceCount': 1,
            'CurrentWeight': 1.0,
            'DeployedImages': [
                {
                    'ResolutionTime': datetime.datetime(2020, 3, 22, 22, 10, 17, 587000, tzinfo=tzlocal()),
                    'ResolvedImage': '174872318107.dkr.ecr.us-west-2.amazonaws.com/linear-learner@sha256:1cc553330fc7ab939e72cc0c0ed4bed61bbb2e7b33b4f838cb0a146d0bb5da9c',
                    'SpecifiedImage': '174872318107.dkr.ecr.us-west-2.amazonaws.com/linear-learner:1'
                }
            ],
            'DesiredInstanceCount': 1,
            'DesiredWeight': 1.0,
            'VariantName': 'Linear-Model'
        },
        {
            'CurrentInstanceCount': 1,
            'CurrentWeight': 1.0,
            'DeployedImages': [
                {
                    'ResolutionTime': datetime.datetime(2020, 3, 22, 22, 10, 17, 842000, tzinfo=tzlocal()),
                    'ResolvedImage': '246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-xgboost@sha256:97ec7833b3e2773d3924b1a863c5742e348dea61eab21b90693ac3c3bdd08522',
                    'SpecifiedImage': '246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-xgboost:0.90-1-cpu-py3'
                }
            ],
            'DesiredInstanceCount': 1,
            'DesiredWeight': 1.0,
            'VariantName': 'XGB-Model'
        }
    ],
    'ResponseMetadata': {
        'HTTPHeaders': {
            'content-length': '1195',
            'content-type': 'application/x-amz-json-1.1',
            'date': 'Sun, 22 Mar 2020 22:18:52 GMT',
            'x-amzn-requestid': 'ce31eb2e-f2c5-401d-8fc2-d69b4a43586d'
        },
        'HTTPStatusCode': 200,
        'RequestId': 'ce31eb2e-f2c5-401d-8fc2-d69b4a43586d',
        'RetryAttempts': 0
    }
}
```

## Updating an Endpoint

Now suppose that we've done our A-B test and the new linear model is working well enough. What we'd
like to do now is to switch our endpoint from sending data to both the XGBoost model and the linear
model to sending data only to the linear model.

Of course, we don't really want to shut down the endpoint to do this as doing so would interrupt
service to whoever depends on our endpoint. Instead, we can ask SageMaker to **update** an endpoint
to a new endpoint configuration.

What is actually happening is that SageMaker will set up a new endpoint with the new characteristics.
Once this new endpoint is running, SageMaker will switch the old endpoint so that it now points at
the newly deployed model, making sure that this happens seamlessly in the background.

```python
session.sagemaker_client.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=linear_endpoint_config_name)
```

```json
{
    "EndpointArn":"arn:aws:sagemaker:us-west-2:171758673694:endpoint/boston-update-endpoint-2020-03-22-22-10-15",
    "ResponseMetadata":{
        "RequestId":"9ca50836-85bd-43eb-8566-c19e964c8490",
        "HTTPStatusCode":200,
        "HTTPHeaders":{
            "x-amzn-requestid":"9ca50836-85bd-43eb-8566-c19e964c8490",
            "content-type":"application/x-amz-json-1.1",
            "content-length":"110",
            "date":"Sun, 22 Mar 2020 22:21:31 GMT"
        },
        "RetryAttempts":0
    }
}
```

To get a glimpse at what is going on, we can ask SageMaker to describe our in-use endpoint now,
before the update process has completed. When we do so, we can see that the in-use endpoint still
has the same characteristics it had before.

```python
pprint(session.sagemaker_client.describe_endpoint(EndpointName=endpoint_name))
```

```json
{
    'CreationTime': datetime.datetime(2020, 3, 22, 22, 10, 15, 958000, tzinfo=tzlocal()),
    'EndpointArn': 'arn:aws:sagemaker:us-west-2:171758673694:endpoint/boston-update-endpoint-2020-03-22-22-10-15',
    'EndpointConfigName': 'boston-combined-endpoint-config-2020-03-22-22-06-47',
    'EndpointName': 'boston-update-endpoint-2020-03-22-22-10-15',
    'EndpointStatus': 'Updating',
    'LastModifiedTime': datetime.datetime(2020, 3, 22, 22, 21, 32, 580000, tzinfo=tzlocal()),
    'ProductionVariants': [
        {
            'CurrentInstanceCount': 1,
            'CurrentWeight': 1.0,
            'DeployedImages': [
                {
                    'ResolutionTime': datetime.datetime(2020, 3, 22, 22, 10, 17, 587000, tzinfo=tzlocal()),
                    'ResolvedImage': '174872318107.dkr.ecr.us-west-2.amazonaws.com/linear-learner@sha256:1cc553330fc7ab939e72cc0c0ed4bed61bbb2e7b33b4f838cb0a146d0bb5da9c',
                    'SpecifiedImage': '174872318107.dkr.ecr.us-west-2.amazonaws.com/linear-learner:1'
                }
            ],
            'DesiredInstanceCount': 1,
            'DesiredWeight': 1.0,
            'VariantName': 'Linear-Model'
        },
        {
            'CurrentInstanceCount': 1,
            'CurrentWeight': 1.0,
            'DeployedImages': [
                {
                    'ResolutionTime': datetime.datetime(2020, 3, 22, 22, 10, 17, 842000, tzinfo=tzlocal()),
                    'ResolvedImage': '246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-xgboost@sha256:97ec7833b3e2773d3924b1a863c5742e348dea61eab21b90693ac3c3bdd08522',
                    'SpecifiedImage': '246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-xgboost:0.90-1-cpu-py3'
                }
            ],
            'DesiredInstanceCount': 1,
            'DesiredWeight': 1.0,
            'VariantName': 'XGB-Model'
        }
    ],
    'ResponseMetadata': {
        'HTTPHeaders': {
            'content-length': '1193',
            'content-type': 'application/x-amz-json-1.1',
            'date': 'Sun, 22 Mar 2020 22:21:43 GMT',
            'x-amzn-requestid': '5878d423-cc5c-4fee-afb2-6e1fbae37db7'
        },
        'HTTPStatusCode': 200,
        'RequestId': '5878d423-cc5c-4fee-afb2-6e1fbae37db7',
        'RetryAttempts': 0
    }
}
```

If we now wait for the update process to complete, and then ask SageMaker to describe the endpoint,
it will return the characteristics of the new endpoint configuration.

```python
endpoint_dec = session.wait_for_endpoint(endpoint_name)
```

```python
pprint(session.sagemaker_client.describe_endpoint(EndpointName=endpoint_name))
```

```json
{
    'CreationTime': datetime.datetime(2020, 3, 22, 22, 10, 15, 958000, tzinfo=tzlocal()),
    'EndpointArn': 'arn:aws:sagemaker:us-west-2:171758673694:endpoint/boston-update-endpoint-2020-03-22-22-10-15',
    'EndpointConfigName': 'boston-linear-endpoint-config-2020-03-22-21-55-35',
    'EndpointName': 'boston-update-endpoint-2020-03-22-22-10-15',
    'EndpointStatus': 'InService',
    'LastModifiedTime': datetime.datetime(2020, 3, 22, 22, 27, 42, 956000, tzinfo=tzlocal()),
    'ProductionVariants': [
        {
            'CurrentInstanceCount': 1,
            'CurrentWeight': 1.0,
            'DeployedImages': [
                {
                    'ResolutionTime': datetime.datetime(2020, 3, 22, 22, 21, 35, 783000, tzinfo=tzlocal()),
                    'ResolvedImage': '174872318107.dkr.ecr.us-west-2.amazonaws.com/linear-learner@sha256:1cc553330fc7ab939e72cc0c0ed4bed61bbb2e7b33b4f838cb0a146d0bb5da9c',
                    'SpecifiedImage': '174872318107.dkr.ecr.us-west-2.amazonaws.com/linear-learner:1'
                }
            ],
            'DesiredInstanceCount': 1,
            'DesiredWeight': 1.0,
            'VariantName': 'Linear-Model'
        }
    ],
    'ResponseMetadata': {
        'HTTPHeaders': {
            'content-length': '770',
            'content-type': 'application/x-amz-json-1.1',
            'date': 'Sun, 22 Mar 2020 22:29:03 GMT',
            'x-amzn-requestid': 'd8aa87b5-fbc8-4ceb-aafb-5f754cde8e9a'
        },
        'HTTPStatusCode': 200,
        'RequestId': 'd8aa87b5-fbc8-4ceb-aafb-5f754cde8e9a',
        'RetryAttempts': 0
    }
}
```

## Shut down the endpoint

Now that we've finished, we need to make sure to shut down the endpoint.

```python
session.sagemaker_client.delete_endpoint(EndpointName = endpoint_name)
```

```json
{
    "ResponseMetadata":{
        "RequestId":"f0a7c173-9505-40b7-bc6b-6b26a7906ff4",
        "HTTPStatusCode":200,
        "HTTPHeaders":{
            "x-amzn-requestid":"f0a7c173-9505-40b7-bc6b-6b26a7906ff4",
            "content-type":"application/x-amz-json-1.1",
            "content-length":"0",
            "date":"Sun, 22 Mar 2020 22:29:09 GMT"
        },
        "RetryAttempts":0
    }
}
```

## Optional: Clean up

The default notebook instance on SageMaker doesn't have a lot of excess disk space available. As you
continue to complete and execute notebooks you will eventually fill up this disk space, leading to
errors which can be difficult to diagnose. Once you are completely finished using a notebook it is a
good idea to remove the files that you created along the way. Of course, you can do this from the
terminal or from the notebook hub if you would like. The cell below contains some commands to clean
up the created files from within the notebook.

```python
# First we will remove all of the files contained in the data_dir directory
!rm $data_dir/*

# And then we delete the directory itself
!rmdir $data_dir
```
