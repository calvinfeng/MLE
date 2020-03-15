# Predicting Boston Housing Prices

## Using XGBoost in SageMaker (Batch Transform)

_Deep Learning Nanodegree Program | Deployment_

---

As an introduction to using SageMaker's High Level Python API we will look at a relatively simple problem. Namely, we will use the [Boston Housing Dataset](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html) to predict the median value of a home in the area of Boston Mass.

The documentation for the high level API can be found on the [ReadTheDocs page](http://sagemaker.readthedocs.io/en/latest/)

## General Outline

Typically, when using a notebook instance with SageMaker, you will proceed through the following steps. Of course, not every step will need to be done with each project. Also, there is quite a lot of room for variation in many of the steps, as you will see throughout these lessons.

1. Download or otherwise retrieve the data.
2. Process / Prepare the data.
3. Upload the processed data to S3.
4. Train a chosen model.
5. Test the trained model (typically using a batch transform job).
6. Deploy the trained model.
7. Use the deployed model.

In this notebook we will only be covering steps 1 through 5 as we just want to get a feel for using SageMaker. In later notebooks we will talk about deploying a trained model in much more detail.

## Step 0: Setting up the notebook

We begin by setting up all of the necessary bits required to run our notebook. To start that means loading all of the Python modules we will need.


```python
%matplotlib inline

import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

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
# object contains some useful information tfhat we will need to access later such as our region.
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

Given that this is clean tabular data, we don't need to do any processing. However, we do need to split the rows in the dataset up into train, test and validation sets.


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

## Step 3: Uploading the data files to S3

When a training job is constructed using SageMaker, a container is executed which performs the training operation. This container is given access to data that is stored in S3. This means that we need to upload the data we want to use for training to S3. In addition, when we perform a batch transform job, SageMaker expects the input data to be stored on S3. We can use the SageMaker API to do this and hide some of the details.

### Save the data locally

First we need to create the test, train and validation csv files which we will then upload to S3.


```python
# This is our local data directory. We need to make sure that it exists.
data_dir = '../data/boston'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
```


```python
# We use pandas to save our test, train and validation data to csv files. Note that we make sure not to include header
# information or an index as this is required by the built in algorithms provided by Amazon. Also, for the train and
# validation data, it is assumed that the first entry in each row is the target variable.

X_test.to_csv(os.path.join(data_dir, 'test.csv'), header=False, index=False)

pd.concat([Y_val, X_val], axis=1).to_csv(os.path.join(data_dir, 'validation.csv'), header=False, index=False)
pd.concat([Y_train, X_train], axis=1).to_csv(os.path.join(data_dir, 'train.csv'), header=False, index=False)
```

### Upload to S3

Since we are currently running inside of a SageMaker session, we can use the object which represents this session to upload our data to the 'default' S3 bucket. Note that it is good practice to provide a custom prefix (essentially an S3 folder) to make sure that you don't accidentally interfere with data uploaded from some other notebook or project.


```python
prefix = 'boston-xgboost-HL'

test_location = session.upload_data(os.path.join(data_dir, 'test.csv'), key_prefix=prefix)
val_location = session.upload_data(os.path.join(data_dir, 'validation.csv'), key_prefix=prefix)
train_location = session.upload_data(os.path.join(data_dir, 'train.csv'), key_prefix=prefix)
```

## Step 4: Train the XGBoost model

Now that we have the training and validation data uploaded to S3, we can construct our XGBoost model and train it. We will be making use of the high level SageMaker API to do this which will make the resulting code a little easier to read at the cost of some flexibility.

To construct an estimator, the object which we wish to train, we need to provide the location of a container which contains the training code. Since we are using a built in algorithm this container is provided by Amazon. However, the full name of the container is a bit lengthy and depends on the region that we are operating in. Fortunately, SageMaker provides a useful utility method called `get_image_uri` that constructs the image name for us.

To use the `get_image_uri` method we need to provide it with our current region, which can be obtained from the session object, and the name of the algorithm we wish to use. In this notebook we will be using XGBoost however you could try another algorithm if you wish. The list of built in algorithms can be found in the list of [Common Parameters](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-algo-docker-registry-paths.html).


```python
# As stated above, we use this utility method to construct the image name for the training container.
container = get_image_uri(session.boto_region_name, 'xgboost', '0.90-1')

# Now that we know which container to use, we can construct the estimator object.
xgb = sagemaker.estimator.Estimator(container, # The image name of the training container
                                    role,      # The IAM role to use (our current role in this case)
                                    train_instance_count=1, # The number of instances to use for training
                                    train_instance_type='ml.m4.xlarge', # The type of instance to use for training
                                    output_path='s3://{}/{}/output'.format(session.default_bucket(), prefix),
                                                                        # Where to save the output (the model artifacts)
                                    sagemaker_session=session) # The current SageMaker session
```

Before asking SageMaker to begin the training job, we should probably set any model specific hyperparameters. There are quite a few that can be set when using the XGBoost algorithm, below are just a few of them. If you would like to change the hyperparameters below or modify additional ones you can find additional information on the [XGBoost hyperparameter page](https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost_hyperparameters.html)


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

Now that we have our estimator object completely set up, it is time to train it. To do this we make sure that SageMaker knows our input data is in csv format and then execute the `fit` method.


```python
# This is a wrapper around the location of our train and validation data, to make sure that SageMaker
# knows our data is in csv format.
s3_input_train = sagemaker.s3_input(s3_data=train_location, content_type='csv')
s3_input_validation = sagemaker.s3_input(s3_data=val_location, content_type='csv')

xgb.fit({'train': s3_input_train, 'validation': s3_input_validation})
```

    2020-03-15 20:00:29 Starting - Starting the training job...
    2020-03-15 20:00:30 Starting - Launching requested ML instances......
    2020-03-15 20:01:58 Starting - Preparing the instances for training......
    2020-03-15 20:02:50 Downloading - Downloading input data...
    2020-03-15 20:03:09 Training - Downloading the training image...
    2020-03-15 20:03:44 Uploading - Uploading generated training model[34mINFO:sagemaker-containers:Imported framework sagemaker_xgboost_container.training[0m
    [34mINFO:sagemaker-containers:Failed to parse hyperparameter objective value reg:linear to Json.[0m
    [34mReturning the value itself[0m
    [34mINFO:sagemaker-containers:No GPUs detected (normal if no gpus installed)[0m
    [34mINFO:sagemaker_xgboost_container.training:Running XGBoost Sagemaker in algorithm mode[0m
    [34mINFO:root:Determined delimiter of CSV input is ','[0m
    [34mINFO:root:Determined delimiter of CSV input is ','[0m
    [34mINFO:root:Determined delimiter of CSV input is ','[0m
    [34m[20:03:41] 227x13 matrix with 2951 entries loaded from /opt/ml/input/data/train?format=csv&label_column=0&delimiter=,[0m
    [34mINFO:root:Determined delimiter of CSV input is ','[0m
    [34m[20:03:41] 112x13 matrix with 1456 entries loaded from /opt/ml/input/data/validation?format=csv&label_column=0&delimiter=,[0m
    [34mINFO:root:Single node training.[0m
    [34mINFO:root:Train matrix has 227 rows[0m
    [34mINFO:root:Validation matrix has 112 rows[0m
    [34m[20:03:41] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.[0m
    [34m[0]#011train-rmse:19.4101#011validation-rmse:19.455[0m
    [34m[1]#011train-rmse:15.9172#011validation-rmse:15.9642[0m
    [34m[2]#011train-rmse:13.0755#011validation-rmse:13.1838[0m
    [34m[3]#011train-rmse:10.7784#011validation-rmse:10.8775[0m
    [34m[4]#011train-rmse:9.01126#011validation-rmse:9.22599[0m
    [34m[5]#011train-rmse:7.5329#011validation-rmse:7.7625[0m
    [34m[6]#011train-rmse:6.38261#011validation-rmse:6.77231[0m
    [34m[7]#011train-rmse:5.44722#011validation-rmse:5.96523[0m
    [34m[8]#011train-rmse:4.74243#011validation-rmse:5.3923[0m
    [34m[9]#011train-rmse:4.17075#011validation-rmse:4.87945[0m
    [34m[10]#011train-rmse:3.75226#011validation-rmse:4.57763[0m
    [34m[11]#011train-rmse:3.4153#011validation-rmse:4.36624[0m
    [34m[12]#011train-rmse:3.14646#011validation-rmse:4.18212[0m
    [34m[13]#011train-rmse:2.9081#011validation-rmse:3.93189[0m
    [34m[14]#011train-rmse:2.73861#011validation-rmse:3.85636[0m
    [34m[15]#011train-rmse:2.58925#011validation-rmse:3.80584[0m
    [34m[16]#011train-rmse:2.49504#011validation-rmse:3.76151[0m
    [34m[17]#011train-rmse:2.36749#011validation-rmse:3.65107[0m
    [34m[18]#011train-rmse:2.30414#011validation-rmse:3.58724[0m
    [34m[19]#011train-rmse:2.25041#011validation-rmse:3.56694[0m
    [34m[20]#011train-rmse:2.15911#011validation-rmse:3.5627[0m
    [34m[21]#011train-rmse:2.09437#011validation-rmse:3.50853[0m
    [34m[22]#011train-rmse:2.06039#011validation-rmse:3.48886[0m
    [34m[23]#011train-rmse:1.99407#011validation-rmse:3.46745[0m
    [34m[24]#011train-rmse:1.93302#011validation-rmse:3.48091[0m
    [34m[25]#011train-rmse:1.88025#011validation-rmse:3.41624[0m
    [34m[26]#011train-rmse:1.84205#011validation-rmse:3.40218[0m
    [34m[27]#011train-rmse:1.78827#011validation-rmse:3.41428[0m
    [34m[28]#011train-rmse:1.74676#011validation-rmse:3.40677[0m
    [34m[29]#011train-rmse:1.71801#011validation-rmse:3.42138[0m
    [34m[30]#011train-rmse:1.69267#011validation-rmse:3.40769[0m
    [34m[31]#011train-rmse:1.66118#011validation-rmse:3.42434[0m
    [34m[32]#011train-rmse:1.64545#011validation-rmse:3.44295[0m
    [34m[33]#011train-rmse:1.61188#011validation-rmse:3.43728[0m
    [34m[34]#011train-rmse:1.58186#011validation-rmse:3.4114[0m
    [34m[35]#011train-rmse:1.54666#011validation-rmse:3.37226[0m
    [34m[36]#011train-rmse:1.51831#011validation-rmse:3.38013[0m
    [34m[37]#011train-rmse:1.49677#011validation-rmse:3.36203[0m
    [34m[38]#011train-rmse:1.47613#011validation-rmse:3.36326[0m
    [34m[39]#011train-rmse:1.45016#011validation-rmse:3.35582[0m
    [34m[40]#011train-rmse:1.4286#011validation-rmse:3.34719[0m
    [34m[41]#011train-rmse:1.40584#011validation-rmse:3.34375[0m
    [34m[42]#011train-rmse:1.38059#011validation-rmse:3.3298[0m
    [34m[43]#011train-rmse:1.36595#011validation-rmse:3.34506[0m
    [34m[44]#011train-rmse:1.35848#011validation-rmse:3.33167[0m
    [34m[45]#011train-rmse:1.32959#011validation-rmse:3.32496[0m
    [34m[46]#011train-rmse:1.28876#011validation-rmse:3.298[0m
    [34m[47]#011train-rmse:1.25641#011validation-rmse:3.26501[0m
    [34m[48]#011train-rmse:1.23215#011validation-rmse:3.28503[0m
    [34m[49]#011train-rmse:1.18595#011validation-rmse:3.28328[0m
    [34m[50]#011train-rmse:1.1715#011validation-rmse:3.28643[0m
    [34m[51]#011train-rmse:1.15997#011validation-rmse:3.29482[0m
    [34m[52]#011train-rmse:1.13718#011validation-rmse:3.30625[0m
    [34m[53]#011train-rmse:1.1137#011validation-rmse:3.308[0m
    [34m[54]#011train-rmse:1.10111#011validation-rmse:3.30573[0m
    [34m[55]#011train-rmse:1.07602#011validation-rmse:3.28826[0m
    [34m[56]#011train-rmse:1.05402#011validation-rmse:3.2927[0m
    [34m[57]#011train-rmse:1.03073#011validation-rmse:3.28223[0m
    
    2020-03-15 20:04:22 Completed - Training job completed
    Training seconds: 92
    Billable seconds: 92


## Step 5: Test the model

Now that we have fit our model to the training data, using the validation data to avoid overfitting, we can test our model. To do this we will make use of SageMaker's Batch Transform functionality. To start with, we need to build a transformer object from our fit model.


```python
xgb_transformer = xgb.transformer(instance_count = 1, instance_type = 'ml.m4.xlarge')
```

Next we ask SageMaker to begin a batch transform job using our trained model and applying it to the test data we previously stored in S3. We need to make sure to provide SageMaker with the type of data that we are providing to our model, in our case `text/csv`, so that it knows how to serialize our data. In addition, we need to make sure to let SageMaker know how to split our data up into chunks if the entire data set happens to be too large to send to our model all at once.

Note that when we ask SageMaker to do this it will execute the batch transform job in the background. Since we need to wait for the results of this job before we can continue, we use the `wait()` method. An added benefit of this is that we get some output from our batch transform job which lets us know if anything went wrong.


```python
xgb_transformer.transform(test_location, content_type='text/csv', split_type='Line')
```


```python
xgb_transformer.wait()
```

    .....................
    [34m[2020-03-15 20:11:17 +0000] [14] [INFO] Starting gunicorn 19.10.0[0m
    [34m[2020-03-15 20:11:17 +0000] [14] [INFO] Listening at: unix:/tmp/gunicorn.sock (14)[0m
    [34m[2020-03-15 20:11:17 +0000] [14] [INFO] Using worker: gevent[0m
    [34m[2020-03-15 20:11:17 +0000] [21] [INFO] Booting worker with pid: 21[0m
    [34m[2020-03-15 20:11:17 +0000] [22] [INFO] Booting worker with pid: 22[0m
    [34m[2020-03-15 20:11:17 +0000] [26] [INFO] Booting worker with pid: 26[0m
    [34m[2020-03-15 20:11:17 +0000] [27] [INFO] Booting worker with pid: 27[0m
    [34m[2020-03-15:20:11:31:INFO] No GPUs detected (normal if no gpus installed)[0m
    [34m169.254.255.130 - - [15/Mar/2020:20:11:31 +0000] "GET /ping HTTP/1.1" 200 0 "-" "Go-http-client/1.1"[0m
    [34m[2020-03-15:20:11:31:INFO] No GPUs detected (normal if no gpus installed)[0m
    [34m169.254.255.130 - - [15/Mar/2020:20:11:31 +0000] "GET /execution-parameters HTTP/1.1" 200 84 "-" "Go-http-client/1.1"[0m
    [35m[2020-03-15:20:11:31:INFO] No GPUs detected (normal if no gpus installed)[0m
    [35m169.254.255.130 - - [15/Mar/2020:20:11:31 +0000] "GET /ping HTTP/1.1" 200 0 "-" "Go-http-client/1.1"[0m
    [35m[2020-03-15:20:11:31:INFO] No GPUs detected (normal if no gpus installed)[0m
    [35m169.254.255.130 - - [15/Mar/2020:20:11:31 +0000] "GET /execution-parameters HTTP/1.1" 200 84 "-" "Go-http-client/1.1"[0m
    [34m[2020-03-15:20:11:31:INFO] No GPUs detected (normal if no gpus installed)[0m
    [34m[2020-03-15:20:11:31:INFO] Determined delimiter of CSV input is ','[0m
    [34m[20:11:31] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.[0m
    [34m169.254.255.130 - - [15/Mar/2020:20:11:31 +0000] "POST /invocations HTTP/1.1" 200 3110 "-" "Go-http-client/1.1"[0m
    [35m[2020-03-15:20:11:31:INFO] No GPUs detected (normal if no gpus installed)[0m
    [35m[2020-03-15:20:11:31:INFO] Determined delimiter of CSV input is ','[0m
    [35m[20:11:31] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.[0m
    [35m169.254.255.130 - - [15/Mar/2020:20:11:31 +0000] "POST /invocations HTTP/1.1" 200 3110 "-" "Go-http-client/1.1"[0m
    [32m2020-03-15T20:11:31.510:[sagemaker logs]: MaxConcurrentTransforms=4, MaxPayloadInMB=6, BatchStrategy=MULTI_RECORD[0m


Now that the batch transform job has finished, the resulting output is stored on S3. Since we wish to analyze the output inside of our notebook we can use a bit of notebook magic to copy the output file from its S3 location and save it locally.


```python
!aws s3 cp --recursive $xgb_transformer.output_path $data_dir
```

    download: s3://sagemaker-us-west-2-171758673694/sagemaker-xgboost-2020-03-15-20-07-51-049/test.csv.out to ../data/boston/test.csv.out


To see how well our model works we can create a simple scatter plot between the predicted and actual values. If the model was completely accurate the resulting scatter plot would look like the line $x=y$. As we can see, our model seems to have done okay but there is room for improvement.


```python
Y_pred = pd.read_csv(os.path.join(data_dir, 'test.csv.out'), header=None)
```


```python
plt.scatter(Y_test, Y_pred)
plt.xlabel("Median Price")
plt.ylabel("Predicted Price")
plt.title("Median Price vs Predicted Price")
```




    Text(0.5, 1.0, 'Median Price vs Predicted Price')


![png](boston_housing_xgboost_batch_transform_high_level_performance.png)


## Optional: Clean up

The default notebook instance on SageMaker doesn't have a lot of excess disk space available. As you continue to complete and execute notebooks you will eventually fill up this disk space, leading to errors which can be difficult to diagnose. Once you are completely finished using a notebook it is a good idea to remove the files that you created along the way. Of course, you can do this from the terminal or from the notebook hub if you would like. The cell below contains some commands to clean up the created files from within the notebook.


```python
# First we will remove all of the files contained in the data_dir directory
!rm $data_dir/*

# And then we delete the directory itself
!rmdir $data_dir
```


```python

```
