# Predicting Boston Housing Prices

## Using XGBoost in SageMaker (Deploy)

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

In this notebook we will be skipping step 5, testing the model. We will still test the model but we will do so by first deploying the model and then sending the test data to the deployed model.

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

## Step 3: Uploading the training and validation files to S3

When a training job is constructed using SageMaker, a container is executed which performs the training operation. This container is given access to data that is stored in S3. This means that we need to upload the data we want to use for training to S3. We can use the SageMaker API to do this and hide some of the details.

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

Since we are currently running inside of a SageMaker session, we can use the object which represents this session to upload our data to the 'default' S3 bucket. Note that it is good practice to provide a custom prefix (essentially an S3 folder) to make sure that you don't accidentally interfere with data uploaded from some other notebook or project.


```python
prefix = 'boston-xgboost-deploy-hl'

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
xgb = sagemaker.estimator.Estimator(container, # The name of the training container
                                    role,      # The IAM role to use (our current role in this case)
                                    train_instance_count=1, # The number of instances to use for training
                                    train_instance_type='ml.m4.xlarge', # The type of instance ot use for training
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
                        num_round=500)
```

Now that we have our estimator object completely set up, it is time to train it. To do this we make sure that SageMaker knows our input data is in csv format and then execute the `fit` method.


```python
# This is a wrapper around the location of our train and validation data, to make sure that SageMaker
# knows our data is in csv format.
s3_input_train = sagemaker.s3_input(s3_data=train_location, content_type='csv')
s3_input_validation = sagemaker.s3_input(s3_data=val_location, content_type='csv')

xgb.fit({'train': s3_input_train, 'validation': s3_input_validation})
```

    2020-03-19 03:08:45 Starting - Starting the training job...
    2020-03-19 03:08:46 Starting - Launching requested ML instances......
    2020-03-19 03:09:46 Starting - Preparing the instances for training...
    2020-03-19 03:10:37 Downloading - Downloading input data...
    2020-03-19 03:10:55 Training - Downloading the training image...
    2020-03-19 03:11:38 Uploading - Uploading generated training model
    2020-03-19 03:11:38 Completed - Training job completed
    [34mINFO:sagemaker-containers:Imported framework sagemaker_xgboost_container.training[0m
    [34mINFO:sagemaker-containers:Failed to parse hyperparameter objective value reg:linear to Json.[0m
    [34mReturning the value itself[0m
    [34mINFO:sagemaker-containers:No GPUs detected (normal if no gpus installed)[0m
    [34mINFO:sagemaker_xgboost_container.training:Running XGBoost Sagemaker in algorithm mode[0m
    [34mINFO:root:Determined delimiter of CSV input is ','[0m
    [34mINFO:root:Determined delimiter of CSV input is ','[0m
    [34mINFO:root:Determined delimiter of CSV input is ','[0m
    [34m[03:11:28] 227x13 matrix with 2951 entries loaded from /opt/ml/input/data/train?format=csv&label_column=0&delimiter=,[0m
    [34mINFO:root:Determined delimiter of CSV input is ','[0m
    [34m[03:11:28] 112x13 matrix with 1456 entries loaded from /opt/ml/input/data/validation?format=csv&label_column=0&delimiter=,[0m
    [34mINFO:root:Single node training.[0m
    [34mINFO:root:Train matrix has 227 rows[0m
    [34mINFO:root:Validation matrix has 112 rows[0m
    [34m[03:11:28] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.[0m
    [34m[0]#011train-rmse:18.9694#011validation-rmse:20.9535[0m
    [34m[1]#011train-rmse:15.4914#011validation-rmse:17.2711[0m
    [34m[2]#011train-rmse:12.6498#011validation-rmse:14.3746[0m
    [34m[3]#011train-rmse:10.3505#011validation-rmse:11.9134[0m
    [34m[4]#011train-rmse:8.53271#011validation-rmse:10.0906[0m
    [34m[5]#011train-rmse:7.10666#011validation-rmse:8.7177[0m
    [34m[6]#011train-rmse:5.99728#011validation-rmse:7.70446[0m
    [34m[7]#011train-rmse:5.08891#011validation-rmse:6.8883[0m
    [34m[8]#011train-rmse:4.40999#011validation-rmse:6.27442[0m
    [34m[9]#011train-rmse:3.85134#011validation-rmse:5.79958[0m
    [34m[10]#011train-rmse:3.36767#011validation-rmse:5.42279[0m
    [34m[11]#011train-rmse:3.02464#011validation-rmse:5.16116[0m
    [34m[12]#011train-rmse:2.76991#011validation-rmse:4.95674[0m
    [34m[13]#011train-rmse:2.56892#011validation-rmse:4.77436[0m
    [34m[14]#011train-rmse:2.40561#011validation-rmse:4.65357[0m
    [34m[15]#011train-rmse:2.25488#011validation-rmse:4.61727[0m
    [34m[16]#011train-rmse:2.16823#011validation-rmse:4.58404[0m
    [34m[17]#011train-rmse:2.07948#011validation-rmse:4.51897[0m
    [34m[18]#011train-rmse:2.00937#011validation-rmse:4.46557[0m
    [34m[19]#011train-rmse:1.91604#011validation-rmse:4.36764[0m
    [34m[20]#011train-rmse:1.86537#011validation-rmse:4.36345[0m
    [34m[21]#011train-rmse:1.78295#011validation-rmse:4.35817[0m
    [34m[22]#011train-rmse:1.73125#011validation-rmse:4.31199[0m
    [34m[23]#011train-rmse:1.69365#011validation-rmse:4.32532[0m
    [34m[24]#011train-rmse:1.64172#011validation-rmse:4.34406[0m
    [34m[25]#011train-rmse:1.61365#011validation-rmse:4.34449[0m
    [34m[26]#011train-rmse:1.56424#011validation-rmse:4.32601[0m
    [34m[27]#011train-rmse:1.51077#011validation-rmse:4.30575[0m
    [34m[28]#011train-rmse:1.47668#011validation-rmse:4.27829[0m
    [34m[29]#011train-rmse:1.44877#011validation-rmse:4.2785[0m
    [34m[30]#011train-rmse:1.39448#011validation-rmse:4.28024[0m
    [34m[31]#011train-rmse:1.35515#011validation-rmse:4.24026[0m
    [34m[32]#011train-rmse:1.33144#011validation-rmse:4.25461[0m
    [34m[33]#011train-rmse:1.3061#011validation-rmse:4.25995[0m
    [34m[34]#011train-rmse:1.28479#011validation-rmse:4.26455[0m
    [34m[35]#011train-rmse:1.249#011validation-rmse:4.30341[0m
    [34m[36]#011train-rmse:1.22055#011validation-rmse:4.31478[0m
    [34m[37]#011train-rmse:1.19374#011validation-rmse:4.2837[0m
    [34m[38]#011train-rmse:1.17948#011validation-rmse:4.26876[0m
    [34m[39]#011train-rmse:1.16107#011validation-rmse:4.25077[0m
    [34m[40]#011train-rmse:1.15132#011validation-rmse:4.24958[0m
    [34m[41]#011train-rmse:1.10926#011validation-rmse:4.22628[0m
    [34m[42]#011train-rmse:1.089#011validation-rmse:4.21181[0m
    [34m[43]#011train-rmse:1.07397#011validation-rmse:4.20253[0m
    [34m[44]#011train-rmse:1.0612#011validation-rmse:4.20834[0m
    [34m[45]#011train-rmse:1.04567#011validation-rmse:4.20017[0m
    [34m[46]#011train-rmse:1.01681#011validation-rmse:4.21658[0m
    [34m[47]#011train-rmse:0.984182#011validation-rmse:4.22414[0m
    [34m[48]#011train-rmse:0.971596#011validation-rmse:4.20906[0m
    [34m[49]#011train-rmse:0.954075#011validation-rmse:4.19286[0m
    [34m[50]#011train-rmse:0.934885#011validation-rmse:4.19715[0m
    [34m[51]#011train-rmse:0.918098#011validation-rmse:4.19439[0m
    [34m[52]#011train-rmse:0.906141#011validation-rmse:4.19102[0m
    [34m[53]#011train-rmse:0.906141#011validation-rmse:4.19104[0m
    [34m[54]#011train-rmse:0.897875#011validation-rmse:4.1724[0m
    [34m[55]#011train-rmse:0.896361#011validation-rmse:4.17338[0m
    [34m[56]#011train-rmse:0.891424#011validation-rmse:4.16723[0m
    [34m[57]#011train-rmse:0.891447#011validation-rmse:4.16664[0m
    [34m[58]#011train-rmse:0.891504#011validation-rmse:4.16599[0m
    [34m[59]#011train-rmse:0.885453#011validation-rmse:4.17862[0m
    [34m[60]#011train-rmse:0.885431#011validation-rmse:4.17876[0m
    [34m[61]#011train-rmse:0.885312#011validation-rmse:4.17987[0m
    [34m[62]#011train-rmse:0.885296#011validation-rmse:4.18024[0m
    [34m[63]#011train-rmse:0.869621#011validation-rmse:4.18436[0m
    [34m[64]#011train-rmse:0.864992#011validation-rmse:4.18805[0m
    [34m[65]#011train-rmse:0.861961#011validation-rmse:4.19922[0m
    [34m[66]#011train-rmse:0.861973#011validation-rmse:4.19975[0m
    [34m[67]#011train-rmse:0.861972#011validation-rmse:4.19972[0m
    [34m[68]#011train-rmse:0.851846#011validation-rmse:4.21148[0m
    Training seconds: 61
    Billable seconds: 61


## Step 5: Test the trained model

We will be skipping this step for now. We will still test our trained model but we are going to do it by using the deployed model, rather than setting up a batch transform job.


## Step 6: Deploy the trained model

Now that we have fit our model to the training data, using the validation data to avoid overfitting, we can deploy our model and test it. Deploying is very simple when we use the high level API, we need only call the `deploy` method of our trained estimator.

**NOTE:** When deploying a model you are asking SageMaker to launch an compute instance that will wait for data to be sent to it. As a result, this compute instance will continue to run until *you* shut it down. This is important to know since the cost of a deployed endpoint depends on how long it has been running for.

In other words **If you are no longer using a deployed endpoint, shut it down!**


```python
xgb_predictor = xgb.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')
```

    ---------------!

## Step 7: Use the model

Now that our model is trained and deployed we can send the test data to it and evaluate the results. Here, because our test data is so small, we can send it all using a single call to our endpoint. If our test dataset was larger we would need to split it up and send the data in chunks, making sure to accumulate the results.


```python
# We need to tell the endpoint what format the data we are sending is in
xgb_predictor.content_type = 'text/csv'
xgb_predictor.serializer = csv_serializer

Y_pred = xgb_predictor.predict(X_test.values).decode('utf-8')
# predictions is currently a comma delimited string and so we would like to break it up
# as a numpy array.
Y_pred = np.fromstring(Y_pred, sep=',')
```

To see how well our model works we can create a simple scatter plot between the predicted and actual values. If the model was completely accurate the resulting scatter plot would look like the line $x=y$. As we can see, our model seems to have done okay but there is room for improvement.


```python
plt.scatter(Y_test, Y_pred)
plt.xlabel("Median Price")
plt.ylabel("Predicted Price")
plt.title("Median Price vs Predicted Price")
```




    Text(0.5, 1.0, 'Median Price vs Predicted Price')




![png](boston_housing_xgboost_deploy_high_level_api_performance.png)


## Delete the endpoint

Since we are no longer using the deployed model we need to make sure to shut it down. Remember that you have to pay for the length of time that your endpoint is deployed so the longer it is left running, the more it costs.


```python
xgb_predictor.delete_endpoint()
```

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
