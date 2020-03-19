# Predicting Boston Housing Prices

## Using XGBoost in SageMaker (Deploy)

As an introduction to using SageMaker's Low Level Python API we will look at a relatively simple problem. Namely, we will use the [Boston Housing Dataset](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html) to predict the median value of a home in the area of Boston Mass.

The documentation reference for the API used in this notebook is the [SageMaker Developer's Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/)

## General Outline

Typically, when using a notebook instance with SageMaker, you will proceed through the following steps. Of course, not every step will need to be done with each project. Also, there is quite a lot of room for variation in many of the steps, as you will see throughout these lessons.

1. Download or otherwise retrieve the data.
2. Process / Prepare the data.
3. Upload the processed data to S3.
4. Train a chosen model.
5. Test the trained model (typically using a batch transform job).
6. Deploy the trained model.
7. Use the deployed model.

In this notebook we will be skipping step 5, testing the model. We will still test the model but we will do so by first deploying it and then sending the test data to the deployed model.

## Step 0: Setting up the notebook

We begin by setting up all of the necessary bits required to run our notebook. To start that means loading all of the Python modules we will need.


```python
%matplotlib inline

import os

import time
from time import gmtime, strftime

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
prefix = 'boston-xgboost-deploy-ll'

val_location = session.upload_data(os.path.join(data_dir, 'validation.csv'), key_prefix=prefix)
train_location = session.upload_data(os.path.join(data_dir, 'train.csv'), key_prefix=prefix)
```

## Step 4: Train and construct the XGBoost model

Now that we have the training and validation data uploaded to S3, we can construct a training job for our XGBoost model and build the model itself.

### Set up the training job

First, we will set up and execute a training job for our model. To do this we need to specify some information that SageMaker will use to set up and properly execute the computation. For additional documentation on constructing a training job, see the [CreateTrainingJob API](https://docs.aws.amazon.com/sagemaker/latest/dg/API_CreateTrainingJob.html) reference.


```python
# We will need to know the name of the container that we want to use for training. SageMaker provides
# a nice utility method to construct this for us.
container = get_image_uri(session.boto_region_name, 'xgboost', '0.90-1')

# We now specify the parameters we wish to use for our training job
training_params = {}

# We need to specify the permissions that this training job will have. For our purposes we can use
# the same permissions that our current SageMaker session has.
training_params['RoleArn'] = role

# Here we describe the algorithm we wish to use. The most important part is the container which
# contains the training code.
training_params['AlgorithmSpecification'] = {
    "TrainingImage": container,
    "TrainingInputMode": "File"
}

# We also need to say where we would like the resulting model artifacst stored.
training_params['OutputDataConfig'] = {
    "S3OutputPath": "s3://" + session.default_bucket() + "/" + prefix + "/output"
}

# We also need to set some parameters for the training job itself. Namely we need to describe what sort of
# compute instance we wish to use along with a stopping condition to handle the case that there is
# some sort of error and the training script doesn't terminate.
training_params['ResourceConfig'] = {
    "InstanceCount": 1,
    "InstanceType": "ml.m4.xlarge",
    "VolumeSizeInGB": 5
}
    
training_params['StoppingCondition'] = {
    "MaxRuntimeInSeconds": 86400
}

# Next we set the algorithm specific hyperparameters. You may wish to change these to see what effect
# there is on the resulting model.
training_params['HyperParameters'] = {
    "max_depth": "5",
    "eta": "0.2",
    "gamma": "4",
    "min_child_weight": "6",
    "subsample": "0.8",
    "objective": "reg:linear",
    "early_stopping_rounds": "10",
    "num_round": "200"
}

# Now we need to tell SageMaker where the data should be retrieved from.
training_params['InputDataConfig'] = [
    {
        "ChannelName": "train",
        "DataSource": {
            "S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": train_location,
                "S3DataDistributionType": "FullyReplicated"
            }
        },
        "ContentType": "csv",
        "CompressionType": "None"
    },
    {
        "ChannelName": "validation",
        "DataSource": {
            "S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": val_location,
                "S3DataDistributionType": "FullyReplicated"
            }
        },
        "ContentType": "csv",
        "CompressionType": "None"
    }
]
```

### Execute the training job

Now that we've built the dict containing the training job parameters, we can ask SageMaker to execute the job.


```python
# First we need to choose a training job name. This is useful for if we want to recall information about our
# training job at a later date. Note that SageMaker requires a training job name and that the name needs to
# be unique, which we accomplish by appending the current timestamp.
training_job_name = "boston-xgboost-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
training_params['TrainingJobName'] = training_job_name

# And now we ask SageMaker to create (and execute) the training job
training_job = session.sagemaker_client.create_training_job(**training_params)
```

The training job has now been created by SageMaker and is currently running. Since we need the output of the training job, we may wish to wait until it has finished. We can do so by asking SageMaker to output the logs generated by the training job and continue doing so until the training job terminates.


```python
session.logs_for_job(training_job_name, wait=True)
```

    2020-03-19 19:03:56 Starting - Preparing the instances for training
    2020-03-19 19:03:56 Downloading - Downloading input data
    2020-03-19 19:03:56 Training - Training image download completed. Training in progress.
    2020-03-19 19:03:56 Uploading - Uploading generated training model
    2020-03-19 19:03:56 Completed - Training job completed[34mINFO:sagemaker-containers:Imported framework sagemaker_xgboost_container.training[0m
    [34mINFO:sagemaker-containers:Failed to parse hyperparameter objective value reg:linear to Json.[0m
    [34mReturning the value itself[0m
    [34mINFO:sagemaker-containers:No GPUs detected (normal if no gpus installed)[0m
    [34mINFO:sagemaker_xgboost_container.training:Running XGBoost Sagemaker in algorithm mode[0m
    [34mINFO:root:Determined delimiter of CSV input is ','[0m
    [34mINFO:root:Determined delimiter of CSV input is ','[0m
    [34mINFO:root:Determined delimiter of CSV input is ','[0m
    [34m[19:03:46] 227x13 matrix with 2951 entries loaded from /opt/ml/input/data/train?format=csv&label_column=0&delimiter=,[0m
    [34mINFO:root:Determined delimiter of CSV input is ','[0m
    [34m[19:03:46] 112x13 matrix with 1456 entries loaded from /opt/ml/input/data/validation?format=csv&label_column=0&delimiter=,[0m
    [34mINFO:root:Single node training.[0m
    [34mINFO:root:Train matrix has 227 rows[0m
    [34mINFO:root:Validation matrix has 112 rows[0m
    [34m[19:03:46] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.[0m
    [34m[0]#011train-rmse:19.374#011validation-rmse:20.5943[0m
    [34m[1]#011train-rmse:15.9056#011validation-rmse:17.0974[0m
    [34m[2]#011train-rmse:13.0565#011validation-rmse:14.288[0m
    [34m[3]#011train-rmse:10.758#011validation-rmse:11.9413[0m
    [34m[4]#011train-rmse:8.95718#011validation-rmse:10.1117[0m
    [34m[5]#011train-rmse:7.44084#011validation-rmse:8.71961[0m
    [34m[6]#011train-rmse:6.29671#011validation-rmse:7.59981[0m
    [34m[7]#011train-rmse:5.337#011validation-rmse:6.79843[0m
    [34m[8]#011train-rmse:4.56975#011validation-rmse:6.16519[0m
    [34m[9]#011train-rmse:3.98444#011validation-rmse:5.7423[0m
    [34m[10]#011train-rmse:3.55828#011validation-rmse:5.3794[0m
    [34m[11]#011train-rmse:3.19943#011validation-rmse:5.25268[0m
    [34m[12]#011train-rmse:2.91466#011validation-rmse:5.12411[0m
    [34m[13]#011train-rmse:2.72112#011validation-rmse:4.98549[0m
    [34m[14]#011train-rmse:2.52299#011validation-rmse:4.85848[0m
    [34m[15]#011train-rmse:2.37062#011validation-rmse:4.82071[0m
    [34m[16]#011train-rmse:2.246#011validation-rmse:4.76704[0m
    [34m[17]#011train-rmse:2.13062#011validation-rmse:4.71539[0m
    [34m[18]#011train-rmse:2.04975#011validation-rmse:4.62498[0m
    [34m[19]#011train-rmse:1.98112#011validation-rmse:4.6257[0m
    [34m[20]#011train-rmse:1.92697#011validation-rmse:4.59102[0m
    [34m[21]#011train-rmse:1.86348#011validation-rmse:4.55636[0m
    [34m[22]#011train-rmse:1.8042#011validation-rmse:4.52745[0m
    [34m[23]#011train-rmse:1.75261#011validation-rmse:4.48974[0m
    [34m[24]#011train-rmse:1.69304#011validation-rmse:4.47752[0m
    [34m[25]#011train-rmse:1.64411#011validation-rmse:4.45976[0m
    [34m[26]#011train-rmse:1.61713#011validation-rmse:4.45793[0m
    [34m[27]#011train-rmse:1.60257#011validation-rmse:4.4527[0m
    [34m[28]#011train-rmse:1.54404#011validation-rmse:4.41863[0m
    [34m[29]#011train-rmse:1.51115#011validation-rmse:4.43804[0m
    [34m[30]#011train-rmse:1.47374#011validation-rmse:4.43461[0m
    [34m[31]#011train-rmse:1.45142#011validation-rmse:4.43122[0m
    [34m[32]#011train-rmse:1.43526#011validation-rmse:4.42464[0m
    [34m[33]#011train-rmse:1.38896#011validation-rmse:4.43223[0m
    [34m[34]#011train-rmse:1.36995#011validation-rmse:4.41703[0m
    [34m[35]#011train-rmse:1.35155#011validation-rmse:4.39074[0m
    [34m[36]#011train-rmse:1.32656#011validation-rmse:4.37263[0m
    [34m[37]#011train-rmse:1.30165#011validation-rmse:4.34523[0m
    [34m[38]#011train-rmse:1.26907#011validation-rmse:4.31869[0m
    [34m[39]#011train-rmse:1.24855#011validation-rmse:4.28489[0m
    [34m[40]#011train-rmse:1.22209#011validation-rmse:4.28453[0m
    [34m[41]#011train-rmse:1.1944#011validation-rmse:4.26412[0m
    [34m[42]#011train-rmse:1.1645#011validation-rmse:4.2729[0m
    [34m[43]#011train-rmse:1.15114#011validation-rmse:4.2923[0m
    [34m[44]#011train-rmse:1.13976#011validation-rmse:4.26943[0m
    [34m[45]#011train-rmse:1.11477#011validation-rmse:4.24267[0m
    [34m[46]#011train-rmse:1.09192#011validation-rmse:4.21242[0m
    [34m[47]#011train-rmse:1.0803#011validation-rmse:4.18807[0m
    [34m[48]#011train-rmse:1.05458#011validation-rmse:4.17345[0m
    [34m[49]#011train-rmse:1.03668#011validation-rmse:4.16559[0m
    [34m[50]#011train-rmse:1.02707#011validation-rmse:4.16014[0m
    [34m[51]#011train-rmse:1.01894#011validation-rmse:4.14964[0m
    [34m[52]#011train-rmse:1.00347#011validation-rmse:4.17238[0m
    [34m[53]#011train-rmse:0.996684#011validation-rmse:4.16841[0m
    [34m[54]#011train-rmse:0.996802#011validation-rmse:4.16751[0m
    [34m[55]#011train-rmse:0.996631#011validation-rmse:4.16899[0m
    [34m[56]#011train-rmse:0.983375#011validation-rmse:4.17121[0m
    [34m[57]#011train-rmse:0.97256#011validation-rmse:4.15505[0m
    [34m[58]#011train-rmse:0.959565#011validation-rmse:4.15559[0m
    [34m[59]#011train-rmse:0.934151#011validation-rmse:4.15709[0m
    [34m[60]#011train-rmse:0.934144#011validation-rmse:4.15846[0m
    [34m[61]#011train-rmse:0.916708#011validation-rmse:4.15852[0m
    Training seconds: 59
    Billable seconds: 59


### Build the model

Now that the training job has completed, we have some model artifacts which we can use to build a model. Note that here we mean SageMaker's definition of a model, which is a collection of information about a specific algorithm along with the artifacts which result from a training job.


```python
# We begin by asking SageMaker to describe for us the results of the training job. The data structure
# returned contains a lot more information than we currently need, try checking it out yourself in
# more detail.
training_job_info = session.sagemaker_client.describe_training_job(TrainingJobName=training_job_name)

model_artifacts = training_job_info['ModelArtifacts']['S3ModelArtifacts']
```


```python
# Just like when we created a training job, the model name must be unique
model_name = training_job_name + "-model"

# We also need to tell SageMaker which container should be used for inference and where it should
# retrieve the model artifacts from. In our case, the xgboost container that we used for training
# can also be used for inference.
primary_container = {
    "Image": container,
    "ModelDataUrl": model_artifacts
}

# And lastly we construct the SageMaker model
model_info = session.sagemaker_client.create_model(
                                ModelName = model_name,
                                ExecutionRoleArn = role,
                                PrimaryContainer = primary_container)
```

## Step 5: Test the trained model

We will be skipping this step for now. We will still test our trained model but we are going to do it by using the deployed model, rather than setting up a batch transform job.

## Step 6: Create and deploy the endpoint

Now that we have trained and constructed a model it is time to build the associated endpoint and deploy it. As in the earlier steps, we first need to construct the appropriate configuration.


```python
# As before, we need to give our endpoint configuration a name which should be unique
endpoint_config_name = "boston-xgboost-endpoint-config-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

# And then we ask SageMaker to construct the endpoint configuration
endpoint_config_info = session.sagemaker_client.create_endpoint_config(
                            EndpointConfigName = endpoint_config_name,
                            ProductionVariants = [{
                                "InstanceType": "ml.m4.xlarge",
                                "InitialVariantWeight": 1,
                                "InitialInstanceCount": 1,
                                "ModelName": model_name,
                                "VariantName": "AllTraffic"
                            }])
```

And now that the endpoint configuration has been created we can deploy the endpoint itself.

**NOTE:** When deploying a model you are asking SageMaker to launch an compute instance that will wait for data to be sent to it. As a result, this compute instance will continue to run until *you* shut it down. This is important to know since the cost of a deployed endpoint depends on how long it has been running for.

In other words **If you are no longer using a deployed endpoint, shut it down!**


```python
# Again, we need a unique name for our endpoint
endpoint_name = "boston-xgboost-endpoint-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

# And then we can deploy our endpoint
endpoint_info = session.sagemaker_client.create_endpoint(
                    EndpointName = endpoint_name,
                    EndpointConfigName = endpoint_config_name)
```

Just like when we created a training job, SageMaker is now requisitioning and launching our endpoint. Since we can't do much until the endpoint has been completely deployed we can wait for it to finish.


```python
endpoint_dec = session.wait_for_endpoint(endpoint_name)
```

    -----------!

## Step 7: Use the model

Now that our model is trained and deployed we can send test data to it and evaluate the results. Here, because our test data is so small, we can send it all using a single call to our endpoint. If our test dataset was larger we would need to split it up and send the data in chunks, making sure to accumulate the results.


```python
# First we need to serialize the input data. In this case we want to send the test data as a csv and
# so we manually do this. Of course, there are many other ways to do this.
payload = [[str(entry) for entry in row] for row in X_test.values]
print("Number of inputs to submit for prediction: {}".format(len(payload)))
payload = '\n'.join([','.join(row) for row in payload])
```

    Number of inputs to submit for prediction: 167


This is example of the request body, which is a CSV file where each row is an input.


```python
print(payload)
```

    0.09266,34.0,6.09,0.0,0.433,6.495,18.4,5.4917,7.0,329.0,16.1,383.61,8.67
    6.28807,0.0,18.1,0.0,0.74,6.341,96.4,2.072,24.0,666.0,20.2,318.01,17.79
    0.03551,25.0,4.86,0.0,0.426,6.167,46.7,5.4007,4.0,281.0,19.0,390.64,7.51
    0.05059,0.0,4.49,0.0,0.449,6.389,48.0,4.7794,3.0,247.0,18.5,396.9,9.62
    5.29305,0.0,18.1,0.0,0.7,6.051,82.5,2.1678,24.0,666.0,20.2,378.38,18.76
    1.22358,0.0,19.58,0.0,0.605,6.943,97.4,1.8773,5.0,403.0,14.7,363.43,4.59
    0.22438,0.0,9.69,0.0,0.585,6.027,79.7,2.4982,6.0,391.0,19.2,396.9,14.33
    6.39312,0.0,18.1,0.0,0.584,6.162,97.4,2.206,24.0,666.0,20.2,302.76,24.1
    0.03705,20.0,3.33,0.0,0.4429,6.968,37.2,5.2447,5.0,216.0,14.9,392.23,4.59
    7.05042,0.0,18.1,0.0,0.614,6.103,85.1,2.0218,24.0,666.0,20.2,2.52,23.29
    8.15174,0.0,18.1,0.0,0.7,5.39,98.9,1.7281,24.0,666.0,20.2,396.9,20.85
    0.11504,0.0,2.89,0.0,0.445,6.163,69.6,3.4952,2.0,276.0,18.0,391.83,11.34
    24.8017,0.0,18.1,0.0,0.693,5.349,96.0,1.7028,24.0,666.0,20.2,396.9,19.77
    0.03548,80.0,3.64,0.0,0.392,5.876,19.1,9.2203,1.0,315.0,16.4,395.18,9.25
    0.05789,12.5,6.07,0.0,0.409,5.878,21.4,6.498,4.0,345.0,18.9,396.21,8.1
    0.27957,0.0,9.69,0.0,0.585,5.926,42.6,2.3817,6.0,391.0,19.2,396.9,13.59
    11.1604,0.0,18.1,0.0,0.74,6.629,94.6,2.1247,24.0,666.0,20.2,109.85,23.27
    0.04527,0.0,11.93,0.0,0.573,6.12,76.7,2.2875,1.0,273.0,21.0,396.9,9.08
    0.1415,0.0,6.91,0.0,0.448,6.169,6.6,5.7209,3.0,233.0,17.9,383.37,5.81
    0.0795,60.0,1.69,0.0,0.411,6.579,35.9,10.7103,4.0,411.0,18.3,370.78,5.49
    0.03237,0.0,2.18,0.0,0.458,6.998,45.8,6.0622,3.0,222.0,18.7,394.63,2.94
    0.55007,20.0,3.97,0.0,0.647,7.206,91.6,1.9301,5.0,264.0,13.0,387.89,8.1
    0.17446,0.0,10.59,1.0,0.489,5.96,92.1,3.8771,4.0,277.0,18.6,393.25,17.27
    0.08265,0.0,13.92,0.0,0.437,6.127,18.4,5.5027,4.0,289.0,16.0,396.9,8.58
    15.8744,0.0,18.1,0.0,0.671,6.545,99.1,1.5192,24.0,666.0,20.2,396.9,21.08
    37.6619,0.0,18.1,0.0,0.679,6.202,78.7,1.8629,24.0,666.0,20.2,18.82,14.52
    0.7842,0.0,8.14,0.0,0.538,5.99,81.7,4.2579,4.0,307.0,21.0,386.75,14.67
    0.22876,0.0,8.56,0.0,0.52,6.405,85.4,2.7147,5.0,384.0,20.9,70.8,10.63
    1.13081,0.0,8.14,0.0,0.538,5.713,94.1,4.233,4.0,307.0,21.0,360.17,22.6
    0.21161,0.0,8.56,0.0,0.52,6.137,87.4,2.7147,5.0,384.0,20.9,394.47,13.44
    3.83684,0.0,18.1,0.0,0.77,6.251,91.1,2.2955,24.0,666.0,20.2,350.65,14.19
    0.10469,40.0,6.41,1.0,0.447,7.267,49.0,4.7872,4.0,254.0,17.6,389.25,6.05
    0.0578,0.0,2.46,0.0,0.488,6.98,58.4,2.829,3.0,193.0,17.8,396.9,5.04
    0.01501,90.0,1.21,1.0,0.401,7.923,24.8,5.885,1.0,198.0,13.6,395.52,3.16
    0.05188,0.0,4.49,0.0,0.449,6.015,45.1,4.4272,3.0,247.0,18.5,395.99,12.86
    0.03466,35.0,6.06,0.0,0.4379,6.031,23.3,6.6407,1.0,304.0,16.9,362.25,7.83
    0.19802,0.0,10.59,0.0,0.489,6.182,42.4,3.9454,4.0,277.0,18.6,393.63,9.47
    4.89822,0.0,18.1,0.0,0.631,4.97,100.0,1.3325,24.0,666.0,20.2,375.52,3.26
    2.81838,0.0,18.1,0.0,0.532,5.762,40.3,4.0983,24.0,666.0,20.2,392.92,10.42
    0.01951,17.5,1.38,0.0,0.4161,7.104,59.5,9.2229,3.0,216.0,18.6,393.24,8.05
    5.70818,0.0,18.1,0.0,0.532,6.75,74.9,3.3317,24.0,666.0,20.2,393.07,7.74
    4.87141,0.0,18.1,0.0,0.614,6.484,93.6,2.3053,24.0,666.0,20.2,396.21,18.68
    0.0459,52.5,5.32,0.0,0.405,6.315,45.6,7.3172,6.0,293.0,16.6,396.9,7.6
    0.03049,55.0,3.78,0.0,0.484,6.874,28.1,6.4654,5.0,370.0,17.6,387.97,4.61
    0.04544,0.0,3.24,0.0,0.46,6.144,32.2,5.8736,4.0,430.0,16.9,368.57,9.09
    7.40389,0.0,18.1,0.0,0.597,5.617,97.9,1.4547,24.0,666.0,20.2,314.64,26.4
    0.08664,45.0,3.44,0.0,0.437,7.178,26.3,6.4798,5.0,398.0,15.2,390.49,2.87
    0.04337,21.0,5.64,0.0,0.439,6.115,63.0,6.8147,4.0,243.0,16.8,393.97,9.43
    7.83932,0.0,18.1,0.0,0.655,6.209,65.4,2.9634,24.0,666.0,20.2,396.9,13.22
    7.67202,0.0,18.1,0.0,0.693,5.747,98.9,1.6334,24.0,666.0,20.2,393.1,19.92
    0.01439,60.0,2.93,0.0,0.401,6.604,18.8,6.2196,1.0,265.0,15.6,376.7,4.38
    0.12932,0.0,13.92,0.0,0.437,6.678,31.1,5.9604,4.0,289.0,16.0,396.9,6.27
    0.05023,35.0,6.06,0.0,0.4379,5.706,28.4,6.6407,1.0,304.0,16.9,394.02,12.43
    10.6718,0.0,18.1,0.0,0.74,6.459,94.8,1.9879,24.0,666.0,20.2,43.06,23.98
    0.12744,0.0,6.91,0.0,0.448,6.77,2.9,5.7209,3.0,233.0,17.9,385.41,4.84
    13.3598,0.0,18.1,0.0,0.693,5.887,94.7,1.7821,24.0,666.0,20.2,396.9,16.35
    0.12083,0.0,2.89,0.0,0.445,8.069,76.0,3.4952,2.0,276.0,18.0,396.9,4.21
    0.38214,0.0,6.2,0.0,0.504,8.04,86.5,3.2157,8.0,307.0,17.4,387.38,3.13
    1.34284,0.0,19.58,0.0,0.605,6.066,100.0,1.7573,5.0,403.0,14.7,353.89,6.43
    0.26363,0.0,8.56,0.0,0.52,6.229,91.2,2.5451,5.0,384.0,20.9,391.23,15.55
    12.8023,0.0,18.1,0.0,0.74,5.854,96.6,1.8956,24.0,666.0,20.2,240.52,23.79
    14.4383,0.0,18.1,0.0,0.597,6.852,100.0,1.4655,24.0,666.0,20.2,179.36,19.78
    5.44114,0.0,18.1,0.0,0.713,6.655,98.2,2.3552,24.0,666.0,20.2,355.29,17.73
    1.35472,0.0,8.14,0.0,0.538,6.072,100.0,4.175,4.0,307.0,21.0,376.73,13.04
    0.03537,34.0,6.09,0.0,0.433,6.59,40.4,5.4917,7.0,329.0,16.1,395.75,9.5
    0.0536,21.0,5.64,0.0,0.439,6.511,21.1,6.8147,4.0,243.0,16.8,396.9,5.28
    0.14476,0.0,10.01,0.0,0.547,5.731,65.2,2.7592,6.0,432.0,17.8,391.5,13.61
    0.18159,0.0,7.38,0.0,0.493,6.376,54.3,4.5404,5.0,287.0,19.6,396.9,6.87
    0.0456,0.0,13.89,1.0,0.55,5.888,56.0,3.1121,5.0,276.0,16.4,392.8,13.51
    9.33889,0.0,18.1,0.0,0.679,6.38,95.6,1.9682,24.0,666.0,20.2,60.72,24.08
    0.04981,21.0,5.64,0.0,0.439,5.998,21.4,6.8147,4.0,243.0,16.8,396.9,8.43
    51.1358,0.0,18.1,0.0,0.597,5.757,100.0,1.413,24.0,666.0,20.2,2.6,10.11
    14.3337,0.0,18.1,0.0,0.614,6.229,88.0,1.9512,24.0,666.0,20.2,383.32,13.11
    0.22188,20.0,6.96,1.0,0.464,7.691,51.8,4.3665,3.0,223.0,18.6,390.77,6.58
    45.7461,0.0,18.1,0.0,0.693,4.519,100.0,1.6582,24.0,666.0,20.2,88.27,36.98
    25.0461,0.0,18.1,0.0,0.693,5.987,100.0,1.5888,24.0,666.0,20.2,396.9,26.77
    0.65665,20.0,3.97,0.0,0.647,6.842,100.0,2.0107,5.0,264.0,13.0,391.93,6.9
    0.05083,0.0,5.19,0.0,0.515,6.316,38.1,6.4584,5.0,224.0,20.2,389.71,5.68
    14.2362,0.0,18.1,0.0,0.693,6.343,100.0,1.5741,24.0,666.0,20.2,396.9,20.32
    0.01432,100.0,1.32,0.0,0.411,6.816,40.5,8.3248,5.0,256.0,15.1,392.9,3.95
    0.15445,25.0,5.13,0.0,0.453,6.145,29.2,7.8148,8.0,284.0,19.7,390.68,6.86
    0.51183,0.0,6.2,0.0,0.507,7.358,71.6,4.148,8.0,307.0,17.4,390.07,4.73
    0.06617,0.0,3.24,0.0,0.46,5.868,25.8,5.2146,4.0,430.0,16.9,382.44,9.97
    0.35233,0.0,21.89,0.0,0.624,6.454,98.4,1.8498,4.0,437.0,21.2,394.08,14.59
    2.63548,0.0,9.9,0.0,0.544,4.973,37.8,2.5194,4.0,304.0,18.4,350.45,12.64
    22.5971,0.0,18.1,0.0,0.7,5.0,89.5,1.5184,24.0,666.0,20.2,396.9,31.99
    0.00632,18.0,2.31,0.0,0.538,6.575,65.2,4.09,1.0,296.0,15.3,396.9,4.98
    0.0686,0.0,2.89,0.0,0.445,7.416,62.5,3.4952,2.0,276.0,18.0,396.9,6.19
    0.24522,0.0,9.9,0.0,0.544,5.782,71.7,4.0317,4.0,304.0,18.4,396.9,15.94
    5.66637,0.0,18.1,0.0,0.74,6.219,100.0,2.0048,24.0,666.0,20.2,395.69,16.59
    0.17331,0.0,9.69,0.0,0.585,5.707,54.0,2.3817,6.0,391.0,19.2,396.9,12.01
    0.33983,22.0,5.86,0.0,0.431,6.108,34.9,8.0555,7.0,330.0,19.1,390.18,9.16
    9.39063,0.0,18.1,0.0,0.74,5.627,93.9,1.8172,24.0,666.0,20.2,396.9,22.88
    0.01096,55.0,2.25,0.0,0.389,6.453,31.9,7.3073,1.0,300.0,15.3,394.72,8.23
    0.06129,20.0,3.33,1.0,0.4429,7.645,49.7,5.2119,5.0,216.0,14.9,377.07,3.01
    15.0234,0.0,18.1,0.0,0.614,5.304,97.3,2.1007,24.0,666.0,20.2,349.48,24.91
    0.0136,75.0,4.0,0.0,0.41,5.888,47.6,7.3197,3.0,469.0,21.1,396.9,14.8
    0.03427,0.0,5.19,0.0,0.515,5.869,46.3,5.2311,5.0,224.0,20.2,396.9,9.8
    0.28955,0.0,10.59,0.0,0.489,5.412,9.8,3.5875,4.0,277.0,18.6,348.93,29.55
    23.6482,0.0,18.1,0.0,0.671,6.38,96.2,1.3861,24.0,666.0,20.2,396.9,23.69
    0.13158,0.0,10.01,0.0,0.547,6.176,72.5,2.7301,6.0,432.0,17.8,393.3,12.04
    18.811,0.0,18.1,0.0,0.597,4.628,100.0,1.5539,24.0,666.0,20.2,28.79,34.37
    0.2896,0.0,9.69,0.0,0.585,5.39,72.9,2.7986,6.0,391.0,19.2,396.9,21.14
    0.12816,12.5,6.07,0.0,0.409,5.885,33.0,6.498,4.0,345.0,18.9,396.9,8.79
    0.10008,0.0,2.46,0.0,0.488,6.563,95.6,2.847,3.0,193.0,17.8,396.9,5.68
    3.53501,0.0,19.58,1.0,0.871,6.152,82.6,1.7455,5.0,403.0,14.7,88.01,15.02
    0.34006,0.0,21.89,0.0,0.624,6.458,98.9,2.1185,4.0,437.0,21.2,395.04,12.6
    14.4208,0.0,18.1,0.0,0.74,6.461,93.3,2.0026,24.0,666.0,20.2,27.49,18.05
    0.13262,0.0,8.56,0.0,0.52,5.851,96.7,2.1069,5.0,384.0,20.9,394.05,16.47
    0.22927,0.0,6.91,0.0,0.448,6.03,85.5,5.6894,3.0,233.0,17.9,392.74,18.8
    1.15172,0.0,8.14,0.0,0.538,5.701,95.0,3.7872,4.0,307.0,21.0,358.77,18.35
    0.17783,0.0,9.69,0.0,0.585,5.569,73.5,2.3999,6.0,391.0,19.2,395.77,15.1
    0.1029,30.0,4.93,0.0,0.428,6.358,52.9,7.0355,6.0,300.0,16.6,372.75,11.22
    2.33099,0.0,19.58,0.0,0.871,5.186,93.8,1.5296,5.0,403.0,14.7,356.99,28.32
    0.3692,0.0,9.9,0.0,0.544,6.567,87.3,3.6023,4.0,304.0,18.4,395.69,9.28
    0.02187,60.0,2.93,0.0,0.401,6.8,9.9,6.2196,1.0,265.0,15.6,393.37,5.03
    2.3139,0.0,19.58,0.0,0.605,5.88,97.3,2.3887,5.0,403.0,14.7,348.13,12.03
    0.28392,0.0,7.38,0.0,0.493,5.708,74.3,4.7211,5.0,287.0,19.6,391.13,11.74
    0.11132,0.0,27.74,0.0,0.609,5.983,83.5,2.1099,4.0,711.0,20.1,396.9,13.35
    0.40202,0.0,9.9,0.0,0.544,6.382,67.2,3.5325,4.0,304.0,18.4,395.21,10.36
    1.12658,0.0,19.58,1.0,0.871,5.012,88.0,1.6102,5.0,403.0,14.7,343.28,12.12
    0.20608,22.0,5.86,0.0,0.431,5.593,76.5,7.9549,7.0,330.0,19.1,372.49,12.5
    0.09068,45.0,3.44,0.0,0.437,6.951,21.5,6.4798,5.0,398.0,15.2,377.68,5.1
    0.01301,35.0,1.52,0.0,0.442,7.241,49.3,7.0379,1.0,284.0,15.5,394.74,5.49
    0.08387,0.0,12.83,0.0,0.437,5.874,36.6,4.5026,5.0,398.0,18.7,396.06,9.1
    0.17004,12.5,7.87,0.0,0.524,6.004,85.9,6.5921,5.0,311.0,15.2,386.71,17.1
    0.03738,0.0,5.19,0.0,0.515,6.31,38.5,6.4584,5.0,224.0,20.2,389.4,6.75
    0.19186,0.0,7.38,0.0,0.493,6.431,14.7,5.4159,5.0,287.0,19.6,393.68,5.08
    0.52693,0.0,6.2,0.0,0.504,8.725,83.0,2.8944,8.0,307.0,17.4,382.0,4.63
    0.07165,0.0,25.65,0.0,0.581,6.004,84.1,2.1974,2.0,188.0,19.1,377.67,14.27
    0.21038,20.0,3.33,0.0,0.4429,6.812,32.2,4.1007,5.0,216.0,14.9,396.9,4.85
    1.19294,0.0,21.89,0.0,0.624,6.326,97.7,2.271,4.0,437.0,21.2,396.9,12.26
    7.36711,0.0,18.1,0.0,0.679,6.193,78.1,1.9356,24.0,666.0,20.2,96.73,21.52
    0.19539,0.0,10.81,0.0,0.413,6.245,6.2,5.2873,4.0,305.0,19.2,377.17,7.54
    0.08199,0.0,13.92,0.0,0.437,6.009,42.3,5.5027,4.0,289.0,16.0,396.9,10.4
    0.23912,0.0,9.69,0.0,0.585,6.019,65.3,2.4091,6.0,391.0,19.2,396.9,12.92
    0.22969,0.0,10.59,0.0,0.489,6.326,52.5,4.3549,4.0,277.0,18.6,394.87,10.97
    0.11747,12.5,7.87,0.0,0.524,6.009,82.9,6.2267,5.0,311.0,15.2,396.9,13.27
    0.38735,0.0,25.65,0.0,0.581,5.613,95.6,1.7572,2.0,188.0,19.1,359.29,27.26
    3.32105,0.0,19.58,1.0,0.871,5.403,100.0,1.3216,5.0,403.0,14.7,396.9,26.82
    0.18836,0.0,6.91,0.0,0.448,5.786,33.3,5.1004,3.0,233.0,17.9,396.9,14.15
    2.01019,0.0,19.58,0.0,0.605,7.929,96.2,2.0459,5.0,403.0,14.7,369.3,3.7
    0.13642,0.0,10.59,0.0,0.489,5.891,22.3,3.9454,4.0,277.0,18.6,396.9,10.87
    0.25356,0.0,9.9,0.0,0.544,5.705,77.7,3.945,4.0,304.0,18.4,396.42,11.5
    0.10659,80.0,1.91,0.0,0.413,5.936,19.5,10.5857,4.0,334.0,22.0,376.04,5.57
    0.59005,0.0,21.89,0.0,0.624,6.372,97.9,2.3274,4.0,437.0,21.2,385.76,11.12
    3.77498,0.0,18.1,0.0,0.655,5.952,84.7,2.8715,24.0,666.0,20.2,22.01,17.15
    0.03615,80.0,4.95,0.0,0.411,6.63,23.4,5.1167,4.0,245.0,19.2,396.9,4.7
    0.0187,85.0,4.15,0.0,0.429,6.516,27.7,8.5353,4.0,351.0,17.9,392.43,6.36
    0.85204,0.0,8.14,0.0,0.538,5.965,89.2,4.0123,4.0,307.0,21.0,392.53,13.83
    0.01965,80.0,1.76,0.0,0.385,6.23,31.5,9.0892,1.0,241.0,18.2,341.6,12.93
    15.288,0.0,18.1,0.0,0.671,6.649,93.3,1.3449,24.0,666.0,20.2,363.02,23.24
    0.1712,0.0,8.56,0.0,0.52,5.836,91.9,2.211,5.0,384.0,20.9,395.67,18.66
    0.09164,0.0,10.81,0.0,0.413,6.065,7.8,5.2873,4.0,305.0,19.2,390.91,5.52
    0.80271,0.0,8.14,0.0,0.538,5.456,36.6,3.7965,4.0,307.0,21.0,288.99,11.69
    0.98843,0.0,8.14,0.0,0.538,5.813,100.0,4.0952,4.0,307.0,21.0,394.54,19.88
    0.24103,0.0,7.38,0.0,0.493,6.083,43.7,5.4159,5.0,287.0,19.6,396.9,12.79
    0.15038,0.0,25.65,0.0,0.581,5.856,97.0,1.9444,2.0,188.0,19.1,370.31,25.41
    0.15098,0.0,10.01,0.0,0.547,6.021,82.6,2.7474,6.0,432.0,17.8,394.51,10.3
    0.08829,12.5,7.87,0.0,0.524,6.012,66.6,5.5605,5.0,311.0,15.2,395.6,12.43
    4.54192,0.0,18.1,0.0,0.77,6.398,88.0,2.5182,24.0,666.0,20.2,374.56,7.79
    0.16439,22.0,5.86,0.0,0.431,6.433,49.1,7.8265,7.0,330.0,19.1,374.71,9.52
    0.44178,0.0,6.2,0.0,0.504,6.552,21.4,3.3751,8.0,307.0,17.4,380.34,3.76
    0.41238,0.0,6.2,0.0,0.504,7.163,79.9,3.2157,8.0,307.0,17.4,372.08,6.36
    0.09252,30.0,4.93,0.0,0.428,6.606,42.2,6.1899,6.0,300.0,16.6,383.78,7.37
    0.37578,0.0,10.59,1.0,0.489,5.404,88.6,3.665,4.0,277.0,18.6,395.24,23.98
    5.58107,0.0,18.1,0.0,0.713,6.436,87.9,2.3158,24.0,666.0,20.2,100.19,16.22



```python
# This time we use the sagemaker runtime client rather than the sagemaker client so that we can invoke
# the endpoint that we created.
response = session.sagemaker_runtime_client.invoke_endpoint(
                                                EndpointName = endpoint_name,
                                                ContentType = 'text/csv',
                                                Body = payload)

# We need to make sure that we deserialize the result of our endpoint call.
result = response['Body'].read().decode("utf-8")
Y_pred = np.fromstring(result, sep=',')
```

The prediction come back as another CSV text file.


```python
print("Number of predictions came back from the endpoint: {}".format(len(Y_pred)))
print(Y_pred)
```

    Number of predictions came back from the endpoint: 167
    [23.44580841 15.9037199  23.0507679  22.70243645 15.29453468 35.65088272
     21.62508583 11.61975002 33.77757645 12.46692562 12.54929447 20.74977493
     10.24889374 17.4729538  19.68718338 22.49187088  9.01094723 16.98608589
     23.87255859 25.11213875 44.76942444 34.42516327 20.66085243 22.92819595
     10.37667179 13.74817562 17.66821289 19.20152283 14.03012943 20.97484589
     18.64076614 36.27371979 35.76951599 45.02554703 21.79822922 18.58917046
     23.31511879 40.5959053  21.0345192  26.11021614 24.9164238  17.48714066
     22.31323624 28.35343361 22.78068733 12.82662773 38.14480209 20.18675613
     21.11404037 11.99268913 26.2344265  28.86267471 17.87236404  8.63697815
     29.63898277 12.35092735 38.42372513 45.45204926 25.94909096 17.45452118
      9.4408474  14.98510265 15.6116066  18.75061226 24.86480141 24.57966805
     21.92589188 23.95388412 22.14761734  9.06180382 20.95449257 24.12928391
     20.39370346 38.28743744  7.6204915   7.17795324 33.88202286 22.26333809
     10.08686447 35.25667191 21.63494873 37.3204155  21.22264099 19.08924484
     21.87445641  8.46276665 26.22470856 28.99702072 19.46490097 15.15663338
     22.61221313 19.47163391 11.7375679  19.3878994  46.87394333 11.42486572
     15.82818508 20.09231949 16.37512207  9.78350925 22.40343666 10.46687794
     17.15299416 20.03124809 32.24276733 17.61249161 19.87075615 11.40579605
     16.93699265 18.50735855 16.13512039 20.34017563 20.69373894 15.4875145
     21.74298096 26.74504089 24.10225487 20.03097153 17.69811058 22.3589859
     21.12917709 19.54560661 27.20138359 32.5811882  19.97025681 19.04721069
     22.5310936  24.47225189 40.71044922 21.40995789 36.92838669 19.28965378
     10.80000401 24.07495308 22.03178978 22.50663948 23.34122849 21.44856644
     15.61993408 14.99038219 21.39032173 51.43477631 22.08493423 20.43500137
     20.102005   19.40231895 14.76822853 28.71664619 22.2153244  18.30685425
     18.46746826 11.44921207 16.2829895  22.57260704 20.22773552 14.90745926
     21.51114655 17.41122055 19.40797615 21.79409218 24.33533859 20.68950653
     33.91294098 35.57676697 25.27508354 16.19690704 12.98216915]


To see how well our model works we can create a simple scatter plot between the predicted and actual values. If the model was completely accurate the resulting scatter plot would look like the line $x=y$. As we can see, our model seems to have done okay but there is room for improvement.


```python
plt.scatter(Y_test, Y_pred)
plt.xlabel("Median Price")
plt.ylabel("Predicted Price")
plt.title("Median Price vs Predicted Price")
```

    Text(0.5, 1.0, 'Median Price vs Predicted Price')

![png](boston_housing_xgboost_deploy_low_level_api_performance.png)

## Delete the endpoint

Since we are no longer using the deployed model we need to make sure to shut it down. Remember that you have to pay for the length of time that your endpoint is deployed so the longer it is left running, the more it costs.


```python
session.sagemaker_client.delete_endpoint(EndpointName = endpoint_name)
```

    {'ResponseMetadata': {'RequestId': '127d55af-d5d3-4b5a-9c30-1e12a97b167a',
      'HTTPStatusCode': 200,
      'HTTPHeaders': {'x-amzn-requestid': '127d55af-d5d3-4b5a-9c30-1e12a97b167a',
       'content-type': 'application/x-amz-json-1.1',
       'content-length': '0',
       'date': 'Thu, 19 Mar 2020 19:23:01 GMT'},
      'RetryAttempts': 0}}

## Optional: Clean up

The default notebook instance on SageMaker doesn't have a lot of excess disk space available. As you continue to complete and execute notebooks you will eventually fill up this disk space, leading to errors which can be difficult to diagnose. Once you are completely finished using a notebook it is a good idea to remove the files that you created along the way. Of course, you can do this from the terminal or from the notebook hub if you would like. The cell below contains some commands to clean up the created files from within the notebook.

```python
# First we will remove all of the files contained in the data_dir directory
!rm $data_dir/*

# And then we delete the directory itself
!rmdir $data_dir
```
