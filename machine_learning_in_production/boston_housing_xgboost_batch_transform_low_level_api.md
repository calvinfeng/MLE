# Predicting Boston Housing Prices

## Using XGBoost in SageMaker (Batch Transform)

As an introduction to using SageMaker's Low Level Python API we will look at a relatively simple
problem. Namely, we will use the [Boston Housing Dataset](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html)
to predict the median value of a home in the area of Boston Mass.

The documentation reference for the API used in this notebook is the [SageMaker Developer's Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/)

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

In this notebook we will only be covering steps 1 through 5 as we just want to get a feel for using
SageMaker. In later notebooks we will talk about deploying a trained model in much more detail.

## Step 0: Setting up the notebook

We begin by setting up all of the necessary bits required to run our notebook. To start that means
loading all of the Python modules we will need.

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

In addition to the modules above, we need to import the various bits of SageMaker that we will be
using.

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

Fortunately, this dataset can be retrieved using sklearn and so this step is relatively
straightforward.

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

## Step 3: Uploading the data files to S3

When a training job is constructed using SageMaker, a container is executed which performs the
training operation. This container is given access to data that is stored in S3. This means that we
need to upload the data we want to use for training to S3. In addition, when we perform a batch
transform job, SageMaker expects the input data to be stored on S3. We can use the SageMaker API to
do this and hide some of the details.

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

Since we are currently running inside of a SageMaker session, we can use the object which represents
this session to upload our data to the 'default' S3 bucket. Note that it is good practice to provide
a custom prefix (essentially an S3 folder) to make sure that you don't accidentally interfere with
data uploaded from some other notebook or project.

```python
prefix = 'boston-xgboost-LL'

test_location = session.upload_data(os.path.join(data_dir, 'test.csv'), key_prefix=prefix)
val_location = session.upload_data(os.path.join(data_dir, 'validation.csv'), key_prefix=prefix)
train_location = session.upload_data(os.path.join(data_dir, 'train.csv'), key_prefix=prefix)
```

## Step 4: Train and construct the XGBoost model

Now that we have the training and validation data uploaded to S3, we can construct a training job
for our XGBoost model and build the model itself.

### Set up the training job

First, we will set up and execute a training job for our model. To do this we need to specify some
information that SageMaker will use to set up and properly execute the computation. For additional
documentation on constructing a training job, see the [CreateTrainingJob API](https://docs.aws.amazon.com/sagemaker/latest/dg/API_CreateTrainingJob.html)
reference.

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

# We also need to say where we would like the resulting model artifacts stored.
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

Now that we've built the dictionary object containing the training job parameters, we can ask
SageMaker to execute the job.

```python
# First we need to choose a training job name. This is useful for if we want to recall information about our
# training job at a later date. Note that SageMaker requires a training job name and that the name needs to
# be unique, which we accomplish by appending the current timestamp.
training_job_name = "boston-xgboost-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
training_params['TrainingJobName'] = training_job_name

# And now we ask SageMaker to create (and execute) the training job
training_job = session.sagemaker_client.create_training_job(**training_params)
```

The training job has now been created by SageMaker and is currently running. Since we need the
output of the training job, we may wish to wait until it has finished. We can do so by asking
SageMaker to output the logs generated by the training job and continue doing so until the training
job terminates.

```python
session.logs_for_job(training_job_name, wait=True)
```

```text
2020-03-18 21:28:57 Starting - Launching requested ML instances......
2020-03-18 21:29:56 Starting - Preparing the instances for training......
2020-03-18 21:30:55 Downloading - Downloading input data...
2020-03-18 21:31:11 Training - Downloading the training image..[34mINFO:sagemaker-containers:Imported framework sagemaker_xgboost_container.training[0m
[34mINFO:sagemaker-containers:Failed to parse hyperparameter objective value reg:linear to Json.[0m
[34mReturning the value itself[0m
[34mINFO:sagemaker-containers:No GPUs detected (normal if no gpus installed)[0m
[34mINFO:sagemaker_xgboost_container.training:Running XGBoost Sagemaker in algorithm mode[0m
[34mINFO:root:Determined delimiter of CSV input is ','[0m
[34mINFO:root:Determined delimiter of CSV input is ','[0m
[34mINFO:root:Determined delimiter of CSV input is ','[0m
[34m[21:31:45] 227x13 matrix with 2951 entries loaded from /opt/ml/input/data/train?format=csv&label_column=0&delimiter=,[0m
[34mINFO:root:Determined delimiter of CSV input is ','[0m
[34m[21:31:45] 112x13 matrix with 1456 entries loaded from /opt/ml/input/data/validation?format=csv&label_column=0&delimiter=,[0m
[34mINFO:root:Single node training.[0m
[34mINFO:root:Train matrix has 227 rows[0m
[34mINFO:root:Validation matrix has 112 rows[0m
[34m[21:31:45] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.[0m
[34m[0]#011train-rmse:19.6015#011validation-rmse:19.9713[0m
[34m[1]#011train-rmse:16.0012#011validation-rmse:16.4917[0m
[34m[2]#011train-rmse:13.1471#011validation-rmse:13.7391[0m
[34m[3]#011train-rmse:10.8318#011validation-rmse:11.546[0m
[34m[4]#011train-rmse:8.99428#011validation-rmse:9.90845[0m
[34m[5]#011train-rmse:7.47775#011validation-rmse:8.61753[0m
[34m[6]#011train-rmse:6.26087#011validation-rmse:7.54242[0m
[34m[7]#011train-rmse:5.30636#011validation-rmse:6.76342[0m
[34m[8]#011train-rmse:4.58561#011validation-rmse:6.31725[0m
[34m[9]#011train-rmse:4.00222#011validation-rmse:5.93388[0m
[34m[10]#011train-rmse:3.59368#011validation-rmse:5.73373[0m
[34m[11]#011train-rmse:3.23297#011validation-rmse:5.53352[0m
[34m[12]#011train-rmse:2.96545#011validation-rmse:5.41433[0m
[34m[13]#011train-rmse:2.76411#011validation-rmse:5.32919[0m
[34m[14]#011train-rmse:2.60116#011validation-rmse:5.21487[0m
[34m[15]#011train-rmse:2.4151#011validation-rmse:5.08712[0m
[34m[16]#011train-rmse:2.29829#011validation-rmse:5.07836[0m
[34m[17]#011train-rmse:2.208#011validation-rmse:5.00698[0m
[34m[18]#011train-rmse:2.13688#011validation-rmse:4.97999[0m
[34m[19]#011train-rmse:2.06558#011validation-rmse:4.97224[0m
[34m[20]#011train-rmse:1.97701#011validation-rmse:4.93851[0m
[34m[21]#011train-rmse:1.89463#011validation-rmse:4.8844[0m
[34m[22]#011train-rmse:1.82083#011validation-rmse:4.84604[0m
[34m[23]#011train-rmse:1.78842#011validation-rmse:4.79984[0m
[34m[24]#011train-rmse:1.75609#011validation-rmse:4.74221[0m
[34m[25]#011train-rmse:1.72153#011validation-rmse:4.71392[0m
[34m[26]#011train-rmse:1.69901#011validation-rmse:4.72[0m
[34m[27]#011train-rmse:1.67513#011validation-rmse:4.70148[0m
[34m[28]#011train-rmse:1.65448#011validation-rmse:4.71672[0m
[34m[29]#011train-rmse:1.62258#011validation-rmse:4.67159[0m
[34m[30]#011train-rmse:1.6057#011validation-rmse:4.66021[0m
[34m[31]#011train-rmse:1.59552#011validation-rmse:4.67154[0m
[34m[32]#011train-rmse:1.55541#011validation-rmse:4.61141[0m
[34m[33]#011train-rmse:1.53717#011validation-rmse:4.61497[0m
[34m[34]#011train-rmse:1.53708#011validation-rmse:4.6202[0m
[34m[35]#011train-rmse:1.49778#011validation-rmse:4.60782[0m
[34m[36]#011train-rmse:1.50075#011validation-rmse:4.62538[0m
[34m[37]#011train-rmse:1.49053#011validation-rmse:4.60922[0m
[34m[38]#011train-rmse:1.48548#011validation-rmse:4.62445[0m
[34m[39]#011train-rmse:1.47539#011validation-rmse:4.62821[0m
[34m[40]#011train-rmse:1.46659#011validation-rmse:4.62153[0m
[34m[41]#011train-rmse:1.45239#011validation-rmse:4.61344[0m
[34m[42]#011train-rmse:1.43587#011validation-rmse:4.57347[0m
[34m[43]#011train-rmse:1.40401#011validation-rmse:4.56752[0m
[34m[44]#011train-rmse:1.38189#011validation-rmse:4.50632[0m
[34m[45]#011train-rmse:1.37594#011validation-rmse:4.51273[0m
[34m[46]#011train-rmse:1.33977#011validation-rmse:4.48602[0m
[34m[47]#011train-rmse:1.34474#011validation-rmse:4.49001[0m
[34m[48]#011train-rmse:1.32822#011validation-rmse:4.48001[0m
[34m[49]#011train-rmse:1.31442#011validation-rmse:4.47066[0m
[34m[50]#011train-rmse:1.30356#011validation-rmse:4.45961[0m
[34m[51]#011train-rmse:1.30427#011validation-rmse:4.46614[0m
[34m[52]#011train-rmse:1.29272#011validation-rmse:4.45275[0m
[34m[53]#011train-rmse:1.26497#011validation-rmse:4.41165[0m
[34m[54]#011train-rmse:1.25659#011validation-rmse:4.40476[0m
[34m[55]#011train-rmse:1.2355#011validation-rmse:4.38478[0m
[34m[56]#011train-rmse:1.21798#011validation-rmse:4.3734[0m
[34m[57]#011train-rmse:1.2046#011validation-rmse:4.3601[0m
[34m[58]#011train-rmse:1.19654#011validation-rmse:4.3496[0m
[34m[59]#011train-rmse:1.1719#011validation-rmse:4.34984[0m
[34m[60]#011train-rmse:1.17144#011validation-rmse:4.3437[0m
[34m[61]#011train-rmse:1.15543#011validation-rmse:4.33572[0m
[34m[62]#011train-rmse:1.15887#011validation-rmse:4.34297[0m
[34m[63]#011train-rmse:1.1552#011validation-rmse:4.33484[0m
[34m[64]#011train-rmse:1.15334#011validation-rmse:4.32953[0m
[34m[65]#011train-rmse:1.14062#011validation-rmse:4.3472[0m
[34m[66]#011train-rmse:1.12507#011validation-rmse:4.34943[0m
[34m[67]#011train-rmse:1.09996#011validation-rmse:4.35931[0m
[34m[68]#011train-rmse:1.09516#011validation-rmse:4.35271[0m
[34m[69]#011train-rmse:1.07561#011validation-rmse:4.32277[0m
[34m[70]#011train-rmse:1.06457#011validation-rmse:4.31648[0m
[34m[71]#011train-rmse:1.0545#011validation-rmse:4.29089[0m
[34m[72]#011train-rmse:1.05028#011validation-rmse:4.28361[0m
[34m[73]#011train-rmse:1.04454#011validation-rmse:4.27559[0m
[34m[74]#011train-rmse:1.03067#011validation-rmse:4.28646[0m
[34m[75]#011train-rmse:1.03158#011validation-rmse:4.29429[0m
[34m[76]#011train-rmse:1.02908#011validation-rmse:4.2898[0m
[34m[77]#011train-rmse:1.0276#011validation-rmse:4.28505[0m
[34m[78]#011train-rmse:1.02806#011validation-rmse:4.283[0m
[34m[79]#011train-rmse:1.01053#011validation-rmse:4.26771[0m
[34m[80]#011train-rmse:1.01054#011validation-rmse:4.26765[0m
[34m[81]#011train-rmse:0.99876#011validation-rmse:4.26786[0m
[34m[82]#011train-rmse:0.997369#011validation-rmse:4.2678[0m
[34m[83]#011train-rmse:0.992551#011validation-rmse:4.25861[0m
[34m[84]#011train-rmse:0.979124#011validation-rmse:4.24851[0m
[34m[85]#011train-rmse:0.979006#011validation-rmse:4.24893[0m
[34m[86]#011train-rmse:0.969096#011validation-rmse:4.27203[0m
[34m[87]#011train-rmse:0.948978#011validation-rmse:4.24257[0m
[34m[88]#011train-rmse:0.946447#011validation-rmse:4.24355[0m
[34m[89]#011train-rmse:0.939564#011validation-rmse:4.25643[0m
[34m[90]#011train-rmse:0.932154#011validation-rmse:4.24137[0m
[34m[91]#011train-rmse:0.923471#011validation-rmse:4.22447[0m
[34m[92]#011train-rmse:0.92377#011validation-rmse:4.22339[0m
[34m[93]#011train-rmse:0.923855#011validation-rmse:4.22577[0m
[34m[94]#011train-rmse:0.923457#011validation-rmse:4.2268[0m
[34m[95]#011train-rmse:0.91434#011validation-rmse:4.23928[0m
[34m[96]#011train-rmse:0.91047#011validation-rmse:4.24151[0m
[34m[97]#011train-rmse:0.902204#011validation-rmse:4.2467[0m
[34m[98]#011train-rmse:0.889396#011validation-rmse:4.25177[0m
[34m[99]#011train-rmse:0.889457#011validation-rmse:4.25294[0m
[34m[100]#011train-rmse:0.888703#011validation-rmse:4.25292[0m
[34m[101]#011train-rmse:0.873455#011validation-rmse:4.24482[0m
[34m[102]#011train-rmse:0.861644#011validation-rmse:4.22572[0m

2020-03-18 21:31:56 Uploading - Uploading generated training model
2020-03-18 21:31:56 Completed - Training job completed
Training seconds: 61
Billable seconds: 61
```

### Build the model

Now that the training job has completed, we have some model artifacts which we can use to build a
model. Note that here we mean SageMaker's definition of a model, which is a collection of
information about a specific algorithm along with the artifacts which result from a training job.

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

## Step 5: Testing the model

Now that we have fit our model to the training data, using the validation data to avoid overfitting,
we can test our model. To do this we will make use of SageMaker's Batch Transform functionality. In
other words, we need to set up and execute a batch transform job, similar to the way that we
constructed the training job earlier.

### Set up the batch transform job

Just like when we were training our model, we first need to provide some information in the form of
a data structure that describes the batch transform job which we wish to execute.

We will only be using some of the options available here but to see some of the additional options
please see the SageMaker documentation for [creating a batch transform job](https://docs.aws.amazon.com/sagemaker/latest/dg/API_CreateTransformJob.html).

```python
# Just like in each of the previous steps, we need to make sure to name our job and the name should be unique.
transform_job_name = 'boston-xgboost-batch-transform-' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

# Now we construct the data structure which will describe the batch transform job.
transform_request = \
{
    "TransformJobName": transform_job_name,

    # This is the name of the model that we created earlier`.
    "ModelName": model_name,

    # This describes how many compute instances should be used at once. If you happen to be doing a very large
    # batch transform job it may be worth running multiple compute instances at once.
    "MaxConcurrentTransforms": 1,

    # This says how big each individual request sent to the model should be, at most. One of the things that
    # SageMaker does in the background is to split our data up into chunks so that each chunks stays under
    # this size limit.
    "MaxPayloadInMB": 6,

    # Sometimes we may want to send only a single sample to our endpoint at a time, however in this case each of
    # the chunks that we send should contain multiple samples of our input data.
    "BatchStrategy": "MultiRecord",

    # This next object describes where the output data should be stored. Some of the more advanced options which
    # we don't cover here also describe how SageMaker should collect output from various batches.
    "TransformOutput": {
        "S3OutputPath": "s3://{}/{}/batch-transform/".format(session.default_bucket(),prefix)
    },

    # Here we describe our input data. Of course, we need to tell SageMaker where on S3 our input data is stored, in
    # addition we need to detail the characteristics of our input data. In particular, since SageMaker may need to
    # split our data up into chunks, it needs to know how the individual samples in our data file appear. In our
    # case each line is its own sample and so we set the split type to 'line'. We also need to tell SageMaker what
    # type of data is being sent, in this case csv, so that it can properly serialize the data.
    "TransformInput": {
        "ContentType": "text/csv",
        "SplitType": "Line",
        "DataSource": {
            "S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": test_location,
            }
        }
    },

    # And lastly we tell SageMaker what sort of compute instance we would like it to use.
    "TransformResources": {
            "InstanceType": "ml.m4.xlarge",
            "InstanceCount": 1
    }
}
```

### Execute the batch transform job

Now that we have created the request data structure, it is time to ask SageMaker to set up and run
our batch transform job. Just like in the previous steps, SageMaker performs these tasks in the
background so that if we want to wait for the transform job to terminate (and ensure the job is
progressing) we can ask SageMaker to wait of the transform job to complete.

```python
transform_response = session.sagemaker_client.create_transform_job(**transform_request)
```

```python
transform_desc = session.wait_for_transform_job(transform_job_name)
```

```text
.............................................!
```

### Analyze the results

Now that the transform job has completed, the results are stored on S3 as we requested. Since we'd
like to do a bit of analysis in the notebook we can use some notebook magic to copy the resulting
output from S3 and save it locally.

```python
transform_output = "s3://{}/{}/batch-transform/".format(session.default_bucket(),prefix)
```

```python
!aws s3 cp --recursive $transform_output $data_dir
```

```text
download: s3://sagemaker-us-west-2-171758673694/boston-xgboost-LL/batch-transform/test.csv.out to ../data/boston/test.csv.out
```

To see how well our model works we can create a simple scatter plot between the predicted and actual
values. If the model was completely accurate the resulting scatter plot would look like the line
$x=y$. As we can see, our model seems to have done okay but there is room for improvement.

```python
Y_pred = pd.read_csv(os.path.join(data_dir, 'test.csv.out'), header=None)
```

```python
plt.scatter(Y_test, Y_pred)
plt.xlabel("Median Price")
plt.ylabel("Predicted Price")
plt.title("Median Price vs Predicted Price")
```

```text
Text(0.5, 1.0, 'Median Price vs Predicted Price')
```

![png](boston_housing_xgboost_batch_transform_low_level_api_performance.png)

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
