# Predicting Boston Housing Prices

## Using XGBoost in SageMaker (Hyperparameter Tuning)

_Deep Learning Nanodegree Program | Deployment_

---

As an introduction to using SageMaker's High Level Python API for hyperparameter tuning, we will look again at the [Boston Housing Dataset](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html) to predict the median value of a home in the area of Boston Mass.

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

In this notebook we will only be covering steps 1 through 5 as we are only interested in creating a tuned model and testing its performance.

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

## Step 3: Uploading the data files to S3

When a training job is constructed using SageMaker, a container is executed which performs the training operation. This container is given access to data that is stored in S3. This means that we need to upload the data we want to use for training to S3. In addition, when we perform a batch transform job, SageMaker expects the input data to be stored on S3. We can use the SageMaker API to do this and hide some of the details.

### Save the data locally

First we need to create the test, train and validation csv files which we will then upload to S3.


```python
# This is our local data directory. We need to make sure that it exists.
data_dir = '../data/boston_housing_xgboost_hyperparameter_tuning_high_level'
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
prefix = 'xgboost-hyperparamter-tuning-high-level'
test_location = session.upload_data(os.path.join(data_dir, 'test.csv'), key_prefix=prefix)
val_location = session.upload_data(os.path.join(data_dir, 'validation.csv'), key_prefix=prefix)
train_location = session.upload_data(os.path.join(data_dir, 'train.csv'), key_prefix=prefix)
```

## Step 4: Train the XGBoost model

Now that we have the training and validation data uploaded to S3, we can construct our XGBoost model and train it. Unlike in the previous notebooks, instead of training a single model, we will use SageMaker's hyperparameter tuning functionality to train multiple models and use the one that performs the best on the validation set.

To begin with, as in the previous approaches, we will need to construct an estimator object.


```python
# As stated above, we use this utility method to construct the image name for the training container.
container = get_image_uri(session.boto_region_name, 'xgboost', '0.90-1')

# Now that we know which container to use, we can construct the estimator object.
xgb = sagemaker.estimator.Estimator(container, # The name of the training container
                                    role,      # The IAM role to use (our current role in this case)
                                    train_instance_count=1, # The number of instances to use for training
                                    train_instance_type='ml.m4.xlarge', # The type of instance ot use for training
                                    output_path='s3://{}/{}/'.format(session.default_bucket(), prefix), # Where to save the output (the model artifacts)
                                    base_job_name='xgboost-training-job',
                                    sagemaker_session=session) # The current SageMaker session
```

Before beginning the hyperparameter tuning, we should make sure to set any model specific hyperparameters that we wish to have default values. There are quite a few that can be set when using the XGBoost algorithm, below are just a few of them. If you would like to change the hyperparameters below or modify additional ones you can find additional information on the [XGBoost hyperparameter page](https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost_hyperparameters.html)


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

Now that we have our estimator object completely set up, it is time to create the hyperparameter tuner. To do this we need to construct a new object which contains each of the parameters we want SageMaker to tune. In this case, we wish to find the best values for the `max_depth`, `eta`, `min_child_weight`, `subsample`, and `gamma` parameters. Note that for each parameter that we want SageMaker to tune we need to specify both the *type* of the parameter and the *range* of values that parameter may take on.

In addition, we specify the *number* of models to construct (`max_jobs`) and the number of those that can be trained in parallel (`max_parallel_jobs`). In the cell below we have chosen to train `20` models, of which we ask that SageMaker train `3` at a time in parallel. Note that this results in a total of `20` training jobs being executed which can take some time, in this case almost a half hour. With more complicated models this can take even longer so be aware!


```python
from sagemaker.tuner import IntegerParameter, ContinuousParameter, HyperparameterTuner

xgb_hyperparameter_tuner = HyperparameterTuner(estimator=xgb, # The estimator object to use as the basis for the training jobs.
                                               objective_metric_name='validation:rmse', # The metric used to compare trained models.
                                               objective_type='Minimize', # Whether we wish to minimize or maximize the metric.
                                               max_jobs=20, # The total number of models to train
                                               max_parallel_jobs=3, # The number of models to train in parallel
                                               hyperparameter_ranges={
                                                    'max_depth': IntegerParameter(3, 12),
                                                    'eta'      : ContinuousParameter(0.05, 0.5),
                                                    'min_child_weight': IntegerParameter(2, 8),
                                                    'subsample': ContinuousParameter(0.5, 0.9),
                                                    'gamma': ContinuousParameter(0, 10),
                                               })
```

Now that we have our hyperparameter tuner object completely set up, it is time to train it. To do this we make sure that SageMaker knows our input data is in csv format and then execute the `fit` method.


```python
# This is a wrapper around the location of our train and validation data, to make sure that SageMaker
# knows our data is in csv format.
s3_input_train = sagemaker.s3_input(s3_data=train_location, content_type='csv')
s3_input_validation = sagemaker.s3_input(s3_data=val_location, content_type='csv')
xgb_hyperparameter_tuner.fit({
    'train': s3_input_train,
    'validation': s3_input_validation
})
```

As in many of the examples we have seen so far, the `fit()` method takes care of setting up and fitting a number of different models, each with different hyperparameters. If we wish to wait for this process to finish, we can call the `wait()` method.


```python
xgb_hyperparameter_tuner.wait()
```

    ...........................................................................................................................................................................................................................................................................................!


Once the hyperamater tuner has finished, we can retrieve information about the best performing model. 


```python
xgb_hyperparameter_tuner.best_training_job()
```




    'sagemaker-xgboost-200325-0443-010-632e4608'



In addition, since we'd like to set up a batch transform job to test the best model, we can construct a new estimator object from the results of the best training job. The `xgb_attached` object below can now be used as though we constructed an estimator with the best performing hyperparameters and then fit it to our training data.


```python
xgb_attached = sagemaker.estimator.Estimator.attach(xgb_hyperparameter_tuner.best_training_job())
```

    2020-03-25 04:56:36 Starting - Preparing the instances for training
    2020-03-25 04:56:36 Downloading - Downloading input data
    2020-03-25 04:56:36 Training - Training image download completed. Training in progress.
    2020-03-25 04:56:36 Uploading - Uploading generated training model
    2020-03-25 04:56:36 Completed - Training job completed[34mINFO:sagemaker-containers:Imported framework sagemaker_xgboost_container.training[0m
    [34mINFO:sagemaker-containers:Failed to parse hyperparameter _tuning_objective_metric value validation:rmse to Json.[0m
    [34mReturning the value itself[0m
    [34mINFO:sagemaker-containers:Failed to parse hyperparameter objective value reg:linear to Json.[0m
    [34mReturning the value itself[0m
    [34mINFO:sagemaker-containers:No GPUs detected (normal if no gpus installed)[0m
    [34mINFO:sagemaker_xgboost_container.training:Running XGBoost Sagemaker in algorithm mode[0m
    [34mINFO:root:Determined delimiter of CSV input is ','[0m
    [34mINFO:root:Determined delimiter of CSV input is ','[0m
    [34mINFO:root:Determined delimiter of CSV input is ','[0m
    [34m[04:56:26] 227x13 matrix with 2951 entries loaded from /opt/ml/input/data/train?format=csv&label_column=0&delimiter=,[0m
    [34mINFO:root:Determined delimiter of CSV input is ','[0m
    [34m[04:56:26] 112x13 matrix with 1456 entries loaded from /opt/ml/input/data/validation?format=csv&label_column=0&delimiter=,[0m
    [34mINFO:root:Single node training.[0m
    [34mINFO:root:Setting up HPO optimized metric to be : rmse[0m
    [34mINFO:root:Train matrix has 227 rows[0m
    [34mINFO:root:Validation matrix has 112 rows[0m
    [34m[04:56:26] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.[0m
    [34m[0]#011train-rmse:22.9765#011validation-rmse:24.019[0m
    [34m[1]#011train-rmse:21.9152#011validation-rmse:22.9233[0m
    [34m[2]#011train-rmse:20.8975#011validation-rmse:21.8724[0m
    [34m[3]#011train-rmse:19.9212#011validation-rmse:20.8688[0m
    [34m[4]#011train-rmse:18.9992#011validation-rmse:19.9157[0m
    [34m[5]#011train-rmse:18.1351#011validation-rmse:19.0388[0m
    [34m[6]#011train-rmse:17.3046#011validation-rmse:18.1947[0m
    [34m[7]#011train-rmse:16.5084#011validation-rmse:17.3721[0m
    [34m[8]#011train-rmse:15.7522#011validation-rmse:16.6077[0m
    [34m[9]#011train-rmse:15.041#011validation-rmse:15.8691[0m
    [34m[10]#011train-rmse:14.373#011validation-rmse:15.2044[0m
    [34m[11]#011train-rmse:13.7411#011validation-rmse:14.5561[0m
    [34m[12]#011train-rmse:13.1316#011validation-rmse:13.9678[0m
    [34m[13]#011train-rmse:12.5642#011validation-rmse:13.413[0m
    [34m[14]#011train-rmse:12.0198#011validation-rmse:12.8438[0m
    [34m[15]#011train-rmse:11.4977#011validation-rmse:12.3251[0m
    [34m[16]#011train-rmse:11.0064#011validation-rmse:11.8518[0m
    [34m[17]#011train-rmse:10.5441#011validation-rmse:11.4028[0m
    [34m[18]#011train-rmse:10.1002#011validation-rmse:10.9416[0m
    [34m[19]#011train-rmse:9.66771#011validation-rmse:10.5108[0m
    [34m[20]#011train-rmse:9.26117#011validation-rmse:10.1154[0m
    [34m[21]#011train-rmse:8.85897#011validation-rmse:9.69352[0m
    [34m[22]#011train-rmse:8.4908#011validation-rmse:9.31363[0m
    [34m[23]#011train-rmse:8.12881#011validation-rmse:8.93984[0m
    [34m[24]#011train-rmse:7.80814#011validation-rmse:8.60596[0m
    [34m[25]#011train-rmse:7.49219#011validation-rmse:8.28855[0m
    [34m[26]#011train-rmse:7.18446#011validation-rmse:7.99713[0m
    [34m[27]#011train-rmse:6.90305#011validation-rmse:7.72991[0m
    [34m[28]#011train-rmse:6.61176#011validation-rmse:7.43752[0m
    [34m[29]#011train-rmse:6.36591#011validation-rmse:7.19754[0m
    [34m[30]#011train-rmse:6.10652#011validation-rmse:6.94225[0m
    [34m[31]#011train-rmse:5.86244#011validation-rmse:6.71259[0m
    [34m[32]#011train-rmse:5.64533#011validation-rmse:6.53594[0m
    [34m[33]#011train-rmse:5.42151#011validation-rmse:6.32822[0m
    [34m[34]#011train-rmse:5.21802#011validation-rmse:6.13441[0m
    [34m[35]#011train-rmse:5.00957#011validation-rmse:5.92814[0m
    [34m[36]#011train-rmse:4.82033#011validation-rmse:5.78189[0m
    [34m[37]#011train-rmse:4.64782#011validation-rmse:5.62965[0m
    [34m[38]#011train-rmse:4.48314#011validation-rmse:5.49492[0m
    [34m[39]#011train-rmse:4.3119#011validation-rmse:5.33745[0m
    [34m[40]#011train-rmse:4.16681#011validation-rmse:5.22646[0m
    [34m[41]#011train-rmse:4.01718#011validation-rmse:5.10249[0m
    [34m[42]#011train-rmse:3.88229#011validation-rmse:4.98746[0m
    [34m[43]#011train-rmse:3.74964#011validation-rmse:4.87621[0m
    [34m[44]#011train-rmse:3.6112#011validation-rmse:4.76555[0m
    [34m[45]#011train-rmse:3.48099#011validation-rmse:4.65815[0m
    [34m[46]#011train-rmse:3.35642#011validation-rmse:4.54674[0m
    [34m[47]#011train-rmse:3.23185#011validation-rmse:4.47582[0m
    [34m[48]#011train-rmse:3.1148#011validation-rmse:4.38055[0m
    [34m[49]#011train-rmse:3.00244#011validation-rmse:4.2897[0m
    [34m[50]#011train-rmse:2.89237#011validation-rmse:4.21536[0m
    [34m[51]#011train-rmse:2.80923#011validation-rmse:4.14758[0m
    [34m[52]#011train-rmse:2.72255#011validation-rmse:4.09429[0m
    [34m[53]#011train-rmse:2.63711#011validation-rmse:4.04844[0m
    [34m[54]#011train-rmse:2.54713#011validation-rmse:4.00031[0m
    [34m[55]#011train-rmse:2.4637#011validation-rmse:3.94159[0m
    [34m[56]#011train-rmse:2.38396#011validation-rmse:3.89786[0m
    [34m[57]#011train-rmse:2.30646#011validation-rmse:3.8473[0m
    [34m[58]#011train-rmse:2.23295#011validation-rmse:3.81377[0m
    [34m[59]#011train-rmse:2.1659#011validation-rmse:3.78285[0m
    [34m[60]#011train-rmse:2.10254#011validation-rmse:3.74064[0m
    [34m[61]#011train-rmse:2.0324#011validation-rmse:3.70186[0m
    [34m[62]#011train-rmse:1.98412#011validation-rmse:3.67799[0m
    [34m[63]#011train-rmse:1.92133#011validation-rmse:3.65926[0m
    [34m[64]#011train-rmse:1.86107#011validation-rmse:3.63088[0m
    [34m[65]#011train-rmse:1.81407#011validation-rmse:3.60843[0m
    [34m[66]#011train-rmse:1.75881#011validation-rmse:3.59112[0m
    [34m[67]#011train-rmse:1.70398#011validation-rmse:3.57673[0m
    [34m[68]#011train-rmse:1.66169#011validation-rmse:3.56457[0m
    [34m[69]#011train-rmse:1.61966#011validation-rmse:3.53626[0m
    [34m[70]#011train-rmse:1.57779#011validation-rmse:3.52236[0m
    [34m[71]#011train-rmse:1.54036#011validation-rmse:3.51773[0m
    [34m[72]#011train-rmse:1.49729#011validation-rmse:3.49866[0m
    [34m[73]#011train-rmse:1.46225#011validation-rmse:3.48303[0m
    [34m[74]#011train-rmse:1.43361#011validation-rmse:3.47494[0m
    [34m[75]#011train-rmse:1.39508#011validation-rmse:3.46661[0m
    [34m[76]#011train-rmse:1.35699#011validation-rmse:3.45364[0m
    [34m[77]#011train-rmse:1.32546#011validation-rmse:3.44801[0m
    [34m[78]#011train-rmse:1.29426#011validation-rmse:3.44057[0m
    [34m[79]#011train-rmse:1.2608#011validation-rmse:3.42987[0m
    [34m[80]#011train-rmse:1.23326#011validation-rmse:3.42543[0m
    [34m[81]#011train-rmse:1.20243#011validation-rmse:3.41828[0m
    [34m[82]#011train-rmse:1.1813#011validation-rmse:3.41692[0m
    [34m[83]#011train-rmse:1.15821#011validation-rmse:3.41219[0m
    [34m[84]#011train-rmse:1.13811#011validation-rmse:3.40513[0m
    [34m[85]#011train-rmse:1.10866#011validation-rmse:3.39517[0m
    [34m[86]#011train-rmse:1.09278#011validation-rmse:3.38679[0m
    [34m[87]#011train-rmse:1.06546#011validation-rmse:3.38231[0m
    [34m[88]#011train-rmse:1.04342#011validation-rmse:3.37812[0m
    [34m[89]#011train-rmse:1.02077#011validation-rmse:3.36811[0m
    [34m[90]#011train-rmse:1.00285#011validation-rmse:3.36054[0m
    [34m[91]#011train-rmse:0.988953#011validation-rmse:3.35515[0m
    [34m[92]#011train-rmse:0.966114#011validation-rmse:3.35052[0m
    [34m[93]#011train-rmse:0.950402#011validation-rmse:3.34418[0m
    [34m[94]#011train-rmse:0.936589#011validation-rmse:3.34307[0m
    [34m[95]#011train-rmse:0.916442#011validation-rmse:3.331[0m
    [34m[96]#011train-rmse:0.903559#011validation-rmse:3.32637[0m
    [34m[97]#011train-rmse:0.888005#011validation-rmse:3.32494[0m
    [34m[98]#011train-rmse:0.87601#011validation-rmse:3.32252[0m
    [34m[99]#011train-rmse:0.867442#011validation-rmse:3.32036[0m
    [34m[100]#011train-rmse:0.852856#011validation-rmse:3.31281[0m
    [34m[101]#011train-rmse:0.837925#011validation-rmse:3.30555[0m
    [34m[102]#011train-rmse:0.830863#011validation-rmse:3.30613[0m
    [34m[103]#011train-rmse:0.817114#011validation-rmse:3.30063[0m
    [34m[104]#011train-rmse:0.807123#011validation-rmse:3.29424[0m
    [34m[105]#011train-rmse:0.797177#011validation-rmse:3.28713[0m
    [34m[106]#011train-rmse:0.784019#011validation-rmse:3.27934[0m
    [34m[107]#011train-rmse:0.774195#011validation-rmse:3.27278[0m
    [34m[108]#011train-rmse:0.768972#011validation-rmse:3.27058[0m
    [34m[109]#011train-rmse:0.763601#011validation-rmse:3.2715[0m
    [34m[110]#011train-rmse:0.760047#011validation-rmse:3.27197[0m
    [34m[111]#011train-rmse:0.754886#011validation-rmse:3.27151[0m
    [34m[112]#011train-rmse:0.747303#011validation-rmse:3.26877[0m
    [34m[113]#011train-rmse:0.73999#011validation-rmse:3.2686[0m
    [34m[114]#011train-rmse:0.733511#011validation-rmse:3.26174[0m
    [34m[115]#011train-rmse:0.723211#011validation-rmse:3.25805[0m
    [34m[116]#011train-rmse:0.719735#011validation-rmse:3.25462[0m
    [34m[117]#011train-rmse:0.715395#011validation-rmse:3.25503[0m
    [34m[118]#011train-rmse:0.709895#011validation-rmse:3.25248[0m
    [34m[119]#011train-rmse:0.701788#011validation-rmse:3.24782[0m
    [34m[120]#011train-rmse:0.693493#011validation-rmse:3.24135[0m
    [34m[121]#011train-rmse:0.688028#011validation-rmse:3.23618[0m
    [34m[122]#011train-rmse:0.68094#011validation-rmse:3.23461[0m
    [34m[123]#011train-rmse:0.674934#011validation-rmse:3.23255[0m
    [34m[124]#011train-rmse:0.666722#011validation-rmse:3.23411[0m
    [34m[125]#011train-rmse:0.660254#011validation-rmse:3.23201[0m
    [34m[126]#011train-rmse:0.651819#011validation-rmse:3.22875[0m
    [34m[127]#011train-rmse:0.644388#011validation-rmse:3.2253[0m
    [34m[128]#011train-rmse:0.64043#011validation-rmse:3.2229[0m
    [34m[129]#011train-rmse:0.63658#011validation-rmse:3.22282[0m
    [34m[130]#011train-rmse:0.63558#011validation-rmse:3.22106[0m
    [34m[131]#011train-rmse:0.633273#011validation-rmse:3.21751[0m
    [34m[132]#011train-rmse:0.630883#011validation-rmse:3.21565[0m
    [34m[133]#011train-rmse:0.628031#011validation-rmse:3.21407[0m
    [34m[134]#011train-rmse:0.622346#011validation-rmse:3.21134[0m
    [34m[135]#011train-rmse:0.618335#011validation-rmse:3.20937[0m
    [34m[136]#011train-rmse:0.617582#011validation-rmse:3.21175[0m
    [34m[137]#011train-rmse:0.614386#011validation-rmse:3.20876[0m
    [34m[138]#011train-rmse:0.608484#011validation-rmse:3.21155[0m
    [34m[139]#011train-rmse:0.603125#011validation-rmse:3.20869[0m
    [34m[140]#011train-rmse:0.599672#011validation-rmse:3.20681[0m
    [34m[141]#011train-rmse:0.594731#011validation-rmse:3.20659[0m
    [34m[142]#011train-rmse:0.588498#011validation-rmse:3.20151[0m
    [34m[143]#011train-rmse:0.584756#011validation-rmse:3.20371[0m
    [34m[144]#011train-rmse:0.582489#011validation-rmse:3.20575[0m
    [34m[145]#011train-rmse:0.576034#011validation-rmse:3.20275[0m
    [34m[146]#011train-rmse:0.570841#011validation-rmse:3.20015[0m
    [34m[147]#011train-rmse:0.567194#011validation-rmse:3.20115[0m
    [34m[148]#011train-rmse:0.567476#011validation-rmse:3.20288[0m
    [34m[149]#011train-rmse:0.566787#011validation-rmse:3.20288[0m
    [34m[150]#011train-rmse:0.564741#011validation-rmse:3.20408[0m
    [34m[151]#011train-rmse:0.564099#011validation-rmse:3.20412[0m
    [34m[152]#011train-rmse:0.563117#011validation-rmse:3.20218[0m
    [34m[153]#011train-rmse:0.557654#011validation-rmse:3.19908[0m
    [34m[154]#011train-rmse:0.551816#011validation-rmse:3.20029[0m
    [34m[155]#011train-rmse:0.548603#011validation-rmse:3.19855[0m
    [34m[156]#011train-rmse:0.547224#011validation-rmse:3.19903[0m
    [34m[157]#011train-rmse:0.544761#011validation-rmse:3.19723[0m
    [34m[158]#011train-rmse:0.538791#011validation-rmse:3.19628[0m
    [34m[159]#011train-rmse:0.538963#011validation-rmse:3.19786[0m
    [34m[160]#011train-rmse:0.538145#011validation-rmse:3.19618[0m
    [34m[161]#011train-rmse:0.535974#011validation-rmse:3.19453[0m
    [34m[162]#011train-rmse:0.531361#011validation-rmse:3.19187[0m
    [34m[163]#011train-rmse:0.529101#011validation-rmse:3.18913[0m
    [34m[164]#011train-rmse:0.526804#011validation-rmse:3.18735[0m
    [34m[165]#011train-rmse:0.525544#011validation-rmse:3.1853[0m
    [34m[166]#011train-rmse:0.524359#011validation-rmse:3.18444[0m
    [34m[167]#011train-rmse:0.523155#011validation-rmse:3.18337[0m
    [34m[168]#011train-rmse:0.519425#011validation-rmse:3.1806[0m
    [34m[169]#011train-rmse:0.515947#011validation-rmse:3.1779[0m
    [34m[170]#011train-rmse:0.513692#011validation-rmse:3.177[0m
    [34m[171]#011train-rmse:0.511426#011validation-rmse:3.17756[0m
    [34m[172]#011train-rmse:0.507754#011validation-rmse:3.17456[0m
    [34m[173]#011train-rmse:0.505584#011validation-rmse:3.17583[0m
    [34m[174]#011train-rmse:0.501595#011validation-rmse:3.17206[0m
    [34m[175]#011train-rmse:0.497798#011validation-rmse:3.16929[0m
    [34m[176]#011train-rmse:0.495894#011validation-rmse:3.16771[0m
    [34m[177]#011train-rmse:0.492881#011validation-rmse:3.16558[0m
    [34m[178]#011train-rmse:0.491018#011validation-rmse:3.16451[0m
    [34m[179]#011train-rmse:0.490556#011validation-rmse:3.16458[0m
    [34m[180]#011train-rmse:0.488146#011validation-rmse:3.16507[0m
    [34m[181]#011train-rmse:0.486202#011validation-rmse:3.16317[0m
    [34m[182]#011train-rmse:0.484817#011validation-rmse:3.16114[0m
    [34m[183]#011train-rmse:0.484832#011validation-rmse:3.1611[0m
    [34m[184]#011train-rmse:0.483512#011validation-rmse:3.15999[0m
    [34m[185]#011train-rmse:0.481192#011validation-rmse:3.16013[0m
    [34m[186]#011train-rmse:0.476721#011validation-rmse:3.16018[0m
    [34m[187]#011train-rmse:0.47672#011validation-rmse:3.16018[0m
    [34m[188]#011train-rmse:0.475942#011validation-rmse:3.15844[0m
    [34m[189]#011train-rmse:0.474417#011validation-rmse:3.15578[0m
    [34m[190]#011train-rmse:0.471076#011validation-rmse:3.15395[0m
    [34m[191]#011train-rmse:0.471082#011validation-rmse:3.15393[0m
    [34m[192]#011train-rmse:0.471081#011validation-rmse:3.15393[0m
    [34m[193]#011train-rmse:0.468338#011validation-rmse:3.15424[0m
    [34m[194]#011train-rmse:0.468459#011validation-rmse:3.15327[0m
    [34m[195]#011train-rmse:0.466324#011validation-rmse:3.15129[0m
    [34m[196]#011train-rmse:0.463832#011validation-rmse:3.15134[0m
    [34m[197]#011train-rmse:0.46116#011validation-rmse:3.15095[0m
    [34m[198]#011train-rmse:0.458851#011validation-rmse:3.14945[0m
    [34m[199]#011train-rmse:0.456419#011validation-rmse:3.15033[0m
    Training seconds: 61
    Billable seconds: 61


## Step 5: Test the model

Now that we have our best performing model, we can test it. To do this we will use the batch transform functionality. To start with, we need to build a transformer object from our fit model.


```python
batch_output = 's3://{}/{}/batch-inference'.format(session.default_bucket(), prefix)
xgb_transformer = xgb_attached.transformer(instance_count=1, instance_type='ml.m4.xlarge', output_path=batch_output)
```

    WARNING:sagemaker:Using already existing model: sagemaker-xgboost-200325-0443-010-632e4608


Next we ask SageMaker to begin a batch transform job using our trained model and applying it to the test data we previous stored in S3. We need to make sure to provide SageMaker with the type of data that we are providing to our model, in our case `text/csv`, so that it knows how to serialize our data. In addition, we need to make sure to let SageMaker know how to split our data up into chunks if the entire data set happens to be too large to send to our model all at once.

Note that when we ask SageMaker to do this it will execute the batch transform job in the background. Since we need to wait for the results of this job before we can continue, we use the `wait()` method. An added benefit of this is that we get some output from our batch transform job which lets us know if anything went wrong.


```python
xgb_transformer.transform(test_location, content_type='text/csv', split_type='Line')
```


```python
xgb_transformer.wait()
```

    ......................[34m[2020-03-25 05:14:18 +0000] [15] [INFO] Starting gunicorn 19.10.0[0m
    [34m[2020-03-25 05:14:18 +0000] [15] [INFO] Listening at: unix:/tmp/gunicorn.sock (15)[0m
    [34m[2020-03-25 05:14:18 +0000] [15] [INFO] Using worker: gevent[0m
    [34m[2020-03-25 05:14:18 +0000] [22] [INFO] Booting worker with pid: 22[0m
    [34m[2020-03-25 05:14:18 +0000] [23] [INFO] Booting worker with pid: 23[0m
    [34m[2020-03-25 05:14:18 +0000] [24] [INFO] Booting worker with pid: 24[0m
    [34m[2020-03-25 05:14:18 +0000] [25] [INFO] Booting worker with pid: 25[0m
    [34m[2020-03-25:05:14:27:INFO] No GPUs detected (normal if no gpus installed)[0m
    [34m169.254.255.130 - - [25/Mar/2020:05:14:27 +0000] "GET /ping HTTP/1.1" 200 0 "-" "Go-http-client/1.1"[0m
    [34m[2020-03-25:05:14:28:INFO] No GPUs detected (normal if no gpus installed)[0m
    [34m169.254.255.130 - - [25/Mar/2020:05:14:28 +0000] "GET /execution-parameters HTTP/1.1" 200 84 "-" "Go-http-client/1.1"[0m
    [34m[2020-03-25:05:14:28:INFO] No GPUs detected (normal if no gpus installed)[0m
    [34m[2020-03-25:05:14:28:INFO] Determined delimiter of CSV input is ','[0m
    [34m[05:14:28] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.[0m
    [35m[2020-03-25:05:14:27:INFO] No GPUs detected (normal if no gpus installed)[0m
    [35m169.254.255.130 - - [25/Mar/2020:05:14:27 +0000] "GET /ping HTTP/1.1" 200 0 "-" "Go-http-client/1.1"[0m
    [35m[2020-03-25:05:14:28:INFO] No GPUs detected (normal if no gpus installed)[0m
    [35m169.254.255.130 - - [25/Mar/2020:05:14:28 +0000] "GET /execution-parameters HTTP/1.1" 200 84 "-" "Go-http-client/1.1"[0m
    [35m[2020-03-25:05:14:28:INFO] No GPUs detected (normal if no gpus installed)[0m
    [35m[2020-03-25:05:14:28:INFO] Determined delimiter of CSV input is ','[0m
    [35m[05:14:28] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.[0m
    [34m169.254.255.130 - - [25/Mar/2020:05:14:28 +0000] "POST /invocations HTTP/1.1" 200 3100 "-" "Go-http-client/1.1"[0m
    [35m169.254.255.130 - - [25/Mar/2020:05:14:28 +0000] "POST /invocations HTTP/1.1" 200 3100 "-" "Go-http-client/1.1"[0m
    [32m2020-03-25T05:14:28.022:[sagemaker logs]: MaxConcurrentTransforms=4, MaxPayloadInMB=6, BatchStrategy=MULTI_RECORD[0m
    


Now that the batch transform job has finished, the resulting output is stored on S3. Since we wish to analyze the output inside of our notebook we can use a bit of notebook magic to copy the output file from its S3 location and save it locally.


```python
!aws s3 cp --recursive $xgb_transformer.output_path $data_dir
```

    download: s3://sagemaker-us-west-2-171758673694/xgboost-hyperparamter-tuning-high-level/batch-inference/test.csv.out to ../data/boston_housing_xgboost_hyperparameter_tuning_high_level/test.csv.out


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




![png](boston_housing_xgboost_hyper_parameter_tuning_high_level_api_performance.png)


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
