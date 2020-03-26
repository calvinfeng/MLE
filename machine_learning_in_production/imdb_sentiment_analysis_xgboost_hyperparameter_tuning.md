# Sentiment Analysis

## Using XGBoost in SageMaker

In this example of using Amazon's SageMaker service we will construct a random tree model to predict
the sentiment of a movie review. You may have seen a version of this example in a pervious lesson
although it would have been done using the sklearn package. Instead, we will be using the XGBoost
package as it is provided to us by Amazon.

## Step 1: Downloading the data

The dataset we are going to use is very popular among researchers in Natural Language Processing,
usually referred to as the [IMDb dataset](http://ai.stanford.edu/~amaas/data/sentiment/). It
consists of movie reviews from the website [imdb.com](http://www.imdb.com/), each labeled as either
'**pos**itive', if the reviewer enjoyed the film, or '**neg**ative' otherwise.

> Maas, Andrew L., et al. [Learning Word Vectors for Sentiment Analysis](http://ai.stanford.edu/~amaas/data/sentiment/).
> In _Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies_. Association for Computational Linguistics, 2011.

We begin by using some Jupyter Notebook magic to download and extract the dataset.

```python
%mkdir ../data
!wget -O ../data/aclImdb_v1.tar.gz http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
!tar -zxf ../data/aclImdb_v1.tar.gz -C ../data
```

```text
mkdir: cannot create directory ‘../data’: File exists
--2020-03-25 05:31:09--  http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
Resolving ai.stanford.edu (ai.stanford.edu)... 171.64.68.10
Connecting to ai.stanford.edu (ai.stanford.edu)|171.64.68.10|:80... connected.
HTTP request sent, awaiting response... 200 OK
Length: 84125825 (80M) [application/x-gzip]
Saving to: ‘../data/aclImdb_v1.tar.gz’
../data/aclImdb_v1. 100%[===================>]  80.23M  39.1MB/s    in 2.0s
2020-03-25 05:31:11 (39.1 MB/s) - ‘../data/aclImdb_v1.tar.gz’ saved [84125825/84125825]
```

## Step 2: Preparing the data

The data we have downloaded is split into various files, each of which contains a single review. It
will be much easier going forward if we combine these individual files into two large files, one for
training and one for testing.

```python
import os
import glob

def read_imdb_data(data_dir='../data/aclImdb'):
    data = {}
    labels = {}

    for data_type in ['train', 'test']:
        data[data_type] = {}
        labels[data_type] = {}

        for sentiment in ['pos', 'neg']:
            data[data_type][sentiment] = []
            labels[data_type][sentiment] = []

            path = os.path.join(data_dir, data_type, sentiment, '*.txt')
            files = glob.glob(path)

            for f in files:
                with open(f) as review:
                    data[data_type][sentiment].append(review.read())
                    # Here we represent a positive review by '1' and a negative review by '0'
                    labels[data_type][sentiment].append(1 if sentiment == 'pos' else 0)

            assert len(data[data_type][sentiment]) == len(labels[data_type][sentiment]), \
                    "{}/{} data size does not match labels size".format(data_type, sentiment)

    return data, labels
```

```python
data, labels = read_imdb_data()
print("IMDB reviews: train = {} pos / {} neg, test = {} pos / {} neg".format(
            len(data['train']['pos']), len(data['train']['neg']),
            len(data['test']['pos']), len(data['test']['neg'])))
```

```text
IMDB reviews: train = 12500 pos / 12500 neg, test = 12500 pos / 12500 neg
```

```python
from sklearn.utils import shuffle

def prepare_imdb_data(data, labels):
    """Prepare training and test sets from IMDb movie reviews."""

    #Combine positive and negative reviews and labels
    data_train = data['train']['pos'] + data['train']['neg']
    data_test = data['test']['pos'] + data['test']['neg']
    labels_train = labels['train']['pos'] + labels['train']['neg']
    labels_test = labels['test']['pos'] + labels['test']['neg']

    #Shuffle reviews and corresponding labels within training and test sets
    data_train, labels_train = shuffle(data_train, labels_train)
    data_test, labels_test = shuffle(data_test, labels_test)

    # Return a unified training data, test data, training labels, test labets
    return data_train, data_test, labels_train, labels_test
```

```python
train_X, test_X, train_y, test_y = prepare_imdb_data(data, labels)
print("IMDb reviews (combined): train = {}, test = {}".format(len(train_X), len(test_X)))
```

```text
IMDb reviews (combined): train = 25000, test = 25000
```

```python
train_X[100]
```

```text
"absolutely trash. i liked Halloween and from then on johnny's been in a downward spiral. this is about the pits. we get it john. pro-lifers are scary! you don't have to make a shitty film that bores the hell out of me to 'tell' me.<br /><br />The pacing is way off here. It feels like john didn't have much to work with here. to his credit it looks like he did not write this junk. There are countless times where the camera just sits and waits for the actors to look dumb or say something dumb. i love the long cut. too bad carpenter doesn't know how to employ it. he needs to bunk up with Herzog and Fassbinder 30 years ago. Please John, stop making a fool of yourself and boring me to death!"
```

## Step 3: Processing the data

Now that we have our training and testing datasets merged and ready to use, we need to start
processing the raw data into something that will be useable by our machine learning algorithm. To
begin with, we remove any html formatting that may appear in the reviews and perform some standard
natural language processing in order to homogenize the data.

```python
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import *
stemmer = PorterStemmer()
```

```text
[nltk_data] Downloading package stopwords to
[nltk_data]     /home/ec2-user/nltk_data...
[nltk_data]   Unzipping corpora/stopwords.zip.
```

```python
import re
from bs4 import BeautifulSoup

def review_to_words(review):
    text = BeautifulSoup(review, "html.parser").get_text() # Remove HTML tags
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) # Convert to lower case
    words = text.split() # Split string into words
    words = [w for w in words if w not in stopwords.words("english")] # Remove stopwords
    words = [PorterStemmer().stem(w) for w in words] # stem

    return words
```

```python
import pickle

cache_dir = os.path.join("../cache", "sentiment_analysis")  # where to store cache files
os.makedirs(cache_dir, exist_ok=True)  # ensure cache directory exists

def preprocess_data(data_train, data_test, labels_train, labels_test,
                    cache_dir=cache_dir, cache_file="preprocessed_data.pkl"):
    """Convert each review to words; read from cache if available."""

    # If cache_file is not None, try to read from it first
    cache_data = None
    if cache_file is not None:
        try:
            with open(os.path.join(cache_dir, cache_file), "rb") as f:
                cache_data = pickle.load(f)
            print("Read preprocessed data from cache file:", cache_file)
        except:
            pass  # unable to read from cache, but that's okay

    # If cache is missing, then do the heavy lifting
    if cache_data is None:
        # Preprocess training and test data to obtain words for each review
        #words_train = list(map(review_to_words, data_train))
        #words_test = list(map(review_to_words, data_test))
        words_train = [review_to_words(review) for review in data_train]
        words_test = [review_to_words(review) for review in data_test]

        # Write to cache file for future runs
        if cache_file is not None:
            cache_data = dict(words_train=words_train, words_test=words_test,
                              labels_train=labels_train, labels_test=labels_test)
            with open(os.path.join(cache_dir, cache_file), "wb") as f:
                pickle.dump(cache_data, f)
            print("Wrote preprocessed data to cache file:", cache_file)
    else:
        # Unpack data loaded from cache file
        words_train, words_test, labels_train, labels_test = (cache_data['words_train'],
                cache_data['words_test'], cache_data['labels_train'], cache_data['labels_test'])

    return words_train, words_test, labels_train, labels_test
```

```python
# Preprocess data
train_X, test_X, train_y, test_y = preprocess_data(train_X, test_X, train_y, test_y)
```

```text
Wrote preprocessed data to cache file: preprocessed_data.pkl
```

### Extract Bag-of-Words features

For the model we will be implementing, rather than using the reviews directly, we are going to
transform each review into a Bag-of-Words feature representation. Keep in mind that 'in the wild' we
will only have access to the training set so our transformer can only use the training set to
construct a representation.

```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
# joblib is an enhanced version of pickle that is more efficient for storing NumPy arrays

def extract_BoW_features(words_train, words_test, vocabulary_size=5000,
                         cache_dir=cache_dir, cache_file="bow_features.pkl"):
    """Extract Bag-of-Words for a given set of documents, already preprocessed into words."""

    # If cache_file is not None, try to read from it first
    cache_data = None
    if cache_file is not None:
        try:
            with open(os.path.join(cache_dir, cache_file), "rb") as f:
                cache_data = joblib.load(f)
            print("Read features from cache file:", cache_file)
        except:
            pass  # unable to read from cache, but that's okay

    # If cache is missing, then do the heavy lifting
    if cache_data is None:
        # Fit a vectorizer to training documents and use it to transform them
        # NOTE: Training documents have already been preprocessed and tokenized into words;
        #       pass in dummy functions to skip those steps, e.g. preprocessor=lambda x: x
        vectorizer = CountVectorizer(max_features=vocabulary_size,
                preprocessor=lambda x: x, tokenizer=lambda x: x)  # already preprocessed
        features_train = vectorizer.fit_transform(words_train).toarray()

        # Apply the same vectorizer to transform the test documents (ignore unknown words)
        features_test = vectorizer.transform(words_test).toarray()

        # NOTE: Remember to convert the features using .toarray() for a compact representation

        # Write to cache file for future runs (store vocabulary as well)
        if cache_file is not None:
            vocabulary = vectorizer.vocabulary_
            cache_data = dict(features_train=features_train, features_test=features_test,
                             vocabulary=vocabulary)
            with open(os.path.join(cache_dir, cache_file), "wb") as f:
                joblib.dump(cache_data, f)
            print("Wrote features to cache file:", cache_file)
    else:
        # Unpack data loaded from cache file
        features_train, features_test, vocabulary = (cache_data['features_train'],
                cache_data['features_test'], cache_data['vocabulary'])

    # Return both the extracted features as well as the vocabulary
    return features_train, features_test, vocabulary
```

```python
# Extract Bag of Words features for both training and test datasets
train_X, test_X, vocabulary = extract_BoW_features(train_X, test_X)
```

```text
Wrote features to cache file: bow_features.pkl
```

## Step 4: Classification using XGBoost

Now that we have created the feature representation of our training (and testing) data, it is time
to start setting up and using the XGBoost classifier provided by SageMaker.

### Writing the dataset

The XGBoost classifier that we will be using requires the dataset to be written to a file and stored
using Amazon S3. To do this, we will start by splitting the training dataset into two parts, the
data we will train the model with and a validation set. Then, we will write those datasets to a file
and upload the files to S3. In addition, we will write the test set input to a file and upload the
file to S3. This is so that we can use SageMakers Batch Transform functionality to test our model
once we've fit it.

```python
import pandas as pd

val_X = pd.DataFrame(train_X[:10000])
train_X = pd.DataFrame(train_X[10000:])

val_y = pd.DataFrame(train_y[:10000])
train_y = pd.DataFrame(train_y[10000:])

test_y = pd.DataFrame(test_y)
test_X = pd.DataFrame(test_X)
```

The documentation for the XGBoost algorithm in SageMaker requires that the saved datasets should
contain no headers or index and that for the training and validation data, the label should occur
first for each sample.

For more information about this and other algorithms, the SageMaker developer documentation can be
found on __[Amazon's website.](https://docs.aws.amazon.com/sagemaker/latest/dg/)__

```python
# First we make sure that the local directory in which we'd like to store the training and validation csv files exists.
data_dir = '../data/sentiment-analysis-xgboost-hyperparameter-tuning'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
```

```python
# First, save the test data to test.csv in the data_dir directory. Note that we do not save the associated ground truth
# labels, instead we will use them later to compare with our model output.
pd.DataFrame(test_X).to_csv(os.path.join(data_dir, 'test.csv'), header=False, index=False)
pd.concat([val_y, val_X], axis=1).to_csv(os.path.join(data_dir, 'validation.csv'), header=False, index=False)
pd.concat([train_y, train_X], axis=1).to_csv(os.path.join(data_dir, 'train.csv'), header=False, index=False)
```

```python
# To save a bit of memory we can set text_X, train_X, val_X, train_y and val_y to None.
train_X = val_X = train_y = val_y = None
```

### Uploading Training / Validation files to S3

Amazon's S3 service allows us to store files that can be access by both the built-in training models
such as the XGBoost model we will be using as well as custom models such as the one we will see a
little later.

For this, and most other tasks we will be doing using SageMaker, there are two methods we could use.
The first is to use the low level functionality of SageMaker which requires knowing each of the
objects involved in the SageMaker environment. The second is to use the high level functionality in
which certain choices have been made on the user's behalf. The low level approach benefits from
allowing the user a great deal of flexibility while the high level approach makes development much
quicker. For our purposes we will opt to use the high level approach although using the low-level
approach is certainly an option.

Recall the method `upload_data()` which is a member of object representing our current SageMaker
session. What this method does is upload the data to the default bucket (which is created if it does
not exist) into the path described by the key_prefix variable. To see this for yourself, once you
have uploaded the data files, go to the S3 console and look to see where the files have been
uploaded.

For additional resources, see the __[SageMaker API documentation](http://sagemaker.readthedocs.io/en/latest/)__
and in addition the __[SageMaker Developer Guide.](https://docs.aws.amazon.com/sagemaker/latest/dg/)__

```python
import sagemaker

session = sagemaker.Session() # Store the current SageMaker session

# S3 prefix (which folder will we use)
prefix = 'sentiment-analysis-xgboost-hyperparameter-tuning'
test_location = session.upload_data(os.path.join(data_dir, 'test.csv'), key_prefix=prefix)
val_location = session.upload_data(os.path.join(data_dir, 'validation.csv'), key_prefix=prefix)
train_location = session.upload_data(os.path.join(data_dir, 'train.csv'), key_prefix=prefix)
```

### (TODO) Creating a hypertuned XGBoost model

Now that the data has been uploaded it is time to create the XGBoost model. As in the Boston Housing
notebook, the first step is to create an estimator object which will be used as the *base* of your
hyperparameter tuning job.

```python
from sagemaker import get_execution_role

# Our current execution role is require when creating the model as the training
# and inference code will need to access the model artifacts.
role = get_execution_role()
```

```python
# We need to retrieve the location of the container which is provided by Amazon for using XGBoost.
# As a matter of convenience, the training and inference code both use the same container.
from sagemaker.amazon.amazon_estimator import get_image_uri

container = get_image_uri(session.boto_region_name, 'xgboost', '0.90-1')
```

```python
# TODO: Create a SageMaker estimator using the container location determined in the previous cell.
#       It is recommended that you use a single training instance of type ml.m4.xlarge. It is also
#       recommended that you use 's3://{}/{}/output'.format(session.default_bucket(), prefix) as the
#       output path.
xgb = sagemaker.estimator.Estimator(container,
                                    role,  
                                    train_instance_count=1,
                                    train_instance_type='ml.m4.xlarge',
                                    output_path='s3://{}/{}/'.format(session.default_bucket(), prefix),
                                    base_job_name='xgboost-hyperparam-tuning-job',
                                    sagemaker_session=session)

# TODO: Set the XGBoost hyperparameters in the xgb object. Don't forget that in this case we have a binary
#       label so we should be using the 'binary:logistic' objective.
xgb.set_hyperparameters(max_depth=5,
                        eta=0.2,
                        gamma=4,
                        min_child_weight=6,
                        subsample=0.8,
                        objective='reg:linear',
                        early_stopping_rounds=10,
                        num_round=200)
```

### (TODO) Create the hyperparameter tuner

Now that the base estimator has been set up we need to construct a hyperparameter tuner object which
we will use to request SageMaker construct a hyperparameter tuning job.

**Note:** Training a single sentiment analysis XGBoost model takes longer than training a Boston
Housing XGBoost model so if you don't want the hyperparameter tuning job to take too long, make sure
to not set the total number of models (jobs) too high.

```python
# First, make sure to import the relevant objects used to construct the tuner
from sagemaker.tuner import IntegerParameter, ContinuousParameter, HyperparameterTuner

# TODO: Create the hyperparameter tuner object
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

### Fit the hyperparameter tuner

Now that the hyperparameter tuner object has been constructed, it is time to fit the various models
and find the best performing model.

```python
s3_input_train = sagemaker.s3_input(s3_data=train_location, content_type='csv')
s3_input_validation = sagemaker.s3_input(s3_data=val_location, content_type='csv')
```

```python
xgb_hyperparameter_tuner.fit({'train': s3_input_train, 'validation': s3_input_validation})
```

Remember that the tuning job is constructed and run in the background so if we want to see the
progress of our training job we need to call the `wait()` method.

```python
xgb_hyperparameter_tuner.wait()
```

### (TODO) Testing the model

Now that we've run our hyperparameter tuning job, it's time to see how well the best performing
model actually performs. To do this we will use SageMaker's Batch Transform functionality. Batch
Transform is a convenient way to perform inference on a large dataset in a way that is not realtime.
That is, we don't necessarily need to use our model's results immediately and instead we can peform
inference on a large number of samples. An example of this in industry might be peforming an end of
month report. This method of inference can also be useful to us as it means to can perform inference
on our entire test set.

Remember that in order to create a transformer object to perform the batch transform job, we need a
trained estimator object. We can do that using the `attach()` method, creating an estimator object
which is attached to the best trained job.

```python
# TODO: Create a new estimator object attached to the best training job found during hyperparameter tuning
xgb_attached = sagemaker.estimator.Estimator.attach(xgb_hyperparameter_tuner.best_training_job())
```

```text
2020-03-25 07:14:04 Starting - Preparing the instances for training
2020-03-25 07:14:04 Downloading - Downloading input data
2020-03-25 07:14:04 Training - Training image download completed. Training in progress.
2020-03-25 07:14:04 Uploading - Uploading generated training model
2020-03-25 07:14:04 Completed - Training job completed[34mINFO:sagemaker-containers:Imported framework sagemaker_xgboost_container.training[0m
[34mINFO:sagemaker-containers:Failed to parse hyperparameter _tuning_objective_metric value validation:rmse to Json.[0m
[34mReturning the value itself[0m
[34mINFO:sagemaker-containers:Failed to parse hyperparameter objective value reg:linear to Json.[0m
[34mReturning the value itself[0m
[34mINFO:sagemaker-containers:No GPUs detected (normal if no gpus installed)[0m
[34mINFO:sagemaker_xgboost_container.training:Running XGBoost Sagemaker in algorithm mode[0m
[34mINFO:root:Determined delimiter of CSV input is ','[0m
[34mINFO:root:Determined delimiter of CSV input is ','[0m
[34mINFO:root:Determined delimiter of CSV input is ','[0m
[34m[07:06:34] 15000x5000 matrix with 75000000 entries loaded from /opt/ml/input/data/train?format=csv&label_column=0&delimiter=,[0m
[34mINFO:root:Determined delimiter of CSV input is ','[0m
[34m[07:06:35] 10000x5000 matrix with 50000000 entries loaded from /opt/ml/input/data/validation?format=csv&label_column=0&delimiter=,[0m
[34mINFO:root:Single node training.[0m
[34mINFO:root:Setting up HPO optimized metric to be : rmse[0m
[34mINFO:root:Train matrix has 15000 rows[0m
[34mINFO:root:Validation matrix has 10000 rows[0m
[34m[07:06:35] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.[0m
[34m[0]#011train-rmse:0.473473#011validation-rmse:0.478049[0m
[34m[1]#011train-rmse:0.452029#011validation-rmse:0.461175[0m
[34m[2]#011train-rmse:0.435042#011validation-rmse:0.448066[0m
[34m[3]#011train-rmse:0.420251#011validation-rmse:0.437589[0m
[34m[4]#011train-rmse:0.408457#011validation-rmse:0.429379[0m
[34m[5]#011train-rmse:0.397599#011validation-rmse:0.422033[0m
[34m[6]#011train-rmse:0.387401#011validation-rmse:0.416546[0m
[34m[7]#011train-rmse:0.378368#011validation-rmse:0.411519[0m
[34m[8]#011train-rmse:0.371087#011validation-rmse:0.406759[0m
[34m[9]#011train-rmse:0.363975#011validation-rmse:0.402781[0m
[34m[10]#011train-rmse:0.358417#011validation-rmse:0.399207[0m
[34m[11]#011train-rmse:0.352734#011validation-rmse:0.395754[0m
[34m[12]#011train-rmse:0.34776#011validation-rmse:0.392536[0m
[34m[13]#011train-rmse:0.342828#011validation-rmse:0.389836[0m
[34m[14]#011train-rmse:0.337449#011validation-rmse:0.387527[0m
[34m[15]#011train-rmse:0.333441#011validation-rmse:0.385331[0m
[34m[16]#011train-rmse:0.329632#011validation-rmse:0.382974[0m
[34m[17]#011train-rmse:0.324897#011validation-rmse:0.381136[0m
[34m[18]#011train-rmse:0.32194#011validation-rmse:0.379099[0m
[34m[19]#011train-rmse:0.319192#011validation-rmse:0.377268[0m
[34m[20]#011train-rmse:0.316679#011validation-rmse:0.375926[0m
[34m[21]#011train-rmse:0.31368#011validation-rmse:0.374523[0m
[34m[22]#011train-rmse:0.310168#011validation-rmse:0.373256[0m
[34m[23]#011train-rmse:0.307211#011validation-rmse:0.371827[0m
[34m[24]#011train-rmse:0.304598#011validation-rmse:0.370587[0m
[34m[25]#011train-rmse:0.302071#011validation-rmse:0.369616[0m
[34m[26]#011train-rmse:0.299743#011validation-rmse:0.368529[0m
[34m[27]#011train-rmse:0.297465#011validation-rmse:0.3674[0m
[34m[28]#011train-rmse:0.295771#011validation-rmse:0.366715[0m
[34m[29]#011train-rmse:0.294083#011validation-rmse:0.365832[0m
[34m[30]#011train-rmse:0.292147#011validation-rmse:0.365122[0m
[34m[31]#011train-rmse:0.290252#011validation-rmse:0.364209[0m
[34m[32]#011train-rmse:0.288682#011validation-rmse:0.363562[0m
[34m[33]#011train-rmse:0.287394#011validation-rmse:0.363086[0m
[34m[34]#011train-rmse:0.286382#011validation-rmse:0.362518[0m
[34m[35]#011train-rmse:0.284963#011validation-rmse:0.361873[0m
[34m[36]#011train-rmse:0.28402#011validation-rmse:0.361175[0m
[34m[37]#011train-rmse:0.282918#011validation-rmse:0.360589[0m
[34m[38]#011train-rmse:0.281748#011validation-rmse:0.360027[0m
[34m[39]#011train-rmse:0.280032#011validation-rmse:0.359511[0m
[34m[40]#011train-rmse:0.278498#011validation-rmse:0.358915[0m
[34m[41]#011train-rmse:0.277374#011validation-rmse:0.358519[0m
[34m[42]#011train-rmse:0.27647#011validation-rmse:0.358024[0m
[34m[43]#011train-rmse:0.275511#011validation-rmse:0.357573[0m
[34m[44]#011train-rmse:0.274461#011validation-rmse:0.357054[0m
[34m[45]#011train-rmse:0.27304#011validation-rmse:0.356582[0m
[34m[46]#011train-rmse:0.271997#011validation-rmse:0.356149[0m
[34m[47]#011train-rmse:0.270752#011validation-rmse:0.355784[0m
[34m[48]#011train-rmse:0.270115#011validation-rmse:0.35559[0m
[34m[49]#011train-rmse:0.269276#011validation-rmse:0.355169[0m
[34m[50]#011train-rmse:0.268458#011validation-rmse:0.35491[0m
[34m[51]#011train-rmse:0.267916#011validation-rmse:0.354477[0m
[34m[52]#011train-rmse:0.267244#011validation-rmse:0.354169[0m
[34m[53]#011train-rmse:0.266325#011validation-rmse:0.354003[0m
[34m[54]#011train-rmse:0.265814#011validation-rmse:0.353737[0m
[34m[55]#011train-rmse:0.264896#011validation-rmse:0.353334[0m
[34m[56]#011train-rmse:0.264487#011validation-rmse:0.352981[0m
[34m[57]#011train-rmse:0.2633#011validation-rmse:0.352483[0m
[34m[58]#011train-rmse:0.262784#011validation-rmse:0.352211[0m
[34m[59]#011train-rmse:0.262178#011validation-rmse:0.352084[0m
[34m[60]#011train-rmse:0.261407#011validation-rmse:0.351894[0m
[34m[61]#011train-rmse:0.260856#011validation-rmse:0.351687[0m
[34m[62]#011train-rmse:0.260421#011validation-rmse:0.351522[0m
[34m[63]#011train-rmse:0.259848#011validation-rmse:0.351273[0m
[34m[64]#011train-rmse:0.259512#011validation-rmse:0.351081[0m
[34m[65]#011train-rmse:0.258727#011validation-rmse:0.350838[0m
[34m[66]#011train-rmse:0.258265#011validation-rmse:0.350841[0m
[34m[67]#011train-rmse:0.257917#011validation-rmse:0.350675[0m
[34m[68]#011train-rmse:0.2576#011validation-rmse:0.350517[0m
[34m[69]#011train-rmse:0.256998#011validation-rmse:0.350344[0m
[34m[70]#011train-rmse:0.256513#011validation-rmse:0.350194[0m
[34m[71]#011train-rmse:0.256221#011validation-rmse:0.350051[0m
[34m[72]#011train-rmse:0.255617#011validation-rmse:0.349862[0m
[34m[73]#011train-rmse:0.255309#011validation-rmse:0.349769[0m
[34m[74]#011train-rmse:0.254611#011validation-rmse:0.349673[0m
[34m[75]#011train-rmse:0.254357#011validation-rmse:0.349471[0m
[34m[76]#011train-rmse:0.253717#011validation-rmse:0.349429[0m
[34m[77]#011train-rmse:0.253415#011validation-rmse:0.349291[0m
[34m[78]#011train-rmse:0.253032#011validation-rmse:0.349079[0m
[34m[79]#011train-rmse:0.252396#011validation-rmse:0.348919[0m
[34m[80]#011train-rmse:0.251997#011validation-rmse:0.348767[0m
[34m[81]#011train-rmse:0.251667#011validation-rmse:0.348651[0m
[34m[82]#011train-rmse:0.251588#011validation-rmse:0.348611[0m
[34m[83]#011train-rmse:0.251364#011validation-rmse:0.348476[0m
[34m[84]#011train-rmse:0.251005#011validation-rmse:0.348353[0m
[34m[85]#011train-rmse:0.250591#011validation-rmse:0.348131[0m
[34m[86]#011train-rmse:0.250345#011validation-rmse:0.348044[0m
[34m[87]#011train-rmse:0.249831#011validation-rmse:0.347956[0m
[34m[88]#011train-rmse:0.249755#011validation-rmse:0.347885[0m
[34m[89]#011train-rmse:0.249535#011validation-rmse:0.347772[0m
[34m[90]#011train-rmse:0.249213#011validation-rmse:0.347647[0m
[34m[91]#011train-rmse:0.248832#011validation-rmse:0.347537[0m
[34m[92]#011train-rmse:0.248577#011validation-rmse:0.347408[0m
[34m[93]#011train-rmse:0.248308#011validation-rmse:0.347217[0m
[34m[94]#011train-rmse:0.248308#011validation-rmse:0.347217[0m
[34m[95]#011train-rmse:0.248033#011validation-rmse:0.347076[0m
[34m[96]#011train-rmse:0.247925#011validation-rmse:0.347035[0m
[34m[97]#011train-rmse:0.247689#011validation-rmse:0.346925[0m
[34m[98]#011train-rmse:0.247452#011validation-rmse:0.346829[0m
[34m[99]#011train-rmse:0.247158#011validation-rmse:0.346735[0m
[34m[100]#011train-rmse:0.246739#011validation-rmse:0.346617[0m
[34m[101]#011train-rmse:0.24637#011validation-rmse:0.346457[0m
[34m[102]#011train-rmse:0.246229#011validation-rmse:0.346443[0m
[34m[103]#011train-rmse:0.246213#011validation-rmse:0.346443[0m
[34m[104]#011train-rmse:0.245744#011validation-rmse:0.346435[0m
[34m[105]#011train-rmse:0.245488#011validation-rmse:0.346426[0m
[34m[106]#011train-rmse:0.245269#011validation-rmse:0.346284[0m
[34m[107]#011train-rmse:0.245269#011validation-rmse:0.346283[0m
[34m[108]#011train-rmse:0.245227#011validation-rmse:0.346263[0m
[34m[109]#011train-rmse:0.245189#011validation-rmse:0.346254[0m
[34m[110]#011train-rmse:0.245008#011validation-rmse:0.346183[0m
[34m[111]#011train-rmse:0.245008#011validation-rmse:0.346184[0m
[34m[112]#011train-rmse:0.245008#011validation-rmse:0.346188[0m
[34m[113]#011train-rmse:0.245008#011validation-rmse:0.346189[0m
[34m[114]#011train-rmse:0.24484#011validation-rmse:0.346132[0m
[34m[115]#011train-rmse:0.244607#011validation-rmse:0.346151[0m
[34m[116]#011train-rmse:0.244607#011validation-rmse:0.346148[0m
[34m[117]#011train-rmse:0.244437#011validation-rmse:0.346129[0m
[34m[118]#011train-rmse:0.244138#011validation-rmse:0.346036[0m
[34m[119]#011train-rmse:0.243761#011validation-rmse:0.345802[0m
[34m[120]#011train-rmse:0.24358#011validation-rmse:0.345734[0m
[34m[121]#011train-rmse:0.243411#011validation-rmse:0.345704[0m
[34m[122]#011train-rmse:0.24324#011validation-rmse:0.345602[0m
[34m[123]#011train-rmse:0.243123#011validation-rmse:0.345583[0m
[34m[124]#011train-rmse:0.243085#011validation-rmse:0.345555[0m
[34m[125]#011train-rmse:0.243085#011validation-rmse:0.345553[0m
[34m[126]#011train-rmse:0.242918#011validation-rmse:0.3455[0m
[34m[127]#011train-rmse:0.24273#011validation-rmse:0.345391[0m
[34m[128]#011train-rmse:0.242507#011validation-rmse:0.345305[0m
[34m[129]#011train-rmse:0.242507#011validation-rmse:0.345298[0m
[34m[130]#011train-rmse:0.242507#011validation-rmse:0.345305[0m
[34m[131]#011train-rmse:0.242243#011validation-rmse:0.345211[0m
[34m[132]#011train-rmse:0.242029#011validation-rmse:0.345069[0m
[34m[133]#011train-rmse:0.24195#011validation-rmse:0.345018[0m
[34m[134]#011train-rmse:0.241739#011validation-rmse:0.344923[0m
[34m[135]#011train-rmse:0.24161#011validation-rmse:0.34487[0m
[34m[136]#011train-rmse:0.24158#011validation-rmse:0.344873[0m
[34m[137]#011train-rmse:0.241438#011validation-rmse:0.344868[0m
[34m[138]#011train-rmse:0.241438#011validation-rmse:0.344868[0m
[34m[139]#011train-rmse:0.241438#011validation-rmse:0.344868[0m
[34m[140]#011train-rmse:0.241438#011validation-rmse:0.344864[0m
[34m[141]#011train-rmse:0.241256#011validation-rmse:0.344848[0m
[34m[142]#011train-rmse:0.241099#011validation-rmse:0.344787[0m
[34m[143]#011train-rmse:0.241099#011validation-rmse:0.344787[0m
[34m[144]#011train-rmse:0.240985#011validation-rmse:0.344727[0m
[34m[145]#011train-rmse:0.240985#011validation-rmse:0.344728[0m
[34m[146]#011train-rmse:0.240804#011validation-rmse:0.344697[0m
[34m[147]#011train-rmse:0.240805#011validation-rmse:0.344701[0m
[34m[148]#011train-rmse:0.240805#011validation-rmse:0.344705[0m
[34m[149]#011train-rmse:0.240805#011validation-rmse:0.344707[0m
[34m[150]#011train-rmse:0.240805#011validation-rmse:0.34471[0m
[34m[151]#011train-rmse:0.240805#011validation-rmse:0.344707[0m
[34m[152]#011train-rmse:0.240569#011validation-rmse:0.344633[0m
[34m[153]#011train-rmse:0.240431#011validation-rmse:0.344557[0m
[34m[154]#011train-rmse:0.240259#011validation-rmse:0.344507[0m
[34m[155]#011train-rmse:0.240094#011validation-rmse:0.344476[0m
[34m[156]#011train-rmse:0.240094#011validation-rmse:0.344477[0m
[34m[157]#011train-rmse:0.239842#011validation-rmse:0.34437[0m
[34m[158]#011train-rmse:0.239843#011validation-rmse:0.344374[0m
[34m[159]#011train-rmse:0.239842#011validation-rmse:0.34437[0m
[34m[160]#011train-rmse:0.239842#011validation-rmse:0.344362[0m
[34m[161]#011train-rmse:0.239627#011validation-rmse:0.34429[0m
[34m[162]#011train-rmse:0.239627#011validation-rmse:0.344292[0m
[34m[163]#011train-rmse:0.239628#011validation-rmse:0.344293[0m
[34m[164]#011train-rmse:0.239474#011validation-rmse:0.34425[0m
[34m[165]#011train-rmse:0.239389#011validation-rmse:0.34421[0m
[34m[166]#011train-rmse:0.239284#011validation-rmse:0.344187[0m
[34m[167]#011train-rmse:0.239137#011validation-rmse:0.344176[0m
[34m[168]#011train-rmse:0.239136#011validation-rmse:0.344171[0m
[34m[169]#011train-rmse:0.239136#011validation-rmse:0.344171[0m
[34m[170]#011train-rmse:0.239136#011validation-rmse:0.344167[0m
[34m[171]#011train-rmse:0.23892#011validation-rmse:0.34415[0m
[34m[172]#011train-rmse:0.23892#011validation-rmse:0.344147[0m
[34m[173]#011train-rmse:0.238848#011validation-rmse:0.344114[0m
[34m[174]#011train-rmse:0.238636#011validation-rmse:0.344079[0m
[34m[175]#011train-rmse:0.238493#011validation-rmse:0.344059[0m
[34m[176]#011train-rmse:0.238493#011validation-rmse:0.344062[0m
[34m[177]#011train-rmse:0.238332#011validation-rmse:0.343983[0m
[34m[178]#011train-rmse:0.238332#011validation-rmse:0.343982[0m
[34m[179]#011train-rmse:0.238332#011validation-rmse:0.343982[0m
[34m[180]#011train-rmse:0.238332#011validation-rmse:0.343979[0m
[34m[181]#011train-rmse:0.238224#011validation-rmse:0.343988[0m
[34m[182]#011train-rmse:0.238056#011validation-rmse:0.343935[0m
[34m[183]#011train-rmse:0.238056#011validation-rmse:0.343932[0m
[34m[184]#011train-rmse:0.238056#011validation-rmse:0.343931[0m
[34m[185]#011train-rmse:0.237996#011validation-rmse:0.343878[0m
[34m[186]#011train-rmse:0.237996#011validation-rmse:0.343872[0m
[34m[187]#011train-rmse:0.237874#011validation-rmse:0.343847[0m
[34m[188]#011train-rmse:0.237874#011validation-rmse:0.343848[0m
[34m[189]#011train-rmse:0.237874#011validation-rmse:0.34385[0m
[34m[190]#011train-rmse:0.2377#011validation-rmse:0.343763[0m
[34m[191]#011train-rmse:0.237655#011validation-rmse:0.343749[0m
[34m[192]#011train-rmse:0.237656#011validation-rmse:0.343744[0m
[34m[193]#011train-rmse:0.237656#011validation-rmse:0.343741[0m
[34m[194]#011train-rmse:0.237656#011validation-rmse:0.34374[0m
[34m[195]#011train-rmse:0.237656#011validation-rmse:0.34374[0m
[34m[196]#011train-rmse:0.237655#011validation-rmse:0.343746[0m
[34m[197]#011train-rmse:0.237655#011validation-rmse:0.34375[0m
[34m[198]#011train-rmse:0.237655#011validation-rmse:0.343749[0m
[34m[199]#011train-rmse:0.237656#011validation-rmse:0.343745[0m
Training seconds: 505
Billable seconds: 505
```

Now that we have an estimator object attached to the correct training job, we can proceed as we
normally would and create a transformer object.

```python
# TODO: Create a transformer object from the attached estimator. Using an instance count of 1 and an instance type of ml.m4.xlarge
#       should be more than enough.
batch_output = 's3://{}/{}/batch-inference'.format(session.default_bucket(), prefix)
xgb_transformer = xgb_attached.transformer(instance_count=1,
                                           instance_type='ml.m4.xlarge',
                                           output_path=batch_output)
```

Next we actually perform the transform job. When doing so we need to make sure to specify the type
of data we are sending so that it is serialized correctly in the background. In our case we are
providing our model with csv data so we specify `text/csv`. Also, if the test data that we have
provided is too large to process all at once then we need to specify how the data file should be
split up. Since each line is a single entry in our data set we tell SageMaker that it can split the
input on each line.

```python
# TODO: Start the transform job. Make sure to specify the content type and the split type of the test data.
xgb_transformer.transform(test_location, content_type='text/csv',
                                         split_type='Line')
```

Currently the transform job is running but it is doing so in the background. Since we wish to wait
until the transform job is done and we would like a bit of feedback we can run the `wait()` method.

```python
xgb_transformer.wait()
```

```text
......................
[34m[2020-03-25 07:57:24 +0000] [15] [INFO] Starting gunicorn 19.10.0[0m
[34m[2020-03-25 07:57:24 +0000] [15] [INFO] Listening at: unix:/tmp/gunicorn.sock (15)[0m
[34m[2020-03-25 07:57:24 +0000] [15] [INFO] Using worker: gevent[0m
[34m[2020-03-25 07:57:24 +0000] [22] [INFO] Booting worker with pid: 22[0m
[34m[2020-03-25 07:57:24 +0000] [23] [INFO] Booting worker with pid: 23[0m
[34m[2020-03-25 07:57:24 +0000] [24] [INFO] Booting worker with pid: 24[0m
[34m[2020-03-25 07:57:24 +0000] [25] [INFO] Booting worker with pid: 25[0m
[34m[2020-03-25:07:57:45:INFO] No GPUs detected (normal if no gpus installed)[0m
[34m169.254.255.130 - - [25/Mar/2020:07:57:45 +0000] "GET /ping HTTP/1.1" 200 0 "-" "Go-http-client/1.1"[0m
[34m[2020-03-25:07:57:45:INFO] No GPUs detected (normal if no gpus installed)[0m
[34m169.254.255.130 - - [25/Mar/2020:07:57:45 +0000] "GET /execution-parameters HTTP/1.1" 200 84 "-" "Go-http-client/1.1"[0m
[35m[2020-03-25:07:57:45:INFO] No GPUs detected (normal if no gpus installed)[0m
[35m169.254.255.130 - - [25/Mar/2020:07:57:45 +0000] "GET /ping HTTP/1.1" 200 0 "-" "Go-http-client/1.1"[0m
[35m[2020-03-25:07:57:45:INFO] No GPUs detected (normal if no gpus installed)[0m
[35m169.254.255.130 - - [25/Mar/2020:07:57:45 +0000] "GET /execution-parameters HTTP/1.1" 200 84 "-" "Go-http-client/1.1"[0m
[32m2020-03-25T07:57:45.857:[sagemaker logs]: MaxConcurrentTransforms=4, MaxPayloadInMB=6, BatchStrategy=MULTI_RECORD[0m
[34m[2020-03-25:07:57:48:INFO] No GPUs detected (normal if no gpus installed)[0m
[34m[2020-03-25:07:57:48:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-25:07:57:48:INFO] No GPUs detected (normal if no gpus installed)[0m
[34m[2020-03-25:07:57:48:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-25:07:57:48:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-25:07:57:48:INFO] No GPUs detected (normal if no gpus installed)[0m
[35m[2020-03-25:07:57:48:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-25:07:57:48:INFO] No GPUs detected (normal if no gpus installed)[0m
[35m[2020-03-25:07:57:48:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-25:07:57:48:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-25:07:57:49:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-25:07:57:49:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [25/Mar/2020:07:57:51 +0000] "POST /invocations HTTP/1.1" 200 12237 "-" "Go-http-client/1.1"[0m
[34m[07:57:51] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.[0m
[34m169.254.255.130 - - [25/Mar/2020:07:57:51 +0000] "POST /invocations HTTP/1.1" 200 12183 "-" "Go-http-client/1.1"[0m
[34m[07:57:51] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.[0m
[34m169.254.255.130 - - [25/Mar/2020:07:57:51 +0000] "POST /invocations HTTP/1.1" 200 12179 "-" "Go-http-client/1.1"[0m
[34m[2020-03-25:07:57:52:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-25:07:57:52:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-25:07:57:52:INFO] Determined delimiter of CSV input is ','[0m
[34m[07:57:52] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.[0m
[34m169.254.255.130 - - [25/Mar/2020:07:57:52 +0000] "POST /invocations HTTP/1.1" 200 12205 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [25/Mar/2020:07:57:51 +0000] "POST /invocations HTTP/1.1" 200 12237 "-" "Go-http-client/1.1"[0m
[35m[07:57:51] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.[0m
[35m169.254.255.130 - - [25/Mar/2020:07:57:51 +0000] "POST /invocations HTTP/1.1" 200 12183 "-" "Go-http-client/1.1"[0m
[35m[07:57:51] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.[0m
[35m169.254.255.130 - - [25/Mar/2020:07:57:51 +0000] "POST /invocations HTTP/1.1" 200 12179 "-" "Go-http-client/1.1"[0m
[35m[2020-03-25:07:57:52:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-25:07:57:52:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-25:07:57:52:INFO] Determined delimiter of CSV input is ','[0m
[35m[07:57:52] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.[0m
[35m169.254.255.130 - - [25/Mar/2020:07:57:52 +0000] "POST /invocations HTTP/1.1" 200 12205 "-" "Go-http-client/1.1"[0m
[34m[2020-03-25:07:57:52:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-25:07:57:52:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [25/Mar/2020:07:57:55 +0000] "POST /invocations HTTP/1.1" 200 12199 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [25/Mar/2020:07:57:55 +0000] "POST /invocations HTTP/1.1" 200 12168 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [25/Mar/2020:07:57:55 +0000] "POST /invocations HTTP/1.1" 200 12193 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [25/Mar/2020:07:57:55 +0000] "POST /invocations HTTP/1.1" 200 12199 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [25/Mar/2020:07:57:55 +0000] "POST /invocations HTTP/1.1" 200 12168 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [25/Mar/2020:07:57:55 +0000] "POST /invocations HTTP/1.1" 200 12193 "-" "Go-http-client/1.1"[0m
[34m[2020-03-25:07:57:55:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [25/Mar/2020:07:57:55 +0000] "POST /invocations HTTP/1.1" 200 12194 "-" "Go-http-client/1.1"[0m
[34m[2020-03-25:07:57:55:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-25:07:57:55:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-25:07:57:55:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-25:07:57:55:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [25/Mar/2020:07:57:55 +0000] "POST /invocations HTTP/1.1" 200 12194 "-" "Go-http-client/1.1"[0m
[35m[2020-03-25:07:57:55:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-25:07:57:55:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-25:07:57:55:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [25/Mar/2020:07:57:58 +0000] "POST /invocations HTTP/1.1" 200 12162 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [25/Mar/2020:07:57:59 +0000] "POST /invocations HTTP/1.1" 200 12182 "-" "Go-http-client/1.1"[0m
[34m[2020-03-25:07:57:59:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [25/Mar/2020:07:57:59 +0000] "POST /invocations HTTP/1.1" 200 12181 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [25/Mar/2020:07:57:59 +0000] "POST /invocations HTTP/1.1" 200 12150 "-" "Go-http-client/1.1"[0m
[34m[2020-03-25:07:57:59:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-25:07:57:59:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-25:07:57:59:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [25/Mar/2020:07:57:58 +0000] "POST /invocations HTTP/1.1" 200 12162 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [25/Mar/2020:07:57:59 +0000] "POST /invocations HTTP/1.1" 200 12182 "-" "Go-http-client/1.1"[0m
[35m[2020-03-25:07:57:59:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [25/Mar/2020:07:57:59 +0000] "POST /invocations HTTP/1.1" 200 12181 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [25/Mar/2020:07:57:59 +0000] "POST /invocations HTTP/1.1" 200 12150 "-" "Go-http-client/1.1"[0m
[35m[2020-03-25:07:57:59:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-25:07:57:59:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-25:07:57:59:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [25/Mar/2020:07:58:02 +0000] "POST /invocations HTTP/1.1" 200 12161 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [25/Mar/2020:07:58:02 +0000] "POST /invocations HTTP/1.1" 200 12161 "-" "Go-http-client/1.1"[0m
[34m[2020-03-25:07:58:02:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [25/Mar/2020:07:58:02 +0000] "POST /invocations HTTP/1.1" 200 12205 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [25/Mar/2020:07:58:02 +0000] "POST /invocations HTTP/1.1" 200 12192 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [25/Mar/2020:07:58:02 +0000] "POST /invocations HTTP/1.1" 200 12147 "-" "Go-http-client/1.1"[0m
[34m[2020-03-25:07:58:02:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-25:07:58:02:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-25:07:58:03:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-25:07:58:02:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [25/Mar/2020:07:58:02 +0000] "POST /invocations HTTP/1.1" 200 12205 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [25/Mar/2020:07:58:02 +0000] "POST /invocations HTTP/1.1" 200 12192 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [25/Mar/2020:07:58:02 +0000] "POST /invocations HTTP/1.1" 200 12147 "-" "Go-http-client/1.1"[0m
[35m[2020-03-25:07:58:02:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-25:07:58:02:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-25:07:58:03:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-25:07:58:06:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-25:07:58:06:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [25/Mar/2020:07:58:09 +0000] "POST /invocations HTTP/1.1" 200 12198 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [25/Mar/2020:07:58:09 +0000] "POST /invocations HTTP/1.1" 200 12198 "-" "Go-http-client/1.1"[0m
[34m[2020-03-25:07:58:09:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [25/Mar/2020:07:58:09 +0000] "POST /invocations HTTP/1.1" 200 12189 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [25/Mar/2020:07:58:09 +0000] "POST /invocations HTTP/1.1" 200 12190 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [25/Mar/2020:07:58:09 +0000] "POST /invocations HTTP/1.1" 200 12210 "-" "Go-http-client/1.1"[0m
[34m[2020-03-25:07:58:09:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-25:07:58:10:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-25:07:58:10:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-25:07:58:09:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [25/Mar/2020:07:58:09 +0000] "POST /invocations HTTP/1.1" 200 12189 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [25/Mar/2020:07:58:09 +0000] "POST /invocations HTTP/1.1" 200 12190 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [25/Mar/2020:07:58:09 +0000] "POST /invocations HTTP/1.1" 200 12210 "-" "Go-http-client/1.1"[0m
[35m[2020-03-25:07:58:09:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-25:07:58:10:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-25:07:58:10:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-25:07:58:13:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-25:07:58:13:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [25/Mar/2020:07:58:16 +0000] "POST /invocations HTTP/1.1" 200 12225 "-" "Go-http-client/1.1"[0m
[34m[2020-03-25:07:58:16:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [25/Mar/2020:07:58:16 +0000] "POST /invocations HTTP/1.1" 200 12225 "-" "Go-http-client/1.1"[0m
[35m[2020-03-25:07:58:16:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [25/Mar/2020:07:58:16 +0000] "POST /invocations HTTP/1.1" 200 12189 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [25/Mar/2020:07:58:16 +0000] "POST /invocations HTTP/1.1" 200 12189 "-" "Go-http-client/1.1"[0m
[34m[2020-03-25:07:58:16:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [25/Mar/2020:07:58:16 +0000] "POST /invocations HTTP/1.1" 200 12179 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [25/Mar/2020:07:58:16 +0000] "POST /invocations HTTP/1.1" 200 12164 "-" "Go-http-client/1.1"[0m
[34m[2020-03-25:07:58:17:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-25:07:58:17:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-25:07:58:16:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [25/Mar/2020:07:58:16 +0000] "POST /invocations HTTP/1.1" 200 12179 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [25/Mar/2020:07:58:16 +0000] "POST /invocations HTTP/1.1" 200 12164 "-" "Go-http-client/1.1"[0m
[35m[2020-03-25:07:58:17:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-25:07:58:17:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [25/Mar/2020:07:58:19 +0000] "POST /invocations HTTP/1.1" 200 12179 "-" "Go-http-client/1.1"[0m
[34m[2020-03-25:07:58:20:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [25/Mar/2020:07:58:20 +0000] "POST /invocations HTTP/1.1" 200 12163 "-" "Go-http-client/1.1"[0m
[34m[2020-03-25:07:58:20:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [25/Mar/2020:07:58:20 +0000] "POST /invocations HTTP/1.1" 200 12216 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [25/Mar/2020:07:58:20 +0000] "POST /invocations HTTP/1.1" 200 12174 "-" "Go-http-client/1.1"[0m
[34m[2020-03-25:07:58:20:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [25/Mar/2020:07:58:19 +0000] "POST /invocations HTTP/1.1" 200 12179 "-" "Go-http-client/1.1"[0m
[35m[2020-03-25:07:58:20:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [25/Mar/2020:07:58:20 +0000] "POST /invocations HTTP/1.1" 200 12163 "-" "Go-http-client/1.1"[0m
[35m[2020-03-25:07:58:20:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [25/Mar/2020:07:58:20 +0000] "POST /invocations HTTP/1.1" 200 12216 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [25/Mar/2020:07:58:20 +0000] "POST /invocations HTTP/1.1" 200 12174 "-" "Go-http-client/1.1"[0m
[35m[2020-03-25:07:58:20:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-25:07:58:20:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-25:07:58:20:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [25/Mar/2020:07:58:23 +0000] "POST /invocations HTTP/1.1" 200 9118 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [25/Mar/2020:07:58:23 +0000] "POST /invocations HTTP/1.1" 200 12215 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [25/Mar/2020:07:58:23 +0000] "POST /invocations HTTP/1.1" 200 12207 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [25/Mar/2020:07:58:23 +0000] "POST /invocations HTTP/1.1" 200 9118 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [25/Mar/2020:07:58:23 +0000] "POST /invocations HTTP/1.1" 200 12215 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [25/Mar/2020:07:58:23 +0000] "POST /invocations HTTP/1.1" 200 12207 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [25/Mar/2020:07:58:23 +0000] "POST /invocations HTTP/1.1" 200 12240 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [25/Mar/2020:07:58:23 +0000] "POST /invocations HTTP/1.1" 200 12240 "-" "Go-http-client/1.1"[0m
```

Now the transform job has executed and the result, the estimated sentiment of each review, has been
saved on S3. Since we would rather work on this file locally we can perform a bit of notebook magic
to copy the file to the `data_dir`.

```python
!aws s3 cp --recursive $xgb_transformer.output_path $data_dir
```

```text
download: s3://sagemaker-us-west-2-171758673694/sentiment-analysis-xgboost-hyperparameter-tuning/batch-inference/test.csv.out
to ../data/sentiment-analysis-xgboost-hyperparameter-tuning/test.csv.out
```

The last step is now to read in the output from our model, convert the output to something a little
more usable, in this case we want the sentiment to be either `1` (positive) or `0` (negative), and
then compare to the ground truth labels.

```python
predictions = pd.read_csv(os.path.join(data_dir, 'test.csv.out'), header=None)
predictions = [round(num) for num in predictions.squeeze().values]
```

```python
from sklearn.metrics import accuracy_score
accuracy_score(test_y, predictions)
```

```text
0.84536
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

# Similarly we will remove the files in the cache_dir directory and the directory itself
!rm $cache_dir/*
!rmdir $cache_dir
```
