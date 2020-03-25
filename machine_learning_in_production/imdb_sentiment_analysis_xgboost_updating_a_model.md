# Sentiment Analysis

## Updating a Model in SageMaker

In this notebook we will consider a situation in which a model that we constructed is no longer
working as we intended. In particular, we will look at the XGBoost sentiment analysis model that we
constructed earlier. In this case, however, we have some new data that our model doesn't seem to
perform very well on. As a result, we will re-train our model and update an existing endpoint so
that it uses our new model.

This notebook starts by re-creating the XGBoost sentiment analysis model that was created in earlier
notebooks. This means that you will have already seen the cells up to the end of Step 4. The new
content in this notebook begins at Step 5.

## Step 1: Downloading the data

The dataset we are going to use is very popular among researchers in Natural Language Processing,
usually referred to as the [IMDb dataset](http://ai.stanford.edu/~amaas/data/sentiment/). It
consists of movie reviews from the website [imdb.com](http://www.imdb.com/), each labeled as either
'**pos**itive', if the reviewer enjoyed the film, or '**neg**ative' otherwise.

> Maas, Andrew L., et al. [Learning Word Vectors for Sentiment Analysis](http://ai.stanford.edu/~amaas/data/sentiment/).
> In _Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies_.
> Association for Computational Linguistics, 2011.

We begin by using some Jupyter Notebook magic to download and extract the dataset.

```python
%mkdir ../data
!wget -O ../data/aclImdb_v1.tar.gz http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
!tar -zxf ../data/aclImdb_v1.tar.gz -C ../data
```

```text
mkdir: cannot create directory ‘../data’: File exists
--2020-03-24 07:16:37--  http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
Resolving ai.stanford.edu (ai.stanford.edu)... 171.64.68.10
Connecting to ai.stanford.edu (ai.stanford.edu)|171.64.68.10|:80... connected.
HTTP request sent, awaiting response... 200 OK
Length: 84125825 (80M) [application/x-gzip]
Saving to: ‘../data/aclImdb_v1.tar.gz’

../data/aclImdb_v1. 100%[===================>]  80.23M  41.3MB/s    in 1.9s

2020-03-24 07:16:39 (41.3 MB/s) - ‘../data/aclImdb_v1.tar.gz’ saved [84125825/84125825]
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
'I wanted so much to enjoy this movie. It moved very slowly and was just boring. If it had been on TV, it would have lasted 15 to 20 minutes, maybe. What happened to the story? A great cast and photographer were working on a faulty foundation. If this is loosely based on the life of the director, why didn\'t he get someone to see that the writing itself was "loose". Then he directed it at a snail\'s pace which may have been the source of a few people nodding off during the movie. The music soars, but for a different film, not this one....for soap opera saga possibly. There were times when the dialogue was not understandable when Armin Meuller Stahl was speaking. I was not alone, because I heard a few rumblings about who said what to whom. Why can\'t Hollywood make better movies? This one had the nugget of a great story, but was just poorly executed.'
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
[nltk_data]   Package stopwords is already up-to-date!
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
review_to_words(train_X[100])[:10]
```

```text
['want', 'much', 'enjoy', 'movi', 'move', 'slowli', 'bore', 'tv', 'would', 'last']
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
Read preprocessed data from cache file: preprocessed_data.pkl
```

### Extract Bag-of-Words features

For the model we will be implementing, rather than using the reviews directly, we are going to
transform each review into a Bag-of-Words feature representation. Keep in mind that 'in the wild'
we will only have access to the training set so our transformer can only use the training set to
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
Read features from cache file: bow_features.pkl
```

```python
len(train_X[100])
```

```text
5000
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

# Earlier we shuffled the training dataset so to make things simple we can just assign
# the first 10 000 reviews to the validation set and use the remaining reviews for training.
val_X = pd.DataFrame(train_X[:10000])
train_X = pd.DataFrame(train_X[10000:])

val_y = pd.DataFrame(train_y[:10000])
train_y = pd.DataFrame(train_y[10000:])
```

The documentation for the XGBoost algorithm in SageMaker requires that the saved datasets should
contain no headers or index and that for the training and validation data, the label should occur
first for each sample.

For more information about this and other algorithms, the SageMaker developer documentation can be
found on __[Amazon's website.](https://docs.aws.amazon.com/sagemaker/latest/dg/)__

```python
# First we make sure that the local directory in which we'd like to store the training and validation csv files exists.
data_dir = '../data/sentiment_analysis_update_a_model'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
```

```python
pd.DataFrame(test_X).to_csv(os.path.join(data_dir, 'test.csv'), header=False, index=False)
pd.concat([val_y, val_X], axis=1).to_csv(os.path.join(data_dir, 'validation.csv'), header=False, index=False)
pd.concat([train_y, train_X], axis=1).to_csv(os.path.join(data_dir, 'train.csv'), header=False, index=False)
```

```python
# To save a bit of memory we can set text_X, train_X, val_X, train_y and val_y to None.
test_X = train_X = val_X = train_y = val_y = None
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
session. What this method does is upload the data to the default bucket (which is created if it does not exist)
into the path described by the key_prefix variable. To see this for yourself, once you have uploaded
the data files, go to the S3 console and look to see where the files have been uploaded.

For additional resources, see the __[SageMaker API documentation](http://sagemaker.readthedocs.io/en/latest/)__ and in addition the __[SageMaker Developer Guide.](https://docs.aws.amazon.com/sagemaker/latest/dg/)__

```python
import sagemaker

session = sagemaker.Session() # Store the current SageMaker session

# S3 prefix (which folder will we use)
prefix = 'sentiment-analysis-update-a-model'
test_location = session.upload_data(os.path.join(data_dir, 'test.csv'), key_prefix=prefix)
val_location = session.upload_data(os.path.join(data_dir, 'validation.csv'), key_prefix=prefix)
train_location = session.upload_data(os.path.join(data_dir, 'train.csv'), key_prefix=prefix)
```

### Creating the XGBoost model

Now that the data has been uploaded it is time to create the XGBoost model. To begin with, we need
to do some setup. At this point it is worth discussing what a model is in SageMaker. It is easiest
to think of a model of comprising three different objects in the SageMaker ecosystem, which interact
with one another.

- Model Artifacts
- Training Code (Container)
- Inference Code (Container)

The Model Artifacts are what you might think of as the actual model itself. For example, if you were
building a neural network, the model artifacts would be the weights of the various layers. In our
case, for an XGBoost model, the artifacts are the actual trees that are created during training.

The other two objects, the training code and the inference code are then used the manipulate the
training artifacts. More precisely, the training code uses the training data that is provided and
creates the model artifacts, while the inference code uses the model artifacts to make predictions
on new data.

The way that SageMaker runs the training and inference code is by making use of Docker containers. For now, think of a container as being a way of packaging code up so that dependencies aren't an issue.

```python
from sagemaker import get_execution_role
from sagemaker.amazon.amazon_estimator import get_image_uri

# Our current execution role is require when creating the model as the training
# and inference code will need to access the model artifacts.
role = get_execution_role()

# We need to retrieve the location of the container which is provided by Amazon for using XGBoost.
# As a matter of convenience, the training and inference code both use the same container.
container = get_image_uri(session.boto_region_name, 'xgboost', '0.90-1')

# First we create a SageMaker estimator object for our model.
xgb = sagemaker.estimator.Estimator(container, # The location of the container we wish to use
                                    role,                                    # What is our current IAM Role
                                    train_instance_count=1,                  # How many compute instances
                                    train_instance_type='ml.m4.xlarge',      # What kind of compute instances
                                    output_path='s3://{}/{}/'.format(session.default_bucket(), prefix),
                                    base_job_name='xgboost-training-job',
                                    sagemaker_session=session)

# And then set the algorithm specific parameters.
xgb.set_hyperparameters(max_depth=5,
                        eta=0.2,
                        gamma=4,
                        min_child_weight=6,
                        subsample=0.8,
                        silent=0,
                        objective='binary:logistic',
                        early_stopping_rounds=10,
                        num_round=500)
```

### Fit the XGBoost model

Now that our model has been set up we simply need to attach the training and validation datasets and
then ask SageMaker to set up the computation.

```python
s3_input_train = sagemaker.s3_input(s3_data=train_location, content_type='csv')
s3_input_validation = sagemaker.s3_input(s3_data=val_location, content_type='csv')

xgb.fit({
    'train': s3_input_train,
    'validation': s3_input_validation
})
```

```text
2020-03-24 07:29:53 Starting - Starting the training job...
2020-03-24 07:29:55 Starting - Launching requested ML instances......
2020-03-24 07:31:23 Starting - Preparing the instances for training......
2020-03-24 07:32:16 Downloading - Downloading input data...
2020-03-24 07:32:35 Training - Downloading the training image...
2020-03-24 07:33:07 Training - Training image download completed. Training in progress..[34mINFO:sagemaker-containers:Imported framework sagemaker_xgboost_container.training[0m
[34mINFO:sagemaker-containers:Failed to parse hyperparameter objective value binary:logistic to Json.[0m
[34mReturning the value itself[0m
[34mINFO:sagemaker-containers:No GPUs detected (normal if no gpus installed)[0m
[34mINFO:sagemaker_xgboost_container.training:Running XGBoost Sagemaker in algorithm mode[0m
[34mINFO:root:Determined delimiter of CSV input is ','[0m
[34mINFO:root:Determined delimiter of CSV input is ','[0m
[34mINFO:root:Determined delimiter of CSV input is ','[0m
[34m[07:33:11] 15000x5000 matrix with 75000000 entries loaded from /opt/ml/input/data/train?format=csv&label_column=0&delimiter=,[0m
[34mINFO:root:Determined delimiter of CSV input is ','[0m
[34m[07:33:12] 10000x5000 matrix with 50000000 entries loaded from /opt/ml/input/data/validation?format=csv&label_column=0&delimiter=,[0m
[34mINFO:root:Single node training.[0m
[34mINFO:root:Train matrix has 15000 rows[0m
[34mINFO:root:Validation matrix has 10000 rows[0m
[34m[0]#011train-error:0.296333#011validation-error:0.3024[0m
[34m[1]#011train-error:0.2774#011validation-error:0.2849[0m
[34m[2]#011train-error:0.2784#011validation-error:0.2859[0m
[34m[3]#011train-error:0.266667#011validation-error:0.2775[0m
[34m[4]#011train-error:0.252#011validation-error:0.266[0m
[34m[5]#011train-error:0.255733#011validation-error:0.2688[0m
[34m[6]#011train-error:0.2476#011validation-error:0.2614[0m
[34m[7]#011train-error:0.2434#011validation-error:0.2595[0m
[34m[8]#011train-error:0.2328#011validation-error:0.2506[0m
[34m[9]#011train-error:0.2222#011validation-error:0.2419[0m
[34m[10]#011train-error:0.217867#011validation-error:0.238[0m
[34m[11]#011train-error:0.215067#011validation-error:0.2363[0m
[34m[12]#011train-error:0.2126#011validation-error:0.233[0m
[34m[13]#011train-error:0.207733#011validation-error:0.2307[0m
[34m[14]#011train-error:0.203733#011validation-error:0.228[0m
[34m[15]#011train-error:0.199467#011validation-error:0.2225[0m
[34m[16]#011train-error:0.195267#011validation-error:0.2178[0m
[34m[17]#011train-error:0.191867#011validation-error:0.2157[0m
[34m[18]#011train-error:0.189#011validation-error:0.2109[0m
[34m[19]#011train-error:0.186467#011validation-error:0.2099[0m
[34m[20]#011train-error:0.1832#011validation-error:0.2071[0m
[34m[21]#011train-error:0.1818#011validation-error:0.204[0m
[34m[22]#011train-error:0.1794#011validation-error:0.2014[0m
[34m[23]#011train-error:0.176733#011validation-error:0.1983[0m
[34m[24]#011train-error:0.1744#011validation-error:0.198[0m
[34m[25]#011train-error:0.172267#011validation-error:0.1971[0m
[34m[26]#011train-error:0.171#011validation-error:0.1963[0m
[34m[27]#011train-error:0.1698#011validation-error:0.1945[0m
[34m[28]#011train-error:0.167267#011validation-error:0.1939[0m
[34m[29]#011train-error:0.1656#011validation-error:0.1944[0m
[34m[30]#011train-error:0.1634#011validation-error:0.1918[0m
[34m[31]#011train-error:0.1614#011validation-error:0.1899[0m
[34m[32]#011train-error:0.160733#011validation-error:0.1888[0m
[34m[33]#011train-error:0.1602#011validation-error:0.1877[0m
[34m[34]#011train-error:0.158333#011validation-error:0.1865[0m
[34m[35]#011train-error:0.1566#011validation-error:0.1843[0m
[34m[36]#011train-error:0.1548#011validation-error:0.1836[0m
[34m[37]#011train-error:0.153133#011validation-error:0.1816[0m
[34m[38]#011train-error:0.152333#011validation-error:0.183[0m
[34m[39]#011train-error:0.151733#011validation-error:0.1807[0m
[34m[40]#011train-error:0.1514#011validation-error:0.1803[0m
[34m[41]#011train-error:0.1502#011validation-error:0.1789[0m
[34m[42]#011train-error:0.148467#011validation-error:0.1779[0m
[34m[43]#011train-error:0.146867#011validation-error:0.1767[0m
[34m[44]#011train-error:0.1452#011validation-error:0.1757[0m
[34m[45]#011train-error:0.143533#011validation-error:0.1745[0m
[34m[46]#011train-error:0.1422#011validation-error:0.1744[0m
[34m[47]#011train-error:0.141933#011validation-error:0.1729[0m
[34m[48]#011train-error:0.1402#011validation-error:0.1721[0m
[34m[49]#011train-error:0.139867#011validation-error:0.1711[0m
[34m[50]#011train-error:0.139667#011validation-error:0.1695[0m
[34m[51]#011train-error:0.138867#011validation-error:0.1697[0m
[34m[52]#011train-error:0.1388#011validation-error:0.1688[0m
[34m[53]#011train-error:0.1378#011validation-error:0.1676[0m
[34m[54]#011train-error:0.1368#011validation-error:0.1662[0m
[34m[55]#011train-error:0.1364#011validation-error:0.1657[0m
[34m[56]#011train-error:0.134133#011validation-error:0.1651[0m
[34m[57]#011train-error:0.132733#011validation-error:0.1655[0m
[34m[58]#011train-error:0.1326#011validation-error:0.1651[0m
[34m[59]#011train-error:0.131333#011validation-error:0.1653[0m
[34m[60]#011train-error:0.129933#011validation-error:0.1654[0m
[34m[61]#011train-error:0.128133#011validation-error:0.1654[0m
[34m[62]#011train-error:0.127467#011validation-error:0.1639[0m
[34m[63]#011train-error:0.1264#011validation-error:0.1649[0m
[34m[64]#011train-error:0.1254#011validation-error:0.164[0m
[34m[65]#011train-error:0.1252#011validation-error:0.1631[0m
[34m[66]#011train-error:0.125533#011validation-error:0.1619[0m
[34m[67]#011train-error:0.125667#011validation-error:0.1615[0m
[34m[68]#011train-error:0.125067#011validation-error:0.1618[0m
[34m[69]#011train-error:0.1236#011validation-error:0.1614[0m
[34m[70]#011train-error:0.122933#011validation-error:0.1601[0m
[34m[71]#011train-error:0.1224#011validation-error:0.1597[0m
[34m[72]#011train-error:0.1218#011validation-error:0.1595[0m
[34m[73]#011train-error:0.121067#011validation-error:0.1588[0m
[34m[74]#011train-error:0.120667#011validation-error:0.1592[0m
[34m[75]#011train-error:0.120333#011validation-error:0.1586[0m
[34m[76]#011train-error:0.1198#011validation-error:0.1575[0m
[34m[77]#011train-error:0.118733#011validation-error:0.1563[0m
[34m[78]#011train-error:0.118267#011validation-error:0.1556[0m
[34m[79]#011train-error:0.117533#011validation-error:0.1549[0m
[34m[80]#011train-error:0.1166#011validation-error:0.1543[0m
[34m[81]#011train-error:0.1158#011validation-error:0.1535[0m
[34m[82]#011train-error:0.115933#011validation-error:0.153[0m
[34m[83]#011train-error:0.115267#011validation-error:0.153[0m
[34m[84]#011train-error:0.114667#011validation-error:0.1527[0m
[34m[85]#011train-error:0.113933#011validation-error:0.1519[0m
[34m[86]#011train-error:0.113467#011validation-error:0.1515[0m
[34m[87]#011train-error:0.112933#011validation-error:0.1512[0m
[34m[88]#011train-error:0.112133#011validation-error:0.1501[0m
[34m[89]#011train-error:0.111667#011validation-error:0.1499[0m
[34m[90]#011train-error:0.112267#011validation-error:0.1502[0m
[34m[91]#011train-error:0.111133#011validation-error:0.1502[0m
[34m[92]#011train-error:0.111267#011validation-error:0.1494[0m
[34m[93]#011train-error:0.110733#011validation-error:0.1492[0m
[34m[94]#011train-error:0.111#011validation-error:0.1483[0m
[34m[95]#011train-error:0.110267#011validation-error:0.1469[0m
[34m[96]#011train-error:0.109867#011validation-error:0.1468[0m
[34m[97]#011train-error:0.109267#011validation-error:0.1468[0m
[34m[98]#011train-error:0.1086#011validation-error:0.1463[0m
[34m[99]#011train-error:0.1078#011validation-error:0.1461[0m
[34m[100]#011train-error:0.107667#011validation-error:0.1461[0m
[34m[101]#011train-error:0.108467#011validation-error:0.1465[0m
[34m[102]#011train-error:0.107467#011validation-error:0.1466[0m
[34m[103]#011train-error:0.107333#011validation-error:0.1464[0m
[34m[104]#011train-error:0.107133#011validation-error:0.1449[0m
[34m[105]#011train-error:0.106733#011validation-error:0.1442[0m
[34m[106]#011train-error:0.106333#011validation-error:0.1446[0m
[34m[107]#011train-error:0.105733#011validation-error:0.144[0m
[34m[108]#011train-error:0.105267#011validation-error:0.1442[0m
[34m[109]#011train-error:0.104267#011validation-error:0.1449[0m
[34m[110]#011train-error:0.104733#011validation-error:0.145[0m
[34m[111]#011train-error:0.1038#011validation-error:0.1442[0m
[34m[112]#011train-error:0.1032#011validation-error:0.143[0m
[34m[113]#011train-error:0.1028#011validation-error:0.1421[0m
[34m[114]#011train-error:0.103133#011validation-error:0.1419[0m
[34m[115]#011train-error:0.102067#011validation-error:0.1425[0m
[34m[116]#011train-error:0.101867#011validation-error:0.1428[0m
[34m[117]#011train-error:0.1016#011validation-error:0.142[0m
[34m[118]#011train-error:0.100133#011validation-error:0.1414[0m
[34m[119]#011train-error:0.100067#011validation-error:0.1416[0m
[34m[120]#011train-error:0.0998#011validation-error:0.1412[0m
[34m[121]#011train-error:0.099667#011validation-error:0.1412[0m
[34m[122]#011train-error:0.099133#011validation-error:0.14[0m
[34m[123]#011train-error:0.0988#011validation-error:0.1392[0m
[34m[124]#011train-error:0.099067#011validation-error:0.1396[0m
[34m[125]#011train-error:0.0978#011validation-error:0.1404[0m
[34m[126]#011train-error:0.0972#011validation-error:0.1399[0m
[34m[127]#011train-error:0.0964#011validation-error:0.1397[0m
[34m[128]#011train-error:0.096133#011validation-error:0.1396[0m
[34m[129]#011train-error:0.0958#011validation-error:0.1391[0m
[34m[130]#011train-error:0.095933#011validation-error:0.1387[0m
[34m[131]#011train-error:0.096133#011validation-error:0.1384[0m
[34m[132]#011train-error:0.0954#011validation-error:0.1387[0m
[34m[133]#011train-error:0.095867#011validation-error:0.1387[0m
[34m[134]#011train-error:0.095133#011validation-error:0.1386[0m
[34m[135]#011train-error:0.0948#011validation-error:0.1388[0m
[34m[136]#011train-error:0.095333#011validation-error:0.139[0m
[34m[137]#011train-error:0.094867#011validation-error:0.139[0m
[34m[138]#011train-error:0.0948#011validation-error:0.1388[0m
[34m[139]#011train-error:0.094067#011validation-error:0.1378[0m
[34m[140]#011train-error:0.093733#011validation-error:0.1372[0m
[34m[141]#011train-error:0.093867#011validation-error:0.1375[0m
[34m[142]#011train-error:0.093667#011validation-error:0.1374[0m
[34m[143]#011train-error:0.0932#011validation-error:0.1377[0m
[34m[144]#011train-error:0.0934#011validation-error:0.1379[0m
[34m[145]#011train-error:0.093333#011validation-error:0.1374[0m
[34m[146]#011train-error:0.093067#011validation-error:0.1365[0m
[34m[147]#011train-error:0.092667#011validation-error:0.1374[0m
[34m[148]#011train-error:0.092533#011validation-error:0.1372[0m
[34m[149]#011train-error:0.092#011validation-error:0.1369[0m
[34m[150]#011train-error:0.091667#011validation-error:0.138[0m
[34m[151]#011train-error:0.091533#011validation-error:0.1378[0m
[34m[152]#011train-error:0.0906#011validation-error:0.1374[0m
[34m[153]#011train-error:0.09#011validation-error:0.1366[0m
[34m[154]#011train-error:0.089867#011validation-error:0.1368[0m
[34m[155]#011train-error:0.0898#011validation-error:0.1371[0m
[34m[156]#011train-error:0.0896#011validation-error:0.1372[0m

2020-03-24 07:36:18 Uploading - Uploading generated training model
2020-03-24 07:36:18 Completed - Training job completed
Training seconds: 242
Billable seconds: 242
```

### Testing the model

Now that we've fit our XGBoost model, it's time to see how well it performs. To do this we will use
SageMakers Batch Transform functionality. Batch Transform is a convenient way to perform inference
on a large dataset in a way that is not realtime. That is, we don't necessarily need to use our
model's results immediately and instead we can peform inference on a large number of samples. An
example of this in industry might be peforming an end of month report. This method of inference can
also be useful to us as it means to can perform inference on our entire test set.

To perform a Batch Transformation we need to first create a transformer objects from our trained
estimator object.

```python
batch_output = 's3://{}/{}/batch-inference'.format(session.default_bucket(), prefix)
xgb_transformer = xgb.transformer(instance_count = 1, instance_type = 'ml.m4.xlarge', output_path=batch_output)
```

Next we actually perform the transform job. When doing so we need to make sure to specify the type
of data we are sending so that it is serialized correctly in the background. In our case we are
providing our model with csv data so we specify `text/csv`. Also, if the test data that we have
provided is too large to process all at once then we need to specify how the data file should be
split up. Since each line is a single entry in our data set we tell SageMaker that it can split the
input on each line.

```python
xgb_transformer.transform(test_location, content_type='text/csv', split_type='Line')
```

Currently the transform job is running but it is doing so in the background. Since we wish to wait
until the transform job is done and we would like a bit of feedback we can run the `wait()` method.

```python
xgb_transformer.wait()
```

```text
..........................
[34m[2020-03-24 07:40:59 +0000] [18] [INFO] Starting gunicorn 19.10.0[0m
[34m[2020-03-24 07:40:59 +0000] [18] [INFO] Listening at: unix:/tmp/gunicorn.sock (18)[0m
[34m[2020-03-24 07:40:59 +0000] [18] [INFO] Using worker: gevent[0m
[34m[2020-03-24 07:40:59 +0000] [25] [INFO] Booting worker with pid: 25[0m
[34m[2020-03-24 07:40:59 +0000] [26] [INFO] Booting worker with pid: 26[0m
[34m[2020-03-24 07:41:00 +0000] [30] [INFO] Booting worker with pid: 30[0m
[34m[2020-03-24 07:41:00 +0000] [31] [INFO] Booting worker with pid: 31[0m
[34m[2020-03-24:07:41:22:INFO] No GPUs detected (normal if no gpus installed)[0m
[35m[2020-03-24:07:41:22:INFO] No GPUs detected (normal if no gpus installed)[0m
[34m169.254.255.130 - - [24/Mar/2020:07:41:22 +0000] "GET /ping HTTP/1.1" 200 0 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:07:41:22 +0000] "GET /execution-parameters HTTP/1.1" 200 84 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:07:41:22 +0000] "GET /ping HTTP/1.1" 200 0 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:07:41:22 +0000] "GET /execution-parameters HTTP/1.1" 200 84 "-" "Go-http-client/1.1"[0m
[32m2020-03-24T07:41:22.833:[sagemaker logs]: MaxConcurrentTransforms=4, MaxPayloadInMB=6, BatchStrategy=MULTI_RECORD[0m
[34m[2020-03-24:07:41:25:INFO] No GPUs detected (normal if no gpus installed)[0m
[34m[2020-03-24:07:41:25:INFO] No GPUs detected (normal if no gpus installed)[0m
[34m[2020-03-24:07:41:25:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:07:41:25:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:07:41:25:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:07:41:25:INFO] No GPUs detected (normal if no gpus installed)[0m
[34m[2020-03-24:07:41:25:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:07:41:25:INFO] No GPUs detected (normal if no gpus installed)[0m
[35m[2020-03-24:07:41:25:INFO] No GPUs detected (normal if no gpus installed)[0m
[35m[2020-03-24:07:41:25:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:07:41:25:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:07:41:25:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:07:41:25:INFO] No GPUs detected (normal if no gpus installed)[0m
[35m[2020-03-24:07:41:25:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:07:41:28 +0000] "POST /invocations HTTP/1.1" 200 12218 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:07:41:29 +0000] "POST /invocations HTTP/1.1" 200 12182 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:07:41:29 +0000] "POST /invocations HTTP/1.1" 200 12192 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:07:41:29 +0000] "POST /invocations HTTP/1.1" 200 12219 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:07:41:28 +0000] "POST /invocations HTTP/1.1" 200 12218 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:07:41:29 +0000] "POST /invocations HTTP/1.1" 200 12182 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:07:41:29 +0000] "POST /invocations HTTP/1.1" 200 12192 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:07:41:29 +0000] "POST /invocations HTTP/1.1" 200 12219 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:07:41:29:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:07:41:29:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:07:41:29:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:07:41:29:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:07:41:29:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:07:41:29:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:07:41:29:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:07:41:29:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:07:41:32 +0000] "POST /invocations HTTP/1.1" 200 12171 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:07:41:32 +0000] "POST /invocations HTTP/1.1" 200 12187 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:07:41:32 +0000] "POST /invocations HTTP/1.1" 200 12180 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:07:41:32:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:07:41:32 +0000] "POST /invocations HTTP/1.1" 200 12195 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:07:41:32:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:07:41:33:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:07:41:33:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [24/Mar/2020:07:41:32 +0000] "POST /invocations HTTP/1.1" 200 12171 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:07:41:32 +0000] "POST /invocations HTTP/1.1" 200 12187 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:07:41:32 +0000] "POST /invocations HTTP/1.1" 200 12180 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:07:41:32:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [24/Mar/2020:07:41:32 +0000] "POST /invocations HTTP/1.1" 200 12195 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:07:41:32:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:07:41:33:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:07:41:33:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:07:41:39 +0000] "POST /invocations HTTP/1.1" 200 12223 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:07:41:39 +0000] "POST /invocations HTTP/1.1" 200 12222 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:07:41:39 +0000] "POST /invocations HTTP/1.1" 200 12199 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:07:41:40:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:07:41:40 +0000] "POST /invocations HTTP/1.1" 200 12225 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:07:41:40:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:07:41:40:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [24/Mar/2020:07:41:39 +0000] "POST /invocations HTTP/1.1" 200 12223 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:07:41:39 +0000] "POST /invocations HTTP/1.1" 200 12222 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:07:41:39 +0000] "POST /invocations HTTP/1.1" 200 12199 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:07:41:40:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [24/Mar/2020:07:41:40 +0000] "POST /invocations HTTP/1.1" 200 12225 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:07:41:40:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:07:41:40:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:07:41:40:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:07:41:40:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:07:41:43 +0000] "POST /invocations HTTP/1.1" 200 12182 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:07:41:43 +0000] "POST /invocations HTTP/1.1" 200 12184 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:07:41:43:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:07:41:43:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:07:41:43 +0000] "POST /invocations HTTP/1.1" 200 12196 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:07:41:43:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:07:41:43:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [24/Mar/2020:07:41:43 +0000] "POST /invocations HTTP/1.1" 200 12182 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:07:41:43 +0000] "POST /invocations HTTP/1.1" 200 12184 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:07:41:43:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:07:41:43:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [24/Mar/2020:07:41:43 +0000] "POST /invocations HTTP/1.1" 200 12196 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:07:41:43:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:07:41:43:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:07:41:46 +0000] "POST /invocations HTTP/1.1" 200 12191 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:07:41:46 +0000] "POST /invocations HTTP/1.1" 200 12154 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:07:41:47:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:07:41:47 +0000] "POST /invocations HTTP/1.1" 200 12226 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:07:41:47:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:07:41:47 +0000] "POST /invocations HTTP/1.1" 200 12205 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:07:41:46 +0000] "POST /invocations HTTP/1.1" 200 12191 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:07:41:46 +0000] "POST /invocations HTTP/1.1" 200 12154 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:07:41:47:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [24/Mar/2020:07:41:47 +0000] "POST /invocations HTTP/1.1" 200 12226 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:07:41:47:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [24/Mar/2020:07:41:47 +0000] "POST /invocations HTTP/1.1" 200 12205 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:07:41:47:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:07:41:47:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:07:41:47:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:07:41:47:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:07:41:50:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:07:41:50 +0000] "POST /invocations HTTP/1.1" 200 12173 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:07:41:50:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:07:41:50 +0000] "POST /invocations HTTP/1.1" 200 12183 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:07:41:50:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:07:41:50:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [24/Mar/2020:07:41:50 +0000] "POST /invocations HTTP/1.1" 200 12173 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:07:41:50:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [24/Mar/2020:07:41:50 +0000] "POST /invocations HTTP/1.1" 200 12183 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:07:41:50:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:07:41:51:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:07:41:51:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:07:41:53 +0000] "POST /invocations HTTP/1.1" 200 12146 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:07:41:53 +0000] "POST /invocations HTTP/1.1" 200 12146 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:07:41:54 +0000] "POST /invocations HTTP/1.1" 200 12211 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:07:41:54:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [24/Mar/2020:07:41:54 +0000] "POST /invocations HTTP/1.1" 200 12211 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:07:41:54:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:07:41:54 +0000] "POST /invocations HTTP/1.1" 200 12200 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:07:41:54:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:07:41:54 +0000] "POST /invocations HTTP/1.1" 200 12193 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:07:41:54:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:07:41:54:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [24/Mar/2020:07:41:54 +0000] "POST /invocations HTTP/1.1" 200 12200 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:07:41:54:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [24/Mar/2020:07:41:54 +0000] "POST /invocations HTTP/1.1" 200 12193 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:07:41:54:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:07:41:54:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:07:41:57 +0000] "POST /invocations HTTP/1.1" 200 12212 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:07:41:57 +0000] "POST /invocations HTTP/1.1" 200 12172 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:07:41:57:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:07:41:57 +0000] "POST /invocations HTTP/1.1" 200 12211 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:07:41:57:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:07:41:57 +0000] "POST /invocations HTTP/1.1" 200 12184 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:07:41:57:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:07:41:58:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [24/Mar/2020:07:41:57 +0000] "POST /invocations HTTP/1.1" 200 12212 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:07:41:57 +0000] "POST /invocations HTTP/1.1" 200 12172 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:07:41:57:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [24/Mar/2020:07:41:57 +0000] "POST /invocations HTTP/1.1" 200 12211 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:07:41:57:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [24/Mar/2020:07:41:57 +0000] "POST /invocations HTTP/1.1" 200 12184 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:07:41:57:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:07:41:58:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:07:42:00 +0000] "POST /invocations HTTP/1.1" 200 9105 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:07:42:00 +0000] "POST /invocations HTTP/1.1" 200 12211 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:07:42:01 +0000] "POST /invocations HTTP/1.1" 200 12194 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:07:42:01 +0000] "POST /invocations HTTP/1.1" 200 12225 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:07:42:00 +0000] "POST /invocations HTTP/1.1" 200 9105 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:07:42:00 +0000] "POST /invocations HTTP/1.1" 200 12211 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:07:42:01 +0000] "POST /invocations HTTP/1.1" 200 12194 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:07:42:01 +0000] "POST /invocations HTTP/1.1" 200 12225 "-" "Go-http-client/1.1"[0m
```

Now the transform job has executed and the result, the estimated sentiment of each review, has been
saved on S3. Since we would rather work on this file locally we can perform a bit of notebook magic
to copy the file to the `data_dir`.

```python
!aws s3 cp --recursive $xgb_transformer.output_path $data_dir
```

```text
download: s3://sagemaker-us-west-2-171758673694/sentiment-analysis-update-a-model/batch-inference/test.csv.out to
../data/sentiment_analysis_update_a_model/test.csv.out
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
0.86032
```

## Step 5: Looking at New Data

So now we have an XGBoost sentiment analysis model that we believe is working pretty well. As a
result, we deployed it and we are using it in some sort of app.

However, as we allow users to use our app we periodically record submitted movie reviews so that we
can perform some quality control on our deployed model. Once we've accumulated enough reviews we go
through them by hand and evaluate whether they are positive or negative (there are many ways you
might do this in practice aside from by hand). The reason for doing this is so that we can check to
see how well our model is doing.

```python
import new_data

new_X, new_Y = new_data.get_new_data()
```

**NOTE:** Part of the fun in this notebook is trying to figure out what exactly is happening with
the new data, so try not to cheat by looking in the `new_data` module. Also, the `new_data` module
assumes that the cache created earlier in Step 3 is still stored in `../cache/sentiment_analysis`.

### (TODO) Testing the current model

Now that we've loaded the new data, let's check to see how our current XGBoost model performs on it.

First, note that the data that has been loaded has already been pre-processed so that each entry in
`new_X` is a list of words that have been processed using `nltk`. However, we have not yet
constructed the bag of words encoding, which we will do now.

First, we use the vocabulary that we constructed earlier using the original training data to
construct a `CountVectorizer` which we will use to transform our new data into its bag of words
encoding.

**TODO:** Create the CountVectorizer object using the vocabulary created earlier and use it to
transform the new data.

```python
# TODO: Create the CountVectorizer using the previously constructed vocabulary
vectorizer = CountVectorizer(vocabulary=vocabulary,
                             preprocessor=lambda x: x,
                             tokenizer=lambda x: x)

# TODO: Transform our new data set and store the transformed data in the variable new_XV
new_XV = vectorizer.fit_transform(new_X).toarray()
```

As a quick sanity check, we make sure that the length of each of our bag of words encoded reviews is
correct. In particular, it must be the same size as the vocabulary which in our case is `5000`.

```python
len(new_XV[100])
```

```text
5000
```

Now that we've performed the data processing that is required by our model we can save it locally
and then upload it to S3 so that we can construct a batch transform job in order to see how well our
model is working.

First, we save the data locally.

**TODO:** Save the new data (after it has been transformed using the original vocabulary) to the
local notebook instance.

```python
# TODO: Save the data contained in new_XV locally in the data_dir with the file name new_data.csv
pd.DataFrame(new_XV).to_csv(os.path.join(data_dir, 'new_data.csv'), header=False, index=False)
```

Next, we upload the data to S3.

**TODO:** Upload the csv file created above to S3.

```python
# TODO: Upload the new_data.csv file contained in the data_dir folder to S3 and save the resulting
#       URI as new_data_location
new_data_location = session.upload_data(os.path.join(data_dir, 'new_data.csv'), key_prefix=prefix)
```

Then, once the new data has been uploaded to S3, we create and run the batch transform job to get
our model's predictions about the sentiment of the new movie reviews.

**TODO:** Using the `xgb_transformer` object that was created earlier (at the end of Step 4 to test
the XGBoost model), transform the data located at `new_data_location`.

```python
# TODO: Using xgb_transformer, transform the new_data_location data. You may wish to **wait** until
#       the batch transform job has finished.
xgb_transformer.transform(new_data_location, content_type='text/csv', split_type='Line')
xgb_transformer.wait()
```

```text
.......................
[34m[2020-03-24 07:49:21 +0000] [15] [INFO] Starting gunicorn 19.10.0[0m
[34m[2020-03-24 07:49:21 +0000] [15] [INFO] Listening at: unix:/tmp/gunicorn.sock (15)[0m
[34m[2020-03-24 07:49:21 +0000] [15] [INFO] Using worker: gevent[0m
[34m[2020-03-24 07:49:21 +0000] [22] [INFO] Booting worker with pid: 22[0m
[34m[2020-03-24 07:49:21 +0000] [23] [INFO] Booting worker with pid: 23[0m
[34m[2020-03-24 07:49:21 +0000] [27] [INFO] Booting worker with pid: 27[0m
[34m[2020-03-24 07:49:21 +0000] [28] [INFO] Booting worker with pid: 28[0m
[34m[2020-03-24:07:49:42:INFO] No GPUs detected (normal if no gpus installed)[0m
[34m169.254.255.130 - - [24/Mar/2020:07:49:42 +0000] "GET /ping HTTP/1.1" 200 0 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:07:49:42:INFO] No GPUs detected (normal if no gpus installed)[0m
[35m169.254.255.130 - - [24/Mar/2020:07:49:42 +0000] "GET /ping HTTP/1.1" 200 0 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:07:49:42:INFO] No GPUs detected (normal if no gpus installed)[0m
[34m169.254.255.130 - - [24/Mar/2020:07:49:42 +0000] "GET /execution-parameters HTTP/1.1" 200 84 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:07:49:42:INFO] No GPUs detected (normal if no gpus installed)[0m
[35m169.254.255.130 - - [24/Mar/2020:07:49:42 +0000] "GET /execution-parameters HTTP/1.1" 200 84 "-" "Go-http-client/1.1"[0m
[32m2020-03-24T07:49:42.330:[sagemaker logs]: MaxConcurrentTransforms=4, MaxPayloadInMB=6, BatchStrategy=MULTI_RECORD[0m
[34m[2020-03-24:07:49:46:INFO] No GPUs detected (normal if no gpus installed)[0m
[34m[2020-03-24:07:49:46:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:07:49:46:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:07:49:46:INFO] No GPUs detected (normal if no gpus installed)[0m
[34m[2020-03-24:07:49:46:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:07:49:46:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:07:49:46:INFO] No GPUs detected (normal if no gpus installed)[0m
[35m[2020-03-24:07:49:46:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:07:49:46:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:07:49:46:INFO] No GPUs detected (normal if no gpus installed)[0m
[35m[2020-03-24:07:49:46:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:07:49:46:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:07:49:50:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:07:49:50:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:07:49:50:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:07:49:50:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:07:49:53 +0000] "POST /invocations HTTP/1.1" 200 12198 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:07:49:53 +0000] "POST /invocations HTTP/1.1" 200 12226 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:07:49:53:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:07:49:53 +0000] "POST /invocations HTTP/1.1" 200 12234 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:07:49:53 +0000] "POST /invocations HTTP/1.1" 200 12203 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:07:49:53:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [24/Mar/2020:07:49:53 +0000] "POST /invocations HTTP/1.1" 200 12198 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:07:49:53 +0000] "POST /invocations HTTP/1.1" 200 12226 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:07:49:53:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [24/Mar/2020:07:49:53 +0000] "POST /invocations HTTP/1.1" 200 12234 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:07:49:53 +0000] "POST /invocations HTTP/1.1" 200 12203 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:07:49:53:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:07:49:53:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:07:49:53:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:07:49:53:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:07:49:53:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:07:49:56 +0000] "POST /invocations HTTP/1.1" 200 12201 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:07:49:56 +0000] "POST /invocations HTTP/1.1" 200 12213 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:07:49:57 +0000] "POST /invocations HTTP/1.1" 200 12201 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:07:49:57:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:07:49:57:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:07:49:57 +0000] "POST /invocations HTTP/1.1" 200 12239 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:07:49:57:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [24/Mar/2020:07:49:56 +0000] "POST /invocations HTTP/1.1" 200 12201 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:07:49:56 +0000] "POST /invocations HTTP/1.1" 200 12213 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:07:49:57 +0000] "POST /invocations HTTP/1.1" 200 12201 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:07:49:57:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:07:49:57:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [24/Mar/2020:07:49:57 +0000] "POST /invocations HTTP/1.1" 200 12239 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:07:49:57:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:07:49:57:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:07:49:57:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:07:50:00 +0000] "POST /invocations HTTP/1.1" 200 12201 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:07:50:00 +0000] "POST /invocations HTTP/1.1" 200 12177 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:07:50:00 +0000] "POST /invocations HTTP/1.1" 200 12183 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:07:50:00 +0000] "POST /invocations HTTP/1.1" 200 12201 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:07:50:00 +0000] "POST /invocations HTTP/1.1" 200 12177 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:07:50:00 +0000] "POST /invocations HTTP/1.1" 200 12183 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:07:50:00:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:07:50:00 +0000] "POST /invocations HTTP/1.1" 200 12214 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:07:50:00:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:07:50:00:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:07:50:00:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:07:50:00:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [24/Mar/2020:07:50:00 +0000] "POST /invocations HTTP/1.1" 200 12214 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:07:50:00:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:07:50:00:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:07:50:00:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:07:50:04 +0000] "POST /invocations HTTP/1.1" 200 12193 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:07:50:04:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:07:50:04:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [24/Mar/2020:07:50:04 +0000] "POST /invocations HTTP/1.1" 200 12193 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:07:50:04:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:07:50:04:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:07:50:07 +0000] "POST /invocations HTTP/1.1" 200 12211 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:07:50:07 +0000] "POST /invocations HTTP/1.1" 200 12240 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:07:50:07 +0000] "POST /invocations HTTP/1.1" 200 12211 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:07:50:07 +0000] "POST /invocations HTTP/1.1" 200 12240 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:07:50:07 +0000] "POST /invocations HTTP/1.1" 200 12220 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:07:50:07 +0000] "POST /invocations HTTP/1.1" 200 12185 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:07:50:07 +0000] "POST /invocations HTTP/1.1" 200 12220 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:07:50:07 +0000] "POST /invocations HTTP/1.1" 200 12185 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:07:50:07:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:07:50:07:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:07:50:07:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:07:50:08:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:07:50:07:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:07:50:07:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:07:50:07:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:07:50:08:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:07:50:11 +0000] "POST /invocations HTTP/1.1" 200 12187 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:07:50:11:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:07:50:11:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:07:50:11:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [24/Mar/2020:07:50:11 +0000] "POST /invocations HTTP/1.1" 200 12187 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:07:50:11:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:07:50:11:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:07:50:11:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:07:50:14 +0000] "POST /invocations HTTP/1.1" 200 12202 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:07:50:14 +0000] "POST /invocations HTTP/1.1" 200 12202 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:07:50:14 +0000] "POST /invocations HTTP/1.1" 200 12240 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:07:50:14:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:07:50:14 +0000] "POST /invocations HTTP/1.1" 200 12193 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:07:50:14 +0000] "POST /invocations HTTP/1.1" 200 12195 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:07:50:14:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:07:50:15:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:07:50:15:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [24/Mar/2020:07:50:14 +0000] "POST /invocations HTTP/1.1" 200 12240 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:07:50:14:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [24/Mar/2020:07:50:14 +0000] "POST /invocations HTTP/1.1" 200 12193 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:07:50:14 +0000] "POST /invocations HTTP/1.1" 200 12195 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:07:50:14:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:07:50:15:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:07:50:15:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:07:50:18 +0000] "POST /invocations HTTP/1.1" 200 12199 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:07:50:18 +0000] "POST /invocations HTTP/1.1" 200 12214 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:07:50:18:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:07:50:18 +0000] "POST /invocations HTTP/1.1" 200 12189 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:07:50:18:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:07:50:18 +0000] "POST /invocations HTTP/1.1" 200 12190 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:07:50:18:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:07:50:18:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [24/Mar/2020:07:50:18 +0000] "POST /invocations HTTP/1.1" 200 12199 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:07:50:18 +0000] "POST /invocations HTTP/1.1" 200 12214 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:07:50:18:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [24/Mar/2020:07:50:18 +0000] "POST /invocations HTTP/1.1" 200 12189 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:07:50:18:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [24/Mar/2020:07:50:18 +0000] "POST /invocations HTTP/1.1" 200 12190 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:07:50:18:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:07:50:18:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:07:50:21 +0000] "POST /invocations HTTP/1.1" 200 9111 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:07:50:21 +0000] "POST /invocations HTTP/1.1" 200 12222 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:07:50:21 +0000] "POST /invocations HTTP/1.1" 200 12180 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:07:50:21 +0000] "POST /invocations HTTP/1.1" 200 9111 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:07:50:21 +0000] "POST /invocations HTTP/1.1" 200 12222 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:07:50:21 +0000] "POST /invocations HTTP/1.1" 200 12180 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:07:50:21 +0000] "POST /invocations HTTP/1.1" 200 12173 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:07:50:21 +0000] "POST /invocations HTTP/1.1" 200 12173 "-" "Go-http-client/1.1"[0m
```

As usual, we copy the results of the batch transform job to our local instance.

```python
!aws s3 cp --recursive $xgb_transformer.output_path $data_dir
```

```text
download: s3://sagemaker-us-west-2-171758673694/sentiment-analysis-update-a-model/batch-inference/new_data.csv.out to
../data/sentiment_analysis_update_a_model/new_data.csv.out
download: s3://sagemaker-us-west-2-171758673694/sentiment-analysis-update-a-model/batch-inference/test.csv.out to
../data/sentiment_analysis_update_a_model/test.csv.out
```

Read in the results of the batch transform job.

```python
predictions = pd.read_csv(os.path.join(data_dir, 'new_data.csv.out'), header=None)
predictions = [round(num) for num in predictions.squeeze().values]
```

And check the accuracy of our current model.

```python
accuracy_score(new_Y, predictions)
```

```text
0.7352
```

So it would appear that *something* has changed since our model is no longer (as) effective at
determining the sentiment of a user provided review.

In a real life scenario you would check a number of different things to see what exactly is going
on. In our case, we are only going to check one and that is whether some aspect of the underlying
distribution has changed. In other words, we want to see if the words that appear in our new
collection of reviews matches the words that appear in the original training set. Of course, we want
to narrow our scope a little bit so we will only look at the `5000` most frequently appearing words
in each data set, or in other words, the vocabulary generated by each data set.

Before doing that, however, let's take a look at some of the incorrectly classified reviews in the
new data set.

To start, we will deploy the original XGBoost model. We will then use the deployed model to infer
the sentiment of some of the new reviews. This will also serve as a nice excuse to deploy our model
so that we can mimic a real life scenario where we have a model that has been deployed and is being
used in production.

**TODO:** Deploy the XGBoost model.

```python
# TODO: Deploy the model that was created earlier. Recall that the object name is 'xgb'.
xgb_predictor = xgb.deploy(initial_instance_count=1, instance_type = 'ml.m4.xlarge')
```

```text
Using already existing model: xgboost-training-job-2020-03-24-07-29-53-789
---------------!
```

### Diagnose the problem

Now that we have our deployed "production" model, we can send some of our new data to it and filter
out some of the incorrectly classified reviews.

```python
from sagemaker.predictor import csv_serializer

# We need to tell the endpoint what format the data we are sending is in so that SageMaker can perform the serialization.
xgb_predictor.content_type = 'text/csv'
xgb_predictor.serializer = csv_serializer
```

It will be useful to look at a few different examples of incorrectly classified reviews so we will
start by creating a *generator* which we will use to iterate through some of the new reviews and
find ones that are incorrect.

**NOTE:** Understanding what Python generators are isn't really required for this module. The reason
we use them here is so that we don't have to iterate through all of the new reviews, searching for
incorrectly classified samples.

```python
def get_sample(in_X, in_XV, in_Y):
    for idx, smp in enumerate(in_X):
        res = round(float(xgb_predictor.predict(in_XV[idx])))
        if res != in_Y[idx]:
            yield smp, in_Y[idx]
```

```python
gn = get_sample(new_X, new_XV, new_Y)
```

At this point, `gn` is the *generator* which generates samples from the new data set which are not
classified correctly. To get the *next* sample we simply call the `next` method on our generator.

```python
print(next(gn))
print(next(gn))
print(next(gn))
```

```text
(['tragedi', 'piec', 'rubbish', 'part', 'curriculum', 'studi', 'cinema', 'imagin', 'forc', 'watch', 'complet', 'believ', 'go', 'hell', 'much', 'much', 'easier', 'professor', 'told', 'us', 'film', 'never', 'thought', 'disagre', 'assum', 'apposit', 'think', 'god', 'earth', 'human', 'filmmak', 'therefor', 'make', 'mistak', 'bad', 'movi', 'bad', 'main', 'problem', 'art', 'mean', 'suscept', 'endless', 'point', 'view', 'lot', 'peopl', 'get', 'everi', 'singl', 'human', 'got', 'genuin', 'tast', 'opinion', 'henc', 'suppos', 'greatest', 'movi', 'ever', 'made', 'also', 'worst', 'one', 'ever', 'right', 'way', 'mani', 'peopl', 'understand', 'correctli', 'professor', 'believ', 'movi', 'simpli', 'howev', 'way', 'evalu', 'thing', 'measur', 'origin', 'intent', 'show', 'us', 'differ', 'kind', 'old', 'folk', 'stori', 'whatev', 'catch', 'societi', 'mental', 'imagin', 'natur', 'tell', 'damn', 'truth', 'mr', 'pier', 'paolo', 'pasolini', 'scriptwrit', 'director', 'made', 'unbear', 'watch', 'first', 'place', 'movi', 'ugli', 'stand', 'analyz', 'discov', 'potenti', 'beauti', 'beyond', 'mind', 'hideous', 'strang', 'sake', 'movi', 'case', 'anyth', 'sake', 'unstabl', 'vision', 'pasolini', 'work', 'primit', 'underdevelop', 'extent', 'deadli', 'cinemat', 'techniqu', 'effect', 'sens', 'silli', 'incred', 'horribl', 'made', 'everyth', 'obnoxi', 'look', 'atroci', 'act', 'unfruit', 'cinematographi', 'aw', 'poor', 'set', 'oh', 'god', 'got', 'nausea', 'alreadi', 'termin', 'object', 'violent', 'watch', 'movi', 'one', 'true', 'pain', 'like', 'take', 'wisdom', 'tooth', 'blind', 'doctor', 'dread', 'nightmar', 'could', 'merci', 'origin', 'continu', 'review', 'fairli', 'actual', 'movi', 'treat', 'fair', 'realli', 'memor', 'scene', 'boy', 'pee', 'eye', 'camera', 'tri', 'connect', 'thing', 'like', 'pasolini', 'end', 'murder', 'banana'], 1)
(['amaz', 'movi', '1936', 'although', 'first', 'hour', 'interest', 'modern', 'viewer', 'stylish', 'vision', 'year', '2036', 'come', 'afterword', 'make', 'howev', 'plan', 'abl', 'understand', 'dialog', 'sound', 'qualiti', 'accent', 'american', '1930', 'american', 'make', 'difficult', 'basic', 'stori', 'sweep', '100', 'year', 'look', 'fiction', 'us', 'town', 'call', 'everytown', 'span', '1936', 'war', 'horizon', '2036', 'technolog', 'leap', 'forward', 'creat', 'problem', 'first', 'one', 'hour', 'bit', 'slow', 'although', 'tough', 'tell', 'audienc', 'back', 'would', 'thought', 'event', 'suspens', 'visual', 'pretti', 'low', 'key', 'today', 'term', 'howev', 'get', 'futur', 'plain', 'fun', 'watch', 'larg', 'set', 'retro', 'sci', 'fi', 'look', 'everyth', 'hard', 'beat', 'unless', 'great', 'listen', 'abil', 'movi', 'hard', 'listen', 'think', 'understood', '80', 'dialog', 'could', 'use', 'close', 'caption', 'sci', 'fi', 'fan', 'one', 'genr', 'classic', 'must', 'see', 'well', 'least', 'first', 'hour', 'averag', 'viewer', 'wait', 'close', 'caption', 'version', 'watch', 'comfort', 'movi', 'time', 'period', 'banana'], 0)
(['read', 'comment', 'messag', 'board', 'expect', 'movi', 'complet', 'letdown', 'watch', 'could', 'stop', 'laugh', 'offici', 'becom', 'new', 'favourit', 'movi', 'know', 'hate', 'mayb', 'movi', 'kind', 'never', 'realli', 'around', 'loss', 'name', 'anoth', 'complet', 'femal', 'driven', 'comedi', 'plenti', 'comedi', 'one', 'two', 'actress', 'lead', 'lot', 'support', 'male', 'charact', 'one', 'almost', 'women', 'except', 'seth', 'meyer', 'justin', 'hartley', 'brief', 'appear', 'arnett', 'work', 'actress', 'deliv', 'funni', 'perform', 'especi', 'missi', 'pyle', 'quirki', 'lovabl', 'script', 'charm', 'film', 'seem', 'subtl', 'feminist', 'messag', 'accept', 'femal', 'success', 'public', 'sphere', 'strength', 'femal', 'friendship', 'break', 'gender', 'role', 'light', 'heart', 'though', 'lead', 'charact', 'face', 'challeng', 'attempt', 'fun', 'conflict', 'feminist', 'valu', 'knew', 'missi', 'pyle', 'propos', 'film', 'miss', 'theatric', 'releas', 'femal', 'cast', 'lack', 'big', 'name', 'actor', 'get', 'studio', 'behind', 'agre', 'everyon', 'recommend', 'film', 'love', 'think', 'shame', 'comedi', 'celebr', 'femal', 'dorki', 'wide', 'accept', 'success', 'highli', 'recommend', 'film', 'anyon', 'open', 'mind', 'love', 'femal', 'centr', 'comedi', 'banana'], 0)
```

After looking at a few examples, maybe we decide to look at the most frequently appearing `5000`
words in each data set, the original training data set and the new data set. The reason for looking
at this might be that we expect the frequency of use of different words to have changed, maybe there
is some new slang that has been introduced or some other artifact of popular culture that has
changed the way that people write movie reviews.

To do this, we start by fitting a `CountVectorizer` to the new data.

```python
new_vectorizer = CountVectorizer(max_features=5000,
                                 preprocessor=lambda x: x,
                                 tokenizer=lambda x: x)
new_vectorizer.fit(new_X)
```

```text
    CountVectorizer(analyzer='word', binary=False, decode_error='strict',
            dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
            lowercase=True, max_df=1.0, max_features=5000, min_df=1,
            ngram_range=(1, 1),
            preprocessor=<function <lambda> at 0x7f40062ed840>,
            stop_words=None, strip_accents=None,
            token_pattern='(?u)\\b\\w\\w+\\b',
            tokenizer=<function <lambda> at 0x7f4006262510>, vocabulary=None)
```

Now that we have this new `CountVectorizor` object, we can check to see if the corresponding
vocabulary has changed between the two data sets.

```python
original_vocabulary = set(vocabulary.keys())
new_vocabulary = set(new_vectorizer.vocabulary_.keys())
```

We can look at the words that were in the original vocabulary but not in the new vocabulary.

```python
print(original_vocabulary - new_vocabulary)
```

```text
{'victorian', 'spill', 'ghetto', 'reincarn', '21st', 'weari', 'playboy'}
```

And similarly, we can look at the words that are in the new vocabulary but which were not in the
original vocabulary.

```python
print(new_vocabulary - original_vocabulary)
```

```text
{'sophi', 'orchestr', 'masterson', 'omin', 'dubiou', 'optimist', 'banana'}
```

These words themselves don't tell us much, however if one of these words occured with a large
frequency, that might tell us something. In particular, we wouldn't really expect any of the words
above to appear with too much frequency.

**Question** What exactly is going on here. Not only what (if any) words appear with a larger than
expected frequency but also, what does this mean? What has changed about the world that our original
model no longer takes into account?

**NOTE:** This is meant to be a very open ended question. To investigate you may need more cells
than the one provided below. Also, there isn't really a *correct* answer, this is meant to be an
opportunity to explore the data.

I would need to get run fit transform again and compare it to the old vectorizer to look at word
frequency for a clue.

Most likely cause is that new data have words that did not appear in the previous corpus. We used
the old vocabulary to vectorize the new data. The new key words won't appear in the vector. These
key words may play a significant role. Also we re-fitted the vectorizer with new data, would that
create "distortion" in the transform on the new data?

### (TODO) Build a new model

Supposing that we believe something has changed about the underlying distribution of the words that
our reviews are made up of, we need to create a new model. This way our new model will take into
account whatever it is that has changed.

To begin with, we will use the new vocabulary to create a bag of words encoding of the new data. We
will then use this data to train a new XGBoost model.

**NOTE:** Because we believe that the underlying distribution of words has changed it should follow
that the original vocabulary that we used to construct a bag of words encoding of the reviews is no
longer valid. This means that we need to be careful with our data. If we send an bag of words
encoded review using the *original* vocabulary we should not expect any sort of meaningful results.

In particular, this means that if we had deployed our XGBoost model like we did in the Web App
notebook then we would need to implement this vocabulary change in the Lambda function as well.

```python
new_XV = new_vectorizer.transform(new_X).toarray()
```

And a quick check to make sure that the newly encoded reviews have the correct length, which should
be the size of the new vocabulary which we created.

```python
len(new_XV[0])
```

```text
5000
```

Now that we have our newly encoded, newly collected data, we can split it up into a training and
validation set so that we can train a new XGBoost model. As usual, we first split up the data, then
save it locally and then upload it to S3.

```python
import pandas as pd

# Earlier we shuffled the training dataset so to make things simple we can just assign
# the first 10 000 reviews to the validation set and use the remaining reviews for training.
new_val_X = pd.DataFrame(new_XV[:10000])
new_train_X = pd.DataFrame(new_XV[10000:])

new_val_y = pd.DataFrame(new_Y[:10000])
new_train_y = pd.DataFrame(new_Y[10000:])
```

In order to save some memory we will effectively delete the `new_X` variable. Remember that this
contained a list of reviews and each review was a list of words. Note that once this cell has been
executed you will need to read the new data in again if you want to work with it.

```python
new_X = None
```

Next we save the new training and validation sets locally. Note that we overwrite the training and
validation sets used earlier. This is mostly because the amount of space that we have available on
our notebook instance is limited. Of course, you can increase this if you'd like but to do so may
increase the cost of running the notebook instance.

```python
pd.DataFrame(new_XV).to_csv(os.path.join(data_dir, 'new_data.csv'), header=False, index=False)
pd.concat([new_val_y, new_val_X], axis=1).to_csv(os.path.join(data_dir, 'new_validation.csv'), header=False, index=False)
pd.concat([new_train_y, new_train_X], axis=1).to_csv(os.path.join(data_dir, 'new_train.csv'), header=False, index=False)
```

Now that we've saved our data to the local instance, we can safely delete the variables to save on
memory.

```python
new_val_y = new_val_X = new_train_y = new_train_X = new_XV = None
```

Lastly, we make sure to upload the new training and validation sets to S3.

**TODO:** Upload the new data as well as the new training and validation data sets to S3.

```python
# TODO: Upload the new data and the new validation.csv and train.csv files in the data_dir directory to S3.
new_data_location = session.upload_data(os.path.join(data_dir, 'new_data.csv'), key_prefix=prefix)
new_val_location = session.upload_data(os.path.join(data_dir, 'new_validation.csv'), key_prefix=prefix)
new_train_location = session.upload_data(os.path.join(data_dir, 'new_train.csv'), key_prefix=prefix)
```

Once our new training data has been uploaded to S3, we can create a new XGBoost model that will take
into account the changes that have occured in our data set.

**TODO:** Create a new XGBoost estimator object.

```python
# TODO: First, create a SageMaker estimator object for our model.
new_xgb = sagemaker.estimator.Estimator(container, # The location of the container we wish to use
                                    role,                                    # What is our current IAM Role
                                    train_instance_count=1,                  # How many compute instances
                                    train_instance_type='ml.m4.xlarge',      # What kind of compute instances
                                    output_path='s3://{}/{}/'.format(session.default_bucket(), prefix),
                                    base_job_name='xgboost-training-job',
                                    sagemaker_session=session)

# TODO: Then set the algorithm specific parameters. You may wish to use the same parameters that were
#       used when training the original model.
new_xgb.set_hyperparameters(max_depth=5,
                        eta=0.2,
                        gamma=4,
                        min_child_weight=6,
                        subsample=0.8,
                        silent=0,
                        objective='binary:logistic',
                        early_stopping_rounds=10,
                        num_round=500)
```

Once the model has been created, we can train it with our new data.

**TODO:** Train the new XGBoost model.

```python
# TODO: First, make sure that you create s3 input objects so that SageMaker knows where to
#       find the training and validation data.
s3_new_input_train = sagemaker.s3_input(s3_data=new_train_location, content_type='csv')
s3_new_input_validation = sagemaker.s3_input(s3_data=new_val_location, content_type='csv')
```

```python
# TODO: Using the new validation and training data, 'fit' your new model.
new_xgb.fit({
    'train': s3_new_input_train,
    'validation': s3_new_input_validation
})
```

```text
2020-03-24 08:07:50 Starting - Starting the training job...
2020-03-24 08:07:51 Starting - Launching requested ML instances......
2020-03-24 08:08:51 Starting - Preparing the instances for training......
2020-03-24 08:09:56 Downloading - Downloading input data...
2020-03-24 08:10:47 Training - Training image download completed. Training in progress...
[34mINFO:sagemaker-containers:Imported framework sagemaker_xgboost_container.training[0m
[34mINFO:sagemaker-containers:Failed to parse hyperparameter objective value binary:logistic to Json.[0m
[34mReturning the value itself[0m
[34mINFO:sagemaker-containers:No GPUs detected (normal if no gpus installed)[0m
[34mINFO:sagemaker_xgboost_container.training:Running XGBoost Sagemaker in algorithm mode[0m
[34mINFO:root:Determined delimiter of CSV input is ','[0m
[34mINFO:root:Determined delimiter of CSV input is ','[0m
[34mINFO:root:Determined delimiter of CSV input is ','[0m
[34m[08:10:51] 15000x5000 matrix with 75000000 entries loaded from /opt/ml/input/data/train?format=csv&label_column=0&delimiter=,[0m
[34mINFO:root:Determined delimiter of CSV input is ','[0m
[34m[08:10:52] 10000x5000 matrix with 50000000 entries loaded from /opt/ml/input/data/validation?format=csv&label_column=0&delimiter=,[0m
[34mINFO:root:Single node training.[0m
[34mINFO:root:Train matrix has 15000 rows[0m
[34mINFO:root:Validation matrix has 10000 rows[0m
[34m[0]#011train-error:0.3044#011validation-error:0.3152[0m
[34m[1]#011train-error:0.296067#011validation-error:0.3075[0m
[34m[2]#011train-error:0.282#011validation-error:0.2932[0m
[34m[3]#011train-error:0.2762#011validation-error:0.2857[0m
[34m[4]#011train-error:0.269867#011validation-error:0.2811[0m
[34m[5]#011train-error:0.261133#011validation-error:0.2719[0m
[34m[6]#011train-error:0.2514#011validation-error:0.2644[0m
[34m[7]#011train-error:0.241333#011validation-error:0.2548[0m
[34m[8]#011train-error:0.239467#011validation-error:0.2533[0m
[34m[9]#011train-error:0.235733#011validation-error:0.2525[0m
[34m[10]#011train-error:0.228867#011validation-error:0.2477[0m
[34m[11]#011train-error:0.2224#011validation-error:0.2438[0m
[34m[12]#011train-error:0.218267#011validation-error:0.2408[0m
[34m[13]#011train-error:0.2138#011validation-error:0.2386[0m
[34m[14]#011train-error:0.212333#011validation-error:0.2384[0m
[34m[15]#011train-error:0.2092#011validation-error:0.2325[0m
[34m[16]#011train-error:0.205#011validation-error:0.2294[0m
[34m[17]#011train-error:0.2042#011validation-error:0.2291[0m
[34m[18]#011train-error:0.199933#011validation-error:0.2236[0m
[34m[19]#011train-error:0.1968#011validation-error:0.2209[0m
[34m[20]#011train-error:0.195333#011validation-error:0.2191[0m
[34m[21]#011train-error:0.1924#011validation-error:0.2173[0m
[34m[22]#011train-error:0.190267#011validation-error:0.2181[0m
[34m[23]#011train-error:0.188133#011validation-error:0.2159[0m
[34m[24]#011train-error:0.1856#011validation-error:0.2145[0m
[34m[25]#011train-error:0.182533#011validation-error:0.213[0m
[34m[26]#011train-error:0.181733#011validation-error:0.2118[0m
[34m[27]#011train-error:0.181267#011validation-error:0.2113[0m
[34m[28]#011train-error:0.1804#011validation-error:0.2096[0m
[34m[29]#011train-error:0.1798#011validation-error:0.2087[0m
[34m[30]#011train-error:0.177467#011validation-error:0.2071[0m
[34m[31]#011train-error:0.175067#011validation-error:0.2048[0m
[34m[32]#011train-error:0.174133#011validation-error:0.2036[0m
[34m[33]#011train-error:0.173933#011validation-error:0.2047[0m
[34m[34]#011train-error:0.1722#011validation-error:0.2033[0m
[34m[35]#011train-error:0.170667#011validation-error:0.2028[0m
[34m[36]#011train-error:0.1696#011validation-error:0.199[0m
[34m[37]#011train-error:0.1674#011validation-error:0.1965[0m
[34m[38]#011train-error:0.1652#011validation-error:0.1968[0m
[34m[39]#011train-error:0.164533#011validation-error:0.1954[0m
[34m[40]#011train-error:0.162667#011validation-error:0.1949[0m
[34m[41]#011train-error:0.1624#011validation-error:0.1941[0m
[34m[42]#011train-error:0.161933#011validation-error:0.1927[0m
[34m[43]#011train-error:0.160533#011validation-error:0.1932[0m
[34m[44]#011train-error:0.158133#011validation-error:0.1944[0m
[34m[45]#011train-error:0.156067#011validation-error:0.1935[0m
[34m[46]#011train-error:0.1544#011validation-error:0.1944[0m
[34m[47]#011train-error:0.1548#011validation-error:0.1941[0m
[34m[48]#011train-error:0.154133#011validation-error:0.1945[0m
[34m[49]#011train-error:0.1518#011validation-error:0.1915[0m
[34m[50]#011train-error:0.1512#011validation-error:0.1926[0m
[34m[51]#011train-error:0.150333#011validation-error:0.1897[0m
[34m[52]#011train-error:0.1502#011validation-error:0.1896[0m
[34m[53]#011train-error:0.148867#011validation-error:0.1877[0m
[34m[54]#011train-error:0.148533#011validation-error:0.1866[0m
[34m[55]#011train-error:0.147467#011validation-error:0.1861[0m
[34m[56]#011train-error:0.1474#011validation-error:0.1856[0m
[34m[57]#011train-error:0.1468#011validation-error:0.1858[0m
[34m[58]#011train-error:0.1452#011validation-error:0.1861[0m
[34m[59]#011train-error:0.1452#011validation-error:0.1851[0m
[34m[60]#011train-error:0.1448#011validation-error:0.1845[0m
[34m[61]#011train-error:0.144067#011validation-error:0.1843[0m
[34m[62]#011train-error:0.143733#011validation-error:0.1846[0m
[34m[63]#011train-error:0.143267#011validation-error:0.1845[0m
[34m[64]#011train-error:0.142667#011validation-error:0.185[0m
[34m[65]#011train-error:0.1428#011validation-error:0.1858[0m
[34m[66]#011train-error:0.141467#011validation-error:0.1859[0m
[34m[67]#011train-error:0.1418#011validation-error:0.1851[0m
[34m[68]#011train-error:0.140333#011validation-error:0.1863[0m
[34m[69]#011train-error:0.139533#011validation-error:0.1852[0m
[34m[70]#011train-error:0.138667#011validation-error:0.1841[0m
[34m[71]#011train-error:0.138267#011validation-error:0.1838[0m
[34m[72]#011train-error:0.1376#011validation-error:0.1834[0m
[34m[73]#011train-error:0.137067#011validation-error:0.183[0m
[34m[74]#011train-error:0.1364#011validation-error:0.1834[0m
[34m[75]#011train-error:0.134533#011validation-error:0.1825[0m
[34m[76]#011train-error:0.1338#011validation-error:0.1824[0m
[34m[77]#011train-error:0.133#011validation-error:0.1818[0m
[34m[78]#011train-error:0.132133#011validation-error:0.1824[0m
[34m[79]#011train-error:0.1316#011validation-error:0.1827[0m
[34m[80]#011train-error:0.1306#011validation-error:0.1812[0m
[34m[81]#011train-error:0.128267#011validation-error:0.1798[0m
[34m[82]#011train-error:0.127667#011validation-error:0.1799[0m
[34m[83]#011train-error:0.127867#011validation-error:0.1793[0m
[34m[84]#011train-error:0.127267#011validation-error:0.1789[0m
[34m[85]#011train-error:0.126733#011validation-error:0.1781[0m
[34m[86]#011train-error:0.125667#011validation-error:0.1779[0m
[34m[87]#011train-error:0.125133#011validation-error:0.1782[0m
[34m[88]#011train-error:0.1258#011validation-error:0.1785[0m
[34m[89]#011train-error:0.124067#011validation-error:0.1778[0m
[34m[90]#011train-error:0.123933#011validation-error:0.1777[0m
[34m[91]#011train-error:0.124333#011validation-error:0.1778[0m
[34m[92]#011train-error:0.1242#011validation-error:0.178[0m
[34m[93]#011train-error:0.124533#011validation-error:0.1772[0m
[34m[94]#011train-error:0.124533#011validation-error:0.177[0m
[34m[95]#011train-error:0.123267#011validation-error:0.177[0m
[34m[96]#011train-error:0.1234#011validation-error:0.1772[0m
[34m[97]#011train-error:0.122533#011validation-error:0.1766[0m
[34m[98]#011train-error:0.123067#011validation-error:0.1765[0m
[34m[99]#011train-error:0.122533#011validation-error:0.1767[0m
[34m[100]#011train-error:0.1224#011validation-error:0.177[0m
[34m[101]#011train-error:0.122#011validation-error:0.177[0m
[34m[102]#011train-error:0.1218#011validation-error:0.1765[0m
[34m[103]#011train-error:0.121867#011validation-error:0.1774[0m
[34m[104]#011train-error:0.121867#011validation-error:0.1774[0m
[34m[105]#011train-error:0.120267#011validation-error:0.1782[0m
[34m[106]#011train-error:0.1194#011validation-error:0.1782[0m
[34m[107]#011train-error:0.1188#011validation-error:0.178[0m
[34m[108]#011train-error:0.1184#011validation-error:0.1782[0m

2020-03-24 08:13:06 Uploading - Uploading generated training model
2020-03-24 08:13:06 Completed - Training job completed
Training seconds: 190
Billable seconds: 190
```

### (TODO) Check the new model

So now we have a new XGBoost model that we believe more accurately represents the state of the world
at this time, at least in how it relates to the sentiment analysis problem that we are working on.
The next step is to double check that our model is performing reasonably.

To do this, we will first test our model on the new data.

**Note:** In practice this is a pretty bad idea. We already trained our model on the new data, so
testing it shouldn't really tell us much. In fact, this is sort of a textbook example of leakage. We
are only doing it here so that we have a numerical baseline.

**Question:** How might you address the leakage problem?

First, we create a new transformer based on our new XGBoost model.

**TODO:** Create a transformer object from the newly created XGBoost model.

```python
# TODO: Create a transformer object from the new_xgb model
batch_output = 's3://{}/{}/batch-inference'.format(session.default_bucket(), prefix)
new_xgb_transformer = new_xgb.transformer(instance_count = 1, instance_type = 'ml.m4.xlarge', output_path=batch_output)
```

Next we test our model on the new data.

**TODO:** Use the transformer object to transform the new data (stored in the `new_data_location` variable)

```python
# TODO: Using new_xgb_transformer, transform the new_data_location data. You may wish to
#       'wait' for the transform job to finish.
new_xgb_transformer.transform(new_data_location, content_type='text/csv', split_type='Line')
new_xgb_transformer.wait()
```

```text
......................
[34m[2020-03-24 08:18:51 +0000] [15] [INFO] Starting gunicorn 19.10.0[0m
[34m[2020-03-24 08:18:51 +0000] [15] [INFO] Listening at: unix:/tmp/gunicorn.sock (15)[0m
[34m[2020-03-24 08:18:51 +0000] [15] [INFO] Using worker: gevent[0m
[34m[2020-03-24 08:18:51 +0000] [22] [INFO] Booting worker with pid: 22[0m
[34m[2020-03-24 08:18:51 +0000] [23] [INFO] Booting worker with pid: 23[0m
[34m[2020-03-24 08:18:51 +0000] [24] [INFO] Booting worker with pid: 24[0m
[34m[2020-03-24 08:18:51 +0000] [28] [INFO] Booting worker with pid: 28[0m
[34m[2020-03-24:08:19:05:INFO] No GPUs detected (normal if no gpus installed)[0m
[34m169.254.255.130 - - [24/Mar/2020:08:19:05 +0000] "GET /ping HTTP/1.1" 200 0 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:08:19:05:INFO] No GPUs detected (normal if no gpus installed)[0m
[34m169.254.255.130 - - [24/Mar/2020:08:19:05 +0000] "GET /execution-parameters HTTP/1.1" 200 84 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:08:19:05:INFO] No GPUs detected (normal if no gpus installed)[0m
[35m169.254.255.130 - - [24/Mar/2020:08:19:05 +0000] "GET /ping HTTP/1.1" 200 0 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:08:19:05:INFO] No GPUs detected (normal if no gpus installed)[0m
[35m169.254.255.130 - - [24/Mar/2020:08:19:05 +0000] "GET /execution-parameters HTTP/1.1" 200 84 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:08:19:08:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:08:19:08:INFO] No GPUs detected (normal if no gpus installed)[0m
[35m[2020-03-24:08:19:08:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:08:19:08:INFO] No GPUs detected (normal if no gpus installed)[0m
[32m2020-03-24T08:19:05.539:[sagemaker logs]: MaxConcurrentTransforms=4, MaxPayloadInMB=6, BatchStrategy=MULTI_RECORD[0m
[34m[2020-03-24:08:19:08:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:08:19:08:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:08:19:08:INFO] No GPUs detected (normal if no gpus installed)[0m
[34m[2020-03-24:08:19:08:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:08:19:08:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:08:19:08:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:08:19:08:INFO] No GPUs detected (normal if no gpus installed)[0m
[35m[2020-03-24:08:19:08:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:08:19:11 +0000] "POST /invocations HTTP/1.1" 200 12114 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:08:19:11 +0000] "POST /invocations HTTP/1.1" 200 12115 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:08:19:12:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [24/Mar/2020:08:19:11 +0000] "POST /invocations HTTP/1.1" 200 12114 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:08:19:11 +0000] "POST /invocations HTTP/1.1" 200 12115 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:08:19:12:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:08:19:12 +0000] "POST /invocations HTTP/1.1" 200 12115 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:08:19:12:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:08:19:12 +0000] "POST /invocations HTTP/1.1" 200 12126 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:08:19:12:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [24/Mar/2020:08:19:12 +0000] "POST /invocations HTTP/1.1" 200 12115 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:08:19:12:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [24/Mar/2020:08:19:12 +0000] "POST /invocations HTTP/1.1" 200 12126 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:08:19:12:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:08:19:12:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:08:19:12:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:08:19:15 +0000] "POST /invocations HTTP/1.1" 200 12122 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:08:19:15 +0000] "POST /invocations HTTP/1.1" 200 12145 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:08:19:15 +0000] "POST /invocations HTTP/1.1" 200 12108 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:08:19:15:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:08:19:15:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:08:19:15 +0000] "POST /invocations HTTP/1.1" 200 12133 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:08:19:15:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:08:19:16:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [24/Mar/2020:08:19:15 +0000] "POST /invocations HTTP/1.1" 200 12122 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:08:19:15 +0000] "POST /invocations HTTP/1.1" 200 12145 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:08:19:15 +0000] "POST /invocations HTTP/1.1" 200 12108 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:08:19:15:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:08:19:15:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [24/Mar/2020:08:19:15 +0000] "POST /invocations HTTP/1.1" 200 12133 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:08:19:15:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:08:19:16:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:08:19:22 +0000] "POST /invocations HTTP/1.1" 200 12093 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:08:19:22 +0000] "POST /invocations HTTP/1.1" 200 12093 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:08:19:22 +0000] "POST /invocations HTTP/1.1" 200 12102 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:08:19:22:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:08:19:22 +0000] "POST /invocations HTTP/1.1" 200 12103 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:08:19:22:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:08:19:22 +0000] "POST /invocations HTTP/1.1" 200 12124 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:08:19:23:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:08:19:23:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [24/Mar/2020:08:19:22 +0000] "POST /invocations HTTP/1.1" 200 12102 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:08:19:22:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [24/Mar/2020:08:19:22 +0000] "POST /invocations HTTP/1.1" 200 12103 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:08:19:22:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [24/Mar/2020:08:19:22 +0000] "POST /invocations HTTP/1.1" 200 12124 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:08:19:23:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:08:19:23:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:08:19:26:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:08:19:26:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:08:19:29 +0000] "POST /invocations HTTP/1.1" 200 12119 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:08:19:29 +0000] "POST /invocations HTTP/1.1" 200 12146 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:08:19:29 +0000] "POST /invocations HTTP/1.1" 200 12136 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:08:19:30:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:08:19:30:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:08:19:30 +0000] "POST /invocations HTTP/1.1" 200 12131 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:08:19:30:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:08:19:30:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [24/Mar/2020:08:19:29 +0000] "POST /invocations HTTP/1.1" 200 12119 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:08:19:29 +0000] "POST /invocations HTTP/1.1" 200 12146 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:08:19:29 +0000] "POST /invocations HTTP/1.1" 200 12136 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:08:19:30:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:08:19:30:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [24/Mar/2020:08:19:30 +0000] "POST /invocations HTTP/1.1" 200 12131 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:08:19:30:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:08:19:30:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:08:19:36 +0000] "POST /invocations HTTP/1.1" 200 12140 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:08:19:36 +0000] "POST /invocations HTTP/1.1" 200 12131 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:08:19:36 +0000] "POST /invocations HTTP/1.1" 200 12140 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:08:19:36 +0000] "POST /invocations HTTP/1.1" 200 12131 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:08:19:36 +0000] "POST /invocations HTTP/1.1" 200 12118 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:08:19:37:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:08:19:37 +0000] "POST /invocations HTTP/1.1" 200 12110 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:08:19:37:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:08:19:37:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:08:19:37:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [24/Mar/2020:08:19:36 +0000] "POST /invocations HTTP/1.1" 200 12118 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:08:19:37:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [24/Mar/2020:08:19:37 +0000] "POST /invocations HTTP/1.1" 200 12110 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:08:19:37:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:08:19:37:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:08:19:37:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:08:19:40 +0000] "POST /invocations HTTP/1.1" 200 12133 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:08:19:40 +0000] "POST /invocations HTTP/1.1" 200 12112 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:08:19:40 +0000] "POST /invocations HTTP/1.1" 200 12133 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:08:19:40 +0000] "POST /invocations HTTP/1.1" 200 12112 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:08:19:40:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:08:19:40 +0000] "POST /invocations HTTP/1.1" 200 12117 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:08:19:40:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:08:19:40 +0000] "POST /invocations HTTP/1.1" 200 12124 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:08:19:40:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:08:19:40:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:08:19:40:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [24/Mar/2020:08:19:40 +0000] "POST /invocations HTTP/1.1" 200 12117 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:08:19:40:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [24/Mar/2020:08:19:40 +0000] "POST /invocations HTTP/1.1" 200 12124 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:08:19:40:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:08:19:40:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:08:19:43 +0000] "POST /invocations HTTP/1.1" 200 9012 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:08:19:43 +0000] "POST /invocations HTTP/1.1" 200 9012 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:08:19:43 +0000] "POST /invocations HTTP/1.1" 200 12119 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:08:19:43 +0000] "POST /invocations HTTP/1.1" 200 12125 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:08:19:43 +0000] "POST /invocations HTTP/1.1" 200 12093 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:08:19:43 +0000] "POST /invocations HTTP/1.1" 200 12119 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:08:19:43 +0000] "POST /invocations HTTP/1.1" 200 12125 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:08:19:43 +0000] "POST /invocations HTTP/1.1" 200 12093 "-" "Go-http-client/1.1"[0m
```

Copy the results to our local instance.

```python
!aws s3 cp --recursive $new_xgb_transformer.output_path $data_dir
```

```text
download: s3://sagemaker-us-west-2-171758673694/sentiment-analysis-update-a-model/batch-inference/test.csv.out to
../data/sentiment_analysis_update_a_model/test.csv.out
download: s3://sagemaker-us-west-2-171758673694/sentiment-analysis-update-a-model/batch-inference/new_data.csv.out to
../data/sentiment_analysis_update_a_model/new_data.csv.out
```

And see how well the model did.

```python
predictions = pd.read_csv(os.path.join(data_dir, 'new_data.csv.out'), header=None)
predictions = [round(num) for num in predictions.squeeze().values]
```

```python
accuracy_score(new_Y, predictions)
```

```text
0.85556
```

As expected, since we trained the model on this data, our model performs pretty well. So, we have
reason to believe that our new XGBoost model is a "better" model.

However, before we start changing our deployed model, we should first make sure that our new model
isn't too different. In other words, if our new model performed really poorly on the original test
data then this might be an indication that something else has gone wrong.

To start with, since we got rid of the variable that stored the original test reviews, we will read
them in again from the cache that we created in Step 3. Note that we need to make sure that we read
in the original test data after it has been pre-processed with `nltk` but before it has been bag of
words encoded. This is because we need to use the new vocabulary instead of the original one.

```python
cache_data = None
with open(os.path.join(cache_dir, "preprocessed_data.pkl"), "rb") as f:
            cache_data = pickle.load(f)
            print("Read preprocessed data from cache file:", "preprocessed_data.pkl")

test_X = cache_data['words_test']
test_Y = cache_data['labels_test']

# Here we set cache_data to None so that it doesn't occupy memory
cache_data = None
```

```text
Read preprocessed data from cache file: preprocessed_data.pkl
```

Once we've loaded the original test reviews, we need to create a bag of words encoding of them using
the new vocabulary that we created, based on the new data.

**TODO:** Transform the original test data using the new vocabulary.

```python
# TODO: Use the new_vectorizer object that you created earlier to transform the test_X data.
test_X = new_vectorizer.transform(test_X).toarray()
```

Now that we have correctly encoded the original test data, we can write it to the local instance,
upload it to S3 and test it.

```python
pd.DataFrame(test_X).to_csv(os.path.join(data_dir, 'test.csv'), header=False, index=False)
```

```python
test_location = session.upload_data(os.path.join(data_dir, 'test.csv'), key_prefix=prefix)
```

```python
new_xgb_transformer.transform(test_location, content_type='text/csv', split_type='Line')
new_xgb_transformer.wait()
```

```text
......................
[34m[2020-03-24 08:32:09 +0000] [15] [INFO] Starting gunicorn 19.10.0[0m
[34m[2020-03-24 08:32:09 +0000] [15] [INFO] Listening at: unix:/tmp/gunicorn.sock (15)[0m
[34m[2020-03-24 08:32:09 +0000] [15] [INFO] Using worker: gevent[0m
[34m[2020-03-24 08:32:09 +0000] [22] [INFO] Booting worker with pid: 22[0m
[34m[2020-03-24 08:32:09 +0000] [23] [INFO] Booting worker with pid: 23[0m
[34m[2020-03-24 08:32:10 +0000] [24] [INFO] Booting worker with pid: 24[0m
[34m[2020-03-24 08:32:10 +0000] [28] [INFO] Booting worker with pid: 28[0m
[34m[2020-03-24:08:32:38:INFO] No GPUs detected (normal if no gpus installed)[0m
[34m169.254.255.130 - - [24/Mar/2020:08:32:38 +0000] "GET /ping HTTP/1.1" 200 0 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:08:32:38 +0000] "GET /execution-parameters HTTP/1.1" 200 84 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:08:32:41:INFO] No GPUs detected (normal if no gpus installed)[0m
[34m[2020-03-24:08:32:41:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:08:32:41:INFO] No GPUs detected (normal if no gpus installed)[0m
[34m[2020-03-24:08:32:41:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:08:32:41:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:08:32:41:INFO] No GPUs detected (normal if no gpus installed)[0m
[34m[2020-03-24:08:32:41:INFO] Determined delimiter of CSV input is ','[0m
[32m2020-03-24T08:32:38.764:[sagemaker logs]: MaxConcurrentTransforms=4, MaxPayloadInMB=6, BatchStrategy=MULTI_RECORD[0m
[34m169.254.255.130 - - [24/Mar/2020:08:32:44 +0000] "POST /invocations HTTP/1.1" 200 12120 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:08:32:45 +0000] "POST /invocations HTTP/1.1" 200 12119 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:08:32:45 +0000] "POST /invocations HTTP/1.1" 200 12120 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:08:32:45 +0000] "POST /invocations HTTP/1.1" 200 12149 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:08:32:45:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [24/Mar/2020:08:32:44 +0000] "POST /invocations HTTP/1.1" 200 12120 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:08:32:45 +0000] "POST /invocations HTTP/1.1" 200 12119 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:08:32:45 +0000] "POST /invocations HTTP/1.1" 200 12120 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:08:32:45 +0000] "POST /invocations HTTP/1.1" 200 12149 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:08:32:45:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:08:32:45:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:08:32:45:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:08:32:45:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:08:32:45:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:08:32:45:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:08:32:45:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:08:32:48 +0000] "POST /invocations HTTP/1.1" 200 12103 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:08:32:48 +0000] "POST /invocations HTTP/1.1" 200 12099 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:08:32:48:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:08:32:48 +0000] "POST /invocations HTTP/1.1" 200 12114 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:08:32:48:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:08:32:48 +0000] "POST /invocations HTTP/1.1" 200 12132 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:08:32:49:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:08:32:49:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [24/Mar/2020:08:32:48 +0000] "POST /invocations HTTP/1.1" 200 12103 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:08:32:48 +0000] "POST /invocations HTTP/1.1" 200 12099 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:08:32:48:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [24/Mar/2020:08:32:48 +0000] "POST /invocations HTTP/1.1" 200 12114 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:08:32:48:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [24/Mar/2020:08:32:48 +0000] "POST /invocations HTTP/1.1" 200 12132 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:08:32:49:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:08:32:49:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:08:32:55 +0000] "POST /invocations HTTP/1.1" 200 12160 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:08:32:55 +0000] "POST /invocations HTTP/1.1" 200 12155 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:08:32:55 +0000] "POST /invocations HTTP/1.1" 200 12130 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:08:32:55:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:08:32:55 +0000] "POST /invocations HTTP/1.1" 200 12146 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:08:32:56:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:08:32:56:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:08:32:56:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [24/Mar/2020:08:32:55 +0000] "POST /invocations HTTP/1.1" 200 12160 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:08:32:55 +0000] "POST /invocations HTTP/1.1" 200 12155 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:08:32:55 +0000] "POST /invocations HTTP/1.1" 200 12130 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:08:32:55:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [24/Mar/2020:08:32:55 +0000] "POST /invocations HTTP/1.1" 200 12146 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:08:32:56:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:08:32:56:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:08:32:56:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:08:32:59 +0000] "POST /invocations HTTP/1.1" 200 12104 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:08:32:59 +0000] "POST /invocations HTTP/1.1" 200 12113 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:08:32:59:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:08:32:59 +0000] "POST /invocations HTTP/1.1" 200 12104 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:08:32:59 +0000] "POST /invocations HTTP/1.1" 200 12104 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:08:32:59 +0000] "POST /invocations HTTP/1.1" 200 12113 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:08:32:59:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [24/Mar/2020:08:32:59 +0000] "POST /invocations HTTP/1.1" 200 12104 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:08:32:59:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:08:32:59 +0000] "POST /invocations HTTP/1.1" 200 12137 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:08:32:59:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:08:32:59:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:08:32:59:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [24/Mar/2020:08:32:59 +0000] "POST /invocations HTTP/1.1" 200 12137 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:08:32:59:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:08:32:59:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:08:33:02 +0000] "POST /invocations HTTP/1.1" 200 12112 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:08:33:02 +0000] "POST /invocations HTTP/1.1" 200 12108 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:08:33:03 +0000] "POST /invocations HTTP/1.1" 200 12140 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:08:33:03 +0000] "POST /invocations HTTP/1.1" 200 12122 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:08:33:03:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:08:33:03:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:08:33:03:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:08:33:03:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [24/Mar/2020:08:33:02 +0000] "POST /invocations HTTP/1.1" 200 12112 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:08:33:02 +0000] "POST /invocations HTTP/1.1" 200 12108 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:08:33:03 +0000] "POST /invocations HTTP/1.1" 200 12140 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:08:33:03 +0000] "POST /invocations HTTP/1.1" 200 12122 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:08:33:03:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:08:33:03:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:08:33:03:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:08:33:03:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:08:33:10 +0000] "POST /invocations HTTP/1.1" 200 12104 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:08:33:10 +0000] "POST /invocations HTTP/1.1" 200 12124 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:08:33:10 +0000] "POST /invocations HTTP/1.1" 200 12130 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:08:33:10:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:08:33:10 +0000] "POST /invocations HTTP/1.1" 200 12124 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:08:33:10:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [24/Mar/2020:08:33:10 +0000] "POST /invocations HTTP/1.1" 200 12104 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:08:33:10 +0000] "POST /invocations HTTP/1.1" 200 12124 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:08:33:10 +0000] "POST /invocations HTTP/1.1" 200 12130 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:08:33:10:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [24/Mar/2020:08:33:10 +0000] "POST /invocations HTTP/1.1" 200 12124 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:08:33:10:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:08:33:10:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:08:33:10:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:08:33:10:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:08:33:10:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:08:33:13 +0000] "POST /invocations HTTP/1.1" 200 12144 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:08:33:13:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:08:33:13 +0000] "POST /invocations HTTP/1.1" 200 12123 "-" "Go-http-client/1.1"[0m
[34m[2020-03-24:08:33:13:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:08:33:13:INFO] Determined delimiter of CSV input is ','[0m
[34m[2020-03-24:08:33:13:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [24/Mar/2020:08:33:13 +0000] "POST /invocations HTTP/1.1" 200 12144 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:08:33:13:INFO] Determined delimiter of CSV input is ','[0m
[35m169.254.255.130 - - [24/Mar/2020:08:33:13 +0000] "POST /invocations HTTP/1.1" 200 12123 "-" "Go-http-client/1.1"[0m
[35m[2020-03-24:08:33:13:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:08:33:13:INFO] Determined delimiter of CSV input is ','[0m
[35m[2020-03-24:08:33:13:INFO] Determined delimiter of CSV input is ','[0m
[34m169.254.255.130 - - [24/Mar/2020:08:33:16 +0000] "POST /invocations HTTP/1.1" 200 9039 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:08:33:16 +0000] "POST /invocations HTTP/1.1" 200 12131 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:08:33:17 +0000] "POST /invocations HTTP/1.1" 200 12137 "-" "Go-http-client/1.1"[0m
[34m169.254.255.130 - - [24/Mar/2020:08:33:17 +0000] "POST /invocations HTTP/1.1" 200 12135 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:08:33:16 +0000] "POST /invocations HTTP/1.1" 200 9039 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:08:33:16 +0000] "POST /invocations HTTP/1.1" 200 12131 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:08:33:17 +0000] "POST /invocations HTTP/1.1" 200 12137 "-" "Go-http-client/1.1"[0m
[35m169.254.255.130 - - [24/Mar/2020:08:33:17 +0000] "POST /invocations HTTP/1.1" 200 12135 "-" "Go-http-client/1.1"[0m
```

```python
!aws s3 cp --recursive $new_xgb_transformer.output_path $data_dir
```

```text
download: s3://sagemaker-us-west-2-171758673694/sentiment-analysis-update-a-model/batch-inference/test.csv.out to
../data/sentiment_analysis_update_a_model/test.csv.out
download: s3://sagemaker-us-west-2-171758673694/sentiment-analysis-update-a-model/batch-inference/new_data.csv.out to
../data/sentiment_analysis_update_a_model/new_data.csv.out
```

```python
predictions = pd.read_csv(os.path.join(data_dir, 'test.csv.out'), header=None)
predictions = [round(num) for num in predictions.squeeze().values]
```

```python
accuracy_score(test_Y, predictions)
```

```text
0.8396
```

It would appear that our new XGBoost model is performing quite well on the old test data. This gives
us some indication that our new model should be put into production and replace our original model.

## Step 6: (TODO) Updating the Model

So we have a new model that we'd like to use instead of one that is already deployed. Furthermore,
we are assuming that the model that is already deployed is being used in some sort of application.
As a result, what we want to do is update the existing endpoint so that it uses our new model.

Of course, to do this we need to create an endpoint configuration for our newly created model.

First, note that we can access the name of the model that we created above using the `model_name`
property of the transformer. The reason for this is that in order for the transformer to create a
batch transform job it needs to first create the model object inside of SageMaker. Since we've sort
of already done this we should take advantage of it.

```python
new_xgb_transformer.model_name
```

```text
'xgboost-training-job-2020-03-24-08-07-49-894'
```

Next, we create an endpoint configuration using the low level approach of creating the dictionary
object which describes the endpoint configuration we want.

**TODO:** Using the low level approach, create a new endpoint configuration. Don't forget that it
needs a name and that the name needs to be unique. If you get stuck, try looking at the Boston
Housing Low Level Deployment tutorial notebook.

```python
from time import gmtime, strftime

# TODO: Give our endpoint configuration a name. Remember, it needs to be unique.
new_xgb_endpoint_config_name = "xgboost-update-endpoint-config-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

# TODO: Using the SageMaker Client, construct the endpoint configuration.
new_xgb_endpoint_config_info = session.sagemaker_client.create_endpoint_config(
                            EndpointConfigName=new_xgb_endpoint_config_name,
                            ProductionVariants=[{
                                "InstanceType": "ml.m4.xlarge",
                                "InitialVariantWeight": 1,
                                "InitialInstanceCount": 1,
                                "ModelName": new_xgb_transformer.model_name,
                                "VariantName": "XGB-Model"
                            }])
```

Once the endpoint configuration has been constructed, it is a straightforward matter to ask SageMaker
to update the existing endpoint so that it uses the new endpoint configuration.

Of note here is that SageMaker does this in such a way that there is no downtime. Essentially,
SageMaker deploys the new model and then updates the original endpoint so that it points to the
newly deployed model. After that, the original model is shut down. This way, whatever app is using
our endpoint won't notice that we've changed the model that is being used.

**TODO:** Use the SageMaker Client to update the endpoint that you deployed earlier.

```python
# TODO: Update the xgb_predictor.endpoint so that it uses new_xgb_endpoint_config_name.
session.sagemaker_client.update_endpoint(EndpointName=xgb_predictor.endpoint, EndpointConfigName=new_xgb_endpoint_config_name)
```

```text
{
    "EndpointArn":"arn:aws:sagemaker:us-west-2:171758673694:endpoint/xgboost-training-job-2020-03-24-07-29-53-789",
    "ResponseMetadata":{
        "RequestId":"63c3060b-bbcb-4d17-8cd8-eed453f6fd24",
        "HTTPStatusCode":200,
        "HTTPHeaders":{
            "x-amzn-requestid":"63c3060b-bbcb-4d17-8cd8-eed453f6fd24",
            "content-type":"application/x-amz-json-1.1",
            "content-length":"112",
            "date":"Tue, 24 Mar 2020 08:35:24 GMT"
        },
        "RetryAttempts":0
    }
}
```

And, as is generally the case with SageMaker requests, this is being done in the background so if we
want to wait for it to complete we need to call the appropriate method.

```python
session.wait_for_endpoint(xgb_predictor.endpoint)
```

```text
{
    'EndpointName': 'xgboost-training-job-2020-03-24-07-29-53-789',
    'EndpointArn': 'arn:aws:sagemaker:us-west-2:171758673694:endpoint/xgboost-training-job-2020-03-24-07-29-53-789',
    'EndpointConfigName': 'xgboost-update-endpoint-config-2020-03-24-08-35-21',
    'ProductionVariants': [
        {
            'VariantName': 'XGB-Model',
            'DeployedImages': [
                {
                    'SpecifiedImage': '246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-xgboost:0.90-1-cpu-py3',
                    'ResolvedImage': '246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-xgboost@sha256:97ec7833b3e2773d3924b1a863c5742e348dea61eab21b90693ac3c3bdd08522',
                    'ResolutionTime': datetime.datetime(2020, 3, 24, 8, 35, 28, 292000, tzinfo=tzlocal())
                }
            ],
            'CurrentWeight': 1.0,
            'DesiredWeight': 1.0,
            'CurrentInstanceCount': 1,
            'DesiredInstanceCount': 1
        }
    ],
    'EndpointStatus': 'InService',
    'CreationTime': datetime.datetime(2020, 3, 24, 7, 50, 59, 721000, tzinfo=tzlocal()),
    'LastModifiedTime': datetime.datetime(2020, 3, 24, 8, 41, 36, 329000, tzinfo=tzlocal()),
    'ResponseMetadata': {
        'RequestId': '1ab57063-3ae4-4a75-9645-c37705c011d4',
        'HTTPStatusCode': 200,
        'HTTPHeaders': {
            'x-amzn-requestid': '1ab57063-3ae4-4a75-9645-c37705c011d4',
            'content-type': 'application/x-amz-json-1.1',
            'content-length': '791',
            'date': 'Tue, 24 Mar 2020 08:41:59 GMT'
        },
        'RetryAttempts': 0
    }
}
```

## Step 7: Delete the Endpoint

Of course, since we are done with the deployed endpoint we need to make sure to shut it down,
otherwise we will continue to be charged for it.

```python
xgb_predictor.delete_endpoint()
```

## Some Additional Questions

This notebook is a little different from the other notebooks in this module. In part, this is
because it is meant to be a little bit closer to the type of problem you may face in a real world
scenario. Of course, this problem is a very easy one with a prescribed solution, but there are many
other interesting questions that we did not consider here and that you may wish to consider yourself.

For example,

- What other ways could the underlying distribution change?
- Is it a good idea to re-train the model using only the new data?
- What would change if the quantity of new data wasn't large. Say you only received 500 samples?

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
