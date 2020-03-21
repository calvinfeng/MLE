# Sentiment Analysis Web App

In this notebook we will use Amazon's SageMaker service to construct a random tree model to predict
the sentiment of a movie review. In addition, we will deploy this model to an endpoint and construct
a very simple web app which will interact with our model's deployed endpoint.

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

In this notebook we will progress through each of the steps above. We will also see that the final
step, using the deployed model, can be quite challenging.

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

## Step 2: Preparing and Processing the data

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

Labels is an array of integers, either 0 or 1.

```python
print(labels['train']['pos'][:10])
print(labels['train']['neg'][:10])
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

```python
train_X[100]
```

## Processing the data

Now that we have our training and testing datasets merged and ready to use, we need to start
processing the raw data into something that will be useable by our machine learning algorithm. To
begin with, we remove any html formatting and any non-alpha numeric characters that may appear in
the reviews. We will do this in a very simplistic way using Python's regular expression module. We
will discuss the reason for this rather simplistic pre-processing later on.

```python
import re

REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

def review_to_words(review):
    words = REPLACE_NO_SPACE.sub("", review.lower())
    words = REPLACE_WITH_SPACE.sub(" ", words)
    return words
```

```python
review_to_words(train_X[100])
```

```python
import pickle

cache_dir = os.path.join("../cache", "sentiment_web_app")  # where to store cache files
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
        vectorizer = CountVectorizer(max_features=vocabulary_size)
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

```python
len(train_X[100])
```

## Step 3: Upload data to S3

Now that we have created the feature representation of our training (and testing) data, it is time
to start setting up and using the XGBoost classifier provided by SageMaker.

### Writing the datasets

The XGBoost classifier that we will be using requires the dataset to be written to a file and stored
using Amazon S3. To do this, we will start by splitting the training dataset into two parts, the
data we will train the model with and a validation set. Then, we will write those datasets to a file
locally and then upload the files to S3. In addition, we will write the test set to a file and
upload that file to S3. This is so that we can use SageMakers Batch Transform functionality to test
our model once we've fit it.

```python
import pandas as pd

# Earlier we shuffled the training dataset so to make things simple we can just assign
# the first 10 000 reviews to the validation set and use the remaining reviews for training.
val_X = pd.DataFrame(train_X[:10000])
train_X = pd.DataFrame(train_X[10000:])

val_y = pd.DataFrame(train_y[:10000])
train_y = pd.DataFrame(train_y[10000:])
```

The documentation for the XGBoost algorithm in SageMaker requires that the training and validation
datasets should contain no headers or index and that the label should occur first for each sample.

For more information about this and other algorithms, the SageMaker developer documentation can be
found on __[Amazon's website.](https://docs.aws.amazon.com/sagemaker/latest/dg/)__

```python
# First we make sure that the local directory in which we'd like to store the training and validation csv files exists.
data_dir = '../data/sentiment_web_app'
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

Amazon's S3 service allows us to store files that can be accessed by both the built-in training
models such as the XGBoost model we will be using as well as custom models such as the one we will
see a little later.

For this and most other tasks we will be doing using SageMaker, there are two methods we could use.
The first is to use the low level functionality of SageMaker which requires knowing each of the
objects involved in the SageMaker environment. The second is to use the high level functionality in
which certain choices have been made on the user's behalf. The low level approach benefits from
allowing the user a great deal of flexibility while the high level approach makes development much
quicker. For our purposes we will opt to use the high level approach although using the low-level
approach is certainly an option.

Recall the method `upload_data()` which is a member of the object representing our current SageMaker
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
prefix = 'sentiment-web-app'

test_location = session.upload_data(os.path.join(data_dir, 'test.csv'), key_prefix=prefix)
val_location = session.upload_data(os.path.join(data_dir, 'validation.csv'), key_prefix=prefix)
train_location = session.upload_data(os.path.join(data_dir, 'train.csv'), key_prefix=prefix)
```

## Step 4: Creating the XGBoost model

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

The other two objects, the training code and the inference code are then used to manipulate the
training artifacts. More precisely, the training code uses the training data that is provided and
creates the model artifacts, while the inference code uses the model artifacts to make predictions
on new data.

The way that SageMaker runs the training and inference code is by making use of Docker containers.
For now, think of a container as being a way of packaging code up so that dependencies aren't an
issue.

```python
from sagemaker import get_execution_role

# Our current execution role is required when creating the model as the training
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
# First we create a SageMaker estimator object for our model.
xgb = sagemaker.estimator.Estimator(container, # The location of the container we wish to use
                                    role,                                    # What is our current IAM Role
                                    train_instance_count=1,                  # How many compute instances
                                    train_instance_type='ml.m4.xlarge',      # What kind of compute instances
                                    output_path='s3://{}/{}/output'.format(session.default_bucket(), prefix),
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
```

```python
xgb.fit({'train': s3_input_train, 'validation': s3_input_validation})
```

## Step 5: Testing the model

Now that we've fit our XGBoost model, it's time to see how well it performs. To do this we will use
SageMakers Batch Transform functionality. Batch Transform is a convenient way to perform inference
on a large dataset in a way that is not realtime. That is, we don't necessarily need to use our
model's results immediately and instead we can perform inference on a large number of samples. An
example of this in industry might be performing an end of month report. This method of inference can
also be useful to us as it means that we can perform inference on our entire test set.

To perform a Batch Transformation we need to first create a transformer objects from our trained estimator object.

```python
xgb_transformer = xgb.transformer(instance_count = 1, instance_type = 'ml.m4.xlarge')
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

Now the transform job has executed and the result, the estimated sentiment of each review, has been
saved on S3. Since we would rather work on this file locally we can perform a bit of notebook magic
to copy the file to the `data_dir`.

```python
!aws s3 cp --recursive $xgb_transformer.output_path $data_dir
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

## Step 6: Deploying the model

Once we construct and fit our model, SageMaker stores the resulting model artifacts and we can use
those to deploy an endpoint (inference code). To see this, look in the SageMaker console and you
should see that a model has been created along with a link to the S3 location where the model
artifacts have been stored.

Deploying an endpoint is a lot like training the model with a few important differences. The first
is that a deployed model doesn't change the model artifacts, so as you send it various testing
instances the model won't change. Another difference is that since we aren't performing a fixed
computation, as we were in the training step or while performing a batch transform, the compute
instance that gets started stays running until we tell it to stop. This is important to note as if
we forget and leave it running we will be charged the entire time.

In other words **If you are no longer using a deployed endpoint, shut it down!**

```python
xgb_predictor = xgb.deploy(initial_instance_count = 1, instance_type = 'ml.m4.xlarge')
```

### Testing the model (again)

Now that we have deployed our endpoint, we can send the testing data to it and get back the
inference results. We already did this earlier using the batch transform functionality of SageMaker,
however, we will test our model again using the newly deployed endpoint so that we can make sure
that it works properly and to get a bit of a feel for how the endpoint works.

When using the created endpoint it is important to know that we are limited in the amount of
information we can send in each call so we need to break the testing data up into chunks and then
send each chunk. Also, we need to serialize our data before we send it to the endpoint to ensure
that our data is transmitted properly. Fortunately, SageMaker can do the serialization part for us
provided we tell it the format of our data.

```python
from sagemaker.predictor import csv_serializer

# We need to tell the endpoint what format the data we are sending is in so that SageMaker can perform the serialization.
xgb_predictor.content_type = 'text/csv'
xgb_predictor.serializer = csv_serializer
```

```python
# We split the data into chunks and send each chunk seperately, accumulating the results.

def predict(data, rows=512):
    split_array = np.array_split(data, int(data.shape[0] / float(rows) + 1))
    predictions = ''
    for array in split_array:
        predictions = ','.join([predictions, xgb_predictor.predict(array).decode('utf-8')])

    return np.fromstring(predictions[1:], sep=',')
```

```python
test_X = pd.read_csv(os.path.join(data_dir, 'test.csv'), header=None).values

predictions = predict(test_X)
predictions = [round(num) for num in predictions]
```

Lastly, we check to see what the accuracy of our model is.

```python
from sklearn.metrics import accuracy_score
accuracy_score(test_y, predictions)
```

And the results here should agree with the model testing that we did earlier using the batch
transform job.

### Cleaning up

Now that we've determined that deploying our model works as expected, we are going to shut it down.
Remember that the longer the endpoint is left running, the greater the cost and since we have a bit
more work to do before we are able to use our endpoint with our simple web app, we should shut
everything down.

```python
xgb_predictor.delete_endpoint()
```

## Step 7: Putting our model to work

As we've mentioned a few times now, our goal is to have our model deployed and then access it using
a very simple web app. The intent is for this web app to take some user submitted data (a review),
send it off to our endpoint (the model) and then display the result.

However, there is a small catch. Currently the only way we can access the endpoint to send it data
is using the SageMaker API. We can, if we wish, expose the actual URL that our model's endpoint is
receiving data from, however, if we just send it data ourselves we will not get anything in return.
This is because the endpoint created by SageMaker requires the entity accessing it have the correct
permissions. So, we would need to somehow authenticate our web app with AWS.

Having a website that authenticates to AWS seems a bit beyond the scope of this lesson so we will
opt for an alternative approach. Namely, we will create a new endpoint which does not require
authentication and which acts as a proxy for the SageMaker endpoint.

As an additional constraint, we will try to avoid doing any data processing in the web app itself.
Remember that when we constructed and tested our model we started with a movie review, then we
simplified it by removing any html formatting and punctuation, then we constructed a bag of words
embedding and the resulting vector is what we sent to our model. All of this needs to be done to our
user input as well.

Fortunately we can do all of this data processing in the backend, using Amazon's Lambda service.

<img src="Web App Diagram.svg">

The diagram above gives an overview of how the various services will work together. On the far right
is the model which we trained above and which will be deployed using SageMaker. On the far left is
our web app that collects a user's movie review, sends it off and expects a positive or negative
sentiment in return.

In the middle is where some of the magic happens. We will construct a Lambda function, which you can
think of as a straightforward Python function that can be executed whenever a specified event
occurs. This Python function will do the data processing we need to perform on a user submitted
review. In addition, we will give this function permission to send and recieve data from a SageMaker
endpoint.

Lastly, the method we will use to execute the Lambda function is a new endpoint that we will create
using API Gateway. This endpoint will be a url that listens for data to be sent to it. Once it gets
some data it will pass that data on to the Lambda function and then return whatever the Lambda
function returns. Essentially it will act as an interface that lets our web app communicate with the
Lambda function.

### Processing a single review

For now, suppose we are given a movie review by our user in the form of a string, like so:

```python
test_review = "Nothing but a disgusting materialistic pageant of glistening abed remote control greed zombies, totally devoid of any heart or heat. A romantic comedy that has zero romantic chemestry and zero laughs!"
```

How do we go from this string to the bag of words feature vector that is expected by our model?

If we recall at the beginning of this notebook, the first step is to remove any unnecessary
characters using the `review_to_words` method. Remember that we intentionally did this in a very
simplistic way. This is because we are going to have to copy this method to our (eventual) Lambda
function (we will go into more detail later) and this means it needs to be rather simplistic.

```python
test_words = review_to_words(test_review)
print(test_words)
```

Next, we need to construct a bag of words embedding of the `test_words` string. To do this, remember
that a bag of words embedding uses a `vocabulary` consisting of the most frequently appearing words
in a set of documents. Then, for each word in the vocabulary we record the number of times that word
appears in `test_words`. We constructed the `vocabulary` earlier using the training set for our
problem so encoding `test_words` is relatively straightforward.

```python
def bow_encoding(words, vocabulary):
    bow = [0] * len(vocabulary) # Start by setting the count for each word in the vocabulary to zero.
    for word in words.split():  # For each word in the string
        if word in vocabulary:  # If the word is one that occurs in the vocabulary, increase its count.
            bow[vocabulary[word]] += 1
    return bow
```

```python
test_bow = bow_encoding(test_words, vocabulary)
print(test_bow[:10])
```

```python
len(test_bow)
```

So now we know how to construct a bag of words encoding of a user provided review, how to we send it
to our endpoint? First, we need to start the endpoint back up.

```python
xgb_predictor = xgb.deploy(initial_instance_count = 1, instance_type = 'ml.m4.xlarge')
```

At this point we could just do the same thing that we did earlier when we tested our deployed model
and send `test_bow` to our endpoint using the `xgb_predictor` object. However, when we eventually
construct our Lambda function we won't have access to this object, so how do we call a SageMaker
endpoint?

It turns out that Python functions that are used in Lambda have access to another Amazon library
called `boto3`. This library provides an API for working with Amazon services, including SageMaker.
To start with, we need to get a handle to the SageMaker runtime.

```python
import boto3

runtime = boto3.Session().client('sagemaker-runtime')
```

And now that we have access to the SageMaker runtime, we can ask it to make use of (invoke) an
endpoint that has already been created. However, we need to provide SageMaker with the name of the
deployed endpoint. To find this out we can print it out using the `xgb_predictor` object.

```python
xgb_predictor.endpoint
```

Using the SageMaker runtime and the name of our endpoint, we can invoke the endpoint and send it
the `test_bow` data.

```python
response = runtime.invoke_endpoint(EndpointName = xgb_predictor.endpoint, # The name of the endpoint we created
                                       ContentType = 'text/csv', # The data format that is expected
                                       Body = test_bow)
```

So why did we get an error?

Because we tried to send the endpoint a list of integers but it expected us to send data of
type `text/csv`. So, we need to convert it.

```python
response = runtime.invoke_endpoint(EndpointName = xgb_predictor.endpoint, # The name of the endpoint we created
                                       ContentType = 'text/csv', # The data format that is expected
                                       Body = ','.join([str(val) for val in test_bow]).encode('utf-8'))
```

```python
print(response)
```

As we can see, the response from our model is a somewhat complicated looking dict that contains a
bunch of information. The bit that we are most interested in is `'Body'` object which is a streaming
object that we need to `read` in order to make use of.

```python
response = response['Body'].read().decode('utf-8')
print(response)
```

Now that we know how to process the incoming user data we can start setting up the infrastructure to
make our simple web app work. To do this we will make use of two different services. Amazon's Lambda
and API Gateway services.

Lambda is a service which allows someone to write some relatively simple code and have it executed
whenever a chosen trigger occurs. For example, you may want to update a database whenever new data
is uploaded to a folder stored on S3.

API Gateway is a service that allows you to create HTTP endpoints (url addresses) which are
connected to other AWS services. One of the benefits to this is that you get to decide what
credentials, if any, are required to access these endpoints.

In our case we are going to set up an HTTP endpoint through API Gateway which is open to the public.
Then, whenever anyone sends data to our public endpoint we will trigger a Lambda function which will
send the input (in our case a review) to our model's endpoint and then return the result.

### Setting up a Lambda function

The first thing we are going to do is set up a Lambda function. This Lambda function will be
executed whenever our public API has data sent to it. When it is executed it will receive the data,
perform any sort of processing that is required, send the data (the review) to the SageMaker
endpoint we've created and then return the result.

#### Part A: Create an IAM Role for the Lambda function

Since we want the Lambda function to call a SageMaker endpoint, we need to make sure that it has
permission to do so. To do this, we will construct a role that we can later give the Lambda function.

Using the AWS Console, navigate to the **IAM** page and click on **Roles**. Then, click on
**Create role**. Make sure that the **AWS service** is the type of trusted entity selected and
choose **Lambda** as the service that will use this role, then click **Next: Permissions**.

In the search box type `sagemaker` and select the check box next to the
**AmazonSageMakerFullAccess** policy. Then, click on **Next: Review**.

Lastly, give this role a name. Make sure you use a name that you will remember later on, for example
`LambdaSageMakerRole`. Then, click on **Create role**.

#### Part B: Create a Lambda function

Now it is time to actually create the Lambda function. Remember from earlier that in order to
process the user provided input and send it to our endpoint we need to gather two pieces of
information:

- The name of the endpoint, and
- the vocabulary object.

We will copy these pieces of information to our Lambda function after we create it.

To start, using the AWS Console, navigate to the AWS Lambda page and click on **Create a function**.
When you get to the next page, make sure that **Author from scratch** is selected. Now, name your
Lambda function, using a name that you will remember later on, for example
`sentiment_analysis_xgboost_func`. Make sure that the **Python 3.6** runtime is selected and then
choose the role that you created in the previous part. Then, click on **Create Function**.

On the next page you will see some information about the Lambda function you've just created. If you
scroll down you should see an editor in which you can write the code that will be executed when your
Lambda function is triggered. Collecting the code we wrote above to process a single review and
adding it to the provided example `lambda_handler` we arrive at the following.

```python
# We need to use the low-level library to interact with SageMaker since the SageMaker API
# is not available natively through Lambda.
import boto3

# And we need the regular expression library to do some of the data processing
import re

REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

def review_to_words(review):
    words = REPLACE_NO_SPACE.sub("", review.lower())
    words = REPLACE_WITH_SPACE.sub(" ", words)
    return words

def bow_encoding(words, vocabulary):
    bow = [0] * len(vocabulary) # Start by setting the count for each word in the vocabulary to zero.
    for word in words.split():  # For each word in the string
        if word in vocabulary:  # If the word is one that occurs in the vocabulary, increase its count.
            bow[vocabulary[word]] += 1
    return bow


def lambda_handler(event, context):

    vocab = "*** ACTUAL VOCABULARY GOES HERE ***"

    words = review_to_words(event['body'])
    bow = bow_encoding(words, vocab)

    # The SageMaker runtime is what allows us to invoke the endpoint that we've created.
    runtime = boto3.Session().client('sagemaker-runtime')

    # Now we use the SageMaker runtime to invoke our endpoint, sending the review we were given
    response = runtime.invoke_endpoint(EndpointName = '***ENDPOINT NAME HERE***',# The name of the endpoint we created
                                       ContentType = 'text/csv',                 # The data format that is expected
                                       Body = ','.join([str(val) for val in bow]).encode('utf-8')) # The actual review

    # The response is an HTTP response whose body contains the result of our inference
    result = response['Body'].read().decode('utf-8')

    # Round the result so that our web app only gets '1' or '0' as a response.
    result = round(float(result))

    return {
        'statusCode' : 200,
        'headers' : { 'Content-Type' : 'text/plain', 'Access-Control-Allow-Origin' : '*' },
        'body' : str(result)
    }
```

Once you have copy and pasted the code above into the Lambda code editor, replace the
`**ENDPOINT NAME HERE**` portion with the name of the endpoint that we deployed earlier. You can
determine the name of the endpoint using the code cell below.

```python
xgb_predictor.endpoint
```

In addition, you will need to copy the vocabulary dict to the appropriate place in the code at the
beginning of the `lambda_handler` method. The cell below prints out the vocabulary dict in a way
that is easy to copy and paste.

```python
print(str(vocabulary))
```

Once you have added the endpoint name to the Lambda function, click on **Save**. Your Lambda
function is now up and running. Next we need to create a way for our web app to execute the Lambda
function.

### Setting up API Gateway

Now that our Lambda function is set up, it is time to create a new API using API Gateway that will
trigger the Lambda function we have just created.

Using AWS Console, navigate to **Amazon API Gateway** and then click on **Get started**.

On the next page, make sure that **New API** is selected and give the new api a name, for example,
`sentiment_analysis_web_app`. Then, click on **Create API**.

Now we have created an API, however it doesn't currently do anything. What we want it to do is to
trigger the Lambda function that we created earlier.

Select the **Actions** dropdown menu and click **Create Method**. A new blank method will be
created, select its dropdown menu and select **POST**, then click on the check mark beside it.

For the integration point, make sure that **Lambda Function** is selected and click on the
**Use Lambda Proxy integration**. This option makes sure that the data that is sent to the API is
then sent directly to the Lambda function with no processing. It also means that the return value
must be a proper response object as it will also not be processed by API Gateway.

Type the name of the Lambda function you created earlier into the **Lambda Function** text entry box
and then click on **Save**. Click on **OK** in the pop-up box that then appears, giving permission
to API Gateway to invoke the Lambda function you created.

The last step in creating the API Gateway is to select the **Actions** dropdown and click on
**Deploy API**. You will need to create a new Deployment stage and name it anything you like, for
example `prod`.

You have now successfully set up a public API to access your SageMaker model. Make sure to copy or
write down the URL provided to invoke your newly created public API as this will be needed in the
next step. This URL can be found at the top of the page, highlighted in blue next to the
text **Invoke URL**.

## Step 7: Deploying our web app

Now that we have a publicly available API, we can start using it in a web app. For our purposes, we
have provided a simple static html file which can make use of the public api you created earlier.

In the `website` folder there should be a file called `index.html`. Download the file to your
computer and open that file up in a text editor of your choice. There should be a line which
contains **\*\*REPLACE WITH PUBLIC API URL\*\***. Replace this string with the url that you wrote
down in the last step and then save the file.

Now, if you open `index.html` on your local computer, your browser will behave as a local web server
and you can use the provided site to interact with your SageMaker model.

If you'd like to go further, you can host this html file anywhere you'd like, for example using
github or hosting a static site on Amazon's S3. Once you have done this you can share the link with
anyone you'd like and have them play with it too!

> **Important Note** In order for the web app to communicate with the SageMaker endpoint, the
> endpoint has to actually be deployed and running. This means that you are paying for it. Make
> sure that the endpoint is running when you want to use the web app but that you shut it down when
> you don't need it, otherwise you will end up with a surprisingly large AWS bill.

### Delete the endpoint

Remember to always shut down your endpoint if you are no longer using it. You are charged for the
length of time that the endpoint is running so if you forget and leave it on you could end up with
an unexpectedly large bill.

```python
xgb_predictor.delete_endpoint()
```

## Optional: Clean up

The default notebook instance on SageMaker doesn't have a lot of excess disk space available. As you
continue to complete and execute notebooks you will eventually fill up this disk space, leading to
errors which can be difficult to diagnose. Once you are completely finished using a notebook it is
a good idea to remove the files that you created along the way. Of course, you can do this from the
terminal or from the notebook hub if you would like. The cell below contains some commands to clean
up the created files from within the notebook.

```python
# First we will remove all of the files contained in the data_dir directory
!rm $data_dir/*

# And then we delete the directory itself
!rmdir $data_dir

# Similarly we remove the files in the cache_dir directory and the directory itself
!rm $cache_dir/*
!rmdir $cache_dir
```
