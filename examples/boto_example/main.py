import boto3
import pandas as pd
import io
import pdb
import matplotlib.pyplot as plt


BUCKET = 'aws-ml-blog-sagemaker-census-segmentation'


def process_data_frame(df):
    print ('loaded data frame of shape {}'.format(df.shape))
    df_cleaned = df.dropna(axis=0, how='any')
    print('cleaned data frame has shape {}'.format(df_cleaned.shape))
    df_cleaned.index = df_cleaned['State'] + '-' + df_cleaned['County']
    df_cleaned.drop(columns=['CensusId', 'State', 'County'])
    print(df_cleaned.head())
    
    for column_name in ['Income', 'IncomePerCap']:
        ax = plt.subplots(figsize=(6,3))
        ax = plt.hist(df_cleaned[column_name], bins=50)
        plt.title('Histogram of {}'.format(column_name))
        plt.show()


def main():
    """
    Boto3 will look for AWS credentials in ~/.aws/credentials file.
    """
    s3_client = boto3.client('s3')

    object_keys = []
    object_list = s3_client.list_objects(Bucket=BUCKET)
    for contents in object_list['Contents']:
        object_keys.append(contents['Key'])

    print("found {} object key(s): {}".format(len(object_keys), object_keys))

    if len(object_keys) > 1:
        raise RuntimeError("S3 bucket returned expected number of keys")

    
    data_object = s3_client.get_object(Bucket=BUCKET, Key=object_keys[0])

    # Data object is a response from S3. This response contains a body.
    # The body is a botocore.response.StreamingBody. We need to use an
    # IO reader to read the bytes out as a data stream.
    data_body = data_object['Body'].read()
    print('data type for {} date body is {}'.format(object_keys[0], type(data_body)))

    data_stream = io.BytesIO(data_body)
    df = pd.read_csv(data_stream, header=0, delimiter=',')
    process_data_frame(df)


if __name__ == '__main__':
    main()
