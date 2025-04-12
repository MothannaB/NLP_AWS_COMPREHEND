import time
import boto3
import pandas as pd
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

def create_classifier(training_data_s3_uri, role_arn):
    # cthe boto3 client automatically detects credentials I  stored in the AWS configuration file  named config
    comprehend = boto3.client('comprehend')
    
    # Create the custom classifier
    response = comprehend.create_document_classifier(
        InputDataConfig={
            'S3Uri': training_data_s3_uri,
            'InputFormat': 'CSV'  # Specify CSV format for the dataset
        },
        OutputDataConfig={
            'S3Uri': 's3://nlpbucket-mothanna/output/'  # Specify the S3 location for output data
        },
        DataAccessRoleArn=role_arn,  # ARN of the IAM role that grants Comprehend access to the data
        LanguageCode='en',
        ClientRequestId=str(int(time.time()))  # Unique request ID for the operation
    )

    # Return the classifier ARN
    return response['DocumentClassifierArn']

# iam role was created using the AWS platform
role_arn = 'arn:aws:iam::097906948960:role/MothannaComprehendS3AccessRole'
training_data_s3_uri = 's3://nlpbucket-mothanna/job_offers_training.csv'
classifier_arn = create_classifier(training_data_s3_uri, role_arn)
print(f"Classifier ARN: {classifier_arn}")




def check_classifier_status(classifier_arn):
    comprehend = boto3.client('comprehend')
    
    while True:
        response = comprehend.describe_document_classifier(
            DocumentClassifierArn=classifier_arn
        )
        status = response['DocumentClassifierProperties']['Status']
        
        print(f"Training status: {status}")
        
        if status in ['TRAINED', 'FAILED']:
            break
        time.sleep(60)  # Check every 60 seconds

    return status

# Usage example
status = check_classifier_status(classifier_arn)
print(f"Training status: {status}")





def classify_text_from_s3(classifier_arn, bucket_name, file_name):
    # Initialize boto3 client
    session = boto3.Session()  # This will load credentials from ~/.aws/credentials file
    comprehend = session.client('comprehend', region_name='us-east-1')
    s3 = session.client('s3')

    try:
        # Download the test dataset from S3
        s3.download_file(bucket_name, file_name, '/tmp/' + file_name)
        print(f"File {file_name} downloaded from S3 bucket {bucket_name}")

        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv('/tmp/' + file_name)
        print(f"Read {len(df)} rows from the dataset.")

        # Iterate over each row in the DataFrame and classify the text
        results = []
        for index, row in df.iterrows():
            text = row['text']  # Assuming the column name containing the text is 'text'
            response = comprehend.batch_classify_document(
                TextList=[text],
                DocumentClassifierArn=classifier_arn
            )

            # Extract the classification result
            result = response['ResultList'][0]
            classes = result['Classes']  # List of classes with confidence scores

            # Append the result
            results.append({
                'text': text,
                'classification': classes
            })

        # Return the results as a DataFrame or a list of dicts
        result_df = pd.DataFrame(results)
        return result_df

    except (NoCredentialsError, PartialCredentialsError) as e:
        print("AWS credentials are not properly configured.")
        raise e
    except Exception as e:
        print(f"Error processing the test dataset: {e}")
        raise e
    


bucket_name = 'nlpbucket-mothanna'
# file hosted on AWS s3
file_name = 'test_offers.csv'
classifier_arn = 'arn:aws:comprehend:us-east-1:097906948960:document-classifier/JobOfferClassifier/version/version3'

result_df = classify_text_from_s3(classifier_arn, bucket_name, file_name)
print(result_df.head())