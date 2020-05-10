import os
import io
import boto3
import json
import csv
from collections import Counter
import sys
import email
import quopri
sys.path.insert(1, '/opt')
from sklearn.feature_extraction.text import TfidfVectorizer

# grab environment variables
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
runtime= boto3.client('runtime.sagemaker')

def lambda_handler(event, context):
    #print("Received event: " + json.dumps(event, indent=2))
    
    
    # In order to d encode the raw data from ses
    # We use the same dataset which we used to train the model to calculate tf-idf as the input
    # Read the dataset from S3 bucket
    s3 = boto3.client('s3')
    csvfile = s3.get_object(Bucket='nyu-cc-final', Key='mycsv.csv')
    csvcontent = csvfile['Body'].read().split(b'\n')
    
    # Fit the tfidf model
    tfidf_model = TfidfVectorizer().fit(csvcontent)
    
    # Extract the data from ses
    # data = json.loads(json.dumps(event))
    # payload = data['data']
    s3 = boto3.client('s3')
    bucket = 'nyu-cc-final'
    msgid = event['Records'][0]['ses']['mail']['messageId']
    email_response = s3.get_object(
        Bucket=bucket,
        Key=msgid
    )
    
    email_body = ""
    content = email_response["Body"].read().decode()
    b = email.message_from_string(content)
    
    if b.is_multipart():
        for payload in b.get_payload():
            # if payload.is_multipart(): ...
            email_body = email_body + str(payload)
    else:
        email_body = email_body + b.get_payload()
    source = b["From"]
    to = b["To"]
    subject = b["Subject"]
    
    payload = email_body
    
    
    
    temp = payload.split(' ')

    # Encode the data   
    sparse_res = tfidf_model.transform(temp)
    array_res = sparse_res.toarray()[0]
    
    # Transform the format for model input 
    res= []
    for i in range(len(array_res)):
        res.append(str(array_res[i]))
    res = res[:7743]
    res_str = ','.join(res)

    # Invoke sagemaker
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                       ContentType='text/csv',
                                       Body=res_str)
    
    print(response)
    # Extract predict score from the response
    result = json.loads(response['Body'].read().decode())
    print(result)
    
    # Label the mail
    #pred = float(result['predictions'][0]['score'])
    predicted_label = 'Spam' if result > 0.5 else 'Ham'
    print(predicted_label)
    
    ##########from event
    send_payload = dict()
    send_payload['EMAIL_RECEIVE_DATE'] = event['Records'][0]['ses']['mail']['timestamp']
    send_payload['EMAIL_SUBJECT'] = subject
    send_payload['From'] = source
    send_payload['To'] = to
    
    send_body = quopri.decodestring(email_body, header=False)
    ########## from s3 object
    if len(send_body) >= 240:
        send_payload['EMAIL_BODY'] = send_body.decode('utf-8')[0:240]
    else:
        send_payload['EMAIL_BODY'] = send_body.decode('utf-8')
    
    ######### from sagemaker
    send_payload['CLASSIFICATION'] = predicted_label
    send_payload['CLASSIFICATION_CONFIDENCE_SCORE'] = result

    
    #print(payload)
    
    send_email(send_payload)
    
def send_email(payload):
    ses = boto3.client("ses")
    
    resp = 'We received your email sent at %s with the subject %s.\n\n' % (payload['EMAIL_RECEIVE_DATE'], payload['EMAIL_SUBJECT'])
    resp += 'Here is a 240 character sample of the email body:\n%s\n\n' % payload['EMAIL_BODY']
    resp += 'The email was categorized as %s with a %s %% confidence.\n\n' % (payload['CLASSIFICATION'], payload['CLASSIFICATION_CONFIDENCE_SCORE'])
    
    response = ses.send_email(
        Source=payload['To'],
        Destination={
            'ToAddresses': [
                payload['From'],
            ],
        },
        Message={
            'Subject': {
                'Data': 'Your Spam Recognition Result',
                'Charset': 'UTF-8'
            },
            'Body': {
                'Text': {
                    'Data': resp,
                    'Charset': 'UTF-8'
                },
            }
        },
    )
