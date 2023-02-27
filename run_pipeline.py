import argparse
import json
import sys
import boto3
import datetime
import pandas as pd
from dateutil.tz import tzlocal

import sagemaker

from pipeline_build import get_pipeline
sagemaker = boto3.client('sagemaker')
s3 = boto3.client('s3')

def main(): 
    result = 0
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", type=str, required=True)
    parser.add_argument("--default-bucket", type=str, required=True)
    args = parser.parse_args()
    
    region = args.region
    default_bucket = args.default_bucket
    print(f'region: {region}')
    print(f'default_bucket: {default_bucket}')

    pipeline = get_pipeline(
        region = region,
        default_bucket=default_bucket
    )

    upsert_response = pipeline.upsert(role_arn={role_arn})
    print(upsert_response)

    execution = pipeline.start()
    execution.wait(delay=60, max_attempts=480)

if __name__ == "__main__":
    result = main()
    exit(result)