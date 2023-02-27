import sys

import os
import json
import boto3
import sagemaker
import sagemaker.session

from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.estimator import Estimator
from sagemaker.processing import ScriptProcessor

from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.steps import TrainingStep
from sagemaker.workflow.properties import PropertyFile

from datetime import datetime

import pytz
import datetime
from dateutil.relativedelta import relativedelta

import pandas as pd

print("###get_session")
def get_session(region, default_bucket):
    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

print("###get_pipeline")
def get_pipeline(
    region,
    default_bucket=None
):
    # Pipeline Local Path
    BASE_DIR = os.path.dirname(os.path.realpath(__file__))
    print(f"BASE_DIR : {BASE_DIR}")
    
    # SageMaker Session
    sagemaker_session = get_session(region, default_bucket)
    # SageMaker Role
    role = sagemaker.session.get_execution_role(sagemaker_session)
    print(f"role : {role}")
    
    # Boto3 Client
    s3 = boto3.client("s3")

    # parameters for pipeline execution
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    processing_instance_type = ParameterString(name="ProcessingInstanceType", default_value="ml.m5.2xlarge")
    
    
    # Preprocessing Step
    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"sklearn-test-process",
        sagemaker_session=sagemaker_session,
        role=role,
    )

    step_process_1 = ProcessingStep(
        name="Preprocessing-test-1",
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(input_name="raw_data",
                            source=f"s3://{default_bucket}/data", 
                            destination="/opt/ml/processing/input"),
        ],
        outputs=[
            ProcessingOutput(output_name="train",
                             source="/opt/ml/processing/train", 
                             destination=f"s3://{default_bucket}/train"),
            ProcessingOutput(output_name="validation",
                             source="/opt/ml/processing/validation", 
                             destination=f"s3://{default_bucket}/validation"),
            ProcessingOutput(output_name="test",
                             source="/opt/ml/processing/test", 
                             destination=f"s3://{default_bucket}/test")
        ],
        code=f"{BASE_DIR}/preprocessing.py"
    )
    
    # 병렬 STEP 테스트용
    step_process_2 = ProcessingStep(
        name="Preprocessing-test-2",
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(input_name="raw_data",
                            source=f"s3://{default_bucket}/data", 
                            destination="/opt/ml/processing/input"),
        ],
        outputs=[
            ProcessingOutput(output_name="train",
                             source="/opt/ml/processing/train", 
                             destination=f"s3://{default_bucket}/train"),
            ProcessingOutput(output_name="validation",
                             source="/opt/ml/processing/validation", 
                             destination=f"s3://{default_bucket}/validation"),
            ProcessingOutput(output_name="test",
                             source="/opt/ml/processing/test", 
                             destination=f"s3://{default_bucket}/test")
        ],
        code=f"{BASE_DIR}/preprocessing.py"
    )
    
    # Training Step
    image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="1.0-1",
        py_version="py3",
        instance_type="ml.m5.xlarge"
    )
    
    xgb_train = Estimator(
        image_uri=image_uri,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        output_path=f"s3://{default_bucket}/model",
        role=role,
    )
    
    xgb_train.set_hyperparameters(
        objective="reg:linear",
        num_round=50,
        max_depth=5,
        eta=0.2,
        gamma=4,
        min_child_weight=6,
        subsample=0.7,
        silent=0
    )
    
    step_train = TrainingStep(
        name="Training-test",
        estimator=xgb_train,
        inputs={
            "train": TrainingInput(
                s3_data=step_process_1.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv"
            ),
            "validation": TrainingInput(
                s3_data=step_process_1.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
                content_type="text/csv"
            )
        },
        depends_on = [step_process_1, step_process_2]
    )
    
    # Evaluation Step
    script_eval = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type="ml.m5.xlarge",
        instance_count=1,
        base_job_name="script-test-eval",
        role=role,
    )
    
    # evaluation_report = PropertyFile(
    #     name="EvaluationReport",
    #     output_name="evaluation",
    #     path="evaluation.json"
    # )
    step_eval = ProcessingStep(
        name="Evaluation-test",
        processor=script_eval,
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model"
            ),
            ProcessingInput(
                source=f"s3://{default_bucket}/test",
                destination="/opt/ml/processing/test"
            )
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation", destination=f"s3://{default_bucket}/evaluation"),
        ],
        code=f"{BASE_DIR}/evaluation.py",
        # property_files=[evaluation_report],
        depends_on = [step_train]
    )
    
    # pipeline instance
    pipeline = Pipeline(
        name="sgmk-pipeline",
        parameters=[
            processing_instance_type, 
            processing_instance_count
        ],
        steps=[
            step_process_1,
            step_process_2,
            step_train,
            step_eval
        ],
        sagemaker_session=sagemaker_session,
    )    
    return pipeline
