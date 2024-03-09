# Boxer Detection SageMaker Endpoint Creation

First create a conda environment, clone this repository, then `cd host-yolov8-on-sagemaker-endpoint` and `pip install -r requirements.txt`.

## Install SageMaker notebook instance
```
$ cd yolov8-pytorch-cdk
$ pip3 install -r requirements.txt
$ cdk synth
$ cdk bootstrap
$ cdk deploy
```

## YOLOv8 PyTorch model deployment on Amazon SageMaker Endpoints:
- From AWS Console, go to [Amazon SageMaker Notebook Instances](https://us-east-1.console.aws.amazon.com/sagemaker/home?region=eu-west-2#/notebook-instances)
- Select the Notebook created by the stack and open it
- Inside SageMaker Notebook, navigate: `sm-notebook` and open the notebooks: `1_DeployEndpoint.ipynb` & `2_TestEndpoint.ipynb`
    1. `1_DeployEndpoint.ipynb`: Upload the model to the notebook root directory, run zip inference code and model to S3, create SageMaker endpoint and deploy it
    2. `2_TestEndpoint.ipynb`: Test the deployed endpoint by running an image and plotting output; Cleanup the endpoint and hosted model

## Test from python code
`python test_inference.py <endpoint-name>`
