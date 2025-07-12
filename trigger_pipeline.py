import kfp
from kfp import Client
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

raw_data = os.getenv("RAW_DATA_PATH", "data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
prediction = os.getenv("PREDICTION_PATH", "data/predictions.csv")
model_name = os.getenv("MODEL_NAME", "Churn Prediction")

# Connect to the Kubeflow Pipelines API server
client = Client()

# Define the pipeline run (use the compiled pipeline function)
client.create_run_from_pipeline_func(
    ml_pipeline,  # Pipeline function defined in pipeline_definition.py
    arguments={
        "data_path": raw_data,  # Path to the raw input data
        "predict_data_path": prediction  # Path where predictions will be saved
        "model_name": model_name
    }
)
