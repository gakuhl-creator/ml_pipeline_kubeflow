import kfp
import os
from kfp import dsl
from dotenv import load_dotenv
from deploy_utils import generate_inference_service_yaml, deploy_model_yaml

# Load environment variables from the .env file
load_dotenv()

clean_data_path = os.get("CLEAN_DATA_PATH", "data/clean_input.csv")
preprocessor_path = os.get("PROCESSSOR_PATH", "data/model_pipeline.pkl")
model_output_path - os.get("MODEL_OUTPUT_PATH", "data/fitted_model_pipeline.pkl")
X_test_path = os.get("X_TEST_PATH", "data/X_test.csv")
y_test_path = os.get("y_TEST_PATH", "data/y_test.csv")
conf_matrix_output_path = os.get("CONF_MATRIX_OUTPUT_PATH", "data/conf_matrix.png")

# Define components for loading, preprocessing, training, evaluation, and prediction
@dsl.python_component
def load_op(raw_path: str, clean_data_path: str) -> (str):
    """clean raw data and save"""
    import subprocess
    subprocess.run(["python", "load.py", "--raw_path", raw_path, "--clean_data_path", clean_data_path])
    return clean_data_path

@dsl.python_component
def preprocess_op(preprocessor_path: str) -> (str):
    """save preprocessor"""
    import subprocess
    subprocess.run(["python", "preprocess.py", "--preprocessor_path", preprocessor_path])
    return preprocessor_path

@dsl.python_component
def train_op(preprocessor: str, preprocessed_data: str, model_output_path: str, x_test_path: str, y_test_path: str) -> (str, str, str):
    import subprocess
    subprocess.run(["python", "train.py", "--preprocessor", preprocessor, "--data", preprocessed_data, "--model_output_path", model_output_path, "--x_test_path", x_test_path, "--y_test_path", y_test_path])
    return model_output_path, x_test_path, y_test_path

@dsl.python_component
def evaluate_op(model: str, conf_matrix_output_path: str):
    import subprocess
    subprocess.run(["python", "evaluate.py", "--model_path", model, "--X_test_path", X_test_path, "--y_test_path", y_test_path, "--conf_matrix_output_path", conf_matrix_output_path])
    return conf_matrix_output_path

@dsl.python_component
def predict_op(model_path: str, data_path: str, output_path: str) -> str:
    import subprocess
    subprocess.run(["python", "predict.py", "--model", model_path, "--data", data_path, "--output", output_path])
    return output_path        

@dsl.python_component
def deploy_model_op(model_path: str, model_name: str) -> str:
    """Deploy the trained model to KFServing and expose it as an API endpoint."""
    
    # Generate the KFServing InferenceService YAML
    inference_service_yaml = generate_inference_service_yaml(model_path, model_name)
    
    # Deploy the model using kubectl
    return deploy_model_yaml(inference_service_yaml, model_name)

# Define the pipeline with dependencies
@dsl.pipeline(
    name="ML Pipeline",
    description="Pipeline for ML training and prediction"
)
def ml_pipeline(data_path: str, predict_data_path: str, model_name: str):
    load = load_op(raw_path=data_path).set_image('load:latest')
    preprocess = preprocess_op(preprocessor_path=preprocessor_path).set_image('preprocess:latest')
    train = train_op(
        preprocessor_path=preprocess.output,
        preprocessed_data=load.output,
        model_output_path=model_output_path,
        x_test_path=X_test_path,
        y_test_path=y_test_path
    ).set_image('train:latest')
    evaluate = evaluate_op(
        model_path=train.output[0],
        X_test_path=train.output[1],
        y_test_path=train.output[2],
        conf_matrix_output_path=conf_matrix_output_path
    ).set_image('evaluate:latest')
    deploy = deploy_model_op(model_path=train.output[0], model_name=model_name).set_image("deploy:latest")
    predict = predict_op(
        model_path=train.output[0],
        data_path=load.output,
        output_path=predict_data_path
    ).set_image('predict:latest')

# Compile the pipeline
kfp.compiler.Compiler().compile(ml_pipeline, 'ml_pipeline.yaml')
