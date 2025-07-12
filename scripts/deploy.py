import argparse
from deploy_utils import generate_inference_service_yaml, deploy_model_yaml

def deploy_model(model_path: str, model_name: str):
    """Deploy the trained model to KFServing and expose it as an API endpoint."""
    
    # Generate the KFServing InferenceService YAML
    inference_service_yaml = generate_inference_service_yaml(model_path, model_name)
    
    # Deploy the model using kubectl
    return deploy_model_yaml(inference_service_yaml, model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help="Path to the model")
    parser.add_argument('--model_name', required=True, help="Name of the model")
    args = parser.parse_args()

    deploy_model(args.model_path, args.model_name)
