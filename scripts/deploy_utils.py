import subprocess

def generate_inference_service_yaml(model_path: str, model_name: str) -> str:
    """Generate the KFServing InferenceService YAML for deployment."""
    inference_service_yaml = f"""
    apiVersion: serving.kserve.io/v1beta1
    kind: InferenceService
    metadata:
      name: {model_name}
    spec:
      predictor:
        tensorflow:
          storageUri: {model_path}
          resources:
            requests:
              cpu: 100m
              memory: 1Gi
          runtimeVersion: "2.4.0"
    """
    return inference_service_yaml

def deploy_model_yaml(yaml_content: str, model_name: str):
    """Deploy the model using kubectl and the generated YAML."""
    yaml_filename = f"{model_name}-inferenceservice.yaml"
    
    # Write the YAML to a file
    with open(yaml_filename, "w") as yaml_file:
        yaml_file.write(yaml_content)

    # Use kubectl to deploy the model in the cluster using the generated YAML
    try:
        print(f"Deploying {model_name} to KFServing...")
        subprocess.run(['kubectl', 'apply', '-f', yaml_filename], check=True)
        print(f"Model {model_name} deployed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error deploying model {model_name}: {e}")
        raise
    return f"Model {model_name} deployed successfully"
