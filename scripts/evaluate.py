import pandas as pd
import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, r2_score

def evaluate(model_path, X_test_path, y_test_path, conf_matrix_output_path):
    model = joblib.load(model_path)
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)

    experiment_name = "churn_prediction"
    mlflow_tracking_uri = "http://127.0.0.1:5000"

    # Predict
    y_pred = model.predict(X_test)

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    # Calculate error metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    # Assemble the metrics we're going to write into a collection
    metrics = {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2, "accuracy", accuracy}

    params = {
        "max_iter": 1000
    }

    experiment = mlflow.set_experiment(experiment_name)
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # Initiate the MLflow run context
    with mlflow.start_run() as run:
        # Log the parameters used for the model fit
        mlflow.log_params(params)

        # Log the error metrics that were calculated during validation
        mlflow.log_metrics(metrics)

        # Log an instance of the trained model for later use
        mlflow.sklearn.log_model(sk_model=model, input_example=X_test, name=experiment_name)

    plt.savefig(conf_matrix_output_path)
    plt.close()

    return conf_matrix_output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help="Path to model")
    parser.add_argument('--X_test_path', required=True, help="Path to X test data")
    parser.add_argument('--y_test_path', required=True, help="Path to y test data")
    parser.add_argument('--conf_matrix_output_path', required=True, help="Path to store confusion matrix")
    args = parser.parse_args()

    evaluation = main(args.model_path, args.X_test_path, args.y_test_path)
    print(f"confusion matrix saved to {evaluation}")