import joblib
import yaml
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from src.features.preprocess import get_preprocessor  # Import the function that returns the preprocessor

def main(preprocessor_path, clean_data, model_output_path, X_test_path, y_test_path):
    # load preprocessor
    preprocessor = joblib.load(preprocessor_path)

    # Load clean data (this is the untransformed data)
    df = pd.read_csv(clean_data)

    # Split the data into features (X) and target (y)
    target_column = "Churn"
    y = df[target_column].map({'Yes': 1, 'No': 0})
    X = df.drop(columns=[target_column])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create the pipeline with both preprocessing and model fitting steps
    pipeline = Pipeline([
        ("preprocessor", preprocessor),  # Preprocessing step (includes added features)
        ("classifier", LogisticRegression(max_iter=1000))  # Estimator (Logistic Regression)
    ])

    # Fit the pipeline with training data (includes preprocessing)
    pipeline.fit(X_train, y_train)

    # Save the trained pipeline (including the model and the preprocessor)
    joblib.dump(pipeline, model_output_path)

    # Save the train_test_splitstest files for evaluation
    X_test.to_csv(X_test_path, index=False)
    y_test.to_csv(y_test_path, index=False)

    return model_output_path, X_test_path, y_test_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocessor', required=True, help="Path to processor")
    parser.add_argument('--clean_data', required=True, help="Path to the clean data")

    parser.add_argument('--model_output_path', required=True, help="Path where the model will be stored")
    parser.add_argument('--x_test_path', required=True, help="Path where X_test data will be stored after train_test_split")
    parser.add_argument('--y_test_path', required=True, help="Path twhere the y_test data will be stored after train_test_split")
    args = parser.parse_args()

    model_output_path, X_test_path, y_test_path = main(args.preprocessor, args.clean_data, args.model_output_path, args.x_test_path, args.y_test_path)
    print(f"Trained model saved to {model_output_path}, X_test data saved to {X_test_path}, and y_test data saved to {y_test_path}")
