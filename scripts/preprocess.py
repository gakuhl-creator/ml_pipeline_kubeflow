import joblib
import argparse
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def main(preprocessor_path):
    # Define numeric and categorical features
    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaperlessBilling", "PaymentMethod"
    ]

    # Numeric feature transformer (Impute and Scale)
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    # Categorical feature transformer (Impute and Encode)
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop='if_binary'))
    ])

    # Combine transformations into a full preprocessing pipeline
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ], remainder='passthrough', verbose_feature_names_out=False)

    # Ensuring the output is a pandas DataFrame
    preprocessor.set_output(transform="pandas")

    # Save preprocessor
    joblib.dump(preprocessor, preprocessor_path)    

    return preprocessor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocessor_path', required=True, help="Path to the preprocessor pipeline")
    args = parser.parse_args()

    # Call the preprocessing function
    preprocessor_path = main(args.preprocessor_path)
    print(f"Preprocessor saved to: {preprocessor_path}")
