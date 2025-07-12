import argparse
import joblib
import pandas as pd

def main(model_path, data, output):

    # Load the trained model
    model = joblib.load(model_path)

    # Load the cleaned data for prediction
    data = pd.read_csv(data)

    # Make predictions using the model
    predictions = model.predict(data)

    # Save predictions to a CSV file
    predictions_df = pd.DataFrame(predictions, columns=["predictions"])
    predictions_df.to_csv(output, index=False)

    print(f"Predictions saved to {output}")

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to the trained model")
    parser.add_argument("--data", required=True, help="Path to the data for prediction (CSV with cleaned data)")
    parser.add_argument("--output", required=True, help="Path to save the predictions")
    args = parser.parse_args()

    main(args.model, args.data, args.output)
