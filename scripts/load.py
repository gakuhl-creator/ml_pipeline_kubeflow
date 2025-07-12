import argparse
import pandas as pd

def load_data(raw_path, clean_data_path):
    # Load raw data
    df = pd.read_csv(raw_path)

    # Clean up any non-numeric values that may exist in numeric columns (convert blank spaces or text to NaN)
    numeric_columns = ["tenure", "MonthlyCharges", "TotalCharges"]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce') 

    # Drop rows with missing values in key columns
    df = df.dropna(subset=["tenure", "MonthlyCharges", "TotalCharges", config["target_column"]])

    df = df.drop(columns=['customerID'])

    # Save cleaned data
    df.to_csv(clean_data_path, index=False)
    return clean_data_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_path', required=True, help="Path to the raw data")
    parser.add_argument('--clean_data_path', required=True, help="Path for storing cleaned data")
    args = parser.parse_args()

    # Call the preprocessing function
    clean_data = main(args.raw_path, args.clean_data_path)
    print(f"Clean data saved to: {clean_data}")