import pandas as pd
from insurance_cost_predicter.logger import logger

def load_data(file_path):
    logger.info(f"Loading dataset from: {file_path}")
    df = pd.read_csv(file_path)
    print("\nDataset Loaded Successfully")
    print("Initial Shape:", df.shape)
    return df

def clean_data(df):
    logger.info("Handling duplicate records")
    initial_rows = len(df)
    df = df.drop_duplicates()
    print(f"Removed {initial_rows - len(df)} duplicate rows.")

    logger.info("Handling missing values")
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Numerical columns: fill with median
            df[col] = df[col].fillna(df[col].median())
        else:
            # String/Object/Categorical columns: fill with mode
            mode_vals = df[col].mode()
            if not mode_vals.empty:
                df[col] = df[col].fillna(mode_vals[0])
            
    print("Missing values handled.")
    return df