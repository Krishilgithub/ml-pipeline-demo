import numpy as np
import pandas as pd
import os
import yaml
import logging
from sklearn.model_selection import train_test_split

#* Logging Configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

#* Console Handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

#* File Handler
file_handler = logging.FileHandler('logs/data_ingestion.log')
file_handler.setLevel(logging.ERROR)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.info(f"Loaded parameters from {params_path}")
        return params
    except FileNotFoundError:
        # print(f"Error: The file '{params_path}' was not found.")
        logger.error(f"Error: The file '{params_path}' was not found.")
        exit(1)
    except yaml.YAMLError as e:
        # print(f"Error parsing YAML file: {e}")
        logger.error(f"Error parsing YAML file: {e}")
        exit(1)

def read_data(data_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        return df
    except FileNotFoundError:
        print(f"Error: The file '{data_path}' was not found.")
        exit(1)
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV file: {e}")
        exit(1)
    except Exception as e:
        print(f"Unexpected error reading data: {e}")
        exit(1)

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.drop(columns=['tweet_id'], inplace=True)
        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])].copy()
        final_df['sentiment'].replace({'happiness': 1, 'sadness': 0}, inplace=True)
        return final_df
    except KeyError as e:
        print(f"Error: Missing expected column in DataFrame - {e}")
        exit(1)
    except Exception as e:
        print(f"Unexpected error during data processing: {e}")
        exit(1)

def save_data(data_path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    try:
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)
    except OSError as e:
        print(f"Error creating or writing to directory '{data_path}': {e}")
        exit(1)
    except Exception as e:
        print(f"Unexpected error while saving data: {e}")
        exit(1)

def main() -> None:
    try:
        params = load_params("params.yaml")
        df = read_data("https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv")
        final_df = process_data(df)
        test_size = params.get('data_ingestion', {}).get('test_size', 0.2)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        data_path = os.path.join("data", "raw")
        save_data(data_path, train_data, test_data)
        print("Data processing and saving completed successfully.")
    except Exception as e:
        print(f"An unexpected error occurred in main: {e}")
        exit(1)

if __name__ == "__main__":
    main()