import numpy as np
import pandas as pd
import os
import yaml
import logging
from typing import Tuple, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Successfully loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {str(e)}")
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load CSV data from the specified file path."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found")
        data = pd.read_csv(file_path)
        logger.info(f"Successfully loaded data from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {str(e)}")
        raise

def clean_data(df: pd.DataFrame, column: str = 'content') -> pd.DataFrame:
    """Fill NaN values in the specified column with empty strings."""
    try:
        df[column] = df[column].fillna('')
        logger.info(f"Successfully cleaned NaN values in {column} column")
        return df
    except Exception as e:
        logger.error(f"Error cleaning data: {str(e)}")
        raise

def apply_tfidf(X: np.ndarray, max_features: int) -> Tuple[np.ndarray, TfidfVectorizer]:
    """Apply Bag of Words transformation using CountVectorizer."""
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)
        X_tfidf = vectorizer.fit_transform(X)
        logger.info("Successfully applied Bag of Words transformation")
        return X_tfidf, vectorizer
    except Exception as e:
        logger.error(f"Error applying Bag of Words: {str(e)}")
        raise

def transform_tfidf(X: np.ndarray, vectorizer: TfidfVectorizer) -> np.ndarray:
    """Transform data using a fitted CountVectorizer."""
    try:
        X_tfidf = vectorizer.transform(X)
        logger.info("Successfully transformed data with Bag of Words")
        return X_tfidf
    except Exception as e:
        logger.error(f"Error transforming data with Bag of Words: {str(e)}")
        raise

def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save DataFrame to the specified file path."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.info(f"Successfully saved data to {file_path}")
    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {str(e)}")
        raise

def main() -> None:
    """Main function to perform feature engineering and save results."""
    try:
        params = load_config("params.yaml")
        max_features = params['feature_engineering']['max_features']
        
        train_data = load_data("data/processed/train_processed.csv")
        test_data = load_data("data/processed/test_processed.csv")
        
        train_data = clean_data(train_data)
        test_data = clean_data(test_data)
        
        X_train = train_data['content'].values
        y_train = train_data['sentiment'].values
        X_test = test_data['content'].values
        y_test = test_data['sentiment'].values
        
        X_train_tfidf, vectorizer = apply_tfidf(X_train, max_features)
        X_test_tfidf = transform_tfidf(X_test, vectorizer)
        
        train_df = pd.DataFrame(X_train_tfidf.toarray())
        train_df['label'] = y_train
        test_df = pd.DataFrame(X_test_tfidf.toarray())
        test_df['label'] = y_test
        
        save_data(train_df, os.path.join("data", "features", "train_tfidf.csv"))
        save_data(test_df, os.path.join("data", "features", "test_tfidf.csv"))
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()