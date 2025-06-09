import numpy as np
import pandas as pd
import os
import re
import nltk
import string
import logging
from typing import List, Set, Union
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_nltk() -> None:
    """Download required NLTK data."""
    try:
        nltk.download('wordnet', quiet=True)
        nltk.download('stopwords', quiet=True)
        logger.info("NLTK data downloaded successfully")
    except Exception as e:
        logger.error(f"Failed to download NLTK data: {str(e)}")
        raise

def lemmatization(text: str, lemmatizer: WordNetLemmatizer = WordNetLemmatizer()) -> str:
    """Lemmatize the input text."""
    try:
        words = text.split()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        return " ".join(lemmatized_words)
    except Exception as e:
        logger.error(f"Error in lemmatization: {str(e)}")
        raise

def remove_stop_words(text: str, stop_words: Set[str] = set(stopwords.words("english"))) -> str:
    """Remove stop words from the input text."""
    try:
        words = str(text).split()
        filtered_words = [word for word in words if word not in stop_words]
        return " ".join(filtered_words)
    except Exception as e:
        logger.error(f"Error removing stop words: {str(e)}")
        raise

def removing_numbers(text: str) -> str:
    """Remove digits from the input text."""
    try:
        return ''.join([char for char in text if not char.isdigit()])
    except Exception as e:
        logger.error(f"Error removing numbers: {str(e)}")
        raise

def lower_case(text: str) -> str:
    """Convert text to lowercase."""
    try:
        words = text.split()
        lowercase_words = [word.lower() for word in words]
        return " ".join(lowercase_words)
    except Exception as e:
        logger.error(f"Error converting to lowercase: {str(e)}")
        raise

def removing_punctuations(text: str) -> str:
    """Remove punctuations and extra whitespace from the input text."""
    try:
        text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
        text = text.replace('؛', '')
        text = re.sub('\s+', ' ', text).strip()
        return text
    except Exception as e:
        logger.error(f"Error removing punctuations: {str(e)}")
        raise

def removing_urls(text: str) -> str:
    """Remove URLs from the input text."""
    try:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub('', text)
    except Exception as e:
        logger.error(f"Error removing URLs: {str(e)}")
        raise

def remove_small_sentences(df: pd.DataFrame, column: str = 'text') -> pd.DataFrame:
    """Remove sentences with fewer than 3 words by setting them to NaN."""
    try:
        for i in range(len(df)):
            if len(str(df[column].iloc[i]).split()) < 3:
                df.loc[i, column] = np.nan
        return df
    except Exception as e:
        logger.error(f"Error removing small sentences: {str(e)}")
        raise

def normalize_text(df: pd.DataFrame, column: str = 'content') -> pd.DataFrame:
    """Apply text normalization pipeline to the specified column of the DataFrame."""
    try:
        df[column] = df[column].apply(lambda x: lower_case(str(x)))
        df[column] = df[column].apply(remove_stop_words)
        df[column] = df[column].apply(removing_numbers)
        df[column] = df[column].apply(removing_punctuations)
        df[column] = df[column].apply(removing_urls)
        df[column] = df[column].apply(lemmatization)
        return df
    except Exception as e:
        logger.error(f"Error normalizing text: {str(e)}")
        raise

def normalized_sentence(sentence: str) -> str:
    """Normalize a single sentence using the full preprocessing pipeline."""
    try:
        sentence = lower_case(sentence)
        sentence = remove_stop_words(sentence)
        sentence = removing_numbers(sentence)
        sentence = removing_punctuations(sentence)
        sentence = removing_urls(sentence)
        sentence = lemmatization(sentence)
        return sentence
    except Exception as e:
        logger.error(f"Error normalizing sentence: {str(e)}")
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
    """Main function to preprocess and save train and test data."""
    try:
        setup_nltk()
        train_data = load_data("data/raw/train.csv")
        test_data = load_data("data/raw/test.csv")
        
        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)
        
        save_data(train_processed_data, os.path.join("data", "processed", "train_processed.csv"))
        save_data(test_processed_data, os.path.join("data", "processed", "test_processed.csv"))
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()