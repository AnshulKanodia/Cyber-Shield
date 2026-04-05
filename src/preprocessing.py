"""
Cyber Shield - Text Preprocessing Pipeline
NLP pipeline for cleaning and normalizing social media text data.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd
from src.utils import SLANG_DICT, CONTRACTIONS, get_logger, timer

logger = get_logger("preprocessing")

# Initialize NLTK components
lemmatizer = WordNetLemmatizer()

# Custom stopwords: keep negation words that are important for sentiment
NEGATION_WORDS = {
    "no", "not", "nor", "neither", "never", "none", "nobody",
    "nothing", "nowhere", "hardly", "barely", "scarcely",
    "don", "dont", "doesn", "doesnt", "didn", "didnt",
    "won", "wont", "wouldn", "wouldnt", "shouldn", "shouldnt",
    "couldn", "couldnt", "isn", "isnt", "aren", "arent",
    "wasn", "wasnt", "weren", "werent", "haven", "havent",
    "hasn", "hasnt", "hadn", "hadnt",
}

try:
    STOP_WORDS = set(stopwords.words("english")) - NEGATION_WORDS
except LookupError:
    nltk.download("stopwords", quiet=True)
    STOP_WORDS = set(stopwords.words("english")) - NEGATION_WORDS


def remove_urls(text: str) -> str:
    """Remove URLs from text."""
    return re.sub(r"http\S+|www\.\S+|https\S+", "", text, flags=re.MULTILINE)


def remove_mentions(text: str) -> str:
    """Remove @username mentions."""
    return re.sub(r"@\w+", "", text)


def remove_hashtag_symbol(text: str) -> str:
    """Remove # symbol but keep the hashtag text."""
    return re.sub(r"#", "", text)


def remove_emojis(text: str) -> str:
    """Remove emojis and special unicode characters."""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"
        "\u3030"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", text)


def remove_special_chars(text: str) -> str:
    """Remove special characters, keeping only alphanumeric and spaces."""
    return re.sub(r"[^a-zA-Z0-9\s]", "", text)


def remove_extra_whitespace(text: str) -> str:
    """Collapse multiple whitespace into single space and strip."""
    return re.sub(r"\s+", " ", text).strip()


def expand_contractions(text: str) -> str:
    """Expand contractions like can't → cannot."""
    words = text.split()
    expanded = []
    for word in words:
        lower_word = word.lower()
        if lower_word in CONTRACTIONS:
            expanded.append(CONTRACTIONS[lower_word])
        else:
            expanded.append(word)
    return " ".join(expanded)


def normalize_slang(text: str) -> str:
    """Replace internet slang and abbreviations with full words."""
    words = text.split()
    normalized = []
    for word in words:
        lower_word = word.lower()
        if lower_word in SLANG_DICT:
            normalized.append(SLANG_DICT[lower_word])
        else:
            normalized.append(word)
    return " ".join(normalized)


def remove_repeated_chars(text: str) -> str:
    """Reduce repeated characters to max 2 (e.g., 'haaappy' → 'haappy')."""
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


def tokenize_and_lemmatize(text: str) -> str:
    """Tokenize, remove stopwords, and lemmatize."""
    try:
        tokens = word_tokenize(text)
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
        tokens = word_tokenize(text)
    
    # Remove stopwords, lemmatize, and filter short tokens
    processed = []
    for token in tokens:
        if token.lower() not in STOP_WORDS and len(token) > 1:
            lemma = lemmatizer.lemmatize(token.lower())
            processed.append(lemma)
    
    return " ".join(processed)


def clean_text(text: str) -> str:
    """
    Full preprocessing pipeline for a single text.
    
    Pipeline order:
    1. Convert to string (handle NaN)
    2. Lowercase
    3. Remove URLs
    4. Remove mentions
    5. Remove hashtag symbols
    6. Remove emojis
    7. Expand contractions
    8. Normalize slang
    9. Remove repeated characters
    10. Remove special characters
    11. Remove extra whitespace
    12. Tokenize, remove stopwords, & lemmatize
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = remove_urls(text)
    text = remove_mentions(text)
    text = remove_hashtag_symbol(text)
    text = remove_emojis(text)
    text = expand_contractions(text)
    text = normalize_slang(text)
    text = remove_repeated_chars(text)
    text = remove_special_chars(text)
    text = remove_extra_whitespace(text)
    text = tokenize_and_lemmatize(text)
    
    return text


@timer
def preprocess_dataframe(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Apply full preprocessing pipeline to a DataFrame.
    
    Args:
        df: DataFrame with text column
        text_col: Name of the text column
    
    Returns:
        DataFrame with cleaned text in 'cleaned_text' column
    """
    logger.info(f"  Preprocessing {len(df)} texts...")
    
    # Apply cleaning
    df = df.copy()
    df["cleaned_text"] = df[text_col].apply(clean_text)
    
    # Remove empty texts after cleaning
    initial_len = len(df)
    df = df[df["cleaned_text"].str.len() > 0].reset_index(drop=True)
    removed = initial_len - len(df)
    if removed > 0:
        logger.info(f"  Removed {removed} empty texts after cleaning")
    
    logger.info(f"  Preprocessing complete: {len(df)} texts remaining")
    
    return df


if __name__ == "__main__":
    # Quick test
    test_texts = [
        "OMG u r such a loser lmaooo 😂😂 #pathetic @someone",
        "I can't believe how stupid you are tbh",
        "Great job on your presentation today! Very impressive.",
        "ur gonna regret this... stfu noob 🤡🤡🤡",
        "The weather is beautiful, let's go for a walk!",
    ]
    
    print("=" * 60)
    print("TEXT PREPROCESSING DEMO")
    print("=" * 60)
    for text in test_texts:
        cleaned = clean_text(text)
        print(f"\nOriginal:  {text}")
        print(f"Cleaned:   {cleaned}")
