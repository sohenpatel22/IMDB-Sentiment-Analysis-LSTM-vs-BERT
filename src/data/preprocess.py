import re
from collections import Counter

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))


def load_data(path):
    return pd.read_csv(path)


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_review(text, remove_stopwords=False):
    words = clean_text(text).split()

    if remove_stopwords:
        words = [word for word in words if word not in STOP_WORDS]

    return words


def get_word_count(text):
    return len(str(text).split())


def build_vocab(reviews, vocab_size=5000, remove_stopwords=False):
    all_words = []

    for review in reviews:
        all_words.extend(preprocess_review(review, remove_stopwords=remove_stopwords))

    word_counts = Counter(all_words)
    common_words = word_counts.most_common(vocab_size)

    stoi = {"<PAD>": 0, "<UNK>": 1}
    for i, (word, _) in enumerate(common_words):
        stoi[word] = i + 2

    return stoi


def encode_review(review, stoi, remove_stopwords=False):
    words = preprocess_review(review, remove_stopwords=remove_stopwords)
    return [stoi.get(word, stoi["<UNK>"]) for word in words]


def encode_labels(labels):
    return np.array([1 if label == "positive" else 0 for label in labels])


def pad_features(reviews_int, seq_length=500):
    features = np.zeros((len(reviews_int), seq_length), dtype=int)

    for i, review in enumerate(reviews_int):
        if len(review) == 0:
            continue

        if len(review) <= seq_length:
            features[i, -len(review):] = np.array(review)
        else:
            features[i, :] = np.array(review[:seq_length])

    return features


def prepare_lstm_data(
    df,
    vocab_size=5000,
    seq_length=500,
    random_state=42,
    remove_stopwords=False
):
    
    X = np.asarray(df["review"].astype(str).tolist(), dtype=object)
    y = np.asarray(df["sentiment"].tolist())

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
    )

    stoi = build_vocab(X_train, vocab_size=vocab_size, remove_stopwords=remove_stopwords)

    X_train_encoded = [encode_review(review, stoi, remove_stopwords=remove_stopwords) for review in X_train]
    X_val_encoded = [encode_review(review, stoi, remove_stopwords=remove_stopwords) for review in X_val]
    X_test_encoded = [encode_review(review, stoi, remove_stopwords=remove_stopwords) for review in X_test]

    X_train_padded = pad_features(X_train_encoded, seq_length=seq_length)
    X_val_padded = pad_features(X_val_encoded, seq_length=seq_length)
    X_test_padded = pad_features(X_test_encoded, seq_length=seq_length)

    y_train_encoded = encode_labels(y_train)
    y_val_encoded = encode_labels(y_val)
    y_test_encoded = encode_labels(y_test)

    return {
        "X_train": X_train_padded,
        "X_val": X_val_padded,
        "X_test": X_test_padded,
        "y_train": y_train_encoded,
        "y_val": y_val_encoded,
        "y_test": y_test_encoded,
        "raw_test_reviews": X_test,
        "raw_test_labels": y_test
    }, stoi