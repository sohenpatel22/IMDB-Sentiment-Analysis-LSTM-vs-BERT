# IMDb Sentiment Analysis: LSTM vs BERT

This project compares three approaches for binary sentiment classification on IMDb movie reviews:

- TF-IDF / classical baseline (optional future extension)
- Word-level LSTM built in PyTorch
- Pretrained BERT classifier using Hugging Face Transformers

The goal is to understand the tradeoffs between traditional sequence models and modern transfer learning for NLP sentiment analysis.

## Project Highlights

- End-to-end text preprocessing pipeline
- Exploratory data analysis on review lengths and sentiment balance
- Word-level tokenization and padding for LSTM
- Transfer learning with `bert-base-cased`
- Training, validation, and test evaluation
- Precision, recall, F1-score, and confusion matrix
- Misclassification analysis
- Optional Streamlit inference app

## Project Structure

```text
imdb-sentiment-analysis/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
├── notebooks/
├── src/
├── outputs/
└── app/