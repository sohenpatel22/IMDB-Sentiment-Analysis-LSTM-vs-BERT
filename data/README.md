# Data

This project uses the **IMDB Movie Review Dataset** for binary sentiment classification.

## Files
- `IMDB Dataset.csv`: Raw dataset containing movie reviews and sentiment labels.
- `README.md`: Documentation for the dataset used in this project.

## Dataset Description
The dataset contains:
- `review`: text of the movie review
- `sentiment`: sentiment label (`positive` or `negative`)

## Notes
- The dataset is used for both the LSTM-based model and the BERT-based model.
- Preprocessing such as text cleaning, tokenization, vocabulary building, and padding is handled inside the `src/data/` pipeline.
- Train/validation/test split is performed in code, not stored as separate files.

## Source
This dataset was provided as part of the course assignment and is based on the IMDB movie review sentiment classification dataset.

## Important
If the dataset file is not included in the repository, place `IMDB Dataset.csv` inside this folder before running the notebooks or training scripts.