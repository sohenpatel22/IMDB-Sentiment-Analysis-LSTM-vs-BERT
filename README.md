# IMDB Sentiment Analysis: LSTM vs BERT

This project builds and compares deep learning models for **binary sentiment classification** on movie reviews using:

- **LSTM (from scratch)**
- **BERT (pretrained transformer)**

The goal is to understand the impact of:
- text preprocessing
- sequence modeling
- transfer learning
- hyperparameter tuning

---

## Demo (Streamlit App)

You can run an interactive web app to test predictions:

```bash
python -m streamlit run app/streamlit_app.py
````

Then open:

```
http://localhost:8501
```

*(Note: Models must be trained first before running the demo. Run the training scripts to generate checkpoints in the `checkpoints/` folder.)*

Features:

* Input custom movie review
* Choose model (LSTM or BERT)
* Get prediction + confidence score

---

## Project Structure

```
Project/
│
├── data/
│   ├── IMDB Dataset.csv
│   └── README.md
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── experiments.ipynb
│   └── outputs/
│       └── experiments/
│           ├── bert_tuning_results.csv
│           └── lstm_tuning_results.csv
│
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── preprocess.py
│   │   └── tokenizer_utils.py
│   │
│   ├── models/
│   │   ├── bert_classifier.py
│   │   └── lstm_model.py
│   │
│   ├── training/
│   │   ├── train_bert.py
│   │   ├── train_lstm.py
│   │   ├── evaluate.py
│   │   └── metrics.py
│   │
│   └── utils/
│       ├── checkpoint.py
│       ├── plotting.py
│       └── seed.py
│
├── app/
│   └── streamlit_app.py
│
├── checkpoints/
├── outputs/
└── README.md
```

---

## Dataset

* Source: IMDB Movie Reviews Dataset
* Task: Binary sentiment classification
* Labels:

  * `positive`
  * `negative`
* Dataset is **balanced (~50/50)**

---

## Exploratory Data Analysis (EDA)

Key findings:

* Reviews vary widely in length (short to 1000+ words)
* Distribution is right-skewed (long reviews exist)
* Positive and negative reviews have similar length distributions
* Raw text contains:

  * HTML tags
  * punctuation
  * noise
* Stopwords include important sentiment indicators (e.g., *not*)

### Implication:

* Sequence length = 500 chosen for LSTM
* Stopword removal tested (not always beneficial)

---

## Models

### 1. LSTM Model

* Embedding layer
* LSTM encoder
* Max + Mean pooling
* Fully connected layers

#### Final Hyperparameters

* `vocab_size = 5000 (+ special tokens)`
* `seq_length = 500`
* `embedding_dim = 128`
* `hidden_dim = 256`
* `dropout = 0.3`
* `batch_size = 32`
* `learning_rate = 1e-3`

---

### 2. BERT Model

* Pretrained: `bert-base-cased`
* Pooled output
* Dropout + Linear classifier

#### Final Hyperparameters

* `max_len = 256`
* `batch_size = 8`
* `learning_rate = 2e-5`
* `weight_decay = 0.01`
* `epochs = 2`

---

## Training

### Train LSTM

```bash
python -m src.training.train_lstm
```

### Train BERT

```bash
python -m src.training.train_bert
```

---

## Hyperparameter Tuning

Performed in:

```
notebooks/experiments.ipynb
```

### Key Observations

#### LSTM

* Larger vocab (5000) improved performance
* Keeping stopwords improved results
* Larger hidden size (256) helped
* Too much dropout hurt performance

#### BERT

* Longer sequences (256) significantly improved performance
* Learning rate `2e-5` was stable and optimal
* Smaller batch size (8) improved generalization

---

## Results

| Model | Validation F1 | Test Performance |
| ----- | ------------- | ---------------- |
| LSTM  | ~0.895        | Good baseline    |
| BERT  | ~0.919        | Best model       |

**BERT significantly outperforms LSTM**, as expected due to contextual embeddings.

---

## Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score

Evaluation is handled through:

* `evaluate.py` (model evaluation)
* `metrics.py` (metric computation)

---

## Key Learnings

* Pretrained transformers outperform traditional RNNs on NLP tasks
* Stopword removal is not always beneficial for sentiment analysis
* Sequence length is critical for capturing context
* Hyperparameter tuning matters more than architecture complexity
* Proper data preprocessing directly impacts model performance

---

## Setup Instructions

### 1. Create environment

```bash
python -m venv .venv
```

### 2. Activate environment

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add dataset

Place:

```
IMDB Dataset.csv
```

inside:

```
data/
```

---

## Future Improvements

* Add attention mechanism to LSTM
* Try DistilBERT for faster inference
* Deploy app on Streamlit Cloud
* Add confusion matrix visualization
* Add explainability (e.g., attention visualization)

---

## Author

Sohen Patel

---

## Notes

This project is designed to:

* demonstrate understanding of NLP pipelines
* compare classical vs transformer models
* showcase model deployment using Streamlit
