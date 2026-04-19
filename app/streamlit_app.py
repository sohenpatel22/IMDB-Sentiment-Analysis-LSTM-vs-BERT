import json
import streamlit as st
import torch

from src.models.bert_classifier import BertSentimentClassifier
from src.models.lstm_model import LSTMSentimentClassifier
from src.data.tokenizer_utils import get_tokenizer
from src.data.preprocess import clean_text, pad_features
from src.utils.checkpoint import load_checkpoint


# CONFIG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LEN_BERT = 256
SEQ_LENGTH_LSTM = 500


# LOAD RESOURCES
@st.cache_resource
def load_tokenizer():
    return get_tokenizer()


@st.cache_resource
def load_stoi():
    with open("checkpoints/stoi.json", "r") as f:
        return json.load(f)


@st.cache_resource
def load_bert_model():
    model = BertSentimentClassifier(n_classes=2, dropout_rate=0.3).to(DEVICE)
    load_checkpoint("checkpoints/bert_best.pt", model, device=DEVICE)
    model.eval()
    return model


@st.cache_resource
def load_lstm_model():
    stoi = load_stoi()

    model = LSTMSentimentClassifier(
        vocab_size=len(stoi),
        embedding_dim=128,
        hidden_dim=256,
        dropout_rate=0.3
    ).to(DEVICE)

    load_checkpoint("checkpoints/lstm_best.pt", model, device=DEVICE)
    model.eval()
    return model


# PREDICTION FUNCTIONS
def predict_bert(text, model, tokenizer):
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=MAX_LEN_BERT,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    return pred, confidence


def predict_lstm(text, model, stoi):
    words = clean_text(text).split()
    encoded = [stoi.get(word, stoi["<UNK>"]) for word in words]

    padded = pad_features([encoded], seq_length=SEQ_LENGTH_LSTM)
    tensor_input = torch.tensor(padded, dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        output = model(tensor_input)
        prob = torch.sigmoid(output).item()
        pred = 1 if prob >= 0.5 else 0

    confidence = prob if pred == 1 else (1 - prob)

    return pred, confidence


# UI
st.title("Movie Review Sentiment Analyzer")
st.write("Enter a movie review and choose a model to predict sentiment.")

model_choice = st.selectbox("Select Model", ["BERT", "LSTM"])
user_input = st.text_area("Enter Review")

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter a review.")
    else:
        if model_choice == "BERT":
            model = load_bert_model()
            tokenizer = load_tokenizer()
            pred, confidence = predict_bert(user_input, model, tokenizer)
        else:
            model = load_lstm_model()
            stoi = load_stoi()
            pred, confidence = predict_lstm(user_input, model, stoi)

        label = "Positive :)" if pred == 1 else "Negative :("

        st.subheader("Prediction")
        st.write(label)

        st.subheader("Confidence")
        st.write(f"{confidence:.4f}")