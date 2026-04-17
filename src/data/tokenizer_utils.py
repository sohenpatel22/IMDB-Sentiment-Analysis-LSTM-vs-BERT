from transformers import BertTokenizer

MODEL_NAME = "bert-base-cased"


def get_tokenizer():
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    return tokenizer