from transformers import BertTokenizer

MODEL_NAME = "bert-base-cased"


def get_tokenizer():
    return BertTokenizer.from_pretrained(MODEL_NAME)