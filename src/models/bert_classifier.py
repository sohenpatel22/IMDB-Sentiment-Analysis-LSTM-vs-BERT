import torch.nn as nn
from transformers import BertModel

MODEL_NAME = "bert-base-cased"


class BertSentimentClassifier(nn.Module):
    def __init__(self, n_classes=2, dropout_rate=0.3):
        super().__init__()

        self.bert = BertModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)

        return logits