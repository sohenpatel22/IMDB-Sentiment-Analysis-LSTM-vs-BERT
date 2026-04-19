import torch
import torch.nn as nn


class LSTMSentimentClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        output_dim=1,
        dropout_rate=0.3
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)

        max_pool = torch.max(out, dim=1)[0]
        mean_pool = torch.mean(out, dim=1)
        out = torch.cat((max_pool, mean_pool), dim=1)

        out = self.dropout(out)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)

        return out.squeeze(1)