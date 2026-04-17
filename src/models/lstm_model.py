import torch
import torch.nn as nn


class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, output_dim=1):
        super(SentimentRNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.embedding(x)

        out, (hidden, cell) = self.lstm(x)

        # taking output from final time step
        out = out[:, -1, :]

        out = self.dropout(out)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)

        return out.squeeze(1)