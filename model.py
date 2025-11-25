import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
    def forward(self, x):
        # x: (batch, seq_len, features)
        outputs, (h_n, c_n) = self.lstm(x)
        # outputs: (batch, seq_len, hidden)
        return outputs, (h_n, c_n)

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.scale = hidden_size ** 0.5
    def forward(self, queries, keys, values, mask=None):
        # queries: (batch, q_len, hidden) - for our use q_len=1 (last hidden)
        Q = self.query_proj(queries)  # (b, q_len, h)
        K = self.key_proj(keys)      # (b, seq_len, h)
        V = self.value_proj(values)  # (b, seq_len, h)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (b, q_len, seq_len)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(attn_scores, dim=-1)  # (b, q_len, seq_len)
        context = torch.matmul(weights, V)  # (b, q_len, h)
        return context, weights

class LSTMAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.encoder = LSTMEncoder(input_size, hidden_size, num_layers=num_layers, dropout=dropout)
        self.attn = AttentionLayer(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        # x: (batch, seq_len, features)
        enc_outputs, (h_n, c_n) = self.encoder(x)
        # use last time-step encoder output as query (batch, 1, hidden)
        query = enc_outputs[:, -1:, :]
        context, weights = self.attn(query, enc_outputs, enc_outputs)
        # context: (batch, 1, hidden)
        out = self.fc(context.squeeze(1))  # (batch, output_size)
        return out, weights

class LSTMBaseline(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        outputs, (h_n, c_n) = self.lstm(x)
        out = self.fc(outputs[:, -1, :])  # last hidden
        return out
