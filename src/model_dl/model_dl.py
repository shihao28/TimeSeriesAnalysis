import torch
from torch import nn
import logging


class RNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, n_out):
        super(RNN, self).__init__()
        # if isinstance(hidden_dim, int):
        #     hidden_dim = [hidden_dim] * rnn_unit_count
        # elif isinstance(hidden_dim, list) and len(hidden_dim) != rnn_unit_count:
        #     logging.info('length of hidden_dim is not equal to rnn_unit_count')
        #     hidden_dim = [hidden_dim[0]] * rnn_unit_count
        self.ln = nn.LayerNorm(in_dim)
        self.lstm = nn.LSTM(
            input_size=in_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
        )
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, 1, 1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, n_out)

    def forward(self, x):
        x = self.ln(x)
        output, (hidden_state, cell_state) = self.lstm(x)
        output = output.transpose(1, 2)
        x = self.conv(output)
        x = self.avg_pool(x).flatten(1)
        logits = self.fc(x)

        return logits


if __name__ == "__main__":
    inputs = torch.rand((64, 10, 32))
    model = RNN(32, 128, 1, 3)
    model(inputs)
