import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class LSTM(nn.Module):
    def __init__(self, configs):
        super(LSTM, self).__init__()
        self.lstm_1 = nn.LSTM(input_size=configs.input_channels, hidden_size=100, num_layers=1, bias=True, batch_first=True)
        self.lstm_2 = nn.LSTM(input_size=100, hidden_size=100, num_layers=1, bias=True, batch_first=True)
        self.fc = nn.Linear(100, configs.num_classes)

    def init_hidden(self, batch_size, dim_hidden):
        return (torch.randn(1, batch_size, dim_hidden).cuda(),
                torch.randn(1, batch_size, dim_hidden).cuda())

    def forward(self, x):
        # x: (b, c, ts)
        bs = x.shape[0]
        out_lstm, (hn, cn) = self.lstm_1(x.transpose(1,2), self.init_hidden(bs, 100))  # (b, ts, 100)
        out_lstm, (hn, cn) = self.lstm_2(out_lstm, self.init_hidden(bs, 100))          # (b, ts, 100)
        out = out_lstm[:, -1, :]   # (b, 100)
        out = self.fc(out)
        return out, None


class Transformer_Model(nn.Module):
    def __init__(self, configs):
        super(Transformer_Model, self).__init__()
        encoder_layers = TransformerEncoderLayer(
            d_model = configs.num_time_steps,
            dim_feedforward = 2*configs.num_time_steps,
            nhead = 2,
            batch_first = True,
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, 2)

        self.logits = nn.Linear(
            configs.input_channels * configs.num_time_steps,
            configs.num_classes
        )

    def forward(self, x_in_t):
        # x_in_t: (b, c, ts)
        x = self.transformer_encoder(x_in_t)
        x_flat = x.reshape(x.shape[0], -1)
        logits = self.logits(x_flat)
        return logits, None


class base_Model(nn.Module):
    def __init__(self, configs):
        super(base_Model, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, 32, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size//2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        model_output_dim = configs.features_len
        self.logits = nn.Linear(model_output_dim * configs.final_out_channels, configs.num_classes)

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x_flat = x.reshape(x.shape[0], -1)
        logits = self.logits(x_flat)
        return logits, x
