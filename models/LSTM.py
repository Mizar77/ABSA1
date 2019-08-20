import torch.nn as nn
from torch.autograd import Variable
import torch


# 加入dropOut最高正确率 77%


class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.args = args
        self.embed = nn.Embedding(args.embed_num, args.embed_dim)
        self.lstm = nn.LSTM(args.embed_dim, args.hidden_size, args.num_layers, batch_first=True)
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(args.hidden_size, args.class_num)

    def forward(self, x, y):
        x = self.embed(x)
        if self.args.static:
            x = Variable(x)

        h0 = torch.zeros(self.args.num_layers, x.size(0), self.args.hidden_size)
        c0 = torch.zeros(self.args.num_layers, x.size(0), self.args.hidden_size)
        # print(h0.size(),x.size())
        # Forward propagate LSTM
        # print(x.size())
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)

        return out


