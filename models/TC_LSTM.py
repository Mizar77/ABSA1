import torch.nn as nn
from torch.autograd import Variable
import torch
# 修改传入参数，xl,xr， target-word左右的句子分别传入， 这个需要另写一个处理数据的函数，在dataset文件的函数中加入split
# 选项，args.splits = (model==td_lstm/tc_lstm)  注意格式处理的部分


class TC_LSTM(nn.Module):
    def __init__(self, args):
        super(TC_LSTM, self).__init__()
        self.args = args
        self.embed = nn.Embedding(args.embed_num, args.embed_dim)
        self.lstml = nn.LSTM(args.embed_dim*2, args.hidden_size, args.num_layers, batch_first=True)
        self.lstmr = nn.LSTM(args.embed_dim*2, args.hidden_size, args.num_layers, batch_first=True)
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(args.hidden_size, args.class_num)

    def forward(self, x, y, aspect):
        x = self.embed(x)
        y = self.embed(y)
        aspect = self.embed(aspect)
        if self.args.static:
            x = Variable(x)
            y = Variable(y)
            aspect = Variable(aspect)
        aspect = aspect.sum(1)/aspect.size(1)
        a1 = aspect.unsqueeze(1).expand_as(x)
        x = torch.cat([x, a1], dim=2)
        a2 = aspect.unsqueeze(1).expand_as(y)

        y = torch.cat([y, a2], dim=2)
        hl = torch.zeros(self.args.num_layers, x.size(0), self.args.hidden_size)
        cl = torch.zeros(self.args.num_layers, x.size(0), self.args.hidden_size)

        hr = torch.zeros(self.args.num_layers, x.size(0), self.args.hidden_size)
        cr = torch.zeros(self.args.num_layers, x.size(0), self.args.hidden_size)

        # target = aspect.sum(1)/y.size(1)

        outl, _ = self.lstml(x, (hl, cl))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        outr, _ = self.lstmr(y, (hr, cr))
        # print(outl.size(), outr.size())
        out = outl[:, -1, :]+outr[:, -1, :]

        # Decode the hidden state of the last time step
        # out = self.dropout(out[:, -1, :])
        out = self.fc(out)

        return out
