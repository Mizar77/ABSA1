import torch
import torch.nn as nn
from math import sqrt
# 85%

class IAN(nn.Module):
    def __init__(self, args):
        super(IAN, self).__init__()
        self.args = args
        self.lstmc = nn.LSTM(args.embed_dim, args.hidden_size, batch_first=True)
        self.lstmt = nn.LSTM(args.embed_dim, args.hidden_size, batch_first=True)
        self.embed = nn.Embedding(args.embed_num, args.embed_dim)
        self.init_param()
        self.dropout= nn.Dropout(self.args.dropout)
        self.fc = nn.Linear(self.args.hidden_size*2, self.args.class_num)

    def init_param(self):
        self.Wc = nn.Parameter(torch.Tensor(self.args.hidden_size,self.args.hidden_size))
        self.bc = nn.Parameter(torch.Tensor(1))
        self.Wt = nn.Parameter(torch.Tensor(self.args.hidden_size,self.args.hidden_size))
        self.bt = nn.Parameter(torch.Tensor(1))
        u = 1 / sqrt(self.args.hidden_size)
        nn.init.uniform_(self.Wc, -u, u)
        nn.init.uniform_(self.bc, -u, u)
        nn.init.uniform_(self.Wt, -u, u)
        nn.init.uniform_(self.bt, -u, u)
        # nn.init.uniform_(self.Wx, -u, u)

    def forward(self, con, tar):  # context,target
        con = self.embed(con)
        tar = self.embed(tar)

        h0=torch.zeros(self.args.num_layers, con.size(0), self.args.hidden_size)
        c0=torch.zeros(self.args.num_layers, con.size(0), self.args.hidden_size)

        Hc, _=self.lstmc(con,(h0,c0))
        Ht, _=self.lstmt(tar,(h0,c0))

        Cavg = Hc.sum(1)/Ht.size(1)
        Tavg = Ht.sum(1)/Ht.size(1)
        #print(Hc.size(), Ht.size())  # [128,49,64]  [128,9,64]
        #print(Cavg.size(),Tavg.size()) #torch.Size([128, 64]) torch.Size([128, 64])

        hh = torch.bmm(Hc, self.Wc.unsqueeze(0).expand(Hc.size(0),self.Wc.size(0),self.Wc.size(1)))
        hh = torch.bmm(hh, Tavg.unsqueeze(2)).squeeze(2)
        extc = self.bc.unsqueeze(0).expand_as(hh)

        alpha = torch.softmax(torch.tanh(hh+extc),dim=1)
        #print(hh.size(), extc.size(),alpha.size())

        hht = torch.bmm(Ht, self.Wt.unsqueeze(0).expand(Ht.size(0), self.Wt.size(0), self.Wt.size(1)))
        hht = torch.bmm(hht, Cavg.unsqueeze(2)).squeeze(2)
        extt = self.bt.unsqueeze(0).expand_as(hht)

        belta = torch.softmax(torch.tanh(hht + extt), dim=1)

        #print(belta.size())

        alpha= alpha.unsqueeze(1)
        belta = belta.unsqueeze(1)

        Cr = torch.bmm(alpha, Hc).squeeze()
        Tr = torch.bmm(belta, Ht).squeeze()
        d = torch.cat([Cr,Tr],dim=1)
        #print(d.size())

        out = self.dropout(d)
        out = self.fc(out)
        out = torch.tanh(out)
        return out

        # return torch.zeros(con.size(0),self.args.class_num)

