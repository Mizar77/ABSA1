import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from math import sqrt
# 最高81%


class AE_LSTM(nn.Module):
    def __init__(self, args):
        super(AE_LSTM, self).__init__()
        self.args = args
        self.init_param()
        self.Trans_matrix = self.init_embed(torch.Tensor(args.embed_num, args.embed_dim))
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(self.args.hidden_size, self.args.class_num)
        self.lstm = nn.LSTM(self.args.embed_dim, self.args.hidden_size, self.args.num_layers, batch_first=True)

    def init_param(self):
        self.Wh = nn.Parameter(torch.Tensor(self.args.hidden_size, self.args.hidden_size))
        self.Wv = nn.Parameter(torch.Tensor(self.args.embed_dim, self.args.embed_dim))
        # torch.tensor 和torch.Tensor的区别
        self.w = nn.Parameter(torch.Tensor(self.args.hidden_size+self.args.embed_dim))
        self.Wx = nn.Parameter(torch.Tensor(self.args.hidden_size, self.args.hidden_size))
        self.Wp = nn.Parameter(torch.Tensor(self.args.hidden_size, self.args.hidden_size))
        u = 1 / sqrt(self.args.hidden_size)
        nn.init.uniform_(self.Wh, -u, u)
        nn.init.uniform_(self.Wv, -u, u)
        nn.init.uniform_(self.w, -u, u)
        nn.init.uniform_(self.Wp, -u, u)
        nn.init.uniform_(self.Wx, -u, u)

        # !!!!!!!!!!!!!!!!!!!init.unifor一定要有，不然loss可能是nan
    def forward(self, x, aspect):
        x = self.Trans_matrix(x)
        aspect = self.Trans_matrix(aspect)

        if self.args.static:
            x = Variable(x)
            aspect = Variable(aspect)
        # print(aspect.size())
        aspect = aspect.sum(1)   # 通过取平均得出aspect embedding

        h0 = torch.zeros(self.args.num_layers, x.size(0), self.args.hidden_size)
        c0 = torch.zeros(self.args.num_layers, x.size(0), self.args.hidden_size)

        H_t, _ = self.lstm(x, (h0, c0))

        # print(aspect.size(), self.Wh.size(), self.args.embed_dim, self.args.hidden_size)
        # print(aspect[:, None].size())
        aspect_matrix = torch.unsqueeze(aspect,2)
        aspect_matrix = aspect_matrix.expand(x.size(0),aspect.size(1),x.size(1))
        aspect_matrix = torch.transpose(aspect_matrix,1,2)
        # print(torch.matmul(self.Wh, H).size(), torch.matmul(self.Wv, aspect_matrix).size())
        # print(torch.matmul(H_t, torch.transpose(self.Wh,0,1)).size())
        # print('aspect',aspect_matrix.size(),self.Wv.size())
        # print(torch.matmul(aspect_matrix, torch.transpose(self.Wv,0,1)).size())
        M = torch.tanh(torch.cat([torch.matmul(H_t, torch.transpose(self.Wh,0,1)), torch.matmul(aspect_matrix, torch.transpose(self.Wv,0,1))],dim =-1))
        # dim=-1??
        # print('M w',M.size(),self.w.size())

        alpha = F.softmax(torch.matmul(M, self.w), dim=1)
        # print('alpha',alpha.size(),'H_t', H_t.size())
        # print(x.size(0),self.Wh.size(0))
        # r=torch.empty(x.size(0),self.Wh.size(0))
        # for i in range(x.size(0)):
        #    r[i,:]=torch.matmul(alpha[i,:],H_t[i,:,:])
        # H_t=H_t.transpose(1,2)
        alpha=torch.unsqueeze(alpha,1)
        #print(alpha.size(),H_t.size())
        r=torch.bmm(alpha,H_t) #bmm for batch multiply
        r=torch.squeeze(r,1)
        #print(r.size())
        hs = torch.tanh(torch.matmul(r,self.Wp)+torch.matmul(H_t[:, -1, :],self.Wx))
        # print(torch.matmul(r,self.Wp).size())
        # print(torch.matmul(H_t[:,-1,:],self.Wx).size())
        out = self.dropout(hs)
        out = self.fc(out)
        return out

    def init_embed(self, embedding):
        num_word, dim_word = embedding.size()
        embed_matrix = nn.Embedding(num_word, dim_word)
        embed_matrix.weight = nn.Parameter(embedding)
        return embed_matrix



