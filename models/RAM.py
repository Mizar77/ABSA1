import torch.nn as nn
from math import sqrt
import torch
#  TODO: 修改dataset/load_data  还原re_sub,彻底解决 story ,  的问题
# dropout的test/train思考
# 91%


class RAM(nn.Module):
    def __init__(self, args):
        super(RAM, self).__init__()
        self.args=args
        self.lstm_hidden_size = args.RAM_hidden_size#
        self.lstm_num_layers = args.RAM_LSTM_num_layers#
        self.embed_dim = args.embed_dim
        self.embed_num = args.embed_num
        self.gru_hidden_size = args.GRU_hidden_size#
        self.class_num = args.class_num
        self.gru_N = args.GRU_timestep
        self.embed = nn.Embedding(self.embed_num, self.embed_dim)

        # self.lstml = nn.LSTM(hidden_size=self.lstm_hidden_size, num_layers=self.lstm_num_layers,
        #                     input_size=self.embed_dim, batch_first=True)
        self.lstm = nn.LSTM(hidden_size=self.lstm_hidden_size, num_layers=self.lstm_num_layers,
                            input_size=self.embed_dim, batch_first=True, bidirectional=True, bias=True)
        self.grucell=nn.GRUCell(input_size=self.lstm_hidden_size*2+1, hidden_size=self.gru_hidden_size)
        self.W_al = nn.Parameter(torch.Tensor(self.gru_N, 2*self.lstm_hidden_size+1+self.gru_hidden_size+self.embed_dim))
        self.b_al = nn.Parameter(torch.Tensor(self.gru_N))
        u = 1 / sqrt(self.gru_hidden_size)
        nn.init.uniform_(self.W_al, -u, u)
        nn.init.uniform_(self.b_al, -u, u)
        self.dropout=nn.Dropout(args.dropout)
        self.fc = nn.Linear(self.gru_hidden_size, self.class_num)

    def forward(self, x, aspect):
        orgX= x
        orgA = aspect
        x = self.embed(x)
        aspect = self.embed(aspect)
        # print(x.size(),aspect.size())
        H, _ = self.lstm(x)
        # print('lstm finished')
        M = self.cal_M(orgX, orgA, H)
        # print('M calculated')
        out = self.gru(M, aspect)
        # print('gru finished')
        # print('gru res.size:', out.size())
        out = out[:, -1]
        out = self.dropout(out)
        out = self.fc(out)
        return out

    def gru(self, M, asp):
        e = torch.zeros(asp.size(0), self.gru_hidden_size)
        asp = asp.mean(1)
        h_list=e.unsqueeze(dim=1)
        for i in range(self.gru_N):
            W_al = self.W_al[i,:].unsqueeze(dim=0).expand(asp.size(0), 1, self.W_al.size(1))
            b_al = self.b_al[i].unsqueeze(dim=0).expand(asp.size(0), 1)
            # print('e,M,asp:', e.size(),M.size(),asp.size())
            #print(asp.unsqueeze(dim=1).expand(asp.size(0),M.size(1),asp.size(1)).size())
            hh = torch.cat([M, e.unsqueeze(dim=1).expand(e.size(0),M.size(1),e.size(1)),
                            asp.unsqueeze(dim=1).expand(asp.size(0),M.size(1),asp.size(1))], dim=2)
            #hh = torch.cat([hh,asp.unsqueeze(dim=1).expand(asp.size(0),M.size(1),asp.size(1))],dim=2)
            # print('hh:',hh.size(),'W_al', W_al.size(),'b_al',b_al.size())
            g = torch.bmm(W_al, hh.transpose(1,2)).squeeze()+b_al
            # g = g.squeeze()
            alpha = torch.softmax(g,dim=1)
            # print('g,alpha:',g.size(),alpha.size())
            alpha = alpha.unsqueeze(2).expand_as(M)
            It = alpha*M
            It = It.sum(1)
            # print('It:',It.size(),e.size())
            e = self.grucell(It, e)
            # print(e.size())
            h_list = torch.cat([h_list, e.unsqueeze(1)], dim=1)
        return h_list

    def cal_M(self, x, aspect, H):
        u = torch.zeros(x.size(0), H.size(1), dtype=torch.float)
        # print('size in cal_M:', x.size(), aspect.size(), H.size(), u.size())
        y = aspect
        for i in range(x.size(0)):  # batch
            # tailX = (torch.range(1, x[i, :].size(0))[x[i, :].eq(1)])
            # if tailX.size() == torch.Size([0]):
            #    tailX = x[i,:].size(0)-1
            tailY = torch.arange(1, y[i, :].size(0) + 1)[y[i, :].eq(1)]
            if tailY.size() == torch.Size([0]):
                tailY = y[i, :].size(0) - 1
            else:
                tailY = tailY[0] - 2
            #print('x:', x[i,:])
            #print('y:', aspect[i,:])
            lIdx = (torch.arange(1, x[i, :].size(0) + 1)[x[i, :].eq(aspect[i, 0])])
            if lIdx.size() == torch.Size([0]):
                #print('x:', x[i,:])
                #print('y:', y[i,:])
                #text = [[self.args.text_field.vocab.itos[s] for s in x[i,:]]]
                #print(text)
                text = [[self.args.text_field.vocab.itos[s] for s in y[i, :]]]
                #print(text)
                #print('aspect not in sentence')
                lIdx = 0
            else:
                lIdx=lIdx[0]-1
            rIdx = lIdx + tailY
            tmax = x.size(1) - rIdx
            if tmax < lIdx:
                tmax = lIdx
            for j in range(x.size(1)):
                if j < lIdx:
                    u[i][j] = float(lIdx - j) / float(tmax)
                else:
                    if j > rIdx:
                        u[i][j] = float((j - rIdx)) / float(tmax)
                    else:
                        u[i][j] = float(0)
                # print('i,j,u[i][j],tmax,,rIdx,lIdx,tailY:', i, j, tmax, u[i][j], rIdx, lIdx, tailY)
        # print(u.size())
        v = torch.unsqueeze(u, dim=2).expand_as(H)
        # print('v:', v)
        M = torch.cat([(1 - v) * H, u.unsqueeze(dim=2)], dim=2)
        return M
