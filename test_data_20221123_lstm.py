import torch
import torch.nn as nn
import sys
os.chdir("c:/Users/48944/finance/")
from importlib import reload
#sys.path.append("c:/Users/48944/finance/")
import data
reload(data)
from data import *
import pdb
import torch.optim as optim

bs.login()
stock_300 = bs.query_hs300_stocks("2020-01-01").get_data()
stocks = stock_300["code"]
t1 = "2015-01-01"
t2 = "2022-01-01"
t3 = "2022-11-01"

sd = stock_dataset(stocks, t1, t2)
sd_test = stock_dataset(stocks, t2, t3)

def self_corr(df, i):
    return pd.concat([df.shift(k) for k in range(i)], axis=1).corr().iloc[0]

def dataset(sd, embed_dim):
    for i in range(sd.ret_d.shape[1] - embed_dim):
        x = torch.tensor(sd.ret_d.values[:, i:(i + embed_dim)], dtype=torch.float32)
        y = torch.tensor(sd.ret_d.values[:, embed_dim + i], dtype=torch.float32)
        yield x * 50, y * 50


c1 = ((sd.corr_inv@sd.ret_d).T / sd.corr_inv_diag).T
c2 = sd.ret_d - c1
c3 = c2.loc["sh.600733"]

test_size = 500
c3_1 = 10 * torch.tensor(c3.iloc[: -1 - test_size], dtype=torch.float32).view(c3.shape[0] - 1 - test_size, 1, 1)
c3_2 = 10 * torch.tensor(c3.iloc[1: -test_size], dtype=torch.float32)
c4_1 = 10 * torch.tensor(c3.iloc[(-1 - test_size): -1], dtype=torch.float32).view(test_size, 1, 1)
c4_2 = 10 * torch.tensor(c3.iloc[( -test_size):], dtype=torch.float32)

hidden_size = 8
input_size = 1
batch_size = 1

class stock_lstm(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(stock_lstm, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size) # (input_size, hidden_size)
        self.nl = nn.Linear(hidden_size, input_size, bias=False)
    def forward(self, input_data, h0=None, c0=None):
        if h0 is None:
            self.h0 = torch.randn(input_size, batch_size, hidden_size) # (batch, hidden_size)
        else:
            self.h0 = h0

        if c0 is None:
            self.c0 = torch.randn(input_size, batch_size, hidden_size) # (batch, hidden_size)
        else:
            self.c0 = c0
        output_data, (hx, cx) = self.rnn(input_data, (self.h0, self.c0))
        self.hx = hx
        self.cx = cx
        return self.nl(output_data).view(output_data.shape[0])


L1 = nn.SmoothL1Loss()
L1 = nn.L1Loss()
L2 = nn.MSELoss()

policy_net = stock_lstm(input_size, hidden_size, batch_size)
output_data = policy_net(c3_1)
optimizer = optim.SGD(policy_net.parameters(), lr=0.01, momentum=0.9)


for i in range(15):
    output_data = policy_net(c3_1)
    loss = L2(output_data, c3_2)
    optimizer.zero_grad()
    loss.backward()
    print(loss)
    optimizer.step()


output_data1 = policy_net(c4_1, policy_net.hx, policy_net.cx)
output_data1.mean()

L2(output_data1, c4_2)
output_data1
pd.DataFrame(np.array([c4_2.detach().numpy(), output_data1.detach().numpy()])).T.corr()







