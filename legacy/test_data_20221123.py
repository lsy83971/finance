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

class ret_attn(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(ret_attn, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.nl = nn.Linear(embed_dim, 1, bias=False)
        
    def forward(self, query, key, value):
        attn_output, attn_output_weights = self.multihead_attn(query, key, value)
        nl_output = self.nl(attn_output)
        return nl_output.view(nl_output.shape[0])

embed_dim = 32
num_heads = 2
batch_size = 1
seq_len_query = 247
seq_len_key = 247

L1 = nn.SmoothL1Loss()

policy_net = ret_attn(embed_dim, num_heads)
optimizer = optim.SGD(policy_net.parameters(), lr=0.01, momentum=0.9)

s1 = dict()
for r in range(10):
    for k, (i, j) in enumerate(dataset(sd, embed_dim)):
        nl_output = policy_net(i, i, i)
        loss = L1(nl_output, j)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        s1[k] = loss.detach().numpy()
        if loss.detach().numpy() > 100:
            break
    print("mean", pd.Series(s1.values()).mean())



sb1 = dataset(sd_test, embed_dim)
sb2 = list()
sb3 = list()
for i, j in sb1:
    nl_output = policy_net(i, i, i)
    sb2.append(nl_output.detach().numpy())
    sb3.append(j.detach().numpy())

s2 = pd.DataFrame(np.array(sb2)).melt()
s3 = pd.DataFrame(np.array(sb3)).melt()


pd.concat([s2["value"], s3["value"]], axis=1).corr()



c1 = ((sd.corr_inv@sd.ret_d).T / sd.corr_inv_diag).T
c2 = sd.ret_d - c1


sd.corr_inv_diag[sd.corr_inv_diag < 2]

import matplotlib.pyplot as plt
plt.plot(c2.loc["sh.600733"]. head(100))

c3 = c2.loc["sh.600733"]
c3.head(50)

sb = (c3 > 0).astype(int)

self_corr(sb, 10)
self_corr(c3, 10)




