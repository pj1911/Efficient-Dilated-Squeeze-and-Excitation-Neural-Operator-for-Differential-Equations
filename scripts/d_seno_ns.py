import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial
from timeit import default_timer
from utilities3 import *

# -------------------- Simple CLI to pass paths --------------------
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default=".", help="Folder containing the .mat file")
parser.add_argument("--train_path", type=str, default="NavierStokes_V1e-5_N1200_T20.mat")
parser.add_argument("--test_path", type=str, default="NavierStokes_V1e-5_N1200_T20.mat")
args = parser.parse_args()
# ------------------------------------------------------------------------

torch.manual_seed(11) # 0, 11, 17
torch.cuda.manual_seed(11)  
torch.cuda.manual_seed_all(11) 

if torch.cuda.is_available():
    device = torch.device("cuda:0") 
    print(f"Running on GPUs: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
else:
    device = torch.device("cpu")
    print("Running on CPU")

# --------------------------------------------------------------------
# squeeze and excitation block
# --------------------------------------------------------------------
class SEBlock(nn.Module):
    def __init__(self, ch: int, reduction: int = 1):
        super().__init__()
        self.fc1 = nn.Conv2d(ch, ch // reduction, 1, bias=True)
        self.fc2 = nn.Conv2d(ch // reduction, ch, 1, bias=True)

    def forward(self, x):
        s = x.mean((2, 3), keepdim=True)
        s = F.gelu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s

# --------------------------------------------------------------------
# DC block : 3x3 and 3x3 D-Conv2d
# --------------------------------------------------------------------
class ResidualSEBlock2d(nn.Module):
    def __init__(self, ch: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv2d(
            ch, ch, 3, padding=dilation, dilation=dilation, bias=True
        )
        
        self.conv5 = nn.Conv2d(
            ch, ch, 3, padding=dilation, dilation=dilation, bias=True
        )

        self.se   = SEBlock(ch)
        self.act  = nn.GELU()

    def forward(self, x):  # b,w,x,y
        y = F.gelu( self.conv(x) )
        y = self.conv5(y)
        y = self.se(y)
        return self.act(x + y)

# --------------------------------------------------------------------
# network body 
# --------------------------------------------------------------------
class Net2d(nn.Module):
    def __init__(self, width: int, H: int, W: int, num_blocks: int = 12):
        super().__init__()
        self.stem = nn.Linear(12, width)
        dilations = [15, 25, 17, 13, 7, 5, 3, 1]
        self.blocks = nn.Sequential(
            *[ResidualSEBlock2d(width, d) for d in dilations]
        )

        self.head = nn.Sequential(
            nn.Conv2d(width, 128, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(128, 1, 1)
        )

    def forward(self, x):
        x = self.stem(x)
        x = x.permute(0, 3, 1, 2) # b,w,x,y
        x = self.blocks(x)
        x = self.head(x)
        return x.permute(0,2,3,1).contiguous()

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

TRAIN_PATH = 'NavierStokes_V1e-5_N1200_T20.mat'
TEST_PATH = 'NavierStokes_V1e-5_N1200_T20.mat'

# -------------------- Point filenames to args.data_dir --------------------
TRAIN_PATH = os.path.join(args.data_dir, args.train_path)
TEST_PATH = os.path.join(args.data_dir, args.test_path)
# -------------------------------------------------------------------------------

ntrain = 1000
ntest = 200

# modes = 12
width = 64

batch_size = 20
batch_size2 = batch_size

epochs = 500
learning_rate = 0.001
scheduler_step = 100
scheduler_gamma = 0.5

runtime = np.zeros(2, )
t1 = default_timer()

sub = 1
S = 64
s=S
T_in = 10
T = 10
step = 1

################################################################
# load data
################################################################

reader = MatReader(TRAIN_PATH)
train_a = reader.read_field('u')[:ntrain,::sub,::sub,:T_in]
train_u = reader.read_field('u')[:ntrain,::sub,::sub,T_in:T+T_in]

reader = MatReader(TEST_PATH)
test_a = reader.read_field('u')[-ntest:,::sub,::sub,:T_in]
test_u = reader.read_field('u')[-ntest:,::sub,::sub,T_in:T+T_in]

print(train_u.shape)
print(test_u.shape)
assert (S == train_u.shape[-2])
assert (T == train_u.shape[-1])

train_a = train_a.reshape(ntrain,S,S,T_in)
test_a = test_a.reshape(ntest,S,S,T_in)

gridx = torch.linspace(0, 1, S, dtype=torch.float).reshape(1, S, 1, 1).repeat(1, 1, S, 1)
gridy = torch.linspace(0, 1, S, dtype=torch.float).reshape(1, 1, S, 1).repeat(1, S, 1, 1)

gridx_train = gridx.repeat(ntrain, 1, 1, 1)
gridy_train = gridy.repeat(ntrain, 1, 1, 1)
gridx_test  = gridx.repeat(ntest,  1, 1, 1)
gridy_test  = gridy.repeat(ntest,  1, 1, 1)

train_a = torch.cat((train_a, gridx_train, gridy_train), dim=-1)
test_a  = torch.cat((test_a,  gridx_test,  gridy_test),  dim=-1)

from torch.utils.data import TensorDataset, DataLoader

train_loader = DataLoader(TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(TensorDataset(test_a,  test_u),  batch_size=batch_size, shuffle=False)

t2 = default_timer()



model = Net2d(width, s,s).cuda()
print(model.count_params())

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

myloss = LpLoss(size_average=False)
gridx = gridx.to(device)
gridy = gridy.to(device)


t3 = default_timer()
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2_step = 0
    train_l2_full = 0
    for xx, yy in train_loader:
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)

        for t in range(0, T, step):
            y = yy[..., t:t + step]
            im = model(xx)
            loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)

            xx = torch.cat((xx[..., step:-2], im,
                            gridx.repeat([batch_size, 1, 1, 1]), gridy.repeat([batch_size, 1, 1, 1])), dim=-1)

        train_l2_step += loss.item()
        l2_full = myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
        train_l2_full += l2_full.item()

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

    test_l2_step = 0
    test_l2_full = 0
    with torch.no_grad():
        for xx, yy in test_loader:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)

            for t in range(0, T, step):
                y = yy[..., t:t + step]
                im = model(xx)
                loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)

                xx = torch.cat((xx[..., step:-2], im,
                                gridx.repeat([batch_size, 1, 1, 1]), gridy.repeat([batch_size, 1, 1, 1])), dim=-1)


            test_l2_step += loss.item()
            test_l2_full += myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()

    t2 = default_timer()
    scheduler.step()
    print(ep, t2 - t1, train_l2_step / ntrain / (T / step), train_l2_full / ntrain, test_l2_step / ntest / (T / step),
          test_l2_full / ntest)
t4 = default_timer()
print("total time = ", ((t4-t3)/epochs))
