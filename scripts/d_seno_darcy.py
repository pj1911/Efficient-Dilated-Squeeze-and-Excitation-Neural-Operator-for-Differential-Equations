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
parser.add_argument("--data_dir", type=str, default=".", help="Folder containing the .mat files")
parser.add_argument("--train_path", type=str, default="piececonst_r421_N1024_smooth1.mat")
parser.add_argument("--test_path", type=str, default="piececonst_r421_N1024_smooth2.mat")
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
# DC block : 3x3 and 5x5 D-Conv2d
# --------------------------------------------------------------------
class ResidualSEBlock2d(nn.Module):
    def __init__(self, ch: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=dilation, dilation=dilation, bias=True)
        self.conv5 = nn.Conv2d(ch, ch, 5, padding= 2 *dilation, dilation=dilation, bias=False)
        self.se   = SEBlock(ch)
        self.act  = nn.GELU()

    def forward(self, x):
        y = self.conv(x)
        y = F.gelu(y)
        y = self.conv5(y)
        y = self.se(y)
        return self.act(x + y)

# --------------------------------------------------------------------
# network body 
# --------------------------------------------------------------------
class Net2d(nn.Module):
    def __init__(self, width: int, H: int, W: int):
        super().__init__()
        self.stem = nn.Conv2d(3, width, 1, bias=True)
        dilations =[1, 3, 5, 9, 13, 19]
        self.blocks = nn.Sequential(*[ResidualSEBlock2d(width, d) for d in dilations])

        self.head = nn.Sequential(
            nn.Conv2d(width, 128, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(128, 1, 1))

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)   # (B,3,H,W)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x.squeeze(1)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

################################################################
# configs
################################################################
TRAIN_PATH = 'piececonst_r421_N1024_smooth1.mat'
TEST_PATH = 'piececonst_r421_N1024_smooth2.mat'

# -------------------- Point filenames to args.data_dir --------------------
TRAIN_PATH = os.path.join(args.data_dir, args.train_path)
TEST_PATH = os.path.join(args.data_dir, args.test_path)
# -------------------------------------------------------------------------------

ntrain = 1000
ntest = 200

batch_size = 20
learning_rate = 0.001

epochs = 500
step_size = 100
gamma = 0.5


width = 48

r = 5
h = int(((421 - 1)/r) + 1)
s = h

################################################################
# load data and data normalization
################################################################
reader = MatReader(TRAIN_PATH)
x_train = reader.read_field('coeff')[:ntrain,::r,::r][:,:s,:s]
y_train = reader.read_field('sol')[:ntrain,::r,::r][:,:s,:s]

reader.load_file(TEST_PATH)
x_test = reader.read_field('coeff')[:ntest,::r,::r][:,:s,:s]
y_test = reader.read_field('sol')[:ntest,::r,::r][:,:s,:s]

x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)

y_normalizer = UnitGaussianNormalizer(y_train)
y_train = y_normalizer.encode(y_train)

grids = []
grids.append(np.linspace(0, 1, s))
grids.append(np.linspace(0, 1, s))
grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
grid = grid.reshape(1,s,s,2)
grid = torch.tensor(grid, dtype=torch.float)
x_train = torch.cat([x_train.reshape(ntrain,s,s,1), grid.repeat(ntrain,1,1,1)], dim=3)
x_test = torch.cat([x_test.reshape(ntest,s,s,1), grid.repeat(ntest,1,1,1)], dim=3)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)



model = Net2d(width, s,s).cuda()
print(model.count_params())

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


myloss = LpLoss(size_average=False)
y_normalizer.cuda()
t3 = default_timer()
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        out = y_normalizer.decode(model(x))
        y = y_normalizer.decode(y)
    
        l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
        l2.backward()
        optimizer.step()
        train_l2 += l2.item()
        
    scheduler.step()
    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            out = y_normalizer.decode(model(x))
            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()
            
    train_l2 /= ntrain
    test_l2 /= ntest

    t2 = default_timer()
    print(f"Epoch: {ep:.1f}, "
    f"Time Elapsed: {t2-t1:.2f}, "
    f"Train Loss: {train_l2:.4f}, "
    f"Test Loss: {test_l2:.4f}")
    
t4 = default_timer()
print("total time = ", ((t4-t3)/500))
