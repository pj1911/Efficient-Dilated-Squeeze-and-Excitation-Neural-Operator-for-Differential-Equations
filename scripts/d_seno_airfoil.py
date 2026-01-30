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

# -------------------- CLI to pass paths --------------------
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default=".", help="Folder containing the .npy files")
parser.add_argument("--input_x", type=str, default="NACA_Cylinder_X.npy")
parser.add_argument("--input_y", type=str, default="NACA_Cylinder_Y.npy")
parser.add_argument("--output_sigma", type=str, default="NACA_Cylinder_Q.npy")
args = parser.parse_args()
# ------------------------------------------------------------------------

torch.manual_seed(11)
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
    def __init__(self, ch: int, dilation):
        super().__init__()
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        pad3 = dilation
        pad5 = (dilation[0] * 2, dilation[1] * 2)
        self.conv  = nn.Conv2d(ch, ch, 3,
                               padding=pad3,
                               dilation=dilation,
                               bias= True)
        self.conv5 = nn.Conv2d(ch, ch, 5,
                               padding=pad5,
                               dilation=dilation,
                               bias=False)

        self.se  = SEBlock(ch)
        self.act = nn.GELU()

    def forward(self, x):
        y = F.gelu(self.conv(x))
        y = self.conv5(y)
        y = self.se(y)
        return self.act(x + y)

# --------------------------------------------------------------------
# network body
# --------------------------------------------------------------------
class Net2d(nn.Module):
    def __init__(self, width: int, H: int, W: int):
        super().__init__()
        self.stem = nn.Conv2d(4, width, 1, bias=True)

        dilations_y = [1, 2, 8, 12, 6, 2, 1]
        dilations_x = [16, 56, 42, 36,  32, 24, 1]

        dilations   = list(zip(dilations_x, dilations_y))

        self.blocks = nn.Sequential(
            *[ResidualSEBlock2d(width, dxy) for dxy in dilations]
        )

        self.head = nn.Sequential(
            nn.Conv2d(width, 128, 1),
            nn.GELU(),
            nn.Conv2d(128, 1, 1)
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)      # (B, 3, H, W)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x.squeeze(1)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

################################################################
# configs
################################################################
INPUT_X = 'NACA_Cylinder_X.npy'
INPUT_Y = 'NACA_Cylinder_Y.npy'
OUTPUT_Sigma = 'NACA_Cylinder_Q.npy'

# -------------------- Point filenames to args.data_dir --------------------
INPUT_X = os.path.join(args.data_dir, args.input_x)
INPUT_Y = os.path.join(args.data_dir, args.input_y)
OUTPUT_Sigma = os.path.join(args.data_dir, args.output_sigma)
# -------------------------------------------------------------------------------

ntrain = 1000
ntest = 200

batch_size = 20
learning_rate = 0.001

epochs = 500
step_size = 100
gamma = 0.5


width = 64

r1=1
r2=1
s1 = int(((221 - 1) / r1) + 1)
s2 = int(((51 - 1) / r2) + 1)


################################################################
# load data and data normalization
################################################################


inputX = np.load(INPUT_X)
inputX = torch.tensor(inputX, dtype=torch.float)
inputY = np.load(INPUT_Y)
inputY = torch.tensor(inputY, dtype=torch.float)
input = torch.stack([inputX, inputY], dim=-1)

output = np.load(OUTPUT_Sigma)[:, 4]
output = torch.tensor(output, dtype=torch.float)
print(input.shape, output.shape)

x_train = input[:ntrain, ::r1, ::r2][:, :s1, :s2]
y_train = output[:ntrain, ::r1, ::r2][:, :s1, :s2]
x_test = input[ntrain:ntrain + ntest, ::r1, ::r2][:, :s1, :s2]
y_test = output[ntrain:ntrain + ntest, ::r1, ::r2][:, :s1, :s2]
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)

y_normalizer = UnitGaussianNormalizer(y_train)
y_train = y_normalizer.encode(y_train)

xs = np.linspace(0, 1, s1)
ys = np.linspace(0, 1, s2)
xx, yy = np.meshgrid(xs, ys, indexing='xy')
grid = np.stack([xx, yy], axis=-1)
grid = grid.transpose(1, 0, 2)
grid = grid[None, ...]
grid_t = torch.tensor(grid, dtype=torch.float32)

grid_train = grid_t.repeat(ntrain, 1, 1, 1)
grid_test  = grid_t.repeat(ntest,  1, 1, 1)


x_train_aug = torch.cat([x_train, grid_train], dim=-1)
x_test_aug  = torch.cat([x_test,  grid_test],  dim=-1)

print(x_train_aug.shape, x_test_aug.shape)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_train_aug, y_train),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_test_aug, y_test),
    batch_size=batch_size, shuffle=False)



model = Net2d(width, s1,s2).cuda()

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
    f"Test Loss: {test_l2:.4f}"
)

t4 = default_timer()
print("total time = ", ((t4-t3)/500))
