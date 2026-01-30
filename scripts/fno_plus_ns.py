import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.fft

import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
from utilities3 import *

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default=".", help="Folder containing the .mat file")
parser.add_argument("--train_path", type=str, default="NavierStokes_V1e-5_N1200_T20.mat")
parser.add_argument("--test_path", type=str, default="NavierStokes_V1e-5_N1200_T20.mat")
args = parser.parse_args()

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"Running on GPUs: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
else:
    device = torch.device("cpu")
    print("Running on CPU")


class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def forward(self, x):
        batchsize, _, height, width = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")

        out_ft = torch.zeros(
            batchsize, self.out_channels, height, width // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )

        out_ft[:, :, : self.modes1, : self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, : self.modes1, : self.modes2],
            self.weights1
        )

        out_ft[:, :, -self.modes1:, : self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, -self.modes1:, : self.modes2],
            self.weights2
        )

        x = torch.fft.irfft2(out_ft, s=(height, width), norm="ortho")
        return x


class SimpleBlock2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(SimpleBlock2d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.fc0 = nn.Linear(12, self.width)

        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = F.gelu(x1 + x2)

        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = F.gelu(x1 + x2)

        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = F.gelu(x1 + x2)

        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


class Net2d(nn.Module):
    def __init__(self, modes, width):
        super(Net2d, self).__init__()
        self.conv1 = SimpleBlock2d(modes, modes, width)

    def forward(self, x):
        x = self.conv1(x)
        return x

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        return c


################################################################
# configs
################################################################
TRAIN_PATH = 'NavierStokes_V1e-5_N1200_T20.mat'
TEST_PATH = 'NavierStokes_V1e-5_N1200_T20.mat'

TRAIN_PATH = os.path.join(args.data_dir, args.train_path)
TEST_PATH = os.path.join(args.data_dir, args.test_path)

ntrain = 1000
ntest = 200

modes = 8
width = 64

batch_size = 20
batch_size2 = batch_size

epochs = 500
learning_rate = 0.001
scheduler_step = 100
scheduler_gamma = 0.5

print(epochs, learning_rate, scheduler_step, scheduler_gamma)

t1 = default_timer()

sub = 1
S = 64
T_in = 10
T = 10
step = 1

################################################################
# load data
################################################################
reader = MatReader(TRAIN_PATH)
train_a = reader.read_field('u')[:ntrain, ::sub, ::sub, :T_in]
train_u = reader.read_field('u')[:ntrain, ::sub, ::sub, T_in:T + T_in]

reader = MatReader(TEST_PATH)
test_a = reader.read_field('u')[-ntest:, ::sub, ::sub, :T_in]
test_u = reader.read_field('u')[-ntest:, ::sub, ::sub, T_in:T + T_in]

print(train_u.shape)
print(test_u.shape)
assert (S == train_u.shape[-2])
assert (T == train_u.shape[-1])

train_a = train_a.reshape(ntrain, S, S, T_in)
test_a = test_a.reshape(ntest, S, S, T_in)

gridx = torch.linspace(0, 1, S, dtype=torch.float).reshape(1, S, 1, 1).repeat(1, 1, S, 1)
gridy = torch.linspace(0, 1, S, dtype=torch.float).reshape(1, 1, S, 1).repeat(1, S, 1, 1)

gridx_train = gridx.repeat(ntrain, 1, 1, 1)
gridy_train = gridy.repeat(ntrain, 1, 1, 1)
gridx_test = gridx.repeat(ntest, 1, 1, 1)
gridy_test = gridy.repeat(ntest, 1, 1, 1)

train_a = torch.cat((train_a, gridx_train, gridy_train), dim=-1)
test_a = torch.cat((test_a, gridx_test, gridy_test), dim=-1)

from torch.utils.data import TensorDataset, DataLoader

train_loader = DataLoader(TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

t2 = default_timer()
print('preprocessing finished, time used:', t2 - t1)
device = torch.device('cuda')

################################################################
# training and evaluation
################################################################
model = Net2d(modes, width).cuda()

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

            xx = torch.cat(
                (xx[..., step:-2], im,
                 gridx.repeat([batch_size, 1, 1, 1]), gridy.repeat([batch_size, 1, 1, 1])),
                dim=-1
            )

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

                xx = torch.cat(
                    (xx[..., step:-2], im,
                     gridx.repeat([batch_size, 1, 1, 1]), gridy.repeat([batch_size, 1, 1, 1])),
                    dim=-1
                )

            test_l2_step += loss.item()
            test_l2_full += myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()

    t2 = default_timer()
    scheduler.step()
    print(
        ep, t2 - t1,
        train_l2_step / ntrain / (T / step), train_l2_full / ntrain,
        test_l2_step / ntest / (T / step), test_l2_full / ntest
    )

t4 = default_timer()
print("total time = ", ((t4 - t3) / epochs))
