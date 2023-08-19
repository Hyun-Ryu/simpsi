import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pyplot
import os
import pdb
import time
import argparse
import datetime
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model import ResNet1D
from dataset import SignalSet

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=64, help='size of the batches')
parser.add_argument("--n_cpu", type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument("--root", type=str, default="/", help='root directory')
parser.add_argument("--data_name", type=str, default="", help='name of the dataset')
parser.add_argument("--exp_name", type=str, default="", help='name of the experiment')
parser.add_argument("--n_class", type=int, default=32, help='number of classes')
parser.add_argument('--seed', default=0, type=int, help='seed value')
opt = parser.parse_args()
print(str(opt) + "\n")

SEED = opt.seed
experiment_log_dir = opt.root + "/experiments/" + opt.exp_name + f"/seed_{SEED}"

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

# Load Models
model = ResNet1D(dim_input=2, dim_hidden=32, dim_output=opt.n_class, pool=True).cuda()
model.load_state_dict(torch.load(experiment_log_dir+'/saved_models/model_epoch_best.pth'))

# Loss
CE = torch.nn.CrossEntropyLoss().cuda()

# Dataset & Dataloader
dataset_test = SignalSet(root=opt.root+'/data/'+opt.data_name, mode='test', n_snr=10)
dataloader_test = DataLoader(
    dataset_test,
    batch_size = opt.batch_size,
    shuffle = False,
    num_workers = opt.n_cpu,
)

prev_time = time.time()

# Testing
loss_test_tot = 0
num_correct_tot_test, num_data_test = 0, 0

for t, sigg in enumerate(dataloader_test):

    # Configure input & gt
    input_ = sigg['yt'].unsqueeze(1).unsqueeze(1)
    input_ = torch.cat((input_.real.type(Tensor), input_.imag.type(Tensor)), dim=1)
    input_ = Variable(input_)                    # input_: (b, 2, 1, 128)
    gt_ = Variable(sigg['xt'].type(LongTensor))  # gt_   : (b, 16)

    # --------------------
    # Inferenece
    # --------------------

    model.eval()

    # model
    output_ = model(input_)          # output_: (b, 2, 16)

    loss_test = CE(output_, gt_)
    loss_test_tot += loss_test.item()

    num_correct_test = (torch.max(output_, dim=1)[1].data==gt_.data).sum()
    num_correct_tot_test += num_correct_test
    num_data_test += (gt_.shape[0] * gt_.shape[1])

print(
    "---Summary---\n\r[Test] [CE: %.4f, Acc: %.2f%%]"
    % (
        loss_test_tot/len(dataloader_test),
        num_correct_tot_test/num_data_test * 100,
    )
)
