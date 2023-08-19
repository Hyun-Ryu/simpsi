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

from model import ResNet1D, Equalizer
from dataset import SignalSet
from augmentations import AugBoostDeep
from utils import _logger

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=50, help='number of training epochs')
parser.add_argument("--batch_size", type=int, default=64, help='size of the batches')
parser.add_argument("--lr", type=float, default=1e-3, help='learning rate')
parser.add_argument("--n_cpu", type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument("--root", type=str, default="/", help='root directory')
parser.add_argument("--data_name", type=str, default="", help='name of the dataset')
parser.add_argument("--exp_name", type=str, default="", help='name of the experiment')
parser.add_argument("--n_class", type=int, default=32, help='number of classes')
parser.add_argument('--aug_list', nargs='+', help='list of augs', required=True)
parser.add_argument('--prior', default='self', type=str, help='self')
parser.add_argument('--equalizer', default='transformer', type=str, help='transformer')
parser.add_argument('--seed', default=0, type=int, help='seed value')
opt = parser.parse_args()

# Seed
SEED = opt.seed
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

experiment_log_dir = opt.root + "/experiments/" + opt.exp_name + f"/seed_{SEED}"
os.makedirs(experiment_log_dir + "/saved_models", exist_ok=True)

# Logging
log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
logger = _logger(log_file_name)
logger.debug(opt)

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

# Load Models
model = ResNet1D(dim_input=2, dim_hidden=32, dim_output=opt.n_class, pool=True).cuda()
equalizer = Equalizer(opt).cuda()

# Loss
criterion_cl = torch.nn.CrossEntropyLoss().cuda()
criterion_ctr = torch.nn.CrossEntropyLoss(reduction='none').cuda()

# Optimizers
model_optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
equalizer_optimizer = torch.optim.Adam(equalizer.parameters(), lr=opt.lr)

# SimPSI
augboost = AugBoostDeep(aug_list=opt.aug_list, prior=opt.prior)

# Dataset & Dataloader
dataset = SignalSet(root=opt.root+'/data/'+opt.data_name, mode='train', n_snr=10)
dataloader = DataLoader(
    dataset,
    batch_size = opt.batch_size,
    shuffle = True,
    num_workers = opt.n_cpu,
)

dataset_valid = SignalSet(root=opt.root+'/data/'+opt.data_name, mode='valid', n_snr=10)
dataloader_valid = DataLoader(
    dataset_valid,
    batch_size = opt.batch_size,
    shuffle = False,
    num_workers = opt.n_cpu,
)

loss_epoch_list, loss_epoch_list_val = [], []
acc_epoch_list, acc_epoch_list_val = [], []
acc_top1 = 0
prev_time = time.time()

for epoch in range(1, opt.n_epochs+1):
    # Training
    loss_cl_tot, loss_ctr_tot = 0, 0
    num_correct_tot, num_data = 0, 0
    model.train()
    equalizer.train()

    for i, sig in enumerate(dataloader):
        # Configure input & gt
        input_ = Variable(sig['yt'].unsqueeze(1).cuda())   # input_: (b, 1, 128) -> cplx
        gt_ = Variable(sig['xt'].type(LongTensor))  # gt_   : (b, 16)

        # --------------------
        # Train Model
        # --------------------

        # 1. Update classifier
        for param in equalizer.parameters():
            param.requires_grad = False
        for param in model.parameters():
            param.requires_grad = True
        model_optimizer.zero_grad()

        data_psi, _, _, _ = augboost(input_, equalizer, gt_, None, model)
        output_psi = model(data_psi.unsqueeze(2))

        loss_cl = criterion_cl(output_psi, gt_)
        loss_cl_tot += loss_cl.item()

        num_correct = (torch.max(output_psi, dim=1)[1].data==gt_.data).sum()
        num_correct_tot += num_correct
        num_data += (gt_.shape[0] * gt_.shape[1])

        loss_cl.backward()
        model_optimizer.step()

        # 2. Update equalizer
        for param in equalizer.parameters():
            param.requires_grad = True
        for param in model.parameters():
            param.requires_grad = False
        equalizer_optimizer.zero_grad()

        data_psi, data_rand, data_lsi, _ = augboost(input_, equalizer, gt_, None, model)
        output_psi, output_rand, output_lsi = model(data_psi.unsqueeze(2)), model(data_rand.unsqueeze(2)), model(data_lsi.unsqueeze(2))
        loss_ctr_psi, loss_ctr_rand, loss_ctr_lsi = criterion_ctr(output_psi, gt_), criterion_ctr(output_rand, gt_), criterion_ctr(output_lsi, gt_)
        # CE is calculated for each batch, and is averaged after.
        tmp_0 = torch.zeros_like(loss_cl).cuda()
        tmp_0.requires_grad = False
        loss_ctr_1 = torch.max(loss_ctr_psi - loss_ctr_rand + 0.010, tmp_0)
        loss_ctr_2 = torch.max(loss_ctr_psi - loss_ctr_lsi + 0.015, tmp_0)
        loss_ctr = (loss_ctr_1 + loss_ctr_2).mean()
        loss_ctr_tot += loss_ctr.item()

        loss_ctr.backward()
        equalizer_optimizer.step()
 
        # --------------------
        # Log Progress
        # --------------------

        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds = batches_left * (time.time() - prev_time))
        prev_time = time.time()

        if i % 10 == 0:
            logger.debug(
                "\r[Epoch %d/%d, Batch %d/%d] [L_cl: %.4f, L_ctr: %.4f, Acc: %.2f%%] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_cl.item(),
                    loss_ctr.item(),
                    num_correct/(gt_.shape[0] * gt_.shape[1]) * 100,
                    time_left,
                )
            )

    # Validation
    loss_valid_tot = 0
    num_correct_tot_valid, num_data_valid = 0, 0

    for t, sigg in enumerate(dataloader_valid):

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

        loss_valid = criterion_cl(output_, gt_)
        loss_valid_tot += loss_valid.item()

        num_correct_valid = (torch.max(output_, dim=1)[1].data==gt_.data).sum()
        num_correct_tot_valid += num_correct_valid
        num_data_valid += (gt_.shape[0] * gt_.shape[1])
        
        # --------------------
        # Log Progress
        # --------------------

        if t % 5 == 0:
            logger.debug(
                    "\r[Epoch %d/%d] [L_cl: %.4f, Acc: %.2f%%]"
                % (
                    epoch,
                    opt.n_epochs,
                    loss_valid.item(),
                    num_correct_valid/(gt_.shape[0] * gt_.shape[1]) * 100,
                )
            )

    logger.debug(
        "---Summary of Epoch %d/%d---\n\r[Train] [L_cl: %.4f, L_ctr: %.4f, Acc: %.2f%%]\n\r[Valid] [L_cl: %.4f, Acc: %.2f%%]"
        % (
            epoch,
            opt.n_epochs,
            loss_cl_tot/len(dataloader),
            loss_ctr_tot/len(dataloader),
            num_correct_tot/num_data * 100,
            loss_valid_tot/len(dataloader_valid),
            num_correct_tot_valid/num_data_valid * 100,
        )
    )

    if (num_correct_tot_valid/num_data_valid * 100) > acc_top1:
        acc_top1 = num_correct_tot_valid/num_data_valid * 100
        loss_top1 = loss_valid_tot/len(dataloader_valid)
        epoch_top1 = epoch
        torch.save(model.state_dict(), experiment_log_dir+'/saved_models/model_epoch_best.pth')
        torch.save(equalizer.state_dict(), experiment_log_dir+'/saved_models/equalizer_epoch_best.pth')

    if epoch % 10 == 0:
        # save model checkpoint
        torch.save(model.state_dict(), experiment_log_dir+'/saved_models/model_epoch_%d.pth' % epoch)
        torch.save(equalizer.state_dict(), experiment_log_dir+'/saved_models/equalizer_epoch_%d.pth' % epoch)
        
    logger.debug(
        "---Summary of TOP1---\n\r[Valid] [Epoch: %d/%d, L_cl: %.4f, Acc: %.2f%%]"
        % (
            epoch_top1,
            opt.n_epochs,
            loss_top1,
            acc_top1,
        )
    )
