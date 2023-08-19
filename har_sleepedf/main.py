import os
import torch
import argparse
import numpy as np
from datetime import datetime

from dataloader.dataloader import data_generator
from models.model import *
from models.equalizer import Equalizer
from dataloader.augmentations import AugBoostDeep
from utils import _logger

# Args selections
start_time = datetime.now()
parser = argparse.ArgumentParser()

######################## Model parameters ########################
home_dir = os.getcwd()
parser.add_argument('--experiment_description', default='Exp1', type=str, help='Experiment Description')
parser.add_argument('--run_description', default='run1', type=str, help='Experiment Description')
parser.add_argument('--seed', default=0, type=int, help='seed value')
parser.add_argument('--training_mode', default='supervised', type=str, help='supervised')
parser.add_argument('--selected_dataset', default='HAR', type=str, help='HAR, SleepEDF')
parser.add_argument('--logs_save_dir', default='experiments_logs', type=str, help='saving directory')
parser.add_argument('--device', default='cuda', type=str, help='cpu or cuda')
parser.add_argument('--home_path', default=home_dir, type=str, help='Project home directory')
parser.add_argument('--aug_list', nargs='+', help='list of augs', required=True)
parser.add_argument('--prior', default='self', type=str, help='mag, slc, self, rnd, none')
parser.add_argument('--equalizer', default='transformer', type=str, help='transformer, conv')
parser.add_argument('--eq_kernel_size', default=9, type=int, help='kernel size')
parser.add_argument('--mode', default='ctr', type=str, help='ctr, ce')
args = parser.parse_args()

if args.mode == 'ctr':
    from trainer.trainer_ctr import Trainer, model_evaluate
elif args.mode == 'ce':
    from trainer.trainer_nonctr import Trainer, model_evaluate
else:
    pass

device = torch.device(args.device)
experiment_description = args.experiment_description
data_type = args.selected_dataset
training_mode = args.training_mode
run_description = args.run_description

logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)


exec(f'from config_files.{data_type}_Configs import Config as Configs')
configs = Configs()

####### fix random seeds for reproducibility ########
SEED = args.seed
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
#####################################################

experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description, f"seed_{SEED}")
os.makedirs(experiment_log_dir, exist_ok=True)


# Logging
log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
logger = _logger(log_file_name)
logger.debug("=" * 45)
logger.debug(f'Dataset: {data_type}')
logger.debug(f'Mode:    {training_mode}')
logger.debug("=" * 45)

# Log arguments
logger.debug(args)

# Load datasets
data_path = f"./data/{data_type}"
train_dl, valid_dl, test_dl = data_generator(data_path, configs, training_mode)
logger.debug("Data loaded ...")

# Load Model
model = base_Model(configs).to(device)
# model = Transformer_Model(configs).to(device)
# model = LSTM(configs).to(device)
equalizer = Equalizer(configs, args).to(device)

# Optimizer
model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
equalizer_optimizer = torch.optim.Adam(equalizer.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)

# Trainer
augboost = AugBoostDeep(aug_list=args.aug_list, prior=args.prior)

Trainer(model, equalizer, model_optimizer, equalizer_optimizer, \
        train_dl, valid_dl, device, logger, configs, experiment_log_dir, augboost)

# Load best model
chkpoint = torch.load(os.path.join(experiment_log_dir, "saved_models", 'ckp_best_valid_acc.pt'))
model.load_state_dict(chkpoint['model_state_dict'])
equalizer.load_state_dict(chkpoint['equalizer_state_dict'])
logger.debug("Loaded best model")

# Testing
outs = model_evaluate(model, test_dl, device, configs, equalizer, experiment_log_dir, args)
test_loss, acc, precision, recall, F1, auroc, auprc = outs
info_ = '[Test] Loss: %.3f | Acc: %.2f | Precision: %.2f | Recall: %.2f | F1: %.2f | AUROC: %.2f | AUPRC: %.2f'\
            % (test_loss, acc*100, precision*100, recall*100, F1*100, auroc*100, auprc*100)
logger.debug(info_)

logger.debug(f"Training time is : {datetime.now()-start_time}")
