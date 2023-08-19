import os
import pickle
import random
import argparse
import numpy as np
from PIL import Image
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, \
    accuracy_score, precision_score, f1_score, recall_score

from tnc.models import WFEncoder
from tnc.equalizer import Equalizer
from tnc.augmentations import AugBoostDeep
from tnc.utils import _logger


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def epoch_train(model, equalizer, dataloader, lr=0.01, augboost=None):
    model.train()
    equalizer.train()

    criterion_cl = nn.CrossEntropyLoss()
    criterion_ctr = nn.CrossEntropyLoss(reduction='none')
    total_loss_cl, total_loss_ctr = [], []
    total_acc = []

    model_optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    equalizer_optimizer = torch.optim.Adam(equalizer.parameters(), lr=lr)

    for batch_idx, (x, y) in enumerate(dataloader):
        # load data
        x = x.to(device)    # x: (b, 2, 2500)
        y = y.to(device)

        # 1. Update classifier
        for param in equalizer.parameters():
            param.requires_grad = False
        for param in model.parameters():
            param.requires_grad = True
        model_optimizer.zero_grad()

        data_psi, _, _, _ = augboost(x, equalizer, y, None)
        predictions_psi = model(data_psi)
        loss_cl = criterion_cl(predictions_psi, y.long())
        total_loss_cl.append(loss_cl.item())
        total_acc.append(y.eq(predictions_psi.detach().argmax(dim=1)).float().mean())

        loss_cl.backward()
        model_optimizer.step()

        # 2. Update mapper
        for param in equalizer.parameters():
            param.requires_grad = True
        for param in model.parameters():
            param.requires_grad = False
        equalizer_optimizer.zero_grad()

        data_psi, data_rand, data_lsi, _ = augboost(x, equalizer, y, None)
        predictions_psi, predictions_rand, predictions_lsi = model(data_psi), model(data_rand), model(data_lsi)
        loss_ctr_psi, loss_ctr_rand, loss_ctr_lsi = criterion_ctr(predictions_psi, y.long()), criterion_ctr(predictions_rand, y.long()), criterion_ctr(predictions_lsi, y.long())
        # CE is calculated for each batch, and is averaged after.
        tmp_0 = torch.zeros_like(loss_ctr_psi).to(device)
        tmp_0.requires_grad = False
        loss_ctr_1 = torch.max(loss_ctr_psi - loss_ctr_rand + 0.010, tmp_0)
        loss_ctr_2 = torch.max(loss_ctr_psi - loss_ctr_lsi + 0.015, tmp_0)
        loss_ctr = (loss_ctr_1 + loss_ctr_2).mean()
        total_loss_ctr.append(loss_ctr.item())

        loss_ctr.backward()
        equalizer_optimizer.step()

    total_loss_cl = torch.tensor(total_loss_cl).mean()
    total_loss_ctr = torch.tensor(total_loss_ctr).mean()
    total_acc = torch.tensor(total_acc).mean()

    return total_loss_cl, total_loss_ctr, total_acc


def epoch_eval(model, equalizer, dataloader, experiment_log_dir):
    model.eval()
    if equalizer is not None:
        equalizer.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            # load data
            x = x.to(device)    # x: (b, 2, 2500)
            y = y.to(device)

            # plot learned prior
            if equalizer is not None:
                x_f = torch.fft.rfft(x)
                x_f = torch.cat((x_f.real, x_f.imag), dim=1)
                boost_ = equalizer(x_f).squeeze(1)

                if batch_idx == 0:
                    boost_all_ = boost_
                    y_all_ = y
                else:
                    boost_all_ = torch.cat((boost_all_, boost_), dim=0)
                    y_all_ = torch.cat((y_all_, y), dim=0)
            
            # inference
            predictions = model(x)

            # loss
            loss = criterion(predictions, y.long())
            total_loss += loss.item()

            # predictions and labels
            scores_np = predictions.detach().cpu().numpy()
            preds_np = np.argmax(scores_np, axis=1)
            labels_np = y.detach().cpu().numpy()
            onehots_np = F.one_hot(y.long(), num_classes=4).detach().cpu().numpy()

            if batch_idx == 0:
                scores_np_all, preds_np_all = scores_np, preds_np
                labels_np_all, onehots_np_all = labels_np, onehots_np
            else:
                scores_np_all = np.concatenate((scores_np_all, scores_np), axis=0)
                preds_np_all = np.concatenate((preds_np_all, preds_np), axis=0)
                labels_np_all = np.concatenate((labels_np_all, labels_np), axis=0)
                onehots_np_all = np.concatenate((onehots_np_all, onehots_np), axis=0)

    # plot learned prior
    if equalizer is not None:
        _, indices = y_all_.sort()
        boost_all_ = boost_all_.detach().cpu().numpy()
        indices = indices.detach().cpu().numpy()
        img = Image.fromarray((boost_all_[indices]*255).astype(np.uint8))
        img.save(os.path.join(experiment_log_dir, 'boost.png'))
    
    total_loss = total_loss / (batch_idx+1)

    # eval metrics
    acc = accuracy_score(labels_np_all, preds_np_all)
    precision = precision_score(labels_np_all, preds_np_all, average='macro')
    recall = recall_score(labels_np_all, preds_np_all, average='macro')
    F1 = f1_score(labels_np_all, preds_np_all, average='macro')

    auroc = roc_auc_score(onehots_np_all, scores_np_all, average="macro", multi_class="ovr")
    auprc = average_precision_score(onehots_np_all, scores_np_all, average="macro")

    return total_loss, acc, precision, recall, F1, auroc, auprc


def train(train_loader, valid_loader, model, equalizer, logger, experiment_log_dir, lr, n_epochs=100, args=None):
    logger.debug("Training started ....")

    best_valid_acc = 0.
    os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
    augboost = AugBoostDeep(aug_list=args.aug_list, prior=args.prior)

    for epoch in range(1, n_epochs+1):
        train_loss_cl, train_loss_ctr, train_acc = epoch_train(model, equalizer, train_loader, lr=lr, augboost=augboost)
        valid_loss, valid_acc, _, _, _, _, valid_auprc = epoch_eval(model, None, valid_loader, None)

        logger.debug(f'\nEpoch : {epoch}\n'
                     f'[Train] L_cl: {train_loss_cl:.3f} | L_ctr: {train_loss_ctr:.4f} | Acc: {train_acc:2.4f}\n'
                     f'[Valid] L_cl: {valid_loss:.3f} | Acc: {valid_acc:2.4f} | AUPRC: {valid_auprc:2.4f}')

        if valid_acc > best_valid_acc:
            logger.debug("Update best model ....")
            chkpoint = {'model_state_dict': model.state_dict(), 'equalizer_state_dict': equalizer.state_dict()}
            torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", 'ckp_best_valid_acc.pt'))
            best_valid_acc = valid_acc
        
        if epoch % 10 == 0:
            chkpoint = {'model_state_dict': model.state_dict(), 'equalizer_state_dict': equalizer.state_dict()}
            torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_epoch_{epoch}.pt'))

    logger.debug("\n################## Training is Done! #########################")


def run_test(data, e2e_lr, data_path, window_size, n_cross_val, args, logger, experiment_log_dir):
    # Load data
    with open(os.path.join(data_path, 'x_train.pkl'), 'rb') as f:
        x = pickle.load(f)
    with open(os.path.join(data_path, 'state_train.pkl'), 'rb') as f:
        y = pickle.load(f)
    with open(os.path.join(data_path, 'x_test.pkl'), 'rb') as f:
        x_test = pickle.load(f)
    with open(os.path.join(data_path, 'state_test.pkl'), 'rb') as f:
        y_test = pickle.load(f)
    
    T = x.shape[-1]
    x_window = np.split(x[:, :, :window_size * (T // window_size)], (T // window_size), -1)
    y_window = np.concatenate(np.split(y[:, :window_size * (T // window_size)], (T // window_size), -1), 0).astype(int)
    x_window = torch.Tensor(np.concatenate(x_window, 0))
    y_window = torch.Tensor(np.array([np.bincount(yy).argmax() for yy in y_window]))

    del x, y, x_test, y_test
    for cv in range(n_cross_val):
        shuffled_inds = list(range(len(x_window)))
        random.shuffle(shuffled_inds)
        x_window = x_window[shuffled_inds]
        y_window = y_window[shuffled_inds]
        n_train = int(0.7*len(x_window))
        X_train, X_test = x_window[:n_train], x_window[n_train:]
        y_train, y_test = y_window[:n_train], y_window[n_train:]

        trainset = torch.utils.data.TensorDataset(X_train, y_train)
        validset = torch.utils.data.TensorDataset(X_test, y_test)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=200, shuffle=False)
        valid_loader = torch.utils.data.DataLoader(validset, batch_size=200, shuffle=False)

        # Define baseline models
        encoding_size = 64
        n_classes = 4
        e2e_model = WFEncoder(encoding_size=encoding_size, classify=True, n_classes=n_classes).to(device)
        e2e_equalizer = Equalizer(None, args).to(device)

        # Train & Validate
        train(train_loader, valid_loader, e2e_model, e2e_equalizer, logger, experiment_log_dir, e2e_lr,\
              n_epochs=args.n_epoch, args=args)
        
        # Load best model
        chkpoint = torch.load(os.path.join(experiment_log_dir, "saved_models", 'ckp_best_valid_acc.pt'))
        e2e_model.load_state_dict(chkpoint['model_state_dict'])
        e2e_equalizer.load_state_dict(chkpoint['equalizer_state_dict'])
        logger.debug("Loaded best model")

        # Test
        # The waveform dataset is very small and sparse. If due to class imbalance there are no samples of a
        # particular class in the test set, report the validation performance
        outs = epoch_eval(e2e_model, e2e_equalizer, valid_loader, experiment_log_dir)
        test_loss, acc, precision, recall, F1, auroc, auprc = outs
        info_ = '[Test] Loss: %.3f | Acc: %.2f | Precision: %.2f | Recall: %.2f | F1: %.2f | AUROC: %.2f | AUPRC: %.2f'\
            % (test_loss, acc*100, precision*100, recall*100, F1*100, auroc*100, auprc*100)
        logger.debug(info_)

        torch.cuda.empty_cache()


if __name__=='__main__':
    start_time = datetime.now()
    parser = argparse.ArgumentParser(description='Run classification test')
    parser.add_argument('--logs_save_dir', default='experiments_logs', type=str, help='saving directory')
    parser.add_argument('--experiment_description', default='Exp1', type=str, help='Experiment Description')
    parser.add_argument('--run_description', default='run1', type=str, help='Experiment Description')
    parser.add_argument('--data', type=str, default='waveform')
    parser.add_argument('--cv', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_epoch', type=int, default=8)
    parser.add_argument('--aug_list', nargs='+', help='list of augs', required=True)
    parser.add_argument('--prior', default='self', type=str, help='self, none')
    parser.add_argument('--equalizer', default='transformer', type=str, help='transformer')
    args = parser.parse_args()

    experiment_description = args.experiment_description
    run_description = args.run_description
    logs_save_dir = args.logs_save_dir
    os.makedirs(logs_save_dir, exist_ok=True)

    # fix random seeds for reproducibility
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    random.seed(SEED)

    experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description, f"seed_{SEED}")
    os.makedirs(experiment_log_dir, exist_ok=True)

    # Logging
    log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
    logger = _logger(log_file_name)
    logger.debug("=" * 45)
    logger.debug(f'Dataset: ECG')
    logger.debug(f'Mode:    supervised')
    logger.debug("=" * 45)

    # Log arguments
    logger.debug(args)

    run_test(data='waveform', e2e_lr=0.0001, data_path='./data/waveform_data/processed', window_size=2500, n_cross_val=args.cv, args=args,\
             logger=logger, experiment_log_dir=experiment_log_dir)

    logger.debug(f"Training time is : {datetime.now()-start_time}")
