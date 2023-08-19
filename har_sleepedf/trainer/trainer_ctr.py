import os
import sys

sys.path.append("..")
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, \
    accuracy_score, precision_score, f1_score, recall_score


def Trainer(model, equalizer, model_optimizer, equalizer_optimizer,\
            train_dl, valid_dl, device, logger, config, experiment_log_dir, augboost):
    # Start training
    logger.debug("Training started ....")
    best_valid_acc = 0.

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')
    os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)

    for epoch in range(1, config.num_epoch + 1):
        # Train and validate
        train_loss_cl, train_loss_ctr, train_acc = model_train(model, equalizer, model_optimizer, equalizer_optimizer,\
                                                                train_dl, device, augboost, epoch)
        valid_loss, valid_acc, _, _, _, _, valid_auprc = model_evaluate(model, valid_dl, device, config, None, None, None)
        scheduler.step(valid_loss)

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


def model_train(model, equalizer, model_optimizer, equalizer_optimizer,\
                train_loader, device, augboost, epoch):
    criterion_cl = nn.CrossEntropyLoss()
    criterion_ctr = nn.CrossEntropyLoss(reduction='none')
    total_loss_cl, total_loss_ctr = [], []
    total_acc = []
    model.train()
    equalizer.train()

    for batch_idx, (data, labels, _, _) in enumerate(train_loader):
        # load data
        data, labels = data.float().to(device), labels.long().to(device)

        # 1. Update classifier
        for param in equalizer.parameters():
            param.requires_grad = False
        for param in model.parameters():
            param.requires_grad = True
        model_optimizer.zero_grad()

        data_psi, _, _, _ = augboost(data, equalizer, labels, model)
        predictions_psi = model(data_psi)[0]
        loss_psi = criterion_cl(predictions_psi, labels)
        loss_cl = loss_psi
        total_loss_cl.append(loss_cl.item())
        total_acc.append(labels.eq(predictions_psi.detach().argmax(dim=1)).float().mean())

        loss_cl.backward()
        model_optimizer.step()

        # 2. Update mapper
        for param in equalizer.parameters():
            param.requires_grad = True
        for param in model.parameters():
            param.requires_grad = False
        equalizer_optimizer.zero_grad()

        data_psi, data_rand, data_lsi, _ = augboost(data, equalizer, labels, model)
        predictions_psi, predictions_rand, predictions_lsi = model(data_psi)[0], model(data_rand)[0], model(data_lsi)[0]
        loss_ctr_psi, loss_ctr_rand, loss_ctr_lsi = criterion_ctr(predictions_psi, labels), criterion_ctr(predictions_rand, labels), criterion_ctr(predictions_lsi, labels)
        # CE is calculated for each batch, and is averaged after.
        tmp_0 = torch.zeros_like(loss_psi).to(device)
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


def model_evaluate(model, test_dl, device, config, equalizer, experiment_log_dir, arg):
    model.eval()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (data, labels, _, _) in enumerate(test_dl):
            # load data
            data, labels = data.float().to(device), labels.long().to(device)

            # plot leanred P
            if equalizer is not None:
                x_f = torch.fft.rfft(data)
                x_f_mag = x_f.abs()
                x_f = torch.cat((x_f.real, x_f.imag), dim=1)
                if arg.prior == 'self':
                    boost_ = equalizer(x_f).squeeze(1)
                elif arg.prior == 'mag':
                    boost_ = equalizer(x_f_mag).squeeze(1)
                else:
                    pass

                if batch_idx == 0:
                    boost_all_ = boost_
                    y_all_ = labels
                else:
                    boost_all_ = torch.cat((boost_all_, boost_), dim=0)
                    y_all_ = torch.cat((y_all_, labels), dim=0)

            # inference
            predictions, _ = model(data)

            # loss
            loss = criterion(predictions, labels)
            total_loss += loss.item()

            # predictions and labels
            scores_np = predictions.detach().cpu().numpy()
            preds_np = np.argmax(scores_np, axis=1)
            labels_np = labels.detach().cpu().numpy()
            onehots_np = F.one_hot(labels, num_classes=config.num_classes).detach().cpu().numpy()

            if batch_idx == 0:
                scores_np_all, preds_np_all = scores_np, preds_np
                labels_np_all, onehots_np_all = labels_np, onehots_np
            else:
                scores_np_all = np.concatenate((scores_np_all, scores_np), axis=0)
                preds_np_all = np.concatenate((preds_np_all, preds_np), axis=0)
                labels_np_all = np.concatenate((labels_np_all, labels_np), axis=0)
                onehots_np_all = np.concatenate((onehots_np_all, onehots_np), axis=0)

    # plot leanred P
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
