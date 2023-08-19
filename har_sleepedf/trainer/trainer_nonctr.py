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

    for epoch in range(1, config.num_epoch + 1):
        # Train and validate
        train_loss_cl, train_acc = model_train(model, equalizer, model_optimizer, equalizer_optimizer,\
                                                train_dl, device, augboost)
        valid_loss, valid_acc, _, _, _, _, valid_auprc = model_evaluate(model, valid_dl, device, config, None, None, None)
        scheduler.step(valid_loss)

        logger.debug(f'\nEpoch : {epoch}\n'
                     f'[Train] Loss: {train_loss_cl:.3f} | Acc: {train_acc:2.4f}\n'
                     f'[Valid] Loss: {valid_loss:.3f} | Acc: {valid_acc:2.4f} | AUPRC: {valid_auprc:2.4f}')

        if valid_acc > best_valid_acc:
            logger.debug("Update best model ....")
            os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
            chkpoint = {'model_state_dict': model.state_dict(), 'equalizer_state_dict': equalizer.state_dict()}
            torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", 'ckp_best_valid_acc.pt'))
            best_valid_acc = valid_acc

    os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
    chkpoint = {'model_state_dict': model.state_dict(), 'equalizer_state_dict': equalizer.state_dict()}
    torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", 'ckp_last.pt'))

    logger.debug("\n################## Training is Done! #########################")


def model_train(model, equalizer, model_optimizer, equalizer_optimizer,\
                train_loader, device, augboost):
    criterion = nn.CrossEntropyLoss()
    total_loss = []
    total_acc = []
    model.train()
    equalizer.train()

    for batch_idx, (data, labels, _, _) in enumerate(train_loader):
        # load data
        data, labels = data.float().to(device), labels.long().to(device)

        model_optimizer.zero_grad()
        equalizer_optimizer.zero_grad()

        data_psi, _, _, _ = augboost(data, equalizer, labels, model)
        # data_psi = data   # no augmentation
        predictions = model(data_psi)[0]
        loss = criterion(predictions, labels)
        total_loss.append(loss.item())
        total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())

        loss.backward()
        model_optimizer.step()
        equalizer_optimizer.step()

    total_loss = torch.tensor(total_loss).mean()
    total_acc = torch.tensor(total_acc).mean()

    return total_loss, total_acc


def model_evaluate(model, test_dl, device, config, equalizer, experiment_log_dir, arg):
    model.eval()
    if equalizer is not None:
        equalizer.eval()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (data, labels, _, _) in enumerate(test_dl):
            # load data
            data, labels = data.float().to(device), labels.long().to(device)

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

    total_loss = total_loss / (batch_idx+1)

    # eval metrics
    acc = accuracy_score(labels_np_all, preds_np_all)
    precision = precision_score(labels_np_all, preds_np_all, average='macro')
    recall = recall_score(labels_np_all, preds_np_all, average='macro')
    F1 = f1_score(labels_np_all, preds_np_all, average='macro')

    auroc = roc_auc_score(onehots_np_all, scores_np_all, average="macro", multi_class="ovr")
    auprc = average_precision_score(onehots_np_all, scores_np_all, average="macro")

    return total_loss, acc, precision, recall, F1, auroc, auprc
