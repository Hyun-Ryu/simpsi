import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np


class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset, config, training_mode, moments_train):
        super(Load_Dataset, self).__init__()
        self.training_mode = training_mode
        mean_train, std_train = moments_train["mean"], moments_train["std"]

        X_train = dataset["samples"]
        y_train = dataset["labels"]

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
            X_train = X_train.permute(0, 2, 1)

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train

        self.len = X_train.shape[0]
        
        # Normalize to unit Gaussian
        self.x_data = (self.x_data - mean_train) / std_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index], self.x_data[index], self.x_data[index]

    def __len__(self):
        return self.len


def data_generator(data_path, configs, training_mode):

    train_dataset = torch.load(os.path.join(data_path, "train.pt"))
    valid_dataset = torch.load(os.path.join(data_path, "val.pt"))
    test_dataset = torch.load(os.path.join(data_path, "test.pt"))

    # Compute mean and std of each variable for training set
    samples_ = train_dataset["samples"]     # (N, c, ts) must be ensured
    assert samples_.shape.index(min(samples_.shape)) == 1, 'Bro check the sample dimension'
    mean_train = torch.mean(samples_, dim=(0,2), keepdim=True)
    std_train = torch.std(samples_, dim=(0,2), keepdim=True)
    moments_train = dict({'mean': mean_train, 'std': std_train})

    train_dataset = Load_Dataset(train_dataset, configs, training_mode, moments_train)
    valid_dataset = Load_Dataset(valid_dataset, configs, training_mode, moments_train)
    test_dataset = Load_Dataset(test_dataset, configs, training_mode, moments_train)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=configs.batch_size,
                                               shuffle=False, drop_last=configs.drop_last,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.batch_size,
                                              shuffle=False, drop_last=False,
                                              num_workers=0)

    return train_loader, valid_loader, test_loader