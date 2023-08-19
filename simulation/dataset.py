import numpy as np
from scipy import io
from glob import glob

import torch
from torch.utils.data import Dataset


class SignalSet(Dataset):
    def __init__(self, root, mode, n_snr):
        self.input_sig_dict = dict()
        self.gt_bit_dict = dict()
        self.n_snr = n_snr
        self.list_snr = list(28-2*np.arange(self.n_snr))
        self.list_snr.reverse()

        ref_point_1 = int(288*0.8)
        ref_point_2 = int(288*0.9)
        if mode == 'train':
            start, end = 0, ref_point_1
        elif mode == 'valid':
            start, end = ref_point_1, ref_point_2
        else:
            start, end = ref_point_2, 288
        self.n_inst = end - start

        mat_list = glob(root + '/*.mat')
        for mat in mat_list:
            snr = str(mat.split('/')[-1][:-6])
            self.input_sig_dict[snr] = io.loadmat(mat)['yt'][start:end, :]
            self.gt_bit_dict[snr] = io.loadmat(mat)['xt'][start:end, :]

    def __getitem__(self, index):
        ind_snr = index // self.n_inst
        ind_inst = index % self.n_inst

        snr = str(self.list_snr[ind_snr])
        input_ = self.input_sig_dict[snr][ind_inst, :]
        gt_ = self.gt_bit_dict[snr][ind_inst, :]
        
        return {'yt': input_, 'xt': gt_, 'snr': snr}

    def __len__(self):
        return self.n_snr * self.n_inst