import random
import numpy as np
from scipy.interpolate import CubicSpline

import torch
import torch.nn as nn
import torch.nn.functional as F

class Freqdropout(nn.Module):
    def __init__(self, rate=0.2):
        super().__init__()
        self.rate = rate
        self.p = 0.5
    
    def forward(self, x_t):
        # x_t: (b, c, timestep)
        mask = torch.rand(x_t.shape[0],1,1).to(x_t.device) < self.p

        x_f = torch.fft.rfft(x_t, dim=-1)
        m = torch.cuda.FloatTensor(x_f.shape).uniform_() < self.rate
        freal = x_f.real.masked_fill(m, 0)
        fimag = x_f.imag.masked_fill(m, 0)
        x_f_aug = torch.complex(freal, fimag)
        x_t_aug = torch.fft.irfft(x_f_aug, dim=-1)
        return x_t*(~mask) + x_t_aug*mask


class Dropout(nn.Module):
    def __init__(self, p_zero=0.2):
        super().__init__()
        self.dropout = nn.Dropout(p_zero)
        self.p = 0.5
    
    def forward(self, x_t):
        # x_t: (b, c, timestep)
        mask = torch.rand(x_t.shape[0],1,1).to(x_t.device) < self.p
        x_t_masked = self.dropout(x_t)
        return x_t*(~mask) + x_t_masked*mask


class Jitter(nn.Module):
    def __init__(self, sigma=0.5):
        super().__init__()
        self.sigma = sigma
        self.p = 0.5

    def forward(self, x_t):
        # x_t: (b, c, timestep)
        mask = torch.rand(x_t.shape[0],1,1).to(x_t.device) < self.p
        return x_t + mask * (torch.randn(x_t.shape).to(x_t.device) * self.sigma)

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"sigma={self.sigma})"
        )
        return s


class Scale(nn.Module):
    def __init__(self, sigma=0.5):
        super().__init__()
        self.sigma = sigma
        self.p = 0.5

    def forward(self, x):
        # x: (b, c, timestep)
        mask = torch.rand(x.shape[0],1,1).to(x.device) < self.p
        return x * (mask * torch.randn((x.shape[0], x.shape[1], 1)).to(x.device) * self.sigma + 1)

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"sigma={self.sigma})"
        )
        return s


class Shift(nn.Module):
    def __init__(self, sigma=0.5):
        super().__init__()
        self.sigma = sigma
        self.p = 0.5

    def forward(self, x):
        # x: (b, c, timestep)
        mask = torch.rand(x.shape[0],1,1).to(x.device) < self.p
        return x + (mask * torch.randn((x.shape[0], x.shape[1], 1)).to(x.device) * self.sigma)

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"sigma={self.sigma})"
        )
        return s


def GenerateRandomCurves(X, sigma, knot):
    # X: (b, timestep, c)
    randcurv = np.empty(shape=X.shape)

    xx = (np.ones((X.shape[2], 1))*(np.arange(0, X.shape[1], (X.shape[1]-1)/(knot+1)))).transpose()    # (knot+2, c)
    yy = np.random.normal(loc=1.0, scale=sigma, size=(X.shape[0], knot+2, X.shape[2]))                 # (b, knot+2, c)
    x_range = np.arange(X.shape[1])

    for b in range(X.shape[0]):
        for c in range(X.shape[2]):
            cs = CubicSpline(xx[:,c], yy[b,:,c])
            randcurv[b,:,c] = cs(x_range)
    return randcurv


class Magwarp(nn.Module):
    def __init__(self, sigma=0.2, knot=4):
        super().__init__()
        self.sigma = sigma
        self.knot = knot
        self.p = 0.5

    def forward(self, x_t):
        # x: (b, c, timestep)
        mask = torch.rand(x_t.shape[0],1,1).to(x_t.device) < self.p
        randcurv_numpy = GenerateRandomCurves(x_t.permute(0,2,1).cpu().numpy(), self.sigma, self.knot)
        randcurv_torch = torch.from_numpy(randcurv_numpy).float().permute(0,2,1).to(x_t.device)
        x_t_mw = x_t * randcurv_torch
        return x_t*(~mask) + x_t_mw*mask

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"sigma={self.sigma}"
            f", knot={self.knot})"
        )
        return s


class Timewarp(nn.Module):
    def __init__(self, sigma=0.2, knot=4):
        super().__init__()
        self.sigma = sigma
        self.knot = knot
        self.p = 0.5

    def forward(self, x_t):
        # x_t: (b, c, timestep)
        mask = torch.rand(x_t.shape[0],1,1).to(x_t.device) < self.p
        x_numpy = x_t.permute(0,2,1).cpu().numpy()
        tt_new = self.DistortTimesteps(x_numpy)
        X_new = np.zeros(x_numpy.shape)
        x_range = np.arange(x_numpy.shape[1])
        for b in range(x_numpy.shape[0]):
            for c in range(x_numpy.shape[2]):
                X_new[b,:,c] = np.interp(x_range, tt_new[b,:,c], x_numpy[b,:,c])
        x_t_tw = torch.from_numpy(X_new).float().permute(0,2,1).to(x_t.device)
        return x_t*(~mask) + x_t_tw*mask
    
    def DistortTimesteps(self, X):
        # X: (b, timestep, c)
        tt = GenerateRandomCurves(X, self.sigma, self.knot)
        tt_cum = np.cumsum(tt, axis=1)

        for b in range(X.shape[0]):
            for c in range(X.shape[2]):
                tt_cum[b,:,c] = tt_cum[b,:,c] * (X.shape[1]-1)/tt_cum[b,-1,c]
        return tt_cum

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"sigma={self.sigma}"
            f", knot={self.knot})"
        )
        return s


class Permute(nn.Module):
    def __init__(self, num_segments=10):
        super().__init__()
        self.num_segments = num_segments
        self.p = 0.5

    def forward(self, x_t):
        # x: (b, c, timestep)
        mask = torch.rand(x_t.shape[0],1,1).to(x_t.device) < self.p
        aug_np = self.time_segment_permutation_transform_improved(x_t.permute(0,2,1).cpu().numpy())
        x_t_perm = torch.from_numpy(aug_np).float().permute(0,2,1).to(x_t.device)
        return x_t*(~mask) + x_t_perm*mask
    
    def time_segment_permutation_transform_improved(self, X):
        """
        Randomly scrambling sections of the signal
        adopted from simclr_har paper; X has size: (batch, timestep, channel), type: np.array
        """
        segment_points_permuted = np.random.choice(X.shape[1], size=(X.shape[0], self.num_segments-1))
        segment_points = np.sort(segment_points_permuted, axis=1)

        X_transformed = np.empty(shape=X.shape)
        for i, (sample, segments) in enumerate(zip(X, segment_points)):
            splitted = np.array(np.split(sample, np.append(segments, X.shape[1])))
            np.random.shuffle(splitted)
            concat = np.concatenate(splitted, axis=0)
            X_transformed[i] = concat
        return X_transformed

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_segments={self.num_segments})"
        )
        return s


'''
SimPSI: A Simple Strategy to Preserve Spectral Information
'''
class AugBoostDeep(nn.Module):
    def __init__(self, aug_list=[], prior='self'):
        super().__init__()
        self.aug_list = aug_list
        self.prior = prior
    
    def forward(self, x_t, eq, y, model):
        # x_t: (b, c, timestep)
        # x_f: (b, c*2, timestep//2 + 1)
        x_t_org = x_t.clone().detach()
        x_f = torch.fft.rfft(x_t)
        x_f_mag = x_f.abs()
        x_f = torch.cat((x_f.real, x_f.imag), dim=1)
        
        # Aug
        for aug in self.aug_list:
            aug_ = eval(aug.capitalize())()
            x_t = aug_(x_t)
        x_t_aug = x_t

        if self.prior == 'none':
            return x_t_aug, None, None, None
        
        # x_t_aug -> x_f_aug
        x_f_aug = torch.fft.rfft(x_t_aug)
        x_f_aug = torch.cat((x_f_aug.real, x_f_aug.imag), dim=1)

        # Boost the prior
        if self.prior == 'mag':
            boost_ = eq(x_f_mag)        # x_f_mag: (b, c, timestep//2 + 1)
        elif self.prior == 'slc':
            # stop gradient
            for param in model.parameters():
                param.requires_grad = False
            model.eval()

            # calculate saliency map
            x_f_ = torch.fft.rfft(x_t_org).clone().detach() # x_f_: (b, c, timestep//2 + 1)
            x_f_.requires_grad = True
            preds, _ = model(torch.fft.irfft(x_f_, n=x_t_org.shape[-1]))
            score = torch.zeros(y.shape).cuda()
            for i in range(y.shape[0]):
                score[i] = preds[i, y[i]]
            score.mean().backward()
            boost_ = eq(torch.abs(x_f_.grad.clone().detach()))

            # resume gradient
            for param in model.parameters():
                param.requires_grad = True
            model.train()
        elif self.prior == 'self':
            boost_ = eq(x_f)            # x_f: (b, c*2, timestep//2 + 1)
        elif self.prior == 'rnd':
            boost_ = torch.rand(x_f.shape[0], 1, x_f.shape[2]).cuda()
        else:
            raise ValueError("Not available prior.")
        x_f_aug_psi = x_f*boost_ + x_f_aug*(1-boost_)

        # contrastive pairs
        boost_rand_ = torch.rand_like(boost_)
        x_f_aug_rand = x_f*boost_rand_ + x_f_aug*(1-boost_rand_)
        x_f_aug_lsi = x_f*(1-boost_) + x_f_aug*boost_

        # x_f_aug_{.} -> x_t_aug_{.}
        C = x_t.shape[1]
        x_f_aug_psi = x_f_aug_psi[:, :C, :] + torch.tensor(1.j)*x_f_aug_psi[:, C:, :]
        x_f_aug_rand = x_f_aug_rand[:, :C, :] + torch.tensor(1.j)*x_f_aug_rand[:, C:, :]
        x_f_aug_lsi = x_f_aug_lsi[:, :C, :] + torch.tensor(1.j)*x_f_aug_lsi[:, C:, :]
        x_t_aug_psi = torch.fft.irfft(x_f_aug_psi, n=x_t.shape[-1])
        x_t_aug_rand = torch.fft.irfft(x_f_aug_rand, n=x_t.shape[-1])
        x_t_aug_lsi = torch.fft.irfft(x_f_aug_lsi, n=x_t.shape[-1])

        return x_t_aug_psi, x_t_aug_rand, x_t_aug_lsi, boost_
