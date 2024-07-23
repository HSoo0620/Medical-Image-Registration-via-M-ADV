import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import math
import torch.nn as nn


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size,
                                                                  window_size).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def _ssim_3D(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv3d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv3d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv3d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


class SSIM3D(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM3D, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window_3D(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window_3D(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return 1-_ssim_3D(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def ssim3D(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _, _) = img1.size()
    window = create_window_3D(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim_3D(img1, img2, window, window_size, channel, size_average)


class Grad(torch.nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        super(Grad, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred, y_true):
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx

        d = torch.mean(dx) + torch.mean(dy)
        grad = d / 2.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad

# 기존

class Grad3d_ssu(torch.nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        super(Grad3d_ssu, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])
        if self.penalty == 'l2':
            dy = dy * dy 
            dx = dx * dx 
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult

        y_pred = y_pred.permute(0, 2, 3, 4, 1)

        b1 = torch.broadcast_to(torch.Tensor([1.0, 0.0, 0.0]).cuda(), (y_pred.shape[1]-1, y_pred.shape[2]-1, y_pred.shape[3]-1, 3))#.permute(3, 0, 1, 2)
        b2 = torch.broadcast_to(torch.Tensor([0.0, 1.0, 0.0]).cuda(), (y_pred.shape[1]-1, y_pred.shape[2]-1, y_pred.shape[3]-1, 3))#.permute(3, 0, 1, 2)
        b3 = torch.broadcast_to(torch.Tensor([0.0, 0.0, 1.0]).cuda(), (y_pred.shape[1]-1, y_pred.shape[2]-1, y_pred.shape[3]-1, 3))#.permute(3, 0, 1, 2)

        dets = (torch.linalg.det((torch.stack([ # 3, D, H, W
            y_pred[:, 1:, :-1, :-1] - y_pred[:, :-1, :-1, :-1] + b1,
            y_pred[:, :-1, 1:, :-1] - y_pred[:, :-1, :-1, :-1] + b2,
            y_pred[:, :-1, :-1, 1:] - y_pred[:, :-1, :-1, :-1] + b3], axis=-1))))

        dets = torch.clamp(dets, max=0.0)

        return grad + torch.mean(dets*dets)

class Grad3d_0311(torch.nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        super(Grad3d_0311, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_true, y_pred):
        y_true = torch.argmax(y_true, 1, keepdim=True)
        y_true[y_true == 14] = 0

        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :]) 
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :]) 
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1]) 
        
        Gy = torch.abs(y_true[:, :, 1:, :, :] - y_true[:, :, :-1, :, :])
        Gx = torch.abs(y_true[:, :, :, 1:, :] - y_true[:, :, :, :-1, :])  
        Gz = torch.abs(y_true[:, :, :, :, 1:] - y_true[:, :, :, :, :-1]) 
    
        w = 0.0
        Gy = torch.where(Gy > 0, torch.tensor(w, dtype=y_pred.dtype).cuda(), torch.tensor(1.0, dtype=y_pred.dtype).cuda())
        Gx = torch.where(Gx > 0, torch.tensor(w, dtype=y_pred.dtype).cuda(), torch.tensor(1.0, dtype=y_pred.dtype).cuda())
        Gz = torch.where(Gz > 0, torch.tensor(w, dtype=y_pred.dtype).cuda(), torch.tensor(1.0, dtype=y_pred.dtype).cuda())


        if self.penalty == 'l2':
            dy = dy * dy * Gy 
            dx = dx * dx * Gx 
            dz = dz * dz * Gz 

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad 

class Grad3d(torch.nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        super(Grad3d, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad


        
class Dice(torch.nn.Module):
    """
    N-D dice for segmentation
    """
    def __init__(self, bg):
        super(Dice, self).__init__()
        self.bg = bg

    def forward(self, y_pred, y_true):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims+2)) 

        y_true = y_true[:, self.bg:, :, :, :]
        y_pred = y_pred[:, self.bg:, :, :, :]

        top = 2 * ((y_true * y_pred).sum())
        bottom = torch.clamp(((y_true + y_pred)).sum(), min=1e-5)
        dice = ((1-(top / bottom)))
        return dice

class Dice_avg(torch.nn.Module):
    """
    N-D dice for segmentation
    """
    def __init__(self, bg):
        super(Dice_avg, self).__init__()
        self.bg = bg

    def forward(self, y_pred, y_true):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims+2)) 

        y_true = y_true[:, self.bg:, :, :, :]
        y_pred = y_pred[:, self.bg:, :, :, :]

        top = 2 * ((y_true * y_pred).sum(dim=vol_axes))
        bottom = torch.clamp(((y_true + y_pred)).sum(dim=vol_axes), min=1e-5)
        dice = ((1-(top / bottom)))
        return torch.mean(dice), dice

class MSE(torch.nn.Module):
    """
    N-D dice for segmentation
    """
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, y_true, y_pred):
        with torch.no_grad():
            mask_t = torch.where(y_true<0.0, torch.tensor(0, dtype=y_true.dtype).cuda(), torch.tensor(1, dtype=y_true.dtype).cuda())
            mask_p = torch.where(y_pred<0.0, torch.tensor(0, dtype=y_pred.dtype).cuda(), torch.tensor(1, dtype=y_pred.dtype).cuda())
            mask = torch.logical_and(mask_t, mask_p)

        diff = y_true - y_pred

        return torch.mean((torch.square(diff)*mask).sum(dim=[1,2,3,4]) / ((mask).sum(dim=[1,2,3,4]) + 1e-5))


class Determ(torch.nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self):
        super(Determ, self).__init__()

    def forward(self, y_pred): # y_pred: DVF. 1, 3, 64, 128, 128
        size = y_pred.shape[2:]
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        grid = grid.cuda()

        y_pred = grid + y_pred

        y_pred = y_pred.permute(0, 2, 3, 4, 1) # 1, 64, 128, 128, 3

        J = np.gradient(y_pred.detach().cpu().numpy(), axis=(1, 2, 3))
        dx = J[0]
        dy = J[1]
        dz = J[2]

        dx = torch.Tensor(dx).cuda()
        dy = torch.Tensor(dy).cuda()
        dz = torch.Tensor(dz).cuda()
        
        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        dets_original =  Jdet0 - Jdet1 + Jdet2

        dets_neg = torch.clamp(dets_original, max=0.0)

        return torch.mean(dets_neg*dets_neg)

class NCC(torch.nn.Module):
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        super(NCC, self).__init__()
        self.win = win

    def forward(self, out, y_pred, out_ccl, y_ccl):

        I = out
        J = y_pred
        
        mask_I = torch.where((out<0.247) & (out_ccl>0), 0.0, 1.0)
        mask_J = torch.where((y_pred<0.247) & (y_ccl>0), 0.0, 1.0)

        ndims = len(list(I.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [64, 128, 128]#[9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = 0# math.floor(win[0]/2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1,1)
            padding = (pad_no, pad_no)
        else:
            stride = (1,1,1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = I * I
        J2 = J * J
        IJ = I * J

        I_sum = conv_fn(I, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(J, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)
        
        return torch.mean(-cc*mask_J*mask_I) 


class NCC_ori(torch.nn.Module):
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        super(NCC_ori, self).__init__()
        self.win = win

    def forward(self, out, y_pred):

        I = out
        J = y_pred

        ndims = len(list(I.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [64, 128, 128]#[9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = 0# math.floor(win[0]/2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1,1)
            padding = (pad_no, pad_no)
        else:
            stride = (1,1,1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = I * I
        J2 = J * J
        IJ = I * J

        I_sum = conv_fn(I, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(J, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return torch.mean(-cc) ## 추가

class Cos_sim(torch.nn.Module):

    def __init__(self, win=None):
        super(Cos_sim, self).__init__()
        self.win = win
        self.cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    def forward(self, f_feature, t_feature):
        result = self.cos(f_feature, t_feature)
        
        return 1-torch.mean(result)
        
class bce_loss(torch.nn.Module):

    def __init__(self):
        super(bce_loss, self).__init__()

    def forward(self, input, target):
        neg_abs = - input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        
        return loss.mean()

    def _contrastive(self, feats_, labels_):
        anchor_num, n_view = feats_.shape[0], feats_.shape[1]

        labels_ = labels_.contiguous().view(-1, 1)
        mask = torch.eq(labels_, torch.transpose(labels_, 0, 1)).float().cuda()

        contrast_count = n_view
        contrast_feature = torch.cat(torch.unbind(feats_, dim=1), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)),
                                        self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask

        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
                                                     0)
        mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def forward(self, feats, labels=None, predict=None):
        labels = labels.unsqueeze(1).float().clone()
        # print(labels.shape)
        labels = torch.nn.functional.interpolate(labels,
                                                 (feats.shape[2], feats.shape[3], feats.shape[4]), mode='nearest')
        labels = labels.squeeze(1).long()
        # print(labels.shape)
        assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)

        batch_size = feats.shape[0]
        labels = labels.contiguous().view(batch_size, -1) # b, class * 8 * 16 * 16 즉 One-hot Encoding 됨
                                                          # b, 8 * 16 *16 
        predict = predict.contiguous().view(batch_size, -1) # b, class * 8 * 16 * 16

        # feats: b, c, d, h, w
        
        feats = feats.permute(0, 2, 3, 4, 1)
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1]) # b, d*h*w, chennels
        # print(feats.shape)

        feats_, labels_ = self._hard_anchor_sampling(feats, labels, predict)

        loss = self._contrastive(feats_, labels_)
        return loss



class Dice_mean(torch.nn.Module):
    """
    N-D dice for segmentation
    """
    def __init__(self, bg):
        super(Dice_mean, self).__init__()
        self.bg = bg

    def forward(self, y_true, y_pred):

        y_true = y_true[:, self.bg:-1, :, :, :]
        y_pred = y_pred[:, self.bg:-1, :, :, :]
        
        top = 2 * ((y_true * y_pred).sum(dim=[2, 3, 4]))
        bottom = torch.clamp(((y_true*y_true + y_pred*y_pred)).sum(dim=[2, 3, 4]), min=1e-5)
        dice = 1 - (top / bottom)

        return torch.mean(dice) 