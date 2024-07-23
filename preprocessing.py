from skimage.transform import resize
import torch
from torch import nn
import numpy as np
import torchvision
import torch.nn.functional as F

from skimage.transform import resize
import torch
from torch import nn
import numpy as np
import torchvision
import torch.nn.functional as F
from einops.einops import rearrange
import nibabel as nib

from skimage.transform import resize
import torch
from torch import nn
import numpy as np
import torchvision
import torch.nn.functional as F

window_level = 10
window_width = 700
window_high = (window_width / 2) + window_level
window_low =  window_level - (window_width / 2)
mode='bilinear'

def get_range(gt) :
    min_slice = 0
    max_slice = 0
    flatten_label = rearrange(gt.clone().detach(), 'w h d -> d (h w)')
    d, hw = flatten_label.shape
    min_voxel = torch.sum(flatten_label >= 0.9) / d / 100
    for cur_d in range(d) :
        mask = (flatten_label[cur_d].long() != 0) & (flatten_label[cur_d].long() != 7) & (flatten_label[cur_d].long() != 8) & (flatten_label[cur_d].long() != 4)
        x = torch.sum(mask)
        if min_slice < 1 and x >= min_voxel :
            min_slice = cur_d
        if x > min_voxel :
            max_slice = cur_d
    start = round(min_slice - (max_slice-min_slice) / 5)
    end = round(max_slice + (max_slice-min_slice) / 5)
    return start, end, min_voxel

def pre_process(img, label, window_low, window_high, crop = False) :
    img = img.astype(np.float32)
    gt = label.astype(np.float32)
    w, h, d = img.shape
    # pre-padding
    img_padder = torch.zeros(w,h,300) - 1024
    gt_padder = torch.zeros(w,h,300)
    img = torch.cat([torch.Tensor(img), img_padder], dim=2)
    gt = torch.cat([torch.Tensor(gt), gt_padder], dim=2)
    # crop
    if crop :
        img = torch.Tensor(img)
        gt = torch.Tensor(gt)
        start, end, min_voxel = get_range(gt)
        start = max(start, 0)
        img = rearrange(img, 'w h d -> d h w')
        gt = rearrange(gt, 'w h d -> d h w')
        print(img.shape)
        img = img[start:end+1]
        gt = gt[start:end+1]
        img = rearrange(img, 'd h w -> w h d')
        gt = rearrange(gt, 'd h w -> w h d')
    w, h, d = img.shape
    print(w, h, d)
    # Windowing
    img[img < window_low] = window_low
    img[img > window_high] = window_high
    # img = torchvision.transforms.functional.equalize(img[..., None])
    # resize
    print(img.shape)
    d = torch.linspace(-1,1,64)
    h = torch.linspace(-1,1,128)
    w = torch.linspace(-1,1,128)
    meshz, meshy, meshx = torch.meshgrid((d,h,w))
    grid = torch.stack((meshx, meshy, meshz), 3).cpu() # (64, 128, 128, 3)
    grid = grid.unsqueeze(0) # (1, 64, 128, 128, 3)
    img = torch.Tensor(img).cpu()
    img = img.unsqueeze(0)
    img = img.unsqueeze(0)
    img = img.permute(0,1,4,3,2)
    img = F.grid_sample(img, grid, mode='bilinear', align_corners=True)
    img = img[0][0]
    gt = torch.Tensor(gt).cpu()
    gt = gt.unsqueeze(0)
    gt = gt.unsqueeze(0)
    gt = gt.permute(0,1,4,3,2)
    gt = F.grid_sample(gt, grid, mode='nearest', align_corners=True)
    gt = gt[0][0]
    print("last shape : ", img.shape)
    img = torch.Tensor.numpy(img)
    gt = torch.Tensor.numpy(gt)
    img_max = np.max(img)
    img_min = np.min(img)
    # min-max normalization
    img = (img-img_min) / (img_max - img_min)
    return img, gt

for a in range(10,51):
    filename1 = './Dataset/AbdomenCT-1K/Image/Organ12_00'+str(a)+'_0000.nii.gz'
    filename2 = './Dataset/AbdomenCT-1K/0309_labeling/Organ12_00'+str(a)+'.nii.gz'
    img = nib.load(filename1).get_fdata()
    img = img[::-1,...]

    label = nib.load(filename2).get_fdata()
    label = label[::-1,...]

    img, label = pre_process(img, label, window_high=window_high, window_low=window_low, crop=True)

    img_name = './Dataset/AbdomenCT-1K/pre_processed/Image/'+str(a)+'.nii.gz'
    img = np.swapaxes(img,0,2)
    x = nib.Nifti1Image(img, None)
    nib.save(x, img_name)

    img_name = './Dataset/AbdomenCT-1K/pre_processed/Mask/'+str(a)+'.nii.gz'
    label = np.swapaxes(label,0,2)
    x = nib.Nifti1Image(label, None)
    nib.save(x, img_name)
    print("clear",a)

