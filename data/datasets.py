import os, glob
import torch, sys
from torch.utils.data import Dataset
from .data_utils import pkload
import matplotlib.pyplot as plt
import random
import numpy as np
import natsort

import nibabel as nib
import os 
import numpy as np
import torchio as tio
import torch.nn.functional as F
import torchvision.transforms as T

def save_img(img, img_name, mode='label') :
    if mode == 'label':
        img = torch.argmax(img, dim=1).float()   
    img = img.squeeze() # d h w 
    img = img.cpu().detach().numpy()
    # print(img.shape)
    img = np.swapaxes(img,0,2) #  w h d
    # img = np.swapaxes(img,1,2) #  w h d
    # print(img.shape)
    x = nib.Nifti1Image(img, None)
    nib.save(x, img_name)

def aff_augmentation(img, label):
    subject = tio.Subject(
            original=tio.ScalarImage(tensor=img),
            mask=tio.LabelMap(tensor=label))
    affine = tio.RandomAffine(scales=(0.8, 1.0), degrees=(10,10,10), translation=(0,4,4))
    subj = affine(subject)
    return subj.original.data, subj.mask.data

def ssl_augmentation(img, label):
    subject = tio.Subject(
            original=tio.ScalarImage(tensor=img),
            mask=tio.LabelMap(tensor=label))
    # affine = tio.RandomAffine(scales=(0.8, 1.0), degrees=(10,10,10), translation=(0,4,4), label_interpolation = 'linear') #LINEAR
    affine = tio.RandomAffine(scales=(0.8, 1.0), degrees=(10,10,10), translation=(0,4,4), label_interpolation = 'nearest') #LINEAR

    ctr_n = 5
    max_displacement = (8, 16, 16)
    deform_transform = tio.RandomElasticDeformation(
        num_control_points=ctr_n,
        locked_borders=0,
        max_displacement = max_displacement,
        label_interpolation = 'nearest'
    )

    composed_transform = tio.Compose([affine, deform_transform])
    transformed_subject = composed_transform(subject)
    transformed_img = transformed_subject['original'].data
    transformed_label = transformed_subject['mask'].data
    return transformed_img, transformed_label

def dir_augmentation(img, label, img2, label2):
    subject = tio.Subject(
            original=tio.ScalarImage(tensor=img),
            mask=tio.LabelMap(tensor=label),
            original2=tio.ScalarImage(tensor=img2),
            mask2=tio.LabelMap(tensor=label2))
    affine = tio.RandomAffine(scales=(0.8, 1.0), degrees=(10,10,10), translation=(0,4,4), label_interpolation = 'nearest') #LINEAR

    ctr_n = 5
    max_displacement = (8, 16, 16)
    deform_transform = tio.RandomElasticDeformation(
        num_control_points=ctr_n,
        locked_borders=0,
        max_displacement = max_displacement,
        # label_interpolation = 'linear'
        label_interpolation = 'nearest'
    )
    composed_transform = tio.Compose([affine])
    transformed_subject = composed_transform(subject)
    transformed_img = transformed_subject['original'].data
    transformed_label = transformed_subject['mask'].data
    transformed_img2 = transformed_subject['original2'].data
    transformed_label2 = transformed_subject['mask2'].data

    return transformed_img, transformed_label, transformed_img2, transformed_label2

def affine_augmentation(img, img2, label, label2):
    subject = tio.Subject(
            original=tio.ScalarImage(tensor=img),
            mask=tio.LabelMap(tensor=label),
            original2=tio.ScalarImage(tensor=img2),
            mask2=tio.LabelMap(tensor=label2))
            
    affine = tio.RandomAffine(scales=(0.9, 1.0), degrees=(10,10,10), translation=(0,4,4), label_interpolation = 'nearest')

    transformed_subject = affine(subject)
    
    transformed_img = transformed_subject['original'].data
    transformed_label = transformed_subject['mask'].data
    transformed_img2 = transformed_subject['original2'].data
    transformed_label2 = transformed_subject['mask2'].data

    return transformed_img, transformed_img2, transformed_label, transformed_label2

def augmentation_hole(img, max_cut, hole_size):
    # hole_size = 4 
    _,d,h,w = img.shape
    cutout_point_x = torch.randint(0, h-hole_size, (max_cut,))
    cutout_point_y = torch.randint(0, h-hole_size, (max_cut,))
    cutout_point_z = torch.randint(0, h-hole_size, (max_cut,))
    for i in range(len(cutout_point_x)):
        img[:, cutout_point_z[i]//2:cutout_point_z[i]//2+(hole_size//2), cutout_point_x[i]:cutout_point_x[i]+hole_size, cutout_point_y[i]:cutout_point_y[i]+hole_size] = 0
    
    return img


class BTCVAffineDataset(Dataset):
    def __init__(self, data_path, transforms):

        self.paths = data_path
        self.transforms = transforms

    def __getitem__(self, index):
        path = self.paths[index]
        tar_list = self.paths.copy()
        tar_list.remove(path)

        ddata = nib.load(path).get_fdata()
        x, y, x_seg, y_seg = ddata[0],ddata[1], ddata[2], ddata[3]

        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])

        x, x_seg = aff_augmentation(x, x_seg)
        y, y_seg =aff_augmentation(y, y_seg)
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)

        x, y, x_seg, y_seg= torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)


class BTCVSSLDataset(Dataset):
    def __init__(self, data_path, transforms):
        # data_path = natsort.natsorted(data_path)
        # print(type(data_path))
        # random.shuffle(data_path)
        self.paths = data_path
        self.transforms = transforms

    def __getitem__(self, index):
        path = self.paths[index]
        tar_list = self.paths.copy()
        tar_list.remove(path)

        ddata = nib.load(path).get_fdata()
        x, _, x_seg, _ = ddata[0],ddata[1], ddata[2], ddata[3]

        x, x_seg = x[None, ...], x_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        
        x_, x_seg_ = ssl_augmentation(x, x_seg)
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        x_ = np.ascontiguousarray(x_)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        x_seg_ = np.ascontiguousarray(x_seg_) 

        x, x_, x_seg, x_seg_ = torch.from_numpy(x),torch.from_numpy(x_), torch.from_numpy(x_seg), torch.from_numpy(x_seg_)
        return x, x_, x_seg, x_seg_

    def __len__(self):
        return len(self.paths)

class BTCVDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def __getitem__(self, index):
        path = self.paths[index]
        tar_list = self.paths.copy()
        tar_list.remove(path)

        ddata = nib.load(path).get_fdata()
        x, y, x_seg, y_seg = ddata[0],ddata[1], ddata[2], ddata[3]

        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])

        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)

def resize_img(img, gt, re_d, re_h, re_w):
    d = torch.linspace(-1,1,re_d)
    h = torch.linspace(-1,1,re_h)
    w = torch.linspace(-1,1,re_w)
    meshz, meshy, meshx = torch.meshgrid((d,h,w))
    grid = torch.stack((meshz, meshy, meshx), 3).cpu() # (64, 128, 128, 3)
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
    return img, gt

class BTCV_Crop_train_Dataset(Dataset):
    def __init__(self, data_path, transforms, patch_size):
        self.paths = data_path
        self.transforms = transforms
        self.patch_size = patch_size

    def __getitem__(self, index):
        path = self.paths[index]
        d, h, w = self.patch_size
        x = nib.load(path[0]).get_fdata()
        y = nib.load(path[1]).get_fdata()
        x_seg = nib.load(path[2]).get_fdata()
        y_seg = nib.load(path[3]).get_fdata()

        x, y, x_seg, y_seg = affine_augmentation(x[None], y[None], x_seg[None], y_seg[None])
        x, y, x_seg, y_seg = x[0], y[0], x_seg[0], y_seg[0]

        x_af, x_af_seg = resize_img(x, x_seg, d, h, w)
        y_af, y_af_seg = resize_img(y, y_seg, d, h, w)

        x_af, y_af = x_af[None, ...], y_af[None, ...]
        x_af_seg, y_af_seg = x_af_seg[None, ...], y_af_seg[None, ...]

        x_af = np.ascontiguousarray(x_af)  # [Bsize,channelsHeight,,Width,Depth]
        y_af = np.ascontiguousarray(y_af)
        x_af_seg = np.ascontiguousarray(x_af_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_af_seg = np.ascontiguousarray(y_af_seg)
        x_af, y_af, x_af_seg, y_af_seg = torch.from_numpy(x_af), torch.from_numpy(y_af), torch.from_numpy(x_af_seg), torch.from_numpy(y_af_seg)

        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]

        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)

        return x_af, y_af, x_af_seg, y_af_seg, x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)

class BTCV_Crop_valid_Dataset(Dataset):
    def __init__(self, data_path, transforms, patch_size):
        self.paths = data_path
        self.transforms = transforms
        self.patch_size = patch_size

    def __getitem__(self, index):
        path = self.paths[index]
        d, h, w = self.patch_size
        
        x = nib.load(path[0]).get_fdata()
        y = nib.load(path[1]).get_fdata()
        x_seg = nib.load(path[2]).get_fdata()
        y_seg = nib.load(path[3]).get_fdata()

        x_af, x_af_seg = resize_img(x, x_seg, d, h, w)
        y_af, y_af_seg = resize_img(y, y_seg, d, h, w)

        x_af, y_af = x_af[None, ...], y_af[None, ...]
        x_af_seg, y_af_seg = x_af_seg[None, ...], y_af_seg[None, ...]


        x_af = np.ascontiguousarray(x_af)  # [Bsize,channelsHeight,,Width,Depth]
        y_af = np.ascontiguousarray(y_af)
        x_af_seg = np.ascontiguousarray(x_af_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_af_seg = np.ascontiguousarray(y_af_seg)
        x_af, y_af, x_af_seg, y_af_seg = torch.from_numpy(x_af), torch.from_numpy(y_af), torch.from_numpy(x_af_seg), torch.from_numpy(y_af_seg)

        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]

        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)

        return x_af, y_af, x_af_seg, y_af_seg, x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)


def volgen_datalist(vol_names, mask_names):
    # convert glob path to filenames
    if isinstance(vol_names, str):
        if os.path.isdir(vol_names):
            vol_names = os.path.join(vol_names, '*')
        vol_names = glob.glob(vol_names)
    
    if isinstance(mask_names, str):
        if os.path.isdir(mask_names):
            mask_names = os.path.join(mask_names, '*')
        mask_names = glob.glob(mask_names)

    return vol_names, mask_names 

class BTCVInferDataset_ori(Dataset):
    def __init__(self, data_path, transforms):
        data_path = natsort.natsorted(data_path)
        self.paths = data_path
        self.transforms = transforms

    def __getitem__(self, index):
        path = self.paths[index]


        ddata = nib.load(path).get_fdata()
        x, y, x_seg, y_seg= ddata[0],ddata[1], ddata[2], ddata[3]

        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg= x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  
        y_seg = np.ascontiguousarray(y_seg)

        x, y, x_seg, y_seg= torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)