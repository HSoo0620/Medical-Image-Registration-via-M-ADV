import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F
from natsort import natsorted
from os.path import join
from os import listdir
from data import datasets, trans
import torchio as tio
import os, glob

def make_Data_case(img_name, label_name):
    train_img_list = natsorted(
        [join(img_name, file_name) for file_name in listdir(img_name)]
    )
    train_label_list = natsorted(
        [join(label_name, file_name) for file_name in listdir(label_name)]
    )
    train_list = datasets.volgen_datalist(train_img_list, train_label_list)

    pairs = []
    for source in range(0, len(train_list[0])): #data_num
        for target in range(0, len(train_list[0])):
            if source == target:
                continue
            pairs.append((train_list[0][source], train_list[0][target], train_list[1][source], train_list[1][target]))
    return pairs

def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=3):
    torch.save(state, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + '*'), reverse=True)
    while len(model_lists) > max_model_num:
        os.remove(model_lists[-1])
        model_lists = natsorted(glob.glob(save_dir + '*'))

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def upsample_img(x, y, x_seg, y_seg, patch_z, patch_y, patch_x):
    x, x_seg = resize_img(x[0][0], x_seg[0][0], patch_z, patch_y, patch_x)
    y, y_seg = resize_img(y[0][0], y_seg[0][0], patch_z, patch_y, patch_x)
    return x, y, x_seg, y_seg
    
def save_img(img, img_name, mode='label') :

    if mode == 'label':
        img = torch.argmax(img, dim=1).float()   
    img = img.squeeze() # d h w 
    img = img.cpu().detach().numpy()
    img = np.swapaxes(img,0,2) #  w h d
    x = nib.Nifti1Image(img, None)
    nib.save(x, img_name)

def restore_dvf(flow, d, h, w):
    upsampling_flow_z = resize_dvf(flow[0][0], d, h, w)
    upsampling_flow_y = resize_dvf(flow[0][1], d, h, w)
    upsampling_flow_x = resize_dvf(flow[0][2], d, h, w)
    
    upsampling_flow_z *= (d / 64)
    upsampling_flow_y *= (h / 128)
    upsampling_flow_x *= (w / 128)
            
    upsampling_flow = torch.cat([upsampling_flow_z[None], upsampling_flow_y[None], upsampling_flow_x[None]], dim=0) # (c, w, h, d)
    upsampling_flow = upsampling_flow[None] # (1, c, w, h, d)
    return upsampling_flow

def resize_dvf(img, re_d, re_h, re_w):
    d = torch.linspace(-1,1,re_d)
    h = torch.linspace(-1,1,re_h)
    w = torch.linspace(-1,1,re_w)
    meshz, meshy, meshx = torch.meshgrid((d,h,w))
    grid = torch.stack((meshz, meshy, meshx), 3).cpu() # (64, 128, 128, 3)
    grid = grid.unsqueeze(0) # (1, 64, 128, 128, 3)
    img = img.cpu()
    img = img.unsqueeze(0)
    img = img.unsqueeze(0)
    img = img.permute(0,1,4,3,2)
    img = F.grid_sample(img, grid, mode='bilinear', align_corners=True)
    img = img[0][0]
    return img

def resize_img(img, gt, re_d, re_h, re_w):
    d = torch.linspace(-1,1,re_d)
    h = torch.linspace(-1,1,re_h)
    w = torch.linspace(-1,1,re_w)
    meshz, meshy, meshx = torch.meshgrid((d,h,w))
    grid = torch.stack((meshz, meshy, meshx), 3).cpu() # (64, 128, 128, 3)
    grid = grid.unsqueeze(0) # (1, 64, 128, 128, 3)
    img = img.cpu()
    img = img.unsqueeze(0)
    img = img.unsqueeze(0)
    img = img.permute(0,1,4,3,2)
    img = F.grid_sample(img, grid, mode='bilinear', align_corners=True)
    # img = img[0][0]
    gt = gt.cpu()
    gt = gt.unsqueeze(0)
    gt = gt.unsqueeze(0)
    gt = gt.permute(0,1,4,3,2)
    gt = F.grid_sample(gt, grid, mode='nearest', align_corners=True)
    # gt = gt[0][0]
    return img, gt

def train_Discriminator(model_d, model, optimizer_D, y, x, mode, slice_num):
    for p in model_d.parameters():
        p.data.clamp_(-0.01, 0.01)

    if mode == 'axial':
        sq_index = 1
        y_slice = y[:,:,slice_num,:]# b c 1

    elif mode == 'sagital':
        sq_index = 3
        y_slice = y[..., slice_num, : ]
    elif mode == 'coronal': 
        sq_index = 4
        y_slice = y[..., slice_num]
    elif mode == '3D':
        y_slice = y

    if mode == 'axial' or mode == 'sagital' or mode == 'coronal':
        y_slice = torch.squeeze(y_slice, sq_index)
    
    y_D_loss = model_d(y_slice)

    with torch.no_grad():   
        x_in = torch.cat((x,y), dim=1)
        fake_out, flow, _, _, _, _ = model(x_in)

    if mode == 'axial':
        x_slice = fake_out[:,:,slice_num,:]# b c 1
    elif mode == 'sagital':
        x_slice = fake_out[..., slice_num, : ]
    elif mode == 'coronal': 
        x_slice = fake_out[..., slice_num]
    elif mode == '3D':  
        x_slice = fake_out

    if mode == 'axial' or mode == 'sagital' or mode == 'coronal':
        x_slice = torch.squeeze(x_slice, sq_index)
        
    x_D_loss = model_d(x_slice)

    d_loss = x_D_loss - y_D_loss
    d_loss.backward()
    optimizer_D.step()

    return x_D_loss, y_D_loss