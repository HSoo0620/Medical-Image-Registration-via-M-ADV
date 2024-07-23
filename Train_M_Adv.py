from tkinter import X
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch import optim
import torch
import torch.nn as nn

import numpy as np
import os, utils, glob, losses, gc, sys, argparse
import matplotlib.pyplot as plt
import nibabel as nib
from data import datasets, trans
from torchvision import transforms
from natsort import natsorted
from models.M_Adv import CONFIGS as CONFIGS_TM
import models.M_Adv as M_Adv
from torch.autograd import Variable
from models.TransMorph_Origin_Affine import CONFIGS as AFF_CONFIGS_TM
import models.TransMorph_Origin_Affine as TransMorph_affine
from os.path import join
from os import listdir

import torch.nn.functional as F
from einops.einops import rearrange
from models.M_Adv_Func import make_Data_case, save_checkpoint, adjust_learning_rate, resize_dvf, restore_dvf, save_img, upsample_img, requires_grad, train_Discriminator, resize_img

class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"logfile.log", "a")    

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', '1'):
        return True
    elif v.lower() in ('no', 'false', '0'):
        return False

def main():
    '''
    Load Affine Model
    '''

    affine_model_dir = args.affine_model
    config_affine = AFF_CONFIGS_TM['TransMorph-Affine']
    affine_model = TransMorph_affine.SwinAffine(config_affine)
    affine_model.load_state_dict(torch.load('experiments/'+ affine_model_dir + natsorted(os.listdir('experiments/'+affine_model_dir), reverse=True)[0])['state_dict'])
    print('Affine Model: {} loaded!'.format(natsorted(os.listdir('experiments/'+ affine_model_dir), reverse=True)[0]))
    affine_model.cuda()
    affine_model.eval()
    
    for param in affine_model.parameters():
        param.requires_grad_(False)

    AffInfer_near = TransMorph_affine.ApplyAffine(mode='nearest')
    AffInfer_near.cuda()
    AffInfer_bi = TransMorph_affine.ApplyAffine(mode='bilinear')
    AffInfer_bi.cuda()

    # Loss function weights
    dir_weights = [0.5, 0.5, 0.5, 1.0]
    save_dir = args.dir_model
    save_result_dir = 'results/' + save_dir
    if not os.path.exists(save_result_dir):
           os.makedirs(save_result_dir)

    if not os.path.exists('experiments/'+save_dir):
        os.makedirs('experiments/'+save_dir)

    if not os.path.exists('logs/'+save_dir):
        os.makedirs('logs/'+save_dir)
    
    sys.stdout = Logger('logs/'+save_dir)
    
    '''hyper parameters'''
    lr = args.learning_rate # learning rate
    lr_D = args.learning_rate
    batch_size = args.batch_size
    epoch_start = 0
    max_epoch = args.max_epoch # max traning epoch
    init_step = 0 
    add_img_to_tensorboard = args.add_img_tensorboard
    validation_iter = args.validation_iter
    patch_size = args.patch_size # d h w patch size
    img_size = args.img_size # d h w img size
    save_validation_img = args.save_validation_img

    patch_z_size, patch_y_size, patch_x_size = patch_size
    img_z_size, img_y_size, img_x_size = img_size 

    # Use Pre-trained Model  
    cont_training_Basic = args.pre_train

    '''
    Initialize model
    '''
    config = CONFIGS_TM['M-Adv']
    model = M_Adv.M_Adv_model(config)
    model.cuda()

    model_Sagital_D = M_Adv.Sagital_Discriminator()
    model_Sagital_D.cuda()

    '''
    If continue from previous training
    '''
    if cont_training_Basic:
        model_dir = 'experiments/'+save_dir
        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch,0.9) ,8)

        best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-1])['state_dict']
        print('Model_Trans_Dual-Stream: {} loaded!'.format(natsorted(os.listdir(model_dir))[-1]))
        model.load_state_dict(best_model)

    else:
        updated_lr = lr
    
    '''
    Initialize spatial transformation function
    '''
    reg_model_bilin = utils.register_model(config.patch_size, 'bilinear')
    reg_model_bilin.cuda()

    reg_model = utils.register_model(config.patch_size, 'nearest')
    reg_model.cuda()

    reg_model_origin = utils.register_model(config.img_size, 'nearest')
    reg_model_origin.cuda()
    reg_model_bilin_origin = utils.register_model(config.img_size, 'bilinear')
    reg_model_bilin_origin.cuda()


    '''
    Initialize training
    '''
    train_composed = transforms.Compose([trans.NumpyType((np.float32, np.float32))])
    val_composed = transforms.Compose([trans.NumpyType((np.float32, np.float32))])

    '''
    Load Data for Train & Valid
    '''
    img_name = args.dataset_dir + 'train/image/'
    label_name = args.dataset_dir + 'train/label/'

    train_pairs = make_Data_case(img_name, label_name)

    print("===Make Train Case ", len(train_pairs), " Combinations")


    img_name = args.dataset_dir + 'test/image/'
    label_name = args.dataset_dir + 'test/label/'
    valid_pairs = make_Data_case(img_name, label_name)
    print("===Make Test Case : ", len(valid_pairs), " Combinations")

    train_set = datasets.BTCV_Crop_train_Dataset(train_pairs, transforms=train_composed, patch_size=args.patch_size)
    val_set = datasets.BTCV_Crop_valid_Dataset(valid_pairs, transforms=val_composed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)

    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    optimizer_Coronal_D = optim.Adam(model_Sagital_D.parameters(), lr=lr_D, weight_decay=0, amsgrad=True)

    criterion_dsc = losses.Dice(bg=1)
    criterion_ncc = losses.NCC_ori()
    criterion_det = losses.Determ()
    criterion_reg = losses.Grad3d(penalty='l2')
    criterion_avg_dsc = losses.Dice_avg(bg=1)
    writer = SummaryWriter(log_dir='logs/'+save_dir)
    writer_d_fake = SummaryWriter(log_dir='logs/'+save_dir+'Dis_False')
    writer_d_True = SummaryWriter(log_dir='logs/'+save_dir+'Dis_True')
    best_dsc = 0

    writer_moved = SummaryWriter(log_dir='logs/'+save_dir+'moved_and_fixed')
    writer_dff_label = SummaryWriter(log_dir='logs/'+save_dir+'m_f_dff_label')

    for epoch in range(epoch_start, max_epoch):
        torch.autograd.set_detect_anomaly(True)
        print('Training Start')
        print('Epoch {} :'.format(epoch))
        '''
        Training
        '''
        loss_all = utils.AverageMeter()
        train_DSC = utils.AverageMeter()
        eval_dsc = utils.AverageMeter()
        eval_avg_dsc = utils.AverageMeter()
        idx = 0
        for data in train_loader:
            idx += 1
            g_Adv_weight = 0.5
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            data = [t.cuda() for t in data]
            x_af = data[0]
            y_af = data[1]
            slice_num = torch.randint(25,36,(1,))
            
            x_af_in = torch.cat((x_af, y_af), dim=1)
            with torch.no_grad():
                out, affine_mat, _, _, _, _ = affine_model(x_af_in)

            x = AffInfer_bi(data[4].float(), affine_mat)
            y = data[5].float()            
            x_seg = AffInfer_near(data[6].float(), affine_mat)
            y_seg = data[7].float()

            x, y, x_seg, y_seg = x.cuda(), y.cuda(), x_seg.cuda(), y_seg.cuda()
            x, y, x_seg, y_seg = upsample_img(x, y, x_seg, y_seg, patch_z_size, patch_y_size, patch_x_size)
            x, y, x_seg, y_seg = x.cuda(), y.cuda(), x_seg.cuda(), y_seg.cuda()

            x_seg = nn.functional.one_hot(x_seg.long(), num_classes=12)# 0~ 14
            x_seg = torch.squeeze(x_seg, 1)
            x_seg = x_seg.permute(0, 4, 1, 2, 3).contiguous()
            
            y_seg = nn.functional.one_hot(y_seg.long(), num_classes=12)# 0~ 14
            y_seg = torch.squeeze(y_seg, 1)
            y_seg = y_seg.permute(0, 4, 1, 2, 3).contiguous()

            ##### 1. Train Discriminator #####
            optimizer_Coronal_D.zero_grad()
            requires_grad(model, False)
            requires_grad(model_Sagital_D, True)

            x_D_loss_a, y_D_loss_a = train_Discriminator(model_Sagital_D, model, optimizer_Coronal_D, y, x, mode='sagital', slice_num = slice_num)
            x_D_loss = x_D_loss_a
            y_D_loss = y_D_loss_a

            ##### 2. Train DIR Model #####
            optimizer.zero_grad()
            requires_grad(model, True)
            requires_grad(model_Sagital_D, False)

            x_in = torch.cat((x, y), dim=1)
            out, flow, _, _, _, _  = model(x_in)
            out_seg = reg_model_bilin([x_seg.float(), flow.float()])
            
            loss_ncc = criterion_ncc(out, y); loss_ncc_w = loss_ncc * dir_weights[0]
            loss_dsc = criterion_dsc(out_seg, y_seg); loss_dsc_w = loss_dsc * dir_weights[1]
            loss_dsc_balance, _ = criterion_avg_dsc(out_seg, y_seg); loss_dsc_balance_w = loss_dsc_balance * dir_weights[1]
            loss_reg = criterion_reg(flow); loss_reg_w = loss_reg * dir_weights[2]
            loss_det = criterion_det(flow); loss_det_w = loss_det * dir_weights[3]

            x_slice_sagital = out[..., slice_num, : ]
            x_slice_sagital = torch.squeeze(x_slice_sagital, 3)
            x_G_loss_3D = model_Sagital_D(x_slice_sagital)
            x_G_loss = x_G_loss_3D * -1

            loss = loss_ncc_w + loss_reg_w + loss_det_w + loss_dsc_balance_w + loss_dsc_w + x_G_loss*g_Adv_weight
            loss_all.update(loss.item(), x.numel())
            train_DSC.update(1-loss_dsc.item(), y.numel())
            writer.add_scalar('Loss/train', loss_all.val, idx+epoch)
            writer.add_scalar('DSC/train', train_DSC.val, idx+epoch)
            writer_d_fake.add_scalar('train', x_D_loss.item(), idx+epoch)
            writer_d_True.add_scalar('train', y_D_loss.item(), idx+epoch)

            optimizer.zero_grad()
            loss.backward() 
            optimizer.step() 
        
            if add_img_to_tensorboard:
                if idx % 50 == 0:            
                    x = rearrange(x.clone().detach(), 'a b d h w ->a b w h d ')
                    y = rearrange(y.clone().detach(), 'a b d h w ->a b w h d ')
                    out = rearrange(out.clone().detach(), 'a b d h w ->a b w h d ')              
                    slide = torch.cat((out[0][0][:,:,slice_num], y[0][0][:,:,slice_num], torch.abs(y[0][0][:,:,slice_num]-out[0][0][:,:,slice_num])), dim=0)
                    slide = torch.squeeze(slide)                
                    writer_moved.add_image('train',slide, idx+epoch*4032+init_step, dataformats='WC')

                    y_seg = torch.argmax(y_seg, dim=1).float()
                    out_seg = reg_model([x_seg.float(), flow.float()])
                    out_seg = torch.argmax(out_seg, dim=1).float()
                    y_seg = rearrange(y_seg.clone().detach(), 'a d h w ->a w h d ')
                    out_seg = rearrange(out_seg.clone().detach(), 'a d h w ->a w h d ')
                    slide = torch.cat((out_seg[0][:,:,slice_num], y_seg[0][:,:,slice_num], torch.abs(y_seg[0][:,:,slice_num]-out_seg[0][:,:,slice_num])), dim=0)
                    slide = torch.squeeze(slide)
                    writer_dff_label.add_image('train',slide, idx+epoch*4032+init_step, dataformats='WC')

            del x, y, x_seg, y_seg, out_seg, out, flow, _
            print('Iter {} of {} loss {:.4f}, NCC: {:.4f}, DSC: {:.4f} CB_DSC: {:.4f}, REG: {:.4f}, DET: {:.4f}, lr: {:.4f}, D_T: {:.4f} D_F: {:.4f} G_T: {:.4f} '.format(idx, len(train_loader),
                                                                            loss.item(),
                                                                            loss_ncc.item(),
                                                                            1-loss_dsc.item(),
                                                                            1-loss_dsc_balance.item(),
                                                                            loss_reg.item(),
                                                                            loss_det.item(),
                                                                            updated_lr,
                                                                            y_D_loss.item(),
                                                                            x_D_loss.item(),
                                                                            x_G_loss.item()*g_Adv_weight))

            '''
            Validation
            '''
            if idx % validation_iter == 0:
                eval_dsc = utils.AverageMeter()
                eval_avg_dsc = utils.AverageMeter()
                with torch.no_grad():
                    eval_idx = 0
                    for data in val_loader:
                        eval_idx +=1
                        model.eval()
                        data = [t.cuda() for t in data]
                        x_af = data[0]
                        y_af = data[1]
                        
                        x_af_in = torch.cat((x_af, y_af), dim=1)
                        with torch.no_grad():
                            _, affine_mat, _, _, _, _ = affine_model(x_af_in)

                        x = AffInfer_bi(data[4].float(), affine_mat)
                        y = data[5].float()            
                        x_seg = AffInfer_near(data[6].float(), affine_mat)
                        y_seg = data[7].float()

                        x_down, y_down, _, _ = upsample_img(x, y, x_seg, y_seg, patch_z_size, patch_y_size, patch_x_size)

                        x_seg = nn.functional.one_hot(x_seg.long(), num_classes=12)# 0~ 14
                        x_seg = torch.squeeze(x_seg, 1)
                        x_seg = x_seg.permute(0, 4, 1, 2, 3).contiguous()
                        y_seg = nn.functional.one_hot(y_seg.long(), num_classes=12)# 0~ 14
                        y_seg = torch.squeeze(y_seg, 1)
                        y_seg = y_seg.permute(0, 4, 1, 2, 3).contiguous()

                        x_in = torch.cat((x_down.cuda(), y_down.cuda()), dim=1)
                        out, flow, _, _, _, _  = model(x_in)

                        upsampling_flow_z = resize_dvf(flow[0][0], img_z_size, img_y_size, img_x_size)
                        upsampling_flow_y = resize_dvf(flow[0][1], img_z_size, img_y_size, img_x_size)
                        upsampling_flow_x = resize_dvf(flow[0][2], img_z_size, img_y_size, img_x_size)
                        
                        upsampling_flow_z *= (img_z_size / patch_z_size)
                        upsampling_flow_y *= (img_y_size / patch_y_size)
                        upsampling_flow_x *= (img_x_size / patch_x_size)
                                
                        upsampling_flow = torch.cat([upsampling_flow_z[None], upsampling_flow_y[None], upsampling_flow_x[None]], dim=0) # (c, w, h, d)
                        upsampling_flow = upsampling_flow[None] # (1, c, w, h, d)

                        out = reg_model_bilin_origin([x.float(), upsampling_flow.float()])
                        out_seg = reg_model_origin([x_seg.float(), upsampling_flow.float()])
                        dsc = 1-criterion_dsc(out_seg, y_seg)

                        eval_dsc.update(dsc.item(), x.size(0))
                        avg_dsc, avg_list = criterion_avg_dsc(out_seg, y_seg)
                        avg_list = avg_list[0]
                        avg_dsc = 1-avg_dsc
                        eval_avg_dsc.update(avg_dsc.item(), x.size(0))
                        
                        print('Idx {} of Val {} DSC:{: .4f} '.format(eval_idx, len(val_loader),dsc.item()))
                        if eval_idx == 240:
                            print("--Validation Dice dsc: {:.5f} +- {:.3f}".format(eval_dsc.avg, eval_dsc.std))
                            print("--Validation Dice avg_dsc: {:.5f} +- {:.3f}".format(eval_avg_dsc.avg, eval_dsc.std))

                        if save_validation_img:
                            if_name = save_result_dir + str(eval_idx)+ '_pred_mask.nii.gz'
                            gt_name = save_result_dir + str(eval_idx)+ '_gt_mask.nii.gz'

                            save_img(out_seg.float(), if_name, 'label') 
                            save_img(y_seg.float(), gt_name, 'label') 

                            if_name = save_result_dir + str(eval_idx)+ '_pred_img.nii.gz'
                            gt_name = save_result_dir + str(eval_idx)+ '_gt_img.nii.gz'

                            save_img(out.float(), if_name, 'img') 
                            save_img(y.float(), gt_name, 'img') 

                        del x, y, x_seg, y_seg, out, out_seg, x_af, y_af, upsampling_flow

                torch.cuda.empty_cache()
                gc.collect()
                best_dsc = max(eval_avg_dsc.avg, best_dsc)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_dsc': best_dsc,
                    'optimizer': optimizer.state_dict(),
                }, save_dir='experiments/'+save_dir, filename='avg_dsc{:.4f} dsc{:.4f} idx{:.1f} epoch{:.1f}.pth.tar'.format(eval_avg_dsc.avg,eval_dsc.avg,idx,epoch))

                writer.add_scalar('DSC/validate', eval_avg_dsc.avg, epoch)

        loss_all.reset()
        eval_dsc.reset()
    writer.close()

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--affine_model', type=str, default='experiments/affine/',
                        help='Affine model load directory')
    parser.add_argument('--dir_model', type=str, default='experiments/test/',
                        help='DIR model load directory')
    parser.add_argument('--dataset_dir', type=str, default='Dataset/BTCV_Abdominal_1k/',
                        help='Dataset directory')
    parser.add_argument('--pre_train', type=str2bool, default='False',
                        help='pre-train load True or False')
    parser.add_argument('--add_img_tensorboard', type=str2bool, default='False',
                        help='add img tensorboard True or False')
    parser.add_argument('--save_validation_img', type=str2bool, default='False',
                        help='save_validation_img True or False')
    parser.add_argument('--img_size', type=int, default=(128, 512, 512),
                        help='size of image')
    parser.add_argument('--patch_size', type=int, default=(64, 128, 128),
                        help='size of patch')
    parser.add_argument('--max_epoch', type=int, default=100,
                        help='max epoch')
    parser.add_argument('--validation_iter', type=int, default=1000,
                        help='validation iter')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch size')                        
    parser.add_argument('--learning_rate', type=int, default=1e-4,
                        help='learning rate')       

    args = parser.parse_args()
    
    main(args)