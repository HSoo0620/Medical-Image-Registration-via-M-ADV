from tkinter import X
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

import numpy as np
import os, utils, losses, argparse
import nibabel as nib
from data import datasets, trans
from torchvision import transforms
from natsort import natsorted
from models.M_Adv import CONFIGS as CONFIGS_TM
import models.M_Adv as M_Adv
from torch.autograd import Variable
from models.TransMorph_Origin_Affine import CONFIGS as AFF_CONFIGS_TM
import models.TransMorph_Origin_Affine as TransMorph_affine


from models.M_Adv_Func import make_Data_case, resize_dvf, resize_img
from monai.metrics import compute_hausdorff_distance, compute_average_surface_distance
import pandas as pd

def calc_hd95(moving, fixed):

    moving = torch.cat([moving[0][:3], moving[0][3+1:]])
    fixed = torch.cat([fixed[0][:3], fixed[0][3+1:]])

    hd95 = compute_hausdorff_distance(moving[None], fixed[None].long(), include_background=False, percentile=95)
    return torch.mean(hd95)

def calc_asd(moving, fixed):
    moving = torch.cat([moving[0][:3], moving[0][3+1:]])
    fixed = torch.cat([fixed[0][:3], fixed[0][3+1:]])
    asd = compute_average_surface_distance(moving[None], fixed[None].long(), include_background=False)
    return torch.mean(asd)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', '1'):
        return True
    elif v.lower() in ('no', 'false', '0'):
        return False

def main():
   
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

    model_dir = args.dir_model
    model_dir = 'experiments/'+model_dir
    
    save_dir = 'results/Train_M_Adv/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    config = CONFIGS_TM['M-Adv']
    model = M_Adv.M_Adv_model(config)

    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-1])['state_dict']
    model.load_state_dict(best_model)
    print('Model: {} loaded!'.format(natsorted(os.listdir(model_dir))[-1]))
    model.cuda()

    test_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16))])


    img_name = args.dataset_dir + 'test/image/'
    label_name = args.dataset_dir + 'test/label/'
    test_pairs = make_Data_case(img_name, label_name)
    print("===Make Test Case : ", len(test_pairs), " Combinations")
    
    patch_size = args.patch_size # d h w patch size
    img_size = args.img_size # d h w img size
    patch_z_size, patch_y_size, patch_x_size = patch_size
    img_z_size, img_y_size, img_x_size = img_size 
    
    test_set = datasets.BTCV_Crop_valid_Dataset((test_pairs), transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=False)

    criterion_dsc = losses.Dice(bg=1)    
    criterion_avg_dsc = losses.Dice_avg(bg=1)
    
    idx = 0

    eval_dsc = utils.AverageMeter()
    eval_avg_dsc = utils.AverageMeter()

    # evaluate each class
    eval_0 = utils.AverageMeter()
    eval_1 = utils.AverageMeter()
    eval_2 = utils.AverageMeter()
    eval_3 = utils.AverageMeter()
    eval_4 = utils.AverageMeter()
    eval_5 = utils.AverageMeter()
    eval_6 = utils.AverageMeter()
    eval_7 = utils.AverageMeter()
    eval_8 = utils.AverageMeter()
    eval_9 = utils.AverageMeter()
    eval_10 = utils.AverageMeter()
    hd_avg =  utils.AverageMeter()
    asd_avg =  utils.AverageMeter()

    with torch.no_grad():
        for data in test_loader:
            model.eval()
            idx += 1

            data = [t.cuda() for t in data]
            x_af = data[0]
            y_af = data[1]

            # start = time.time()

            x_af_in = torch.cat((x_af, y_af), dim=1)
            with torch.no_grad():
                out, affine_mat, inv_mats, Rigid_out, Rigid_mat, Rigid_inv_mat = affine_model(x_af_in)

            x = AffInfer_bi(data[4].float(), affine_mat)
            y = data[5].float()            
            x_seg = AffInfer_near(data[6].float(), affine_mat)
            y_seg = data[7].float()

            x_seg = nn.functional.one_hot(x_seg.long(), num_classes=12)# 0~ 14
            x_seg = torch.squeeze(x_seg, 1)
            x_seg = x_seg.permute(0, 4, 1, 2, 3).contiguous()
            y_seg = nn.functional.one_hot(y_seg.long(), num_classes=12)# 0~ 14
            y_seg = torch.squeeze(y_seg, 1)
            y_seg = y_seg.permute(0, 4, 1, 2, 3).contiguous()

            x_down = resize_img(x[0][0], patch_z_size, patch_y_size, patch_x_size)
            y_down = resize_img(y[0][0], patch_z_size, patch_y_size, patch_x_size)

            x_in = torch.cat((x_down.cuda(), y_down.cuda()), dim=1)
            out, flow, _, _, _, _ = model(x_in)

            upsampling_flow_z = resize_dvf(flow[0][0], img_z_size, img_y_size, img_x_size)
            upsampling_flow_y = resize_dvf(flow[0][1], img_z_size, img_y_size, img_x_size)
            upsampling_flow_x = resize_dvf(flow[0][2], img_z_size, img_y_size, img_x_size)
            
            upsampling_flow_z *= (img_z_size / patch_z_size)
            upsampling_flow_y *= (img_y_size / patch_y_size)
            upsampling_flow_x *= (img_x_size / patch_x_size)
                    
            upsampling_flow = torch.cat([upsampling_flow_z[None], upsampling_flow_y[None], upsampling_flow_x[None]], dim=0) # (c, w, h, d)
            upsampling_flow = upsampling_flow[None] # (1, c, w, h, d)

            reg_model = utils.register_model((img_z_size, img_y_size, img_x_size), 'nearest')
            reg_model.cuda()
            reg_model_bilin = utils.register_model((img_z_size, img_y_size, img_x_size), 'bilinear')
            reg_model_bilin.cuda()

            out, new_locs = reg_model_bilin([x.float(), upsampling_flow.float()])
            out_seg, _ = reg_model([x_seg.float(), upsampling_flow.float()])
            
            dsc = 1-criterion_dsc(out_seg, y_seg.long())

            avg_dsc, avg_list = criterion_avg_dsc(out_seg, y_seg.long())
            avg_list = avg_list[0]

            hd95 = calc_hd95(out_seg, y_seg)
            asd = calc_asd(out_seg, y_seg)

            hd_avg.update(hd95.item(), x.size(0))
            asd_avg.update(asd.item(), x.size(0))

            eval_dsc.update(dsc.item(), x.size(0))
            eval_avg_dsc.update(1-avg_dsc.item(), x.size(0))

            eval_0.update(1-avg_list[0].item(), x.size(0))
            eval_1.update(1-avg_list[1].item(), x.size(0))
            eval_2.update(1-avg_list[2].item(), x.size(0))
            eval_3.update(1-avg_list[3].item(), x.size(0))
            eval_4.update(1-avg_list[4].item(), x.size(0))
            eval_5.update(1-avg_list[5].item(), x.size(0))
            eval_6.update(1-avg_list[6].item(), x.size(0))
            eval_7.update(1-avg_list[7].item(), x.size(0))
            eval_8.update(1-avg_list[8].item(), x.size(0))
            eval_9.update(1-avg_list[9].item(), x.size(0))
            eval_10.update(1-avg_list[10].item(), x.size(0))

            del out_seg, x, y, x_seg, y_seg, flow
            print('Idx {} of Val {} DSC:{: .4f} '.format(idx, len(test_loader),dsc.item()))
            if idx == 240:
                print("Dice eval_dsc: {:.5f} +- {:.3f}".format(eval_dsc.avg, eval_dsc.std))
                print("--------------------------------------------")
                print("Dice eval_avg_dsc0: {:.5f} +- {:.3f}".format(eval_0.avg, eval_0.std))
                print("Dice eval_avg_dsc1: {:.5f} +- {:.3f}".format(eval_1.avg, eval_1.std))
                print("Dice eval_avg_dsc2: {:.5f} +- {:.3f}".format(eval_2.avg, eval_2.std))
                print("Dice eval_avg_dsc3: {:.5f} +- {:.3f}".format(eval_3.avg, eval_3.std))
                print("Dice eval_avg_dsc4: {:.5f} +- {:.3f}".format(eval_4.avg, eval_4.std))
                print("Dice eval_avg_dsc5: {:.5f} +- {:.3f}".format(eval_5.avg, eval_5.std))
                print("Dice eval_avg_dsc6: {:.5f} +- {:.3f}".format(eval_6.avg, eval_6.std))
                print("Dice eval_avg_dsc7: {:.5f} +- {:.3f}".format(eval_7.avg, eval_7.std))
                print("Dice eval_avg_dsc8: {:.5f} +- {:.3f}".format(eval_8.avg, eval_8.std))
                print("Dice eval_avg_dsc9: {:.5f} +- {:.3f}".format(eval_9.avg, eval_9.std))
                print("Dice eval_avg_dsc10: {:.5f} +- {:.3f}".format(eval_10.avg, eval_10.std))
                print("Dice eval_avg_dsc: {:.5f} +- {:.3f}".format(eval_avg_dsc.avg, eval_dsc.std))
                print("Dice eval_dsc: {:.5f} +- {:.3f}".format(eval_dsc.avg, eval_dsc.std))

                print("--------------------------------------------")
                print("{:.5f} +- {:.3f}".format(eval_0.avg, eval_0.std))
                print("{:.5f} +- {:.3f}".format(eval_1.avg, eval_1.std))
                print("{:.5f} +- {:.3f}".format(eval_2.avg, eval_2.std))
                print("{:.5f} +- {:.3f}".format(eval_3.avg, eval_3.std))
                print("{:.5f} +- {:.3f}".format(eval_4.avg, eval_4.std))
                print("{:.5f} +- {:.3f}".format(eval_5.avg, eval_5.std))
                print("{:.5f} +- {:.3f}".format(eval_6.avg, eval_6.std))
                print("{:.5f} +- {:.3f}".format(eval_7.avg, eval_7.std))
                print("{:.5f} +- {:.3f}".format(eval_8.avg, eval_8.std))
                print("{:.5f} +- {:.3f}".format(eval_9.avg, eval_9.std))
                print("{:.5f} +- {:.3f}".format(eval_10.avg, eval_10.std))
                print("{:.5f} +- {:.3f}".format(eval_avg_dsc.avg, eval_dsc.std))
                print("{:.5f} +- {:.3f}".format(eval_dsc.avg, eval_dsc.std))

                
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
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--affine_model', type=str, default='experiments/affine/',
                        help='Affine model load directory')
    parser.add_argument('--dir_model', type=str, default='Train_M_Adv/',
                        help='DIR model load directory')
    parser.add_argument('--dataset_dir', type=str, default='Dataset/BTCV_Abdominal_1k/',
                        help='Dataset directory')
    parser.add_argument('--img_size', type=int, default=(128, 512, 512),
                        help='img_size')
    parser.add_argument('--patch_size', type=int, default=(64, 128, 128),
                        help='patch_size')
    
    args = parser.parse_args()

    main(args)