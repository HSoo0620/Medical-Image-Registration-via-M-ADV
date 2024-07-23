from torch.utils.tensorboard import SummaryWriter
import os, utils, glob, losses, gc, sys, argparse
import sys
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from natsort import natsorted
from models.TransMorph_Origin_Affine import CONFIGS as CONFIGS_TM
import models.TransMorph_Origin_Affine as TransMorph_affine
import nibabel as nib
from models.M_Adv_Func import save_checkpoint, adjust_learning_rate, save_img

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', '1'):
        return True
    elif v.lower() in ('no', 'false', '0'):
        return False

class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def main():
    batch_size = args.batch_size
    train_dir = args.dataset_dir + 'train/'
    val_dir = args.dataset_dir + 'test/'
    save_validation_img = args.save_validation_img

    weights = [1.0, 0.5] # loss weights
    save_dir = 'test/'
    if not os.path.exists('experiments/'+save_dir):
        os.makedirs('experiments/'+save_dir)
    if not os.path.exists('logs/'+save_dir):
        os.makedirs('logs/'+save_dir)
    sys.stdout = Logger('logs/'+save_dir)
    lr = args.learning_rate # learning rate
    epoch_start = 0
    max_epoch = args.max_epoch #max traning epoch

    # Use Pre-trained Model  
    cont_training = args.pre_train

    '''
    Initialize model
    '''
    config = CONFIGS_TM['TransMorph-Affine']
    model = TransMorph_affine.SwinAffine(config)
    model.cuda()

    AffInfer = TransMorph_affine.ApplyAffine()
    AffInfer.cuda()

    '''
    Continue training
    '''
    if cont_training:
        model_dir = 'experiments/'+save_dir
        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch,0.9),8)
        best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[0])['state_dict']
        print('Affine: {} loaded!'.format(natsorted(os.listdir(model_dir))[-1]))
        model.load_state_dict(best_model)
    else:
        updated_lr = lr

    '''
    Initialize training
    '''

    train_composed = transforms.Compose([trans.NumpyType((np.float32, np.float32))])
    val_composed = transforms.Compose([trans.NumpyType((np.float32, np.float32))])

    train_set = datasets.BTCVAffineDataset(glob.glob(train_dir + '*.nii.gz'), transforms=train_composed)
    val_set = datasets.BTCVInferDataset_ori(glob.glob(val_dir + '*.nii.gz'), transforms=val_composed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)
    
    # Optimizers
    optimizer = optim.Adam(model.parameters(), lr=updated_lr, amsgrad=True)
    criterion_dsc_ori = losses.Dice(bg=1)
    criterion_ncc_ori = losses.NCC_ori()

    best_dsc = 1e-10
    writer = SummaryWriter(log_dir='logs/'+save_dir)
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        print('Epoch {} :'.format(epoch))
        '''
        Training
        '''
        loss_all = utils.AverageMeter()
        idx = 0
        for data in train_loader:
            idx += 1
            model.train()
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            data = [t.cuda() for t in data]

            ####################
            # Affine transform
            ####################
            x = data[0]
            y = data[1]
            x_seg = nn.functional.one_hot(data[2].long(), num_classes=12)# 0~ 14
            x_seg = torch.squeeze(x_seg, 1)
            x_seg = x_seg.permute(0, 4, 1, 2, 3).contiguous()
            
            y_seg = nn.functional.one_hot(data[3].long(), num_classes=12)# 0~ 14
            y_seg = torch.squeeze(y_seg, 1)
            y_seg = y_seg.permute(0, 4, 1, 2, 3).contiguous()

            x_in = torch.cat((x, y), dim=1)
            out, mat, inv_mat, Rigid_out, Rigid_mat, Rigid_inv_mat = model(x_in)
            out_seg = AffInfer(x_seg.cuda().float(), mat)

            loss_ncc = criterion_ncc_ori(out, y)
            loss_ncc_w = loss_ncc * weights[0]

    
            loss_dsc = criterion_dsc_ori(out_seg, y_seg)
            loss_dsc_w = loss_dsc * weights[1]
            loss = loss_ncc_w + loss_dsc_w

            loss_all.update(loss.item(), x.numel())
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()
            print('Iter {} of {} LOSS: {:.6f} NCC: {:.6f}, DSC: {:.6f}'.format(idx, len(train_loader),
                                                         loss.item(), loss_ncc.item(), 1-loss_dsc.item()))
            del out, out_seg
        writer.add_scalar('Loss/train', loss_all.avg, epoch)
        print('Epoch {}, loss {:.4f}'.format(epoch, loss_all.avg))

        '''
        Validation
        '''
        eval_dsc_ori = utils.AverageMeter()
        with torch.no_grad():
            idx = 0
            for data in val_loader:
                idx += 1
                model.eval()
                data = [t.cuda() for t in data]
                x = data[0]
                y = data[1]
                x_seg = nn.functional.one_hot(data[2].long(), num_classes=12)# 0~ 14
                x_seg = torch.squeeze(x_seg, 1)
                x_seg = x_seg.permute(0, 4, 1, 2, 3).contiguous()
                
                y_seg = nn.functional.one_hot(data[3].long(), num_classes=12)# 0~ 14
                y_seg = torch.squeeze(y_seg, 1)
                y_seg = y_seg.permute(0, 4, 1, 2, 3).contiguous()


                x_in = torch.cat((x, y), dim=1)
                out, mat, inv_mat, Rigid_out, Rigid_mat, Rigid_inv_mat = model(x_in)
                def_out = AffInfer(x_seg.cuda().float(), mat)   

                dsc2 = 1-criterion_dsc_ori(def_out, y_seg)
                eval_dsc_ori.update(dsc2.item(), x.size(0))
                print('Iter {} of {} DSC: {:.6f}'.format(idx, len(val_loader),
                                                       dsc2.item()))
                if idx == 30:
                    print("--everage Dice: {:.5f} +- {:.3f}".format(eval_dsc_ori.avg, eval_dsc_ori.std))
                
                save_result_dir = 'results/0313_Affine/'

                if not os.path.exists(save_result_dir):
                  os.makedirs(save_result_dir)
        
                if save_validation_img:
                    if_name = save_result_dir + str(idx)+ '_pred_mask.nii.gz'
                    gt_name = save_result_dir + str(idx)+ '_gt_mask.nii.gz'

                    save_img(def_out.float(), if_name, 'label') 
                    save_img(y_seg.float(), gt_name, 'label') 
                
        best_dsc = max(eval_dsc_ori.avg, best_dsc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_dsc': best_dsc,
            'optimizer': optimizer.state_dict(),
        }, save_dir='experiments/'+save_dir, filename='dsc{:.4f} epoch{:.1f}.pth.tar'.format(eval_dsc_ori.avg,epoch))
        writer.add_scalar('Loss_GF/val', eval_dsc_ori.avg, epoch)

        loss_all.reset()
        eval_dsc_ori.reset()

    writer.close()

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)

def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=3):
    torch.save(state, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + '*'), reverse=True)
    while len(model_lists) > max_model_num:
        os.remove(model_lists[-1])
        model_lists = natsorted(glob.glob(save_dir + '*'))

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
    parser.add_argument('--dir_model', type=str, default='experiments/test/',
                        help='DIR model load directory')
    parser.add_argument('--dataset_dir', type=str, default='Dataset/BTCV_Abdominal_1k/',
                        help='Dataset directory')
    parser.add_argument('--pre_train', type=str2bool, default='False',
                        help='pre-train load True or False')
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