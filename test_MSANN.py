# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 16:20:26 2020

@author: Administrator
"""
import os, time, scipy.io, shutil
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
import cv2
import time
import scipy.misc


from MSANN_model import DSPNet
#from makedataset import Dataset
#
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_checkpoint(checkpoint_dir, num_input_channels):
    if num_input_channels ==3:
        
        if os.path.exists(checkpoint_dir + 'checkpoint.pth.tar'):
            # load existing model
            model_info = torch.load(checkpoint_dir + 'checkpoint.pth.tar')
            print('==> loading existing model:', checkpoint_dir + 'checkpoint.pth.tar')
            net = DSPNet()
            device_ids = [0]
            model = nn.DataParallel(net, device_ids=device_ids).cuda()
            model.load_state_dict(model_info['state_dict'])
            optimizer = torch.optim.Adam(model.parameters())
            optimizer.load_state_dict(model_info['optimizer'])
            cur_epoch = model_info['epoch']
            
        else:
            # create model
            net = DSPNet()
            device_ids = [0]
            model = nn.DataParallel(net, device_ids=device_ids).cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            cur_epoch = 0

    else:
        if os.path.exists(checkpoint_dir + 'checkpoint.pth.tar'):
            # load existing model
            model_info = torch.load(checkpoint_dir + 'checkpoint.pth.tar')
            print('==> loading existing model:', checkpoint_dir + 'checkpoint.pth.tar')
            net = DSPNet()
            device_ids = [0]
            model = nn.DataParallel(net, device_ids=device_ids).cuda()
            model.load_state_dict(model_info['state_dict'])
            optimizer = torch.optim.Adam(model.parameters())
            optimizer.load_state_dict(model_info['optimizer'])
            cur_epoch = model_info['epoch']
            
        else:
            # create model
            net = DSPNet()
            device_ids = [0]
            model = nn.DataParallel(net, device_ids=device_ids).cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            cur_epoch = 0


    return model, optimizer,cur_epoch

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
	torch.save(state, checkpoint_dir + 'checkpoint.pth.tar')
	if is_best:
		shutil.copyfile(checkpoint_dir + 'checkpoint.pth.tar',checkpoint_dir + 'model_best.pth.tar')

def adjust_learning_rate(optimizer, epoch, lr_update_freq):
	if not epoch % lr_update_freq and epoch:
		for param_group in optimizer.param_groups:
			param_group['lr'] = param_group['lr'] * 0.1
			print( param_group['lr'])
	return optimizer

def test_synthetic(test_syn,result_syn, model):
    '''test synthetic gamma images and set noiselevel(15,30,50,75)'''
    
    files = os.listdir(test_syn)
    time_all = 0
    for j in range(len(files)):   
        model.eval()
        with torch.no_grad():
            img_c =  cv2.imread(test_syn + '/' + files[j],0)/255.
            start = time.time()

            #add noise noiselevel=[5,10,15,20]
            #noise_img = -np.log(img_c+1e-3) + (-np.log(np.random.gamma(shape=noiselevel[i],scale = 1/noiselevel[i],size =(w,h))+1e-3))
            input_var = torch.from_numpy(img_c.copy()).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)
            input_var = input_var.cuda()
            
            input_var = -torch.log(input_var+1e-3)
            
            _,_,output = model(input_var)  
            output = torch.exp(-output)
            end = time.time()
            output_np = output.squeeze().cpu().detach().numpy()           

            time_all = time_all + end-start

            cv2.imwrite(result_syn + '/'  + files[j][:-4]+'_MSANN'+files[j][-4:],np.clip( output_np*255,0.0,255.0))                 
    print('Average Running Time: %f'%(time_all/len(files)))     
    print(time_all/len(files))
            
if __name__ == '__main__': 
    checkpoint_dir = './checkpoint/'
    model, optimizer,_ = load_checkpoint(checkpoint_dir,1)
    test_syn = './input'
    result_syn = './output'
    test_synthetic(test_syn,result_syn, model)


