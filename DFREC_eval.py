import torch
import numpy as np
import cv2
import sys,os
from hashlib import md5
import shutil
import torch.nn.functional as F
import torchvision
from torch import nn, optim
from tqdm import tqdm
import imageio
from torchvision import transforms
import segmentation_models as smp
from models.unet import SourceRecoverNet_Attention2,TargetRecoverNet
from models.mae import MAE
from PIL import Image
def load_model(gunet,recover1,recover2, path):
    ckpt = torch.load(path, map_location="cpu")
    # print(ckpt)
    start_epoch = ckpt.get("epoch", 0)
    best_acc = ckpt.get("acc1", 0.0)
    gunet.load_state_dict(ckpt["state_dict"],strict=False)
    recover1.load_state_dict(ckpt["recover1_state_dict"])
    recover2.load_state_dict(ckpt["recover2_state_dict"])
    return gunet,recover1,recover2

def buildmodel():
    unet = smp.DeepLabV3Plus(
        encoder_name="efficientnet-b0",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pretrained weights for encoder initialization                    
        classes=1,
        activation='sigmoid'   
    )
    return unet

def idreceval(imgpath,savepath1,savepath2):
    pilimg=Image.open(imgpath).convert('RGB').resize((224,224))
    imgtensor=transforms.ToTensor()(pilimg).float()
    anchorimg=imgtensor.unsqueeze(0)
    anchorimg = anchorimg.cuda()
    maskpred = gunet(anchorimg)
    anchor1=maskpred*anchorimg
    anchor2=(1-maskpred)*anchorimg
    t1,f2,fsrc,ftgt=recovery1(anchor1)
    x_pred, anchorpatch,targetpatch, t2,maemask=recovery2(anchorimg,anchorimg,maskpred,f2)
    t1img=t1.squeeze(0).detach().cpu().numpy()
    t1img=np.array(np.transpose(t1img*255,(1,2,0)),dtype=np.uint8)
    t2img=t2.squeeze(0).detach().cpu().numpy()
    t2img=np.array(np.transpose(t2img*255,(1,2,0)),dtype=np.uint8)
    Image.fromarray(t1img).save(savepath1)
    Image.fromarray(t2img).save(savepath2)

gunet=buildmodel().cuda()
recovery1 = SourceRecoverNet_Attention2(idout=98).cuda()
vit = torchvision.models.vit_l_16().cuda()
recovery2 = MAE(vit,maskratio=0.5,threshold=250).cuda()

gunet,recovery1,recovery2=load_model(gunet,recovery1,recovery2,'splitidentity_ftm.pth')
print('IDREC model loaded!')
gunet.eval()
recovery1.eval()
recovery2.eval()
torch.autograd.set_grad_enabled(False)   
for file in os.listdir('imgs'):
    idreceval('imgs/'+file,'outputs/rec_source_'+file,'outputs/rec_target_'+file)
    print('imgs/'+file,'finished')