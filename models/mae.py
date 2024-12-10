# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import torch
import torchvision
from torch import nn
import torchvision.utils as vutils
from lightly.models import utils
from lightly.models.modules import masked_autoencoder
from lightly.transforms.mae_transform import MAETransform
import random

class MAE(nn.Module):
    def __init__(self, vit,maskratio=0.75,threshold=250,para=False):
        super().__init__()

        decoder_dim = 512
        self.para=para
        if para:
            self.threshold=nn.Parameter(torch.tensor(threshold).float())
        else:
            self.threshold=torch.tensor(threshold)
        self.mask_ratio = maskratio
        self.patch_size = vit.patch_size
        self.sequence_length = vit.seq_length
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.backbone = masked_autoencoder.MAEBackbone.from_vit(vit)
        self.decoder = masked_autoencoder.MAEDecoder(
            seq_length=vit.seq_length,
            num_layers=1,
            num_heads=16,
            embed_input_dim=vit.hidden_dim+196,
            hidden_dim=decoder_dim,
            mlp_dim=decoder_dim * 4,
            out_dim=vit.patch_size**2 * 3,
            dropout=0,
            attention_dropout=0,
        )
        

    def forward_encoder(self, images, idx_keep=None):
        return self.backbone.encode(images, idx_keep)

    def forward_decoder(self, x_encoded, idx_keep, idx_mask):
        # build decoder input
        batch_size = x_encoded.shape[0]
        x_decode = self.decoder.embed(x_encoded)
        x_masked = utils.repeat_token(
            self.mask_token, (batch_size, self.sequence_length)
        )
        x_masked = utils.set_at_index(x_masked, idx_keep, x_decode.type_as(x_masked))

        # decoder forward pass
        x_decoded = self.decoder.decode(x_masked)

        # predict pixel values for masked tokens
        x_pred = utils.get_at_index(x_decoded, idx_mask)
        x_pred = self.decoder.predict(x_pred)
        return x_pred
    
    def calcmask(self,diff):
        patches = utils.patchify(diff, self.patch_size)
        initweight=torch.sum(patches,dim=2)
        meanweight=initweight.mean(dim=1)
        noise = torch.rand(patches.shape[0], patches.shape[1]).cuda()
        if self.para:
            threshold=nn.Sigmoid()(self.threshold)*256
            initweight[meanweight>=threshold]=noise[meanweight>=threshold]
            initweight[meanweight<threshold]=torch.where(initweight[meanweight<threshold]<=50,noise[meanweight<threshold],initweight[meanweight<threshold])
        else:
            initweight[meanweight>=self.threshold]=noise[meanweight>=self.threshold]
            initweight[meanweight<self.threshold]=torch.where(initweight[meanweight<self.threshold]<=50,noise[meanweight<self.threshold],initweight[meanweight<self.threshold])

            
        num_keep = int(self.sequence_length * (1 - self.mask_ratio))
        # get indices of tokens to keep
        indices = torch.argsort(initweight, dim=1)
        idx_keep = indices[:, :num_keep]
        idx_mask = indices[:, num_keep:]

        return idx_keep,idx_mask       


    def forward(self, images,targetimg1,maskimg,feature):
        batch_size = images.shape[0]
        idx_keep,idx_mask=self.calcmask(maskimg)
        x_encoded = self.forward_encoder(images, idx_keep)
        x_encoded = torch.cat((x_encoded,feature),dim=2)
        x_pred = self.forward_decoder(x_encoded, idx_keep, idx_mask)

        # get image patches for masked tokens
        patches = utils.patchify(images, self.patch_size)

        # must adjust idx_mask for missing class token
        target = utils.get_at_index(patches, idx_mask)
        targetimg1patches = utils.patchify(targetimg1, self.patch_size)
        targetimg1patch=utils.get_at_index(targetimg1patches, idx_mask)
        x_pred =nn.Sigmoid()(x_pred)
        reconstruct=utils.set_at_index(patches, idx_mask,x_pred)

        reconstruct=utils.unpatchify(reconstruct, self.patch_size)
        maskzeros=torch.zeros_like(x_pred)
        maskconstruct=utils.set_at_index(patches, idx_mask,maskzeros)
        maskconstruct=utils.unpatchify(maskconstruct, self.patch_size)
        return x_pred, target, targetimg1patch, reconstruct,maskconstruct

