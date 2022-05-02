# jupyter 不支持 DDP 所以很不幸我需要用.py运行
import os
import re
import argparse
import random
import time
import datetime
import numpy as np
import math
from math import exp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.backends import cudnn
from glob import glob # https://www.pynote.net/archives/852
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.cuda.amp import autocast,GradScaler
from einops import rearrange, repeat
import torchvision
from utils import torchPSNR,torchSSIM,get_fbp
from timm.models.layers import trunc_normal_

def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True
        
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank',type=int,default=0,help='node rank for DDP')
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(
    backend='nccl',
    rank=args.local_rank
)
init_seeds(1 + torch.distributed.get_rank())

# Input Projection
class InputProj(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, kernel_size=3, stride=1, norm_layer=None,act_layer=nn.LeakyReLU,isPadding=False):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=kernel_size//2 if isPadding == True else 0),
            act_layer(inplace=True)
        )
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        if self.norm is not None:
            x = self.norm(x)
        return x
# Output Projection
class OutputProj(nn.Module):
    def __init__(self, in_channel=64, out_channel=3, kernel_size=3, stride=1, norm_layer=None,act_layer=None,isPadding=False):
        super().__init__()
        self.proj = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=kernel_size//2 if isPadding == True else 0),
        )
        if act_layer is not None:
            self.proj.add_module(act_layer(inplace=True))
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x
#########################################
######## Embedding for q,k,v ########
# LinearProjection(dim,num_heads,dim//num_heads,bias=qkv_bias)
class LinearProjection(nn.Module):
    def __init__(self, dim = 512, heads = 8, dim_head = 64, dropout = 0., bias=True):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias = bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = bias)

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        attn_kv = x if attn_kv is None else attn_kv
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(attn_kv).reshape(B_, N, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4) 
        q = q[0]
        k, v = kv[0], kv[1] 
        return q,k,v
class LeFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU,drop = 0.):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
                                act_layer())
        self.dwconv = nn.Sequential(nn.Conv2d(hidden_dim,hidden_dim,groups=hidden_dim,kernel_size=3,stride=1,padding=1),
                        act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))

    def forward(self, x):
        # bs x hw x c
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))

        x = self.linear1(x)

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h = hh, w = hh)
        # bs,hidden_dim,32x32

        x = self.dwconv(x)

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h = hh, w = hh)

        x = self.linear2(x)

        return x
#########################################
########### window-based self-attention #############
class WindowAttention(nn.Module):
    def __init__(self, dim, win_size,num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.win_size = win_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.win_size[0]) # [0,...,Wh-1]
        coords_w = torch.arange(self.win_size[1]) # [0,...,Ww-1]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = LinearProjection(dim,num_heads,dim//num_heads,bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, attn_kv=None, mask=None):
        B_, N, C = x.shape
        q, k, v = self.qkv(x,attn_kv)

        q = q * self.scale
        
        attn = (q @ k.transpose(-2, -1))
#         print(attn.shape)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
#         print(relative_position_bias.shape)
        ratio = attn.size(-1)//relative_position_bias.size(-1)
        relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d = ratio)
#         print(relative_position_bias.shape)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            mask = repeat(mask, 'nW m n -> nW m (n d)',d = ratio)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N*ratio) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N*ratio)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
########### window operation#############
def window_partition(x, win_size, dilation_rate=1):
    B, H, W, C = x.shape
    if dilation_rate !=1:
        x = x.permute(0,3,1,2) # B, C, H, W
        assert type(dilation_rate) is int, 'dilation_rate should be a int'
        x = F.unfold(x, kernel_size=win_size,dilation=dilation_rate,padding=4*(dilation_rate-1),stride=win_size) # B, C*Wh*Ww, H/Wh*W/Ww
        windows = x.permute(0,2,1).contiguous().view(-1, C, win_size, win_size) # B' ,C ,Wh ,Ww
        windows = windows.permute(0,2,3,1).contiguous() # B' ,Wh ,Ww ,C
    else:
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C) # B' ,Wh ,Ww ,C
    return windows

def window_reverse(windows, win_size, H, W, dilation_rate=1):
    # B' ,Wh ,Ww ,C
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    if dilation_rate !=1:
        x = windows.permute(0,5,3,4,1,2).contiguous() # B, C*Wh*Ww, H/Wh*W/Ww
        x = F.fold(x, (H, W), kernel_size=win_size, dilation=dilation_rate, padding=4*(dilation_rate-1),stride=win_size)
    else:
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x
#########################################
########### LeWinTransformer #############
class LeWinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, win_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        assert 0 <= self.shift_size < self.win_size, "shift_size must in 0-win_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, win_size=[self.win_size,self.win_size], num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = LeFF(dim,mlp_hidden_dim,act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        assert self.win_size <= H, "win_size:{},H:{}".format(self.win_size,H)
        
        ## input mask
        if mask != None:
            input_mask = F.interpolate(mask, size=(H,W)).permute(0,2,3,1)
            input_mask_windows = window_partition(input_mask, self.win_size) # nW, win_size, win_size, 1
            attn_mask = input_mask_windows.view(-1, self.win_size * self.win_size) # nW, win_size*win_size
            attn_mask = attn_mask.unsqueeze(2)*attn_mask.unsqueeze(1) # nW, win_size*win_size, win_size*win_size
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        ## shift mask
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            shift_mask = torch.zeros((1, H, W, 1)).type_as(x)
            h_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    shift_mask[:, h, w, :] = cnt
                    cnt += 1
            shift_mask_windows = window_partition(shift_mask, self.win_size)  # nW, win_size, win_size, 1
            shift_mask_windows = shift_mask_windows.view(-1, self.win_size * self.win_size) # nW, win_size*win_size
            shift_attn_mask = shift_mask_windows.unsqueeze(1) - shift_mask_windows.unsqueeze(2) # nW, win_size*win_size, win_size*win_size
            attn_mask = attn_mask or shift_attn_mask
            attn_mask = attn_mask.masked_fill(shift_attn_mask != 0, float(-100.0))
            
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.win_size)  # nW*B, win_size, win_size, C  N*C->C
        x_windows = x_windows.view(-1, self.win_size * self.win_size, C)  # nW*B, win_size*win_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, win_size*win_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C)
        shifted_x = window_reverse(attn_windows, self.win_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        del attn_mask
        return x
class DenseBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DenseBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1),
            nn.GELU()
        )

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.conv(x).flatten(2).transpose(1,2).contiguous()  # B H*W C
        return out
    
#########################################
########### Basic layer of Uformer ################
class BasicBlock(nn.Module):
    def __init__(self, input_chn,dim, depth, num_heads, win_size,
                 mlp_ratio=4., qkv_bias=True,  drop=0., attn_drop=0., norm_layer=nn.LayerNorm):
        super(BasicBlock,self).__init__()

        # build blocks
        self.blocks = nn.ModuleList([
            LeWinTransformerBlock(dim=dim,
                                 num_heads=num_heads, win_size=win_size,
                                 shift_size=0 if (i % 2 == 0) else win_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 norm_layer=norm_layer)
            for i in range(depth)])
        self.cat2 = DenseBlock(in_channel=dim*2, out_channel=dim)
    def forward(self, x):
        xs = []
        for blk in self.blocks:
            xs.append(blk(x))
        xs = self.cat2(torch.cat([xs[0],xs[1]],dim=-1))
        return xs
class sinogram_domain(nn.Module):
    def __init__(self,input_channels,embed_dim=64):
        super().__init__()
        layer = []
        self.input_proj = InputProj(in_channel=input_channels, out_channel=embed_dim, 
                                    kernel_size=4, stride=4, act_layer=nn.LeakyReLU)
        self.output_proj = OutputProj(in_channel=embed_dim, out_channel=input_channels, kernel_size=4, stride=4)
        self.num_layers = 5
        for i in range(self.num_layers):
            layer.append(BasicBlock(input_chn=embed_dim,
                        dim=embed_dim, 
                        depth=2,
                        num_heads=16,
                        win_size=8,
                        mlp_ratio=4.,
                        qkv_bias=True,
                        drop=0.2, attn_drop=0.2))
        self.net = nn.Sequential(*layer)
    def forward(self,X):
        for i,layer in enumerate(self.net):
            if i == 0 :
                temp1 = X
                X = self.input_proj(X)
            elif i == 1 :
                temp2 = X
            elif i == self.num_layers -1 :
                X = X + temp2 
            else:
                temp2 = X + temp2
            X = layer(X)
        return self.output_proj(X) + temp1
class image_domain(nn.Module):
    def __init__(self,input_channels,embed_dim=64):
        super().__init__()
        layer = []
        self.input_proj = InputProj(in_channel=input_channels, out_channel=embed_dim, 
                                    kernel_size=4, stride=4, act_layer=nn.LeakyReLU)
        self.output_proj = OutputProj(in_channel=embed_dim, out_channel=input_channels, kernel_size=4, stride=4)
        self.num_layers = 7
        for i in range(self.num_layers):
            layer.append(BasicBlock(input_chn=embed_dim,
                        dim=embed_dim, 
                        depth=2,
                        num_heads=16,
                        win_size=8,
                        mlp_ratio=4.,
                        qkv_bias=True,
                        drop=0.2, attn_drop=0.2))
        self.net = nn.Sequential(*layer)
    def forward(self,X):
        for i,layer in enumerate(self.net):
            if i == 0 :
                temp1 = X
                X = self.input_proj(X)
            elif i == 1 :
                temp2 = X
            elif i == self.num_layers -1 :
                X = X + temp2 
            else:
                temp2 = X + temp2
            X = layer(X)
        return self.output_proj(X) + temp1
#dataloader
class ct_dataset(Dataset):
    def __init__(self,data_path,patientsID):
        super().__init__()
        input_path = sorted(glob(os.path.join(data_path,'*_sparse.npy')))
        target_path = sorted(glob(os.path.join(data_path,'*_full.npy')))
        self.input_ = [f for f in input_path if re.search(r"(C|N|L)\d{3}", f)[0] in patientsID]
        self.target_ = [f for f in target_path if re.search(r"(C|N|L)\d{3}", f)[0] in patientsID]
        
        self.upsample = nn.Upsample(size=(512), mode='bilinear',align_corners=False)
        
    def __len__(self):
        return len(self.input_)
    def __getitem__(self, idx):

        input_img, target_img = self.input_[idx], self.target_[idx]
        input_img, target_img = np.load(input_img),np.load(target_img)
        input_img,target_img = torch.Tensor(input_img),torch.Tensor(target_img)
        # return input_img.unsqueeze(0),target_img.unsqueeze(0)
        return self.upsample(input_img.unsqueeze(0).unsqueeze(0)).squeeze(0),target_img.unsqueeze(0)
    
def get_loader(data_path=None,patientsID_path=None,batch_size=32,num_workers=8):
    train_patientsID = np.load(os.path.join(patientsID_path,'train_patientsID.npy')).flatten()
    valid_patientsID = np.load(os.path.join(patientsID_path,'valid_patientsID.npy')).flatten()
    test_patientsID  = np.load(os.path.join(patientsID_path,'test_patientsID.npy')).flatten()

    train_dataset_ = ct_dataset(data_path,train_patientsID)
    valid_dataset_ = ct_dataset(data_path,valid_patientsID)
    test_dataset_  = ct_dataset(data_path,test_patientsID)
    sampler1 = DistributedSampler(train_dataset_)
    sampler2 = DistributedSampler(valid_dataset_)
    sampler3 = DistributedSampler(test_dataset_)
    
    print('train_dataset numbers:{}\nvalid_dataset numbers:{}\ntest_dataset numbers:{}\n'.format(len(train_dataset_),len(valid_dataset_),len(test_dataset_)))
          
    return (DataLoader(dataset = train_dataset_,batch_size=batch_size,num_workers=num_workers,sampler=sampler1,pin_memory=True),
            DataLoader(dataset = valid_dataset_,batch_size=batch_size,num_workers=num_workers,sampler=sampler2,pin_memory=True),
            DataLoader(dataset = test_dataset_,batch_size=batch_size,num_workers=num_workers,sampler=sampler3,pin_memory=True))

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-4):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss
    
def train2(net,train_iter,valid_iter,num_gpus,num_epochs,lr,full_view_size=512,save_name = None): 

    optimizer = torch.optim.Adam(net.parameters(),lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[int(num_epochs*(3/5)),num_epochs],gamma = 1)
    
    device = torch.device('cuda',args.local_rank)
    net.to(device)
    net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = nn.parallel.DistributedDataParallel(
        net,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters = True
        )
    torch.backends.cudnn.benchmark = True
    
    loss = CharbonnierLoss(1e-4).cuda()
    # upsample = nn.Upsample(size=(full_view_size,512), mode='bilinear',align_corners=False)
    sd_net = sinogram_domain(1,embed_dim=32)
    sd_net.load_state_dict(torch.load('./result_model/sinogram_domain_64.pkl'),strict=True)
    sd_net.to(device)
    sd_net = nn.SyncBatchNorm.convert_sync_batchnorm(sd_net)
    sd_net = nn.parallel.DistributedDataParallel(
        sd_net,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters = True
        )
    sd_net.eval()
    best_psnr = 34.88  # 先定义成根据 psnr 值来保存模型
    checkpoint = len(train_iter) // 10   # origin value: // 5
    train_loss,valid_psnr,valid_loss = [],[],[]
    net.train()
    for epoch in range(num_epochs):
        print("epoch:{}".format(epoch))
        epoch_start_time = time.time()
        # train
        train_iter.sampler.set_epoch(epoch)
        scaler = GradScaler()
        for i,(input_,target_) in enumerate(train_iter): 
            # input_ = upsample(input_)
            input_,target_ = input_.to(device),target_.to(device)
            with torch.no_grad():    
                input_ = sd_net(input_)
                input_,target_ = get_fbp(input_,n_angles=full_view_size),get_fbp(target_,n_angles=full_view_size)            
            optimizer.zero_grad()
            pred_ = net(input_)
            l = loss(pred_,target_)
            scaler.scale(l).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss.append(l)
            
            # valid
            # if i >= 4*checkpoint and  i % checkpoint == 0 :
            if i != 0 and i % checkpoint == 0 :
                train_loss = torch.stack(train_loss).mean().item()
                print("\t i:{} \t train loss:{}".format(i,train_loss))
                net.eval()
                # 验证集的batch-size为1
                for i,(input_,target_) in enumerate(valid_iter):
                    # input_ = upsample(input_)
                    input_,target_ = input_.to(device),target_.to(device)
                    with torch.no_grad():      # 验证的时候不用计算梯度，取消掉它减少时间和空间的损耗
                        input_ = sd_net(input_)
                        input_,target_ = get_fbp(input_,n_angles=full_view_size),get_fbp(target_,n_angles=full_view_size)            
                        pred_ = net(input_)
                        l = loss(pred_,target_)
                        valid_loss.append(l)
                        valid_psnr.append(torchPSNR(pred_, target_))
                valid_loss = torch.stack(valid_loss).mean().item()
                valid_psnr  = torch.stack(valid_psnr).mean().item()
                print('\tvalid_loss: {:.6f}   valid_psnr: {:.6f}'.format(valid_loss,valid_psnr))
                if dist.get_rank() == 0 and best_psnr < valid_psnr:
                    best_psnr = valid_psnr
                    print('best psnr update: {:.2f}'.format(best_psnr))
                    if save_name:
                        torch.save(net.module.state_dict(), save_name)
                        print('saving model with psnr {:.2f}'.format(best_psnr))
                train_loss,valid_psnr,valid_loss = [],[],[]
                net.train()
        scheduler.step()
        if dist.get_rank() == 0:
            print("Epoch: {}\tTime: {:.4f}\t".format(epoch, time.time()-epoch_start_time))
        
if __name__ == '__main__':      
    batch_size = 5
    lr,num_epochs,num_gpus,num_workers = 1e-4,30,2,8
    data_path,patientsID_path= './2020_mayo_sinogram_img','./2020_mayo_patientsID'
    # data_path,patientsID_path= './mix_sinogram_img','./2020_mayo_patientsID'
    train_iter,valid_iter,_=get_loader(data_path=data_path,
                                       patientsID_path=patientsID_path,batch_size=batch_size,num_workers=num_workers)

    save_name ='./result_model/image_domain_64.pkl'  #2是每个BasicUformerLayer做了跳跃连接，1是没做。3是在DenseConv加了Gelu
    net = image_domain(1,embed_dim=32)
    net.load_state_dict(torch.load(save_name),strict=True)
    # save_name ='./result_model/image_domain_64.pkl'
    train2(net,train_iter,valid_iter,num_gpus,num_epochs,lr,save_name=save_name)
    torch.cuda.empty_cache()