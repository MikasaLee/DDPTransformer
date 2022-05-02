import os
import re
import numpy as np
from math import exp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.backends import cudnn
from glob import glob # https://www.pynote.net/archives/852
from torch.utils.data import Dataset,DataLoader
import torchvision
from torch_radon import Radon, RadonFanbeam

def getNPS_X(size=512):
    fft = np.fft.fftfreq(size)
    fftMultiplyFactor = 10.
    return fft[fft>0]*fftMultiplyFactor,fft
def getNPS_Y(img,ref,fft):
    def nps_cal(img,ref):
        f = np.fft.fft2(img - ref)
        return np.square(np.absolute(f))
    def noisePowerRadAv(im):
        """noisePowerRadAv(2D array), returns radius vector with
            axially averaged data. From CTQA-CP (Erlend Andersen)."""
        x,y=im.shape
        rran=np.arange(x)
        theta=np.linspace(0,np.pi/2.0,x*2,endpoint=False)
        r=np.zeros(x)
        for ang in theta:
            yran=np.rint(rran*np.sin(ang)).astype(int)
            xran=np.rint(rran*np.cos(ang)).astype(int)
            r+=np.ravel(im[[xran],[yran]])
        return r/float(x)
    result = nps_cal(img,ref)
    FFT = noisePowerRadAv(result)
    return FFT[fft>0]

#PSNR
def torchPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img,0,1) - torch.clamp(tar_img,0,1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    return ps

def numpyPSNR(tar_img, prd_img):
    imdff = np.float32(prd_img) - np.float32(tar_img)
    rmse = np.sqrt(np.mean(imdff**2))
    ps = 20*np.log10(255/rmse)
    return ps
#---------------------------------------------------------------------------------------------------------------------
#SSIM
def gaussian(window_size,sigma):
    gauss = torch.Tensor([exp(-(x-window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()
    
def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window
    
def torchSSIM(img1,img2,data_range=1,window_size=11,channel=1,size_average=True):
    if len(img1.size()) == 2:
        shape_ = img1.shape[-1]
        img1 = img1.view(1,1,shape_,shape_)
        img2 = img2.view(1,1,shape_,shape_)
    window = create_window(window_size,channel)
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size//2)
    mu2 = F.conv2d(img2, window, padding=window_size//2)
    mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2) - mu1_mu2

    C1, C2 = (0.01*data_range)**2, (0.03*data_range)**2
    #C1, C2 = 0.01**2, 0.03**2

    ssim_map = ((2*mu1_mu2+C1)*(2*sigma12+C2)) / ((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))
    return ssim_map.mean()



#---------------------------------------------------------------------------------------------------------------------

#dataloader
class ct_dataset(Dataset):
    def __init__(self,data_path,patientsID):
        super().__init__()
        input_path = sorted(glob(os.path.join(data_path,'*_sparse.npy')))
        target_path = sorted(glob(os.path.join(data_path,'*_full.npy')))
        self.input_ = [f for f in input_path if re.search(r"(C|N|L)\d{3}", f)[0] in patientsID]
        self.target_ = [f for f in target_path if re.search(r"(C|N|L)\d{3}", f)[0] in patientsID]
        # new add
        self.upsample = nn.Upsample(size=(512), mode='bilinear',align_corners=False)
    def __len__(self):
        return len(self.input_)
    def __getitem__(self, idx):

        input_img, target_img = self.input_[idx], self.target_[idx]
        input_img, target_img = np.load(input_img),np.load(target_img)
        input_img, target_img =torch.Tensor(input_img),torch.Tensor(target_img)
        return self.upsample(input_img.unsqueeze(0).unsqueeze(0)).squeeze(0),target_img.unsqueeze(0)
    
def get_loader(data_path=None,patientsID_path=None,batch_size=32,num_workers=8):
    train_patientsID = np.load(os.path.join(patientsID_path,'train_patientsID.npy')).flatten()
    valid_patientsID = np.load(os.path.join(patientsID_path,'valid_patientsID.npy')).flatten()
    test_patientsID  = np.load(os.path.join(patientsID_path,'test_patientsID.npy')).flatten()
    
    train_dataset_ = ct_dataset(data_path,train_patientsID)
    valid_dataset_ = ct_dataset(data_path,valid_patientsID)
    test_dataset_  = ct_dataset(data_path,test_patientsID)
    
    print('train_dataset numbers:{}\nvalid_dataset numbers:{}\ntest_dataset numbers:{}\n'.format(len(train_dataset_),len(valid_dataset_),len(test_dataset_)))
          
    return (DataLoader(dataset = train_dataset_,batch_size=batch_size,shuffle=True,num_workers=num_workers),
            DataLoader(dataset = valid_dataset_,batch_size=batch_size,shuffle=True,num_workers=num_workers),
            DataLoader(dataset = test_dataset_,batch_size=batch_size,shuffle=True,num_workers=num_workers) )




#--------------------------------------------------------------------------------------------------------------------------------------------
#dataloader
class ct_dataset2(Dataset):
    def __init__(self,img_path,sinogram_path,patientsID):
        super().__init__()

        img_input_path = sorted(glob(os.path.join(sinogram_path,'*_sparse.npy')))
        img_target_path = sorted(glob(os.path.join(img_path,'*_full.npy'))) 
        
        self.img_input_ = [f for f in img_input_path if re.search(r"(C|N|L)\d{3}", f)[0] in patientsID]
        self.img_target_ = [f for f in img_target_path if re.search(r"(C|N|L)\d{3}", f)[0] in patientsID]
    def __len__(self):
        return len(self.img_input_)
    def __getitem__(self, idx):

        input_img, target_img = self.img_input_[idx], self.img_target_[idx]
        input_img, target_img = np.load(input_img),np.load(target_img)
        input_img, target_img = torch.Tensor(input_img),torch.Tensor(target_img)
        return input_img.unsqueeze(0),target_img.unsqueeze(0)
    

def get_loader2(img_path,sinogram_path,patientsID_path=None,batch_size=32,num_workers=8):
    train_patientsID = np.load(os.path.join(patientsID_path,'train_patientsID.npy')).flatten()
    valid_patientsID = np.load(os.path.join(patientsID_path,'valid_patientsID.npy')).flatten()
    test_patientsID  = np.load(os.path.join(patientsID_path,'test_patientsID.npy')).flatten()
    
    train_dataset_ = ct_dataset2(img_path,sinogram_path,train_patientsID)
    valid_dataset_ = ct_dataset2(img_path,sinogram_path,valid_patientsID)
    test_dataset_  = ct_dataset2(img_path,sinogram_path,test_patientsID)
    
    print('train_dataset numbers:{}\nvalid_dataset numbers:{}\ntest_dataset numbers:{}\n'.format(len(train_dataset_),len(valid_dataset_),len(test_dataset_)))
          
    return (DataLoader(dataset = train_dataset_,batch_size=batch_size,shuffle=True,num_workers=num_workers),
            DataLoader(dataset = valid_dataset_,batch_size=batch_size,shuffle=True,num_workers=num_workers),
            DataLoader(dataset = test_dataset_,batch_size=batch_size,shuffle=True,num_workers=num_workers) )




#---------------------------------------------------------------------------------------------------------------------------
# Fan Beam Projection
def get_fbp(x,n_angles=512,image_size=512):
    fanbeam_angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
    radon_fanbeam = RadonFanbeam(image_size, fanbeam_angles, source_distance=512, det_distance=512, det_spacing=2.5)

    filtered_sinogram = radon_fanbeam.filter_sinogram(x)
    fbp = radon_fanbeam.backprojection(filtered_sinogram)
    return fbp

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