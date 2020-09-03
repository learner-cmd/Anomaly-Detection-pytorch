import cv2 as cv
import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader
import itertools
from torch import nn
from torch.nn import functional as F
import torch as t

#create marks for anomaly objects
def mark(fold,frames,block,d,distance,r):
    image_fold = "D:/data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/Test_rs" + "%03d"%(fold)
    if(d>r*r):
        for k in range(10):
            frame = (frames-1)*10+k
            image_name = "%03d" % (frame + 1) + ".tif"
            image_dir = os.path.join(image_fold, image_name)
            image = cv.imread(image_dir)
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            temp = np.zeros((20,20))
            i = block%18
            j = int(block/18)
            temp[0:20,0:20] = image[(80+j*20):(80+j*20+20), i*20:(i*20+20)]
            temp[0:2] = 255
            temp[18:20] = 255
            temp[:,0:2] = 255
            temp[:,18:20] = 255
            image[80+j*20:(80+j*20+20), i*20:(i*20+20)] = temp[0:20,0:20]
            cv.imwrite(image_dir,image)
            
class dataTest(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
    def __len__(self):
        return 12*12*18*8

    def __getitem__(self, idx):
        fold = int(idx/(18*8*12)) +1 #第fold文件夹
        frame = int((idx-(fold-1)*18*8*12)/(18*8))+1 #第frame个十帧
        i = int((idx-(fold-1)*18*8*12-(frame-1)*(18*8))/18) #坐标（i，j）个像素块
        j = (idx-(fold-1)*18*8*12-(frame-1)*(18*8))%18
        sample = self.get_single_video_x(fold,frame,i,j)
        if self.transform:
            sample = self.transform(sample)
        return sample


    def get_single_video_x(self, fold, frame, i, j ):
        ablock = np.zeros((1,10,160,360))
        mblock = np.zeros((1,20,160,360))
        gblock = np.zeros((1,10,160,360))
        video_a = np.zeros((1,10,20,20))
        video_m = np.zeros((1,20,20,20))
        video_g = np.zeros((1,10,20,20))

        afold_path = "D:/data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/Test"+"%03d"%(fold)
        mfold_path = "D:/data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/opflowTest"+"%03d"%(fold)
        gfold_path = "D:/data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/Test"+"%03d"%(fold)+"_gt"
        aframe_start = (frame-1)*10+1
        mframe_start = (frame-1)*20+1
        for k in range(10):
            aimage_name = "%03d"%(aframe_start+k)+".tif"
            aimage_path=os.path.join(afold_path,aimage_name)
            aimage = cv.imread(aimage_path)
            aimage = cv.cvtColor(aimage, cv.COLOR_BGR2GRAY)
            ablock[0,k,:,:] =aimage[80:240,0:360]
        for k in range(20):
            mimage_name = "%03d"%(mframe_start+k)+".npy"
            mimage_path=os.path.join(mfold_path,mimage_name)
            mimage = np.load(mimage_path)
            mblock[0,k,:,:] =mimage[80:240,0:360] 
        for k in range(10):
            if (fold==6 or fold==9 or fold==12):
                gimage_name = "frame%03d"%(aframe_start+k)+".bmp"
            else:
                gimage_name = "%03d"%(aframe_start+k)+".bmp"
            gimage_path=os.path.join(gfold_path,gimage_name)
            gimage = cv.imread(gimage_path)
            gimage = cv.cvtColor(gimage, cv.COLOR_BGR2GRAY)
            gblock[0,k,:,:] =gimage[80:240,0:360]
        video_a[0,:,:,:] = ablock[0,0:10,i*20:i*20+20,j*20:j*20+20]
        video_m[0,:,:,:] = mblock[0,0:20,i*20:i*20+20,j*20:j*20+20]
        video_g[0,:,:,:] = gblock[0,0:10,i*20:i*20+20,j*20:j*20+20]
        
        return video_a,video_m,video_g
        
class block1(nn.Module):
    def __init__(self, inchannel, hidden1, outchannel, stride):
        super(block1, self).__init__()
        self.left = nn.Sequential(
            nn.Conv3d(in_channels=inchannel,out_channels= hidden1,kernel_size= 1,stride= stride,padding= 0),
            nn.BatchNorm3d(hidden1),
            nn.ReLU(inplace=True),

            nn.Conv3d(in_channels=hidden1,out_channels= hidden1,kernel_size= 3,stride= 1,padding= 1),
            nn.BatchNorm3d(hidden1),
            nn.ReLU(inplace=True),

            nn.Conv3d(in_channels=hidden1,out_channels= outchannel,kernel_size= 1,stride= 1,padding= 0),
            nn.BatchNorm3d(outchannel),
        )
        self.right = nn.Sequential()
        if inchannel != outchannel:
            self.right = nn.Sequential(
                nn.Conv3d(inchannel, outchannel, 1, 2, 0),
                nn.BatchNorm3d(outchannel)
             )

    def forward(self, x):
        x_l = self.left(x)
        x_r = self.right(x)
        x = x_l+x_r
        x = F.relu(x)
        return x


class block2(nn.Module):
    def __init__(self, inchannel, outchannel, stride, outpadding=0):
        super(block2, self).__init__()
        self.left = nn.Sequential(
            nn.ConvTranspose3d(inchannel, outchannel, 1, stride, 0, output_padding=outpadding),
            nn.BatchNorm3d(outchannel),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(outchannel, outchannel, 3, 1, 1),
            nn.BatchNorm3d(outchannel),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(outchannel, outchannel, 1, 1, 0),
            nn.BatchNorm3d(outchannel),
        )
        self.right = nn.Sequential()
        if inchannel != outchannel:
            self.right = nn.Sequential(
                nn.ConvTranspose3d(inchannel, outchannel, 1, stride, 0, output_padding=outpadding),
                nn.BatchNorm3d(outchannel)
            )

    def forward(self, x):
        x_l = self.left(x)
        x_r = self.right(x)
        x = x_l + x_r
        x = F.relu(x)
        return x

class neta(nn.Module):
    def __init__(self):
        super(neta, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2), (1, 2, 2), 0)
        )
        self.conv2_1 = block1(32, 16, 64, 2)
        self.conv2_2 = block1(64, 16, 64, 1)
        self.conv3_1 = block1(64, 32, 128, 2)
        self.conv3_2 = block1(128, 32, 128, 1)
        self.conv4 = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

class netm(nn.Module):
    def __init__(self):
        super(netm, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(6, 3, 3), stride=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2), (2, 2, 2), 0)
        )
        self.conv2_1=block1(32, 16, 64, 2)
        self.conv2_2=block1(64, 16, 64, 1)
        self.conv3_1=block1(64, 32, 128, 2)
        self.conv3_2=block1(128, 32, 128,1)
        self.conv4 = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0),
        )

    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1, 3, 2), 'constant', 0)
        x = self.conv1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

class netfa(nn.Module):
    def __init__(self):
        super(netfa, self).__init__()
        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )
        self.conv3 = block2(128, 64, 2)
        self.conv2 = block2(64, 32, 2, 1)
        self.conv1 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=32, out_channels=1, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), output_padding=(0, 1, 1))
        )
    def forward(self, x):
        x = self.conv5(x)
        x = self.conv4(x)
        x = self.conv3(x)
        x = self.conv2(x)
        x = self.conv1(x)
        return x

class netfm(nn.Module):
    def __init__(self):
        super(netfm, self).__init__()
        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )
        self.conv3 = block2(128, 64, 2)
        self.conv2 = block2(64, 32, 2, 1)
        self.conv1 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=32, out_channels=1, kernel_size=(6, 3, 3), stride=2, padding=(2, 1, 1), output_padding=(0, 1, 1))
        )
    def forward(self,x):
        x=self.conv5(x)
        x=self.conv4(x)
        x=self.conv3(x)
        x=self.conv2(x)
        x=self.conv1(x)
        return x
        
def test():

    R = torch.tensor([1.283447])
    R = R.cuda()
    r = R.item()
    neta.cuda()
    netm.cuda()
    neta.load_state_dict(t.load('E:/MODEL_BACKUP/neta.pth'))
    netm.load_state_dict(t.load('E:/MODEL_BACKUP/netm.pth'))

    data_set = dataTest()
    data_loader = DataLoader(data_set, batch_size=48,shuffle=False)

    for i,data in enumerate(data_loader):

        adata,mdata,gdata = data
        adata = (adata)
        mdata = Variable(mdata)
        gdata = Variable(gdata)
        adata = adata/255
        adata = adata.type(torch.FloatTensor)
        mdata = mdata.type(torch.FloatTensor)
        adata = adata.cuda()
        mdata = mdata.cuda()

        y_a = (neta(adata)).cuda()
        y_m = (netm(mdata)).cuda()

        y = (0.3*y_a + 0.7*y_m).cuda()
#         print(y.size())
        y_l = (y*y).cuda()
        
        for k in range(48):      
            d = t.sum(y_l,dim = 1)
            d = d.squeeze()
            distance = t.sum(gdata[k])/(80*10*20*20)
            fold = int(i/36)+1
            frame = int((i%36)/3)+1
            block = ((i%36)%3)*48+k 
            mark(fold,frame,block,distance,d,r)   
        
        
                
