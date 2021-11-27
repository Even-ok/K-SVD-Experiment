"""
Implementation of the Deep K-SVD Denoising model, presented in
Deep K-SVD Denoising
M Scetbon, M Elad, P Milanfar
"""

import os
from skimage import io
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List
from cbam import *
import gc

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imagesc as imagesc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.nn.functional as F


def order_F_to_C(n):
    idx = np.arange(0, n ** 2)
    idx = idx.reshape(n, n, order="F")
    idx = idx.reshape(n ** 2, order="C")
    idx = list(idx)
    return idx


def init_dct(n, m):  #8*16
    """ Compute the Overcomplete Discrete Cosinus Transform. """
    oc_dictionary = np.zeros((n, m))
    for k in range(m):
        V = np.cos(np.arange(0, n) * k * np.pi / m)  #长度为8的向量
        if k > 0:
            V = V - np.mean(V)
        oc_dictionary[:, k] = V / np.linalg.norm(V)   #更新一列
    oc_dictionary = np.kron(oc_dictionary, oc_dictionary)   #行乘行大小，列乘列大小，克罗内克乘积
    oc_dictionary = oc_dictionary.dot(np.diag(1 / np.sqrt(np.sum(oc_dictionary ** 2, axis=0))))  #点乘。一行的每个数平方，相加，再开方，取倒数（特征值的算法吗）
    idx = np.arange(0, n ** 2)    #步长为1
    idx = idx.reshape(n, n, order="F")  #8*8   竖着排列
    idx = idx.reshape(n ** 2, order="C")  #横着排列
    oc_dictionary = oc_dictionary[idx, :]
    oc_dictionary = torch.from_numpy(oc_dictionary).float()
    return oc_dictionary  #[64,256]


class SubImagesDataset(Dataset):   #用子图来训练
    def __init__(self, root_dir, image_names, sub_image_size, sigma, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            image_names (list): List of the images names.
            sub_image_size (integer): Width of the square sub image.
            sigma (float): Level of the noise.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.sub_image_size = sub_image_size
        self.transform = transform
        self.root_dir = root_dir
        self.sigma = sigma
        self.image_names = image_names

        self.dataset_list = [io.imread(os.path.join(self.root_dir, name)) for name in self.image_names]

        w, h = np.shape(self.dataset_list[0])
        self.number_sub_images = int(
            (w - sub_image_size + 1) * (h - sub_image_size + 1)  #子图个数
        ) 

        self.number_images = len(self.image_names)

    @staticmethod
    def extract_sub_image_from_image(image, sub_image_size, idx_sub_image):   #得到一个128*128的子图
        w, h = np.shape(image)
        w_idx, h_idx = np.unravel_index(idx_sub_image, (int(w - sub_image_size + 1), int(h - sub_image_size + 1)))
        sub_image = image[w_idx: w_idx + sub_image_size, h_idx: h_idx + sub_image_size]
        sub_image = sub_image.reshape(1, sub_image_size, sub_image_size)
        return sub_image

    def __len__(self):
        return self.number_images * self.number_sub_images

    def __getitem__(self, idx):
        idx_im, idx_sub_image = np.unravel_index(idx, (self.number_images, self.number_sub_images))

        image = self.dataset_list[idx_im]
        sub_image = self.extract_sub_image_from_image(image, self.sub_image_size, idx_sub_image)

        np.random.seed(idx)
        noise = np.random.randn(self.sub_image_size, self.sub_image_size)  #生成具有正态分布的，128*128大小的噪音

        sub_image_noise = sub_image + self.sigma * noise

        if self.transform:
            sub_image = self.transform(sub_image)
            sub_image_noise = self.transform(sub_image_noise)     #归一化

        return sub_image.float(), sub_image_noise.float()  #转换为float形式输出


class FullImagesDataset(Dataset):   #获取完整图像，用来进行测试
    def __init__(self, root_dir: str, image_names: List[str], sigma: float, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            image_names (list): List of the name of the images.
            sigma (float): Level of the noise.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.transform = transform
        self.root_dir = root_dir
        self.sigma = sigma
        self.image_names = image_names

        self.dataset_list = [io.imread(os.path.join(self.root_dir, name)) for name in self.image_names]
        self.dataset_list_noise = [self._add_noise_to_image(np_im, k + 1e7) for (k, np_im) in
                                   enumerate(self.dataset_list)]

        self.number_images = len(self.image_names)

    def _add_noise_to_image(self, np_image, seed):
        w, h = np.shape(np_image)
        np.random.seed(int(seed))
        noise = np.random.randn(w, h)  #高斯分布噪声
        np_im_noise = np_image + self.sigma * noise
        return np_im_noise

    def __len__(self):
        return self.number_images

    def __getitem__(self, idx):
        image = self.dataset_list[idx]
        w, h = np.shape(image)
        image = image.reshape(1, w, h)
        image_noise = self.dataset_list_noise[idx]
        image_noise = image_noise.reshape(1, w, h)

        if self.transform:
            image = self.transform(image)
            image_noise = self.transform(image_noise)
        return image.float(), image_noise.float()


class ToTensor(object):
    """ Convert ndarrays to Tensors. """

    def __call__(self, image):
        return torch.from_numpy(image)


class Normalize(object):
    """ Normalize the images. """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        return (image - self.mean) / self.std


class DenoisingNet_MLP(torch.nn.Module):
    def __init__(
            self,
            patch_size, #8
            D_in, #64
            H_1,  #128
            H_2,  #64
            H_3,  #32
            D_out_lam,  #1
            T,
            min_v,
            max_v,
            Dict_init,
            c_init,
            w_init,
            device,
    ):
        super(DenoisingNet_MLP, self).__init__()
        self.patch_size = patch_size

        self.T = T
        self.min_v = min_v
        self.max_v = max_v

        q, l = Dict_init.shape
        soft_comp = torch.zeros(l).to(device)  #16
        Identity = torch.eye(l).to(device)   #16

        self.soft_comp = soft_comp
        self.Identity = Identity
        self.device = device

        self.Dict = torch.nn.Parameter(Dict_init)      #self.Dict, self.c和w都是可以被训练的权重
        self.c = torch.nn.Parameter(c_init)
        self.unfold = torch.nn.Unfold(kernel_size=(self.patch_size, self.patch_size))
        #卷积核大小是8*8   做展平操作，降了1维

        self.linear1 = torch.nn.Linear(D_in, H_1, bias=True) # 用于设置网络中的全连接层，需要注意的是全连接层的输入与输出都是二维张量，一般形状为[batch_size, size]
        self.linear2 = torch.nn.Linear(H_1, H_2, bias=True)
        self.linear3 = torch.nn.Linear(H_2, H_3, bias=True)
        self.linear4 = torch.nn.Linear(H_3, D_out_lam, bias=True)
        self.linear5 = torch.nn.Linear(D_out_lam, D_out_lam, bias=True) #加了一个z到lam的转变
        self.relu1 = torch.nn.ReLU(inplace=True)


        self.w = torch.nn.Parameter(w_init)
        self.attention4 = CBAM(D_out_lam, 16)   #要的应该是这个
        # self.attention4 = CBAM(256, 16)

    def soft_thresh(self, x, l):   #软阈值函数
        return torch.sign(x) * torch.max(torch.abs(x) - l, self.soft_comp)   #l是一维的话，可以通过广播。l是256维的话，可以直接减

    

    def change_shape(self, x,w,h):
        batch_size,w_2,channel = x.shape
        x_1 = x.transpose(1,2)
        x_2 = x_1.reshape((batch_size,channel,w,h))  #[1,128,121,121]
        return x_2

    def restore_shape(self, x):
        batch_size,channel,w,h = x.shape  #[1,128,121,121]
        x_1 = x.reshape((batch_size,channel,w*h)) #[1,128,14641]
        x_2 = x_1.transpose(1,2)
        return x_2

    def forward(self, x):   #放进来的大小是128*128  1,1,128,128
        N, C, w, h = x.shape   #一开始获取一个批量图像的形状 ——batch,通道，高宽，通过滑块，将图像分成小块


        #batch，通道，宽，高
        trans_w = w - self.patch_size+1
        trans_h = h - self.patch_size+1

        

        unfold = self.unfold(x)   #用8*8的核将其平展,[1, 64, 14641]  (只有单通道，所以一直向右延展) 64个小长条，做14641次
        N, d, number_patches = unfold.shape   
        #print("number_patches=",number_patches)
        unfold = unfold.transpose(1, 2)  #将这些小块形状拉平成向量，方便神经网络输入，  改变维度，[1, 14641, 64]

        
        gc.collect()
        torch.cuda.empty_cache()

        lin = self.linear1(unfold).clamp(min=0)   #将输入input张量每个元素的范围限制到区间 [min,max]，返回结果到一个新张量 [1, 14641, 128]
        lin = self.linear2(lin).clamp(min=0) #[1, 14641, 64]
        lin = self.linear3(lin).clamp(min=0) #[1, 14641, 32] 
        lam = self.linear4(lin)   #[1, 14641, 1]
        '''
        ###################################################################画图代码主要在下方#########################################################
        
        '''
        '''
        画出attention前的mesh图
        '''
        lam = self.change_shape(lam,trans_w,trans_h)  #[1,256,121,121]

        fig_M1 = imagesc.plot(lam[0,0,:,:].cpu().detach().numpy())
        imagesc.savefig(fig_M1, './pre.png')
        x = np.arange(0,trans_w,1)
        y = np.arange(0,trans_h,1)
        
        xx, yy = np.meshgrid(x, y)  # 转换成二维的矩阵坐标

        fig = plt.figure(1, figsize=(12, 8)) 
        zz =  lam[0, 0, :, :].cpu().detach().numpy()
        ax = fig.add_subplot(2, 2, 1, projection='3d')
        ax.set_top_view()
        
        ax.plot_surface(xx, yy, np.transpose(zz), rstride=1, cstride=1, cmap='rainbow')
        plt.show()

        '''
        画出attention后的mesh图
        '''
        lam = self.attention4(lam)  #[1,1,121,121]

        '''
        画出某一维度的二维热力图
        '''
        fig_M1 = imagesc.plot(lam[0,0,:,:].cpu().detach().numpy())
        imagesc.savefig(fig_M1, './post.png')
        fig = plt.figure(1, figsize=(12, 8))

        zz =  lam[0, 0, :, :].cpu().detach().numpy()
        ax = fig.add_subplot(2, 2, 2, projection='3d')
        ax.set_top_view()
        
        ax.plot_surface(xx, yy, np.transpose(zz), rstride=1, cstride=1, cmap='rainbow')
        plt.show()

        lam = self.restore_shape(lam)  #[1,14641,1]

        #把attention加在lam上，找到相似性和结构性
        l = lam / self.c   #c_init = (linalg.norm(Dict_init.cpu(), ord=2)) ** 2  求特征值，A的转置共轭矩阵与矩阵A的积的最大特征根的平方根值（又给乘方了） torch.Size([1, 14641, 256]
        y = torch.matmul(unfold, self.Dict)  #字典大小：[64, 256]  y:[1, 14641, 256] 
        S = self.Identity - (1 / self.c) * self.Dict.t().mm(self.Dict)   #[256, 256]  
        S = S.t()  #[256, 256]

        '''
        联合lambda和z一起训练,adaptive
        '''
        # z = self.soft_thresh(y, l)  #[1, 14641, 256](软阈值函数)  稀疏系数,原子相关性

        # for t in range(self.T):   #T=5，做5次        

        #     z_lam = self.linear5(z)  #将z转变为lam
        #     #加个relu
        #     z_lam = self.relu1(z_lam)
        #     new_lam = lam+z_lam    #lam为定值，因为放进来的图片是固定的
        #     l = new_lam / self.c  
        
        #     z = self.soft_thresh(torch.matmul(z, S) + (1 / self.c) * y, l)  #z变了，l也变了

        '''
        原始的z
        '''

        z = self.soft_thresh(y, l) #y:[1, 14641, 256]  l:[1, 14641, 256]
        for t in range(self.T):
            z = self.soft_thresh(torch.matmul(z, S) + (1 / self.c) * y, l)  #缺陷，在于这个l是不变的


        '''
        重建模块
        '''
        x_pred = torch.matmul(z, self.Dict.t())   #与字典相乘，做计算    软阈值了之后再乘
        x_pred = torch.clamp(x_pred, min=self.min_v, max=self.max_v)  #将输入input张量每个元素的夹紧到区间 [min,max][-1,1]，并返回结果到一个新张量。
        x_pred = self.w * x_pred
        x_pred = x_pred.transpose(1, 2)

        normalize = torch.ones(N, number_patches, d)
        normalize = normalize.to(self.device)
        normalize = self.w * normalize  #乘以权重
        normalize = normalize.transpose(1, 2)

        fold = torch.nn.functional.fold(     #返回128*128的sub_image_size大小
            x_pred, output_size=(w, h), kernel_size=(self.patch_size, self.patch_size)
        )

        norm = torch.nn.functional.fold(
            normalize,
            output_size=(w, h),
            kernel_size=(self.patch_size, self.patch_size),
        )

        res = fold / norm   #进行反标准化输出，得到与原始图像类似的矩阵

        return res


class DenoisingNet_MLP_2(torch.nn.Module):
    def __init__(
            self,
            patch_size,
            D_in,
            H_1,
            H_2,
            H_3,
            D_out_lam,
            T,
            min_v,
            max_v,
            Dict_init,
            c_init,
            w_1_init,
            w_2_init,
            device,
    ):

        super(DenoisingNet_MLP_2, self).__init__()
        self.patch_size = patch_size

        self.T = T
        self.min_v = min_v
        self.max_v = max_v

        q, l = Dict_init.shape
        soft_comp = torch.zeros(l).to(device)
        Identity = torch.eye(l).to(device)
        self.soft_comp = soft_comp
        self.Identity = Identity
        self.device = device

        self.Dict = torch.nn.Parameter(Dict_init)
        self.c = torch.nn.Parameter(c_init)
        self.unfold = torch.nn.Unfold(kernel_size=(self.patch_size, self.patch_size))

        #### First Stage ####
        self.linear1 = torch.nn.Linear(D_in, H_1, bias=True)
        self.linear2 = torch.nn.Linear(H_1, H_2, bias=True)
        self.linear3 = torch.nn.Linear(H_2, H_3, bias=True)
        self.linear4 = torch.nn.Linear(H_3, D_out_lam, bias=True)

        self.w_1 = torch.nn.Parameter(w_1_init)
        ######################

        #### Second Stage ####
        self.linear1_2 = torch.nn.Linear(D_in, H_1, bias=True)
        self.linear2_2 = torch.nn.Linear(H_1, H_2, bias=True)
        self.linear3_2 = torch.nn.Linear(H_2, H_3, bias=True)
        self.linear4_2 = torch.nn.Linear(H_3, D_out_lam, bias=True)

        self.w_2 = torch.nn.Parameter(w_2_init)
        ######################

    def soft_thresh(self, x, l):
        return torch.sign(x) * torch.max(torch.abs(x) - l, self.soft_comp)

    def forward(self, x):

        N, C, w, h = x.shape

        unfold = self.unfold(x)
        N, d, number_patches = unfold.shape

        unfold = unfold.transpose(1, 2)

        lin = self.linear1(unfold).clamp(min=0)
        lin = self.linear2(lin).clamp(min=0)
        lin = self.linear3(lin).clamp(min=0)
        lam = self.linear4(lin)

        l = lam / self.c
        y = torch.matmul(unfold, self.Dict)
        S = self.Identity - (1 / self.c) * self.Dict.t().mm(self.Dict)
        S = S.t()

        z = self.soft_thresh(y, l)
        for t in range(self.T):
            z = self.soft_thresh(torch.matmul(z, S) + (1 / self.c) * y, l)

        x_pred = torch.matmul(z, self.Dict.t())
        x_pred = torch.clamp(x_pred, min=self.min_v, max=self.max_v)
        x_pred = self.w_1 * x_pred
        x_pred = x_pred.transpose(1, 2)

        normalize = torch.ones(N, number_patches, d)
        normalize = normalize.to(self.device)
        normalize = self.w_1 * normalize
        normalize = normalize.transpose(1, 2)

        fold = torch.nn.functional.fold(
            x_pred, output_size=(w, h), kernel_size=(self.patch_size, self.patch_size)
        )

        norm = torch.nn.functional.fold(
            normalize,
            output_size=(w, h),
            kernel_size=(self.patch_size, self.patch_size),
        )

        res = fold / norm

        ### Second Stage ###
        unfold = self.unfold(res)
        unfold = unfold.transpose(1, 2)

        lin = self.linear1_2(unfold).clamp(min=0)
        lin = self.linear2_2(lin).clamp(min=0)
        lin = self.linear3_2(lin).clamp(min=0)
        lam = self.linear4_2(lin)

        l = lam / self.c
        y = torch.matmul(unfold, self.Dict)

        z = self.soft_thresh(y, l)
        for t in range(self.T):
            z = self.soft_thresh(torch.matmul(z, S) + (1 / self.c) * y, l)

        x_pred = torch.matmul(z, self.Dict.t())
        x_pred = torch.clamp(x_pred, min=self.min_v, max=self.max_v)
        x_pred = self.w_2 * x_pred
        x_pred = x_pred.transpose(1, 2)

        normalize = torch.ones(N, number_patches, d)
        normalize = normalize.to(self.device)
        normalize = self.w_2 * normalize
        normalize = normalize.transpose(1, 2)

        fold = torch.nn.functional.fold(
            x_pred, output_size=(w, h), kernel_size=(self.patch_size, self.patch_size)
        )

        norm = torch.nn.functional.fold(
            normalize,
            output_size=(w, h),
            kernel_size=(self.patch_size, self.patch_size),
        )

        res = fold / norm

        return res


class DenoisingNet_MLP_3(torch.nn.Module):
    def __init__(
            self,
            patch_size,
            D_in,
            H_1,
            H_2,
            H_3,
            D_out_lam,
            T,
            min_v,
            max_v,
            Dict_init,
            c,
            w_1_init,
            w_2_init,
            w_3_init,
            device,
    ):

        super(DenoisingNet_MLP_3, self).__init__()
        self.patch_size = patch_size

        self.T = T
        self.min_v = min_v
        self.max_v = max_v

        q, l = Dict_init.shape
        soft_comp = torch.zeros(l).to(device)
        Identity = torch.eye(l).to(device)
        self.soft_comp = soft_comp
        self.Identity = Identity
        self.device = device

        self.Dict = torch.nn.Parameter(Dict_init)
        self.c = torch.nn.Parameter(c)
        self.unfold = torch.nn.Unfold(kernel_size=(self.patch_size, self.patch_size))

        #### First Stage ####
        self.linear1 = torch.nn.Linear(D_in, H_1, bias=True)
        self.linear2 = torch.nn.Linear(H_1, H_2, bias=True)
        self.linear3 = torch.nn.Linear(H_2, H_3, bias=True)
        self.linear4 = torch.nn.Linear(H_3, D_out_lam, bias=True)

        self.w_1 = torch.nn.Parameter(w_1_init)
        #####################

        #### Second Stage ####
        self.linear1_2 = torch.nn.Linear(D_in, H_1, bias=True)
        self.linear2_2 = torch.nn.Linear(H_1, H_2, bias=True)
        self.linear3_2 = torch.nn.Linear(H_2, H_3, bias=True)
        self.linear4_2 = torch.nn.Linear(H_3, D_out_lam, bias=True)

        self.w_2 = torch.nn.Parameter(w_2_init)
        ######################

        #### Third Stage ####
        self.linear1_3 = torch.nn.Linear(D_in, H_1, bias=True)
        self.linear2_3 = torch.nn.Linear(H_1, H_2, bias=True)
        self.linear3_3 = torch.nn.Linear(H_2, H_3, bias=True)
        self.linear4_3 = torch.nn.Linear(H_3, D_out_lam, bias=True)

        self.w_3 = torch.nn.Parameter(w_3_init)
        ######################

    def soft_thresh(self, x, l):
        return torch.sign(x) * torch.max(torch.abs(x) - l, self.soft_comp)

    def forward(self, x):

        N, C, w, h = x.shape

        unfold = self.unfold(x)
        N, d, number_patches = unfold.shape

        unfold = unfold.transpose(1, 2)

        lin = self.linear1(unfold).clamp(min=0)
        lin = self.linear2(lin).clamp(min=0)
        lin = self.linear3(lin).clamp(min=0)
        lam = self.linear4(lin)

        l = lam / self.c
        y = torch.matmul(unfold, self.Dict)
        S = self.Identity - (1 / self.c) * self.Dict.t().mm(self.Dict)
        S = S.t()

        z = self.soft_thresh(y, l)
        for t in range(self.T):
            z = self.soft_thresh(torch.matmul(z, S) + (1 / self.c) * y, l)

        x_pred = torch.matmul(z, self.Dict.t())
        x_pred = torch.clamp(x_pred, min=self.min_v, max=self.max_v)
        x_pred = self.w * x_pred
        x_pred = x_pred.transpose(1, 2)

        normalize = torch.ones(N, number_patches, d)
        normalize = normalize.to(self.device)
        normalize = self.w_1 * normalize
        normalize = normalize.transpose(1, 2)

        fold = torch.nn.functional.fold(
            x_pred, output_size=(w, h), kernel_size=(self.patch_size, self.patch_size)
        )

        norm = torch.nn.functional.fold(
            normalize,
            output_size=(w, h),
            kernel_size=(self.patch_size, self.patch_size),
        )

        res = fold / norm

        ### Second Stage ###
        unfold = self.unfold(res)
        unfold = unfold.transpose(1, 2)

        lin = self.linear1_2(unfold).clamp(min=0)
        lin = self.linear2_2(lin).clamp(min=0)
        lin = self.linear3_2(lin).clamp(min=0)
        lam = self.linear4_2(lin)

        l = lam / self.c
        y = torch.matmul(unfold, self.Dict)

        z = self.soft_thresh(y, l)
        for t in range(self.T):
            z = self.soft_thresh(torch.matmul(z, S) + (1 / self.c) * y, l)

        x_pred = torch.matmul(z, self.Dict.t())
        x_pred = torch.clamp(x_pred, min=self.min_v, max=self.max_v)
        x_pred = self.w_2 * x_pred
        x_pred = x_pred.transpose(1, 2)

        normalize = torch.ones(N, number_patches, d)
        normalize = normalize.to(self.device)
        normalize = self.w_2 * normalize
        normalize = normalize.transpose(1, 2)

        fold = torch.nn.functional.fold(
            x_pred, output_size=(w, h), kernel_size=(self.patch_size, self.patch_size)
        )

        norm = torch.nn.functional.fold(
            normalize,
            output_size=(w, h),
            kernel_size=(self.patch_size, self.patch_size),
        )

        res = fold / norm

        ### Third Stage ###
        unfold = self.unfold(res)
        unfold = unfold.transpose(1, 2)

        lin = self.linear1_3(unfold).clamp(min=0)
        lin = self.linear2_3(lin).clamp(min=0)
        lin = self.linear3_3(lin).clamp(min=0)
        lam = self.linear4_3(lin)

        l = lam / self.c
        y = torch.matmul(unfold, self.Dict)

        z = self.soft_thresh(y, l)
        for t in range(self.T):
            z = self.soft_thresh(torch.matmul(z, S) + (1 / self.c) * y, l)

        x_pred = torch.matmul(z, self.Dict.t())
        x_pred = torch.clamp(x_pred, min=self.min_v, max=self.max_v)
        x_pred = self.w_3 * x_pred
        x_pred = x_pred.transpose(1, 2)

        normalize = torch.ones(N, number_patches, d)
        normalize = normalize.to(self.device)
        normalize = self.w_3 * normalize
        normalize = normalize.transpose(1, 2)

        fold = torch.nn.functional.fold(
            x_pred, output_size=(w, h), kernel_size=(self.patch_size, self.patch_size)
        )

        norm = torch.nn.functional.fold(
            normalize,
            output_size=(w, h),
            kernel_size=(self.patch_size, self.patch_size),
        )

        res = fold / norm

        return res
