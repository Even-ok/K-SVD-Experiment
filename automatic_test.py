'''
封装测试代码
'''

import torch
from skimage import io
import numpy as np
import Deep_KSVD
from scipy import linalg
import time
import os
import math
from skimage.metrics import structural_similarity

def normalize(img):
    mean=255/2
    std=255/2
    return (img - mean) / std

def inNormalize(img):
    mean=255/2
    std=255/2
    return img*std+mean

def get_all_result (ori_img,noise_img,model_url,denoise_path):
    '''
    读取原图
    '''
    img1 = io.imread(ori_img)
    img1_nor = normalize(img1)
    
    '''
    读取噪声图
    '''
    img1_noise1 = io.imread(noise_img)
    img1_noise1_nor = normalize(img1_noise1)

    '''
    使用模型降噪
    '''

    patch_size = 8
    m = 16
    device = torch.device("cpu")

    Dict_init = Deep_KSVD.init_dct(patch_size, m)
    Dict_init = Dict_init.to(device)

    # Squared Spectral norm:
    c_init = linalg.norm(Dict_init, ord=2) ** 2
    c_init = torch.FloatTensor((c_init,))
    c_init = c_init.to(device)

    # Average weight:
    w_1_init = torch.normal(mean=1, std=1 / 10 * torch.ones(patch_size ** 2)).float()
    w_1_init = w_1_init.to(device)

    w_2_init = torch.normal(mean=1, std=1 / 10 * torch.ones(patch_size ** 2)).float()
    w_2_init = w_2_init.to(device)

    w_3_init = torch.normal(mean=1, std=1 / 10 * torch.ones(patch_size ** 2)).float()
    w_3_init = w_3_init.to(device)

    # Deep-KSVD:
    D_in, H_1, H_2, H_3, D_out_lam, T, min_v, max_v = patch_size ** 2, 128, 64, 32, 1, 7, -1, 1
    # D_in, H_1, H_2, H_3, D_out_lam, T, min_v, max_v = patch_size ** 2, 512,1024,512,256, 5, -1, 1
    model = Deep_KSVD.DenoisingNet_MLP_3(
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
        w_3_init,
        device,
    )
    # model.add_module('attention4', CBAM(256, 16))
    # print(model)
    model.load_state_dict(torch.load(model_url,map_location='cuda:0'))
    model.to(device)

    '''
    进行去噪操作，统计时间
    '''
    w,h=img1_noise1_nor.shape

    with torch.no_grad():
        start_time=time.time()
        img2 = img1_noise1_nor.reshape(1,1,w,h)
        img2 = np.float32(img2)
        img_denoise1 = model(torch.from_numpy(img2))
        image_dnoise1 = img_denoise1[0, 0, :, :]
        real_denoise_1 = inNormalize(image_dnoise1).numpy()
        real_denoise = inNormalize(image_dnoise1).numpy().astype('uint8')
        end_time=time.time()   #结束时间
        dur_time = end_time-start_time
        print("time:%d"  % (dur_time))  #结束时间-开始时间
        #io.imshow(real_denoise)
        filename = os.path.split(noise_img)[1]
        name = os.path.splitext(filename)[0]
        ext = os.path.splitext(filename)[1]
        newFileName = name + '_denoise'+ext
        #io.imsave('D:\\1课程资料\\大三上\\专业实习\\模拟数据\\+++复原图像\\sigma50\\walkbridge_10.png',real_denoise)
        io.imsave(denoise_path+newFileName,real_denoise)

    '''
     计算PSNR
    '''
    #计算原图和噪声图的PSNR(PSNR_o)
    PSNR_init_1 = 20 * np.log10(255 / math.sqrt(np.mean((img1.astype(float) - img1_noise1.astype(float)) ** 2)))
    print('ori_PSNR:',PSNR_init_1)

    PSNR_init_2 = 20 * np.log10(255 / math.sqrt(np.mean((img1.astype(float) - real_denoise.astype(float)) ** 2)))  #这是转成可观测的图（也就是上面的降噪图的）
    print('denoise_PSNR:',PSNR_init_2)


    '''
    计算SSIM
    '''
    #用库的方法计算SSIM
    #ssim1 = structural_similarity(img1_nor, img1_noise1_nor, data_range=1.0)
    noise_ssim2 = structural_similarity(img1, img1_noise1, data_range=255)#一般取这个
    #print('SSIM1: ', ssim1)
    print('noise_SSIM: ', noise_ssim2)

    #降噪图
    denoise_ssim2 = structural_similarity(img1,real_denoise_1, data_range=255)
    print('denoise_SSIM: ', denoise_ssim2)

    '''
    计算MAE
    '''
    abs_diff_ori = img1 - img1_noise1
    abs_diff_ori = np.abs(abs_diff_ori)
    abs_diff_re = img1 - real_denoise
    abs_diff_re = np.abs(abs_diff_re)

    #噪声图的
    abs_sum = 0
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            abs_sum += np.abs(np.float32(img1[i][j])-np.float(img1_noise1[i][j]))
    noise_MAE = abs_sum/(img1.shape[0]*img1.shape[1])
    print('noise_MAE:',noise_MAE)

    #降噪图的
    abs_sum = 0
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            abs_sum += np.abs(np.float32(img1[i][j])-np.float(real_denoise_1[i][j]))
    denoise_MAE = abs_sum/(img1.shape[0]*img1.shape[1])
    print('denoise_MAE:',denoise_MAE)

    return PSNR_init_1,PSNR_init_2,noise_ssim2,denoise_ssim2,noise_MAE,denoise_MAE

if __name__=='__main__':
    # ori_file = 
    # noise_file = 
    # ori_img_name = 
    # noise_img_name = 
    # ori_img = ori_file+ori_img_name
    # noise_img = noise_file + noise_img_name
    ori_img = 'D:\\1课程资料\\大三上\\专业实习\\模拟数据\\原始图像\\light.png'
    noise_img = 'D:\\1课程资料\\大三上\\专业实习\\模拟数据\\噪声图像\\light_50.png'
    model_url = 'D:\\1课程资料\\大三上\\专业实习\\K-SVD-Experiment\\Model\\T=7K=3sigma50Model\\model.pth'
    denoise_path = 'D:\\1课程资料\\大三上\\专业实习\\模拟数据\\+++复原图像\\sigma50_1\\'
    get_all_result(ori_img,noise_img,model_url,denoise_path)