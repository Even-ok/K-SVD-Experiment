"""
"""

import numpy as np
from scipy import linalg
import pickle
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import Deep_KSVD

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
# Overcomplete Discrete Cosinus Transform:
patch_size = 8
m = 16
Dict_init = Deep_KSVD.init_dct(patch_size, m)
Dict_init = Dict_init.to(device)

# Squared Spectral norm:
c_init = linalg.norm(Dict_init, ord=2) ** 2
c_init = torch.FloatTensor((c_init,))
c_init = c_init.to(device)

# Average weight:
w_init = torch.normal(mean=1, std=1 / 10 * torch.ones(patch_size ** 2)).float()
w_init = w_init.to(device)

# Deep-KSVD:
D_in, H_1, H_2, H_3, D_out_lam, T, min_v, max_v = patch_size ** 2, 128, 64, 32, 1, 7, -1, 1
model = Deep_KSVD.DenoisingNet_MLP(
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
    w_init,
    device,
)

model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.to(device)

# Test image names:
file_test = open("test_gray.txt", "r")
onlyfiles_test = []
for e in file_test:
    onlyfiles_test.append(e[:-1])

# Rescaling in [-1, 1]:
mean = 255 / 2
std = 255 / 2
data_transform = transforms.Compose(
    [Deep_KSVD.Normalize(mean=mean, std=std), Deep_KSVD.ToTensor()]
)
# Noise level:
sigma = 25

# Test Dataset:
my_Data_test = Deep_KSVD.FullImagesDataset(
    root_dir="gray", image_names=onlyfiles_test, sigma=sigma, transform=data_transform
)

dataloader_test = DataLoader(my_Data_test, batch_size=1, shuffle=False, num_workers=0)

# List PSNR:
file_to_print = open("list_test_PSNR.csv", "w")
file_to_print.write(str(device) + "\n")
file_to_print.flush()

#psnr是“Peak Signal to Noise Ratio”的缩写，即峰值信噪比，是一种评价图像的客观标准，
# 它具有局限性，一般是用于最大值信号和背景噪音之间的一个工程项目。
#它是原图像与被处理图像之间的均方误差相对于(2^n-1)^2的对数值(信号最大值的平方，n是每个采样值的比特数)，它的单位是dB。 MATLAB用法的公式如下：
#PSNR=10*log10((2^n-1)^2/MSE)
with open("list_test_PSNR.txt", "wb") as fp:
    with torch.no_grad():
        list_PSNR = []
        list_PSNR_init = []
        PSNR = 0
        for k, (image_true, image_noise) in enumerate(dataloader_test, 0):

            image_true_t = image_true[0, 0, :, :]
            image_true_t = image_true_t.to(device)

            image_noise_0 = image_noise[0, 0, :, :]
            image_noise_0 = image_noise_0.to(device)

            image_noise_t = image_noise.to(device)
            image_restored_t = model(image_noise_t)
            image_restored_t = image_restored_t[0, 0, :, :]

            PSNR_init = 10 * torch.log10(
                4 / torch.mean((image_true_t - image_noise_0) ** 2)
            )
            file_to_print.write("Init:" + " " + str(PSNR_init) + "\n")
            file_to_print.flush()

            list_PSNR_init.append(PSNR_init)

            PSNR = 10 * torch.log10(
                4 / torch.mean((image_true_t - image_restored_t) ** 2)
            )
            PSNR = PSNR.cpu()
            file_to_print.write("Test:" + " " + str(PSNR) + "\n")
            file_to_print.flush()

            list_PSNR.append(PSNR)

            # imsave("im_noisy_"+str(q)+'.pdf',image_noise_0)
            # imsave("im_restored_"+str(q)+'.pdf',image_restored_t)

    mean = np.mean(list_PSNR)
    file_to_print.write("FINAL" + " " + str(mean) + "\n")
    file_to_print.flush()
    pickle.dump(list_PSNR, fp)
