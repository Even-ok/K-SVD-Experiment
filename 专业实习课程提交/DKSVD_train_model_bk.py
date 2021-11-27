"""
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import time
import Deep_KSVD
from scipy import linalg
from skimage.transform import resize

# List of the test image names BSD68:
file_test = open("test_gray.txt", "r")  #放入图片的名字
onlyfiles_test = []
for e in file_test:
    onlyfiles_test.append(e[:-1])

# List of the train image names:
file_train = open("train_gray.txt", "r")  
onlyfiles_train = []
for e in file_train:
    onlyfiles_train.append(e[:-1])

# Rescaling in [-1, 1]:  图片归一化
mean = 255 / 2
std = 255 / 2
data_transform = transforms.Compose(
    [Deep_KSVD.Normalize(mean=mean, std=std), Deep_KSVD.ToTensor()]   #图片归一化，为了让某些激活函数的梯度不致于过小，加快收敛。
)
# Noise level:  #噪声等级
sigma = 25
# Sub Image Size: sub_image是灰度值的相减,size是相减后图像边长
sub_image_size = 128  #一个patch的边长
# Training Dataset:
my_Data_train = Deep_KSVD.SubImagesDataset(   #输入的是灰度图
    root_dir="gray",
    image_names=onlyfiles_train,
    sub_image_size=sub_image_size,
    sigma=sigma,
    transform=data_transform,
)
# Test Dataset:
my_Data_test = Deep_KSVD.FullImagesDataset(
    root_dir="gray", image_names=onlyfiles_test, sigma=sigma, transform=data_transform
)

# Dataloader of the test set:
num_images_test = 5
indices_test = np.random.randint(0, 68, num_images_test).tolist()
my_Data_test_sub = torch.utils.data.Subset(my_Data_test, indices_test)
dataloader_test = DataLoader(
    my_Data_test_sub, batch_size=1, shuffle=False, num_workers=0
)

# Dataloader of the training set:
#Batch_size=1，也就是每次只训练一个样本。这就是在线学习(Online Learning)。
# 理论上说batch_size=1是最好的，不过实际上调的时候，会出现batch_size太小导致网络收敛不稳定，最后结果比较差。
batch_size = 1
dataloader_train = DataLoader(
    my_Data_train, batch_size=batch_size, shuffle=True, num_workers=0
)
#print(enumerate(dataloader_train, 0))   #29668032


# Create a file to see the output during the training:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
file_to_print = open("results_training.csv", "w")
file_to_print.write(str(device) + "\n")
print(str(device) + "\n")
#刷新缓冲区
file_to_print.flush()

# Initialization:
patch_size = 8   
#在CNN学习训练过程中，不是一次来处理一整张图片，
# 而是先将图片划分为多个小的块，内核 kernel (或过滤器或特征检测器)
# 每次只查看图像的一个块，这一个小块就称为 patch
# 然后过滤器移动到图像的另一个patch，以此类推。
m = 16
Dict_init = Deep_KSVD.init_dct(patch_size, m)
Dict_init = Dict_init.to(device)  #[64, 256]

c_init = (linalg.norm(Dict_init.cpu(), ord=2)) ** 2  #取字典的二范式的平方值
c_init = torch.FloatTensor((c_init,))
c_init = c_init.to(device)  #c就是一个值

w_init = torch.normal(mean=1, std=1 / 10 * torch.ones(patch_size ** 2)).float()  #[64]
w_init = w_init.to(device)

D_in, H_1, H_2, H_3, D_out_lam, T, min_v, max_v = patch_size ** 2, 128, 64, 32, 1, 5, -1, 1
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
model.to(device)

# Construct our loss function and an Optimizer:
criterion = torch.nn.MSELoss(reduction="mean")
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

start = time.time()
epochs = 3
running_loss = 0.0
print_every = 1
train_losses, test_losses = [], []
# print(enumerate(dataloader_train, 0))
print(type(dataloader_train))

for epoch in range(epochs):  # loop over the dataset multiple times
    for i, (sub_images, sub_images_noise) in enumerate(dataloader_train, 0):   #29668032
        # get the inputs
        sub_images, sub_images_noise = (
            sub_images.to(device),
            sub_images_noise.to(device),
        )

        # zero the parameter gradients
        # 把梯度置零，也就是把loss关于weight的导数变成0.
        optimizer.zero_grad()

        # N,C,w,h = sub_images_noise.shape
        # if w!=h:  #在这里就更改测试图像大小
        #     w_1 = max(w,h)  #选最大的那个
        #     sub_images_noise = resize(patches_noise.cpu(),(N,C,w_1,w_1))
        #     sub_images_noise = torch.from_numpy(patches_noise).to(device)

        # forward + backward + optimize
        outputs = model(sub_images_noise)
        loss = criterion(outputs, sub_images)   #降噪图与原始子图的MSELoss
        loss.backward()  #后向传播
        optimizer.step()      #此时w,Dict,c都得到了更新

        # print statistics
        running_loss += loss.item()
        if i % print_every == print_every - 1:  # print every x mini-batches
            train_losses.append(running_loss / print_every)

            end = time.time()
            time_curr = end - start
            file_to_print.write("time:" + " " + str(time_curr) + "\n")
            print("time:" + " " + str(time_curr) + "\n")
            start = time.time()

            with torch.no_grad():
                test_loss = 0

                for patches_t, patches_noise_t in dataloader_test:  #len(dataloader_test)=5
                    patches, patches_noise = (
                        patches_t.to(device),
                        patches_noise_t.to(device),
                    )
                    N,C,w,h = patches_noise.shape
                    # if w!=h:  #在这里就更改测试图像大小
                    #     w_1 = max(w,h)  #选最大的那个
                    #     patches_noise = resize(patches_noise.cpu(),(N,C,w_1,w_1))
                    #     patches_noise = torch.from_numpy(patches_noise).to(device)
                    #     patches = resize(patches.cpu(),(N,C,w_1,w_1))
                    #     patches = torch.from_numpy(patches).to(device)

                    outputs = model(patches_noise)   #[1,1,321,481]
                    loss = criterion(outputs, patches)
                    test_loss += loss.item()

                test_loss = test_loss / len(dataloader_test)

            end = time.time()
            time_curr = end - start
            file_to_print.write("time:" + " " + str(time_curr) + "\n")
            print("time:" + " " + str(time_curr) + "\n")
            start = time.time()

            test_losses.append(test_loss)
            s = "[%d, %d] loss_train: %f, loss_test: %f" % (
                epoch + 1,
                (i + 1) * batch_size,
                running_loss / print_every,
                test_loss,
            )
            s = s + "\n"
            file_to_print.write(s)
            print(s)
            file_to_print.flush()
            running_loss = 0.0

        if i % (10 * print_every) == (10 * print_every) - 1:
            torch.save(model.state_dict(), "model.pth")
            np.savez(
                "losses.npz", train=np.array(test_losses), test=np.array(train_losses)
            )


file_to_print.write("Finished Training")
