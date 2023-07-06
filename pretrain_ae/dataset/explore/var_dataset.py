import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from imageio import imread
import copy
filepath = '/media/yangliwei/yangliwei/yangliwei/Domain_Generalization/code/AdaptSegNet-master/data/GTA5/images/'  # 数据集目录
pathDir = os.listdir(filepath)


max_r = 0
max_g = 0
max_b = 0
max_r_corr_g = 0
max_r_corr_b = 0
max_g_corr_r = 0
max_g_corr_b = 0
max_b_corr_r = 0
max_b_corr_g = 0
r_channel =[]
id_R = []
id_G = []
id_B = []
for idx in np.arange(len(pathDir)):
    filename = pathDir[idx]
    img = imread(os.path.join(filepath, filename))
    image_shape = img.shape
    print(image_shape)
    for i in np.arange(image_shape[0]):
        for j in np.arange(image_shape[1]):
            if img[i,j,0] > max_r:
                max_r = img[i,j,0]
                max_r_corr_g = img[i,j,1]
                max_r_corr_b = img[i,j,2]
            if img[i,j,1] > max_g:
                max_g = img[i,j,1]
                max_g_corr_r = img[i,j,0]
                max_g_corr_b = img[i,j,2]
            if img[i,j,2] > max_b:
                max_b = img[i,j,2]
                max_b_corr_r = img[i,j,0]
                max_b_corr_g = img[i,j,1]
    if idx % 100 == 0:
        print(idx)
print("max r  (rgb)" ,max_r,max_r_corr_g,max_r_corr_b )
print("max g  (rgb)" ,max_g_corr_r,max_g,max_g_corr_b )
print("max b  (rgb)" ,max_b_corr_r,max_b_corr_g,max_b )
    # R_channel = R_channel + np.sum(img[:, :, 0])
    # G_channel = G_channel + np.sum(img[:, :, 1])
    # B_channel = B_channel + np.sum(img[:, :, 2])

# R_channel = 0
# G_channel = 0
# B_channel = 0
# for idx in np.arange(len(pathDir)):
#     filename = pathDir[idx]
#     img = imread(os.path.join(filepath, filename))
#     R_channel = R_channel + np.sum(img[:, :, 0])
#     G_channel = G_channel + np.sum(img[:, :, 1])
#     B_channel = B_channel + np.sum(img[:, :, 2])
#     if idx %100 ==0:
#         print(idx)
#
# num = len(pathDir) *1914 * 1052  # 这里（384,512）是每幅图片的大小，所有图片尺寸都一样
# R_mean = R_channel / num
# G_mean = G_channel / num
# B_mean = B_channel / num
#
# R_channel = 0
# G_channel = 0
# B_channel = 0
# max_r_var = 0
# max_g_var = 0
# max_b_var = 0
# max_r_corr_g = 0
# max_r_corr_b = 0
# max_g_corr_r = 0
# max_g_corr_b = 0
# max_b_corr_r = 0
# max_b_corr_g = 0
# max_idx_r = 0
# max_idx_g = 0
# max_idx_b = 0
# for idx in np.arange(len(pathDir)):
#     if idx % 100 == 0:
#         print(idx)
#     filename = pathDir[idx]
#     img = imread(os.path.join(filepath, filename))
#     r_var = np.sum((img[:, :, 0] - R_mean) ** 2)
#     g_var = np.sum((img[:, :, 1] - G_mean) ** 2)
#     b_var = np.sum((img[:, :, 2] - B_mean) ** 2)
#     if max_r_var < r_var:
#         max_r_var = copy.deepcopy(r_var)
#         max_r_corr_g = copy.deepcopy(g_var)
#         max_r_corr_b =copy.deepcopy( b_var)
#         max_idx_r = copy.deepcopy(idx)
#     if max_g_var < g_var:
#         max_g_var = copy.deepcopy(g_var)
#         max_g_corr_r = copy.deepcopy(r_var)
#         max_g_corr_b = copy.deepcopy(b_var)
#         max_idx_g = copy.deepcopy(idx)
#     if max_b_var <  b_var:
#         max_b_var = copy.deepcopy(b_var)
#         max_b_corr_r = copy.deepcopy(r_var)
#         max_b_corr_g = copy.deepcopy(g_var)
#         max_idx_b = copy.deepcopy(idx)
#     R_channel = R_channel + r_var
#     G_channel = G_channel + g_var
#     B_channel = B_channel + b_var
#
# R_var = R_channel / num
# G_var = G_channel / num
# B_var = B_channel / num
#
# print("R_mean is %f, G_mean is %f, B_mean is %f" % (R_mean, G_mean, B_mean))
# print("R_var is %f, G_var is %f, B_var is %f" % (R_var, G_var, B_var))
# print("max r var (rgb)" ,max_idx_r,max_r_var,max_r_corr_g,max_r_corr_b )
# print("max g var (rgb)" ,max_idx_g,max_g_corr_r,max_g_var,max_g_corr_b )
# print("max b var (rgb)" ,max_idx_b,max_b_corr_r,max_b_corr_g,max_b_var )
# print("R max , G max, B max")





#R_mean is 112.770339, G_mean is 111.680680, B_mean is 108.277730
#R_var is 4434.354744, G_var is 4235.189658, B_var is 4052.808410

#sum  12722.352812  average root 65.12130427645523

# R_mean is 112.770339, G_mean is 111.680680, B_mean is 108.277730
# R_var is 4434.354744, G_var is 4235.189658, B_var is 4052.808410
# max r var (rgb) 3426 0. 18960003946.699795 17183522740.23713
# max g var (rgb) 4024 18667310808.906654 19324135595.459194 14896735309.87932
# max b var (rgb) 719 14180878825.207827 15739340146.619009 18086205368.481102