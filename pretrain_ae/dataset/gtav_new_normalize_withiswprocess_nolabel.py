import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image
import copy
import torchvision.transforms as transforms
from skimage.util.dtype import img_as_float
import torchvision.transforms as standard_transforms
from skimage.filters import gaussian
import scipy.misc as m

import dataset.cityscapes_labels as cityscapes_labels

import time
from skimage.util import random_noise
trainid_to_trainid = cityscapes_labels.trainId2trainId
color_to_trainid = cityscapes_labels.color2trainId
ignore_label = 255

import transforms.joint_transforms as joint_transforms

# import cv2
#
# def contrast_demo(img1,c,b):
#     rows,clos,channel = img1.shape
#     blank = np.zeros([rows, clos, channel],img1.dtype)
#     dst = cv2.addWeighted(img1,c,blank,1-c,b)
#     return dst
# import cv2
#
# def contrast_demo(img1,c,b):
#     rows,clos,channel = img1.shape
#     blank = np.zeros([rows, clos, channel],img1.dtype)
#     dst = cv2.addWeighted(img1,c,blank,1-c,b)
#     return dst
class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()

class RandomGaussianBlur(object):
    """
    Apply Gaussian Blur
    """
    def __call__(self, img):
        sigma = 0.15 + random.random() * 1.15
        blurred_img = gaussian(np.array(img), sigma=sigma, multichannel=True)
        blurred_img *= 255
        return Image.fromarray(blurred_img.astype(np.uint8))


def get_target_transforms():
    """
    Get target transforms
    Args:
        args: input config arguments
        dataset: dataset class object

    return: target_transform, target_train_transform, target_aux_train_transform
    """

    target_transform = MaskToTensor()

    target_train_transform = MaskToTensor()

    target_aux_train_transform = MaskToTensor()

    return target_transform, target_train_transform, target_aux_train_transform



class RandomGaussianNoise(object):
    def __call__(self, img):
        img = img_as_float(img)
        if img.min() < 0:
            low_clip = -1.
        else:
            low_clip = 0.
        noise = np.random.normal(0., 0.005 ** 0.5,img.shape)
        noised_img = img + noise

        noised_img = np.clip(noised_img,low_clip,1.0)
        noised_img *= 255
        return Image.fromarray(noised_img.astype(np.uint8))
# def RandomGaussianNoise(img,mean,var):
#
#     img = img_as_float(img)
#     if img.min() < 0:
#         low_clip = -1.
#     else:
#         low_clip = 0.
#     noise = np.random.normal(mean, var,img.shape)
#     noised_img = img + noise
#
#     noised_img = np.clip(noised_img,low_clip,1.0)
#     noised_img *= 255
#     return Image.fromarray(noised_img.astype(np.uint8))
def ran(a,b):
    x =  np.random.uniform(a,b)
    return x
def get_train_transform():
     train_input_transform = []
     train_input_transform += [standard_transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)]
     train_input_transform += [standard_transforms.RandomApply([RandomGaussianNoise()], p=0.5)]
     train_input_transform += [standard_transforms.ToTensor()]
     train_input_transform = standard_transforms.Compose(train_input_transform)
     return train_input_transform
def get_train_joint_transform(args):
    """
    Get train joint transform
    Args:
        args: input config arguments
        dataset: dataset class object

    return: train_joint_transform_list, train_joint_transform
    """
    ignore_label = 255
    # Geometric image transformations
    train_joint_transform_list = []
    train_joint_transform_list += [
        joint_transforms.RandomSizeAndCrop(args.input_size,
                                           crop_nopad=args.crop_nopad,
                                           pre_size=args.pre_size,
                                           scale_min=args.scale_min,
                                           scale_max=args.scale_max,
                                           ignore_index=ignore_label),
        joint_transforms.Resize(args.crop_size),
        joint_transforms.RandomHorizontallyFlip()]


    train_joint_transform = joint_transforms.Compose(train_joint_transform_list)

    # return the raw list for class uniform sampling
    return train_joint_transform_list, train_joint_transform

class GTA5DataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.is_mirror = mirror
        self.train_input_transform = get_train_transform()
        self.target_transform, self.target_train_transform, self.target_aux_transform = get_target_transforms()

        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join('/public/home/yangliwei/dataset/gtav/GTAV/images/train/folder/%s'% name)
            label_file = osp.join( '/public/home/yangliwei/dataset/gtav/GTAV/labels/train/folder/%s'% name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)



    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        name = datafiles["name"]
        image = image.resize(self.crop_size, Image.BICUBIC)


        ###v2###  use transform function to transform
        image_transform = copy.deepcopy(image)
       
        image_transform =  self.train_input_transform(image_transform)
        rgb_mean_std_gt = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        image = transforms.Normalize(*rgb_mean_std_gt)(transforms.ToTensor()(image))
        image_transform = transforms.Normalize(*rgb_mean_std_gt)(image_transform)
        image = np.asarray(image, np.float32)
        image_transform = np.asarray(image_transform, np.float32)
        size = image.size

        ###v2 ###
        # print(np.max(image_cl_input))
        # image_record = copy.deepcopy(image)
        # re-assign labels to match the format of Cityscapes


        # return image.copy(), label_copy.copy(), np.array(size), name,image_cb_input.copy(),image_cc_input.copy(),image_cbc_input.copy(),image_gauss_noise_input.copy()
        return image.copy(), np.array(size), name,image_transform.copy()




if __name__ == '__main__':
    dst = GTA5DataSet("./data", is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]

            plt.imshow(img)
            plt.show()
