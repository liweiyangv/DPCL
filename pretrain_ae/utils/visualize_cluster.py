import argparse
import scipy
from scipy import ndimage
import numpy as np
import sys
from packaging import version
from dataset.gta5_dataset import GTA5DataSet
import torch.nn.functional as F

import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo

from model.deeplab import Deeplab_Single
# from model.Autoencoder import AutoEncoder
# from new_autoencoder import AutoEncoder
from autoencoder.cyc_autoencoder import AutoEncoder
# from dataset.cityscapes_dataset import cityscapesDataSet
from dataset.gtav_iou import GTAV_IOU
from collections import OrderedDict
import os
from PIL import Image
from torchvision import utils as vutils
import copy
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from utils.use_function import ED,local_normalization,local_normal_slow_version
DATA_DIRECTORY = '/media/yangliwei/yangliwei/yangliwei/Domain_Generalization/dataset/gtav/valid'
DATA_LIST_PATH = '/media/yangliwei/yangliwei/yangliwei/Domain_Generalization/code/RobustNet-main/RobustNet-main/split_data/gtav_split_val.txt'
# DATA_DIRECTORY = '/media/yangliwei/yangliwei/yangliwei/Domain_Generalization/dataset/gtav/test'
# DATA_LIST_PATH = '/media/yangliwei/yangliwei/yangliwei/Domain_Generalization/code/RobustNet-main/RobustNet-main/split_data/gtav_split_test.txt'
SAVE_PATH = './pre_result/cityscapes/1'

IGNORE_LABEL = 255
NUM_CLASSES = 19
NUM_STEPS = 500  # Number of images in the validation set.
#RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_multi-ed35151c.pth'
RESTORE_FROM = '/media/yangliwei/yangliwei/yangliwei/Domain_Generalization/experimental_data/fixed_point/snapshots/seg_and_ae/GTA5_only_seg_5000.pth'
RESTORE_FROM_VGG = 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_vgg-ac4ac9f6.pth'
RESTORE_FROM_ORC = 'http://vllab1.ucmerced.edu/~whung/adaptSeg/cityscapes_oracle-b7b9934.pth'
SET = 'val'
PRETRAINED_AUTOENCODER = '/media/yangliwei/lemon/fixed_point/snapshots/ae/add_distanglement/prototype/1/GTA5_only_transfer_150000.pth'
INPUT_SIZE = '1280,720'
ORI_K_MEANS_CENTOR = '/media/yangliwei/lemon/fixed_point/result/ae/add_distanglement/prototype/1_cluster_pixel/initial_use_online_k_means_1024_cluster/local_normal/100/k_means_centor_3000.pt'
ORI_KMEANS_MEAN = '/media/yangliwei/lemon/fixed_point/result/ae/add_distanglement/prototype/1_cluster_pixel/initial_use_online_k_means_1024_cluster/local_normal/100/k_means_centor_mean_3000.pt'
ORI_KMEANS_VAR = '/media/yangliwei/lemon/fixed_point/result/ae/add_distanglement/prototype/1_cluster_pixel/initial_use_online_k_means_1024_cluster/local_normal/100/k_means_centor_var_3000.pt'
IMG_SAVE = '/media/yangliwei/lemon/fixed_point/pre_result/seg_and_ae_cbc_add_distanglement/prototype/1/cluster_pixel/pretrain_visualize/new_local_normal/'
MODEL = 'Deeplab_Single'

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model_name",type=str,default=None,help="the name of the model")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="Model Choice (DeeplabMulti/DeeplabVGG/Oracle).")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    # parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
    #                     help="Where restore model parameters from.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose evaluation set.")
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result.")
    parser.add_argument("--img_save", type=str, default=IMG_SAVE,
                        help="Path to save image result.")
    parser.add_argument("--pretrained_autoencoder",type=str,default=PRETRAINED_AUTOENCODER,help="the path to load autoencoder")
    parser.add_argument("--add_infer",type=bool,default=False,help="whether to use the test infer")
    parser.add_argument("--lamda_iter", type=float, default=50000,help="hyperparamter")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--ori_k_means_center",type=str,default=ORI_K_MEANS_CENTOR,help=" path of initialize centor")
    parser.add_argument("--ori_k_means_mean",type=str,default=ORI_KMEANS_MEAN,help=" path of initialize mean var")
    parser.add_argument("--ori_k_means_var",type=str,default=ORI_KMEANS_VAR,help=" path of initialize mean var")

    return parser.parse_args()


def main():
    """Create the model and start the evaluation process."""
    # interp_1 = nn.Upsample(size=(512, 1024), mode='bilinear', align_corners=True)
    # interp_1 = nn.Upsample(size=(1280,720), mode='bilinear', align_corners=True)

    args = get_arguments()

    w, h = map(int, args.input_size.split(','))  # format is  '1280,720'
    input_size = (w, h)
    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear')
    interp_1 = nn.Upsample(size=(512,1024), mode='bilinear',align_corners=True)

    gpu0 = args.gpu

    if not os.path.exists(args.save):
        os.makedirs(args.save)


    model_transfer = AutoEncoder()
    model_transfer.load_state_dict(torch.load(args.pretrained_autoencoder))

    model_transfer.eval()
    model_transfer.cuda(gpu0)

    if not os.path.exists(args.save):
        os.makedirs(args.save)
    if not os.path.exists(args.img_save):
        os.makedirs(args.img_save)
    print('model name', args.model_name)
    print('restore_from',args.restore_from)
    print('pretrained_autoencoder',args.pretrained_autoencoder)
    print('save_path', args.save)
    print('img_save_path', args.img_save)
    print('centor_path',args.ori_k_means_center)



    testloader = data.DataLoader(
        GTAV_IOU(args.data_dir, args.data_list, crop_size=(1280,720), scale=False,
                          mirror=True, set=args.set), batch_size=2, shuffle=False, pin_memory=True)


    kmeans_centers = torch.load(args.ori_k_means_center)
    kmeans_centers_mean = torch.load(args.ori_k_means_mean)
    kmeans_centers_var = torch.load(args.ori_k_means_var)

    # intial_centor = F.normalize(torch.rand(( 256, 180, 320), dtype=torch.float),dim=1).cuda()  # Initialize the cluster
    # intial_centor = intial_centor.reshape((256,180*320))
    # for i in np.arange(180*320):
    #     intial_centor[:,i] = kmeans_centers[1,:]
    # intial_centor = intial_centor.reshape((256,180,320))
    # reconstruct = intial_centor * torch.sqrt(kmeans_centers_mean_var[1,0]) + torch.mean(kmeans_centers_mean_var[1,1])
    # reconstruct = torch.unsqueeze(reconstruct,dim=0)
    # reconstruct_image = model_transfer.decoder(reconstruct)
    # vutils.save_image(reconstruct_image, '%s/reconstruct_image.png' % (args.img_save), normalize=True)

    for index, batch in enumerate(testloader):
        for param in model_transfer.parameters():
            param.requires_grad = False

        # VALID

        if index % 100 == 0:
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),'%d processd' % index)
        ori_image, _, name = batch
        if args.model == 'Deeplab_Single':
            if args.add_infer :
                if index ==0:
                    print('add infer_iter in test')
                pass
            else:
                if index == 0:
                    print('no infer_iter')
                ori_image_1 = Variable(ori_image[0,:,:,:]).cuda(gpu0)
                ori_image_2 = Variable(ori_image[1,:,:,:]).cuda(args.gpu)
                ori_image_1 = torch.unsqueeze(ori_image_1,dim=0)
                ori_image_2 = torch.unsqueeze(ori_image_2,dim=0)
                ori_image_encode_1 = model_transfer.encoder(ori_image_1)
                ori_image_encode_2 = model_transfer.encoder(ori_image_2)
                ori_image_normalize_1,feature_mean_1,feature_var_1 = local_normalization(ori_image_encode_1,window_size=(2,2))
                ori_image_normalize_2,feature_mean_2,feature_var_2 = local_normalization(ori_image_encode_2,window_size=(2,2))

                reconstruct_1 = ori_image_normalize_1 * torch.sqrt(feature_var_2) + feature_mean_2
                reconstruct_2 = ori_image_normalize_2 * torch.sqrt(feature_var_1) + feature_mean_1
                project_image_1 = model_transfer.decoder(reconstruct_1)
                # recon_feature = ori_image_normalize * torch.sqrt(feature_var) + feature_mean
                project_image_2 = model_transfer.decoder(reconstruct_2)
                recon_featre_mean_1 = model_transfer.decoder(feature_mean_1)
                recon_featre_mean_2 = model_transfer.decoder(feature_mean_2)
                recon_featre_var_1 = model_transfer.decoder(feature_var_1)
                recon_featre_var_2 = model_transfer.decoder(feature_var_2)

                #
                vutils.save_image(project_image_1, '%s/local_norm_22_project_1_%s.png' % (args.img_save, index), normalize=True)
                vutils.save_image(project_image_2, '%s/local_norm_22_project_2_%s.png' % (args.img_save, index), normalize=True)
                vutils.save_image(recon_featre_mean_1, '%s/local_norm_22_mean_1_%s.png' % (args.img_save, index), normalize=True)
                vutils.save_image(recon_featre_mean_2, '%s/local_norm_22_mean_2_%s.png' % (args.img_save, index), normalize=True)
                vutils.save_image(recon_featre_var_1, '%s/local_norm_22_var_1_%s.png' % (args.img_save, index), normalize=True)
                vutils.save_image(recon_featre_var_2, '%s/local_norm_22_var_2_%s.png' % (args.img_save, index), normalize=True)
                vutils.save_image(ori_image_1, '%s/ori_1_%s.png' % (args.img_save, index), normalize=True)
                vutils.save_image(ori_image_2, '%s/ori_2_%s.png' % (args.img_save, index), normalize=True)

                # vutils.save_image(project_image_2, '%s/slow_project_image_%s.png' % (args.img_save, index),
                #                   normalize=True)
                # vutils.save_image(ori_recon, '%s/normalize_feature_image_%s.png' % (args.img_save, index),
                #                   normalize=True)
                print('1')
                # ori_image_normalize_2,feature_mean_2,feature_var_2 =  local_normal_slow_version(torch.squeeze(ori_image_encode))
                # ori_image_normalize = torch.squeeze(ori_image_normalize)
                # x, y, z = ori_image_normalize.shape
                # ori_image_normalize = ori_image_normalize.reshape(x, y * z)
                # distance = ED(ori_image_normalize, kmeans_centers)
                # min_index = torch.argmin(distance, dim=1)
                # alter_mean = kmeans_centers_mean[min_index, :].reshape(feature_mean.shape).cuda(args.gpu)
                #
                # alter_var = kmeans_centers_var[min_index, :].reshape(feature_var.shape).cuda(args.gpu)
                # ori_image_normalize = ori_image_normalize.reshape(ori_image_encode.shape).cuda(args.gpu)
                # reconstruct_feature = ori_image_normalize * torch.sqrt(alter_var) + alter_mean
                # ori_image_normalize = ori_image_normalize.reshape(ori_image_encode.shape).cuda(
                #     args.gpu)
                # ori_image_encode_mean = torch.mean(ori_image_encode, dim=1, keepdim=True)
                # ori_image_encode_var = torch.var(ori_image_encode, dim=1, keepdim=True)
                # ori_image_normalize = (ori_image_encode - ori_image_encode_mean) / torch.sqrt(ori_image_encode_var)
                # ori_image_normalize = torch.squeeze(ori_image_normalize)
                # x, y, z = ori_image_normalize.shape
                # ori_image_normalize = ori_image_normalize.reshape(x, y * z)
                # distance = ED(ori_image_normalize, kmeans_centers)
                # min_index = torch.argmin(distance, dim=1)
                # alter_mean = kmeans_centers_mean_var[min_index, 0].reshape(ori_image_encode_mean.shape).cuda(
                #     args.gpu)
                # alter_var = kmeans_centers_mean_var[min_index, 1].reshape(ori_image_encode_var.shape).cuda(
                #     args.gpu)
                # ori_image_normalize = ori_image_normalize.reshape(ori_image_encode.shape).cuda(
                #     args.gpu)
                # reconstruct_feature = ori_image_normalize * torch.sqrt(alter_var) + alter_mean

                # recon = model_transfer.decoder(recon_feature)
                # ori_recon = model_transfer.decoder(ori_image_normalize)
                # loss = torch.norm((recon_feature - ori_image_encode), p=1)


                # feature_mean_1 = torch.mean(ori_image_encode_1,dim=1)
                # feature_mean_2 = torch.mean(ori_image_encode_2,dim=1)
                # feature_var_1 = torch.var(ori_image_encode_1,dim=1)
                # feature_var_2 = torch.var(ori_image_encode_2,dim=1)
                # ori_image_normalize_1 = (ori_image_encode_1-feature_mean_1)/torch.sqrt(feature_var_1)
                # ori_image_normalize_2 = (ori_image_encode_2-feature_mean_2)/torch.sqrt(feature_var_2)

if __name__ == '__main__':
    main()