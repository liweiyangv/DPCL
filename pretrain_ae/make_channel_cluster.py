import argparse
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import sys
import os
import os.path as osp
import matplotlib.pyplot as plt
import random
import copy
from torchvision import utils as vutils
from skimage import exposure
from model.deeplab import Deeplab_Single
# from model.Autoencoder import AutoEncoder
# from new_autoencoder import AutoEncoder
from autoencoder.cyc_autoencoder import AutoEncoder
from utils.loss import CrossEntropy2d
from dataset.gtav_new_normalize import GTA5DataSet
import time
import torchvision

from dataset.gtav_iou import GTAV_IOU
from kmeans import KMeans as fast_k_means


MODEL = 'DeepLab'
BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 4
DATA_DIRECTORY = '/media/SuperJisk/yangliwei/dataset/city/gtav/train/'
DATA_LIST_PATH = '/media/SuperHisk/yangliwei/code/fixed_point_master/split_data/gtav_split_train.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '1280,720'
# DATA_DIRECTORY_TARGET = './data/Cityscapes/data'
# DATA_LIST_PATH_TARGET = './dataset/cityscapes_list/train.txt'
transfer_learning_rate=1e-4
MOMENTUM = 0.9
NUM_CLASSES = 19
NUM_STEPS = 300000
NUM_STEPS_STOP = 300000  # early stopping
POWER = 0.9
RANDOM_SEED = 1234
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 5000
VISUAL_RESULT = '/media/SuperJisk/yangliwei/experiment_data/fixed_point/splitted/result/ae/add_distanglement/prototype/2_undersampling_alter_mean_var_without_train_with_seg//mean_5_var_5/'
PRETRAINED_AUTOENCODER='/media/SuperJisk/yangliwei/experiment_data/fixed_point/splitted/snapshots/ae/2_undersampling_alter_mean_var/GTA5_only_transfer_105000.pth'
WEIGHT_DECAY = 0.0005

GAN = 'LS'

TARGET = 'cityscapes'
SET = 'train'


def get_arguments():
    """Parse all the arguments pr   d from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab")
    parser.add_argument("--target", type=str, default=TARGET,
                        help="available options : cityscapes")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")

    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")

    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    # parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
    #                     help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=2,
                        help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")

    parser.add_argument("--visual_result",type=str,default=VISUAL_RESULT,help="where to save the visual result")
    parser.add_argument("--pretrained_autoencoder",type=str,default=PRETRAINED_AUTOENCODER,help="the path to load autoencoder")
    parser.add_argument("--transfer_learning_rate",type=float,default=transfer_learning_rate,help="the learning rate of the autoencoder")
    parser.add_argument("--begin_iter",type=int,default=0,help="the beginning iter ")
    parser.add_argument("--cluster_number",type=int,default=5,help="number of centor in k-means ")
    parser.add_argument("--k_means_iter",type=int,default=10,help=" number of collected data to k-means ")

    return parser.parse_args()


args = get_arguments()




def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))



def recover_image():
    pass

def main():
    """Create the model and start the training."""

    w, h = map(int, args.input_size.split(','))  # format is  '1280,720'
    input_size = (w, h)
    # w, h = map(int, args.input_size_target.split(','))
    # input_size_target = (w, h)

    cudnn.enabled = True  # optimize the efficiency
    gpu = args.gpu

    # Create network


 # load the pretrained autoencoder for iter 100000
    model_transfer = AutoEncoder()
    pram = torch.load(args.pretrained_autoencoder)
    model_transfer.load_state_dict(pram)

    model_transfer.eval()
    model_transfer.cuda(args.gpu)

    cudnn.benchmark = True  # optimize the efficiency  follow the cudnn.enabled = true

    # if not os.path.exists(args.snapshot_dir):
    #     os.makedirs(args.snapshot_dir)
    if not os.path.exists(args.visual_result):
        os.makedirs(args.visual_result)

    trainloader = data.DataLoader(
        GTAV_IOU(args.data_dir, args.data_list, crop_size=input_size, scale=False,
                          mirror=False, set=args.set), batch_size=1, shuffle=False, pin_memory=True)
    print(len(trainloader))
    trainloader_iter = enumerate(trainloader)
    for param in model_transfer.parameters():
        param.requires_grad = False

    fast_kmeans_mean = fast_k_means(args.cluster_number, mode='euclidean', verbose=1)
    fast_kmeans_var = fast_k_means(args.cluster_number,mode='euclidean',verbose=1)
    # fast_kmeans_mean = fast_k_means(args.cluster_number, mode='cosine', verbose=1)
    # fast_kmeans_var = fast_k_means(args.cluster_number,mode='cosine',verbose=1)
    #
    # mean_data = torch.load('/media/SuperJisk/yangliwei/experiment_data/fixed_point/splitted/result/ae/add_distanglement/prototype/2_undersampling_alter_mean_var/use_center_after_train/mean_data_12388.pt').cuda(gpu)
    # var_data = torch.load('/media/SuperJisk/yangliwei/experiment_data/fixed_point/splitted/result/ae/add_distanglement/prototype/2_undersampling_alter_mean_var/use_center_after_train/var_data_12388.pt').cuda(gpu)

    #
    mean_data = torch.empty((0, 128)).cuda(args.gpu)
    var_data = torch.empty((0, 128)).cuda(args.gpu)
    for i_iter in range(len(trainloader)):
        if i_iter % 100 ==0:
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),'processing:',i_iter)
        _, batch = trainloader_iter.__next__()
        images_1, _,_,name = batch
        images_1 = Variable(images_1).cuda(args.gpu)
        ori_encode = model_transfer.encoder(images_1)
        ori_encode_mean = torch.mean(ori_encode,dim=[2,3],keepdim=True)
        ori_encode_var = torch.var(ori_encode,dim=[2,3],keepdim=True)
        ori_encode_mean = ori_encode_mean.squeeze(3).squeeze(2)
        ori_encode_var = ori_encode_var.squeeze(3).squeeze(2)
        mean_data = torch.cat([mean_data, ori_encode_mean], (0))
        var_data = torch.cat([var_data, ori_encode_var], (0))

        if i_iter % 5000 ==0:
            torch.save(mean_data, osp.join(args.visual_result, 'mean_data_%s' % str(i_iter + 1) + '.pt'))
            torch.save(var_data, osp.join(args.visual_result, 'var_data_%s' % str(i_iter + 1) + '.pt'))

    torch.save(mean_data, osp.join(args.visual_result, 'mean_data_%s' % str(i_iter + 1) + '.pt'))
    torch.save(var_data, osp.join(args.visual_result, 'var_data_%s' % str(i_iter + 1) + '.pt'))
    label_mean,mean_centor = fast_kmeans_mean.fit_predict(mean_data,centroids=None)
    label_var,var_centor = fast_kmeans_var.fit_predict(var_data,centroids=None)
    torch.save(mean_centor, osp.join(args.visual_result, 'mean_center_%s' % str(i_iter+1) + '.pt'))
    torch.save(var_centor, osp.join(args.visual_result, 'var_center_%s' % str(i_iter+1) + '.pt'))
    #
    # torch.save(label, osp.join(args.visual_result, 'label_%s' % (i_iter) + '.pt'))
    # kmeans_centers = torch.load('%s/ori_normalize_feature_centor_12387.pt'%args.visual_result)
    # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),'begin select existed feature')
    # for i_iter in np.arange(len(trainloader)):
    #     _, batch = trainloader_iter.__next__()
    #     if i_iter % 100 ==0:
    #         print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),'processing:',i_iter)
    #     images_1, _, name = batch
    #     images_1 = Variable(images_1).cuda(args.gpu)
    #     ori_encode = model_transfer.encoder(images_1)
    #     ori_encode_mean = torch.mean(ori_encode, dim=[2, 3], keepdim=True)
    #     ori_encode_var = torch.var(ori_encode, dim=[2, 3], keepdim=True)
    #     ori_encode_mean_squeeze = torch.squeeze(torch.squeeze(ori_encode_mean, axis=3), axis=2)
    #     ori_encode_var_squeeze = torch.squeeze(torch.squeeze(ori_encode_var, axis=3), axis=2)
    #     ori_normalize = (ori_encode - ori_encode_mean) / torch.sqrt(ori_encode_var)
    #     feature_mean_var = torch.cat([ori_encode_mean_squeeze, ori_encode_var_squeeze], 1)
    #     if i_iter % args.k_means_iter == 0:
    #         if i_iter > 0:
    #             if i_iter == args.k_means_iter :
    #                 print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),'calulate distance')
    #                 distance , existed_feature,index = measure_distance(normalized_feature_cluster,kmeans_centers)
    #                 mean_var_existed = normalized_feature_mean_var[index,:]
    #             else:
    #                 print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),'calulate distance')
    #                 new_distance,new_existed_feature,new_index = measure_distance(normalized_feature_cluster,kmeans_centers)
    #                 new_mean_var_existed = normalized_feature_mean_var[new_index,:]
    #                 for j in np.arange(len(new_distance)):
    #                     if new_distance[j]<distance[j]:
    #                         distance[j] = new_distance[j].clone()
    #                         existed_feature[j] = new_existed_feature[j,:].clone()
    #                         mean_var_existed[j] = new_mean_var_existed[j,:].clone()
    #         normalized_feature_cluster = ori_normalize.clone()
    #         normalized_feature_mean_var = feature_mean_var.clone()
    #     else:
    #         normalized_feature_cluster = torch.cat((normalized_feature_cluster, ori_normalize), axis=0)
    #         normalized_feature_mean_var = torch.cat((normalized_feature_mean_var, feature_mean_var), axis=0)
    # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 'calulate distance')
    # new_distance, new_existed_feature, new_index = measure_distance(normalized_feature_cluster, kmeans_centers)
    # new_mean_var_existed = normalized_feature_mean_var[new_index, :]
    # for j in np.arange(len(new_distance)):   #choose the min distance in all the data
    #     if new_distance[j] < distance[j]:
    #         distance[j] = new_distance[j].clone()
    #         existed_feature[j] = new_existed_feature[j, :].clone()
    #         mean_var_existed[j] = new_mean_var_existed[j, :].clone()
    #
    # torch.save(existed_feature, osp.join(args.visual_result, 'exist_normalize_feature_centor'  + '.pt'))
    # torch.save(mean_var_existed, osp.join(args.visual_result, 'exist_feature_centor_mean_var' + '.pt'))



if __name__ == '__main__':
    main()
