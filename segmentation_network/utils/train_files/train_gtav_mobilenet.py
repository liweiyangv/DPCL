"""
training code
"""
from __future__ import absolute_import
from __future__ import division
import argparse
import logging
import os
import torch

from config import cfg, assert_and_infer_cfg
from utils.misc_prototype import AverageMeter, prep_experiment, evaluate_eval, fast_hist
import datasets
import loss
import network
import optimizer_prototype as optimizer
import time
import torchvision.utils as vutils
import torch.nn.functional as F
from network.mynn import freeze_weights, unfreeze_weights
import numpy as np
import random
from skimage.measure import label as sklabel

from autoencoder.cyc_autoencoder import AutoEncoder
from torch.autograd import Variable
import os.path as osp
from utils.use_function import colorize_mask
import ignite.metrics.confusion_matrix as ConfusionMatrix
from sklearn.metrics import confusion_matrix as confusionmatrix

PRETRAINED_AUTOENCODER = '/public/home/yangliwei/olddata/yangliwei/DG_project/snapshots/ae/pretrained_ae/alter_mean_var/color_jitt_0_2_0_2_0_2_0_1_gauusian_noise/GTA5_only_transfer_105000.pth'
SNAPSHOT_DIR = '/public/home/yangliwei/DG_project/snapshots/disturb_recon_ori_with_alter_mean_var/hard_pixel_new_pixel_contrast_instance_triplet_orthometric_mobilenet_bs8_le_3_40000_v14/'
VISUAL_RESULT = '/public/home/yangliwei/DG_project/result/disturb_recon_ori_with_alter_mean_var/hard_pixel_new_pixel_contrast_instance_triplet_orthometric_mobilenet_bs8_le_3_40000_v14/'
PRE_RESULT = '/public/home/yangliwei/DG_project/pre_result/prototype_classifer/hard_pixel_new_pixel_contrast_instance_triplet_orthometric_mobilenet_bs8_le_3_40000_v14/'

# MEAN_CENTER = '/media/SuperJisk/yangliwei/experiment_data/fixed_point/splitted/result/ae/add_distanglement/prototype/2_undersampling_alter_mean_var/use_center_after_train/mean_20_var_20/mean_center_12388.pt'
# VAR_CENTER = '/media/SuperJisk/yangliwei/experiment_data/fixed_point/splitted/result/ae/add_distanglement/prototype/2_undersampling_alter_mean_var/use_center_after_train/mean_10_var_10/var_center_12388.pt'

BG_LABEL = [0,1,2,3,4,8,9,10]
FG_LABEL = [5,6,7,11,12,13,14,15,16,17,18]

# Argument Parser
parser = argparse.ArgumentParser(description='Semantic Segmentation')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--arch', type=str, default='network.deepv3_ori_prototype_dec2.DeepMobileNetV3PlusD',
                    help='Network architecture. We have DeepSRNX50V3PlusD (backbone: ResNeXt50) \
                    and deepWV3Plus (backbone: WideResNet38).')
parser.add_argument('--dataset', nargs='*', type=str, default=['gtav'],
                    help='a list of datasets; cityscapes, mapillary, camvid, kitti, gtav, mapillary, synthia')
parser.add_argument('--image_uniform_sampling', action='store_true', default=False,
                    help='uniformly sample images across the multiple source domains')
parser.add_argument('--val_dataset', nargs='*', type=str, default=['cityscapes'],
                    help='a list consists of cityscapes, mapillary, gtav, bdd100k, synthia')
parser.add_argument('--covstat_val_dataset', nargs='*', type=str, default=['gtav'],
                    help='a list consists of cityscapes, mapillary, gtav, bdd100k, synthia')
parser.add_argument('--cv', type=int, default=0,
                    help='cross-validation split id to use. Default # of splits set to 3 in config')
parser.add_argument('--class_uniform_pct', type=float, default=0.5,
                    help='What fraction of images is uniformly sampled')
parser.add_argument('--class_uniform_tile', type=int, default=1024,
                    help='tile size for class uniform sampling')
parser.add_argument('--coarse_boost_classes', type=str, default=None,
                    help='use coarse annotations to boost fine data with specific classes')

parser.add_argument('--img_wt_loss', action='store_true', default=False,
                    help='per-image class-weighted loss')
parser.add_argument('--cls_wt_loss', action='store_true', default=False,
                    help='class-weighted loss')
parser.add_argument('--batch_weighting', action='store_true', default=False,
                    help='Batch weighting for class (use nll class weighting using batch stats')

parser.add_argument('--jointwtborder', action='store_true', default=False,
                    help='Enable boundary label relaxation')
parser.add_argument('--strict_bdr_cls', type=str, default='',
                    help='Enable boundary label relaxation for specific classes')
parser.add_argument('--rlx_off_iter', type=int, default=-1,
                    help='Turn off border relaxation after specific epoch count')
parser.add_argument('--rescale', type=float, default=1.0,
                    help='Warm Restarts new learning rate ratio compared to original lr')
parser.add_argument('--repoly', type=float, default=1.5,
                    help='Warm Restart new poly exp')

parser.add_argument('--fp16', action='store_true', default=False,
                    help='Use Nvidia Apex AMP')
parser.add_argument('--local_rank', default=0, type=int,
                    help='parameter used by apex library')

parser.add_argument('--sgd', action='store_true', default=True)
parser.add_argument('--adam', action='store_true', default=False)
parser.add_argument('--amsgrad', action='store_true', default=False)

parser.add_argument('--freeze_trunk', action='store_true', default=False)
parser.add_argument('--hardnm', default=0, type=int,
                    help='0 means no aug, 1 means hard negative mining iter 1,' +
                    '2 means hard negative mining iter 2')

parser.add_argument('--trunk', type=str, default='resnet101',
                    help='trunk model, can be: resnet101 (default), resnet50')
parser.add_argument('--max_epoch', type=int, default=180)
parser.add_argument('--max_iter', type=int, default=40000)
parser.add_argument('--max_cu_epoch', type=int, default=100000,
                    help='Class Uniform Max Epochs')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--crop_nopad', action='store_true', default=False)
parser.add_argument('--rrotate', type=int,
                    default=0, help='degree of random roate')
parser.add_argument('--color_aug', type=float,
                    default=0.5, help='level of color augmentation')
parser.add_argument('--gblur', action='store_true', default=False,
                    help='Use Guassian Blur Augmentation')
parser.add_argument('--bblur', action='store_true', default=False,
                    help='Use Bilateral Blur Augmentation')
parser.add_argument('--lr_schedule', type=str, default='poly',
                    help='name of lr schedule: poly')
parser.add_argument('--poly_exp', type=float, default=0.9,
                    help='polynomial LR exponent')
parser.add_argument('--bs_mult', type=int, default=8,
                    help='Batch size for training per gpu')
parser.add_argument('--bs_mult_val', type=int, default=1,
                    help='Batch size for Validation per gpu')
parser.add_argument('--crop_size', type=int, default=768,
                    help='training crop size')
parser.add_argument('--pre_size', type=int, default=None,
                    help='resize image shorter edge to this before augmentation')
parser.add_argument('--scale_min', type=float, default=0.5,
                    help='dynamically scale training images down to this size')
parser.add_argument('--scale_max', type=float, default=2.0,
                    help='dynamically scale training images up to this size')
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--snapshot', type=str, default=None)
parser.add_argument('--restore_optimizer', action='store_true', default=False)


parser.add_argument('--city_mode', type=str, default='train',
                    help='experiment directory date name')
parser.add_argument('--date', type=str, default='2022_8_1_619',
                    help='experiment directory date name')
parser.add_argument('--exp', type=str, default='hard_pixel_new_pixel_contrast_instance_triplet_orthometric_mobilenet_bs8_le_2_40000_v14',
                    help='experiment directory name')
parser.add_argument('--tb_tag', type=str, default='',
                    help='add tag to tb dir')
parser.add_argument('--ckpt', type=str, default='./logs/ckpt',
                    help='Save Checkpoint Point')
parser.add_argument('--tb_path', type=str, default='./logs/tb',
                    help='Save Tensorboard Path')
parser.add_argument('--syncbn', action='store_true', default=False,
                    help='Use Synchronized BN')
parser.add_argument('--dump_augmentation_images', action='store_true', default=False,
                    help='Dump Augmentated Images for sanity check')
parser.add_argument('--test_mode', action='store_true', default=False,
                    help='Minimum testing to verify nothing failed, ' +
                    'Runs code for 1 epoch of train and val')
parser.add_argument('-wb', '--wt_bound', type=float, default=1.0,
                    help='Weight Scaling for the losses')
parser.add_argument('--maxSkip', type=int, default=0,
                    help='Skip x number of  frames of video augmented dataset')
parser.add_argument('--scf', action='store_true', default=False,
                    help='scale correction factor')
parser.add_argument('--dist_url', default='tcp://127.0.0.1:', type=str,
                    help='url used to set up distributed training')

parser.add_argument('--wt_layer', nargs='*', type=int, default=[0,0,0,0,0,0,0],
                    help='0: None, 1: IW/IRW, 2: ISW, 3: IS, 4: IN (IBNNet: 0 0 4 4 4 0 0)')
parser.add_argument('--wt_reg_weight', type=float, default=0.0)
parser.add_argument('--relax_denom', type=float, default=0.0)
parser.add_argument('--clusters', type=int, default=50)
parser.add_argument('--trials', type=int, default=10)
parser.add_argument('--dynamic', action='store_true', default=False)

parser.add_argument('--image_in', action='store_true', default=False,
                    help='Input Image Instance Norm')
parser.add_argument('--cov_stat_epoch', type=int, default=0,
                    help='cov_stat_epoch')
parser.add_argument('--visualize_feature', action='store_true', default=False,
                    help='Visualize intermediate feature')
parser.add_argument('--use_wtloss', action='store_true', default=False,
                    help='Automatic setting from wt_layer')
parser.add_argument('--use_isw', action='store_true', default=False,help='Automatic setting from wt_layer')

parser.add_argument("--gpu", type=int, default=0, help="choose gpu device.")
parser.add_argument("--RANDOM_SEED", type=int, default=304, help="choose gpu device.")
parser.add_argument("--BATCHSIZE", type=int, default=8, help="choose gpu device.")

"""
AUTOENCODER PARSER
"""
parser.add_argument("--pretrained_autoencoder",type=str,default=PRETRAINED_AUTOENCODER,help='path of pretrained autoencoder')
parser.add_argument("--visual_result", type=str, default=VISUAL_RESULT, help="where to save the visual result")
parser.add_argument("--snapshot_dir", type=str, default=SNAPSHOT_DIR, help="Where to save snapshots of the model.")
# parser.add_argument("--mean_center", type=str, default=MEAN_CENTER, help=" path of initialize centor")
# parser.add_argument("--var_center", type=str, default=VAR_CENTER,help=" path of initialize mean var")
parser.add_argument("--transfer_learning_rate", type=float, default=1e-5,help="the learning rate of the autoencoder")
parser.add_argument("--cluster_number", type=int, default=10, help="number of centor in k-means ")
parser.add_argument("--num_big_momentum", type=int, default=100, help="number of centor in k-means ")
parser.add_argument("--use_warmup", type=bool, default=False, help="number of centor in k-means ")
parser.add_argument("--pre_result", type=str, default=PRE_RESULT, help="where to save the visual result")
parser.add_argument("--warmup_steps", type=int, default=30000, help="number of centor in k-means ")
parser.add_argument("--warmup_end_epoch", type=int, default=9, help="number of end warmup epcoh ")
parser.add_argument("--random_select_num", type=int, default=30, help="number of select pixel in pixel_contrast")


args = parser.parse_args()

# Enable CUDNN Benchmarking optimization
#torch.backends.cudnn.benchmark = True
random_seed = args.RANDOM_SEED  #304
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
# torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

# args.world_size = 1

# Test Mode run two epochs with a few iterations of training and val
if args.test_mode:
    args.max_epoch = 2

# if 'WORLD_SIZE' in os.environ:
#     # args.apex = int(os.environ['WORLD_SIZE']) > 1
#     args.world_size = int(os.environ['WORLD_SIZE'])
#     print("Total world size: ", int(os.environ['WORLD_SIZE']))

torch.cuda.set_device(args.gpu)
# print('My Rank:', args.local_rank)
# Initialize distributed communication
# args.dist_url = args.dist_url + str(8000 + (int(time.time()%1000))//10)

# torch.distributed.init_process_group(backend='nccl',
#                                      init_method=args.dist_url,
#                                      world_size=args.world_size,
#                                      rank=args.local_rank)

# for i in range(len(args.wt_layer)):
#     if args.wt_layer[i] == 1:
#         args.use_wtloss = True
#     if args.wt_layer[i] == 2:
#         args.use_wtloss = True
#         args.use_isw = True
def seg_label(label,current_label):
    segs = []
    mask = label==current_label
    if torch.sum(mask)>0:
        masknp = mask.cpu().numpy().astype(int)
        seg, forenum = sklabel(masknp, background=0, return_num=True, connectivity=2)
        seg = torch.LongTensor(seg).cuda()
        pixelnum = np.zeros(forenum, dtype=int)
        for i in range(forenum):
            pixelnum[i] = torch.sum(seg==(i+1)).item()
        segs.append([seg, pixelnum])
    else:
        segs.append([mask.long(), np.zeros(0)])
    return segs

def random_select_pixel(feature_num,select_num):
    feature =feature_num.t()
    pixel_num = feature.shape[0]
    min_select = min(pixel_num,select_num)
    random_index = random.sample(range(0,pixel_num),min_select)
    selected_feature = feature[random_index,:]
    normalized_feature = torch.nn.functional.normalize(selected_feature,p=2,dim=1)
    return normalized_feature

def random_select_half_hard_pixel(feature_clone_2,easy_index,hard_index,select_num):
    hard_num = torch.sum(hard_index)
    easy_num = torch.sum(easy_index)
    hard_prop = 1/3
    easy_prop = 2/3
    hard_prop_num = int(select_num*hard_prop)
    easy_prop_num = int(select_num*easy_prop)
    
    easy_feature = feature_clone_2[:,easy_index.squeeze(1)].t()
    hard_feature = feature_clone_2[:,hard_index.squeeze(1)].t()


    more_hard_index = hard_num - hard_prop_num
    more_easy_index = easy_num - easy_prop_num

    if more_hard_index * more_easy_index >0:
        hard_select_num = min(hard_prop_num,hard_num)
        easy_select_num = min(easy_prop_num,easy_num)
    else:
        if more_hard_index < 0:
            hard_select_num = hard_num
            easy_select_num = min(easy_num,select_num-hard_num)
        elif more_hard_index == 0:
            hard_select_num = hard_num
            easy_select_num = min(easy_num,easy_prop_num)
        else:
            easy_select_num = easy_num
            hard_select_num = min(hard_num,select_num-easy_num)

    random_hard_index = random.sample(range(0,hard_num),hard_select_num)
    random_easy_index = random.sample(range(0,easy_num),easy_select_num)
    selected_hard_feature = hard_feature[random_hard_index,:]
    selected_easy_feature = easy_feature[random_easy_index,:]
    selected_feature = torch.cat((selected_easy_feature,selected_hard_feature),dim=0)
    normalized_feature = torch.nn.functional.normalize(selected_feature,p=2,dim=1)
    return normalized_feature

def cal_pixel_contrast_loss(pixel_feature,temperature):
    cat_feature = torch.cat(pixel_feature,dim=0)
    mask_matrix = torch.ones(0,0).cuda(args.gpu)

    for pixel_pos_index in np.arange(len(pixel_feature)):
        current_shape = pixel_feature[pixel_pos_index].shape[0]
        mask_matrix = torch.block_diag(mask_matrix, torch.ones(current_shape, current_shape).cuda(args.gpu))

    anchor_dot_contrast = torch.div(torch.matmul(cat_feature,cat_feature.t()),temperature)
    logits_max,_ = torch.max(anchor_dot_contrast,dim=1,keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()
    # logits = torch.div(torch.matmul(cat_feature,cat_feature.t())-torch.diag_embed(torch.ones(cat_feature.shape[0])).cuda(args.gpu),temperature)
    exp_logits = torch.exp(logits)

    logits_mask = torch.ones_like(mask_matrix).cuda(args.gpu) - torch.eye(cat_feature.shape[0]).cuda(args.gpu)
    positives_mask = mask_matrix * logits_mask
    negatives_mask = 1. - mask_matrix

    num_positives_per_row = torch.sum(positives_mask,axis=1)
    denominator = torch.sum(exp_logits*negatives_mask,axis=1,keepdim=True) + torch.sum(exp_logits * positives_mask,axis=1,keepdim=True)

    log_probs = logits - torch.log(denominator)
    if torch.any(torch.isnan(log_probs)):
        raise ValueError("Log_prob has nan")

    # log_probs_ = torch.sum(log_probs*positives_mask,axis=1)[num_positives_per_row>0]
    log_probs_ = torch.mean(log_probs*positives_mask,axis=1)[num_positives_per_row>0]

    pixel_wise_contrastive_loss = -log_probs_

    return pixel_wise_contrastive_loss.mean()

def js_div(p_logits, q_logits, get_softmax=True):
    """
    Function that measures JS divergence between target and output logits:
    """
    KLDivLoss = torch.nn.KLDivLoss(reduction='batchmean')
    if get_softmax:
        p_output = F.softmax(p_logits)
        q_output = F.softmax(q_logits)
    else:
        p_output = p_logits
        q_output = q_logits
    log_mean_output = ((p_output + q_output) / 2).log()
    return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output)) / 2

def js_div_2(p_logits, q_logits, get_softmax=True):
    """
    Function that measures JS divergence between target and output logits:
    """
    KLDivLoss = torch.nn.KLDivLoss(reduction='mean')
    if get_softmax:
        p_output = F.softmax(p_logits)
        q_output = F.softmax(q_logits)
    else:
        p_output = p_logits
        q_output = q_logits
    log_mean_output = ((p_output + q_output) / 2).log()
    return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output)) / 2

def cal_pixel_contrast_all_relation_loss(pixel_feature,temperature):
    # pixel_contrastive_loss = torch.Tensor([0]).cuda(args.gpu)

    cat_feature = torch.cat(pixel_feature,dim=0)
    mask_matrix = torch.ones(0,0).cuda(args.gpu)
    pixel_num = cat_feature.shape[0]
    for pixel_pos_index in np.arange(len(pixel_feature)):
        current_shape = pixel_feature[pixel_pos_index].shape[0]
        mask_matrix = torch.block_diag(mask_matrix, torch.ones(current_shape, current_shape).cuda(args.gpu))

    anchor_dot_contrast = torch.div(torch.matmul(cat_feature,cat_feature.t()),temperature)
    exp_anchor_dot_contrast = torch.exp(anchor_dot_contrast)
    exp_anchor = torch.nn.functional.normalize(exp_anchor_dot_contrast,dim=1,p=1)
    mask_normalize = torch.nn.functional.normalize(mask_matrix,dim=1,p=1)
    # mask_F_normalize = mask_normalize / torch.norm(mask_normalize)
    # exp_anchor_F_normalize = exp_anchor / torch.norm(exp_anchor)

    # for loss_index in np.arange(pixel_num):
    #     js_loss = js_div(exp_anchor[loss_index,:],mask_normalize[loss_index,:],get_softmax=False)
    #     # KL_loss_1 = F.kl_div(exp_anchor[loss_index,:].softmax(dim=-1).log(),mean_output[loss_index,:].softmax(dim=-1), reduction='mean')
    #     # KL_loss_2 = F.kl_div(mask_normalize[loss_index,:].softmax(dim=-1).log(),mean_output[loss_index,:].softmax(dim=-1), reduction='mean')
    #     pixel_contrastive_loss = pixel_contrastive_loss + js_loss
    #
    #     if js_loss < 0 or torch.isnan(js_loss) or torch.isinf(js_loss):
    #         raise ValueError("js_loss wrong ")
    pixel_contrastive_loss = js_div(exp_anchor,mask_normalize,get_softmax=False)
    return pixel_contrastive_loss
def supervised_prototype_memory_prototype(instance_protoype,prototype_normalize,prototype_temp):
    prototype_contrastive_loss = torch.Tensor([0]).cuda(args.gpu)
    cal_contrast_num = 0
    for list_index in np.arange(len(instance_protoype)):
        instance_num = instance_protoype[list_index].shape[0]
        if instance_num == 0:
            continue
        current_prototype = instance_protoype[list_index]
        current_sim_total_prototype = torch.matmul(current_prototype, prototype_normalize.t())
        current_sim_total_prototype_temp = torch.exp(current_sim_total_prototype / prototype_temp)
        current_sim_total_prototype_temp_sum = torch.sum(current_sim_total_prototype_temp, dim=1)

        prototype_contrastive_loss += torch.mean(-torch.log(current_sim_total_prototype_temp[:, list_index] / (current_sim_total_prototype_temp_sum)))
        cal_contrast_num += 1

    mean_prototype_contrastive_loss = prototype_contrastive_loss / cal_contrast_num
    return mean_prototype_contrastive_loss

def supervised_pixel_instance_hierarchical_loss(pixel_contrast_feature,instance_protoype,prototype_temp):
    prototype_contrastive_loss = torch.Tensor([0]).cuda(args.gpu)
    cal_contrast_num = 0
    for list_index in np.arange(len(instance_protoype)):
        instance_num = instance_protoype[list_index].shape[0]
        pixel_num = pixel_contrast_feature[list_index].shape[0]

        if pixel_num == 0:
            continue
        if instance_num == 0:  #for instance-pic pair if each pic has one instance pic_prototype will be 0 instance_will be two
            if pixel_num > 1:
                current_pixel = pixel_contrast_feature[list_index]
                colone_pixel = pixel_contrast_feature.copy()
                del_pixel = colone_pixel.pop(list_index)

                pixel_pos_sample = torch.matmul(current_pixel, current_pixel.t())
                pixel_pos_sample_temp = torch.exp(pixel_pos_sample / prototype_temp)
                pos_mask = torch.ones(pixel_num, pixel_num).cuda(args.gpu) - torch.eye(pixel_num).cuda(args.gpu)

                neg_pixel = torch.cat(colone_pixel, dim=0)
                pixel_neg_sample = torch.matmul(current_pixel, neg_pixel.t())

                pixel_neg_sample_temp = torch.exp(pixel_neg_sample / prototype_temp)
                pixel_neg_sample_temp_sum = torch.sum(pixel_neg_sample_temp, dim=1)
                pixel_neg_sample_temp_sum_repeat = pixel_neg_sample_temp_sum.repeat(pixel_num, 1).t()

                # sup_contrastive_loss = torch.sum(-torch.log(pixel_pos_sample_temp / (pixel_pos_sample_temp + pixel_neg_sample_temp_sum_repeat)) * pos_mask, dim=1) # 15 v1 v2
                sup_contrastive_loss = torch.mean(-torch.log(pixel_pos_sample_temp / (pixel_pos_sample_temp + pixel_neg_sample_temp_sum_repeat)) * pos_mask, dim=1)  # 15 v3

                prototype_contrastive_loss += torch.mean(sup_contrastive_loss)
                cal_contrast_num += 1

        else:
            if pixel_num > 1:
                current_prototype = instance_protoype[list_index]
                current_pixel = pixel_contrast_feature[list_index]
                colone_instance_prototype = instance_protoype.copy()
                colone_pixel = pixel_contrast_feature.copy()
                del_pixel = colone_pixel.pop(list_index)
                del_instance_prototype = colone_instance_prototype.pop(list_index)

                pixel_pos_pixel_sim = torch.matmul(current_pixel,current_pixel.t())
                pixel_pos_instance_prototype_sim = torch.matmul(current_pixel,current_prototype.t())
                pixel_pos_sample = torch.cat((pixel_pos_pixel_sim,pixel_pos_instance_prototype_sim),dim=1)

                pixel_pos_sample_temp = torch.exp(pixel_pos_sample / prototype_temp)
                pos_mask = torch.cat((torch.ones(pixel_num,pixel_num)-torch.eye(pixel_num),torch.ones(pixel_num,instance_num)),dim=1).cuda(args.gpu)

                neg_pixel = torch.cat(colone_pixel,dim=0)
                neg_instance_prototype = torch.cat(colone_instance_prototype,dim=0)
                pixel_neg_pixel_sim = torch.matmul(current_pixel,neg_pixel.t())
                pixel_neg_instance_prototype_sim = torch.matmul(current_pixel,neg_instance_prototype.t())
                pixel_neg_sample = torch.cat((pixel_neg_pixel_sim,pixel_neg_instance_prototype_sim),dim=1)

                pixel_neg_sample_temp = torch.exp(pixel_neg_sample / prototype_temp)
                pixel_neg_sample_temp_sum = torch.sum(pixel_neg_sample_temp,dim=1)
                pixel_neg_sample_temp_sum_repeat = pixel_neg_sample_temp_sum.repeat(pixel_num + instance_num,1).t()

                # sup_contrastive_loss = torch.sum(-torch.log(pixel_pos_sample_temp / (pixel_pos_sample_temp + pixel_neg_sample_temp_sum_repeat)) * pos_mask ,dim=1)  # 15 v1v2
                sup_contrastive_loss = torch.mean(-torch.log(pixel_pos_sample_temp / (pixel_pos_sample_temp + pixel_neg_sample_temp_sum_repeat)) * pos_mask ,dim=1)   # 15 v3

                # cat_sim_pos = torch.cat((pixel_pos_pixel_sim,pixel_pos_instance_prototype_sim),dim=1)
                # cat_sim_pos_div = torch.div(cat_sim_pos,prototype_temp)
                # logits_max,_ = torch.max(cat_sim_pos_div,dim=1,keepdim=True)
                # logits_max_repeat = logits_max.squeeze().repeat(pixel_num+instance_num,1).t()
                #
                # logits = cat_sim_pos_div - logits_max_repeat *(1. - pos_mask)
                # pixel_neg_sample_div = torch.div(pixel_neg_sample,prototype_temp)
                #
                # exp_logits_pos = torch.exp(logits)
                # exp_logits_neg = torch.exp(pixel_neg_sample_div)
                #
                # dominator = exp_logits_pos + torch.sum(exp_logits_neg,dim=1,keepdim=True)
                # a3 = logits - torch.log(dominator)

                prototype_contrastive_loss += torch.mean(sup_contrastive_loss)
                cal_contrast_num += 1

    mean_prototype_contrastive_loss = prototype_contrastive_loss / cal_contrast_num
    return mean_prototype_contrastive_loss


def supervised_instance_pic_memory_hierarchical_loss(pixel_contrast_feature, instance_protoype,prototype_normalize, prototype_temp):
    prototype_contrastive_loss = torch.Tensor([0]).cuda(args.gpu)
    cal_contrast_num = 0
    for list_index in np.arange(len(instance_protoype)):
        instance_num = instance_protoype[list_index].shape[0]
        pixel_num = pixel_contrast_feature[list_index].shape[0]
        memory_num = prototype_normalize[[list_index],:].shape[0]
        if pixel_num == 0:
            continue
        if instance_num == 0:
            if pixel_num > 1:# for instance-pic pair if each pic has one instance pic_prototype will be 0 instance_will be two
                current_pixel = pixel_contrast_feature[list_index]
                colone_instance_prototype = instance_protoype.copy()
                colone_pixel = pixel_contrast_feature.copy()
                del_pixel = colone_pixel.pop(list_index)
                del_instance_prototype = colone_instance_prototype.pop(list_index)

                pixel_pos_pixel_sim = torch.matmul(current_pixel, current_pixel.t())
                pixel_pos_pixel_sim_temp = torch.exp(pixel_pos_pixel_sim / prototype_temp)

                pixel_memory_sim = torch.matmul(current_pixel, prototype_normalize.t())
                pixel_memory_sim_temp = torch.exp(pixel_memory_sim / prototype_temp)

                pixel_pos_memory = pixel_memory_sim_temp[:, [list_index]]
                pixel_memory_neg_sum = torch.sum(pixel_memory_sim_temp, dim=1, keepdim=True) - pixel_pos_memory

                pixel_pos_total_sample_temp = torch.cat((pixel_pos_pixel_sim_temp, pixel_pos_memory), dim=1)

                pos_mask = torch.cat((torch.ones(pixel_num, pixel_num) - torch.eye(pixel_num),torch.ones(pixel_num, instance_num + memory_num)), dim=1).cuda(args.gpu)

                neg_pixel = torch.cat(colone_pixel, dim=0)
                neg_instance_prototype = torch.cat(colone_instance_prototype, dim=0)
                pixel_neg_pixel_sim = torch.matmul(current_pixel, neg_pixel.t())
                pixel_neg_instance_prototype_sim = torch.matmul(current_pixel, neg_instance_prototype.t())
                pixel_neg_sample = torch.cat((pixel_neg_pixel_sim, pixel_neg_instance_prototype_sim), dim=1)

                pixel_neg_sample_temp = torch.exp(pixel_neg_sample / prototype_temp)
                pixel_neg_sample_temp_sum = torch.sum(pixel_neg_sample_temp, dim=1, keepdim=True)

                pixel_neg_total_sum = pixel_memory_neg_sum + pixel_neg_sample_temp_sum
                pixel_neg_sample_temp_sum_repeat = pixel_neg_total_sum.squeeze().repeat(pixel_num + instance_num + memory_num, 1).t()

                # sup_contrastive_loss = torch.sum(-torch.log(pixel_pos_total_sample_temp / (pixel_pos_total_sample_temp + pixel_neg_sample_temp_sum_repeat)) * pos_mask, dim=1)  #sum for v1,v2
                sup_contrastive_loss = torch.mean(-torch.log(pixel_pos_total_sample_temp / (pixel_pos_total_sample_temp + pixel_neg_sample_temp_sum_repeat)) * pos_mask, dim=1)   #mean for v3

                prototype_contrastive_loss += torch.mean(sup_contrastive_loss)
                cal_contrast_num += 1
            else:
                current_pixel = pixel_contrast_feature[list_index]
                colone_instance_prototype = instance_protoype.copy()
                colone_pixel = pixel_contrast_feature.copy()
                del_pixel = colone_pixel.pop(list_index)
                del_instance_prototype = colone_instance_prototype.pop(list_index)

                pixel_memory_sim = torch.matmul(current_pixel, prototype_normalize.t())
                pixel_memory_sim_temp = torch.exp(pixel_memory_sim / prototype_temp)

                pixel_pos_memory = pixel_memory_sim_temp[:, [list_index]]
                pixel_memory_neg_sum = torch.sum(pixel_memory_sim_temp, dim=1, keepdim=True) - pixel_pos_memory

                neg_pixel = torch.cat(colone_pixel, dim=0)
                neg_instance_prototype = torch.cat(colone_instance_prototype, dim=0)
                pixel_neg_pixel_sim = torch.matmul(current_pixel, neg_pixel.t())
                pixel_neg_instance_prototype_sim = torch.matmul(current_pixel, neg_instance_prototype.t())
                pixel_neg_sample = torch.cat((pixel_neg_pixel_sim, pixel_neg_instance_prototype_sim), dim=1)

                pixel_neg_sample_temp = torch.exp(pixel_neg_sample / prototype_temp)
                pixel_neg_sample_temp_sum = torch.sum(pixel_neg_sample_temp, dim=1, keepdim=True)

                pixel_neg_total_sum = pixel_memory_neg_sum + pixel_neg_sample_temp_sum
                # sup_contrastive_loss = torch.sum(-torch.log(pixel_pos_memory / (pixel_pos_memory + pixel_neg_total_sum)), dim=1)  #sum for v1 v2
                sup_contrastive_loss = torch.mean(-torch.log(pixel_pos_memory / (pixel_pos_memory + pixel_neg_total_sum)), dim=1)   #mean for v3

                prototype_contrastive_loss += torch.mean(sup_contrastive_loss)
                cal_contrast_num += 1

        else:
            current_prototype = instance_protoype[list_index]
            current_pixel = pixel_contrast_feature[list_index]
            colone_instance_prototype = instance_protoype.copy()
            colone_pixel = pixel_contrast_feature.copy()
            del_pixel = colone_pixel.pop(list_index)
            del_instance_prototype = colone_instance_prototype.pop(list_index)

            pixel_pos_pixel_sim = torch.matmul(current_pixel, current_pixel.t())
            pixel_pos_instance_prototype_sim = torch.matmul(current_pixel, current_prototype.t())
            pixel_pos_sample = torch.cat((pixel_pos_pixel_sim, pixel_pos_instance_prototype_sim), dim=1)
            pixel_pos_sample_temp = torch.exp(pixel_pos_sample / prototype_temp)

            pixel_memory_sim = torch.matmul(current_pixel,prototype_normalize.t())
            pixel_memory_sim_temp = torch.exp(pixel_memory_sim / prototype_temp)

            pixel_pos_memory = pixel_memory_sim_temp[:,[list_index]]
            pixel_memory_neg_sum = torch.sum(pixel_memory_sim_temp,dim=1,keepdim=True)-pixel_pos_memory

            pixel_pos_total_sample_temp = torch.cat((pixel_pos_sample_temp, pixel_pos_memory), dim=1)

            pos_mask = torch.cat((torch.ones(pixel_num, pixel_num) - torch.eye(pixel_num), torch.ones(pixel_num, instance_num+memory_num)),dim=1).cuda(args.gpu)

            neg_pixel = torch.cat(colone_pixel, dim=0)
            neg_instance_prototype = torch.cat(colone_instance_prototype, dim=0)
            pixel_neg_pixel_sim = torch.matmul(current_pixel, neg_pixel.t())
            pixel_neg_instance_prototype_sim = torch.matmul(current_pixel, neg_instance_prototype.t())
            pixel_neg_sample = torch.cat((pixel_neg_pixel_sim, pixel_neg_instance_prototype_sim), dim=1)

            pixel_neg_sample_temp = torch.exp(pixel_neg_sample / prototype_temp)
            pixel_neg_sample_temp_sum = torch.sum(pixel_neg_sample_temp, dim=1,keepdim=True)

            pixel_neg_total_sum = pixel_memory_neg_sum + pixel_neg_sample_temp_sum
            pixel_neg_sample_temp_sum_repeat = pixel_neg_total_sum.squeeze().repeat(pixel_num + instance_num + memory_num, 1).t()

            # sup_contrastive_loss = torch.sum(-torch.log(pixel_pos_total_sample_temp / (pixel_pos_total_sample_temp + pixel_neg_sample_temp_sum_repeat)) * pos_mask,dim=1)  #sum for v1 v2
            sup_contrastive_loss = torch.mean(-torch.log(pixel_pos_total_sample_temp / (pixel_pos_total_sample_temp + pixel_neg_sample_temp_sum_repeat)) * pos_mask,dim=1)    # mean for v3

            current_prototype_memory_sim = torch.matmul(current_prototype,prototype_normalize.t())
            current_prototype_memory_sim_temp = torch.exp(current_prototype_memory_sim / prototype_temp)
            current_prototype_wise_neg = torch.matmul(current_prototype,neg_instance_prototype.t())
            current_prototype_wise_neg_temp = torch.exp(current_prototype_wise_neg / prototype_temp)
            current_prototype_memory_pos = current_prototype_memory_sim_temp[:,[list_index]]
            current_prototype_memory_neg_sum = torch.sum(current_prototype_memory_sim_temp,dim=1,keepdim=True) - current_prototype_memory_pos
            current_prototype_total_neg_sum = torch.sum(current_prototype_wise_neg_temp,dim=1,keepdim=True) + current_prototype_memory_neg_sum
            if instance_num > 2:   #prototype-wise
                current_prototype_wise_pos = torch.matmul(current_prototype,current_prototype.t())
                pos_instance_mask = torch.cat((torch.ones(instance_num, instance_num) - torch.eye(instance_num),torch.ones(instance_num, 1)), dim=1).cuda(args.gpu)
                current_prototype_wise_pos_temp = torch.exp(current_prototype_wise_pos / prototype_temp)
                currnet_prototype_total_pos = torch.cat((current_prototype_wise_pos_temp,current_prototype_memory_pos),dim=1)
                current_prototype_wise_contrastive_loss = torch.mean(-torch.log(currnet_prototype_total_pos / (currnet_prototype_total_pos + current_prototype_total_neg_sum)) * pos_instance_mask,dim=1,keepdim=True)    # mean for v3

            elif instance_num == 2:   #prototype-wise
                current_prototype_wise_pos = torch.matmul(current_prototype[[0],:],current_prototype[[1],:].t())
                current_prototype_wise_pos_temp = torch.exp(current_prototype_wise_pos / prototype_temp)
                currnet_prototype_total_pos = torch.cat((current_prototype_wise_pos_temp.repeat(instance_num,1),current_prototype_memory_pos),dim=1)
                current_prototype_wise_contrastive_loss = -torch.log(currnet_prototype_total_pos /(currnet_prototype_total_pos + current_prototype_total_neg_sum) )
            else:
                current_prototype_wise_contrastive_loss = -torch.log(current_prototype_memory_pos/(current_prototype_memory_pos + current_prototype_total_neg_sum))

            # sum_current_prototype_wise_contrastive_loss = torch.sum(current_prototype_wise_contrastive_loss,dim=1) #mean for v1 v2
            sum_current_prototype_wise_contrastive_loss = torch.mean(current_prototype_wise_contrastive_loss,dim=1) #mean for v3

            prototype_contrastive_loss += torch.mean(torch.cat((sup_contrastive_loss,sum_current_prototype_wise_contrastive_loss),dim=0))
            cal_contrast_num += 1

    mean_prototype_contrastive_loss = prototype_contrastive_loss / cal_contrast_num
    return mean_prototype_contrastive_loss

def instance_triplet_loss(instance_protoype,prototype_normalize, margin):
    current_triplet_loss = torch.Tensor([0]).cuda(args.gpu)
    cal_contrast_num = 0
    for list_index in np.arange(len(instance_protoype)):
        instance_num = instance_protoype[list_index].shape[0]
        if instance_num == 0:
            continue
        if instance_num > 1:
            pos_instance_prototype = instance_protoype[list_index]
            colone_instance_prototype = instance_protoype.copy()
            del_instance_prototype = colone_instance_prototype.pop(list_index)
            neg_instance_prototype = torch.cat(colone_instance_prototype, dim=0)
            anchor_prototype = prototype_normalize[[list_index],:]
            pos_anchor_distance = torch.norm((pos_instance_prototype[:,None]-anchor_prototype),dim=2,p=2)
            neg_anchor_distance = torch.norm((neg_instance_prototype[:,None]-anchor_prototype),dim=2,p=2)
            delta_distance = pos_anchor_distance - neg_anchor_distance.t()
            delta_margin = delta_distance + margin
            current_triplet_loss += torch.clamp(delta_margin,min=0).mean()
            cal_contrast_num += 1
    mean_triplet_loss = current_triplet_loss / cal_contrast_num
    return mean_triplet_loss

def supervised_pixel_instance_memory_hierarchical_loss(pixel_contrast_feature, instance_protoype,prototype_normalize, prototype_temp):
    prototype_contrastive_loss = torch.Tensor([0]).cuda(args.gpu)
    cal_contrast_num = 0
    for list_index in np.arange(len(instance_protoype)):
        instance_num = instance_protoype[list_index].shape[0]
        pixel_num = pixel_contrast_feature[list_index].shape[0]
        memory_num = prototype_normalize[[list_index],:].shape[0]
        if pixel_num == 0:
            continue
        if pixel_num > 1:
            current_prototype = instance_protoype[list_index]
            current_pixel = pixel_contrast_feature[list_index]
            colone_instance_prototype = instance_protoype.copy()
            colone_pixel = pixel_contrast_feature.copy()
            del_pixel = colone_pixel.pop(list_index)
            del_instance_prototype = colone_instance_prototype.pop(list_index)

            pixel_pos_pixel_sim = torch.matmul(current_pixel, current_pixel.t())
            pixel_pos_instance_prototype_sim = torch.matmul(current_pixel, current_prototype.t())
            pixel_pos_sample = torch.cat((pixel_pos_pixel_sim, pixel_pos_instance_prototype_sim), dim=1)
            pixel_pos_sample_temp = torch.exp(pixel_pos_sample / prototype_temp)

            pixel_memory_sim = torch.matmul(current_pixel,prototype_normalize.t())
            pixel_memory_sim_temp = torch.exp(pixel_memory_sim / prototype_temp)

            pixel_pos_memory = pixel_memory_sim_temp[:,[list_index]]
            pixel_memory_neg_sum = torch.sum(pixel_memory_sim_temp,dim=1,keepdim=True)-pixel_pos_memory

            pixel_pos_total_sample_temp = torch.cat((pixel_pos_sample_temp, pixel_pos_memory), dim=1)

            pos_mask = torch.cat((torch.ones(pixel_num, pixel_num) - torch.eye(pixel_num), torch.ones(pixel_num, instance_num+memory_num)),dim=1).cuda(args.gpu)

            neg_pixel = torch.cat(colone_pixel, dim=0)
            neg_instance_prototype = torch.cat(colone_instance_prototype, dim=0)
            pixel_neg_pixel_sim = torch.matmul(current_pixel, neg_pixel.t())
            pixel_neg_instance_prototype_sim = torch.matmul(current_pixel, neg_instance_prototype.t())
            pixel_neg_sample = torch.cat((pixel_neg_pixel_sim, pixel_neg_instance_prototype_sim), dim=1)

            pixel_neg_sample_temp = torch.exp(pixel_neg_sample / prototype_temp)
            pixel_neg_sample_temp_sum = torch.sum(pixel_neg_sample_temp, dim=1,keepdim=True)

            pixel_neg_total_sum = pixel_memory_neg_sum + pixel_neg_sample_temp_sum
            pixel_neg_sample_temp_sum_repeat = pixel_neg_total_sum.squeeze().repeat(pixel_num + instance_num + memory_num, 1).t()

            sup_contrastive_loss = torch.sum(-torch.log(pixel_pos_total_sample_temp / (pixel_pos_total_sample_temp + pixel_neg_sample_temp_sum_repeat)) * pos_mask,dim=1)

            prototype_contrastive_loss += torch.mean(sup_contrastive_loss)
            cal_contrast_num += 1

        else:
            current_pixel = pixel_contrast_feature[list_index]
            colone_instance_prototype = instance_protoype.copy()
            colone_pixel = pixel_contrast_feature.copy()
            del_pixel = colone_pixel.pop(list_index)
            del_instance_prototype = colone_instance_prototype.pop(list_index)


            pixel_memory_sim = torch.matmul(current_pixel,prototype_normalize.t())
            pixel_memory_sim_temp = torch.exp(pixel_memory_sim / prototype_temp)

            pixel_pos_memory = pixel_memory_sim_temp[:,[list_index]]
            pixel_memory_neg_sum = torch.sum(pixel_memory_sim_temp,dim=1,keepdim=True)-pixel_pos_memory


            neg_pixel = torch.cat(colone_pixel, dim=0)
            neg_instance_prototype = torch.cat(colone_instance_prototype, dim=0)
            pixel_neg_pixel_sim = torch.matmul(current_pixel, neg_pixel.t())
            pixel_neg_instance_prototype_sim = torch.matmul(current_pixel, neg_instance_prototype.t())
            pixel_neg_sample = torch.cat((pixel_neg_pixel_sim, pixel_neg_instance_prototype_sim), dim=1)

            pixel_neg_sample_temp = torch.exp(pixel_neg_sample / prototype_temp)
            pixel_neg_sample_temp_sum = torch.sum(pixel_neg_sample_temp, dim=1,keepdim=True)

            pixel_neg_total_sum = pixel_memory_neg_sum + pixel_neg_sample_temp_sum
            sup_contrastive_loss = torch.sum(-torch.log(pixel_pos_memory / (pixel_pos_memory + pixel_neg_total_sum)),dim=1)

            prototype_contrastive_loss += torch.mean(sup_contrastive_loss)
            cal_contrast_num += 1

    mean_prototype_contrastive_loss = prototype_contrastive_loss / cal_contrast_num
    return mean_prototype_contrastive_loss

def main():
    """
    Main Function
    """
    # Set up the Arguments, Tensorboard Writer, Dataloader, Loss Fn, Optimizer
    assert_and_infer_cfg(args)
    writer = prep_experiment(args, parser)
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    if not os.path.exists(args.visual_result):
        os.makedirs(args.visual_result)
    train_loader, val_loaders, train_obj, extra_val_loaders, covstat_val_loaders = datasets.setup_loaders(args)

    criterion, criterion_val = loss.get_loss(args)
    criterion_aux = loss.get_loss_aux(args)
    net = network.get_net(args, criterion, criterion_aux)
    net.cuda(args.gpu)
    optim, scheduler = optimizer.get_optimizer(args, net)
    model_transfer = AutoEncoder()
    model_transfer = model_transfer.cuda(args.gpu)
    optimizer_transfer = torch.optim.Adam(model_transfer.parameters(),lr=args.transfer_learning_rate,  weight_decay=0.0005)
    epoch = 0
    i = 0
    confusion_matrix = None
    weighted_matrix = None
    if args.use_warmup:
        print('warm up------------------------------------------------------------------------------------------')
        # #BS=2 WARM UP
        # seg_checkpoint = torch.load('/media/yangliwei/lemon/fixed_point/snapshots/prototype_classifer/11_with_prototype_grad_and_classifier_with_warm_up_confusion_matrix_prototype_contrast_correct_confusion_matrix_net_train/GTA5_only_seg_epoch_4.pth')
        # model_transfer.load_state_dict(torch.load('/media/yangliwei/lemon/fixed_point/snapshots/prototype_classifer/11_with_prototype_grad_and_classifier_with_warm_up_confusion_matrix_prototype_contrast_correct_confusion_matrix_net_train/GTA5_only_transfer_epoch_4.pth'))
        # net = optimizer.forgiving_state_restore(net,seg_checkpoint)
        # optim.load_state_dict(torch.load('/media/yangliwei/lemon/fixed_point/snapshots/prototype_classifer/11_with_prototype_grad_and_classifier_with_warm_up_confusion_matrix_prototype_contrast_correct_confusion_matrix_net_train/optim_epoch_4.pth'))
        # scheduler.load_state_dict(torch.load('/media/yangliwei/lemon/fixed_point/snapshots/prototype_classifer/11_with_prototype_grad_and_classifier_with_warm_up_confusion_matrix_prototype_contrast_correct_confusion_matrix_net_train/scheduler_epoch_4.pth'))
        # optimizer_transfer.load_state_dict(torch.load('/media/yangliwei/lemon/fixed_point/snapshots/prototype_classifer/11_with_prototype_grad_and_classifier_with_warm_up_confusion_matrix_prototype_contrast_correct_confusion_matrix_net_train/optimizer_transfer_epoch_4.pth'))
        # epoch = 5
        # seg_checkpoint = torch.load(
        #     '/media/SuperGisk/yangliwei/experiment_data/fixed_point/splitted/snapshots/prototype_classifer/pc/bs6_619_15_warmup/GTA5_only_seg_epoch_9.pth')
        # model_transfer.load_state_dict(torch.load(
        #     '/media/SuperGisk/yangliwei/experiment_data/fixed_point/splitted/snapshots/prototype_classifer/pc/bs6_619_15_warmup/GTA5_only_transfer_epoch_9.pth'))
        # net = optimizer.forgiving_state_restore(net, seg_checkpoint)
        # old_optim =torch.load(
        #     '/media/SuperGisk/yangliwei/experiment_data/fixed_point/splitted/snapshots/prototype_classifer/pc/bs6_619_15_warmup/optim_epoch_9.pth')
        # optim.param_groups[0]['lr'] = old_optim['param_groups'][0]['lr']
        # scheduler.load_state_dict(torch.load(
        #     '/media/SuperGisk/yangliwei/experiment_data/fixed_point/splitted/snapshots/prototype_classifer/pc/bs6_619_15_warmup/scheduler_epoch_9.pth'))
        # optimizer_transfer.load_state_dict(torch.load(
        #     '/media/SuperGisk/yangliwei/experiment_data/fixed_point/splitted/snapshots/prototype_classifer/pc/bs6_619_15_warmup/optimizer_transfer_epoch_9.pth'))
        # # weighted_matrix = torch.load(
        # #     '/media/SuperGisk/yangliwei/experiment_data/fixed_point/splitted/snapshots/prototype_classifer/pc/bs6_619_15_warmup/weighted_matrix_epoch_div_max_epoch_0.pt')
        # epoch = 10
        ####bs=6 lr=0.001 max_iter = 700000
        seg_checkpoint = torch.load(
            '/public/home/yangliwei/DG_project/snapshots/disturb_recon_ori_with_alter_mean_var/hard_pixel_new_pixel_contrast_instance_triplet_orthometric_mobilenet_bs8_le_3_40000_v10/GTA5_only_seg_epoch_9.pth',map_location={'cuda:0':'cuda:1'})
        model_transfer.load_state_dict(torch.load(
            '/public/home/yangliwei/olddata/yangliwei/DG_project/snapshots/ae/pretrained_ae/alter_mean_var/color_jitt_0_2_0_2_0_2_0_1_gauusian_noise/GTA5_only_transfer_105000.pth',map_location={'cuda:0':'cuda:1'}))
        net = optimizer.forgiving_state_restore(net, seg_checkpoint)
        optim_old = torch.load(
            '/public/home/yangliwei/DG_project/snapshots/disturb_recon_ori_with_alter_mean_var/hard_pixel_new_pixel_contrast_instance_triplet_orthometric_mobilenet_bs8_le_3_40000_v10/optim_epoch_9.pth',map_location={'cuda:0':'cuda:1'})
        optim.load_state_dict(optim_old)
        scheduler.load_state_dict(torch.load(
            '/public/home/yangliwei/DG_project/snapshots/disturb_recon_ori_with_alter_mean_var/hard_pixel_new_pixel_contrast_instance_triplet_orthometric_mobilenet_bs8_le_3_40000_v10/scheduler_epoch_9.pth',map_location={'cuda:0':'cuda:1'}))
        epoch = 10
    else:
        print('do not warm up------------------------------------------------------------------------------------------')
        model_transfer.load_state_dict(torch.load(args.pretrained_autoencoder,map_location='cpu'))

    cal_update_num = np.zeros(19)
    class_prototype = torch.randn(19,256).cuda(args.gpu)

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    if not os.path.exists(args.visual_result):
        os.makedirs(args.visual_result)
    if not os.path.exists(args.visual_result + '/recon/'):
        os.makedirs(args.visual_result + '/recon/')

    if not os.path.exists(args.visual_result + '/mask/'):
        os.makedirs(args.visual_result + '/mask/')
    if not os.path.exists(args.pre_result):
        os.makedirs(args.pre_result)



    if args.snapshot:
        print('use snapshot')
        net, optim, scheduler, epoch, mean_iu,model_transfer,optimizer_transfer,class_prototype = optimizer.load_weights(net, optim, scheduler, args.snapshot, args.restore_optimizer,model_transfer,optimizer_transfer,class_prototype)
        class_prototype = class_prototype.cuda(args.gpu)
        if args.restore_optimizer is True:
            iter_per_epoch = len(train_loader)
            epoch = epoch + 1
            i = iter_per_epoch * epoch
        else:
            epoch = 0

    print("#### iteration", i)
    torch.cuda.empty_cache()
    # Main Loop
    # for epoch in range(args.start_epoch, args.max_epoch):
    #
    # weighted_matrix = None

    while i < args.max_iter:
        # Update EPOCH CTR
        cfg.immutable(False)
        cfg.ITER = i
        cfg.immutable(True)
        if epoch <= args.warmup_end_epoch:
            i,class_prototype,cal_update_num = train(train_loader, net, model_transfer,optim, optimizer_transfer,epoch, writer, scheduler,class_prototype,cal_update_num,weighted_matrix, args.max_iter)
        elif epoch== args.warmup_end_epoch + 1:
            # for dataset, val_loader in val_loaders.items():
            #     confusion_matrix,weighted_matrix = cal_confusion_matrix(val_loader, dataset, net, criterion_val, optim, scheduler, epoch, writer, i, model_transfer,optimizer_transfer, class_prototype,save_pth=False)
            i,class_prototype,cal_update_num = train(train_loader, net, model_transfer,optim, optimizer_transfer,epoch, writer, scheduler,class_prototype,cal_update_num,weighted_matrix, args.max_iter)
        else:
            # if (epoch - args.warmup_end_epoch -1 ) % 5==0 :
            #     for dataset, val_loader in val_loaders.items():
            #         confusion_matrix,weighted_matrix = cal_confusion_matrix(val_loader, dataset, net, criterion_val, optim, scheduler, epoch, writer, i, model_transfer,optimizer_transfer, class_prototype,save_pth=False)

            i,class_prototype,cal_update_num = train(train_loader, net, model_transfer,optim, optimizer_transfer,epoch, writer, scheduler,class_prototype,cal_update_num,weighted_matrix, args.max_iter)

        # train_loader.sampler.set_epoch(epoch + 1)

        # if (args.dynamic and args.use_isw and epoch % (args.cov_stat_epoch + 1) == args.cov_stat_epoch) \
        #    or (args.dynamic is False and args.use_isw and epoch == args.cov_stat_epoch):
        #     net.module.reset_mask_matrix()
        #     for trial in range(args.trials):
        #         for dataset, val_loader in covstat_val_loaders.items():  # For get the statistics of covariance
        #             validate_for_cov_stat(val_loader, dataset, net, criterion_val, optim, scheduler, epoch, writer, i,
        #                                   save_pth=False)
        #             net.module.set_mask_matrix()

        if args.gpu == args.gpu:
        #
            print("Saving pth file...")
            evaluate_eval(args, net, optim, scheduler,None, None, [],
                        writer, epoch, "None", None, i, save_pth=True,model_transfer=model_transfer,optimizer_transfer=optimizer_transfer,class_prototype=class_prototype)
            torch.save(net.state_dict(), osp.join(args.snapshot_dir, 'GTA5_only_seg_epoch_' + str(epoch) + '.pth'))
            torch.save(scheduler.state_dict(), osp.join(args.snapshot_dir, 'scheduler_epoch_' + str(epoch) + '.pth'))
            # torch.save(optimizer_transfer.state_dict(), osp.join(args.snapshot_dir, 'optimizer_transfer_epoch_' + str(epoch) + '.pth'))
            torch.save(optim.state_dict(), osp.join(args.snapshot_dir, 'optim_epoch_' + str(epoch) + '.pth'))
            # torch.save(model_transfer.state_dict(), osp.join(args.snapshot_dir, 'GTA5_only_transfer_epoch_' + str(epoch) + '.pth'))
            torch.save(class_prototype, osp.join(args.snapshot_dir, 'class_prototype_epoch_' + str(epoch) + '.pt'))
            # torch.save(confusion_matrix, osp.join(args.snapshot_dir, 'confusion_matrix_epoch_together_' + str(epoch) + '.pt'))
            # torch.save(weighted_matrix, osp.join(args.snapshot_dir, 'weighted_matrix_epoch_together_' + str(epoch) + '.pt'))

        for dataset, val_loader in extra_val_loaders.items():
            print("Extra validating... This won't save pth file")
            validate(val_loader, dataset, net, criterion_val, optim, scheduler, epoch, writer, i, model_transfer,
                     optimizer_transfer, class_prototype, save_pth=False)

        # if args.class_uniform_pct:
        #     if epoch >= args.max_cu_epoch:
        #         train_obj.build_epoch(cut=True)
        #         train_loader.sampler.set_num_samples()
        #     else:
        #         train_obj.build_epoch()

        epoch += 1

    # Validation after epochs
    if len(val_loaders) == 1:
        # Run validation only one time - To save models
        for dataset, val_loader in val_loaders.items():
            validate(val_loader, dataset, net, criterion_val, optim, scheduler, epoch, writer, i,model_transfer, optimizer_transfer,class_prototype)
    else:
        if args.local_rank == 0:
            print("Saving pth file...")
            evaluate_eval(args, net, optim, scheduler, None, None, [],
                        writer, epoch, "None", None, i, save_pth=True)

    for dataset, val_loader in extra_val_loaders.items():
        print("Extra validating... This won't save pth file")
        validate(val_loader, dataset, net, criterion_val, optim, scheduler, epoch, writer, i, model_transfer, optimizer_transfer,class_prototype,save_pth=False)


def train(train_loader, net, model_transfer, optim, optimizer_transfer, curr_epoch, writer,scheduler,class_prototype, cal_update_num,weighted_matrix,max_iter):
    """
    Runs the training loop per epoch
    train_loader: Data loader for train
    net: thet network
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return:
    """
    net.train()
    model_transfer.eval()
    for param in model_transfer.parameters():
        param.requires_grad = False
    train_total_loss = AverageMeter()
    train_recon_loss = AverageMeter()
    train_pixel_memory_contrastive_loss = AverageMeter()
    train_instance_pic_refer_memory_contrastive_loss = AverageMeter()
    train_orthormetric_loss = AverageMeter()
    train_seg_loss = AverageMeter()
    train_pixel_wise_contrastive_loss = AverageMeter()
    # train_prototype_sim_loss = AverageMeter()
    time_meter = AverageMeter()
    class_num = 19
    criterion = net.criterion

    curr_iter = curr_epoch * len(train_loader)
    criterion_l1 = torch.nn.L1Loss()
    for i, data in enumerate(train_loader):

        if curr_iter >= max_iter:
            break

        ori_image,inputs, gts, _, aux_gts = data

        # Multi source and AGG case
        if len(inputs.shape) == 5:
            B, D, C, H, W = inputs.shape
            num_domains = D
            inputs = inputs.transpose(0, 1)
            gts = gts.transpose(0, 1).squeeze(2)
            aux_gts = aux_gts.transpose(0, 1).squeeze(2)

            inputs = [input.squeeze(0) for input in torch.chunk(inputs, num_domains, 0)]
            gts = [gt.squeeze(0) for gt in torch.chunk(gts, num_domains, 0)]
            aux_gts = [aux_gt.squeeze(0) for aux_gt in torch.chunk(aux_gts, num_domains, 0)]
        else:
            B, C, H, W = inputs.shape
            num_domains = 1
            inputs = [inputs]
            gts = [gts]
            aux_gts = [aux_gts]
            ori_image = [ori_image]

        batch_pixel_size = C * H * W
        for di, ingredients in enumerate(zip(ori_image,inputs, gts, aux_gts)):
            ori_image, input, gt, aux_gt = ingredients

            start_ts = time.time()

            img_gt = None
            input, gt,aux_gt = input.cuda(), gt.cuda(),aux_gt.cuda()
            ori_image = ori_image.cuda()
            # optimizer_transfer.zero_grad()
            optim.zero_grad()

            image_transform_encode = model_transfer.encoder(input)
            image_transform_encode_mean = torch.mean(image_transform_encode,dim=[2,3],keepdim=True)
            image_transform_encode_var = torch.var(image_transform_encode,dim=[2,3],keepdim=True)
            image_transform_normalize = (image_transform_encode - image_transform_encode_mean) / torch.sqrt(image_transform_encode_var)

            ori_image_encode = model_transfer.encoder(ori_image)
            ori_image_encode_mean = torch.mean(ori_image_encode,dim=[2,3],keepdim=True)
            ori_image_encode_var = torch.var(ori_image_encode,dim=[2,3],keepdim=True)

            recon_ori_feature = image_transform_normalize * torch.sqrt(ori_image_encode_var) + ori_image_encode_mean
            recon = model_transfer.decoder(recon_ori_feature)

            feature,pre_mask,outputs = net(recon, gts=gt, aux_gts=aux_gt, img_gt=img_gt, visualize=args.visualize_feature)

            if curr_epoch > args.warmup_end_epoch:
                feature_clone = feature.clone()
                _batch, _w, _h = gt.shape
                source_label_downsampled = gt.reshape([_batch, 1, _w, _h]).float()
                source_label_downsampled = F.interpolate(source_label_downsampled.float(),size=feature_clone.size()[2:], mode='nearest')
                source_label_downsampled = torch.tensor(source_label_downsampled, dtype=torch.long)

                new_class_prototype = torch.zeros(19,256).cuda(args.gpu)
                feature_clone_2 = feature_clone.transpose(1, 0)
                current_update_index = np.zeros(19)
                instance_protoype = []
                pixel_contrast_feature = []
                pic_prototype = []
                old_prototype = class_prototype.clone()

                for label_id in np.arange(class_num):
                    index = source_label_downsampled == label_id
                    sum = torch.sum(index)
                    if sum > 0:
                        feature_num = feature_clone_2[:, index.squeeze(1)]

                        _,pre_mask_max = torch.max(pre_mask,dim=1,keepdim=True)
                        wrong_index = pre_mask_max != source_label_downsampled
                        corrent_index = pre_mask_max == source_label_downsampled
                        hard_index = index & wrong_index
                        easy_index = index & corrent_index
                        current_pixel_feature = random_select_half_hard_pixel(feature_clone_2,easy_index,hard_index,args.random_select_num)
                        pixel_contrast_feature.append(current_pixel_feature)
                        new_class_prototype[label_id, :] = torch.mean(feature_num, dim=1)
                        if curr_epoch == args.warmup_end_epoch + 1:
                            if cal_update_num[label_id] == 0:
                                print('initial :', label_id)
                                class_prototype[label_id, :] = torch.mean(feature_num, dim=1)
                        cal_update_num[label_id] += 1
                        current_update_index[label_id] += 1
                        if label_id in BG_LABEL:
                            current_instance_prototype = []
                            current_pic_prototype = []
                            for _batch_index in np.arange(_batch):
                                current_label = source_label_downsampled[_batch_index, 0, :]
                                mask_current = current_label==label_id
                                if torch.sum(mask_current)<2:
                                    continue
                                bg_feature = feature_clone[_batch_index, :, mask_current]
                                bg_feature_prototype = torch.mean(bg_feature, dim=1).unsqueeze(0)
                                current_instance_prototype.append(bg_feature_prototype)

                            if len(current_instance_prototype) >0:
                                cat_prototype = torch.cat(current_instance_prototype,dim=0)
                                cat_normalize_prototype = torch.nn.functional.normalize(cat_prototype,dim=1,p=2)
                                instance_protoype.append(cat_normalize_prototype)
                            else:
                                current_instance_prototype = torch.empty(0, 256).cuda(args.gpu)
                                instance_protoype.append(current_instance_prototype)

                        else:
                            current_instance_prototype = []
                            current_pic_prototype = []

                            for _batch_index in np.arange(_batch):
                                current_label = source_label_downsampled[_batch_index,0, :]
                                seg_ins = seg_label(current_label,label_id)
                                segmask, pixelnum = seg_ins[0]
                                if len(pixelnum) == 0:
                                    continue
                                sortmax = np.argsort(pixelnum)[::-1]
                                for update_instance_index in range(min(5, len(sortmax))):  # update the largest 10 num in fg_label
                                    mask_current = segmask == (sortmax[update_instance_index] + 1)
                                # for update_instance_index in range(len(pixelnum)):  # update the largest 10 num in fg_label
                                #     mask_current = segmask == (update_instance_index + 1)
                                    fg_feature = feature_clone[_batch_index, :, mask_current]
                                    fg_feature_prototype = torch.mean(fg_feature, dim=1).unsqueeze(0)
                                    current_instance_prototype.append(fg_feature_prototype)
                            cat_prototype = torch.cat(current_instance_prototype,dim=0)
                            cat_normalize_prototype = torch.nn.functional.normalize(cat_prototype,dim=1,p=2)
                            instance_protoype.append(cat_normalize_prototype)

                    else:
                        current_instance_prototype = torch.empty(0, 256).cuda(args.gpu)
                        instance_protoype.append(current_instance_prototype)

                        current_pixel_feature = torch.empty(0, 256).cuda(args.gpu)
                        pixel_contrast_feature.append(current_pixel_feature)


                prototype_temp = 0.1
                prototype_normalize = torch.nn.functional.normalize(old_prototype,dim=1,p=2)
                pixel_wise_contrastive_loss = cal_pixel_contrast_all_relation_loss(pixel_contrast_feature,prototype_temp)

                # instance_pic_memory_contrastive_loss = supervised_instance_pic_memory_hierarchical_loss(instance_protoype,pic_prototype,prototype_normalize,prototype_temp)
                instance_pic_memory_contrastive_loss = instance_triplet_loss(instance_protoype, prototype_normalize, 0.)
                #feature = feature.transpose(1,0)
                b, c, w, h = feature.shape

                feature = feature.reshape(b,c,w * h)
                source_label_downsampled = source_label_downsampled.reshape(b,w * h)
                source_label_downsampled = torch.tensor(source_label_downsampled, dtype=torch.long)
                feature_normalize = torch.nn.functional.normalize(feature,dim=1,p=2)

                sim_sum = torch.matmul(prototype_normalize.unsqueeze(0),feature_normalize)

                # weighted_label = torch.where(source_label_downsampled<19,source_label_downsampled,0)
                # sim_weight = weighted_matrix[weighted_label,:]
                # sim_weight_permute = sim_weight.permute(0,2,1)
                # sim_sum_weighted = sim_sum * sim_weight_permute
                contrastive_loss = criterion(sim_sum/0.1, source_label_downsampled)


                # loss_ae = criterion_l1(recon, ori_image)
                outputs_index = 0
                main_loss = outputs[outputs_index]
                outputs_index += 1
                aux_loss = outputs[outputs_index]
                outputs_index += 1

                seg_loss = main_loss + (0.4 * aux_loss)

                # total_loss = 0.1 * loss_ae + contrastive_loss + seg_loss + mean_prototype_contrastive_loss

                # total_loss = main_loss + (0.4 * aux_loss)
                if curr_iter < args.num_big_momentum + (args.warmup_end_epoch + 1) * len(train_loader) + 1:
                    print('big step')
                    index_update = np.where(current_update_index > 0, True, False)  # update index
                    class_prototype[index_update, :] = 0.7 * old_prototype[index_update, :] + 0.3 * new_class_prototype[index_update, :]
                else:
                    if curr_epoch == args.warmup_end_epoch + 1:
                        print('first update epoch')
                        index_update = np.where(current_update_index > 0, True, False)  # update index
                        index_big = np.where(cal_update_num < 5., True, False)
                        index_big_update = index_big & index_update
                        index_small = np.where(cal_update_num >= 5., True, False)
                        index_small_update = index_small & index_update
                        class_prototype[index_big_update, :] = 0.7 * old_prototype[index_big_update,:] + 0.3 * new_class_prototype[index_big_update, :]
                        class_prototype[index_small_update, :] = 0.999 * old_prototype[index_small_update,:] + 0.001 * new_class_prototype[index_small_update, :]
                    else:
                        index_update = np.where(current_update_index > 0, True, False)  # update index
                        class_prototype[index_update, :] = 0.999 * old_prototype[index_update,:] + 0.001 * new_class_prototype[index_update, :]

                # class_prototype_normalize = torch.nn.functional.normalize(class_prototype,dim=1,p=2)
                # class_prototype_sim = torch.matmul(class_prototype_normalize,class_prototype_normalize.t())-torch.diag_embed(torch.ones(19)).cuda(args.gpu)
                # zeros_matrix = torch.zeros(19,19).cuda(args.gpu)
                # prototype_similarity_loss = torch.mean(torch.where(class_prototype_sim>0,class_prototype_sim,zeros_matrix))
                class_prototype_2 = class_prototype.clone()
                class_prototype_2_normalize = torch.nn.functional.normalize(class_prototype_2,dim=1)
                orthormetric_loss = torch.clamp(torch.matmul(class_prototype_2_normalize,class_prototype_2_normalize.t())-torch.eye(class_num).cuda(args.gpu),min=0).mean()
                total_loss = contrastive_loss + seg_loss + instance_pic_memory_contrastive_loss + 5 * pixel_wise_contrastive_loss + orthormetric_loss
                log_total_loss = total_loss.clone().detach_()
                loss_seg = seg_loss.clone().detach_()
                #loss_recon = loss_ae.clone().detach_()
                loss_pixel_memory_contrastive = contrastive_loss.clone().detach_()
                loss_instance_refer_pic_memory_contrastive = instance_pic_memory_contrastive_loss.clone().detach_()
                loss_pixel_wise_contrastive = pixel_wise_contrastive_loss.clone().detach_()
                loss_orthormetric = orthormetric_loss.clone().detach_()

                # loss_prototype_similarity = prototype_similarity_loss.clone().detach_()

                # torch.distributed.all_reduce(log_total_loss, torch.distributed.ReduceOp.SUM)
                # log_total_loss = log_total_loss / args.world_size
                train_total_loss.update(log_total_loss.item(), batch_pixel_size)
                #train_recon_loss.update(loss_recon.item(), batch_pixel_size)
                train_pixel_memory_contrastive_loss.update(loss_pixel_memory_contrastive.item(), batch_pixel_size)
                train_instance_pic_refer_memory_contrastive_loss.update(loss_instance_refer_pic_memory_contrastive.item(), batch_pixel_size)
                train_pixel_wise_contrastive_loss.update(loss_pixel_wise_contrastive.item(), batch_pixel_size)
                train_seg_loss.update(loss_seg.item(), batch_pixel_size)
                train_orthormetric_loss.update(loss_orthormetric.item(), batch_pixel_size)

                # train_prototype_sim_loss.update(loss_prototype_similarity.item(),batch_pixel_size)



                total_loss.backward()
                optim.step()
                class_prototype = class_prototype.detach()

                #optimizer_transfer.step()



            else:
                # loss_ae = criterion_l1(recon, ori_image)
                outputs_index = 0
                main_loss = outputs[outputs_index]
                outputs_index += 1
                aux_loss = outputs[outputs_index]
                outputs_index += 1
                seg_loss = main_loss + (0.4 * aux_loss)
                total_loss = seg_loss

                log_total_loss = total_loss.clone().detach_()
                loss_seg = seg_loss.clone().detach_()
                #loss_recon = loss_ae.clone().detach_()

                train_total_loss.update(log_total_loss.item(), batch_pixel_size)
                #train_recon_loss.update(loss_recon.item(), batch_pixel_size)
                train_seg_loss.update(loss_seg.item(), batch_pixel_size)

                total_loss.backward()
                optim.step()
                #optimizer_transfer.step()

            time_meter.update(time.time() - start_ts)

            del total_loss, log_total_loss
            del image_transform_normalize,ori_image_encode,image_transform_encode

            # if args.local_rank == 0:
            if args.gpu == args.gpu:
                if i % 50 == 49:
                    if args.visualize_feature:
                        visualize_matrix(writer, f_cor_arr, curr_iter, '/Covariance/Feature-')

                    msg = '[epoch {}], [iter {} / {} : {}], [loss_total {:0.6f}],[loss_seg {:0.6f}][loss_pixel_memory_contrastive {:0.6f}], [loss_instance-instance_refer_pic_memory_contrastive {:0.6f}], [loss pixel-pixel_contrast {:0.6f}],[loss_orthormetric {:0.6f}],[loss_recon {:0.6f}],[lr {:0.6f}], [time {:0.4f}]'.format(
                        curr_epoch, i + 1, len(train_loader), curr_iter, train_total_loss.avg,train_seg_loss.avg,train_pixel_memory_contrastive_loss.avg,train_instance_pic_refer_memory_contrastive_loss.avg,train_pixel_wise_contrastive_loss.avg,train_orthormetric_loss.avg,train_recon_loss.avg,
                        optim.param_groups[-1]['lr'], time_meter.avg / args.train_batch_size)

                    logging.info(msg)
                    if args.use_wtloss:
                        print("Whitening Loss", wt_loss)

                    # Log tensorboard metrics for each iteration of the training phase
                    writer.add_scalar('loss/train_total_loss', (train_total_loss.avg),curr_iter)
                    writer.add_scalar('loss/recon_loss', (train_recon_loss.avg),curr_iter)

                    train_total_loss.reset()
                    train_recon_loss.reset()
                    train_seg_loss.reset()
                    train_pixel_memory_contrastive_loss.reset()
                    train_instance_pic_refer_memory_contrastive_loss.reset()
                    train_pixel_wise_contrastive_loss.reset()
                    train_orthormetric_loss.reset()
                    time_meter.reset()

        curr_iter += 1
        scheduler.step()

        if curr_iter % 500 == 0:
            vutils.save_image(input, '%s/aeinput_%s.png' % (args.visual_result, str(curr_iter)),normalize=True)
            # vutils.save_image(recon_image, '%s/recon/recon_ori_%s.png'%(args.visual_result,str(curr_iter)), normalize=True)
            vutils.save_image(recon, '%s/recon/project_%s.png'%(args.visual_result,str(curr_iter)), normalize=True)
            vutils.save_image(ori_image, '%s/ori_%s.png'%(args.visual_result,str(curr_iter)), normalize=True)
            # pre_mask = pre_mask[0,:].transpose(1, 2, 0)
            pre_mask = pre_mask.cpu().data[0].numpy()
            pre_mask = pre_mask.transpose(1, 2, 0)

            pre_mask = np.asarray(np.argmax(pre_mask, axis=2), dtype=np.uint8)
            output_col = colorize_mask(pre_mask)
            output_col.save('%s/mask/%s_color.png' % (args.visual_result, str(curr_iter)))


        # if (curr_iter+1) % 5000 == 0:
        #     torch.save(net.state_dict(), osp.join(args.snapshot_dir, 'GTA5_only_seg_' + str(curr_iter+1) + '.pth'))
        #     torch.save(model_transfer.state_dict(), osp.join(args.snapshot_dir, 'GTA5_only_transfer_' + str(curr_iter+1) + '.pth'))
        #     torch.save(optim.state_dict(), osp.join(args.snapshot_dir, 'optim_' + str(curr_iter+1) + '.pth'))
        #     torch.save(optimizer_transfer.state_dict(), osp.join(args.snapshot_dir, 'optim_transfer' + str(curr_iter+1) + '.pth'))
        #     torch.save(scheduler.state_dict(), osp.join(args.snapshot_dir, 'scheduler_seg_' + str(curr_iter+1) + '.pth'))
        #     torch.save(class_prototype, osp.join(args.snapshot_dir, 'class_prototype_' + str(curr_iter+1) + '.pt'))



        if i > 5 and args.test_mode:
            return curr_iter,class_prototype

    return curr_iter,class_prototype,cal_update_num

def validate(val_loader, dataset, net, criterion, optim, scheduler, curr_epoch, writer, curr_iter, model_transfer, optimizer_transfer,class_prototype,save_pth=True):
    """
    Runs the validation loop after each training epoch
    val_loader: Data loader for validation
    dataset: dataset name (str)
    net: thet network
    criterion: loss fn
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return: val_avg for step function if required
    """

    net.eval()
    model_transfer.eval()
    val_loss = AverageMeter()
    val_classifier_loss = AverageMeter()
    val_prototype_loss = AverageMeter()
    iou_acc = 0
    iou_acc_classifier = 0
    iou_acc_prototype = 0
    error_acc = 0
    dump_images = []
    if not os.path.exists(args.pre_result+'/epoch_%s/'%str(curr_epoch)):
        os.makedirs(args.pre_result+'/epoch_%s/'%str(curr_epoch))

    for val_idx, data in enumerate(val_loader):
        # input        = torch.Size([1, 3, 713, 713])
        # gt_image           = torch.Size([1, 713, 713])
        inputs, gt_image, img_names, _ = data

        if len(inputs.shape) == 5:
            B, D, C, H, W = inputs.shape
            inputs = inputs.view(-1, C, H, W)
            gt_image = gt_image.view(-1, 1, H, W)

        assert len(inputs.size()) == 4 and len(gt_image.size()) == 3
        assert inputs.size()[2:] == gt_image.size()[1:]

        batch_pixel_size = inputs.size(0) * inputs.size(2) * inputs.size(3)
        inputs, gt_cuda = inputs.cuda(), gt_image.cuda()

        with torch.no_grad():
            if args.use_wtloss:
                output, f_cor_arr = net(inputs, visualize=True)
            else:
                image_transform = inputs.cuda(args.gpu)

                _,project_image = model_transfer(image_transform)
                class_prototype_clone = class_prototype.clone().detach()
                feature,classifier_output = net(project_image)
                classifier_prob = torch.softmax(classifier_output,dim=1)
                b,c,x,y = feature.shape
                feature = feature.reshape(c,x*y)

                distance = -torch.norm(class_prototype_clone[:,None]-feature.t(),p=2,dim=2)
                distance_softmax = torch.softmax(distance,dim=0).reshape(b,19,x,y)
                # distance_softmax = distance.reshape(distance.shape[0],x,y).unsqueeze(0)
                distance_prob = torch.nn.functional.interpolate(distance_softmax, size=classifier_prob.size()[2:], mode='bilinear',align_corners=True)
                output = torch.mean(torch.cat((classifier_prob,distance_prob),dim=0),dim=0,keepdim=True)




        del inputs

        assert output.size()[2:] == gt_image.size()[1:]
        assert output.size()[1] == datasets.num_classes

        val_loss.update(criterion(output, gt_cuda).item(), batch_pixel_size)
        val_classifier_loss.update(criterion(classifier_output, gt_cuda).item(), batch_pixel_size)
        val_prototype_loss.update(criterion(distance_prob, gt_cuda).item(), batch_pixel_size)

        del gt_cuda

        # Collect data from different GPU to a single GPU since
        # encoding.parallel.criterionparallel function calculates distributed loss
        # functions
        predictions = output.data.max(1)[1].cpu()
        pred_only_classifier = classifier_output.data.max(1)[1].cpu()
        pred_prototype = distance_prob.max(1)[1].cpu()

        # Logging
        if val_idx % 20 == 0:
            if args.local_rank == 0:
                logging.info("validating: %d / %d", val_idx + 1, len(val_loader))
                output = output.cpu().data[0].numpy()
                output = output.transpose(1, 2, 0)
                output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
                output_col = colorize_mask(output)
                output_col.save('%s/epoch_%s/%s_mean_output.png' % (args.pre_result, str(curr_epoch),str(val_idx)))

                classifier_prob_output = classifier_prob.cpu().data[0].numpy()
                classifier_prob_output = classifier_prob_output.transpose(1, 2, 0)
                classifier_prob_output = np.asarray(np.argmax(classifier_prob_output, axis=2), dtype=np.uint8)
                classifier_prob_output_col = colorize_mask(classifier_prob_output)
                classifier_prob_output_col.save('%s/epoch_%s/%s_classifier_output.png' % (args.pre_result, str(curr_epoch), str(val_idx)))

                pred_prototype_output = distance_prob.cpu().data[0].numpy()
                pred_prototype_output = pred_prototype_output.transpose(1, 2, 0)
                pred_prototype_output = np.asarray(np.argmax(pred_prototype_output, axis=2), dtype=np.uint8)
                pred_prototype_output_col = colorize_mask(pred_prototype_output)
                pred_prototype_output_col.save('%s/epoch_%s/%s_prototype_output.png' % (args.pre_result, str(curr_epoch), str(val_idx)))

                gt_show = gt_image.cpu().data[0].numpy()
                gt_show = np.asarray(gt_show, dtype=np.uint8)
                gt_show_col = colorize_mask(gt_show)
                gt_show_col.save('%s/epoch_%s/%s_gt.png' % (args.pre_result,str(curr_epoch),str(val_idx)))

        if val_idx > 10 and args.test_mode:
            break

        # Image Dumps
        if val_idx < 10:
            dump_images.append([gt_image, predictions, img_names])

        iou_acc += fast_hist(predictions.numpy().flatten(), gt_image.numpy().flatten(),
                             datasets.num_classes)
        iou_acc_classifier += fast_hist(pred_only_classifier.numpy().flatten(), gt_image.numpy().flatten(),datasets.num_classes)
        iou_acc_prototype += fast_hist(pred_prototype.numpy().flatten(), gt_image.numpy().flatten(),datasets.num_classes)

        del output, val_idx, data

    iou_acc_tensor = torch.cuda.FloatTensor(iou_acc)
    # torch.distributed.all_reduce(iou_acc_tensor, op=torch.distributed.ReduceOp.SUM)
    iou_acc = iou_acc_tensor.cpu().numpy()

    iou_acc_classifier_tensor = torch.cuda.FloatTensor(iou_acc_classifier)
    iou_acc_classifier = iou_acc_classifier_tensor.cpu().numpy()

    iou_acc_prototype_tensor = torch.cuda.FloatTensor(iou_acc_prototype)
    iou_acc_prototype = iou_acc_prototype_tensor.cpu().numpy()

    if args.gpu == args.gpu:
        evaluate_eval(args, net, optim, scheduler,val_loss, iou_acc, dump_images,
                    writer, curr_epoch, dataset, None, curr_iter,
                    save_pth=save_pth,model_transfer=model_transfer,optimizer_transfer=optimizer_transfer,class_prototype=class_prototype)
        evaluate_eval(args, net, optim, scheduler,val_classifier_loss, iou_acc_classifier, dump_images,
                    writer, curr_epoch, dataset, None, curr_iter,
                    save_pth=save_pth,model_transfer=model_transfer,optimizer_transfer=optimizer_transfer,class_prototype=class_prototype)
        evaluate_eval(args, net, optim, scheduler,val_prototype_loss, iou_acc_prototype, dump_images,
                    writer, curr_epoch, dataset, None, curr_iter,
                    save_pth=save_pth,model_transfer=model_transfer,optimizer_transfer=optimizer_transfer,class_prototype=class_prototype)
        if args.use_wtloss:
            visualize_matrix(writer, f_cor_arr, curr_iter, '/Covariance/Feature-')

    return val_loss.avg

def cal_confusion_matrix(val_loader, dataset, net, criterion, optim, scheduler, curr_epoch, writer, curr_iter, model_transfer, optimizer_transfer,class_prototype,save_pth=False):
    """
    Runs the validation loop after each training epoch
    val_loader: Data loader for validation
    dataset: dataset name (str)
    net: thet network
    criterion: loss fn
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return: val_avg for step function if required
    """

    net.eval()
    model_transfer.eval()
    val_loss = AverageMeter()
    val_classifier_loss = AverageMeter()
    val_prototype_loss = AverageMeter()
    iou_acc = 0
    iou_acc_classifier = 0
    iou_acc_prototype = 0
    error_acc = 0
    dump_images = []
    confusion_matrix = torch.tensor(torch.zeros(19,19),dtype=torch.int64).cuda(args.gpu)
    if not os.path.exists(args.pre_result+'/epoch_%s/'%str(curr_epoch)):
        os.makedirs(args.pre_result+'/epoch_%s/'%str(curr_epoch))

    for val_idx, data in enumerate(val_loader):
        # input        = torch.Size([1, 3, 713, 713])
        # gt_image           = torch.Size([1, 713, 713])
        inputs, gt_image, img_names, _ = data

        if len(inputs.shape) == 5:
            B, D, C, H, W = inputs.shape
            inputs = inputs.view(-1, C, H, W)
            gt_image = gt_image.view(-1, 1, H, W)

        assert len(inputs.size()) == 4 and len(gt_image.size()) == 3
        assert inputs.size()[2:] == gt_image.size()[1:]

        batch_pixel_size = inputs.size(0) * inputs.size(2) * inputs.size(3)
        inputs, gt_cuda = inputs.cuda(), gt_image.cuda()

        with torch.no_grad():
            if args.use_wtloss:
                output, f_cor_arr = net(inputs, visualize=True)
            else:
                image_transform = inputs.cuda(args.gpu)

                _,project_image = model_transfer(image_transform)
                class_prototype_clone = class_prototype.clone().detach()
                feature,classifier_output = net(project_image)
                classifier_prob = torch.softmax(classifier_output,dim=1)
                b,c,x,y = feature.shape
                feature = feature.reshape(c,x*y)

                distance = -torch.norm(class_prototype_clone[:,None]-feature.t(),p=2,dim=2)
                distance_softmax = torch.softmax(distance,dim=0).reshape(b,19,x,y)
                # distance_softmax = distance.reshape(distance.shape[0],x,y).unsqueeze(0)
                distance_prob = torch.nn.functional.interpolate(distance_softmax, size=classifier_prob.size()[2:], mode='bilinear',align_corners=True)
                output = torch.mean(torch.cat((classifier_prob,distance_prob),dim=0),dim=0,keepdim=True)

                b,w,h = gt_cuda.shape
                real_label_2 = gt_cuda.clone().reshape(w*h)
                pre_label_2 = classifier_output.reshape(19,w*h).t()
                a = ConfusionMatrix.ConfusionMatrix(num_classes=19)
                a.update([pre_label_2,real_label_2])
                confusion_matrix += a.confusion_matrix.cuda(args.gpu)

        del inputs

        assert output.size()[2:] == gt_image.size()[1:]
        assert output.size()[1] == datasets.num_classes

        val_loss.update(criterion(output, gt_cuda).item(), batch_pixel_size)
        val_classifier_loss.update(criterion(classifier_output, gt_cuda).item(), batch_pixel_size)
        val_prototype_loss.update(criterion(distance_prob, gt_cuda).item(), batch_pixel_size)

        del gt_cuda

        # Collect data from different GPU to a single GPU since
        # encoding.parallel.criterionparallel function calculates distributed loss
        # functions
        predictions = output.data.max(1)[1].cpu()
        pred_only_classifier = classifier_output.data.max(1)[1].cpu()
        pred_prototype = distance_prob.max(1)[1].cpu()

        # Logging
        if val_idx % 20 == 0:
            if args.local_rank == 0:
                logging.info("validating: %d / %d", val_idx + 1, len(val_loader))
                output = output.cpu().data[0].numpy()
                output = output.transpose(1, 2, 0)
                output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
                output_col = colorize_mask(output)
                output_col.save('%s/epoch_%s/%s_mean_output.png' % (args.pre_result, str(curr_epoch), str(val_idx)))

                classifier_prob_output = classifier_prob.cpu().data[0].numpy()
                classifier_prob_output = classifier_prob_output.transpose(1, 2, 0)
                classifier_prob_output = np.asarray(np.argmax(classifier_prob_output, axis=2), dtype=np.uint8)
                classifier_prob_output_col = colorize_mask(classifier_prob_output)
                classifier_prob_output_col.save('%s/epoch_%s/%s_classifier_output.png' % (args.pre_result, str(curr_epoch), str(val_idx)))

                pred_prototype_output = distance_prob.cpu().data[0].numpy()
                pred_prototype_output = pred_prototype_output.transpose(1, 2, 0)
                pred_prototype_output = np.asarray(np.argmax(pred_prototype_output, axis=2), dtype=np.uint8)
                pred_prototype_output_col = colorize_mask(pred_prototype_output)
                pred_prototype_output_col.save('%s/epoch_%s/%s_prototype_output.png' % (args.pre_result, str(curr_epoch), str(val_idx)))

                gt_show = gt_image.cpu().data[0].numpy()
                gt_show = np.asarray(gt_show, dtype=np.uint8)
                gt_show_col = colorize_mask(gt_show)
                gt_show_col.save('%s/epoch_%s/%s_gt.png' % (args.pre_result, str(curr_epoch), str(val_idx)))

        if val_idx > 10 and args.test_mode:
            break

        # Image Dumps
        if val_idx < 10:
            dump_images.append([gt_image, predictions, img_names])

        iou_acc += fast_hist(predictions.numpy().flatten(), gt_image.numpy().flatten(),
                             datasets.num_classes)
        iou_acc_classifier += fast_hist(pred_only_classifier.numpy().flatten(), gt_image.numpy().flatten(),datasets.num_classes)
        iou_acc_prototype += fast_hist(pred_prototype.numpy().flatten(), gt_image.numpy().flatten(),datasets.num_classes)

        del output, val_idx, data

    iou_acc_tensor = torch.cuda.FloatTensor(iou_acc)
    # torch.distributed.all_reduce(iou_acc_tensor, op=torch.distributed.ReduceOp.SUM)
    iou_acc = iou_acc_tensor.cpu().numpy()

    iou_acc_classifier_tensor = torch.cuda.FloatTensor(iou_acc_classifier)
    iou_acc_classifier = iou_acc_classifier_tensor.cpu().numpy()

    iou_acc_prototype_tensor = torch.cuda.FloatTensor(iou_acc_prototype)
    iou_acc_prototype = iou_acc_prototype_tensor.cpu().numpy()

    if args.gpu == args.gpu:
        evaluate_eval(args, net, optim, scheduler,val_loss, iou_acc, dump_images,
                    writer, curr_epoch, dataset, None, curr_iter,
                    save_pth=save_pth,model_transfer=model_transfer,optimizer_transfer=optimizer_transfer,class_prototype=class_prototype)
        evaluate_eval(args, net, optim, scheduler,val_classifier_loss, iou_acc_classifier, dump_images,
                    writer, curr_epoch, dataset, None, curr_iter,
                    save_pth=save_pth,model_transfer=model_transfer,optimizer_transfer=optimizer_transfer,class_prototype=class_prototype)
        evaluate_eval(args, net, optim, scheduler,val_prototype_loss, iou_acc_prototype, dump_images,
                    writer, curr_epoch, dataset, None, curr_iter,
                    save_pth=save_pth,model_transfer=model_transfer,optimizer_transfer=optimizer_transfer,class_prototype=class_prototype)
        if args.use_wtloss:
            visualize_matrix(writer, f_cor_arr, curr_iter, '/Covariance/Feature-')

    torch.save(confusion_matrix, osp.join(args.snapshot_dir, 'confusion_matrix_epoch_' + str(curr_epoch) + '.pt'))
    confusion_matrix_diag = torch.diag_embed(torch.diag(confusion_matrix))
    confusion_matrix_no_diag = confusion_matrix - confusion_matrix_diag
    confusion_matrix_no_diag = confusion_matrix_no_diag.float()
    confusion_matrix_max = torch.max(confusion_matrix_no_diag, dim=1)[0]
    confusion_matrix_max_repeat = confusion_matrix_max.repeat(19, 1).t()
    confusion_matrix_no_diag_div_max = torch.div(confusion_matrix_no_diag, confusion_matrix_max_repeat)

    identity_matrix = torch.diag_embed(torch.ones(19)).cuda()
    all_ones_matrix = torch.ones(19, 19).cuda()
    confusion_matrix_no_diag_div_max_add_1 = confusion_matrix_no_diag_div_max + all_ones_matrix - identity_matrix

    confusion_matrix_no_diag_normalize = torch.nn.functional.normalize(confusion_matrix_no_diag_div_max_add_1, dim=1,p=1)
    scale_weighted_matrix = 18 * confusion_matrix_no_diag_normalize
    weighted_matrix = identity_matrix + scale_weighted_matrix
    torch.save(weighted_matrix, osp.join(args.snapshot_dir, 'weighted_matrix_epoch_div_max_epoch_' + str(0) + '.pt'))

    return confusion_matrix,weighted_matrix

def validate_for_cov_stat(val_loader, dataset, net, criterion, optim, scheduler, curr_epoch, writer, curr_iter, save_pth=True):
    """
    Runs the validation loop after each training epoch
    val_loader: Data loader for validation
    dataset: dataset name (str)
    net: thet network
    criterion: loss fn
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return: val_avg for step function if required
    """

    # net.train()#eval()
    net.eval()

    for val_idx, data in enumerate(val_loader):
        img_or, img_photometric, img_geometric, img_name = data   # img_geometric is not used.
        img_or, img_photometric = img_or.cuda(), img_photometric.cuda()

        with torch.no_grad():
            net([img_photometric, img_or], cal_covstat=True)

        del img_or, img_photometric, img_geometric

        # Logging
        if val_idx % 20 == 0:
            if args.local_rank == 0:
                logging.info("validating: %d / 100", val_idx + 1)
        del data

        if val_idx >= 499:
            return


def visualize_matrix(writer, matrix_arr, iteration, title_str):
    stage = 'valid'

    for i in range(len(matrix_arr)):
        C = matrix_arr[i].shape[1]
        matrix = matrix_arr[i][0].unsqueeze(0)    # 1 X C X C
        matrix = torch.clamp(torch.abs(matrix), max=1)
        matrix = torch.cat((torch.ones(1, C, C).cuda(), torch.abs(matrix - 1.0),
                        torch.abs(matrix - 1.0)), 0)
        matrix = vutils.make_grid(matrix, padding=5, normalize=False, range=(0,1))
        writer.add_image(stage + title_str + str(i), matrix, iteration)


def save_feature_numpy(feature_maps, iteration):
    file_fullpath = '/home/userA/projects/visualization/feature_map/'
    file_name = str(args.date) + '_' + str(args.exp)
    B, C, H, W = feature_maps.shape
    for i in range(B):
        feature_map = feature_maps[i]
        feature_map = feature_map.data.cpu().numpy()   # H X D
        file_name_post = '_' + str(iteration * B + i)
        np.save(file_fullpath + file_name + file_name_post, feature_map)


if __name__ == '__main__':
    main()

