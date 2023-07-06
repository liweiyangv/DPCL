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
import copy

PRETRAINED_AUTOENCODER = '/public/home/yangliwei/olddata/yangliwei/DG_project/snapshots/ae/pretrained_ae/alter_mean_var/color_jitt_0_2_0_2_0_2_0_1_gauusian_noise/GTA5_only_transfer_105000.pth'
SNAPSHOT_DIR = '/public/home/yangliwei/DG_project/snapshots/disturb_recon_ori_with_alter_mean_var/mobilnet_iter_cityscpaes_entropy_ours/'
VISUAL_RESULT = '/public/home/yangliwei/DG_project/result/disturb_recon_ori_with_alter_mean_var/mobilnet_iter_cityscpaes_entropy_ours/'
PRE_RESULT = '/public/home/yangliwei/DG_project/pre_result/prototype_classifer/mobilnet_iter_cityscpaes_entropy_ours/'
BG_LABEL = [0,1,2,3,4,8,9,10]
FG_LABEL = [5,6,7,11,12,13,14,15,16,17,18]

# Argument Parser
parser = argparse.ArgumentParser(description='Semantic Segmentation')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--arch', type=str, default='network.deepv3_with_prototype_test_time.DeepMobileNetV3PlusD',
                    help='Network architecture. We have DeepSRNX50V3PlusD (backbone: ResNeXt50) \
                    and deepWV3Plus (backbone: WideResNet38).')
parser.add_argument('--dataset', nargs='*', type=str, default=['gtav'],
                    help='a list of datasets; cityscapes, mapillary, camvid, kitti, gtav, mapillary, synthia')
parser.add_argument('--image_uniform_sampling', action='store_true', default=False,
                    help='uniformly sample images across the multiple source domains')
parser.add_argument('--val_dataset', nargs='*', type=str, default=['cityscapes','bdd100k','mapillary'],
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
parser.add_argument('--max_iter', type=int, default=70000)
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
parser.add_argument('--bs_mult', type=int, default=2,
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
# parser.add_argument('--snapshot', type=str, default='/media/SuperHisk/yangliwei/experiment_data/fixed_point/splitted/snapshots/best/gtav/resnet50/best_total.pth')
parser.add_argument('--snapshot', type=str, default='./logs/ckpt/2022_8_1_619/hard_pixel_new_pixel_contrast_instance_triplet_orthometric_mobilenet_bs8_le_2_40000_v14/08_01_09/last_None_epoch_25_mean-iu_0.00000.pth')


parser.add_argument('--restore_optimizer', action='store_true', default=False)


parser.add_argument('--city_mode', type=str, default='train',
                    help='experiment directory date name')
parser.add_argument('--date', type=str, default='2022_10_23',
                    help='experiment directory date name')
parser.add_argument('--exp', type=str, default='mobilnet_iter_cityscpaes_entropy_ours',
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
parser.add_argument("--BATCHSIZE", type=int, default=2, help="choose gpu device.")

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
parser.add_argument("--num_big_momentum", type=int, default=20, help="number of centor in k-means ")
parser.add_argument("--use_warmup", type=bool, default=False, help="number of centor in k-means ")
parser.add_argument("--pre_result", type=str, default=PRE_RESULT, help="where to save the visual result")
parser.add_argument("--warmup_steps", type=int, default=30000, help="number of centor in k-means ")
parser.add_argument("--warmup_end_epoch", type=int, default=4, help="number of end warmup epcoh ")
parser.add_argument("--random_select_num", type=int, default=10, help="number of select pixel in pixel_contrast")


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

def cal_pixel_contrast_loss(pixel_feature,temperature):
    cat_feature = torch.cat(pixel_feature,dim=0)
    mask_matrix = torch.ones(0,0).cuda(args.gpu)

    for pixel_pos_index in np.arange(len(pixel_feature)):
        current_shape = pixel_feature[pixel_pos_index].shape[0]
        mask_matrix = torch.block_diag(mask_matrix, torch.ones(current_shape, current_shape).cuda(args.gpu))

    # anchor_dot_contrast = torch.div(torch.matmul(cat_feature,cat_feature.t()),temperature)
    # logits_max,_ = torch.max(anchor_dot_contrast,dim=1,keepdim=True)
    # logits = anchor_dot_contrast - logits_max.detach()
    logits = torch.div(torch.matmul(cat_feature,cat_feature.t())-torch.diag_embed(torch.ones(cat_feature.shape[0])).cuda(args.gpu),temperature)
    exp_logits = torch.exp(logits)

    logits_mask = torch.ones_like(mask_matrix).cuda(args.gpu) - torch.eye(cat_feature.shape[0]).cuda(args.gpu)
    positives_mask = mask_matrix * logits_mask
    negatives_mask = 1. - mask_matrix

    num_positives_per_row = torch.sum(positives_mask,axis=1)
    denominator = torch.sum(exp_logits*negatives_mask,axis=1,keepdim=True) + torch.sum(exp_logits * positives_mask,axis=1,keepdim=True)

    log_probs = logits - torch.log(denominator)
    if torch.any(torch.isnan(log_probs)):
        raise ValueError("Log_prob has nan")

    log_probs_ = torch.sum(log_probs*positives_mask,axis=1)[num_positives_per_row>0]
    pixel_wise_contrastive_loss = -log_probs_

    return pixel_wise_contrastive_loss.mean()


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
        seg_checkpoint = torch.load('/media/SuperHisk/yangliwei/experiment_data/fixed_point/splitted/snapshots/prototype_classifer/warm_4_epoch_confusion_matrix_train_mode/GTA5_only_seg_epoch_4.pth')
        model_transfer.load_state_dict(torch.load('/media/SuperHisk/yangliwei/experiment_data/fixed_point/splitted/snapshots/prototype_classifer/warm_4_epoch_confusion_matrix_train_mode/GTA5_only_transfer_epoch_4.pth'))
        net = optimizer.forgiving_state_restore(net,seg_checkpoint)
        optim.load_state_dict(torch.load('/media/SuperHisk/yangliwei/experiment_data/fixed_point/splitted/snapshots/prototype_classifer/warm_4_epoch_confusion_matrix_train_mode/optim_epoch_4.pth'))
        scheduler.load_state_dict(torch.load('/media/SuperHisk/yangliwei/experiment_data/fixed_point/splitted/snapshots/prototype_classifer/warm_4_epoch_confusion_matrix_train_mode/scheduler_epoch_4.pth'))
        optimizer_transfer.load_state_dict(torch.load('/media/SuperHisk/yangliwei/experiment_data/fixed_point/splitted/snapshots/prototype_classifer/warm_4_epoch_confusion_matrix_train_mode/optimizer_transfer_epoch_4.pth'))
        weighted_matrix = torch.load('/media/SuperHisk/yangliwei/experiment_data/fixed_point/splitted/snapshots/prototype_classifer/warm_4_epoch_confusion_matrix_train_mode/weighted_matrix_epoch_div_max_epoch_0.pt')
        epoch = 5
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
            i = iter_per_epoch * epoch
            epoch = epoch + 1
        else:
            epoch = 0

    print("#### iteration", i)
    torch.cuda.empty_cache()
    # for num in np.arange(20):
    #     epoch = int(1000-(num) * 50)

    for dataset, val_loader in extra_val_loaders.items():
        print("Extra validating... This won't save pth file")
        validate(val_loader, dataset, net, criterion_val, optim, scheduler, epoch, writer, i, model_transfer, optimizer_transfer,class_prototype,save_pth=False)

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

def cal_pixel_contrast_all_relation_loss(pixel_feature,temperature):
    # pixel_contrastive_loss = torch.Tensor([0]).cuda(args.gpu)

    cat_feature = torch.cat(pixel_feature,dim=0)
    mask_matrix = torch.ones(0,0).cuda(args.gpu)
    pixel_num = cat_feature.shape[0]
    for pixel_pos_index in np.arange(len(pixel_feature)):
        current_shape = pixel_feature[pixel_pos_index].shape[0]
        mask_matrix = torch.block_diag(mask_matrix, torch.ones(current_shape, current_shape).cuda(args.gpu))
    a = torch.matmul(cat_feature,cat_feature.t())
    anchor_dot_contrast = torch.div(torch.matmul(cat_feature,cat_feature.t()),temperature)
    exp_anchor_dot_contrast = torch.exp(anchor_dot_contrast)
    exp_anchor = torch.nn.functional.normalize(exp_anchor_dot_contrast,dim=1,p=1)
    mask_normalize = torch.nn.functional.normalize(mask_matrix,dim=1,p=1)
    pixel_contrastive_loss = js_div(exp_anchor,mask_normalize,get_softmax=False)
    return pixel_contrastive_loss



def cal_test_contrastive_entropy_loss(net,image,prototype,threshold,num):
    feature, classifier_output_low, classifier_output = net(image)
    classifier_prob_low = torch.softmax(classifier_output_low, dim=1)
    b, c, x, y = feature.shape
    feature = feature.reshape(c, x * y)
    distance = -torch.norm(prototype[:, None] - feature.t(), p=2, dim=2)
    distance_softmax = torch.softmax(distance, dim=0).reshape(b, 19, x, y)
    output = torch.mean(torch.cat((classifier_prob_low, distance_softmax), dim=0), dim=0, keepdim=True)

    loss_entropy = torch.sum(torch.sum(-output * torch.log2(output+1e-10),dim=1))
    loss_entropy_mean = 1/x * 1/y * loss_entropy
    predictions = output.data.max(1)[1]
    max_pred, _ = torch.max(output, dim=1)
    confident_index = torch.where(max_pred > threshold, True, False).cuda(args.gpu)
    pixel_feature = []
    for label_id in np.arange(19):
        pred_id = predictions == label_id
        update_id = pred_id & confident_index
        if torch.sum(update_id) > 10:
            # current_predict = output[0,label_id,update_id.squeeze()]
            # select_num = min(1000,int(current_predict.shape[0]))
            # _, idx1 = torch.sort(current_predict, descending=True)
            # idx = idx1[:select_num]
            current_feature = feature[:, update_id.squeeze().reshape(x * y)]
            # select_num = min(1000,int(current_feature.shape[1]*0.5))

            # select_feature = current_feature[:,idx].t()
            # select_feature_normalize = torch.nn.functional.normalize(select_feature,dim=1,p=2)
            select_feature_normalize = random_select_pixel(current_feature,num)
            pixel_feature.append(select_feature_normalize)
        else:
            current_pixel = torch.empty((0,256)).cuda(args.gpu)
            pixel_feature.append(current_pixel)

    # class_prototype_up = update_prototype_topk(prototype.clone(), output, feature, 0.5, 0.9, 0.8)

    loss_contrast = cal_pixel_contrast_all_relation_loss(pixel_feature,1)
    # return loss_contrast,class_prototype_up
    return loss_contrast+loss_entropy_mean,prototype

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
    for param in model_transfer.parameters():
        param.requires_grad = False

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
    mean_data = torch.load('./logs/ckpt/619/ae/mean_center_12388.pt').cuda(args.gpu)
    var_data = torch.load('./logs/ckpt/619/ae/var_center_12388.pt').cuda(args.gpu)

    for val_idx, data in enumerate(val_loader):
        # input        = torch.Size([1, 3, 713, 713])
        # gt_image           = torch.Size([1, 713, 713])
        if str(dataset)=='gtav':
            inputs, gt_image, img_names, _, border = data
        else:
            inputs, gt_image, img_names, _ = data

        if len(inputs.shape) == 5:
            B, D, C, H, W = inputs.shape
            inputs = inputs.view(-1, C, H, W)
            gt_image = gt_image.view(-1, 1, H, W)

        assert len(inputs.size()) == 4 and len(gt_image.size()) == 3
        if str(dataset)!='gtav':
            assert inputs.size()[2:] == gt_image.size()[1:]

        batch_pixel_size = inputs.size(0) * inputs.size(2) * inputs.size(3)
        if str(dataset)=='gtav':
            b1,b2,b3,b4 = border[0],border[1],border[2],border[3]
            ww = inputs.size(2)
            hh=inputs.size(3)
        inputs, gt_cuda = inputs.cuda(), gt_image.cuda()
        # net,optim = iter_all_net_params(net)

        # with torch.no_grad():
        if args.use_wtloss:
            output, f_cor_arr = net(inputs, visualize=True)
        else:

            image_transform = inputs.cuda(args.gpu)
            # image_transform = Variable(image_transform, requires_grad=True).cuda(args.gpu)
            lamda = 1e-6

            image_transform_encode = model_transfer.encoder(image_transform)
            image_transform_encode_mean = torch.mean(image_transform_encode, dim=[2, 3], keepdim=True)
            image_transform_encode_var = torch.var(image_transform_encode, dim=[2, 3], keepdim=True)
            image_transform_normalize = (image_transform_encode - image_transform_encode_mean) / torch.sqrt(image_transform_encode_var)
            image_transform_encode_mean_ = image_transform_encode_mean.squeeze(2).squeeze(2)
            image_transform_encode_var_ = image_transform_encode_var.squeeze(2).squeeze(2)
            distance_mean = torch.norm(mean_data[:,None]-image_transform_encode_mean_,dim=2,p=2)
            distance_var = torch.norm(var_data[:,None]-image_transform_encode_var_,dim=2,p=2)
            _,mean_sort_id = torch.sort(distance_mean,dim=0,descending=False)
            _,var_sort_id = torch.sort(distance_var,dim=0,descending=False)
            alter_mean = mean_data[[mean_sort_id[0]],:].unsqueeze(2).unsqueeze(3)
            alter_var = var_data[[var_sort_id[0]],:].unsqueeze(2).unsqueeze(3)
            recon_feature = image_transform_normalize * torch.sqrt(alter_var) + alter_mean
            project_image = model_transfer.decoder(recon_feature)
            if str(dataset) == 'gtav':
                project_image = project_image[:, :, b2:ww - b4, b1:hh - b3]
                assert project_image.size()[2:] == gt_image.size()[1:]
            # _, project_image = model_transfer(image_transform)
            class_prototype_clone = class_prototype.clone().detach()
            project_image = Variable(project_image, requires_grad=True).cuda(args.gpu)

            # iter_image_1 = iter_avg_output(net, project_image,class_prototype_clone, lamda)
            #
            test_contrastive_loss,class_prototype_clone = cal_test_contrastive_entropy_loss(net, project_image,class_prototype_clone,0.,1000)
            # test_contrastive_loss_2,class_prototype_clone = cal_test_contrastive_loss(net, project_image,class_prototype_clone,0.5,curr_epoch)

            grad_contrastive = torch.autograd.grad(test_contrastive_loss,project_image)
            iter_image_1 = project_image - 1 * grad_contrastive[0]
            iter_image_1 = iter_image_1.detach()
            torch.cuda.empty_cache()

            with torch.no_grad():
                class_prototype_clone = class_prototype.clone().detach()
                feature, classifier_output_low, classifier_output = net(iter_image_1)
                classifier_prob = torch.softmax(classifier_output, dim=1)
                classifier_prob_low = torch.softmax(classifier_output_low, dim=1)

                b, c, x, y = feature.shape
                feature = feature.reshape(c, x * y)
                distance = -torch.norm(class_prototype_clone[:, None] - feature.t(), p=2, dim=2)


                distance_softmax = torch.softmax(distance, dim=0).reshape(b, 19, x, y)
                output = torch.mean(torch.cat((classifier_prob_low, distance_softmax), dim=0), dim=0, keepdim=True)
                output = torch.nn.functional.interpolate(output, size=classifier_output.size()[2:], mode='bilinear',align_corners=True)
                distance_prob = torch.nn.functional.interpolate(distance_softmax, size=classifier_output.size()[2:],mode='bilinear', align_corners=True)

        del inputs
        torch.cuda.empty_cache()
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
        if val_idx % 1 == 0:
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
        logging.info((np.diag(iou_acc) / (iou_acc.sum(axis=1) + iou_acc.sum(axis=0) - np.diag(iou_acc))).mean())

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

    if args.gpu == 0:
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

