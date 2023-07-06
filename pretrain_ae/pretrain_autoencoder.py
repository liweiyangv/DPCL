import argparse
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
from PIL import Image
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
from autoencoder.cyc_autoencoder import AutoEncoder
from dataset.gtav_new_normalize_withiswprocess_nolabel import GTA5DataSet


import time
import torchvision
BATCH_SIZE = 2
ITER_SIZE = 1
NUM_WORKERS = 4

DATA_DIRECTORY = '/public/home/yangliwei/dataset/gtav/GTAV/images/train/folder/'
DATA_LIST_PATH = './dataset/text/gtav_split_train.txt'

IGNORE_LABEL = 255
INPUT_SIZE = '1280,720'
# INPUT_SIZE = '1024,512'

INPUT_SIZE_TARGET = '1024,512'
LEARNING_RATE = 2.5e-4
transfer_learning_rate=1e-4
MOMENTUM = 0.9
NUM_CLASSES = 19
NUM_STEPS = 300000
NUM_STEPS_STOP = 120000  # early stopping
POWER = 0.9
RANDOM_SEED = 304

SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 2500

IMG_MEAN_TORCH = torch.tensor([0.5, 0.5,0.5],dtype=torch.float32,device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

SNAPSHOT_DIR = '/public/home/yangliwei/DG_project/snapshots/ae/alter_mean_var/2_undersampling_alter_2023_check/'
VISUAL_RESULT = '/public/home/yangliwei/DG_project/result/ae/alter_mean_var/2_undersampling_alter_2023_check/'
PRETRAINED_AUTOENCODER='/public/home/yangliwei/DG_project/snapshots/ae/alter_mean_var/2_undersampling/isw_base/GTA5_only_transfer_105000.pth'
WEIGHT_DECAY = 0.0005
RESTORE_FROM_EXIST =None
LEARNING_RATE_D = 2.5e-4
LAMBDA_SEG = 0.0

SET = 'train'



def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="pretrain ae")
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
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    # parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
    #                     help="Path to the directory containing the target dataset.")
    # parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
    #                     help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")

    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
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
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--visual_result",type=str,default=VISUAL_RESULT,help="where to save the visual result")
    parser.add_argument("--snapshot_dir",type=str,default=SNAPSHOT_DIR,help="Where to save snapshots of the model.")

    parser.add_argument("--pretrained_autoencoder",type=str,default=PRETRAINED_AUTOENCODER,help="the path to load autoencoder")
    parser.add_argument("--transfer_learning_rate",type=float,default=transfer_learning_rate,help="the learning rate of the autoencoder")
    parser.add_argument("--continue_train",type=bool,default=False,help="whether to continue train the model")
    parser.add_argument("--begin_iter",type=int,default=0,help="the beginning iter ")
    parser.add_argument("--restore_from_exist", type=str, default=RESTORE_FROM_EXIST,help="Where restore model parameters from.")
    parser.add_argument("--use_other_augmentation", type=bool, default=False,help="Whether using other augmentations except change the contrast and brightness.")

    parser.add_argument('--lr_schedule', type=str, default='poly',help='name of lr schedule: poly')
    parser.add_argument('--max_iter', type=int, default=300000)
    parser.add_argument('--poly_exp', type=float, default=0.9,help='polynomial LR exponent')
    parser.add_argument('--crop_nopad', action='store_true', default=False)
    parser.add_argument('--wt_layer', nargs='*', type=int, default=[0, 0, 0, 0, 0, 0, 0],
                        help='0: None, 1: IW/IRW, 2: ISW, 3: IS, 4: IN (IBNNet: 0 0 4 4 4 0 0)')

    return parser.parse_args()


args = get_arguments()




def main():
    """Create the model and start the training."""
    w, h = map(int, args.input_size.split(','))  # format is  '1280,720'
    input_size = (w, h)

    # w, h = map(int, args.input_size_target.split(','))
    # input_size_target = (w, h)
    random_seed = args.random_seed #304
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    gpu = args.gpu
    criterion_entropy = nn.CrossEntropyLoss(weight=None, reduction='mean',
                                    ignore_index=255).cuda()
    #if args.continue_train:
      #   print('continue train loading ---------------------------------------')

        # model_transfer = AutoEncoder()
        # model_transfer.load_state_dict(torch.load(args.pretrained_autoencoder))
        # model_transfer.train()
        # model_transfer.cuda(args.gpu)
         #optimizer_transfer = optim.Adam(model_transfer.parameters(),lr=args.transfer_learning_rate,  weight_decay=0.0005)
         #optimizer_transfer.load_state_dict(torch.load('/public/home/yangliwei/DG_project/snapshots/ae/alter_mean_var/2_undersampling/isw_base/GTA5_optimizer_transfer_105000.pth'))

    #else:
    print('with out pretrainded')
    model_transfer = AutoEncoder()
       #model_transfer.load_state_dict(torch.load(args.pretrained_autoencoder))
    model_transfer.train()
        # model_transfer.eval()   # fixed the autoencoder
    model_transfer.cuda(args.gpu)    
    optimizer_transfer = optim.Adam(model_transfer.parameters(),lr=args.transfer_learning_rate,  weight_decay=0.0005)

    # Get computing device

 # load the pretrained autoencoder for iter 100000


    print('pretrained_ae',args.pretrained_autoencoder)
    print('snapshot_path',args.snapshot_dir)
    print('visual_result',args.visual_result)

    # model_transfer.eval()   # fixed the autoencoder

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    if not os.path.exists(args.visual_result):
        os.makedirs(args.visual_result)
    if not os.path.exists(args.visual_result+'/recon/'):
        os.makedirs(args.visual_result+'/recon/')

    criterion = nn.L1Loss()
    trainloader = data.DataLoader(
        GTA5DataSet(args.data_dir, args.data_list, max_iters=args.max_iter,
                    crop_size=input_size,
                    scale=args.random_scale, mirror=args.random_mirror),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    trainloader_iter = enumerate(trainloader)


    optimizer_transfer.zero_grad()

    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear',align_corners=True)

    for param in model_transfer.parameters():
      param.requires_grad = True
      
    for i_iter in range(len(trainloader)):

        # seg loss

        # ae loss
        _, batch = trainloader_iter.__next__()
            # images_1, labels, _, _, _, _, images_cbc,image_guass_noise = batch
        images_1, _, _, image_transform = batch
        images_1 = images_1.cuda(args.gpu)
        image_transform = image_transform.cuda(args.gpu)
        image_transform_encode = model_transfer.encoder(image_transform)
        image_transform_encode_mean = torch.mean(image_transform_encode,dim=[2,3],keepdim=True)
        image_transform_encode_var = torch.var(image_transform_encode,dim=[2,3],keepdim=True)
        image_transform_normalize = (image_transform_encode - image_transform_encode_mean) / torch.sqrt(image_transform_encode_var)

        ori_image_encode = model_transfer.encoder(images_1)
        ori_image_encode_mean = torch.mean(ori_image_encode,dim=[2,3],keepdim=True)
        ori_image_encode_var = torch.var(ori_image_encode,dim=[2,3],keepdim=True)
        
        recon_ori_feature = image_transform_normalize * torch.sqrt(ori_image_encode_var) + ori_image_encode_mean
        recon_ori = model_transfer.decoder(recon_ori_feature)
        #recon_ori = model_transfer.decoder(image_transform_encode)
        loss_ae = criterion(recon_ori,images_1)
        optimizer_transfer.zero_grad()
        loss_ae.backward()
        optimizer_transfer.step()
        if (i_iter + 1) % 500 == 0:
            vutils.save_image(image_transform, '%s/aeinput_%s.png' % (args.visual_result, str(i_iter + 1)),normalize=True)
            vutils.save_image(recon_ori, '%s/recon/input_recon_ori_%s.png'%(args.visual_result,str(i_iter + 1)), normalize=True)
            vutils.save_image(images_1, '%s/ori_%s.png'%(args.visual_result,str(i_iter + 1)), normalize=True)
        if (i_iter + 1) % 500 == 0 :
            test_file = open('%s/loss_file.txt'%(args.visual_result),'a')
            test_file.writelines('iter:  ' + str(i_iter + 1) + '\n')
            test_file.writelines('loss_ae:' + str(loss_ae) +'\n')
            test_file.close()
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                      'iter = {0:8d}/{1:8d} '
                      ' loss_recon_ori = {2:.3f} '.format(
                          i_iter, args.num_steps,loss_ae))

        if i_iter >= args.num_steps_stop - 1:
            print
            'save model ...'
            torch.save(model_transfer.state_dict(), osp.join(args.snapshot_dir, 'GTA5_only_transfer_' + str(args.num_steps_stop) + '.pth'))
            torch.save(optimizer_transfer.state_dict(), osp.join(args.snapshot_dir, 'GTA5_optimizer_transfer_' + str(args.num_steps_stop) + '.pth'))

            break
        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print
            'taking snapshot ...'
            torch.save(model_transfer.state_dict(), osp.join(args.snapshot_dir, 'GTA5_only_transfer_' + str(i_iter) + '.pth'))
            torch.save(optimizer_transfer.state_dict(), osp.join(args.snapshot_dir, 'GTA5_optimizer_transfer_' + str(i_iter) + '.pth'))




if __name__ == '__main__':
    main()