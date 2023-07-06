import torch
import numpy as np
from kmeans import KMeans as fast_k_means
import torch.nn.functional as F
import math
import time
####use local context normalization
def local_normalization(input,channels_per_group=1,window_size=(3,3),eps=1e-05):
    x = input.clone()
    N, C, H, W = x.size()
    G = C // channels_per_group
    assert C % channels_per_group == 0
    if window_size[0] < H and window_size[1] < W:
        # Build integral image
        device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else 'cpu')
        x_squared = x * x
        integral_img = x.cumsum(dim=2).cumsum(dim=3)
        integral_img_sq = x_squared.cumsum(dim=2).cumsum(dim=3)
        # Dilation
        d = (1, window_size[0], window_size[1])
        integral_img = torch.unsqueeze(integral_img, dim=1)
        integral_img_sq = torch.unsqueeze(integral_img_sq, dim=1)
        kernel = torch.tensor([[[[[1., -1.], [-1., 1.]]]]]).to(device)
        c_kernel = torch.ones((1, 1, channels_per_group, 1, 1)).to(device)
        with torch.no_grad():
            # Dilated conv
            sums = F.conv3d(integral_img, kernel, stride=[1, 1, 1], dilation=d)
            sums = F.conv3d(sums, c_kernel, stride=[channels_per_group, 1, 1])
            squares = F.conv3d(integral_img_sq, kernel, stride=[1, 1, 1], dilation=d)
            squares = F.conv3d(squares, c_kernel, stride=[channels_per_group, 1, 1])
        n = window_size[0] * window_size[1] * channels_per_group
        means = torch.squeeze(sums / n, dim=1)
        var = torch.squeeze(1.0 / n * torch.abs(squares - sums * sums / n), dim=1)
        # for i in np.arange(a.shape[3]):
        #     for j in np.arange(a.shape[4]):
        #         if a[0,0,0,i,j]<0:
        #
        #             print(i,j)
        #             print('mean----------------')
        #
        #             print(torch.sum(x[0, 0, 1:4, 1:4]))
        #             print(integral_img[0, 0,0, 3, 3] + integral_img[0, 0,0, 0, 0] - integral_img[0, 0, 0,0, 3] -
        #                   integral_img[0, 0,0, 3, 0])
        #             print(sums[0,0,0,0,0])
        #             print('var----------------')
        #
        #
        #             print(torch.sum(x_squared[0, 0, 3:6, 92:95]))
        #             print(integral_img_sq[0, 0,0, 5, 94] + integral_img_sq[0, 0,0, 2, 91] - integral_img_sq[0, 0, 0,5, 91] -
        #                   integral_img_sq[0, 0,0, 2, 94])
        #
        #             print(squares[0,0,0,i,j])
        #             print(sums[0,0,0,i,j]*sums[0,0,0,i,j]/n)
        #             print('ok')
        #             break
        _, _, h, w = means.size()
        pad2d = (int(math.floor((W - w) / 2)), int(math.ceil((W - w) / 2)), int(math.floor((H - h) / 2)),
                 int(math.ceil((H - h) / 2)))
        padded_means = F.pad(means, pad2d, 'replicate')
        padded_vars = F.pad(var, pad2d, 'replicate') + eps
        old = x.clone()
        for i in range(G):
            x[:, i * channels_per_group:i * channels_per_group + channels_per_group, :, :] = \
                (x[:, i * channels_per_group:i * channels_per_group + channels_per_group, :, :] -
                 torch.unsqueeze(padded_means[:, i, :, :], dim=1).to(device)) /\
                torch.sqrt(((torch.unsqueeze(padded_vars[:, i, :, :], dim=1)).to(device)))
        del integral_img
        del integral_img_sq
    else:
        x = x.view(N, G, -1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)
        x = (x - mean) / (var + eps).sqrt()
        x = x.view(N, C, H, W)


    return x,padded_means,padded_vars

def  k_means_update_local_mean_var(cluster_num,input_data,mean_data,var_data,input_mean_centor=None,input_var_centor=None,input_centroids=None,eps=1e-5):

    input_data = input_data.t()
    mean_data = mean_data.t()
    var_data = var_data.t()
    fast_kmeans = fast_k_means(cluster_num,mode='euclidean',verbose=1)
    label,centor = fast_kmeans.predict(input_data,input_centroids)
    new_centroids = torch.zeros((cluster_num,input_data.shape[1])).cuda()
    mean_centor = torch.zeros((cluster_num,mean_data.shape[1])).cuda()
    var_centor = torch.zeros((cluster_num,var_data.shape[1])).cuda()
    for i in np.arange(cluster_num):
        if input_data[label == i, :].shape[0] <= 50 or mean_data[label == i,:].shape[0] <= 50 or var_data[label == i,:].shape[0] <= 50 :
            if input_centroids == None :
                print('cluster_%s has no data for inital'%str(i))
                new_centroids[i, :] = centor[i, :]
                var_centor[i,:] = var_centor[i,:] + eps

            else:
                # print('label_%s_do not update' % str(i))
                new_centroids[i, :] = input_centroids[i, :]
                mean_centor[i, :] = input_mean_centor[i, :]
                var_centor[i, :] = input_var_centor[i, :]
        else:
            new_centroids[i, :] = torch.sum(input_data[label == i, :], dim=0) / input_data[label == i, :].shape[0]
            mean_centor[i, :] = torch.sum(mean_data[label == i, :], dim=0) / mean_data[label == i, :].shape[0]

            var_centor[i, :] = torch.sum(var_data[label == i, :], dim=0) / var_data[label == i, :].shape[0]


    if input_centroids == None:
        print('input centor = none')
        new_centroids = 0.3 * centor + 0.7 * new_centroids
    return new_centroids,mean_centor,var_centor

def local_normal_slow_version(input_feature):
    x , y ,z =input_feature.shape
    # normalized_feature = torch.zeros(input_feature.shape)
    feature_mean = torch.zeros(input_feature.shape).cuda()
    feature_var = torch.zeros(input_feature.shape).cuda()
    a = torch.mean(input_feature[:,0:2,0:2],dim=[1,2])
    feature_mean[:,0,0] = torch.mean(input_feature[:,0:2,0:2],dim=[1,2])
    feature_var[:,0,0] = torch.var(input_feature[:,0:2,0:2],dim=[1,2])
    feature_mean[:,0,z-1] = torch.mean(input_feature[:,0:2,z-2:z],dim=[1,2])
    feature_var[:,0,z-1] = torch.var(input_feature[:,0:2,z-2:z],dim=[1,2])
    feature_mean[:,y-1,0] = torch.mean(input_feature[:,y-2:y,0:2],dim=[1,2])
    feature_var[:,y-1,0] = torch.var(input_feature[:,y-2:y,0:2],dim=[1,2])
    feature_mean[:,y-1,z-1] = torch.mean(input_feature[:,y-2:y,z-2:z],dim=[1,2])
    feature_var[:,y-1,z-1] = torch.var(input_feature[:,y-2:y,z-2:z],dim=[1,2])
    for i in np.arange(z-2):
        feature_mean[:,0,i+1] = torch.mean(input_feature[:,0:2,i:i+3],dim=[1,2])
        feature_var[:,0,i+1] = torch.var(input_feature[:,0:2,i:i+3],dim=[1,2])
        feature_mean[:,y-1,i+1] = torch.mean(input_feature[:,y-2:y,i:i+3],dim=[1,2])
        feature_var[:,y-1,i+1] = torch.var(input_feature[:,y-2:y,i:i+3],dim=[1,2])
        for j in np.arange(y-2):
            feature_mean[:,j+1,i+1] = torch.mean(input_feature[:,j:j+3,i:i+3],dim=[1,2])
            feature_var[:,j+1,i+1] = torch.var(input_feature[:,j:j+3,i:i+3],dim=[1,2])
    for i in np.arange(y-2):
        feature_mean[:,i+1,0] = torch.mean(input_feature[:,i:i+3,0:2],dim=[1,2])
        feature_var[:,i+1,0] = torch.var(input_feature[:,i:i+3,0:2],dim=[1,2])
        feature_mean[:,i+1,z-1] = torch.mean(input_feature[:,i:i+3,z-2:z+1],dim=[1,2])
        feature_var[:,i+1,z-1] = torch.var(input_feature[:,i:i+3,z-2:z],dim=[1,2])
    normalized_feature  = (input_feature-feature_mean)/ torch.sqrt(feature_var)
    return normalized_feature,feature_mean,feature_var

def ED(a,b):
    a = a.t()
    sq_a = a**2
    sq_b = b**2
    sum_sq_a = torch.sum(sq_a,dim=1).unsqueeze(1)
    sum_sq_b = torch.sum(sq_b,dim=1).unsqueeze(0)
    bt = b.t()
    return torch.sqrt(sum_sq_a+sum_sq_b-2*a.mm(bt))

def k_means_update(cluster_num,input_data,mean_var_data,input_mean_var_centor,input_centroids=None):

    input_data = input_data.t()
    device = input_centroids.device
    fast_kmeans = fast_k_means(cluster_num,mode='euclidean',verbose=1)
    label,centor = fast_kmeans.predict(input_data,input_centroids)
    new_centroids = torch.zeros((cluster_num,input_data.shape[1])).cuda(device)
    mean_var_centor = torch.zeros((cluster_num,mean_var_data.shape[0])).cuda(device)
    for i in np.arange(cluster_num):
        if input_data[label == i,:].shape[0] <= 10 or mean_var_data[:, label == i].shape[1] <= 10:
            new_centroids[i, :] = input_centroids[i, :]
            mean_var_centor[i, :] = input_mean_var_centor[i, :]
        else:
            new_centroids[i, :] = torch.sum(input_data[label == i, :], dim=0) / input_data[label == i, :].shape[0]
            mean_var_centor[i, :] = torch.sum(mean_var_data[:, label == i], dim=1) / mean_var_data[:, label == i].shape[1]

    if input_centroids == None:
        print('input centor = none')
        new_centroids = 0.7 * centor + 0.3 * new_centroids

    return new_centroids.cuda(device),mean_var_centor.cuda(device)

# def update_centor(trainloader,trainloader_iter,kmeans_centers,model_transfer,save_path,present_number):
#     print('begin update centor',len(trainloader))
#     target_path = save_path+'/'+str(present_number)
#     if not os.path.exists(target_path):
#         os.makedirs(target_path)
#     for i_iter in range(len(trainloader)):
#         print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 'begin update centor processing:', i_iter)
#
#         if i_iter % 100 ==0:
#             print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),'begin update centor processing:',i_iter)
#         _, batch = trainloader_iter.__next__()
#         images_1, _, name = batch
#         images_1 = Variable(images_1).cuda(args.gpu)
#         ori_encode = model_transfer.encoder(images_1)
#         ori_encode_mean = torch.mean(ori_encode,dim=[2,3],keepdim=True)
#         ori_encode_var = torch.var(ori_encode,dim=[2,3],keepdim=True)
#         ori_normalize = (ori_encode - ori_encode_mean) / torch.sqrt(ori_encode_var)
#         if i_iter % args.k_means_iter ==0:
#             if i_iter > 0: # use once k-means
#                 print('k-means processing')
#                 if i_iter == args.k_means_iter:
#                     kmeans_centers, label = k_means(args.cluster_number, model_cluster, kmeans_centers)
#                 else:
#                     old_kmeans_centers = kmeans_centers.clone()
#                     kmeans_centers, label = k_means(args.cluster_number, model_cluster, kmeans_centers)
#                     kmeans_centers = 0.1 * kmeans_centers + 0.9 * old_kmeans_centers
#                 # torch.save(kmeans_centers, osp.join(target_path, 'ori_normalize_feature_centor_%s'%(str(i_iter)) + '.pt'))
#                 # torch.save(label, osp.join(target_path, 'label_%s'%(str(i_iter)) + '.pt'))
#             model_cluster = ori_normalize.clone()
#         else:
#             model_cluster = torch.cat((model_cluster,ori_normalize),axis=0)
#     old_kmeans_centers = kmeans_centers.clone()
#     kmeans_centers, label = k_means(args.cluster_number, model_cluster, kmeans_centers)
#     kmeans_centers = 0.1 * kmeans_centers + 0.9 * old_kmeans_centers
#     torch.save(kmeans_centers, osp.join(target_path, 'ori_normalize_feature_centor_%s' % str(i_iter) + '.pt'))
#     # torch.save(label, osp.join(target_path, 'label_%s' % (i_iter) + '.pt'))
#     return kmeans_centers
#
# def find_centor(trainloader,trainloader_iter,kmeans_centers,model_transfer,save_path,present_number):
#     print('begin find centor',len(trainloader))
#     target_path = save_path+'/'+str(present_number)
#     if not os.path.exists(target_path):
#         os.makedirs(target_path)
#     for i_iter in np.arange(len(trainloader)):
#         _, batch = trainloader_iter.__next__()
#         if i_iter % 100 == 0:
#             print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 'begin find centor processing:', i_iter)
#         images_1, _, name = batch
#         images_1 = Variable(images_1).cuda(args.gpu)
#         ori_encode = model_transfer.encoder(images_1)
#         ori_encode_mean = torch.mean(ori_encode, dim=[2, 3], keepdim=True)
#         ori_encode_var = torch.var(ori_encode, dim=[2, 3], keepdim=True)
#         ori_encode_mean_squeeze = torch.squeeze(torch.squeeze(ori_encode_mean, axis=3), axis=2)
#         ori_encode_var_squeeze = torch.squeeze(torch.squeeze(ori_encode_var, axis=3), axis=2)
#         ori_normalize = (ori_encode - ori_encode_mean) / torch.sqrt(ori_encode_var)
#         feature_mean_var = torch.cat([ori_encode_mean_squeeze, ori_encode_var_squeeze], 1)
#         if i_iter % args.k_means_iter == 0:
#             if i_iter > 0:
#                 if i_iter == args.k_means_iter:
#                     print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 'calulate distance')
#                     distance, existed_feature, index = measure_distance(normalized_feature_cluster, kmeans_centers)
#                     mean_var_existed = normalized_feature_mean_var[index, :]
#                 else:
#                     print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 'calulate distance')
#                     new_distance, new_existed_feature, new_index = measure_distance(normalized_feature_cluster,
#                                                                                     kmeans_centers)
#                     new_mean_var_existed = normalized_feature_mean_var[new_index, :]
#                     for j in np.arange(len(new_distance)):
#                         if new_distance[j] < distance[j]:
#                             distance[j] = new_distance[j].clone()
#                             existed_feature[j] = new_existed_feature[j, :].clone()
#                             mean_var_existed[j] = new_mean_var_existed[j, :].clone()
#             normalized_feature_cluster = ori_normalize.clone()
#             normalized_feature_mean_var = feature_mean_var.clone()
#         else:
#             normalized_feature_cluster = torch.cat((normalized_feature_cluster, ori_normalize), axis=0)
#             normalized_feature_mean_var = torch.cat((normalized_feature_mean_var, feature_mean_var), axis=0)
#     print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 'The last calulate distance')
#     new_distance, new_existed_feature, new_index = measure_distance(normalized_feature_cluster, kmeans_centers)
#     new_mean_var_existed = normalized_feature_mean_var[new_index, :]
#     for j in np.arange(len(new_distance)):  # choose the min distance in all the data
#         if new_distance[j] < distance[j]:
#             distance[j] = new_distance[j].clone()
#             existed_feature[j] = new_existed_feature[j, :].clone()
#             mean_var_existed[j] = new_mean_var_existed[j, :].clone()
#     torch.save(existed_feature, osp.join(target_path, 'exist_normalize_feature_centor' + '.pt'))
#     torch.save(mean_var_existed, osp.join(target_path, 'exist_feature_centor_mean_var' + '.pt'))
#     return existed_feature,mean_var_existed