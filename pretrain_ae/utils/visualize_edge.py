import os
import cv2
import numpy as np
import torch



def image_normalization(img, img_min=0, img_max=255,
                        epsilon=1e-12):
    """This is a typical image normalization function
    where the minimum and maximum of the image is needed
    source: https://en.wikipedia.org/wiki/Normalization_(image_processing)

    :param img: an image could be gray scale or color
    :param img_min:  for default is 0
    :param img_max: for default is 255

    :return: a normalized image, if max is 255 the dtype is uint8
    """

    img = np.float32(img)
    # whenever an inconsistent image
    img = (img - np.min(img)) * (img_max - img_min) / \
        ((np.max(img) - np.min(img)) + epsilon) + img_min
    return img




def save_image_batch_to_disk(tensor, output_dir, file_name, is_inchannel=False):

    os.makedirs(output_dir, exist_ok=True)
    if is_inchannel:

        tensor, tensor2 = tensor
        fuse_name = 'fusedCH'
        av_name = 'avgCH'
        is_2tensors = True
        edge_maps2 = []
        for i in tensor2:
            tmp = torch.sigmoid(i).cpu().detach().numpy()
            edge_maps2.append(tmp)
        tensor2 = np.array(edge_maps2)
    else:
        fuse_name = 'fused'
        av_name = 'avg'
        tensor2 = None
        tmp_img2 = None

    output_dir_f = os.path.join(output_dir, fuse_name)
    output_dir_a = os.path.join(output_dir, av_name)
    os.makedirs(output_dir_f, exist_ok=True)
    os.makedirs(output_dir_a, exist_ok=True)

    # 255.0 * (1.0 - em_a)
    edge_maps = []
    for i in tensor:
        tmp = torch.sigmoid(i).cpu().detach().numpy()
        edge_maps.append(tmp)
    tensor = np.array(edge_maps)
    # print(f"tensor shape: {tensor.shape}")

    idx = 0
    i_shape = (1280,720)

    tmp = tensor[:, idx, ...]
    tmp2 = tensor2[:, idx, ...] if tensor2 is not None else None
    # tmp = np.transpose(np.squeeze(tmp), [0, 1, 2])
    tmp = np.squeeze(tmp)
    tmp2 = np.squeeze(tmp2) if tensor2 is not None else None

    # Iterate our all 7 NN outputs for a particular image
    preds = []
    for i in range(tmp.shape[0]):
        tmp_img = tmp[i]
        tmp_img = np.uint8(image_normalization(tmp_img))
        tmp_img = cv2.bitwise_not(tmp_img)
        # tmp_img[tmp_img < 0.0] = 0.0
        # tmp_img = 255.0 * (1.0 - tmp_img)
        if tmp2 is not None:
            tmp_img2 = tmp2[i]
            tmp_img2 = np.uint8(image_normalization(tmp_img2))
            tmp_img2 = cv2.bitwise_not(tmp_img2)

        # Resize prediction to match input image size
        if not tmp_img.shape[1] == i_shape[0] or not tmp_img.shape[0] == i_shape[1]:
            tmp_img = cv2.resize(tmp_img, (i_shape[0], i_shape[1]))
            tmp_img2 = cv2.resize(tmp_img2, (i_shape[0], i_shape[1])) if tmp2 is not None else None

        if tmp2 is not None:
            tmp_mask = np.logical_and(tmp_img > 128, tmp_img2 < 128)
            tmp_img = np.where(tmp_mask, tmp_img2, tmp_img)
            preds.append(tmp_img)

        else:
            preds.append(tmp_img)

        if i == 6:
            fuse = tmp_img
            fuse = fuse.astype(np.uint8)
            if tmp_img2 is not None:
                fuse2 = tmp_img2
                fuse2 = fuse2.astype(np.uint8)
                # fuse = fuse-fuse2
                fuse_mask = np.logical_and(fuse > 128, fuse2 < 128)
                fuse = np.where(fuse_mask, fuse2, fuse)

                # print(fuse.shape, fuse_mask.shape)

    # Get the mean prediction of all the 7 outputs
    average = np.array(preds, dtype=np.float32)
    average = np.uint8(np.mean(average, axis=0))
    output_file_name_f = os.path.join(output_dir_f, file_name)
    output_file_name_a = os.path.join(output_dir_a, file_name)
    cv2.imwrite(output_file_name_f, fuse)
    cv2.imwrite(output_file_name_a, average)

    idx += 1
# def save_image_batch_to_disk(tensor, output_dir, file_names, img_shape=None, is_inchannel=False):
#
#     os.makedirs(output_dir, exist_ok=True)
#     if is_inchannel:
#
#         tensor, tensor2 = tensor
#         fuse_name = 'fusedCH'
#         av_name = 'avgCH'
#         is_2tensors = True
#         edge_maps2 = []
#         for i in tensor2:
#             tmp = torch.sigmoid(i).cpu().detach().numpy()
#             edge_maps2.append(tmp)
#         tensor2 = np.array(edge_maps2)
#     else:
#         fuse_name = 'fused'
#         av_name = 'avg'
#         tensor2 = None
#         tmp_img2 = None
#
#     output_dir_f = os.path.join(output_dir, fuse_name)
#     output_dir_a = os.path.join(output_dir, av_name)
#     os.makedirs(output_dir_f, exist_ok=True)
#     os.makedirs(output_dir_a, exist_ok=True)
#
#     # 255.0 * (1.0 - em_a)
#     edge_maps = []
#     for i in tensor:
#         tmp = torch.sigmoid(i).cpu().detach().numpy()
#         edge_maps.append(tmp)
#     tensor = np.array(edge_maps)
#     # print(f"tensor shape: {tensor.shape}")
#
#     image_shape = [x.cpu().detach().numpy() for x in img_shape]
#     # (H, W) -> (W, H)
#     image_shape = [[y, x] for x, y in zip(image_shape[0], image_shape[1])]
#
#     assert len(image_shape) == len(file_names)
#
#     idx = 0
#     for i_shape, file_name in zip(image_shape, file_names):
#         tmp = tensor[:, idx, ...]
#         tmp2 = tensor2[:, idx, ...] if tensor2 is not None else None
#         # tmp = np.transpose(np.squeeze(tmp), [0, 1, 2])
#         tmp = np.squeeze(tmp)
#         tmp2 = np.squeeze(tmp2) if tensor2 is not None else None
#
#         # Iterate our all 7 NN outputs for a particular image
#         preds = []
#         for i in range(tmp.shape[0]):
#             tmp_img = tmp[i]
#             tmp_img = np.uint8(image_normalization(tmp_img))
#             tmp_img = cv2.bitwise_not(tmp_img)
#             # tmp_img[tmp_img < 0.0] = 0.0
#             # tmp_img = 255.0 * (1.0 - tmp_img)
#             if tmp2 is not None:
#                 tmp_img2 = tmp2[i]
#                 tmp_img2 = np.uint8(image_normalization(tmp_img2))
#                 tmp_img2 = cv2.bitwise_not(tmp_img2)
#
#             # Resize prediction to match input image size
#             if not tmp_img.shape[1] == i_shape[0] or not tmp_img.shape[0] == i_shape[1]:
#                 tmp_img = cv2.resize(tmp_img, (i_shape[0], i_shape[1]))
#                 tmp_img2 = cv2.resize(tmp_img2, (i_shape[0], i_shape[1])) if tmp2 is not None else None
#
#             if tmp2 is not None:
#                 tmp_mask = np.logical_and(tmp_img > 128, tmp_img2 < 128)
#                 tmp_img = np.where(tmp_mask, tmp_img2, tmp_img)
#                 preds.append(tmp_img)
#
#             else:
#                 preds.append(tmp_img)
#
#             if i == 6:
#                 fuse = tmp_img
#                 fuse = fuse.astype(np.uint8)
#                 if tmp_img2 is not None:
#                     fuse2 = tmp_img2
#                     fuse2 = fuse2.astype(np.uint8)
#                     # fuse = fuse-fuse2
#                     fuse_mask = np.logical_and(fuse > 128, fuse2 < 128)
#                     fuse = np.where(fuse_mask, fuse2, fuse)
#
#                     # print(fuse.shape, fuse_mask.shape)
#
#         # Get the mean prediction of all the 7 outputs
#         average = np.array(preds, dtype=np.float32)
#         average = np.uint8(np.mean(average, axis=0))
#         output_file_name_f = os.path.join(output_dir_f, file_name)
#         output_file_name_a = os.path.join(output_dir_a, file_name)
#         cv2.imwrite(output_file_name_f, fuse)
#         cv2.imwrite(output_file_name_a, average)
#
#         idx += 1

