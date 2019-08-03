# a set of tools used to visualize features and 
# inner layers of resnet and transformer
import torch
import torchvision
import numpy as np
import cv2

import pdb

# inputs are torch tensors on CPU
def cam(raw_imgs, feature_maps, proj_mat, target_ids):
    # raw_imgs: B x 3 x W_i x H_i
    # feature_maps: B x C x W x H
    # proj_mat: C x N(num_class)
    # target_ids: B
    # reorder feature maps 
    B, C, W, H = feature_maps.size() # destructuring
    W_i, H_i = raw_imgs.size(2), raw_imgs.size(3)
    # B x W*H x C
    feature_maps = feature_maps.permute(0, 2, 3, 1).view(B, -1, C)

    # select out target class proj features
    # B x C x N
    target_proj_feats = proj_mat.unsqueeze(0).repeat(B, 1, 1)
    # B x C x 1
    target_ids = target_ids.unsqueeze(1).unsqueeze(1).repeat(1, C, 1)
    target_proj_feats = target_proj_feats.gather(2, target_ids)

    # calculate class activation 
    # B x W*H x 1
    activations = torch.bmm(feature_maps, target_proj_feats)
    # B x W x H
    activations = activations.squeeze(-1).view(B, W, H)
    # [B x overlay_imgs(w x h x 3)]
    outputs = []
    for idx in range(B):
        cam = activations[idx].detach().numpy()
        # normalize
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        cam_img = np.uint8(255 * cam)
        cam_img = cv2.resize(cam_img, (W_i, H_i))
        heatmap = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
        # W_i x H_i x 3
        img = raw_imgs[idx].permute(1, 2, 0).numpy()
        overlay = img * 1
        outputs.append(overlay)

    return outputs

