# a set of tools used to visualize features and 
# inner layers of resnet and transformer
import sys
import time
import torch
import torchvision
import numpy as np
import cv2

import pdb

# Inputs
# model: pre-trained model for image classification on GPU 
# model_layer: any layer whose output's shape follows B x C x W x H
# dataloader: dataloader of images
# Output 
# for each filter, top k images that activate the filter the most
# with the activation heatmap overlaid on top
# Example 
# activated_imgs = vis.top_k_activated_images(model, model.res_layers[-1][-1].conv1, val_loader, K=3)

def top_k_activated_images(model, model_layer,
    dataloader, K=5):
    activations = [None]
    def hook_feature(module, input, output):
        activations[0] = output
    model_layer.register_forward_hook(hook_feature)
    top_k_stats = None

    model.eval()
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        start = time.time()
        inputs = inputs.cuda()
        targets = targets.cuda()
        outputs = model(inputs)
        # make sure that layer is indeed from the model
        # otherwise, activations[0] won't be assigned
        assert activations[0] is not None
        B, C, W, H = activations[0].size()
        pdb.set_trace()
        if top_k_stats is None:
            # initialize top_k_stats based on the number of output channels
            # [C x [K(unordered) x [activation_map, image]]]
            top_k_stats = [[] for _ in range(C)]
        inputs = inputs.permute(0, 2, 3, 1).cpu().numpy()
        acts = activations[0].detach().cpu().numpy()
        for b in range(B):
            img = inputs[b]
            for c in range(C):
                act = acts[b, c]
                curr_top_k = top_k_stats[c]
                if len(curr_top_k) < K:                    
                    curr_top_k.append([act, img])
                else:
                    # get the min activated value and idx from current top k choices
                    min_i = -1
                    min_v = sys.maxsize
                    for i in range(len(curr_top_k)):
                        if curr_top_k[i][0].max() < min_v:
                            min_v = curr_top_k[i][0].max()
                            min_i = i
                    if act.max() > min_v:
                        curr_top_k[min_i] = [act, img]
        delta = time.time() - start
        print('batch time: %d' % delta)

    # overlay receptive field onto image
    # [num_filter x [K x overlay_image]]
    outputs = []
    for f in range(len(top_k_stats)):
        filter_stats = top_k_stats[f]
        filter_output = []
        for k in range(len(filter_stats)):
            # filter out inactivated regions using relu
            act = _normalize(_relu(filter_stats[k][0]))
            img = _normalize(filter_stats[k][1])
            act = cv2.resize(act, (img.shape[0], img.shape[1]))
            act = cv2.applyColorMap(act, cv2.COLORMAP_JET)
            overlay = img * 0.5 + act * 0.3
            filter_output.append(overlay)
        outputs.append(filter_output)
    
    return outputs

# useful for visualizing filters at low layers
def conv_templates(conv_weights, size=64):
    # conv_weights: C x W x H
    conv_weights = conv_weights.numpy()
    templates = _normalize(conv_weights)
    return templates


# To use cam, define global variables proj_weight and feature_blobs and register
# a hook as follows:
    # def hook_feature(module, input, output):
    #     feature_blobs.append(input[0])
    # model._modules.get('avg_pool').register_forward_hook(hook_feature)
    # proj_weight = model._modules.get('projection').weight.transpose(0, 1)
    
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
        cam_img = _normalize(cam)
        cam_img = cv2.resize(cam_img, (W_i, H_i))
        heatmap = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
        # W_i x H_i x 3
        img = raw_imgs[idx].permute(1, 2, 0).numpy()
        img = _normalize(img)
        overlay = img * 0.5 + heatmap * 0.3
        outputs.append(overlay)

    return outputs

def _normalize(img):
    img = img - np.min(img)
    img = img / np.max(img)
    img = np.uint8(255 * img)
    return img

def _relu(img):
    img[img < 0 ] = 0
    return img
