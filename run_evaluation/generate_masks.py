# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# MAE Medical Anomaly Detection - https://github.com/lilygeorgescu/MAE-medical-anomaly-detection?tab=readme-ov-file
# --------------------------------------------------------

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

def get_reconstructions(model_, imgs_, idx, mask_ratio, use_validation):
    x = torch.tensor(imgs_)

    x = torch.einsum('nhwc->nchw', x) 
    x = x.to('cuda')
    _, result, mask = model_(x.float(), mask_ratio=mask_ratio, idx_masking=idx, is_testing=use_validation)
    result = model_.unpatchify(result)
    result = torch.einsum('nchw->nhwc', result).detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model_.patch_embed.patch_size[0]**2 * 1)  # (N, H*W, p*p*3)
    mask = model_.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

    # MAE reconstruction pasted with visible patches
    im_paste = torch.einsum('nchw->nhwc', x).detach().cpu() * (1 - mask) + result * mask

    return im_paste.numpy()

def get_reconstructions_multi(model_, imgs_, num_trials, mask_ratio, use_validation):
    results = None
    for idx in range(num_trials):
        result = get_reconstructions(model_, imgs_, idx, mask_ratio, use_validation)
        if results is None:
            results = result
        else:
            results += result

    results = results / num_trials
    return results

def postprocess_image(input_img): 
    assert np.min(input_img) >= 0 and np.max(input_img) <= 255
    assert type(input_img).__module__ == np.__name__
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))
    contrast_img = clahe.apply(input_img)
    denoised_img = cv2.fastNlMeansDenoising(contrast_img, None, 5,7,21)
    return denoised_img 

def connected_components(img, min_area=10): 
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)
    refined_mask = np.zeros_like(img)
    for i in range(1, num_labels):  
        area = stats[i, cv2.CC_STAT_AREA]
        if area > min_area: 
            refined_mask[labels==i] = 1
    return refined_mask

def create_mask(input_img, percentile=95):
    percent = np.percentile(input_img, percentile) 
    _, thresholded_img = cv2.threshold(input_img, percent, 255, cv2.THRESH_BINARY)
    masked_img = connected_components(thresholded_img)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    closed_mask = cv2.morphologyEx(masked_img, cv2.MORPH_CLOSE, kernel, iterations=1)
    return closed_mask


def visualise(image, reconstruction, difference, processed, mask): 
    image_list = [image, reconstruction, difference, processed, mask]
    title_list = ["Input", "Reconstruction", "abs(Input - Reconstruction)", "Post-processed difference", "Generated Mask"]
    for idx, (img, title) in enumerate(zip(image_list, title_list)): 
        plt.subplot(1, 5, idx+1)
        plt.imshow(img, cmap='gray')
        plt.title(title, fontsize=28)
        plt.axis('off')
    plt.show()
