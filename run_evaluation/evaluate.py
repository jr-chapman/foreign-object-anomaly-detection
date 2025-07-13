# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# MAE Medical Anomaly Detection - https://github.com/lilygeorgescu/MAE-medical-anomaly-detection?tab=readme-ov-file
# --------------------------------------------------------

import os 
import torch
import json
import numpy as np
from PIL import Image

from skimage.metrics import structural_similarity as ssim

import models_mae
import run_evaluation.generate_masks as generate_masks
import run_evaluation.calculate_metrics as calculate_metrics
import run_evaluation.calculate_threshold as calculate_threshold

def get_threshold():
    try:
        file_path = os.path.join("config", "threshold.json")
        with open(file_path, "r") as f: 
            data = json.load(f)
        return data["threshold"]
    except Exception as e:
        raise RuntimeError("Threshold not set. Run threshold generation before evaluation.") from e

def prepare_model(model_path, arch='mae_vit_base_patch16'):
    img_size =224
    patch_size = 16 
    model = getattr(models_mae, arch)(img_size=img_size, patch_size=patch_size)
    checkpoint = torch.load(model_path, weights_only=False)
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    model = model.to('cuda')
    model.eval()
    print("Model successfully loaded")
    return model

def save_outputs(img, filename, folder): 
    img_file = os.path.join(folder, filename)
    Image.fromarray(img).save(img_file)

class MAEEvaluator: 
    def __init__(self, logger, error, mask_ratio, num_trials, output_dir): 
        self.logger = logger
        self.error = error
        self.mask_ratio = mask_ratio
        self.num_trials = num_trials
        self.output_dir = output_dir

    def calculate_ssim(self, imgs_, reconstructions, ground_truth_labels): 
        ssim_normal = []
        ssim_abnormal = []
        for target_, recon_, label in zip(imgs_, reconstructions, ground_truth_labels):
            target_range = target_[:,:,0].max() - target_[:,:,0].min()
            if target_range == 0.0: 
                target_range = 1.0 
            err = -ssim(target_[:, :, 0], recon_[:, :, 0], data_range=target_range)
            if label == 1:
                    ssim_abnormal.append(err)
            else:
                ssim_normal.append(err)
        return ssim_normal, ssim_abnormal 
    
    def get_anomaly_score(self, normalised_differences, ground_truth_labels): 
        anomaly_score_normal = []
        anomaly_score_abnormal = []
        for difference, label in zip(normalised_differences, ground_truth_labels): 
            if label == 1: 
                anomaly_score_abnormal.append(np.mean(difference))
            else: 
                anomaly_score_normal.append(np.mean(difference))
        return anomaly_score_normal, anomaly_score_abnormal 

    def evaluate_difference(self, model, images, filenames, ground_truth_labels, mode="complete", img_folder=None, mask_folder=None, use_validation=True):
        ground_truth_labels = ground_truth_labels.numpy()
        imgs_ = [torch.einsum('chw->hwc', img).numpy() for img in images]
        imgs_ = np.array(imgs_, np.float32)

        reconstructions = generate_masks.get_reconstructions_multi(model, imgs_, self.num_trials, self.mask_ratio, use_validation)

        mask_score_total = []
        mask_score_abnormal = []
        mask_score_normal = []
        differences = []
        for (img_, reconstruction_, gt_label_, filename_) in zip(imgs_, reconstructions, ground_truth_labels, filenames):
            recon_ = np.clip(reconstruction_, 0.0, 1.0) 
            diff_ = np.abs(img_-recon_)
            differences.append(diff_)
            diff_norm = (diff_ - np.min(diff_)) / (np.max(diff_) - np.min(diff_))
            diff_img = (diff_norm.squeeze()*255).astype(np.uint8) 
            processed_ = generate_masks.postprocess_image(diff_img)
            mask_ = generate_masks.create_mask(processed_)
            mask_score = np.mean(mask_)
            mask_score_total.append(mask_score)

            if mode=="threshold": 
                continue

            #generate_masks.visualise(img_, recon_, diff_norm, processed_, mask_)
            save_outputs((img_.squeeze()*255).astype(np.uint8), filename_, img_folder)
            save_outputs((mask_*255).astype(np.uint8), filename_, mask_folder)
            if gt_label_ == 1: #as validation sets contain normal and abnormal images, this check has to be added 
                mask_score_abnormal.append(mask_score)
            else: 
                mask_score_normal.append(mask_score)
        
        if mode == "threshold": 
            return mask_score_total
        
        results = {
            'mask_score_normal': mask_score_normal, 
            'mask_score_abnormal': mask_score_abnormal, 
            'mask_score_total': mask_score_total
        }

        if self.error: 
            if 'ssim' in self.error: 
                ssim_normal, ssim_abnormal = self.calculate_ssim(imgs_, reconstructions, ground_truth_labels)
                results['ssim_score_normal'] = ssim_normal
                results['ssim_score_abnormal'] = ssim_abnormal 
            
            if 'anomaly' in self.error: 
                anomaly_score_normal, anomaly_score_abnormal = self.get_anomaly_score(differences, ground_truth_labels) 
                results['anomaly_score_normal'] = anomaly_score_normal
                results['anomaly_score_abnormal'] = anomaly_score_abnormal

            if not any(error in self.error for error in ['ssim', 'anomaly']):
                self.logger.error(f"error {self.error} not supported!")
                raise ValueError(f"error {self.error} not supported!")
  
        return results

    def evaluate_dataloader(self, model, dataloader, img_folder, mask_folder): 
        result_keys = ["anomaly_score_normal", "anomaly_score_abnormal", 
                       "mask_score_normal", "mask_score_abnormal", 
                       "ssim_score_normal", "ssim_score_abnormal", 
                       "mask_score_total"]
        metrics = {key: [] for key in result_keys}
        metrics['labels'] = []
        for data_iter_step, (samples, labels, filenames) in enumerate(dataloader): 
            print(f"Evaluating batch {data_iter_step} with {len(samples)} images")
            batch_results = self.evaluate_difference(model, samples, filenames, labels, img_folder=img_folder, mask_folder=mask_folder)
            for key in result_keys: 
                if key in batch_results: 
                    metrics[key].extend(batch_results[key])
            metrics['labels'].extend(labels.numpy())
        return metrics


    def run_evaluation(self, model, dataloader): 
        img_folder= os.path.join(self.output_dir, 'images/')
        mask_folder = os.path.join(self.output_dir, 'masks/')
        os.makedirs(img_folder, exist_ok=True)
        os.makedirs(mask_folder, exist_ok=True)

        if isinstance(model, str): 
            model = prepare_model(model)

        results = self.evaluate_dataloader(model, dataloader=dataloader, img_folder=img_folder, mask_folder=mask_folder)
        THRESHOLD = get_threshold()
        predictions = results['mask_score_total']
        label_predictions = (np.array(predictions) > THRESHOLD).astype(np.uint8)
        calculate_metrics.calculate_metrics(self.logger, predictions, label_predictions, results) 

        print("Model evaluation complete")

    def generate_threshold(self, model, dataloader): 
        if isinstance(model, str): 
            model = prepare_model(model)

        mask_scores = []
        for data_iter_step, (samples, labels, filenames) in enumerate(dataloader): 
            print(f"Generating masks for threshold calculation. Batch {data_iter_step}")
            batch_mask_scores = self.evaluate_difference(model, images=samples, filenames=filenames, ground_truth_labels=labels, mode="threshold")
            mask_scores.extend(batch_mask_scores)
        calculate_threshold.calculate_threshold(self.logger, mask_scores)
