# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import numpy as np 
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def calculate_metrics(logger, predictions, label_predictions, results): 
    logger.info(64*"-")
    logger.info("Evaluation metrics")
    
    score_types = {
        "anomaly_score_normal": "Anomaly score for negative images", 
        "mask_score_normal": "Mask score for negative images", 
        "ssim_score_normal": "SSIM score for negative images"
    }

    for key, label in score_types.items(): 
        if key in results and len(results[key]) > 0: 
            logger.info(f"{label}: {np.mean(results[key])}")
    
    if any(results.get(key) for key in ['anomaly_score_abnormal', 'mask_score_abnormal', 'ssim_score_abnormal']): 
        logger.info("-" * 64)
        logger.info("Abnormal images present - logging metrics for abnormal images")

        anomalous_score_types = {
            "anomaly_score_abnormal": "Anomaly score for positive images", 
            "mask_score_abnormal": "Mask score for positive images", 
            "ssim_score_abnormal": "SSIM score for positive images"
        }
        for key, label in anomalous_score_types.items(): 
            if key in results and len(results[key]) > 0: 
                logger.info(f"{label}: {np.mean(results[key])}")
    
        logger.info("-" * 64)
        
        label_list = results.get('labels')

        metrics = {}

        metrics["accuracy"] = accuracy_score(label_list, label_predictions)
        metrics["precision"] = precision_score(label_list, label_predictions)
        metrics["recall"] = recall_score(label_list, label_predictions)
        metrics["f1"] = f1_score(label_list, label_predictions)
        metrics["conf_matrix"] = confusion_matrix(label_list, label_predictions)
        metrics["auc"] = roc_auc_score(label_list, predictions)

        for key, value in metrics.items(): 
            if key!="conf_matrix": 
                logger.info(f"{key.capitalize()}: {value}")
            else: 
                logger.info(f"Confusion Matrix: {value}")
        
        return metrics 
        