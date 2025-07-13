# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import os 
import json
import numpy as np

def calculate_threshold(logger, mask_scores, percentile=85): 
    logger.info(f"Calculating threshold at {percentile}th percentile of negative images")
    threshold = np.percentile(mask_scores, percentile) 
    config_dir = "./config"
    os.makedirs(config_dir, exist_ok=True)
    file_path = os.path.join(config_dir, f"threshold.json")
    with open(file_path, "w") as f: 
        json.dump({"threshold": threshold}, f)
    logger.info(f"The generated threshold has been saved to {file_path}")
    return threshold
