# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import os 
from PIL import Image
import pandas as pd 
import numpy as np 
from torch.utils.data import Dataset

class processInput(Dataset): 
    def __init__(self, df, folder_path, logger, transform): 
        self.df = df 
        self.folder_path = folder_path
        self.logger = logger
        self.transform = transform
        self.valid_samples = self._check_directory()

    def __len__(self): 
        return len(self.valid_samples)

    def _check_directory(self): 
        valid = []
        file_paths = []
        for _, row in self.df.iterrows(): 
            sub_directory = row['directory']
            filename = row['fileName']
            full_directory = os.path.join(self.folder_path, str(sub_directory), filename)
            if os.path.exists(full_directory):
                valid.append(row)
                file_paths.append(full_directory)
        valid = pd.DataFrame(valid)
        valid['filePath'] = file_paths
        self.logger.info(64*"-")
        self.logger.info(f'{str(valid.shape[0])} valid files found')
        return valid 
    

    def __getitem__(self, idx): 
        row = self.valid_samples.iloc[idx]
        filepath = row['filePath']
        filename = row['fileName']
        label = row['objects']
            
        try: 
            if not os.path.exists(filepath): 
                self.logger.warning(f'File {filepath} not found')
                raise FileNotFoundError(f'File {filepath} not found') 

            image = Image.open(filepath)
            if image.mode.startswith("I"): 
                np_img = np.array(image).astype(np.float32)
                arr = (np_img - np_img.min()) * 255 // (np_img.max() - np_img.min() + 1e-10)
                img = np.array(arr).astype(np.uint8)
                image = Image.fromarray(img, mode='L')
            transformed_image = self.transform(image)

        except Exception as e: 
            self.logger.error(f'Error opening or processing image: {e}')
            return None

        return transformed_image, label, filename