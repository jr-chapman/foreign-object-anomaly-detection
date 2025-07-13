# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import os
import ast
import pandas as pd
from sklearn.model_selection import train_test_split


FOREIGN_OBJECTS = [
    "tracheostomy tube",
    "endotracheal tube",
    "NSG tube",
    "chest drain tube",
    "ventriculoperitoneal drain tube",
    "gastrostomy tube",
    "nephrostomy tube",
    "double J stent",
    "catheter",
    "central venous catheter",
    "central venous catheter via subclavian vein",
    "central venous catheter via jugular vein",
    "reservoir central venous catheter",
    "central venous catheter via umbilical vein",
    "electrical device",
    "dual chamber device",
    "single chamber device",
    "pacemaker",
    "dai",
    "artificial heart valve",
    "artificial mitral heart valve",
    "artificial aortic heart valve",
    "metal",
    "osteosynthesis material",
    "sternotomy",
    "suture material",
    "bone cement",
    "prosthesis",
    "humeral prosthesis",
    "mammary prosthesis",
    "endoprosthesis",
    "aortic endoprosthesis",
    "abnormal foreign body",
    "external foreign body"
]

PROJECTIONS_EXCLUDE = ['L', 'UNK', 'EXCLUDE', 'COSTAL']
LABELS_EXCLUDE = ['unchanged', 'exclude', 'suboptimal study']

def filter_data(file, logger):
    required_columns = ['ImageID', 'ImageDir', 'Labels', 'Projection']

    logger.info(f"Excluding the following projections: {PROJECTIONS_EXCLUDE}")
    logger.info(f"Excluding the following labels: {LABELS_EXCLUDE}")
    projection_counter = 0 
    excluded_labels_counter = 0

    foreign_object_df = []
    df = file[required_columns]

    for i, element in df.iterrows():
        labels = element['Labels']
        if pd.isna(labels): 
            continue

        try:
            labels_cleaned = [label.strip().lower() for label in ast.literal_eval(labels)]
        except Exception as e: 
            logger.warning(f"Invalid labels at row {i}")
            continue

        projection = element['Projection']
        directory = element['ImageDir']
        normal = 0
        objects = 0
        objects_present = []

        try:
            if projection in PROJECTIONS_EXCLUDE: 
                projection_counter += 1
                continue
            if any(label in labels_cleaned for label in LABELS_EXCLUDE): 
                excluded_labels_counter += 1
                continue
            if 'normal' in labels_cleaned:
                normal = 1
            for foreign_object in FOREIGN_OBJECTS:
                if foreign_object in labels_cleaned:
                    objects = 1
                    objects_present.append(foreign_object)

        except Exception as e:
            logger.warning("Failure processing row: " + str(i))
            continue

        foreign_object_df.append({
            'index':i,
            'fileName': element['ImageID'],
            'directory': directory,
            'normal': normal,
            'objects': objects,
            'objects_present': objects_present
        })

        if i % 10000 == 0 and i != 0:
            print("Finished processing row: " + str(i))

    logger.info(f'Number of images removed due to incorrect projection: {projection_counter}')
    logger.info(f'Number of images removed due to faulty labels: {excluded_labels_counter}')
    return pd.DataFrame(foreign_object_df)

def save_data_split(directory, path, df): 
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, f"{path}.csv") 
    df.to_csv(file_path, index=False)

def data_split(logger, training_size, negative_df, positive_df, directory):
    logger.info(64*"-")
    negative_training, negative_holdout = train_test_split(negative_df, train_size=training_size)
    logger.info(f"Training size set to {training_size*100}% of negative data samples.")

    #split negative data samples
    training_data, validation_negative = train_test_split(negative_training, train_size=0.9)
    validation_negative, testing_negative = train_test_split(validation_negative, train_size=0.5)
    logger.info(f"Total training observations: {str(negative_training.shape[0])}")
    logger.info(f"Number of observations for training: {str(training_data.shape[0])}")
    logger.info(f"Number of observations to validate training: {str(validation_negative.shape[0])}")
    logger.info(f"Number of observations to test training: {str(testing_negative.shape[0])}")
    logger.info(64*"-")

    #split positive data samples 
    testing_data = pd.concat([positive_df, negative_holdout])
    testing_positive, validation_positive = train_test_split(testing_data, train_size=0.8)
    logger.info(f"Total testing observations: {str(testing_data.shape[0])}")
    logger.info(f"Number of observations for validation: {str(validation_positive.shape[0])}")
    logger.info(f"Number of observations for testing: {str(testing_positive.shape[0])}")
    logger.info(64*"-")

    #save generated splits 
    save_data_split(directory, "training", training_data)
    save_data_split(directory, "validation_negative", validation_negative)
    save_data_split(directory, "testing_negative", testing_negative)
    save_data_split(directory, "validation_positive", validation_positive)
    save_data_split(directory, "testing_positive", testing_positive)

    return training_data, validation_negative, testing_negative, validation_positive, testing_positive

def prepare_execution_data(df, logger, training_size=None, split_path=None): 
    #using only normal images for training 
    negative_samples = df[df['objects'] == 0]
    positive_samples = df[df['objects'] == 1]
    logger.info(f"Total observations without foreign object (negative samples): {str(negative_samples.shape[0])}")
    logger.info(f"Total observations with foreign object (positive samples): {str(positive_samples.shape[0])}")

    if training_size: 
        if os.path.exists(split_path): 
            logger.info(64*"-")
            logger.info(f"Using existing split files in {split_path}")
            return tuple(pd.read_csv(os.path.join(split_path, f"{name}.csv")) 
                         for name in ["training", "validation_negative", "testing_negative", 
                                      "validation_positive", "testing_positive"])
        logger.info(64*"-")
        logger.info(f"Generating dataset splits")
        return data_split(logger, training_size, negative_samples, positive_samples, split_path)
    return df
    
def prepare_dataframe(file, df_dir, logger): 
    df_path = os.path.join(df_dir, "padchest_dataframe.csv")
    if os.path.exists(df_path): 
        logger.info(f"DataFrame already generated. Using file present in {df_path}")
        df = pd.read_csv(df_path)
    else: 
        logger.info(f"No existing DataFrame at path {df_path}. Generating DataFrame")
        df = filter_data(file, logger)
        df.to_csv(df_path, index=False)
    logger.info(64*"-")
    logger.info("Total observations in extracted dataframe: " + str(df.shape[0]))
    return df
