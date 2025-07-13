# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# MAE Medical Anomaly Detection - https://github.com/lilygeorgescu/MAE-medical-anomaly-detection?tab=readme-ov-file
# --------------------------------------------------------

import os
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import pandas as pd
import os 
from torchvision import transforms

from config import log
from preprocessing import create_dataframe
from preprocessing.process_input import processInput

from config.args import get_args_parser, validate_arguments
import util.misc as misc
from util.misc import save_info

from run_training.train import train
import run_evaluation.evaluate as evaluate

def main(args):
    validate_arguments(args)
    misc.setup_print()

    logging_directory = "logs"
    logging_prefix = "LOG"
    logger, listener = log.logging_setup(logging_directory, logging_prefix)
    logger.info('Implementation started')
    logger.info(64*"-")

    #CSV file is downloaded and saved to ./dataset
    file = pd.read_csv("./dataset/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv", dtype={19: 'string', 20: 'string'}, low_memory=False)
    df_directory = "./dataset"
    split_directory = "./dataset/splits"
    images_folder = args.data_path
    training_size = 0.9 
   
    model = None

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("Set arguments: ")
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    
    metadata_df = create_dataframe.prepare_dataframe(file, df_directory, logger)

    transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.Grayscale(num_output_channels=1), 
            transforms.ToTensor()
        ])

    evaluator = evaluate.MAEEvaluator(logger=logger, 
                                      error=args.err, 
                                      mask_ratio=args.mask_ratio, 
                                      num_trials=args.num_trials, 
                                      output_dir=args.output_dir)

    if args.train or args.evaluate: 
        training_data, validation_negative, testing_negative, validation_positive, testing_positive  = create_dataframe.prepare_execution_data(metadata_df, logger, training_size, split_directory)

    if args.train: 
        save_info("Training argument set - beginning training", logger=logger)

        train_dataset = processInput(training_data, images_folder, logger, transform)
        data_loader_train = torch.utils.data.DataLoader(
            train_dataset, sampler=torch.utils.data.RandomSampler(train_dataset),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True
        )

        model = train(args=args, device=device, trainDataLoader=data_loader_train, logger=logger)

        save_info("Training complete, validating for threshold calculation", logger=logger)
        val_dataset = processInput(testing_negative, images_folder, logger, transform)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        evaluator.generate_threshold(model, val_dataloader)

    if args.evaluate: 
        evaluate.get_threshold()
        splits = {
            "validation_negative": validation_negative,
            "testing_negative": testing_negative, 
            "validation_positive": validation_positive, 
            "testing_positive": testing_positive
        }
        logger.info(64*"-")
        save_info("Evaluation argument set - beginning evaluation", logger=logger)

        selected_split = splits[args.evaluation_set]
        val_dataset = processInput(selected_split, images_folder, logger, transform)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        logger.info(64*"-")
        
        if model:  
            logger.info("Evaluating with trained model")
            evaluator.run_evaluation(model, val_dataloader)
        else: 
            logger.info(f"Evaluating with model checkpoint provided in arguments: {args.model_path}")
            evaluator.run_evaluation(args.model_path, val_dataloader)

    if args.inference: 
        evaluate.get_threshold()
        logger.info(64*"-")
        save_info("Inference argument set - beginning inference", logger=logger)
        logger.info(64*"-")
        df = create_dataframe.prepare_execution_data(metadata_df, logger)
        inference_dataset = processInput(df, images_folder, logger, transform)
        inference_dataloader = torch.utils.data.DataLoader(inference_dataset, batch_size=args.batch_size, shuffle=False)
        logger.info(64*"-")
        save_info(f"Using provided model for inference: {args.model_path}", logger)
        evaluator.run_evaluation(args.model_path, inference_dataloader)

    if not args.train and not args.evaluate and not args.inference: 
        print("Please set argument for execution: --train=True for training, --evaluate=True for evaluation or --inference=True for inference")
    
    
    if listener: 
        listener.stop()

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
