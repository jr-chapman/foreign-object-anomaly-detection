# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# MAE Medical Anomaly Detection - https://github.com/lilygeorgescu/MAE-medical-anomaly-detection?tab=readme-ov-file
# --------------------------------------------------------

import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--model_path', type=str, default=None, 
                        help='Path to pretrained model for continued training or model to use for validation / testing')
    parser.add_argument('--img_size', default=224, type=int,
                        help='Input image size')
    parser.add_argument('--patch_size', default=16, type=int,
                        help='Patch size')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches)')

    # Training parameters 
    parser.add_argument('--train', action='store_true',  
                        help="Set model for training")
    parser.add_argument('--resume', default=None, type=str,
                        help='Resume training from provided checkpoint. Set model path here instead of in model_path')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=100, type=int, 
                        help='Number of epochs to train for')
    parser.add_argument('--warmup_epochs', default=5, type=int, metavar='N',
                        help='Number of epochs to warmup LR')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='Epoch to start training from')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)
    parser.add_argument('--apply_loss_on_unmasked', action='store_true',
                        help='Apply the loss for the unmasked tokens.')
    parser.set_defaults(apply_loss_on_unmasked=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='Learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='Base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0.0, metavar='LR',
                        help='Lower lr bound for cyclic schedulers that hit 0')

    # Dataset parameters
    parser.add_argument('--data_path', default=None, type=str,
                        help='Path to the dataset')
    parser.add_argument('--output_dir', default='./output_dir', type=str, 
                        help='Path to save output to')

    # Execution parameters
    parser.add_argument('--num_workers', default=4, type=int, 
                        help='Number of CPU workers for execution')
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--device', default='cuda', type=str,
                        help='Device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int,
                        help='Seed set for reproducability')

    # Run evaluation parameters 
    parser.add_argument('--validate', action='store_true',
                        help='Set model for validation')
    parser.add_argument('--validation_set', default=None, type=str, 
                        help="Dataset split to use for validation: validation_negative, testing_negative, validation_positive, testing_positive")
    parser.add_argument('--inference', action='store_true', 
                        help='Set model for inference using a pre-trained model on provided data')
    parser.add_argument('--err', type=lambda s: [err.strip() for err in s.split(',')], default=None, 
                        help="Additional evaluation metrics: use 'ssim' for SSIM score or 'anomaly' for anomaly score or 'ssim, anomaly' for both")
    parser.add_argument('--num_trials', type=int, default=4, 
                        help="Number of reconstructions to perform")
    return parser

def assert_arguments(argument, name):
    assert argument is not None, f"The following argument is required but not provided: {name}"

def validate_arguments(args): 
    assert_arguments(args.data_path, "data_path")
    if args.validate: 
        assert_arguments(args.validation_set, "validation_set")
    if args.inference: 
        assert_arguments(args.model_path, "model_path")

    