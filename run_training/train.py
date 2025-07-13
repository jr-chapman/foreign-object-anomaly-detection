# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# MAE Medical Anomaly Detection - https://github.com/lilygeorgescu/MAE-medical-anomaly-detection?tab=readme-ov-file
# Weight decay - https://huggingface.co/spaces/Roll20/pet_score/blob/3653888366407445408f2bfa8c68d6cdbdd4cba6/lib/timm/optim/optim_factory.py
# --------------------------------------------------------

import torch 
import datetime
import time 

import models_mae
import util.misc as misc
from run_training.engine_train import train_one_epoch
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.misc import save_info

def add_weight_decay(model, weight_decay=0.05):
    decay=[]
    no_decay=[]
    for name, param in model.named_parameters(): 
        if not param.requires_grad: 
            continue 
        if len(param.shape)==1 or "bias" in name or "bn" in name: 
            no_decay.append(param)
        else: 
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.0}, 
        {'params': decay, 'weight_decay': weight_decay}]


def train(args, device, trainDataLoader, logger): 
        
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss,
                                            apply_loss_on_unmasked=args.apply_loss_on_unmasked,
                                            img_size=args.img_size,
                                            patch_size=args.patch_size)
    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    
    logger.info(64*"-")
    save_info("base lr: %.2e" % (args.lr * 256 / eff_batch_size), logger=logger)
    save_info("actual lr: %.2e" % args.lr, logger=logger)
    save_info("accumulate grad iterations: %d" % args.accum_iter, logger=logger)
    save_info("effective batch size: %d" % eff_batch_size, logger=logger)


    # following timm: set wd as 0 for bias and norm layers
    param_groups = add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    if args.model_path: 
        checkpoint = torch.load(args.model_path, weights_only=False)
        save_info("Loading pre-trained model from checkpoint", logger=logger)
        model_without_ddp.load_state_dict(checkpoint['model'], strict = False)
        save_info(f"Pre-trained weightes loaded from: {args.model_path}", logger=logger)

    else: 
        misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, logger=logger)

    save_info(f"Start training for {args.epochs} epochs", logger=logger)
    logger.info(64*"-")
    start_time = time.time()

    losses = []
    lrs = []

    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, trainDataLoader,
            optimizer, device, epoch, loss_scaler,
            args=args
        )
        if args.output_dir and (epoch % 10 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}
        logger.info(log_stats)
        for key, value in train_stats.items(): 
            if key == 'loss': 
                losses.append(value)
            if key == 'lr': 
                lrs.append(value)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(64*"-")
    save_info('Training time {}'.format(total_time_str), logger=logger)

    logger.info(64*"-")
    logger.info("Training Finished")
    logger.info(f"Losses per epoch: {losses}" )
    logger.info(f"Learning rate per epoch: {lrs}")
    
    return model 