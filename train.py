# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

# Default imports
import os
import time
import json
import shutil
import argparse
import importlib
import numpy as np
from tqdm import tqdm
from datetime import datetime
from contextlib import contextmanager
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 24})

import warnings
warnings.filterwarnings('ignore')

# Torch
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import WeightedRandomSampler

from tacotron2.model import Tacotron2

import torch.multiprocessing as mp
import torch.distributed as dist

# Distributed + AMP
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import autocast, GradScaler
#from apex.parallel import DistributedDataParallel as DDP

from tacotron2.data_function import batch_to_gpu, TextMelLoader, TextMelCollate, DistributedBucketSampler
from tacotron2.loss_function import Tacotron2Loss

import utils

import bitsandbytes as bnb

# Reload Config
configs = importlib.import_module('configs')
configs = importlib.reload(configs)

Config = configs.Config
PConfig = configs.PreprocessingConfig

# Config dependent imports
from tacotron2.text import text_to_sequence
from router import models, loss_functions, data_functions

from common.utils import remove_crackle


def reduce_tensor(tensor, num_gpus):
    """

    :param tensor:
    :param num_gpus:
    :return:
    """
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt = float(rt)/num_gpus
    return rt


def restore_checkpoint(restore_path, model, optimizer):
    """

    :param restore_path:
    :param model_name:
    :return:
    """
    checkpoint = torch.load(restore_path, map_location='cpu')
    start_epoch = checkpoint['epoch'] + 1
    start_iter = checkpoint['iteration'] + 1

    print('Restoring from `{}` checkpoint'.format(restore_path))

    # Unwrap distributed
    model_dict = {}
    for key, value in checkpoint['state_dict'].items():
        new_key = key.replace('module.1.', '')
        new_key = new_key.replace('module.', '')
        model_dict[new_key] = value

    model.load_state_dict(model_dict)
    
    print('Restoring optimizer state')
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer, start_epoch, start_iter


def save_checkpoint(model, epoch, iteration, config, optimizer, filepath):
    """

    :param model:
    :param epoch:
    :param config:
    :param optimizer:
    :param filepath:
    :return:
    """
    print('Saving model and optimizer state at epoch {} to {}'.format(epoch, filepath))
    torch.save({'epoch': epoch,
                'iteration': iteration,
                'config': config,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, filepath)

# adapted from: https://discuss.pytorch.org/t/opinion-eval-should-be-a-context-manager/18998/3
# Following snippet is licensed under MIT license


@contextmanager
def evaluating(model):
    """
    Temporarily switch to evaluation mode.

    :param model:
    :return:
    """
    istrain = model.training
    try:
        model.eval()
        yield model
    finally:
        if istrain:
            model.train()


def validate(model, criterion, valset, batch_size, world_size, collate_fn, distributed_run, batch_to_gpu):
    """
    Handles all the validation scoring and printing

    :param model:
    :param criterion:
    :param valset:
    :param batch_size:
    :param world_size:
    :param collate_fn:
    :param distributed_run:
    :param batch_to_gpu:
    :return:
    """
    with evaluating(model), torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(valset, num_workers=1, shuffle=False, sampler=val_sampler,
                                batch_size=batch_size, pin_memory=False, collate_fn=collate_fn)
        val_loss = 0.0

        for i, batch in enumerate(val_loader):
            x, y, len_x = batch_to_gpu(batch)
            y_pred = model(x)

            loss = balance_loss(x, y, y_pred, criterion) if Config.use_loss_coefficients else criterion(y_pred, y)

            if distributed_run:
                reduced_val_loss = reduce_tensor(loss.data, world_size)
            else:
                reduced_val_loss = loss.item()

            val_loss += reduced_val_loss
        val_loss = val_loss / (i + 1)

    return val_loss


def adjust_learning_rate(epoch, optimizer, learning_rate, anneal_steps, anneal_factor):
    """

    :param epoch:
    :param optimizer:
    :param learning_rate:
    :param anneal_steps:
    :param anneal_factor:
    :return:
    """
    p = 0
    if anneal_steps is not None:
        for i, a_step in enumerate(anneal_steps):
            if epoch >= int(a_step):
                p = p+1

    if anneal_factor == 0.3:
        lr = learning_rate*((0.1 ** (p//2))*(1.0 if p % 2 == 0 else 0.3))
    else:
        lr = learning_rate*(anneal_factor ** p)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def balance_loss(x, y, y_pred, criterion):
    """
    Args:
        x: model input
        y: labels
        y_pred: predictions
        criterion: loss function

    Returns: balanced loss, torch.tensor
    """
    _, _, _, _, _, speaker_ids, emotion_ids = x

    batch_size = speaker_ids.shape[0]
    loss_balanced = 0

    for i in range(batch_size):
        yi = [el[i] for el in y]
        yi_p = [el[i] for el in y_pred]
        
        s_c = Config.speaker_coefficients[str(speaker_ids[i].item())]

        single_loss = s_c * criterion(yi_p, yi)
        
        if Config.use_emotions:
            e_c = Config.emotion_coefficients[str(emotion_ids[i].item())]
            single_loss *= e_c

        loss_balanced += single_loss

    loss = loss_balanced / batch_size

    return Config.loss_scale * loss


def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    n_gpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "65530"

    mp.spawn(
        train_and_eval,
        nprocs=n_gpus,
        args=(
            n_gpus,
        ),
    )

def train_and_eval(rank, n_gpus):
    if rank == 0:
        logger = utils.get_logger(Config.model_dir)
        logger.info(Config)
        writer = SummaryWriter(log_dir=Config.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(Config.model_dir, "eval"))

    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=n_gpus, rank=rank
    )
    torch.manual_seed(Config.seed)
    torch.cuda.set_device(rank)

    # Weight Sampler
    with open(Config.training_files, "r") as txt_file:
        lines = txt_file.readlines()
        
    attr_names_samples = np.array([item.split("|")[1] for item in lines])
    unique_attr_names = np.unique(attr_names_samples).tolist()
    attr_idx = [unique_attr_names.index(l) for l in attr_names_samples]
    attr_count = np.array(
        [len(np.where(attr_names_samples == l)[0]) for l in unique_attr_names]
    )

    weight_attr = 1.0 / attr_count
    dataset_samples_weight = np.array([weight_attr[l] for l in attr_idx])
    dataset_samples_weight = dataset_samples_weight / np.linalg.norm(
        dataset_samples_weight
    )

    weights, attr_names, attr_weights = (
        torch.from_numpy(dataset_samples_weight).float(),
        unique_attr_names,
        np.unique(dataset_samples_weight).tolist(),
    )
    weights = weights * 1.0
    w_sampler = WeightedRandomSampler(weights, len(weights))

    train_dataset = TextMelLoader(Config.training_files, Config.text_cleaners,
                                    Config.load_mel_from_dist, Config.max_wav_value,
                                    Config.sampling_rate, Config.filter_length, Config.hop_length,
                                    Config.win_length, Config.n_mel_channels, Config.mel_fmin,
                                    Config.mel_fmax, Config)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        Config.batch_size,
        [32, 300, 400, 500, 600, 700, 800, 900, 1000],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
        weights=w_sampler,
    )
    collate_fn = TextMelCollate(Config.n_frames_per_step)
    train_loader = DataLoader(
        train_dataset,
        num_workers=8,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
    )

    if rank == 0:
        val_dataset = TextMelLoader(Config.validation_files, Config.text_cleaners,
                                    Config.load_mel_from_dist, Config.max_wav_value,
                                    Config.sampling_rate, Config.filter_length, Config.hop_length,
                                    Config.win_length, Config.n_mel_channels, Config.mel_fmin,
                                    Config.mel_fmax, Config)

    model = Tacotron2(**Config).cuda()

    optimizer = bnb.optim.AdamW(
        model.parameters(),
        Config.learning_rate,
        betas=Config.betas,
        eps=Config.eps,
    )

    model = DistributedDataParallel(model, device_ids=[rank])

    start_epoch = 0
    start_iter = 0

    scaler = GradScaler(enabled=Config.fp16_run)

    if Config.warm_start:
        model = utils.warm_start_model(Config.warm_start_checkpoint, model, Config.ignored_layer)
    else:
        try:
            print('Loading Checkpoint state')
            model, optimizer, start_epoch, start_iter = restore_checkpoint(utils.latest_checkpoint_path(Config.model_dir, "tc2emo_*.pt"), model, optimizer)
            #utils.load_checkpoint(model, optimizer, scaler, start_epoch, start_iter, Config.fp16_run, )
        except Exception as e:
            print("Not Found Checkpoint")
            print(e)

    sigma = None
    criterion = Tacotron2Loss()

    iteration = start_iter
    model.train()

    # Training loop
    if start_epoch >= Config.epochs:
        print('Checkpoint epoch {} >= total epochs {}'.format(start_epoch, Config.epochs))
    else:
        for epoch in range(start_epoch, Config.epochs):
            epoch_start_time = time.time()

            # Used to calculate avg items/sec over epoch
            reduced_num_items_epoch = 0

            # Used to calculate avg loss over epoch
            train_epoch_avg_loss = 0.0
            train_epoch_avg_items_per_sec = 0.0
            num_iters = 0

            if rank == 0:
                pb = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch: {epoch}/{Config.epochs}')
            else:
                pb = enumerate(train_loader)

            for i, batch in pb:
                iter_start_time = time.time()
                adjust_learning_rate(epoch, optimizer, learning_rate=Config.learning_rate, anneal_steps=Config.anneal_steps, anneal_factor=Config.anneal_factor)
                model.zero_grad()
                x, y, num_items = batch_to_gpu(batch)

                with autocast(enabled=Config.amp_run):
                    y_pred = model(x)

                    loss = balance_loss(x, y, y_pred, criterion) if Config.use_loss_coefficients else criterion(y_pred, y)

                if Config.amp_run:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if Config.distributed_run:
                    reduced_loss = reduce_tensor(loss.data, n_gpus)
                    reduced_num_items = reduce_tensor(num_items.data, 1)
                else:
                    reduced_loss = loss.item()
                    reduced_num_items = num_items.item()

                if np.isnan(reduced_loss):
                    raise Exception('loss is NaN')

                train_epoch_avg_loss += reduced_loss
                num_iters += 1

                # Accumulate number of items processed in this epoch
                reduced_num_items_epoch += reduced_num_items

                if Config.amp_run:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), Config.grad_clip_thresh)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), Config.grad_clip_thresh)
                    optimizer.step()

                iteration += 1

                iter_stop_time = time.time()
                iter_time = iter_stop_time - iter_start_time
                items_per_sec = reduced_num_items/iter_time
                train_epoch_avg_items_per_sec += items_per_sec

            epoch_stop_time = time.time()

            epoch_time = epoch_stop_time - epoch_start_time
            train_epoch_items_per_sec = reduced_num_items_epoch / epoch_time

            train_epoch_avg_items_per_sec = train_epoch_avg_items_per_sec / num_iters if num_iters > 0 else 0.0
            train_epoch_avg_loss = train_epoch_avg_loss / num_iters if num_iters > 0 else 0.0
            epoch_val_loss = validate(model, criterion, val_dataset, Config.batch_size, n_gpus, collate_fn, Config.distributed_run, batch_to_gpu)

            if rank == 0:
                utils.summarize(
                    writer=writer,
                    global_step=iteration,
                    scalars={
                        "loss/g/total": train_epoch_avg_loss,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "grad_norm": grad_norm,
                        "train_stats/epoch_items_per_sec": train_epoch_items_per_sec,
                        "train_stats/epoch_avg_items_per_sec": train_epoch_avg_items_per_sec,
                        "train_stats/epoch_time": epoch_time,
                    },
                )

                utils.summarize(
                    writer=writer_eval,
                    global_step=iteration,
                    scalars={
                        "loss/g/total": epoch_val_loss,
                    },
                )

            if epoch != 0 and epoch % Config.epochs_per_checkpoint == 0 and rank == 0:
                checkpoint_path = os.path.join(Config.model_dir, 'checkpoint_{}'.format(epoch))
                save_checkpoint(model, epoch, iteration, Config, optimizer, checkpoint_path)

                with evaluating(model), torch.no_grad():
                    val_sampler = DistributedSampler(val_dataset) if Config.distributed_run else None
                    val_loader = DataLoader(val_dataset, num_workers=1, shuffle=False, sampler=val_sampler, batch_size=Config.batch_size, pin_memory=False, collate_fn=collate_fn)

                    for i, batch in enumerate(val_loader):
                        if i > 2:
                            break
                        
                        batch = batch[:1]
                        x, y, len_x = batch_to_gpu(batch)
                        _, mel, _, alignments = model.infer(x[0], langs=x[5], speakers=x[6], emos=x[7])
                        alignments_numpy = alignments[0].data.cpu().numpy()

                        utils.summarize(
                            writer=writer,
                            global_step=iteration,
                            images={
                                i + "_y_org": utils.plot_spectrogram_to_numpy(
                                    y[0].data.cpu().numpy()
                                ),
                                i + "_y_gen": utils.plot_spectrogram_to_numpy(
                                    mel.data.cpu().numpy()
                                ),
                                i + "_attn": alignments_numpy,
                            },
                        )

if __name__ == '__main__':
    main()
