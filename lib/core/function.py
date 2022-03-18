# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import logging
import os
import time

import numpy as np
import numpy.ma as ma
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.utils import AverageMeter
from utils.utils import get_confusion_matrix
from utils.utils import adjust_learning_rate

import utils.distributed as dist

# flag for tcr versions
# flag = 0 -> tcr version 1 (add one tcr branch)
# flag = 1 -> tcr version 2 (add cloud information)
flag = 1

def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that 
    process with rank 0 has the averaged results.
    """
    world_size = dist.get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        torch.distributed.reduce(reduced_inp, dst=0)
    return reduced_inp / world_size


def train(config, epoch, num_epoch, epoch_iters, base_lr,
          num_iters, trainloader, optimizer, model, writer_dict, alpha=0.1):
    # Training
    model.train()

    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    # record ce_loss and tcr_loss
    ave_ce_loss = AverageMeter()
    if flag == 0:
        ave_tcr_loss = AverageMeter()
    elif flag == 1:
        # record separate tcr_loss
        # tcr_01_loss, tcr_00_loss, tcr_11_loss
        ave_tcr_01_loss = AverageMeter()
        ave_tcr_00_loss = AverageMeter()
        ave_tcr_11_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']

    if flag == 0:
        for i_iter, batch in enumerate(trainloader, 0):
            # images, labels, _, _ = batch
            # images = images.cuda()
            # labels = labels.long().cuda()

            # losses, _ = model(images, labels)
            # loss = losses.mean()

            """
            TCR: two branches, CE1 + CE2 + TCR
            """
            images, labels, _, _, images1, labels1, _, _ = batch
            images = images.cuda()
            labels = labels.long().cuda()
            images1 = images1.cuda()
            labels1 = labels1.long().cuda()

            # losses, _ = model(images, images1, labels, labels1)
            # show ce_loss, tcr_loss separately
            ce_losses, tcr_losses, _ = model(images, images1, labels, labels1)
            losses = alpha * ce_losses + (1 - alpha) * tcr_losses
            ce_loss = ce_losses.mean()
            tcr_loss = tcr_losses.mean()

            loss = losses.mean()

            if dist.is_distributed():
                reduced_loss = reduce_tensor(loss)
                # record ce_loss and tcr_loss
                reduced_ce_loss = reduce_tensor(ce_loss)
                reduced_tcr_loss = reduce_tensor(tcr_loss)
            else:
                reduced_loss = loss
                # record ce_loss and tcr_loss
                reduced_ce_loss = ce_loss
                reduced_tcr_loss = tcr_loss

            model.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - tic)
            tic = time.time()

            # update average loss
            ave_loss.update(reduced_loss.item())

            # update average ce_loss and tcr_loss
            ave_ce_loss.update(reduced_ce_loss.item())
            ave_tcr_loss.update(reduced_tcr_loss.item())

            lr = adjust_learning_rate(optimizer,
                                    base_lr,
                                    num_iters,
                                    i_iter+cur_iters)

            if i_iter % config.PRINT_FREQ == 0 and dist.get_rank() == 0:
                # msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                #       'lr: {}, Loss: {:.6f}' .format(
                #           epoch, num_epoch, i_iter, epoch_iters,
                #           batch_time.average(), [x['lr'] for x in optimizer.param_groups], ave_loss.average())

                # record average ce_loss and tcr_loss
                msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                    'lr: {}, Loss: {:.6f}, CE_Loss: {:.6f}, TCR_Loss: {:.6f}' .format(
                        epoch, num_epoch, i_iter, epoch_iters,
                        batch_time.average(), [x['lr'] for x in optimizer.param_groups], ave_loss.average(), ave_ce_loss.average(), ave_tcr_loss.average())
                logging.info(msg)

        writer.add_scalar('train_loss', ave_loss.average(), global_steps)
        # record average ce_loss and tcr_loss
        writer.add_scalar('train_ce_loss', ave_ce_loss.average(), global_steps)
        writer.add_scalar('train_tcr_loss', ave_tcr_loss.average(), global_steps)

        writer_dict['train_global_steps'] = global_steps + 1
    elif flag == 1:
        for i_iter, batch in enumerate(trainloader, 0):
            """
            TCR: two branches, CE1 + CE2 + TCR
            """
            images, labels, clouds, _, _, images1, labels1, clouds1, _, _ = batch
            images = images.cuda()
            labels = labels.long().cuda()
            clouds = clouds.long().cuda()
            images1 = images1.cuda()
            labels1 = labels1.long().cuda()
            clouds1 = clouds1.long().cuda()

            # show ce_loss, 3 separate tcr_losses
            ce_losses, tcr_01_losses, tcr_00_losses, tcr_11_losses, _ = model(images, images1, labels, labels1, clouds, clouds1)
            # losses = alpha * ce_losses + (1 - alpha) * (0.8 * tcr_01_losses + 0.1 * tcr_00_losses + 0.1 * tcr_11_losses)
            losses = config.LOSS.WEIGHT_CE * ce_losses + (1 - config.LOSS.WEIGHT_CE) * (config.LOSS.WEIGHT_TCR_01 * tcr_01_losses + config.LOSS.WEIGHT_TCR_00 * tcr_00_losses + config.LOSS.WEIGHT_TCR_11 * tcr_11_losses)
            ce_loss = ce_losses.mean()
            tcr_01_loss = tcr_01_losses.mean()
            tcr_00_loss = tcr_00_losses.mean()
            tcr_11_loss = tcr_11_losses.mean()
            loss = losses.mean()

            if dist.is_distributed():
                reduced_loss = reduce_tensor(loss)
                # record ce_loss and 3 separate tcr_losses
                reduced_ce_loss = reduce_tensor(ce_loss)
                reduced_tcr_01_loss = reduce_tensor(tcr_01_loss)
                reduced_tcr_00_loss = reduce_tensor(tcr_00_loss)
                reduced_tcr_11_loss = reduce_tensor(tcr_11_loss)
            else:
                reduced_loss = loss
                # record ce_loss and 3 separate tcr_losses
                reduced_ce_loss = ce_loss
                reduced_tcr_01_loss = tcr_01_loss
                reduced_tcr_00_loss = tcr_00_loss
                reduced_tcr_11_loss = tcr_11_loss

            model.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - tic)
            tic = time.time()

            # update average loss
            ave_loss.update(reduced_loss.item())

            # update average ce_loss and 3 separate tcr_losses
            ave_ce_loss.update(reduced_ce_loss.item())
            ave_tcr_01_loss.update(reduced_tcr_01_loss.item())
            ave_tcr_00_loss.update(reduced_tcr_00_loss.item())
            ave_tcr_11_loss.update(reduced_tcr_11_loss.item())

            lr = adjust_learning_rate(optimizer,
                                    base_lr,
                                    num_iters,
                                    i_iter+cur_iters)

            if i_iter % config.PRINT_FREQ == 0 and dist.get_rank() == 0:
                # msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                #       'lr: {}, Loss: {:.6f}' .format(
                #           epoch, num_epoch, i_iter, epoch_iters,
                #           batch_time.average(), [x['lr'] for x in optimizer.param_groups], ave_loss.average())

                # record average ce_loss and 3 separate tcr_losses
                msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                    'lr: {}, Loss: {:.6f}, CE_Loss: {:.6f}, TCR_01_Loss: {:.6f}, TCR_00_Loss: {:.6f}, TCR_11_Loss: {:.6f}' .format(
                        epoch, num_epoch, i_iter, epoch_iters,
                        batch_time.average(), [x['lr'] for x in optimizer.param_groups], ave_loss.average(), ave_ce_loss.average(),
                        ave_tcr_01_loss.average(), ave_tcr_00_loss.average(), ave_tcr_11_loss.average())
                logging.info(msg)

        writer.add_scalar('train_loss', ave_loss.average(), global_steps)
        # record average ce_loss and 3 separate tcr_losses
        writer.add_scalar('train_ce_loss', ave_ce_loss.average(), global_steps)
        writer.add_scalar('train_tcr_01_loss', ave_tcr_01_loss.average(), global_steps)
        writer.add_scalar('train_tcr_00_loss', ave_tcr_00_loss.average(), global_steps)
        writer.add_scalar('train_tcr_11_loss', ave_tcr_11_loss.average(), global_steps)
        
        writer_dict['train_global_steps'] = global_steps + 1

def validate(config, testloader, model, writer_dict, alpha=0.1):
    model.eval()
    ave_loss = AverageMeter()
    nums = config.MODEL.NUM_OUTPUTS
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums))
    with torch.no_grad():
        for idx, batch in enumerate(testloader):
            # image, label, _, _ = batch
            # size = label.size()
            # image = image.cuda()
            # label = label.long().cuda()

            # losses, pred = model(image, label)

            """
            TCR: two branches, CE1 + CE2 + TCR
            """
            if flag == 0:
                image, label, _, _, image1, label1, _, _ = batch
                size = label.size()
                image = image.cuda()
                label = label.long().cuda()
                image1 = image1.cuda()
                label1 = label1.long().cuda()

                # losses, pred = model(image, image1, label, label1)
                # loss function
                ce_losses, tcr_losses, pred = model(image, image1, label, label1)
                losses = alpha * ce_losses + (1 - alpha) * tcr_losses

            elif flag == 1:
                image, label, cloud, _, _, image1, label1, cloud1, _, _ = batch
                size = label.size()
                image = image.cuda()
                label = label.long().cuda()
                cloud = cloud.long().cuda()
                image1 = image1.cuda()
                label1 = label1.long().cuda()
                cloud1 = cloud1.long().cuda()

                ce_losses, tcr_01_losses, tcr_00_losses, tcr_11_losses, pred = model(image, image1, label, label1, cloud, cloud1)
                # losses = alpha * ce_losses + (1 - alpha) * (0.8 * tcr_01_losses + 0.1 * tcr_00_losses + 0.1 * tcr_11_losses)
                losses = config.LOSS.WEIGHT_CE * ce_losses + (1 - config.LOSS.WEIGHT_CE) * (config.LOSS.WEIGHT_TCR_01 * tcr_01_losses + config.LOSS.WEIGHT_TCR_00 * tcr_00_losses + config.LOSS.WEIGHT_TCR_11 * tcr_11_losses)

            if not isinstance(pred, (list, tuple)):
                pred = [pred]
            for i, x in enumerate(pred):
                x = F.interpolate(
                    input=x, size=size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

                confusion_matrix[..., i] += get_confusion_matrix(
                    label,
                    x,
                    size,
                    config.DATASET.NUM_CLASSES,
                    config.TRAIN.IGNORE_LABEL
                )

            if idx % 10 == 0:
                print(idx)

            loss = losses.mean()
            if dist.is_distributed():
                reduced_loss = reduce_tensor(loss)
            else:
                reduced_loss = loss
            ave_loss.update(reduced_loss.item())

    if dist.is_distributed():
        confusion_matrix = torch.from_numpy(confusion_matrix).cuda()
        reduced_confusion_matrix = reduce_tensor(confusion_matrix)
        confusion_matrix = reduced_confusion_matrix.cpu().numpy()

    for i in range(nums):
        pos = confusion_matrix[..., i].sum(1)
        res = confusion_matrix[..., i].sum(0)
        tp = np.diag(confusion_matrix[..., i])
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()
        if dist.get_rank() <= 0:
            logging.info('{} {} {}'.format(i, IoU_array, mean_IoU))

    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
    writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1
    return ave_loss.average(), mean_IoU, IoU_array


def testval(config, test_dataset, testloader, model,
            sv_dir='', sv_pred=False):
    model.eval()
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, _, name, *border_padding = batch
            size = label.size()
            pred = test_dataset.multi_scale_inference(
                config,
                model,
                image,
                scales=config.TEST.SCALE_LIST,
                flip=config.TEST.FLIP_TEST)

            if len(border_padding) > 0:
                border_padding = border_padding[0]
                pred = pred[:, :, 0:pred.size(2) - border_padding[0], 0:pred.size(3) - border_padding[1]]

            if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

            confusion_matrix += get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)

            if sv_pred:
                sv_path = os.path.join(sv_dir, 'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)

            if index % 100 == 0:
                logging.info('processing: %d images' % index)
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IoU = IoU_array.mean()
                logging.info('mIoU: %.4f' % (mean_IoU))

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum()/pos.sum()
    mean_acc = (tp/np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    return mean_IoU, IoU_array, pixel_acc, mean_acc


def test(config, test_dataset, testloader, model,
         sv_dir='', sv_pred=True):
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            image, size, name = batch
            size = size[0]
            pred = test_dataset.multi_scale_inference(
                config,
                model,
                image,
                scales=config.TEST.SCALE_LIST,
                flip=config.TEST.FLIP_TEST)

            if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

            if sv_pred:
                sv_path = os.path.join(sv_dir, 'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)
