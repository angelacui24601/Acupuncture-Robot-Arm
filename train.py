from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import json
import torch
import sys
import datetime
import torch.utils.data
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import time
from opts import opts
from src.dataset.dataset_factory import get_dataset
from src.model.model import create_model, load_checkpoint, save_checkpoint
from src.util.lr_scheduler import get_scheduler
from src.util.lars import LARS
from src.util.logger import setup_logger
from src.util.util import AverageMeter
from chainercv.evaluations import eval_semantic_segmentation
from src.model.loss import Dice_loss, Focal_Loss, KeypointLoss, CE_Loss
from src.model.utils import calculate_mse, calculate_rmse, calculate_nmse
import warnings
warnings.filterwarnings("ignore")
from thop import profile
from src.util.data_parallel import DataParallel

""" Calculate the time taken """
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def get_optimizer(opt, model):

    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=opt.batch_size * opt.world_size / 256 * opt.base_lr,
            # lr=opt.base_lr,
            momentum=opt.momentum,
            weight_decay=opt.weight_decay)
    elif opt.optimizer == 'adam':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            # lr=opt.batch_size * opt.world_size / 256 * opt.base_lr)
            lr=0.0001)
    elif opt.optimizer == 'lars':
        optimizer = LARS(
            model.parameters(),
            lr=opt.batch_size * opt.world_size / 256 * opt.base_lr,
            momentum=opt.momentum,
            weight_decay=opt.weight_decay)

    return optimizer

def train(model, train_loader, vaild_loader, optimizer, scaler, scheduler, epoch, logger, opt):
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    miou_meter = AverageMeter()
    oa_meter = AverageMeter()
    mse_meter = AverageMeter()
    rmse_meter = AverageMeter()
    nmse_meter = AverageMeter()
    end = time.time()
    time1 = time.time()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    # keypoint_loss = KeypointLoss()
    keypoint_loss = nn.MSELoss()
    model.train()
    train_len = len(train_loader)
    # for crops_a, coords, flags, crops_RGB in train_loader:
    for i, (img, acupoint, segm, seg_label, keypoint_label) in enumerate(train_loader):

        # crops = [crop.cuda(device=opt.gpus[0]) for crop in crops_a]
        # coords = [coord.cuda(device=opt.gpus[0]) for coord in coords]
        # flags = [flag.cuda(device=opt.gpus[0]) for flag in flags]
        # crops_RGB = [crops_R.cuda(device=opt.gpus[0]) for crops_R in crops_RGB]

        img = img.cuda(non_blocking=True).type(torch.float32)
        acupoint = acupoint.cuda(non_blocking=True).type(torch.float32)
        segm = segm.cuda(non_blocking=True).type(torch.float32)
        seg_label = seg_label.cuda(non_blocking=True).type(torch.float32)
        keypoint_label = keypoint_label.cuda(non_blocking=True).type(torch.float32)
        weights = torch.from_numpy(np.ones([15], np.float32)).cuda(non_blocking=True)

        # compute output and loss
        with torch.cuda.amp.autocast(scaler is not None):
            masks, points = model(img, acupoint)

        # loss_focal = Focal_Loss(masks, segm, weights, num_classes=15)
        # loss_dice = Dice_loss(masks, seg_label)
        # loss_segm = loss_focal + loss_dice

        # loss_segm = criterion(masks, segm.type(torch.int64))

        loss_segm = CE_Loss(masks, segm, weights, num_classes=-1)

        # kpt_mask = keypoint_label[..., 2] != 0 if keypoint_label.shape[-1] == 3 else torch.full_like(keypoint_label[..., 0], True)
        kpts_loss = keypoint_loss(points, keypoint_label)  # pose loss

        loss = loss_segm + kpts_loss

        if torch.isnan(loss):
            ValueError("Loss is NaN")

            # loss = loss.mean()
        optimizer.zero_grad()

        if opt.fp16:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
        else:

            loss.backward()
            optimizer.step()
        scheduler.step()

        # avg loss from batch size
        loss_meter.update(loss.item(), img[0].size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        masks = masks.argmax(dim=1)
        # miou = iou_mean(pred, label.type(torch.int64), opt.n_classes)
        result = eval_semantic_segmentation(masks.detach().cpu().numpy(), segm.type(torch.int64).detach().cpu().numpy())

        valid_idx = segm != -1
        acc = (masks[valid_idx] == segm[valid_idx]).sum() / masks[valid_idx].numel()

        if i % 10 == 0:
            lr = optimizer.param_groups[0]['lr']
            etas = batch_time.avg * (train_len - i)
            logger.info(
                f'Train: [{epoch}/{opt.num_epochs}][{i}/{train_len}]  '
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.4f}  '
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})  '
                f'loss {loss_meter.val:.5f} ({loss_meter.avg:.5f})  '
                f'loss_segm {loss_segm:.5f} '
                f'loss_keypoint {kpts_loss:.5f}  '
                f'miou {result["miou"]:.5f}  '
                f'acc {acc:.5f}  '
            )

    """验证"""
    model.eval()
    logger.info(
        f'-------------------------------\n'
    )
    for i, (img, acupoint, segm, seg_label, keypoint_label) in enumerate(vaild_loader):

        img = img.cuda(non_blocking=True).type(torch.float32)
        acupoint = acupoint.cuda(non_blocking=True).type(torch.float32)
        segm = segm.cuda(non_blocking=True).type(torch.float32)
        seg_label = seg_label.cuda(non_blocking=True).type(torch.float32)
        keypoint_label = keypoint_label.cuda(non_blocking=True).type(torch.float32)

        # compute output and loss
        with torch.cuda.amp.autocast(scaler is not None):
            masks, points = model(img, acupoint)

        masks = masks.argmax(dim=1)
        # miou = iou_mean(pred, label.type(torch.int64), opt.n_classes)
        result = eval_semantic_segmentation(masks.detach().cpu().numpy(), segm.type(torch.int64).detach().cpu().numpy())

        valid_idx = segm != -1
        acc = (masks[valid_idx] == segm[valid_idx]).sum() / masks[valid_idx].numel()

        mse = calculate_mse(points, keypoint_label)
        rmse = calculate_rmse(points, keypoint_label)
        nmse = calculate_nmse(points, keypoint_label)

        miou_meter.update(result['miou'], img[0].size(0))
        oa_meter.update(acc, img[0].size(0))
        mse_meter.update(mse, img[0].size(0))
        rmse_meter.update(rmse, img[0].size(0))
        nmse_meter.update(nmse, img[0].size(0))

        # fwiou_meter.update(fwiou, batch[0].size(0))
        logger.info(
            f'Train: [{epoch}/{opt.num_epochs}][{i}/{train_len}]  '
            f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.4f}  '
            f'time {batch_time.val:.4f} ({batch_time.avg:.4f})  '
            f'miou {miou_meter.val:.5f} ({miou_meter.avg:.5f}) '
            f'oa {oa_meter.val:.5f} ({oa_meter.avg:.5f}) '
            f'MSE {mse_meter.val:.5f} ({mse_meter.avg:.5f}) '
            f'RMSE {rmse_meter.val:.5f} ({rmse_meter.avg:.5f}) '
            f'NMSE {nmse_meter.val:.5f} ({nmse_meter.avg:.5f}) '
        )


    logger.info(
        f'-------------------------------\n')
    logger.info(
        f'iou: [{result["iou"]}]  '
        # f'macro_acc {macro_acc:.5f}  '
    )


def main(opt):
    """ Dataset and loader """
    # opt.local_rank = 0
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # device = torch.device('cuda:0')
    # torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', world_size=1, rank=0)

    if os.environ["LOCAL_RANK"] is not None:
        opt.local_rank = int(os.environ["LOCAL_RANK"])
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3'
    torch.cuda.device_count()
    torch.cuda.set_device(opt.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    cudnn.benchmark = True

    opt.world_size = dist.get_world_size()
    opt.batch_size = int(opt.batch_size / opt.world_size)

    logger = setup_logger(output=opt.output_dir, distributed_rank=dist.get_rank(), name="HSNet")

    if dist.get_rank() == 0:
        path = os.path.join(opt.output_dir, "config.json")
        with open(path, 'w') as f:
            json.dump(vars(opt), f, indent=2)
        logger.info("Full config saved to {}".format(path))

    # print args
    logger.info(
        "\n".join("%s: %s" % (k, str(v))
                  for k, v in sorted(dict(vars(opt)).items()))
    )

    Dataset = get_dataset(opt.dataset)
    train_dataset = Dataset('train', opt=opt)
    valid_dataset = Dataset('val', opt=opt)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    vaild_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)

    print("loading trainset...")
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=opt.batch_size,
        # shuffle=(train_sampler is None),
        shuffle=False,
        num_workers=opt.num_workers,
        sampler=train_sampler,
        pin_memory=True,
        drop_last=True
    )

    vaild_loader = DataLoader(
        dataset=valid_dataset,
        # batch_size=opt.batch_size,
        batch_size=int(opt.batch_size / 2),
        # shuffle=(train_sampler is None),
        shuffle=False,
        num_workers=opt.num_workers,
        sampler=vaild_sampler,
        pin_memory=True,
        drop_last=True
    )

    opt.num_instances = len(train_loader.dataset)
    logger.info(f"length of training dataset: {opt.num_instances}")

    # create model
    logger.info("=> creating model '{}'".format(opt.arch))
    model = create_model(opt=opt).cuda()
    # input1 = torch.randn(1, 250, 256, 256)
    #
    # macs, params = profile(model, inputs=(input1,))
    #
    # print("FLOPS:", str(2 * macs))
    # print("params:", str(params))

    # model = torch.nn.DataParallel(model, device_ids=opt.gpus)  # 指定要用到的设备
    # model = DataParallel(model, device_ids=opt.gpus, chunk_sizes=opt.chunk_sizes).cuda(device=opt.gpus[0])  # 指定要用到的设备
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.local_rank], find_unused_parameters=True)


    # logger.info(model)

    # create optimizer
    optimizer = get_optimizer(opt, model)

    # for state in optimizer.state.values():
    #     for k, v in state.items():
    #         if isinstance(v, torch.Tensor):
    #             state[k] = v.to(device=opt.device, non_blocking=True)

    # define scheduler
    scheduler = get_scheduler(optimizer, len(train_loader), opt)

    # define scaler
    if opt.fp16:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    if opt.load_model != '':
        # opt.load_model = os.path.join(opt.load_model, "ckpt_epoch_10.pth")
        load_checkpoint(logger, opt, model, optimizer, scheduler, scaler)

    print("start...")
    start_epoch = 0
    """ Training the model """
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        opt.now_epcho = epoch
        train_sampler.set_epoch(epoch)
        # start_time = time.time()
        train(model, train_loader, vaild_loader, optimizer, scaler, scheduler, epoch, logger, opt)

        save_checkpoint(logger, opt, epoch, model, optimizer, scheduler, scaler)

        # if dist.get_rank() == 0 and (epoch % int(opt.save_freq) == 0 or epoch in opt.save_point):
        #     save_checkpoint(logger, opt, epoch, model, optimizer, scheduler, scaler)

        if dist.get_rank() == 0 and epoch % int(opt.save_freq) == 0:
            logger.info('==> Saving...')
            state = {
                'args': opt,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
            }
            if opt.fp16:
                state['scaler'] = scaler.state_dict()
            file_name = os.path.join(opt.output_dir, f'ckpt_epoch_{epoch}.pth')
            torch.save(state, file_name)

        # end_time = time.time()
        # epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        #
        # data_str = f'{opt.exp_id} | {opt.arch} | Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
        # data_str += f'\t{opt.exp_id} | {opt.arch} | Best Valid Loss: {best_valid_loss:.4f}\n'
        # # data_str += f'\t Val. Loss: {valid_loss:.3f}\n'
        # print(data_str)

if __name__ == '__main__':
    opt = opts().parse()
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True

    # opt.exp_id = "test"

    """ Create a directory. """
    if opt.exp_id == 'default':
        print("exp_id null !!!")
        sys.exit(1)
    else:
        opt.output_dir = os.path.join('..', "results", opt.arch, opt.exp_id)

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    main(opt)

