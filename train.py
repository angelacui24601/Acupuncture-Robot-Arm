from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os
import time
import warnings

import torch
import torch.nn as nn

from opts import opts
from src.engine import (
    build_dataloader,
    compute_segmentation_metrics,
    evaluate_segmentation,
    move_batch_to_device,
    resolve_output_dir,
    set_random_seed,
    setup_runtime,
)
from src.model.loss import CE_Loss
from src.model.model import create_model, load_checkpoint, save_checkpoint
from src.util.lars import LARS
from src.util.lr_scheduler import get_scheduler
from src.util.util import AverageMeter

warnings.filterwarnings('ignore')


def get_optimizer(opt, model):
    if opt.optimizer == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=opt.batch_size * opt.world_size / 256 * opt.base_lr,
            momentum=opt.momentum,
            weight_decay=opt.weight_decay,
        )
    if opt.optimizer == 'adam':
        return torch.optim.AdamW(model.parameters(), lr=0.0001)
    if opt.optimizer == 'lars':
        return LARS(
            model.parameters(),
            lr=opt.batch_size * opt.world_size / 256 * opt.base_lr,
            momentum=opt.momentum,
            weight_decay=opt.weight_decay,
        )
    raise ValueError(f'Unsupported optimizer: {opt.optimizer}')


def train_one_epoch(model, train_loader, optimizer, scaler, scheduler, epoch, logger, opt, device):
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    keypoint_loss = nn.MSELoss()
    model.train()
    end = time.time()
    train_steps = len(train_loader)

    for step, batch in enumerate(train_loader):
        img, acupoint, segm, _seg_label, keypoint_label = move_batch_to_device(batch, device)
        weights = torch.ones(int(opt.num_classes), dtype=torch.float32, device=device)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            masks, points = model(img, acupoint)
            loss_segm = CE_Loss(masks, segm, weights, num_classes=-1)
            loss_kpt = keypoint_loss(points, keypoint_label)
            loss = opt.seg_loss_weight * loss_segm + opt.kpt_loss_weight * loss_kpt

        if torch.isnan(loss).any():
            raise ValueError('Loss is NaN')

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        scheduler.step()

        loss_meter.update(loss.item(), img.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if step % max(1, int(opt.print_freq)) == 0:
            pred = masks.argmax(dim=1)
            seg_metrics = compute_segmentation_metrics(
                pred.detach().cpu().numpy(),
                segm.type(torch.int64).detach().cpu().numpy(),
                int(opt.num_classes),
            )
            eta_seconds = batch_time.avg * max(0, train_steps - step - 1)
            lr = optimizer.param_groups[0]['lr']
            logger.info(
                f'Train: [{epoch}/{opt.num_epochs}][{step}/{train_steps}] '
                f'eta {datetime.timedelta(seconds=int(eta_seconds))} '
                f'lr {lr:.6f} '
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f}) '
                f'loss {loss_meter.val:.5f} ({loss_meter.avg:.5f}) '
                f'loss_segm {loss_segm.item():.5f} '
                f'loss_keypoint {loss_kpt.item():.5f} '
                f'miou {seg_metrics["miou"]:.5f} '
                f'oa {seg_metrics["oa"]:.5f}'
            )


def save_periodic_checkpoint(opt, epoch, model, optimizer, scheduler, scaler=None):
    state = {
        'args': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
    }
    if scaler is not None:
        state['scaler'] = scaler.state_dict()
    file_name = os.path.join(opt.output_dir, f'ckpt_epoch_{epoch}.pth')
    torch.save(state, file_name)


def main(opt):
    resolve_output_dir(opt)
    runtime, logger = setup_runtime(opt)

    train_dataset, train_loader, train_sampler = build_dataloader(
        opt,
        phase='train',
        batch_size=opt.batch_size,
        is_distributed=runtime.is_distributed,
        shuffle=True,
        drop_last=True,
    )
    _valid_dataset, valid_loader, _valid_sampler = build_dataloader(
        opt,
        phase='val',
        batch_size=max(1, int(opt.batch_size / 2)),
        is_distributed=runtime.is_distributed,
        shuffle=False,
        drop_last=True,
    )

    opt.num_instances = len(train_dataset)
    logger.info(f'length of training dataset: {opt.num_instances}')
    logger.info("=> creating model '%s'", opt.arch)

    model = create_model(opt=opt).to(runtime.device)
    optimizer = get_optimizer(opt, model)
    scheduler = get_scheduler(optimizer, len(train_loader), opt)
    scaler = torch.cuda.amp.GradScaler() if opt.fp16 and runtime.device.type == 'cuda' else None

    if opt.load_model:
        load_checkpoint(logger, opt, model, optimizer, scheduler, scaler)

    if runtime.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.local_rank],
            find_unused_parameters=True,
        )

    for epoch in range(opt.start_epoch, opt.num_epochs + 1):
        opt.now_epcho = epoch
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_one_epoch(model, train_loader, optimizer, scaler, scheduler, epoch, logger, opt, runtime.device)
        evaluate_segmentation(model, valid_loader, logger, opt, runtime.device, is_distributed=runtime.is_distributed)

        if runtime.is_main_process:
            save_checkpoint(logger, opt, epoch, model, optimizer, scheduler, scaler)
            if epoch % int(opt.save_freq) == 0 or epoch in opt.save_point:
                save_periodic_checkpoint(opt, epoch, model, optimizer, scheduler, scaler)


if __name__ == '__main__':
    options = opts().parse()
    set_random_seed(options.seed)
    main(options)

