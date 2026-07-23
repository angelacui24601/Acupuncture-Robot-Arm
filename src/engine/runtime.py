from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from src.dataset.dataset_factory import get_dataset
from src.util.logger import setup_logger


@dataclass
class RuntimeContext:
    device: torch.device
    is_distributed: bool
    distributed_rank: int
    is_main_process: bool


def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def resolve_output_dir(opt) -> str:
    if opt.exp_id == 'default':
        raise ValueError('exp_id null !!!')

    output_dir = os.path.join(opt.output_root, opt.arch, opt.exp_id)
    os.makedirs(output_dir, exist_ok=True)
    opt.output_dir = output_dir
    return output_dir


def setup_runtime(opt, logger_name: str = 'HSNet'):
    is_distributed = 'LOCAL_RANK' in os.environ
    if is_distributed:
        if not torch.cuda.is_available():
            raise RuntimeError('Distributed launch requires CUDA availability.')
        opt.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        device = torch.device('cuda', opt.local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group(backend='nccl', init_method='env://')
        opt.world_size = dist.get_world_size()
        distributed_rank = dist.get_rank()
    else:
        opt.local_rank = 0
        if torch.cuda.is_available() and len(opt.gpus) > 0 and opt.gpus[0] >= 0:
            device = torch.device('cuda', opt.gpus[0])
            torch.cuda.set_device(device)
        else:
            device = torch.device('cpu')
        opt.world_size = 1
        distributed_rank = 0

    cudnn.benchmark = torch.cuda.is_available() and not opt.not_cuda_benchmark
    opt.batch_size = max(1, int(opt.batch_size / opt.world_size))
    opt.device = str(device)

    logger = setup_logger(output=opt.output_dir, distributed_rank=distributed_rank, name=logger_name)
    is_main_process = distributed_rank == 0

    if is_main_process:
        path = os.path.join(opt.output_dir, 'config.json')
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(vars(opt), handle, indent=2, ensure_ascii=False)
        logger.info('Full config saved to %s', path)

    logger.info('\n'.join('%s: %s' % (key, str(value)) for key, value in sorted(dict(vars(opt)).items())))

    return RuntimeContext(
        device=device,
        is_distributed=is_distributed,
        distributed_rank=distributed_rank,
        is_main_process=is_main_process,
    ), logger


def build_dataloader(
    opt,
    phase: str,
    batch_size: int,
    is_distributed: bool,
    shuffle: bool,
    drop_last: bool,
):
    dataset_cls = get_dataset(opt.dataset)
    dataset = dataset_cls(phase, opt=opt)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle) if is_distributed else None
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        num_workers=opt.num_workers,
        sampler=sampler,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
    )
    return dataset, loader, sampler


def move_batch_to_device(batch, device: torch.device):
    moved = []
    for item in batch:
        if torch.is_tensor(item):
            moved.append(item.to(device=device, non_blocking=device.type == 'cuda').float())
        else:
            moved.append(item)
    return tuple(moved)
