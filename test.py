from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

from opts import opts
from src.engine import build_dataloader, evaluate_segmentation, resolve_output_dir, set_random_seed, setup_runtime, visualize_predictions
from src.model.model import create_model, load_checkpoint


def main(opt):
    resolve_output_dir(opt)
    runtime, logger = setup_runtime(opt)

    _train_dataset, _train_loader, _train_sampler = build_dataloader(
        opt,
        phase='train',
        batch_size=opt.batch_size,
        is_distributed=runtime.is_distributed,
        shuffle=True,
        drop_last=True,
    )
    valid_dataset, valid_loader, _valid_sampler = build_dataloader(
        opt,
        phase='val',
        batch_size=max(1, opt.batch_size),
        is_distributed=runtime.is_distributed,
        shuffle=False,
        drop_last=True,
    )

    opt.num_instances = len(valid_dataset)
    logger.info(f'length of validation dataset: {opt.num_instances}')
    logger.info("=> creating model '%s'", opt.arch)

    model = create_model(opt=opt).to(runtime.device)
    if opt.load_model:
        load_checkpoint(logger, opt, model, type='plot')

    if runtime.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.local_rank],
            find_unused_parameters=True,
        )

    if opt.eval_only:
        evaluate_segmentation(model, valid_loader, logger, opt, runtime.device, is_distributed=runtime.is_distributed)
    else:
        visualize_predictions(model, valid_loader, opt, runtime.device)


if __name__ == '__main__':
    options = opts().parse()
    set_random_seed(options.seed)
    main(options)

