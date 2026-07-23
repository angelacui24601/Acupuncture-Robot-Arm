from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.engine.runtime import move_batch_to_device


PALETTE = [
    [30, 144, 255],
    [0, 255, 0],
    [255, 0, 0],
    [0, 255, 255],
    [255, 0, 255],
    [255, 255, 0],
    [128, 0, 255],
    [255, 128, 0],
    [138, 43, 226],
    [128, 128, 128],
    [255, 240, 245],
    [144, 238, 144],
    [176, 196, 222],
    [0, 0, 205],
    [220, 20, 60],
    [85, 107, 47],
    [0, 191, 255],
    [173, 255, 47],
    [230, 230, 250],
    [135, 206, 250],
    [186, 85, 211],
    [218, 112, 214],
    [160, 82, 45],
    [255, 99, 71],
]


def _colorize_mask(mask):
    colorized = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_index, color in enumerate(PALETTE):
        colorized[mask == class_index] = color
    return colorized


def plot_segmentation(image, pred_mask, label_mask):
    image = (image * 255).astype(np.uint8).transpose(1, 2, 0)
    pred_mask = pred_mask.copy()
    pred_mask[label_mask == 0] = 0

    pred_result = _colorize_mask(pred_mask)
    label_result = _colorize_mask(label_mask)
    white_bar = (np.ones((image.shape[0], 10, 3)) * 255).astype(np.uint8)

    result = np.concatenate([image, white_bar, pred_result, white_bar, label_result], axis=1)
    plt.imshow(result)
    plt.show()


@torch.no_grad()
def visualize_predictions(model, data_loader, opt, device):
    model.eval()
    max_vis = max(1, int(opt.max_vis_samples))

    for index, batch in enumerate(data_loader):
        if index >= max_vis:
            break

        img, acupoint, segm, _seg_label, _keypoint_label = move_batch_to_device(batch, device)
        masks, _points = model(img, acupoint)
        pred = masks.argmax(dim=1)
        plot_segmentation(
            img[0].detach().cpu().numpy(),
            pred[0].detach().cpu().numpy(),
            segm[0].detach().cpu().numpy().astype(np.int64),
        )