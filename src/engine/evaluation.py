from __future__ import annotations

import csv
import json
import os

import numpy as np
import torch
import torch.distributed as dist

from src.engine.runtime import move_batch_to_device
from src.model.utils import calculate_mean_point_error, calculate_mse, calculate_nmse, calculate_pck, calculate_rmse


def compute_segmentation_metrics(pred, target, num_classes: int, ignore_index: int = -1):
    pred = np.asarray(pred, dtype=np.int64)
    target = np.asarray(target, dtype=np.int64)

    valid_mask = target != ignore_index
    pred = pred[valid_mask]
    target = target[valid_mask]

    if target.size == 0:
        return {'iou': np.zeros(num_classes, dtype=np.float64), 'miou': 0.0, 'oa': 0.0}

    in_range = (target >= 0) & (target < num_classes)
    pred = pred[in_range]
    target = target[in_range]

    confusion = np.bincount(
        target * num_classes + pred,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes).astype(np.float64)

    true_positive = np.diag(confusion)
    target_count = confusion.sum(axis=1)
    pred_count = confusion.sum(axis=0)
    union = target_count + pred_count - true_positive
    iou = np.divide(true_positive, union, out=np.zeros_like(true_positive), where=union > 0)
    valid_classes = target_count > 0
    miou = float(iou[valid_classes].mean()) if np.any(valid_classes) else 0.0
    oa = float(true_positive.sum() / max(1.0, confusion.sum()))

    return {'iou': iou, 'miou': miou, 'oa': oa}


def _flatten_sample_metrics(row, thresholds):
    flat_row = {
        'sample_index': row['sample_index'],
        'miou': row['miou'],
        'oa': row['oa'],
        'mse': row['mse'],
        'rmse': row['rmse'],
        'nmse': row['nmse'],
        'point_error': row['point_error'],
    }
    for threshold in thresholds:
        flat_row[f'pck_{threshold}'] = row['pck'].get(str(threshold), 0.0)
    return flat_row


@torch.no_grad()
def evaluate_segmentation(model, data_loader, logger, opt, device, is_distributed: bool = False):
    model.eval()
    metric_device = device if device.type == 'cuda' else torch.device('cpu')

    metric_sums = {
        'miou': 0.0,
        'oa': 0.0,
        'mse': 0.0,
        'rmse': 0.0,
        'nmse': 0.0,
        'count': 0.0,
    }
    class_iou_sum = np.zeros(int(opt.num_classes), dtype=np.float64)
    per_sample_rows = []
    sample_index = 0

    for batch in data_loader:
        img, acupoint, segm, _seg_label, keypoint_label = move_batch_to_device(batch, device)

        masks, points = model(img, acupoint)
        pred = masks.argmax(dim=1)

        batch_size = int(img.size(0))
        for sample_offset in range(batch_size):
            sample_pred = pred[sample_offset].detach().cpu().numpy()
            sample_segm = segm[sample_offset].type(torch.int64).detach().cpu().numpy()
            seg_metrics = compute_segmentation_metrics(sample_pred, sample_segm, int(opt.num_classes))

            point_pred = points[sample_offset:sample_offset + 1]
            point_gt = keypoint_label[sample_offset:sample_offset + 1]

            metric_sums['miou'] += seg_metrics['miou']
            metric_sums['oa'] += seg_metrics['oa']
            metric_sums['mse'] += float(calculate_mse(point_pred, point_gt))
            metric_sums['rmse'] += float(calculate_rmse(point_pred, point_gt))
            metric_sums['nmse'] += float(calculate_nmse(point_pred, point_gt))
            metric_sums['count'] += 1.0
            class_iou_sum += seg_metrics['iou']

            per_sample_rows.append(
                {
                    'sample_index': sample_index,
                    'miou': seg_metrics['miou'],
                    'oa': seg_metrics['oa'],
                    'mse': float(calculate_mse(point_pred, point_gt)),
                    'rmse': float(calculate_rmse(point_pred, point_gt)),
                    'nmse': float(calculate_nmse(point_pred, point_gt)),
                    'point_error': float(calculate_mean_point_error(point_pred, point_gt)),
                    'pck': calculate_pck(point_pred.detach().cpu(), point_gt.detach().cpu(), opt.pck_thresholds),
                }
            )
            sample_index += 1

    metric_tensor = torch.tensor(
        [
            metric_sums['miou'],
            metric_sums['oa'],
            metric_sums['mse'],
            metric_sums['rmse'],
            metric_sums['nmse'],
            metric_sums['count'],
        ],
        device=metric_device,
        dtype=torch.float64,
    )
    class_iou_tensor = torch.tensor(class_iou_sum, device=metric_device, dtype=torch.float64)

    if is_distributed:
        dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(class_iou_tensor, op=dist.ReduceOp.SUM)

    total_count = max(1.0, float(metric_tensor[5].item()))
    metrics = {
        'miou': float(metric_tensor[0].item() / total_count),
        'oa': float(metric_tensor[1].item() / total_count),
        'mse': float(metric_tensor[2].item() / total_count),
        'rmse': float(metric_tensor[3].item() / total_count),
        'nmse': float(metric_tensor[4].item() / total_count),
        'samples': int(total_count),
    }

    class_iou_avg = (class_iou_tensor / total_count).detach().cpu().numpy().tolist()
    metrics['class_iou'] = class_iou_avg
    metrics['worst_class_iou'] = float(np.min(class_iou_avg)) if class_iou_avg else 0.0

    pck_summary = {}
    if per_sample_rows:
        for threshold in opt.pck_thresholds:
            values = [row['pck'].get(str(threshold), 0.0) for row in per_sample_rows]
            pck_summary[f'pck_{threshold}'] = float(np.mean(values)) if values else 0.0
    metrics['pck'] = pck_summary

    detail_path = os.path.join(opt.output_dir, 'evaluation_details.csv')
    if per_sample_rows:
        flat_rows = [_flatten_sample_metrics(row, opt.pck_thresholds) for row in per_sample_rows]
        with open(detail_path, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0].keys()))
            writer.writeheader()
            writer.writerows(flat_rows)

    worst_path = os.path.join(opt.output_dir, 'worst_samples.csv')
    if per_sample_rows and opt.topk_worst > 0:
        worst_rows = sorted(per_sample_rows, key=lambda row: row['point_error'], reverse=True)[: opt.topk_worst]
        flat_worst = [_flatten_sample_metrics(row, opt.pck_thresholds) for row in worst_rows]
        with open(worst_path, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat_worst[0].keys()))
            writer.writeheader()
            writer.writerows(flat_worst)

    output_path = os.path.join(opt.output_dir, 'metrics_summary.json')
    with open(output_path, 'w', encoding='utf-8') as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)

    logger.info('评估完成，指标已保存: %s', output_path)
    logger.info('评估指标: %s', metrics)
    logger.info('评估详情已保存: %s', detail_path)
    logger.info('最差样本已保存: %s', worst_path)
    return metrics
