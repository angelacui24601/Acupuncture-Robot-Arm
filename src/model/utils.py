import numpy as np

import sys
import time
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from sklearn.metrics import mean_squared_error, r2_score
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore

from copy import deepcopy
from typing import Tuple
from einops.layers.torch import Rearrange
from einops import repeat, rearrange


def PCA_svd(X, k, center=True):
    """
    参数：
        X: 形状为 BxCxHxW 的输入张量
        k: 主成分数量
    返回：
        降维后的特征张量
    """
    B, C, H, W = X.shape
    X = X.permute(0, 2, 3, 1)  # BxHxWxC
    X = X.reshape(B, H * W, C)
    U, S, V = torch.pca_lowrank(X, center=center)
    Y = torch.bmm(X, V[:, :, :k])
    Y = Y.reshape(B, H, W, k)
    Y = Y.permute(0, 3, 1, 2)  # BxHxWxk
    return Y


def calculate_metrics(y_true, y_pred):
    """处理真实标签"""
    y_true = y_true.detach().cpu().numpy()
    # y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    """处理预测结果"""
    y_pred = y_pred.detach().cpu().numpy()
    # y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    score_f1 = f1_score(y_true, y_pred,  average='macro')
    score_recall = recall_score(y_true, y_pred, average='macro')
    score_precision = precision_score(y_true, y_pred, average='macro')

    return score_f1, score_recall, score_precision

def RMSE(result,test_y):
    return np.sqrt(mean_squared_error(result, test_y))

def calculate_rmse(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_true = y_true.reshape(-1)
    y_pred = y_pred.detach().cpu().numpy()
    y_pred = y_pred.reshape(-1)
    """计算 RMSE"""
    score_RMSE = RMSE(y_pred, y_true)
    return score_RMSE

def calculate_nmse(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    # y_true = y_true > 0.5
    #
    y_true = y_true.reshape(-1)

    y_pred = y_pred.detach().cpu().numpy()
    y_pred = y_pred.reshape(-1)

    """计算 NMSE"""
    MSE = mean_squared_error(y_true, y_pred)
    NMSE = MSE / np.var(y_true)

    return NMSE

def calculate_ssim(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_true = y_true.reshape(-1)
    y_pred = y_pred.detach().cpu().numpy()
    y_pred = y_pred.reshape(-1)
    """计算 SSIM"""
    SSIM = structural_similarity(y_true, y_pred)
    return SSIM

def calculate_pnsr(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_true = y_true.reshape(-1)
    y_pred = y_pred.detach().cpu().numpy()
    y_pred = y_pred.reshape(-1)
    """计算 PSNR"""
    PNSR = peak_signal_noise_ratio(y_true, y_pred)
    return PNSR

def calculate_mse(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_true = y_true.reshape(-1)
    y_pred = y_pred.detach().cpu().numpy()
    y_pred = y_pred.reshape(-1)
    """计算 MSE"""
    MSE = mean_squared_error(y_true, y_pred)
    return MSE


def calculate_mean_point_error(y_true, y_pred):
    """计算平均关键点欧氏距离，适用于形状为 (B, K, 3) 的预测/真值。"""
    if isinstance(y_true, torch.Tensor):
        y_true_t = y_true.detach().cpu()
    else:
        y_true_t = torch.as_tensor(y_true)
    if isinstance(y_pred, torch.Tensor):
        y_pred_t = y_pred.detach().cpu()
    else:
        y_pred_t = torch.as_tensor(y_pred)

    if y_true_t.ndim == 2:
        y_true_t = y_true_t.unsqueeze(0)
    if y_pred_t.ndim == 2:
        y_pred_t = y_pred_t.unsqueeze(0)

    pred_coords = y_pred_t[..., :2].float()
    gt_coords = y_true_t[..., :2].float()

    if y_true_t.shape[-1] >= 3:
        valid_mask = y_true_t[..., 2] > 0
    else:
        valid_mask = torch.ones_like(gt_coords[..., 0], dtype=torch.bool)

    dist = torch.linalg.vector_norm(pred_coords - gt_coords, dim=-1)
    dist = dist[valid_mask]
    if dist.numel() == 0:
        return 0.0
    return float(dist.mean().item())


def calculate_pck(y_true, y_pred, thresholds):
    """计算 PCK（Percentage of Correct Keypoints）。"""
    if isinstance(y_true, torch.Tensor):
        y_true_t = y_true.detach().cpu()
    else:
        y_true_t = torch.as_tensor(y_true)
    if isinstance(y_pred, torch.Tensor):
        y_pred_t = y_pred.detach().cpu()
    else:
        y_pred_t = torch.as_tensor(y_pred)

    if y_true_t.ndim == 2:
        y_true_t = y_true_t.unsqueeze(0)
    if y_pred_t.ndim == 2:
        y_pred_t = y_pred_t.unsqueeze(0)

    pred_coords = y_pred_t[..., :2].float()
    gt_coords = y_true_t[..., :2].float()

    if y_true_t.shape[-1] >= 3:
        valid_mask = y_true_t[..., 2] > 0
    else:
        valid_mask = torch.ones_like(gt_coords[..., 0], dtype=torch.bool)

    dist = torch.linalg.vector_norm(pred_coords - gt_coords, dim=-1)
    if dist.ndim == 1:
        dist = dist.unsqueeze(0)

    scores = {}
    for threshold in thresholds:
        threshold_value = float(threshold)
        correct = (dist[valid_mask] <= threshold_value).float().mean().item()
        scores[str(threshold_value)] = float(correct)
    return scores

class ResizeLongestSide:
    """
    将图像最长边缩放到 target_length。
    同时提供坐标和框的同步缩放方法。
    支持 numpy 数组与 batched torch 张量两种输入。
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        输入应为 uint8 的 numpy 数组，形状 HxWxC。
        """
        target_size = self.get_preprocess_shape(image.shape[2], image.shape[3], self.target_length)
        return np.array(resize(to_pil_image(image), target_size))

    def apply_coords(self, coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        输入坐标最后一维长度应为 2。
        需要传入原始图像尺寸 (H, W)。
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        # coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        输入边界框形状应为 Bx4。
        需要传入原始图像尺寸 (H, W)。
        """
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    def apply_image_torch(self, image: torch.Tensor) -> torch.Tensor:
        """
        输入应为浮点格式的 batched 图像，形状 BxCxHxW。
        该变换与 apply_image 可能不完全一致。
        模型默认期望的是 apply_image 的变换行为。
        """
        # 输入格式为 BCHW，与 apply_image 结果可能存在轻微差异
        target_size = self.get_preprocess_shape(image.shape[2], image.shape[3], self.target_length)
        return F.interpolate(
            image, target_size, mode="bilinear", align_corners=False, antialias=True
        )

    def apply_coords_torch(
        self, coords: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        输入应为 torch 张量，最后一维长度为 2。
        需要传入原始图像尺寸 (H, W)。
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).to(torch.float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes_torch(
        self, boxes: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        输入应为 torch 张量，形状 Bx4。
        需要传入原始图像尺寸 (H, W)。
        """
        boxes = self.apply_coords_torch(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        根据输入尺寸和目标最长边计算输出尺寸。
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)