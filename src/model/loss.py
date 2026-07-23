import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

def CE_Loss(inputs, target, cls_weights, num_classes=21):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    CE_loss  = nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes)(temp_inputs, temp_target.long())
    return CE_loss


def Focal_Loss(inputs, target, cls_weights, num_classes=21, alpha=0.5, gamma=2):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    # logpt  = -nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes, reduction='none')(temp_inputs, temp_target.long())
    logpt = -nn.CrossEntropyLoss(weight=cls_weights, ignore_index=0, reduction='none')(temp_inputs,
                                                                                                 temp_target.long())
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt
    loss = loss.mean()
    return loss


def Dice_loss(inputs, target, beta=1, smooth=1e-5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = target.view(n, -1, ct)

    # --------------------------------------------#
    #   计算 Dice Loss
    # --------------------------------------------#
    tp = torch.sum(temp_target[..., :-1] * temp_inputs, axis=[0, 1])
    fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
    fn = torch.sum(temp_target[..., :-1], axis=[0, 1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss

# class KeypointLoss(nn.Module):
#     """关键点训练损失函数。"""
#
#     def __init__(self, sigmas) -> None:
#         """初始化 KeypointLoss。"""
#         super().__init__()
#         self.sigmas = sigmas
#
#     def forward(self, pred_kpts, gt_kpts, kpt_mask, area=None):
#         """计算关键点损失因子和欧氏距离损失。"""
#         d = (pred_kpts[..., 0] - gt_kpts[..., 0]) ** 2 + (pred_kpts[..., 1] - gt_kpts[..., 1]) ** 2
#         kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
#         e = d / (2 * (1 * self.sigmas) ** 2 + 1e-9)  # 公式版本
#         # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # 公式版本
#         # e = d / (2 * self.sigmas) ** 2 / (area + 1e-9) / 2  # cocoeval 版本
#         return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class KeypointLoss(nn.Module):

    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean', ):
        super(KeypointLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        '''
        logits 与 label 形状相同，label 类型为 long。
        参数：
            logits: 形状为 (N, ...) 的张量
            label: 形状为 (N, ...) 的张量
        '''

        # 计算损失
        logits = logits.float()  # 若 logits 为 fp16，则转为 fp32 计算
        # 表明以下两步不用求梯度
        # 可以看出alpha不是一个值，在正样本的情况下（1）其系数为alpha
        with torch.no_grad():
            alpha = torch.empty_like(logits).fill_(1 - self.alpha)
            alpha[label == 1] = self.alpha
        # 将输出结果映射到概率
        probs = torch.sigmoid(logits)
        # label==1的地方用probs代替，不等于1的地方用1 - probs代替
        pt = torch.where(label == 1, probs, 1 - probs)
        # 基础损失为 BCEWithLogitsLoss
        ce_loss = self.crit(logits, label.float())
        loss = (alpha * torch.pow(1 - pt, self.gamma) * ce_loss)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss
