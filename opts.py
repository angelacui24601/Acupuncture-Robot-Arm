from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import json
import torch

class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser("HSNet")
        # 基础实验配置
        self.parser.add_argument(
            '--dataset',
            default='acusim',
            choices=['hand', 'cervicocranial', 'acusim'],
            help='数据集名称'
        )
        self.parser.add_argument('--test_dataset', default='', help='测试集名称，可为空')
        self.parser.add_argument('--exp_id', default='default')
        self.parser.add_argument('--test', action='store_true')
        self.parser.add_argument('--no_pause', action='store_true')
        self.parser.add_argument('--load_model', default='', help='预训练模型路径')
        self.parser.add_argument('--output_root', default='../results', help='实验输出根目录')

        # 系统配置
        self.parser.add_argument('--gpus', default='2,3', help='CPU 使用 -1，多卡用逗号分隔')
        self.parser.add_argument('--num-workers', type=int, default=0, help='DataLoader 线程数，0 表示单线程')
        self.parser.add_argument('--not_cuda_benchmark', action='store_true', help='输入尺寸不固定时关闭 cudnn benchmark')
        self.parser.add_argument('--seed', type=int, default=317, help='随机种子')  # 来自 CornerNet 常用设置
        self.parser.add_argument('--not_set_cuda_env', action='store_true', help='在 slurm 等集群环境下使用')

        # 模型配置
        self.parser.add_argument(
            '--arch',
            default='acupointmm',
            choices=['acupointmm', 'baseline_cnn'],
            help='模型名称'
        )
        self.parser.add_argument('--patch_size', default=16, help='patch 大小')
        self.parser.add_argument('--in_chans', default=3, help='输入通道数（RGB=3）')
        self.parser.add_argument('--embed_dim', default=768, help='ViT 嵌入维度')
        self.parser.add_argument('--depth', default=12, help='ViT 层数')
        self.parser.add_argument('--num_heads', default=12, help='ViT 注意力头数')
        self.parser.add_argument('--mlp_ratio', default=4.0, help='MLP 扩展比例')
        self.parser.add_argument('--out_chans', default=256, help='输出通道数')

        self.parser.add_argument('--prompt_embed_dim', default=256, help='提示向量维度，通常与 out_chans 相同')

        # QuadTree 配置
        self.parser.add_argument('--min_patch_size', type=int, default=16, help='QuadTree 最小 patch 尺寸')
        self.parser.add_argument('--max_patch_size', type=int, default=64, help='QuadTree 最大 patch 尺寸')
        self.parser.add_argument('--num_patches', type=int, default=100, help='patch 数量')
        self.parser.add_argument('--num_scales', type=int, default=3, help='输入尺度数量')

        # 输入与数据配置
        self.parser.add_argument('--image_size', type=int, default=512, help='输入图像尺寸')
        self.parser.add_argument('--num_classes', type=int, default=15, help='分割类别数')
        self.parser.add_argument('--cervico_dataset_root', type=str, default='acuSim/dataset/main/dataset',
                     help='CervicoCranial 数据根目录，需包含 train/image 和 train/label')
        self.parser.add_argument('--cervico_test_image_dir', type=str, default='acuSim/test_imgs',
                     help='CervicoCranial 验证图像兜底目录')
        self.parser.add_argument('--cervico_test_label_dir', type=str, default='acuSim/test_label',
                     help='CervicoCranial 验证标注兜底目录')
        self.parser.add_argument('--cervico_image_subdir', type=str, default='img_512',
                     help='图像分辨率子目录，例如 img_512')
        self.parser.add_argument('--cervico_map_file', type=str, default='',
                     help='可选：显式指定 map.txt 路径，用于关键点顺序')
        self.parser.add_argument('--cervico_keypoints', type=str, default='',
                     help='逗号分隔的关键点名称；为空时默认读取 map.txt 前 16 项')

        # 训练配置
        self.parser.add_argument('--optimizer', default='adam', choices=['sgd', 'lars', 'adam'], help='优化器类型')
        self.parser.add_argument('--base-lr', type=float, default=1.0, help='基础学习率（按 batch size 线性缩放）')
        self.parser.add_argument('--num_epochs', type=int, default=400, help='训练总轮数')
        self.parser.add_argument('--start_epoch', type=int, default=1, help='恢复训练时的起始轮数')
        self.parser.add_argument('--batch_size', type=int, default=2, help='批大小')
        self.parser.add_argument('--warmup-epoch', type=int, default=5, help='warmup 轮数')
        self.parser.add_argument('--warmup-multiplier', type=int, default=100, help='warmup 放大倍数')
        self.parser.add_argument('--weight-decay', type=float, default=1e-5, help='权重衰减')
        self.parser.add_argument('--momentum', type=float, default=0.9, help='SGD 动量')
        self.parser.add_argument('--fp16', action='store_true', default=False, help='是否启用自动混合精度')
        self.parser.add_argument('--master_batch_size', type=int, default=-1,help='主卡 batch 大小')
        self.parser.add_argument('--save_point', type=str, default='10,120', help='额外保存检查点的轮数')
        self.parser.add_argument('--save-freq', type=str, default='30', help='按固定间隔保存检查点')
        self.parser.add_argument('--auto-resume', action='store_true', help='是否从 current.pth 自动恢复')
        self.parser.add_argument('--print-freq', type=int, default=10, help='日志打印间隔')
        self.parser.add_argument('--seg_loss_weight', type=float, default=1.0, help='分割损失权重')
        self.parser.add_argument('--kpt_loss_weight', type=float, default=1.0, help='关键点损失权重')



        # 测试配置
        self.parser.add_argument('--flip-test', action='store_true', help='测试时是否使用翻转增强')
        self.parser.add_argument('--save-log', default='False', help='是否保存日志')
        self.parser.add_argument('--eval_only', action='store_true', help='仅执行评估并输出指标文件，不做可视化')
        self.parser.add_argument('--max_vis_samples', type=int, default=20, help='可视化模式下最多展示样本数')
        self.parser.add_argument('--pck_thresholds', type=str, default='0.02,0.05,0.1', help='PCK 阈值列表，逗号分隔')
        self.parser.add_argument('--topk_worst', type=int, default=20, help='导出误差最高样本数量')

        # 消融配置
        self.parser.add_argument('--ablate_no_prompt', action='store_true', help='消融：关闭 prompt 注入')

        # 掩码配置
        self.parser.add_argument('--mim_masking_ratio', type=float, default=0.7, help='MIM 掩码比例')
        self.parser.add_argument('--mim_intermediate_losses', type=bool, default=False, help='是否启用中间层损失')
        self.parser.add_argument('--mim_mask_patch_size', type=int, default=4, help='MIM 掩码 patch 大小')
        self.parser.add_argument('--spectral_pos_embed', type=bool, default=False, help='是否使用光谱位置编码')
        self.parser.add_argument('--blockwise_patch_embed', type=bool, default=True, help='是否启用块级 patch 嵌入')
        self.parser.add_argument('--spectral_only', type=bool, default=False, help='是否仅使用光谱分支')
        self.parser.add_argument('--to_pixels_per_spectral_block', type=bool, default=False, help='是否按光谱块映射到像素')
        self.parser.add_argument('--tube_masking', type=bool, default=True, help='是否启用 tube masking')


    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        if opt.test_dataset == '':
            opt.test_dataset = opt.dataset

        opt.gpus_str = opt.gpus
        opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
        # opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >=0 else [-1]
        # opt.lr_step = [int(i) for i in opt.lr_step.split(',')]
        opt.save_point = [int(i) for i in opt.save_point.split(',')]
        opt.pck_thresholds = [float(x) for x in opt.pck_thresholds.split(',') if x.strip()]
        if len(opt.pck_thresholds) == 0:
            opt.pck_thresholds = [0.05]
        opt.seg_loss_weight = max(0.0, float(opt.seg_loss_weight))
        opt.kpt_loss_weight = max(0.0, float(opt.kpt_loss_weight))
        opt.output_root = os.path.normpath(opt.output_root)
        # opt.num_workers = max(opt.num_workers, 2 * len(opt.gpus))



        # # log dirs
        # opt.root_dir = os.path.join(os.path.dirname(__file__), '..', '..')
        # opt.data_dir = os.path.join(opt.root_dir, 'data')
        # opt.exp_dir = os.path.join(opt.root_dir, 'exp', opt.task)
        # opt.save_dir = os.path.join(opt.exp_dir, opt.exp_id)
        # opt.debug_dir = os.path.join(opt.save_dir, 'debug')
        #
        # if opt.resume and opt.load_model == '':
        #     opt.load_model = os.path.join(opt.save_dir, 'model_last.pth')
        return opt


    def init(self, args=''):
        print("OK")

