import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, Type, Any
import numpy as np
import torch.nn.functional as F
from .common import LayerNorm2d, MLPBlock, Adapter
from .mask_decoder import MaskDecoder
from .transformer import TwoWayTransformer
from .image_encoder import ImageEncoderViT
import math
import warnings
from itertools import repeat
import torch.distributed as dist
import torchvision
import mediapipe as mp
from einops import repeat, rearrange
from einops.layers.torch import Rearrange

class PositionEmbeddingRandom(nn.Module):
    """
    使用随机空间频率的位置编码。
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """对归一化到 [0,1] 的点进行位置编码。"""
        # 假设 coords 位于 [0,1]^2，形状为 d_1 x ... x d_n x 2
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # 输出形状为 d_1 x ... x d_n x C
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: int) -> torch.Tensor:
        """为指定大小网格生成位置编码。"""
        h, w = size, size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

class AcupointMM(nn.Module):
    def __init__(self, opts):
        super().__init__()

        self.image_size = opts.image_size
        self.patch_size = opts.patch_size
        self.in_chans = opts.in_chans
        self.embed_dim = opts.embed_dim
        self.depth = opts.depth
        self.num_heads = opts.num_heads
        self.mlp_ratio = opts.mlp_ratio
        self.out_chans = opts.out_chans

        self.prompt_embed_dim = opts.prompt_embed_dim
        self.ablate_no_prompt = bool(getattr(opts, "ablate_no_prompt", False))

        self.shared_mlp = nn.Linear(self.embed_dim // 32, self.embed_dim)
        for i in range(self.depth):

            lightweight_mlp = nn.Sequential(
                nn.Linear(16 * 3, self.embed_dim // 32),
                nn.GELU(),
                #nn.Linear(self.embed_dim//self.scale_factor, self.embed_dim)
            )
            setattr(self, 'lightweight_mlp_{}'.format(str(i)), lightweight_mlp)

        self.encoder = ImageEncoderViT(
            img_size=self.image_size,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            out_chans=self.out_chans,
            qkv_bias=True,
            norm_layer=nn.LayerNorm,
            act_layer=nn.GELU,
            use_abs_pos=True,
            use_rel_pos=False,
            rel_pos_zero_init=True,
            window_size=0
        )

        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            num_classes=opts.num_classes
        )

        self.pe_layer = PositionEmbeddingRandom(self.prompt_embed_dim // 2)
        self.image_embedding_size = self.image_size // self.patch_size
        self.no_mask_embed = nn.Embedding(1, self.prompt_embed_dim)

        self.keypoint_head = nn.Sequential(
            nn.Linear(self.prompt_embed_dim, self.prompt_embed_dim * 2),
            nn.GELU(),
            nn.Linear(self.prompt_embed_dim * 2, self.prompt_embed_dim * 2),
            nn.GELU(),
            nn.Linear(self.prompt_embed_dim * 2, 16 * 3),
            nn.GELU(),
        )


    def forward(self, img, acupoint):
        bs, c, h, w = img.shape
        bs_ac, n_ac, c_ac = acupoint.shape

        prompt = self.get_prompt(acupoint)
        features = self.encoder(img, prompt)

        segm_features = rearrange(
            features[:, 1:, :],
            "b (h w) c -> b c h w",
            h=self.image_embedding_size,
            w=self.image_embedding_size,
        )
        keypoint_features = features[:, 0, :]

        # 构造提示嵌入
        sparse_embeddings = torch.empty((bs, 0, self.prompt_embed_dim)).cuda(non_blocking=True)
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )

        # 预测分割掩码
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=segm_features,
            image_pe=self.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        # 将低分辨率掩码上采样到输入分辨率
        masks = self.postprocess_masks(low_res_masks, self.image_size, self.image_size)
        points = self.keypoint_head(keypoint_features).view(bs_ac, n_ac, c_ac)

        return masks, points


    def get_prompt(self, x):
        B, N, C = x.shape
        if self.ablate_no_prompt:
            return [
                torch.zeros((B, self.embed_dim), device=x.device, dtype=x.dtype)
                for _ in range(self.depth)
            ]

        handcrafted_feature = x.view(B, N * C)
        prompts = []
        for i in range(self.depth):
            lightweight_mlp = getattr(self, 'lightweight_mlp_{}'.format(str(i)))
            prompt = lightweight_mlp(handcrafted_feature)
            prompts.append(self.shared_mlp(prompt))
        return prompts

    def get_dense_pe(self) -> torch.Tensor:
        """
                返回用于点提示编码的稠密位置编码。
                其空间尺寸与图像编码一致。

        Returns:
                    torch.Tensor: 形状为
                        1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
                去除填充并将掩码上采样到目标图像尺寸。

        Arguments:
                    masks (torch.Tensor): 来自 mask_decoder 的批量掩码，
                        形状为 BxCxHxW。
                    input_size (tuple(int, int)): 模型输入尺寸 (H, W)，
                        用于裁剪填充区域。
                    original_size (tuple(int, int)): 原始图像尺寸 (H, W)。

        Returns:
                    (torch.Tensor): 批量掩码，形状为 BxCxHxW，
                        其中 (H, W) 为 original_size。
        """
        masks = masks[:,0,:,:]
        masks = F.interpolate(
            masks,
            (input_size, input_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size, : input_size]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks
