from models.vit import semantic_encoder
from models.unet import UNet as unet

from config import Config
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# LateFusion(vit, unet).to(device)
class LateFusion(nn.Module):
	def __init__(self, vit:semantic_encoder, unet:unet, num_classes=13):
		super().__init__()
		# 意味抽出（CLIP-ViT）を初期化
		self.vit = semantic_encoder()
		# 幾何抽出（UNet/ResNet）を初期化
		self.unet = unet(Config())
		
		self.vit_adapter = nn.Sequential(
			nn.Conv2d(768, 256, kernel_size=1),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True)
		)

		# 最終的なUNet出力(u)との融合
		self.fusion_layer = nn.Sequential(
			nn.Conv2d(512, 256, kernel_size=3, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			nn.Dropout2d(p=0.3),
		)

		# Class分類
		self.classifier = nn.Sequential(
			nn.Conv2d(128, 128, kernel_size=3, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
			nn.Dropout2d(p=0.2), # クラス判定の過学習を抑制
			nn.Conv2d(128, num_classes, kernel_size=1),
		)

	def forward(self, rgb, depth):

		u, skips = self.unet.encode(depth)
		# ViT側 (14(21)x14(21) 特徴量 v を抽出)/特徴入力
		x = self.vit.forward_features(rgb)

		B, N, C = x.shape
		num_patches = N - 1
		grid_size = int(round(num_patches**0.5))
		v = x[:, 1:, :].permute(0, 2, 1).reshape(B, C, grid_size, grid_size) # ViT:[B, 768, 14, 14]
		
		# ViTパッチを段階的に復元する: 224 => 336
		v = self.vit_adapter(v) # 14(21)x14(21) [B, 256, 14, 14]
		fused = torch.cat([v, u], dim=1) # 14(21)x14(21), 512ch=>256ch
		fused = self.fusion_layer(fused) # 14(21)x14(21), 256ch

		# UNetデコーダ：UNetのDepth特徴(u)とViTの意味(v)を結合
		combined = self.unet.decode(fused, skips) # 128ch
		return self.classifier(combined) # 128ch