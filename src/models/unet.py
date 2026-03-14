# https://github.com/ml-explore/mlx-examples/tree/main/llms/llama
import mlx.core as mx
import mlx.nn as mlx_nn
import mlx.optimizers as optim

from config import Config
import torchvision.models as models
import numpy as np
import torch
import torch.nn as nn

class UNet(nn.Module):
	def __init__(self, config: Config):
		super().__init__()
		# Encoder
		self.encoder = Resnet18()
		# Decorder: 512 -> ... -> 32
		self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
		self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
		self.up3 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
		self.up4 = nn.ConvTranspose2d(64, 128, kernel_size=2, stride=2)

		# 出力（特徴量）- Resnet-18
		features_dimension = 128

		self.mask_head = nn.Conv2d(features_dimension, config.NUM_CLASSES, kernel_size=1)
		self.depth_head = nn.Conv2d(features_dimension, config.NUM_CLASSES, kernel_size=1)
	
	def encode(self, x):
		# エンコーダ：特徴とスキップ接続用の特徴量を返す
		x1 = nn.functional.relu(self.encoder.bn1(self.encoder.conv1(x)))
		x2 = self.encoder.layer1(self.encoder.maxpool(x1))
		x3 = self.encoder.layer2(x2)
		y = self.encoder.layer3(x3)
		return y, (x1, x2, x3)

	def decode(self, y, skips):
		# 次元をあわせた状態でViTの特徴（意味）とUNetエンコーダ(Resnet)の特徴とスキップ接続用の特徴量をパッチサイズから128chに統一して復元する
		x1, x2, x3 = skips
		y = nn.functional.relu(self.up1(y))
		# skip connections
		y = (y + x3).contiguous()
		y = nn.functional.relu(self.up2(y))
		y = (y + x2).contiguous()
		y = nn.functional.relu(self.up3(y))
		y = (y + x1).contiguous()
		y = nn.functional.relu(self.up4(y))
		return y 

# Resnet Based
class Resnet(nn.Module):
	def __init__(self, in_channels, out_channels, stride=1):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
		self.bn2 = nn.BatchNorm2d(out_channels)

		self.sequential = nn.Sequential()

		if stride != 1 or in_channels != out_channels:
			self.sequential = nn.Sequential(
				nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
				nn.BatchNorm2d(out_channels)
			)
	def forward(self, x):
		y = nn.functional.relu(self.bn1(self.conv1(x)))
		y = self.bn2(self.conv2(y))
		y += self.sequential(x) if self.sequential is not None else x
		return nn.functional.relu(y)

# Resnet18 based custom encoder
class Resnet18(nn.Module):
	def __init__(self, in_channels=1):
		super().__init__()
		
		self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

		self.layer1 = self.layers(64, 64, 2, stride=1)
		self.layer2 = self.layers(64, 128, 2, stride=2)
		self.layer3 = self.layers(128, 256, 2, stride=2)

	def layers(self, in_channels, out_channels, blocks, stride):
		layers = [Resnet(in_channels, out_channels, stride)]
		for _ in range(1, blocks):
			layers.append(Resnet(out_channels, out_channels))
		return nn.Sequential(*layers)
	
	def forward(self, x):
		x = nn.functional.relu(self.bn1(self.conv1(x)))
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		return x