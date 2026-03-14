# https://github.com/timHau/vit_mlx/blob/main/src/vit_mlx/vit.py
import mlx.core as mx 
import mlx.nn as nn
import mlx.optimizers as optim

import torch
import torch.nn as tch_nn
import numpy as np
from torchvision import models
from config import Config
import timm
import torch.nn.functional as F

def semantic_encoder():
	config = Config()
	# 336pxで事前学習されたモデル（CLIP）を直接呼ぶ
	model = timm.create_model('vit_base_patch16_clip_224.openai', pretrained=True, num_classes=config.NUM_CLASSES, img_size=config.TARGET_SIZE[0])

	model = model.to(config.device)
	return model