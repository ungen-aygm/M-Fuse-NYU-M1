import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import random_split, DataLoader
from torch import optim
from torchvision.transforms import (
    Compose,
    ToTensor,
    Normalize,
)
from utils.augmentations import Augmentation, AugmentationParams
from utils.dataset import ColorMap, NYUv2Dataset, NYUv2DatasetTest
from utils.preprocess import Preprocess as Loader
from sklearn.model_selection import train_test_split
from models.vit import semantic_encoder
from models.unet import UNet as unet
from models.late_fusion import LateFusion
from utils.diceloss import FocalDiceLoss as HybridLoss
from utils.normalize import ImageNet as nm
from utils.functions import *
from config import Config

class SegSetup(nn.Module):
	def __init__(self, config:Config):
		super(SegSetup, self).__init__()
		self.config = config

	def __call__(self, is_load_model=False, is_freeze=False):
		model = LateFusion(semantic_encoder, unet).to(self.config.device)
		model = load_model(model, self.config.LATEST_MODEL, is_load=is_load_model)
		# モデル内すべての層の入力前にメモリを整理する
		for module in model.modules():
			module.register_forward_pre_hook(ensure_contiguous_hook)
		
		# ViTのフリーズ（必要な場合だけ）
		if is_freeze is True:
			self.freeze_model(model)

		# 最適化関数
		optimizer = None
		if is_freeze is True:
			# フリーズ：最適化関数
			optimizer = self.optimizer_freeze(model)
		else:
			optimizer = self.optimizer(model)

		# 目的関数
		criterion = self.loss_fs()
		# 学習率管理
		scheduler = self.scheduler(optimizer)
		return model, optimizer, criterion, scheduler
	
	def freeze_model(self, model):
		for param in model.vit.parameters():
			param.requires_grad = False
	
	def optimizer_freeze(self, model):
		return optim.AdamW(
			filter(lambda p: p.requires_grad, model.parameters()),
			lr=self.config.LR["base"], # デコーダーをしっかり動かす
			weight_decay=1e-2,
		)
	
	def optimizer(self, model):
		return optim.AdamW([
			{'params': model.vit.parameters(), 'lr': self.config.LR["base"] * self.config.LR["ViT"]},
			{'params': model.unet.parameters(), 'lr': self.config.LR["base"] * self.config.LR["UNet"]},
			{'params': model.fusion_layer.parameters(), 'lr': self.config.LR["base"] * self.config.LR["LateFusion"]},
		])
	
	# Focal Loss + Dice Loss
	def loss_fs(self):
		return HybridLoss(ignore_index=self.config.IGNORE_INDEX, focal_weight=self.config.FOCALLOSS, dice_weight=self.config.DICELLOSS, weight=self.config.CLASS_WEIGHT.to(self.config.device).float())

	# 学習率管理
	def scheduler(self, optimizer):
		return optim.lr_scheduler.ReduceLROnPlateau(
			optimizer, 
			mode=self.config.LR_MODE,    
			factor=self.config.FACTOR,   
			patience=self.config.PATIENCE,    
			min_lr=self.config.MIN_LR
		)

class PrepareDatasets:
	def __init__(self, config:Config, mode="train", img=None, depth=None, mask=None):
		if mode == "test":
			# 評価用
			loader = Loader()
			self.test_image, self.test_depth, self.test_mask = loader.loads(is_test=True)
		else:
			if mode == "train":
				self.train_img, self.train_depth, self.train_mask = img, depth, mask
			elif mode == "valid":
				self.valid_img, self.valid_depth, self.valid_mask = img, depth, mask

		self.config = config
		self.mode = mode
		self.transform = self.trans()
		self.transform_depth = self.trans_depth()
		self.is_augmentation = True if mode=="train" else False
		self.augmentation = Augmentation(AugmentationParams(
			degrees=config.DEGREES, # Rotate（角度）
			hflip=config.HFLIP,# 左右反転確率(0.0-1.0)
			crop=config.CROP,
			crop_size=config.TARGET_SIZE,
			crop_scale=config.CROP_SCALE,
			crop_ratio=config.CROP_RATIO,
		))
	
	def __call__(self, augmentation=None, is_augmentation=False, batch_scale=2):
		if self.mode=="train":
			return DataLoader(self.train_dataset(augmentation, is_augmentation), batch_size=self.config.BATCH_SIZE, shuffle=True, num_workers=0)

		elif self.mode == "valid":
			return DataLoader(self.vaild_dataset(), batch_size=self.config.BATCH_SIZE * batch_scale, shuffle=False, num_workers=0)

		elif self.mode == "test":
			return DataLoader(self.test_dataset(), batch_size=self.config.BATCH_SIZE * batch_scale, shuffle=False, num_workers=0)

	def train_dataset(self, augmentation=None, is_augmentation=False):
		return NYUv2Dataset(
			self.train_img,
			self.train_depth,
			self.train_mask,
			transform=self.transform,
			depth_transform=self.transform_depth,
			augmentation=augmentation,
			is_aug=is_augmentation# 使う時だけ、is_augをTrueへ
		)
	def vaild_dataset(self):
		return NYUv2Dataset(
			self.valid_img,
			self.valid_depth,
			self.valid_mask,
			transform=self.transform,
			depth_transform=self.transform_depth,
			is_aug=False
		)
	def test_dataset(self):
		return NYUv2Dataset(
			self.test_image,
			self.test_depth,
			self.test_mask,
			transform=self.transform,
			depth_transform=self.transform_depth,
			is_aug=False
		)

	# 正規化と次元化
	def trans(self):
		return Compose([
			ToTensor(),
			Normalize(mean=nm.MEAN, std=nm.STD),
		])
	# Tensor化
	def trans_depth(self):
		return Compose([
			ToTensor(),
		])

# データセット作成（訓練・検証）
def split_data():
	config = Config()
	loader = Loader()
	image, depth, mask = loader.loads()
	train_img, valid_img, train_depth, valid_depth, train_mask, valid_mask = train_test_split(
		image, depth, mask, test_size=config.SPLIT_SIZE, random_state=config.RANDOM_STATE
	)
	return train_img, valid_img, train_depth, valid_depth, train_mask, valid_mask