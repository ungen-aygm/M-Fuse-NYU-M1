import numpy as np
import os, random
from PIL import Image
import torch
from torchvision.datasets import VisionDataset
from config import Config
from glob import glob as find

class ColorMap:
	def __init__(self):
		pass
	def __call__(self, N=256, normalized=False):
		dtype = 'float32' if normalized else 'uint8'
		cmap = np.zeros((N, 3), dtype=dtype)
		for i in range(N):
			r = g = b = 0
			c = i
			for j in range(8):
				r = r | (self.bitget(c, 0) << 7-j)
				g = g | (self.bitget(c, 1) << 7-j)
				b = b | (self.bitget(c, 2) << 7-j)
				c = c >> 3
			cmap[i] = np.array([r, g, b])
		cmap = cmap/255 if normalized else cmap
		return cmap
	def bitget(self, byteval, idx):
		return ((byteval & (1 << idx)) != 0)

class NYUv2Dataset(VisionDataset):
	def __init__(
		self, 
		image, 
		depth, 
		mask=None, 
		transform=None, 
		depth_transform=None, 
		augmentation=None, 
		is_aug=True
	):
		super().__init__()
		self.image = image
		self.depth = depth
		self.mask = mask
		self.transform = transform
		self.depth_transform = depth_transform
		self.augmentation = augmentation
		self.is_aug = is_aug
		self.config = Config()

	def __len__(self):
		return len(self.image)
	
	def __getitem__(self, idx):
		image = np.array(self.image[idx].squeeze()).astype(np.uint8)
		depth = np.array(self.depth[idx].squeeze()).astype(np.uint8)
		mask = np.array(self.mask[idx].squeeze()).astype(np.uint8) if self.mask is not None else None

		# PIL変換
		image_PIL = Image.fromarray(image).resize(self.config.TARGET_SIZE, Image.BILINEAR)
		depth_PIL = Image.fromarray(depth).resize(self.config.TARGET_SIZE, Image.NEAREST)
		mask_PIL = None
		if mask is not None:
			mask_PIL = Image.fromarray(mask).resize(self.config.TARGET_SIZE, Image.NEAREST)

		if self.is_aug and self.augmentation:
			image_PIL, depth_PIL, mask_PIL = self.augmentation(image_PIL, depth_PIL, mask_PIL)
		
		image = self.transform(image_PIL)
		depth = self.depth_transform(depth_PIL)
		# 評価モード
		if mask_PIL is not None:
			mask_PIL = mask_PIL
			mask = torch.from_numpy(np.array(mask_PIL)).long()
		else:
			mask = torch.zeros(self.config.TARGET_SIZE, dtype=torch.long)

		return image, depth, mask
# 評価用
class NYUv2DatasetTest(NYUv2Dataset):
	def __init__(
		self, 
		image, 
		depth, 
		mask=None, 
		transform=None, 
		depth_transform=None, 
		augmentation=None, 
		is_aug=True
	):
		super(NYUv2DatasetTest, self).__init__(
			image, 
			depth, 
			mask, 
			transform, 
			depth_transform,
			None,
			False
		)
		self.config = Config()