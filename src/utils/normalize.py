import mlx.core as mx
import numpy as np
from config import Config

class ImageNet:
	# ImageNet standard normalization constants
    # Reference: https://pytorch.org/vision/stable/models.html
    # These values are calculated from millions of images in the ImageNet dataset.
	MEAN = Config.MEAN
	STD = Config.STD

	@staticmethod
	def norm_rgb(array: list):
		# チャネルを 初期化(B, W, H, C)として定義
		mean = ImageNet.MEAN.reshape(1, 1, 1, 3)
		std = ImageNet.STD.reshape(1, 1, 1, 3)
		x = mx.array(np.array(array)) / 255.0
		# 3 チャネルを(1, W, H, C)に拡張
		if len(x.shape) == 3:
			x = x[None]
		return (x - mean) / std
	
	@staticmethod
	def denorm_rgb(array: mx.array):
		# チャネルを(B, W, H, C)として定義
		mean = ImageNet.MEAN.reshape(1, 1, 1, 3)
		std = ImageNet.STD.reshape(1, 1, 1, 3)
		if len(array.shape) == 3:
			array = array[None]
		x = mx.clip((array * std) + mean, 0.0, 1.0).astype(mx.float32)
		x = mx.squeeze(x)
		return np.array(x.astype(mx.float32))
	
	@staticmethod
	def norm_gray(array: list):
		# Numpyへ変換
		x = np.array(array).astype(np.float32)
		# 固定値: 255.0で正規化
		x = x / 255.0
		# MLX配列に変換
		x = mx.array(x)
		# チャネルチェックと拡張
		if len(x.shape) == 2:
			x = x[..., None]
		return x

	@staticmethod
	def denorm_gray(array: mx.array):
		# 深度マップも (1, H, W, 1) -> (H, W, 1) に戻す必要がある
		if len(array.shape) == 4 and array.shape[0] == 1:
			array = mx.squeeze(array, axis=0)
		x = mx.clip(array, 0.0, 1.0)
		return np.array(x)