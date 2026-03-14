import mlx.core as mx
import numpy as np
import os
from glob import glob as find
from tqdm import tqdm
from PIL import Image
from config import Config
# Projects
from utils.normalize import ImageNet as nm

class Preprocess:
	DEFAULT_PATH=Config.DEFAULT_PATH
	def __init__(self):
		self.cfg = Config()
		self.path = {
			# 訓練データ
			"train":{
				"img": os.path.join(self.DEFAULT_PATH, "train/image"),
				"label": os.path.join(self.DEFAULT_PATH, "train/label"),
				"depth": os.path.join(self.DEFAULT_PATH, "train/depth"),
				"numpy": os.path.join(self.DEFAULT_PATH, "train/numpy"),
				"mask": os.path.join(self.DEFAULT_PATH, "train/mask"),
			},
			# 評価用
			"test":{
				"img": os.path.join(self.DEFAULT_PATH, "test/image"),
				"label": os.path.join(self.DEFAULT_PATH, "test/label"),
				"depth": os.path.join(self.DEFAULT_PATH, "test/depth"),
				"numpy": os.path.join(self.DEFAULT_PATH, "test/numpy"),
				"mask": os.path.join(self.DEFAULT_PATH, "test/mask"),
			}
		}
		for p in [self.path["train"]["numpy"], self.path["train"]["mask"], self.path["test"]["numpy"], self.path["test"]["mask"]]:
			os.makedirs(p, exist_ok=True)

	def search_path(self, is_test):
		"""Return numpy and mask directories for train/test."""
		mode = "test" if is_test else "train"
		img = self.path[mode]['numpy']
		mask = self.path[mode]['mask']
		return img, mask

	# combine data and label data exchanges numpy.
	def combine_and_label_numpy(self, img_path, dep_path, mask_path=None):		
		train_image = Image.open(img_path).convert("RGB").resize(Config.TARGET_SIZE, Image.BILINEAR)
		depth_image = Image.open(dep_path).resize(Config.TARGET_SIZE, Image.NEAREST)

		train_np = np.array(train_image, dtype=np.uint8)
		depth_np = np.array(depth_image, dtype=np.float32)

		if depth_np.max() > 0:
			depth_np = (depth_np / depth_np.max() * 255.0)
		# uint8 に型変換
		depth_np = depth_np.astype(np.uint8)

		# 画像・深度の結合（3チャネル + 1チャネル結合）
		combined = np.concatenate([train_np, depth_np[..., None]], axis=-1)

		mask_np = None
		if mask_path:
			# 正解画像の変換（訓練のみ）
			mask_image = Image.open(mask_path).resize(Config.TARGET_SIZE, Image.NEAREST)
			mask_np = np.array(mask_image)

		return combined, mask_np

	def convert(self):
		# 共通化
		for mode in ["test", "train"]:
			imgs = sorted(find(f"{self.path[mode]['img']}/*"))
			depths = sorted(find(f"{self.path[mode]['depth']}/*"))
			labels = sorted(find(f"{self.path[mode]['label']}/*"))
			if len(labels) == 0:
				labels = [None] * len(imgs)
			for i, (img_path, dep_path, mask_path) in enumerate(tqdm(zip(imgs, depths, labels), desc=f"{mode} convert...")):
				combined, mask_np = self.combine_and_label_numpy(img_path, dep_path, mask_path)
				np.save(f"{self.path[mode]['numpy']}/img_{i:05d}.npy", combined.astype(np.uint8))
				# Label用のファイルの保存
				if mask_np is not None:
					np.save(f"{self.path[mode]['mask']}/mask_{i:05d}.npy", mask_np.astype(np.uint8))
	
	def loads(self, is_test=False):
		mode = "test" if is_test else "train"
		data = sorted(find(f"{self.path[mode]['numpy']}/*"))
		mask = sorted(find(f"{self.path[mode]['mask']}/*"))
		images, depths, masks = [], [], []

		if len(data) == 0:
			# fallback diagnostic to show available test numpy files
			print(sorted(find(f"{self.path['test']['numpy']}/*")))
			print(f"Error: no data found ({len(data)}) in {self.path[mode]['numpy']}")
			return images, depths, masks

		for i, path in enumerate(tqdm(data, desc="Preload: RGB/Depth")):
			np_data = np.load(path)
			img = np_data[:, :, :3]
			depth = np_data[:, :, 3]
			images.append(img)
			depths.append(depth)

		if mask is not None:
			for i, path in enumerate(tqdm(mask, desc="Preload: Mask")):
				np_data = np.load(path)
				if np_data.ndim == 2:
					np_data = np_data[..., np.newaxis]
				masks.append(np_data)
		
		return images, depths, masks