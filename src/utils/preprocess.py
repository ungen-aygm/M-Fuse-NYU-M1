import mlx.core as mx
import numpy as np
import os, h5py
from glob import glob as find
from tqdm import tqdm
from PIL import Image
from config import Config
# Projects
from utils.normalize import ImageNet as nm
import scipy.io

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
		self.make_dataset_dir()

	def make_dataset_dir(self):
		lists = [self.path["train"]["img"], self.path["train"]["depth"], self.path["train"]["label"], self.path["train"]["numpy"], self.path["train"]["mask"], self.path["test"]["img"], self.path["test"]["depth"], self.path["test"]["label"], self.path["test"]["numpy"], self.path["test"]["mask"]]
		for p in lists:
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

"""
初期学習用にNYUv2公式からデータセットを展開し、前処理する。
NYUV2からダウンロードした.matファイルをtrain/testに分割する。
"""
class Setup_NYUV2(Preprocess):
	MAT_FILE = os.path.join("datasets", "nyu_depth_v2_labeled.mat")
	SPLIT = 795
	def __init__(self, config):
		super(Setup_NYUV2, self).__init__()
		self.make_dataset_dir()
		print(f"Make Dir(Train): {self.path['train']['img']}, OK!")
		print(f"Make Dir(Test): {self.path['test']['img']}, OK!")
		print("Starting .mat converting...")

		self.map_40 = self.mapping40()
		self.map_13 = self.mapping13()
	
	def __call__(self):
		if self.check_mat_file() is True:
			self.converting_mat_file()
			print(f"Finished Converting .mat => png(image/depth/label)")

	def check_mat_file(self):
		file = os.path.isfile(self.MAT_FILE)
		if file is False:
			print(f"{self.MAT_FILE}が見つかりません。data_setup.shを実行してください。")
			return False
		return True

	# 開く
	def converting_mat_file(self):
		file = self.MAT_FILE
		with h5py.File(file, 'r') as f:
			labels = f['labels'][:]
			images = f['images'][:]
			depths = f['depths'][:]
			
			print(f"Label: 894 => 40 => 13 Converting...")
			labels = self.convert_label(labels)
			
			N = images.shape[0]
			step = 1
			indices = np.arange(0, N, step)

			# train/test分割
			train_idx = indices[:self.SPLIT]
			test_idx  = indices[self.SPLIT:]

			print(f"Starting Train/Test Batching...")
			for mode, lists in [("train", train_idx), ("test", test_idx)]:
				for i, n in enumerate(tqdm(lists, desc="Saving...", leave=False)):
					image = images[n].transpose(1, 2, 0)
					image = np.rot90(image, k=3)
					depth = depths[n]
					depth = (depth / depth.max() * 255) if depth.max() > 0 else depth
					depth = np.rot90(depth, k=3)
					label = labels[n]
					label = np.rot90(label, k=3)
					# 画像に保存
					image_file = os.path.join(self.path[mode]['img'], f"{i:05d}.png")
					depth_file = os.path.join(self.path[mode]['depth'], f"{i:05d}.png")
					label_file = os.path.join(self.path[mode]['label'], f"{i:05d}.png")
					Image.fromarray(image.astype(np.uint8)).save(image_file)
					Image.fromarray(depth.astype(np.uint8)).save(depth_file)
					Image.fromarray(label.astype(np.uint8)).save(label_file)
	
	def view_mapped_value(self, map):
		# Labelが正しく変換されたかをチェック
		print("unique mapped:", np.unique(map))
		print("max:", map.max())
	
	def mapping40(self):
		mat_file = 'datasets/classMapping40.mat'
		mat = scipy.io.loadmat(mat_file)
		mapping = mat['mapClass']
		return mapping.flatten().astype(np.int32) - 1
	
	def mapping13(self):
		mat_file = 'datasets/class13Mapping.mat'
		mat = scipy.io.loadmat(mat_file)
		mapping = mat['classMapping13']
		return mapping[0, 0][0].flatten().astype(np.int32) - 1
	
	def convert_label(self, label):
		label = label.astype(np.int32) - 1
		mapped_40 = self.map_40[label]
		mapped_13 = self.map_13[mapped_40]
		mapped_13[mapped_40 >= 37] = 255
		return mapped_13