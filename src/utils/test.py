from config import Config 
import os, yaml, sys, h5py
import scipy.io
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
from glob import glob as find
from PIL import Image

class NYUv2:
	label_url = "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
	class13_mat = "datasets/class13Mapping.mat"
	nyu_depth_v2_labeled_mat = "datasets/nyu_depth_v2_labeled.mat"
	test_dir = "datasets/nyuv2/test"
	# numpy ファイルの保存場所
	rgb_depth_dir = "numpy"
	mask_dir = "mask"

	def __init__(self):
		self.base_image_path = f"{self.test_dir}/{self.rgb_depth_dir}"
		self.base_mask_path = f"{self.test_dir}/{self.mask_dir}"
		print(f"CLASS 13 MATLIB: {self.class13_mat}")
		print(f"NYUv2 LAB LABEL MATLIB: {self.nyu_depth_v2_labeled_mat}")
		self.config = Config()
	
	def __call__(self):
		self.image_dir = "image"
		self.depth_dir = "depth"
		self.base_image_path = f"{self.test_dir}/{self.image_dir}"
		self.base_depth_path = f"{self.test_dir}/{self.depth_dir}"
		# Image
		if os.path.isdir(f"{self.base_image_path}") is True:
			image_paths = sorted(find(f"{self.base_image_path}/*"))
			depth_paths = sorted(find(f"{self.base_depth_path}/*"))
			nums = len(image_paths)
			print(depth_paths[-1])
			loop = tqdm(range(0, nums), desc="Image...")
			for i, (num) in enumerate(loop):
				# Image
				image_path = image_paths[num]
				image = Image.open(image_path).convert('RGB')
				image = image.resize(self.config.TARGET_SIZE, Image.BILINEAR)
				image_np = np.array(image)

				# depth
				depth_path = depth_paths[num]
				depth = Image.open(depth_path)
				depth = depth.resize(self.config.TARGET_SIZE, Image.NEAREST)
				depth_np = np.array(depth).astype(np.float32) / 255.0
				self.save_images(image_np, depth_np, num)
	
	def save_images(self, img, depth, num):
		# 正解ラベルのない評価データ：
		plt.figure(figsize=(12, 4))
		row = 1
		col = 2
		plt.subplot(row, col, 1)
		plt.title("Depth + Image")
		plt.imshow(img / 255, alpha=0.1, cmap='tab20b')
		plt.imshow(depth, alpha=0.7, cmap='YlOrRd', vmin=0.1)
		plt.axis("off")

		plt.subplot(row, col, 2)
		plt.title("Image")
		plt.imshow(img / 255, cmap='tab20b')
		plt.axis("off")

		plt.savefig(f"{self.test_dir}/plt/fig_{num}.png")
		plt.close()

	def from_mat_to_numpy(self, rgb_depth_dir="rgbd", mask_dir="rgbd_label"):
		if os.path.isfile(self.nyu_depth_v2_labeled_mat) is False:
			print(f"File not found, {self.nyu_depth_v2_labeled_mat}")
			return

		self.rgb_depth_dir = rgb_depth_dir
		self.mask_dir = mask_dir

		# ディレクトリ作成
		os.makedirs(os.path.join(self.test_dir, self.rgb_depth_dir), exist_ok=True)
		os.makedirs(os.path.join(self.test_dir, self.mask_dir), exist_ok=True)
		# ディレクトリ情報の更新
		self.base_image_path = f"{self.test_dir}/{self.rgb_depth_dir}"
		self.base_mask_path = f"{self.test_dir}/{self.mask_air}"

		print(f"TEST DATASET(RGB): {self.base_image_path}")
		print(f"TEST DATASET(LABEL): {self.base_mask_path}")

		# 894 cls -> 40 cls -> 13 clsに変換してゆく
		# 1. 13クラスマッピング (40 -> 13) の準備
		map_mat = scipy.io.loadmat(self.class13_mat)
		c40_to_13 = map_mat['classMapping13']['labels13'][0, 0].flatten().astype(np.int32)
		c40_to_13_map = np.insert(c40_to_13, 0, 255).astype(np.uint8)
		full_map = np.full(895, 255, dtype=np.uint8)

		# 0〜40 までのマッピングを流し込む (背景0を255にする処理を含む)
		full_map[0] = 255 # 背景
		for i, val in enumerate(c40_to_13):
			if i < 40:
				full_map[i+1] = val - 1 # 1〜40番目に 13クラスの値を代入
		full_map[0] = 255 # 背景は評価対象外

		with h5py.File(self.nyu_depth_v2_labeled_mat, 'r') as f:
			max_label = int(np.max(f['labels']))
			print(max_label)
			images = f["images"]
			depths = f["depths"]
			labels = f["labels"]
			num_images = len(labels)
			# 1. 安全装置：1000要素の巨大マップ（これなら 380 が来ても落ちない）
			c21_to_13_safe_map = np.full(1000, 255, dtype=np.uint8)
			for i in range(13):
				c21_to_13_safe_map[i] = i

			# 2. ラベルデータセットへの参照を保持
			labels_ds = f['labels']

			loop = tqdm(range(795, num_images), desc="Validate...")
			for i, (num) in enumerate(loop):

				# --- 1. RGB画像の処理 ---
				# (3, 640, 480) -> (480, 640, 3) 
				# NumPy結合時の型合わせのため、あえて float32 にキャストしておきます
				img_np = np.array(images[num]).transpose(2, 1, 0).astype(np.float32)
				
				# --- 2. Depth画像の処理 ---
				# (640, 480) -> (480, 640)
				depth_np = np.array(depths[num]).transpose(1, 0).astype(np.float32)
				# |-> (480, 640, 1)
				depth_np = np.expand_dims(depth_np, axis=-1) 
				
				# --- 3. RGBとDepthの結合 (4チャネル化) ---
				rgbd_np = np.concatenate([img_np, depth_np], axis=-1)
				
				# --- 4. 正解ラベル(Mask)の処理 ---
				label_raw = labels_ds[num, :, :].transpose(1, 0).astype(np.int32)
				mask_np = full_map[label_raw]
				if num == 800:
					print(f"DEBUG: label_raw shape = {label_raw.shape}")
					print(f"DEBUG: label_raw sample values = {label_raw[240, 155:157]}")
					print(f"Image Shape: {img_np.shape}") # (480, 640, 3) であるべき
					print(f"Mask Shape: {mask_np.shape}") # (480, 640) であるべき
					print(f"Unique values in GT: {np.unique(mask_np)}")
					self.save(num, img_np, depth_np, mask_np)

				# --- 保存処理 ---
				# RGB-DとMaskをそれぞれ .npy 形式で保存
				np.save(os.path.join(self.test_dir, self.rgb_depth_dir, f"{num:04d}.npy"), rgbd_np)
				np.save(os.path.join(self.test_dir, self.mask_dir, f"{num:04d}.npy"), mask_np)

	def save(self, num, img_np, depth_np, mask_np):
		# ラベルと画像を重ねながら確認用画像を作成
		plt.figure(figsize=(12, 4))
		# 重ねて表示（透過表示）
		plt.subplot(1, 4, 1)
		plt.imshow(mask_np, alpha=0.5, cmap='tab20b') # ラベルを半透明で被せる
		plt.imshow(depth_np, alpha=0.3, cmap='jet', vmin=0.1) # ラベルを半透明で被せる
		plt.title("Depth + Label")

		plt.subplot(1, 4, 2)
		plt.imshow(mask_np, cmap="tab20b", vmin=1, vmax=13)
		plt.title("Mask Labels (0-12)")

		plt.subplot(1, 4, 3)
		plt.imshow(img_np/255)
		plt.title("RGB")

		plt.subplot(1, 4, 4)
		plt.imshow(depth_np, cmap='YlOrRd', vmin=0.1) # ラベルを半透明で被せる
		plt.title("Depth")
		plt.savefig(f"{self.test_dir}/plt/gt_fig_{num}.png")
		plt.close()

# nyu = NYUv2()
# nyu.from_mat_to_numpy("numpy", "mask")
# nyu()
