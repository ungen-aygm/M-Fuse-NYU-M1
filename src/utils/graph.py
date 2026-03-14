import mlx.core as mx
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import numpy as np
from PIL import Image
# Project: Normarizatiion, Preprocess, ColorMap
from utils.normalize import ImageNet as nm
from utils.preprocess import Preprocess as Loader
from utils.dataset import ColorMap
from tqdm import tqdm
import torch
from config import Config
from datetime import datetime
from utils.miou import mIoU
import numpy as np

class Visualizer:
	DEFAULT_SIZE = (224, 224)
	figsize=(12, 8)
	row = 4
	column = 6
	def __init__(self):
		# for mx array
		self.image_mx = None
		self.date = datetime.now().strftime("%Y%m%d-%H%M%S")

	def cmap(self, vmax=14, cmap="tab20b"):
		vmin, vmax = 1, vmax
		cmap = plt.get_cmap(cmap)
		norm = mcolors.BoundaryNorm(np.arange(vmin, vmax + 2), cmap.N)
		# EXAM) plt.imshow(img.squeeze(), cmap=cmap, norm=norm)
		return cmap, norm
	
	def confirm(self, dataset):
		config = Config()
		device = config.device
		cmap, norm = self.cmap(14, cmap="tab20c")
		loop = tqdm(dataset, desc="Export Color Label", leave=True)
		indices = sorted(np.random.choice(len(dataset), 1, replace=False).tolist())
		plt.figure(figsize=(15, 5))
		for i, (_rgb, _depth, _targets) in enumerate(loop):
			rgb = _rgb.to(device)
			depth = _depth.to(device)
			targets = _targets.to(device)
			if i in indices:
				raw_rgb = np.array(nm.denorm_rgb(mx.array(rgb[0].cpu().numpy().transpose(1, 2, 0)))) * 255
				raw_rgb = raw_rgb.astype(np.uint8)
				depth = np.array(nm.denorm_gray(mx.array(depth[0].cpu().numpy().transpose(1, 2, 0))))
				colored_targets = cmap(targets[0].cpu().numpy())[:, :, :3]

				plt.subplot(1, 3, 1)
				plt.title(f"RGB")
				plt.imshow(raw_rgb)
				plt.axis("off")

				plt.subplot(1, 3, 2)
				plt.title(f"Depth")
				plt.imshow(depth)
				plt.axis("off")

				plt.subplot(1, 3, 3)
				plt.title(f"Label")
				plt.imshow(colored_targets)
				plt.axis("off")
			
		plt.show()

	def export(self, model, dataset, cmax=1, save_dir=""):
		model.eval()
		config = Config()
		tracker = mIoU(num_classes=config.NUM_CLASSES)
		device = config.device
		count = 0
		cmap, norm = self.cmap(14, cmap="tab20c")
		loop = tqdm(dataset, desc="Export ColorLabel...", leave=True)
		path = "runs/Plot"
		if len(dataset) < cmax:
			cmax = len(dataset)
		indices = sorted(np.random.choice(len(dataset), cmax, replace=False).tolist())
		with torch.no_grad():
			for i, (_rgb, _depth, _targets) in enumerate(loop):
				rgb = _rgb.to(device)
				depth = _depth.to(device)
				targets = _targets.to(device)
				outputs = model(rgb, depth)
				predicted = torch.argmax(outputs, dim=1)
				tracker.update(predicted, targets)
				if i in indices:
					iou_score, miou = tracker.compute()
					self.view(rgb, depth, predicted, targets, mode="save", idx=i, miou=miou, save_dir=save_dir)
					count += 1

				if count >= cmax:
					print(f"最大枚数：{cmax}枚 保存しました。処理を中断します。")
					break
	
	def view(self, rgb, depth, predict, target, mode="save", idx=0, miou=0.0, save_dir=""):
		# Numpyデータを表示するので、torch.tensorなどのGPU上からcpuに移動させ,numpyを扱う
		cmap, norm = self.cmap(14, cmap="tab20c")
		path = "runs/Plot"
		# numpy変換
		rgb = self._to_numpy(rgb).squeeze()
		rgb = nm.denorm_rgb(mx.array(rgb))
		predict = self._to_numpy(predict).squeeze()
		target = self._to_numpy(target).squeeze()

		plt.figure(figsize=(15, 5))
		plt.subplot(1, 3, 1)
		plt.title(f"RGB: {idx}")
		plt.imshow(rgb)
		plt.axis("off")
		
		plt.subplot(1, 3, 2)
		plt.title("Target")
		plt.imshow(target[0], cmap=cmap, norm=norm)
		plt.axis("off")

		plt.subplot(1, 3, 3)
		plt.title(f"Predict (mIoU: {miou.item():.3f})", fontsize=12, color='blue' if miou.item() > 0.5 else 'black')
		plt.imshow(predict[0], cmap=cmap, norm=norm)
		plt.axis("off")

		filename = f"{save_dir}/plot_{idx}_{self.date}.png" if save_dir != "" else f"{path}/plot_{idx}_{self.date}.png"
		
		if mode == "save":
			plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
			plt.close()
		else:
			plt.show()

	def cls_rank(self, iou_score, miou, export=False):
		"""
		クラスごとにランキング形式で表示する
		"""
		config = Config()
		class_names = config.CLASS_NAMES
		# ランキング用に整形
		ranking = []
		for i, name in enumerate(class_names):
			ranking.append({
				"Class ID": i,
				"Class Name": name,
				"IoU": iou_score[i].item()
			})
		# IoU(降順)
		self.view_cls_rank(sorted(ranking, key=lambda x: x["IoU"], reverse=True), miou)
	
	def view_cls_rank(self, sorted_ranking:list, miou):

		print("\n" + "="*50)
		print(f"1. Class-wise IoU Ranking (mIoU: {miou.item():.4f})")
		print("="*50)
		
		# ヘッダー
		print(f"{'Class ID':<10} {'Class Name':<20} {'IoU':<10}")
		print("-" * 40)
		# 行表示
		for row in sorted_ranking:
			print(f"{row['Class ID']:<10} {row['Class Name']:<20} {row['IoU']:.4f}")
		print("="*50 + "\n")

	def _to_numpy(self, data, idx=0):
		"""あらゆる型を2D NumPyに変換"""
		# PyTorch
		if isinstance(data, torch.Tensor):
			data = data.detach().cpu().numpy()
		elif 'mlx.core' in str(type(data)):
			data = np.array(data)
			
		# 1. 次元整形(4->3) (B, C, H, W) -> (C, H, W)
		if data.ndim == 4:
			data = data[idx]
		# 2. データの種類に応じた整形(転置): RGB (3, H, W)  -> (H, W, 3) 
		if data.ndim == 3 and data.shape[0] == 3:
			data = data.transpose(1, 2, 0)
		# ラベル/深度 (1, H, W)  -> (H, W) 
		if data.ndim == 3 and data.shape[0] == 1:
			data = data.squeeze(0)
		
		return data

	# Show Plot( 3 photo * 4 columns and rows )
	def plot_triplet_grid(self, images:list, labels=[]):
		plt.figure(figsize=self.figsize)
		number = 1
		for row in range(1, (self.row * self.column)+1):
			plt.subplot(self.row, self.column, i)
			if row % 3 == 1:
				# 学習画像
				plt.imshow(images[i-1])
			elif row % 3 == 2:
				# 正解ラベル(正規化を外す)
				mask_raw = (images[i-1] * 255).round().astype(np.int32)
				unique_vals = np.unique(mask_raw)
				mask_mapped = np.zeros_like(mask_raw)
				for idx, val in enumerate(unique_vals):
					# 255 (境界線) は 0 または適当な ID に固定
					if val == 255:
						mask_mapped[mask_raw == val] = 0 # 例: 背景と同じにする
					elif idx < 14:
						mask_mapped[mask_raw == val] = idx
				im = plt.imshow(mask_mapped, cmap=self.label(), vmin=0, vmax=14)
				plt.colorbar(ticks=np.arange(self.label().N))
			else:
				# 深度
				plt.imshow(images[i-1], cmap="YlOrRd", vmin=0.1, vmax=0.9)
			plt.axis("off")
			if i >= len(images):
				break
		# Layout
		plt.tight_layout()
		plt.show()

# 訓練画像表示
def plot_training_batch(loader:Loader, num_samples: int=8, is_test=False):
	# 1. 画像の読込み
	plot = Visualizer()
	(image, depth, mask) = loader.loads(is_test=is_test)
	images = []
	# 3. リストの作成
	for _ in range(num_samples):
		# 2. 正規化を外す
		index = np.random.randint(0, len(image))
		image_mx = np.array(nm.denorm_rgb(image[index]))
		depth_mx = np.array(nm.denorm_gray(depth[index]))
		# 正解ラベル
		mask_mx = None
		if len(mask) > 0:
			mask_mx = np.array(mask[index])

		images.append(image_mx)
		if mask_mx is not None:
			images.append(mask_mx)
		else:
			images.append(image_mx)
		images.append(depth_mx)

	del depth_mx, depth, image_mx, mask_mx
	# 推論結果（可視化）
	plot.plot_triplet_grid(images)

	del image