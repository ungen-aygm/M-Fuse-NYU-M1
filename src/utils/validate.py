import mlx.core as mx
import torch
import numpy as np
from config import Config
from utils.dataset import ColorMap
from utils.normalize import ImageNet as nm
from tqdm import tqdm
from utils.miou import mIoU
from torchvision.utils import make_grid
from utils.functions import *

# 検証構成(勾配なし):TensorBoardに画像送信
def validate(model, dataloader, criterion, epoch, device, writer, path):
	# 評価モード(not train)
	model.eval()
	config = Config()
	val_loss = 0
	validate_preds = []
	# mIoU Tracker
	tracker = mIoU(num_classes=config.NUM_CLASSES)
	predicted, target_tensor = None, None
	# 評価用
	valid_predicts, valid_targets = [], []
	# 勾配なし
	with torch.no_grad():
		loop = tqdm(dataloader, desc="Fusion Validating...", leave=True)
		for idx, (_rgb, _depth, _targets) in enumerate(loop):

			rgb = _rgb.to(device)
			depth = _depth.to(device)
			targets = _targets.to(device)

			outputs = model(rgb, depth)
			loss = criterion(outputs, targets)
			val_loss += loss.item()

			# 1. 予測をクラスIDに変換
			predicted = torch.argmax(outputs, dim=1)
			# 評価用
			valid_indices = (targets != 255)
			valid_predicts.append(predicted[valid_indices].detach().cpu().numpy())
			valid_targets.append(targets[valid_indices].detach().cpu().numpy())
			#######################################################################
			mask = (targets != 255)
			tracker.update(predicted[mask], targets[mask])
			# --- 最初のバッチだけ可視化 ---
			if idx == 0:
				# 2. カラーマップ変換関数の呼び出し（後述）
				cmap = ColorMap()(N=256)
				raw_rgb = torch.from_numpy(np.array(nm.denorm_rgb(mx.array(rgb[0].cpu().numpy().transpose(1, 2, 0))))).permute(2, 0, 1) * 255
				# 表示用に変換 (255 を 0 に置換してインデックスエラー回避)
				vis_targets = targets[0].clone()
				vis_targets[vis_targets == 255] = 0
				# 予測
				colored_preds = torch.from_numpy(cmap[predicted[0].cpu().numpy()].transpose(2, 0, 1))
				# GT(Grand True)
				colored_targets = torch.from_numpy(cmap[vis_targets.cpu().numpy()].transpose(2, 0, 1))
				# 元画像/GT画像/予測画像の順番で１列の画像にする
				grid = make_grid([raw_rgb, colored_targets, colored_preds], padding=2).float() / 255.0
				# 画像の送信
				writer.add_image(f'Val/Comparison Grid Image', grid, epoch)

	class_ious, miou = tracker.compute()
	# miou
	writer.add_scalar(f'mIoU/Validation', miou.item(), epoch)
	# 平均Lossの記録
	writer.add_scalar('Loss/Validation', val_loss / len(dataloader), epoch)
	# クラス別 IoU
	class_names = config.CLASS_NAMES
	for c, iou in enumerate(class_ious.cpu().numpy()):
		if not np.isnan(iou):
			# クラス名
			writer.add_scalar(f'Class_IoU/{class_names[c]}', iou, epoch)

	# 混同行列(適合率/再現率/F1 Score)
	precision_s, recalls, f1_scores = 0, 0, 0
	if len(valid_predicts) > 0:
		all_precisions = np.concatenate([p.flatten() for p in valid_predicts])
		all_targets = np.concatenate([t.flatten() for t in valid_targets])
		# 適合率/再現率/F1 ScoreをTensorBoardに送信
		precision_s, recalls, f1_scores = cal_metrix(all_precisions, all_targets)
		writer.add_scalar(f'Metrics/Precision Validation', precision_s, epoch)
		writer.add_scalar(f'Metrics/Recall Validation', recalls, epoch)
		writer.add_scalar(f'Metrics/F1 Score Validation', f1_scores, epoch)
		valid_predicts.clear()
		valid_targets.clear()
		
	return miou.item(), (precision_s, recalls, f1_scores)