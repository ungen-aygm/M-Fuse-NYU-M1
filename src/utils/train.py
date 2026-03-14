import torch
import numpy as np
from tqdm import tqdm
from utils.miou import mIoU
from config import Config
from utils.functions import *

# 訓練
def train(model, dataloader, optimizer, criterion, epoches, device, writer):
	# 学習モード
	model.train()
	config = Config()
	running_loss = 0.0
	current_acc = 0.0
	correct = 0
	total = 0
	processed_samples = 0
	# 評価用
	valid_predicts, valid_targets = [], []
	# mIoU Tracker
	tracker = mIoU(num_classes=config.NUM_CLASSES)
	miou = None
	# for start in tqdm(range(0, total, batch_size), desc="Training"):
	loop = tqdm(dataloader, desc="Fusion Training", leave=True)
	for idx, (_rgb, _depth, _targets) in enumerate(loop):
		
		rgb = _rgb.to(device)
		depth = _depth.to(device)
		targets = _targets.to(device)

		optimizer.zero_grad()
		# 順伝播 (Late Fusion)
		outputs = model(rgb, depth)

		# 逆伝播と更新
		loss = criterion(outputs, targets)

		loss.backward()
		optimizer.step()

		# 統計計算
		running_loss += loss.item()
		_, predicted = outputs.max(1)
		mask = (targets != 255)
		tracker.update(predicted[mask], targets[mask])

		valid_indices = (targets > 0) & (targets != 255)

		valid_predicted = predicted[valid_indices].detach().cpu().numpy()
		valid_target = targets[valid_indices].detach().cpu().numpy()

		valid_predicts.append(valid_predicted)
		valid_targets.append(valid_target)

		correct += (valid_predicted == (valid_target - 1)).sum().item()
		processed_samples += valid_predicted.size
		
		# 5. プログレスバーの右側に現在の Loss と Accuracy を表示
		current_acc = 100. * correct / (processed_samples + 1e-6)
		loop.set_postfix({
			'loss': f"{loss.item():.4f}",
			'acc': f"{current_acc:.2f}%"
		})
		torch.mps.empty_cache()
		
	num_batches = len(loop)
	# mIoU更新
	class_ious, miou = tracker.compute()
	# mIoU
	writer.add_scalar(f'mIoU/Train', miou.item(), epoches)
	# Pixel正解率
	writer.add_scalar('Accuracy/Train', current_acc, epoches)
	writer.add_scalar('Loss/Train', running_loss / num_batches, epoches)

	# 混同行列(適合率/再現率/F1 Score)
	precision_s, recalls, f1_scores = 0, 0, 0
	if len(valid_predicts) > 0:
		all_precisions = np.concatenate(valid_predicts)
		all_targets = np.concatenate(valid_targets)
		precision_s, recalls, f1_scores = cal_metrix(all_precisions, all_targets)
		writer.add_scalar(f'Metrics/Precision Train', precision_s, epoches)
		writer.add_scalar(f'Metrics/Recall Train', recalls, epoches)
		writer.add_scalar(f'Metrics/F1 Score Train', f1_scores, epoches)
		valid_predicts.clear()
		valid_targets.clear()

	return miou.item(), (precision_s, recalls, f1_scores)

