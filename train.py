from tqdm import tqdm
import torch
import numpy as np
import time
from config import Config
from utils.miou import mIoU
# tensorboard
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from utils.log import Log
from utils.dataset import ColorMap, NYUv2Dataset
from utils.normalize import ImageNet as nm
from utils.train import train
from utils.validate import validate 
from utils.functions import *
from segsetup import SegSetup, PrepareDatasets, split_data

# モデル、オプティマイザ、デバイスの準備
config = Config()
device = torch.device(config.device)
set_seed(config.SEED)

#################################################################################
# ハイパーパラメータ
title = f"[Base] Late Fusion 128ch Learning Log: {datetime.now().strftime("%Y%m%d-%H:%M:%S")}"
log = Log(title)
print(f"TensorBoard Log DIR => {log.dir}")

# 1e-4(AdamW)
lr = config.LR["base"]
epochs = config.EPOCHS
batch_size = config.BATCH_SIZE
# データ拡張
is_augmentation = True
# FTする時True, 初期学習時はFalse
is_load_model = True
# FT対象：モデル指定
model_path = config.LATEST_MODEL
################################################################################
# データセット作成（訓練・検証・評価）
# 訓練・検証用
################################################################################
# 安全策（データリーク防止）
# 訓練データから検証データを作成する
train_img, val_img, train_depth, val_depth, train_mask, val_mask = split_data()
# 訓練DS
train_dataset = PrepareDatasets(config, mode="train", img=train_img, depth=train_depth, mask=train_mask)
train_data = train_dataset(
		augmentation=train_dataset.augmentation, 
		is_augmentation=is_augmentation
)
# 検証DS
valid_dataset = PrepareDatasets(config, mode="valid", img=val_img, depth=val_depth, mask=val_mask)
valid_data = valid_dataset()
# 評価データが存在する場合は評価データを利用する（コンペ形式であると、ないことも多い）
# test_dataset = PrepareDatasets(config, mode="test")
# test_data = test_dataset()

# ##############################################################################
# モデル/最適化関数/目的関数/学習率スケジューラー
# ##############################################################################
setup = SegSetup(config)
model, optimizer, criterion, scheduler = setup(is_load_model=is_load_model)
#################################################################################
# 学習設定 / TensorBoardへ書き出し / 精度用配列:accuracy
#################################################################################
writer = SummaryWriter(log_dir=log.dir)
accuracy = {"name":"Accuracy Log", "Accuracy":[], "miou":[]}
log(
	model={
		"name": config.MODEL_NAME, 
		"input_shape": [(config.TARGET_SIZE[0], config.TARGET_SIZE[1], 3),(config.TARGET_SIZE[0], config.TARGET_SIZE[1], 1),(config.TARGET_SIZE[0], config.TARGET_SIZE[1], 1)], "num_classes":config.NUM_CLASSES, 
		"dropout_rate":0.0, 
		"state_dict":[]
	}, 
	training={
		"epoch":epochs, 
		"batch_size":batch_size, 
		"optimizer":optimizer.__class__.__name__, 
		"learning_rate":lr, 
		"loss_function":criterion.__class__.__name__},
	dataset={
		"path":config.DEFAULT_PATH, 
		"augmentation":is_augmentation, 
		"train_split":config.SPLIT_SIZE
	},
	paths={
		"train_dir": f"{log.dir}/train", 
		"checkpoint_dir": f"{log.dir}/checkpoints", 
		"validation_dir": f"{log.dir}/validation", 
		"predict_dir": f"{log.dir}/predict"
	},
	is_save=True
)

# MAIN関数
def main(train_data, valid_data, optimizer, criterion, writer, log):
	# 学習メイン
	config = Config()
	device = config.device

	for epoch in range(0, epochs):
		epoch_start_time = time.time()
		# 訓練
		t0 = time.time()
		
		miou, metrics = train(model, train_data, optimizer, criterion, epoch, device, writer)
		print(f"{epoch} Epoch => Mean IoU(Train): {(miou * 100):.4f}%")
		# ファイル書き出し用（miou）
		accuracy["miou"].append({"epoch":f"Train/{epoch} epoch", "miou": (miou)})
		# 訓練時間測定
		train_time = (time.time() - t0) / 60
		writer.add_scalar('Performance/Train Time', train_time, epoch)
		print(f"Performance/Train Time: {train_time} min.")

		save_model(model, epoch, miou, log.dir)
		# epoch最後にモデルを保存する。
		if epoch == (epochs - 1):
			# save_model(model, epoch, miou, log.dir)
			save_path = f"{log.dir}/checkpoints/latefusion_epochs_{epoch}_miou_{miou:.2f}.pth"
			torch.save(model.state_dict(), save_path)
			print(f"--- Model saved at Last Epoch {epoch} (mIoU: {miou:.4f}) ---")

		# 検証(2epochごとに実行)
		if epoch % 2 == 0:
			t1 = time.time()
			miou, metrics = validate(model, valid_data, criterion, epoch, device, writer, log.dir)
			epoch_end_time = time.time()
			elapsed_seconds = epoch_end_time - epoch_start_time
			time_elapsed = elapsed_seconds / 60
			# 検証測定
			val_time = (time.time() - t1) / 60
			print(f"{epoch} Epoch => Mean IoU(Validate): {(miou * 100):.4f}%")
			writer.add_scalar('Performance/Val Time', val_time, epoch)
			print(f"Performance/Val Time: {val_time} min.") 
			writer.add_scalar('Performance/Epoch_Time_Min', time_elapsed, epoch)
			print(f"{epoch} Epoch finished in {time_elapsed:.2f} minutes.")
			# 現在の学習率をログへ
			scheduler.step(miou)
		# ファイル書き出し用（miou）
		accuracy["miou"].append({"epoch":f"Val/{epoch} epoch", "miou": (miou)})
		current_lr = optimizer.param_groups[0]['lr']
		writer.add_scalar('Train/Learning Rate', current_lr, epoch)
		print("Next ---")

	log.save(f"{log.dir}/accuracy.yaml", accuracy)

if __name__ == '__main__':
	main(train_data, valid_data, optimizer, criterion, writer, log)
