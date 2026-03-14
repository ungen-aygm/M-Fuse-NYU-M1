import torch
import mlx.core as mx
class Config:
	##################################################
	# ハイパーパラメータ
	##################################################
	# 基本設定
	CONFIG_PATH="src/config.yaml"
	# Tensor Board
	LOG_PATH="runs/"

	# MODEL (LATEST)
	LATEST_MODEL="runs/fase6/LateFution_128ch_20260313-181004/checkpoints/latefusion_epochs_4_miou_0.79.pth"
	
	# 入力解像度
	TARGET_SIZE=(448, 448) # (224, 224) or (336, 336)

	# BATCH SIZE
	BATCH_SIZE = 8 # (448, 448) *ViT Freeze / 8 or 16

	# EPOCH
	EPOCHS = 5

	# LEARNING RATES (差分学習率設定)
	LR = {
		"base":5e-5, # 1e-4
		"ViT": 1e-1, # lr * 1e-2
		"UNet": 1.0, # lr * 3.0
		"LateFusion": 1.0, # lr * 1.0
	}

	# Learning Schedule(学習スケジュール)
	# mIoUを最大化したい=>max
	LR_MODE="max"
	# 停滞したら学習率を3/4
	FACTOR=0.75
	# 5 epoch連続で改善しなかったら発動
	PATIENCE=5
	# 下限学習率
	MIN_LR=1e-7

	# DATASET
	DEFAULT_PATH="datasets/nyuv2/"

	SPLIT_SIZE=0.125
	SEED=42
	RANDOM_STATE=42

	# CLASSES
	NUM_CLASSES=13 # 40の場合もできるよう
	IGNORE_INDEX=255

	# Augment
	CROP=True
	DEGREES=20
	HFLIP=0.5
	CROP_SCALE=(0.5, 1.0)
	CROP_RATIO=(0.75, 1.33)
	# Loss
	FOCALLOSS = 0.5
	DICELLOSS = 0.5
	# EXAM: JITTER_PARAM={"brightness":0.2, "contrast":0.2, "saturation":0.1, "hue":0.05},
	JITTER_PARAM = None
	
	##################################################
	# 定数
	##################################################
	# MPS型指定
	DTYPE = torch.float32
	# Schedule
	T_MAX=30
	# Model(Late Fusion)
	MODEL_NAME = "LateFution_128ch"
	# Model(UNet)
	UNet_BACKBONE = "resnet18"
	# Model(ViT-tiny)
	ViT_BACKBONE = "vit_base_patch16_clip_224.openai"
	PATCH_SIZE = (16, 16)

	# Normarize
	MEAN = mx.array([0.485, 0.456, 0.406])
	STD = mx.array([0.229, 0.224, 0.225])

	###############################################################
	# クラス不均衡問題: book/0.0のため、facal/diceの損失関数や重み調整を行う
	###############################################################
	"""
	==================================================
	1. Class-wise IoU Ranking (mIoU: 0.6745)
	==================================================
	Class ID   Class Name           IoU       
	----------------------------------------
	0          bed                  0.9029
	4          floor                0.8569
	...
	9          table                0.5799
	1          books                0.0000
	==================================================
	以下のクラスIDに注目するための重み
	"""
	# クラス（独自定義）
	CLASS_WEIGHT=torch.tensor([
		5.0,  # 0: bed
		20.0, # 1: books
		2.0,  # 2: ceiling
		4.0,  # 3: chair
		0.8,  # 4: floor
		4.0,  # 5: furniture
		8.0,  # 6: objects
		8.0, # 7: picture
		5.0,  # 8: sofa
		5.0,  # 9: table
		10.0, # 10: tv
		0.5,  # 11: wall
		8.0   # 12: window
	])
	# クラス名（独自定義）
	CLASS_NAMES = [
		"bed",       # 0
		"books",     # 1
		"ceiling",   # 2
		"chair",     # 3
		"floor",     # 4
		"furniture", # 5
		"objects",   # 6
		"picture",   # 7
		"sofa",      # 8
		"table",     # 9
		"tv",        # 10
		"wall",      # 11
		"window"     # 12
	]
	@property
	def device(self):
		if torch.backends.mps.is_available():
			return torch.device("mps") 
		elif torch.cuda.is_available("cuda"):
			return torch.device("cuda")
		else:
			return torch.device("cpu")

	# 訓練時に使いやすくなればいい。
	def __call__(self, 
		TARGET_SIZE=(224, 224),
		BATCH_SIZE=16,
		EPOCHS=10,
	):
		self.TARGET_SIZE = TARGET_SIZE
		self.BATCH_SIZE = BATCH_SIZE
		self.EPOCHS = EPOCHS

	def __repr__(self):
		return f"Target: {self.TARGET_SIZE}, Device: {self.device}"