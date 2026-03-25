import matplotlib.patches as mpatches
from tqdm import tqdm
import torch
import numpy as np
from utils.dataset import ColorMap, NYUv2Dataset
from utils.normalize import ImageNet as nm
from utils.preprocess import Preprocess as Loader
from utils.augmentations import Augmentation, AugmentationParams
from config import Config
import mlx.core as mx
from utils.graph import Visualizer
from utils.functions import *
from utils.evaluate import evaluate
from segsetup import SegSetup, PrepareDatasets, split_data

def setup_augmentation(config:Config):
	# データ拡張：configにパラメータの上書き
	return Augmentation(AugmentationParams(
		degrees=30, # Rotate（角度）
		hflip=config.HFLIP,# 左右反転確率(0.0-1.0)
		crop=config.CROP,
		crop_size=config.TARGET_SIZE,
		crop_scale=config.CROP_SCALE,
		crop_ratio=config.CROP_RATIO,
	))

def setup_env(config):
	set_seed(config.SEED)
	setup = SegSetup(config)
	model, _, _, _ = setup(is_load_model=True)
	return model

def setup_test(config):
	# 評価DS
	test_dataset = PrepareDatasets(config, mode="test")
	test_data = test_dataset()
	return test_data

# 訓練・検証DS
def setup_train_and_val(config, is_augmentation=True):
	# 訓練データからの分割
	train_img, val_img, train_depth, val_depth, train_mask, val_mask = split_data()
	# 訓練DS
	train_dataset = PrepareDatasets(config, mode="train", img=train_img, depth=train_depth, mask=train_mask)
	# データ拡張：setup_augmentationを適宜変更
	train_data = train_dataset(
		augmentation=setup_augmentation,
		is_augmentation=is_augmentation
	)
	# 検証DS
	valid_dataset = PrepareDatasets(config, mode="valid", img=val_img, depth=val_depth, mask=val_mask)
	valid_data = valid_dataset()
	# 使う場合は、訓練・検証データを返す
	# train_data, valid_data = setup_train_and_val(config, is_augmentation=True)
	return train_data, valid_data

def check_label(dataset):
	_, _ , mask = next(iter(dataset))
	# mask(Label)の分布を確認する
	print(f"label check...,", np.unique(mask))

def check_image(model, dataset):
	vis = Visualizer()
	vis.export(model, dataset, cmax=32)

def check_augmentation(dataset):
	vis = Visualizer()
	vis.confirm(dataset)

def main():
	config = Config()
	model = setup_env(config)
	# 検証・評価用（現在のモデルから）
	test_data = setup_test(config)
	# ラベル分布確認用
	# check_label(test_data)
	evaluate(model, test_data, None, None)

if __name__ == "__main__":
	main()