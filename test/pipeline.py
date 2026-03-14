import matplotlib.patches as mpatches
from tqdm import tqdm
import torch
import numpy as np
from datetime import datetime
from utils.dataset import ColorMap, NYUv2Dataset
from utils.normalize import ImageNet as nm
from utils.preprocess import Preprocess as Loader
from utils.augmentations import Augmentation, AugmentationParams
from config import Config
from PIL import Image
import mlx.core as mx
from time import time
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
from utils.graph import Visualizer
from glob import glob as find
from utils.evaluate import evaluate
from segsetup import SegSetup, PrepareDatasets, split_data

def label_check(dataset):
	_, _ , mask = next(iter(dataset))
	# mask(Label)の分布を確認する
	print(f"label check...,", np.unique(mask))

def augmentation(config:Config):
	# データ拡張：configにパラメータの上書き
	config.DEGREES = 30

def main():
	config = Config()
	is_augmentation=True
	setup = SegSetup(config)
	model, _, _, _ = setup(is_load_model=True)
	######################################################################
	# 分割
	train_img, val_img, train_depth, val_depth, train_mask, val_mask = split_data()
	# 検証DS（データリークなし）
	valid_dataset = PrepareDatasets(config, mode="valid", img=val_img, depth=val_depth, mask=val_mask)
	valid_data = valid_dataset()
	# 評価DS
	# test_dataset = PrepareDatasets(config, mode="test")
	# test_data = test_dataset()
	# 訓練データセット
	train_dataset = PrepareDatasets(config, mode="train", img=train_img, depth=train_depth, mask=train_mask)
	# データ拡張：パラメータ設定
	augmentation(config)
	train_data = train_dataset(
		augmentation=train_dataset.augmentation,
		# データ拡張: ON/OFF
		is_augmentation=True
	)
	# ラベルチェック
	# label_check(valid_data)
	vis = Visualizer()
	# 検証画像などを出す。
	vis.export(model, valid_data, cmax=32)
	# データ拡張確認
	# vis.confirm(train_data)
	# 検証・評価用（現在のモデルから）
	# evaluate(model, valid_data, None, None)

if __name__ == "__main__":
	main()