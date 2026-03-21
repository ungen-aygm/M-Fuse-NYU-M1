from utils.preprocess import Setup_NYUV2
from config import Config
from glob import glob
import os
import numpy as np
from PIL import Image
from utils.graph import Visualizer
from segsetup import SegSetup, PrepareDatasets, split_data
import matplotlib.pyplot as plt
import argparse as arg 

def Parser():
	parser = arg.ArgumentParser(description="")
	parser.add_argument("-c", "--command", type=str, default="setup", help="")
	return parser.parse_args()

def main():
	config = Config()
	setup = Setup_NYUV2(config)
	args = Parser()
	if args.command == "view":
		plotter(setup, config)
	else:
		setup()
		print(f"画像/深度/ラベル抽出完了。")

def plotter(setup, config):
	vis = Visualizer()
	count = len(glob(os.path.join(setup.path["train"]["img"], "*")))
	if count > 0:
		vis.confirm_raw_data(setup)
	else:
		# 分割
		train_img, val_img, train_depth, val_depth, train_mask, val_mask = split_data()
		# 検証DS
		valid_dataset = PrepareDatasets(config, mode="valid", img=val_img, depth=val_depth, mask=val_mask)
		valid_data = valid_dataset()
		# 評価
		# test_dataset = PrepareDatasets(config, mode="test")
		# test_data = test_dataset()
		vis.confirm(valid_data)

if __name__ == "__main__":
	main()