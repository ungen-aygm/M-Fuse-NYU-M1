import os, json, yaml, shutil
from datetime import datetime
import numpy as np

from config import Config

def date():
	return datetime.now().strftime("%Y%m%d-%H%M%S")

class Parameters:
	parameter = {}
	def __init__(
		self, 
		name:str="",
		seed:int=0,
		model={},
		training={},
		dataset={},
		paths={},
	):
		self.parameter = {}
		self.parameter["experiment_name"] = name
		self.parameter["date"] = date()
		self.parameter["seed"] = seed
		self.parameter["model"] = model
		self.parameter["training"] = training
		self.parameter["dataset"] = dataset
		self.parameter["paths"] = paths

class Log:
	def __init__(self, title:str):
		self.config = Config()
		model_name = self.config.MODEL_NAME
		self.parameter = Parameters(name=title, seed=self.config.SEED)
		self.date = datetime.now().strftime("%Y%m%d-%H%M%S")
	
	def __call__(self, model, training, dataset, paths, is_save=False):
		self.parameter.parameter["model"] = model
		self.parameter.parameter["training"] = training
		self.parameter.parameter["dataset"] = dataset
		self.parameter.parameter["paths"] = paths
		if is_save:
			os.makedirs(self.dir, exist_ok=True)
			os.makedirs(f"{paths['train_dir']}", exist_ok=True)
			os.makedirs(f"{paths['validation_dir']}", exist_ok=True)
			os.makedirs(f"{paths['checkpoint_dir']}", exist_ok=True)
			os.makedirs(f"{paths['predict_dir']}", exist_ok=True)
			# os.makedirs(f"{paths['model_dir']}", exist_ok=True)
			self.save(f"{self.dir}/config.yaml",{
				"experiment_name": self.parameter.parameter["experiment_name"],
				"date": self.parameter.parameter["date"],
				"model": model,
				"training": training,
				"dataset": dataset,
				"paths": paths,
			})
	
	@property
	def dir(self):
		return os.path.join("runs", f"{self.config.MODEL_NAME}_{self.date}")

	def save(self, save_path:str, config={}):
		with open(save_path, 'w') as f:
			yaml.dump(config, f)
		print(f"Config File[{save_path}] Create, {self.date}.")
	
	def load(self, path:str):
		config = {}
		if not os.path.is_file(savepath):
			print(f"Not Found: {path}.")
			return config

		with open(path, 'r') as f:
			config = yaml.safe_load(f)
		return config

	def copy(self):
		config_yaml = self.config.CONFIG_PATH
		shutil.copy(config_yaml, os.path.join(self.dir, "config.yaml"))
		print(f"Config Saved: {self.dir}.")
	
	def dump(self):
		print("TensorBoard Logs:-------------------------------------------")
		print(f"Log DIR > {self.dir}")
		print(self.parameter.parameter["experiment_name"])
		print(self.parameter.parameter["model"])
		print(self.parameter.parameter["training"])
		print(self.parameter.parameter["dataset"])
		print(self.parameter.parameter["paths"])