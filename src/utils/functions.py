import torch
from config import Config
from sklearn.metrics import precision_score, recall_score, f1_score
import torch.nn.functional as F
import random
import numpy as np

# mlx用
def ensure_contiguous_hook(module, input):
	return tuple(i.contiguous() if isinstance(i, torch.Tensor) else i for i in input)

# モデル保存
def save_model(model, count:int, miou:float, path:str):
	config = Config()
	device = config.device
	model = model.to(device)
	if count % 5 == 0 and count > 0:
		save_path = f"{path}/checkpoints/latefusion_epochs_{count}_miou_{miou:.2f}.pth"
		torch.save(model.state_dict(), save_path)
		print(f"--- Model saved at epoch {count} (mIoU: {miou:.4f}) ---")

# モデル読込
def load_model(model, path:str, is_load=False):
	config = Config()
	device = config.device
	if is_load is True:
		model = model.to(device)
		checkpoints = torch.load(path, map_location=device)
		# timmによるアップデート
		model_dict = model.state_dict()
		matched_dict = {
			k: v for k, v in checkpoints.items() 
			if k in model_dict and v.size() == model_dict[k].size()
		}
		model_dict.update(matched_dict)
		##############################################################
		model.load_state_dict(model_dict, strict=False)
		print(f"Successfully loaded {len(matched_dict)} layers from {path}")
		print("Note: ViT layers were skipped and kept as timm pretrained weights.")
		model.eval()
	return model

# 混同行列(適合率/再現率/F1 Score)
def cal_metrix(precisions, targets):
	precision_s = precision_score(targets, precisions, average='macro', zero_division=0)
	recalls = recall_score(targets, precisions, average='macro', zero_division=0)
	f1_scores = f1_score(targets, precisions, average='macro', zero_division=0)
	return precision_s, recalls, f1_scores

def set_seed(seed):
    """
    シードを固定する．

    Parameters
    ----------
    seed : int
        乱数生成に用いるシード値．
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False