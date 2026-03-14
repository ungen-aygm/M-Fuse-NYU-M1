import mlx.core as mx
import torch
import numpy as np
from config import Config
from utils.dataset import ColorMap
from utils.normalize import ImageNet as nm
from utils.miou import mIoU
from tqdm import tqdm
from config import Config
import matplotlib.pyplot as plt
from utils.graph import Visualizer as ClsRank
from utils.functions import *

# 評価用
def evaluate(model, dataloader, writer=None, log=None):
    config = Config()
    device = config.device
    set_seed(config.SEED)
    model.eval()
    tracker = mIoU(num_classes=config.NUM_CLASSES)
    # クラスランキング
    rank = ClsRank()
    loop = tqdm(dataloader, desc="Fusion Evaluate...", leave=True)
    miou = torch.tensor(0.0).to(device)
    with torch.no_grad():
        for idx, (_rgb, _depth, _targets) in enumerate(loop):
            rgb, depth, targets = _rgb.to(device), _depth.to(device), _targets.to(device)
            # 1. 推論
            outputs = model(rgb, depth)
            predicted = torch.argmax(outputs, dim=1)
            tracker.update(predicted, targets)
            rank.view(rgb, depth, predicted, targets, idx=idx, miou=miou.detach().cpu().numpy())
    
    iou_score, miou = tracker.compute()
    print(f"\n[Results / Evaluate mIoU:{miou.item():.4f}]")
    rank.cls_rank(iou_score, miou)
    return miou, 0.0