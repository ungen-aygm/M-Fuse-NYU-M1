import torch
from models.late_fusion import LateFusion
from models.vit import semantic_encoder
from models.unet import UNet as unet
from config import Config


config = Config()
device = config.device
model = LateFusion(semantic_encoder, unet)
model.cpu()
model.eval()
try:
	dummy_rgb = torch.randn(1, 3, config.TARGET_SIZE[0], config.TARGET_SIZE[1]).cpu()
	dummy_depth = torch.randn(1, 1, config.TARGET_SIZE[0], config.TARGET_SIZE[1]).cpu()
	torch.onnx.export(model, (dummy_rgb, dummy_depth), "model.onnx")
except Exception as e:
	print(f"Export failed: {e}")