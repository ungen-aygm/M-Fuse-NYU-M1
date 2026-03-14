import torch
import torch.nn as nn
import torch.nn.functional as F
# Focal Loss + Dice Loss(目的関数・勾配計算損失関数)
class FocalDiceLoss(nn.Module):
	def __init__(self, ignore_index=255, smooth=1.0,focal_weight=0.2, dice_weight=0.8, gamma=2.0, weight=None):
		super(FocalDiceLoss, self).__init__()
		self.ignore_index = ignore_index
		self.smooth = smooth
		self.focal_weight = focal_weight
		self.dice_weight = dice_weight
		self.focal_loss = FocalLoss(ignore_index=ignore_index, gamma=gamma, weight=weight)

	def forward(self, inputs, targets):
		# inputs: [Batch, Classes, H, W] (Logits)
		# targets(labels): [Batch, H, W] (Class Indices)		
		# 1. Cross Entropy計算
		focal_loss = self.focal_loss(inputs, targets)
		
		# 2. Dice Loss計算   
		# Logitsを確率(0~1)に変換
		probs = F.softmax(inputs, dim=1)
		num_classes = inputs.size(1)

		target_mask = (targets != self.ignore_index).unsqueeze(1).float()
		targets_safe = targets.clone()
		targets_safe[targets == self.ignore_index] = 0
		# TargetをOne-hotに変換 [B, H, W] -> [B, C, H, W]
		targets_one_hots = F.one_hot(targets_safe, num_classes).permute(0, 3, 1, 2).float()

		# dims = (0, 2, 3) # Batch, H, W 全体で集計
		dims = (2, 3) # Batch, H, W 全体で集計
		intersection = torch.sum(probs * targets_one_hots * target_mask, dims)
		cardinality = torch.sum((probs + targets_one_hots) * target_mask, dims)
		dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
		# 計算安定化
		dice_loss = (1 - dice_score).mean() # 全クラスの平均を取る
		
		# 3. Focal Loss + Dice Loss
		total_loss = (self.focal_weight * focal_loss) + (self.dice_weight * dice_loss)
		return total_loss

class FocalLoss(nn.Module):
	def __init__(self, ignore_index=255, gamma=2.0, weight=None):
		super(FocalLoss, self).__init__()
		self.gamma = gamma
		self.weight = weight
		self.ignore_index = ignore_index

	def forward(self, x, target):
		# x:[N, C, H, W], target:[N, H, W]
		ce_loss = F.cross_entropy(x, target, reduction='none', weight=self.weight, ignore_index=self.ignore_index)
		pt = torch.exp(-ce_loss)
		facal_term = (1 - pt) ** self.gamma
		return (facal_term * ce_loss).mean()