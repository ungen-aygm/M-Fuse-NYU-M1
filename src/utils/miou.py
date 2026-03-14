import torch
import numpy as np
"""
出典：Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J., & Zisserman, A. (2010). The Pascal Visual Object Classes (VOC) Challenge. International Journal of Computer Vision.
URL: https://link.springer.com/article/10.1007/s11263-009-0275-4
最終課題の評価はmIoUで行われています。
"""
class mIoU:
    def __init__(self, num_classes, ignore_index=255, device="cpu"):
      self.num_classes = num_classes
      self.ignore_index = ignore_index
      self.device = device
      self.confusion_matrix = torch.zeros(
            (self.num_classes, self.num_classes),
            dtype=torch.int64,
            device=self.device  # GPU上で計算するためにdeviceを指定
      )

    def reset(self):
      # 混同行列のリセット
      self.confusion_matrix.zero_()

    def update(self, pred, target):
      # 元のコード:
      mask = (target != self.ignore_index)

      pred = pred[mask]
      target = target[mask]

      conf_mat_flat = target.view(-1) * self.num_classes + pred.view(-1)
      bincount = torch.bincount(conf_mat_flat, minlength=self.num_classes**2)
      batch_conf = bincount.reshape(self.num_classes, self.num_classes)
      self.confusion_matrix += batch_conf.to(self.confusion_matrix.device)

    def compute(self):
      # 現在の混同行列からmIoUを計算する
      # conf_mat = self.confusion_matrix.double()
      conf_mat = self.confusion_matrix.float()
      """
      TP: True Positive（正しく予測されたピクセル数）
      FP: False Positive（誤って予測されたピクセル数）
      FN: False Negative（見逃したピクセル数）
      """
      tp = torch.diag(conf_mat)
      fp = conf_mat.sum(dim=0) - tp
      fn = conf_mat.sum(dim=1) - tp

      # ゼロ除算を避けるための微小な値 (epsilon)
      epsilon = 1e-6

      iou = tp / (tp + fp + fn + epsilon)

      # 全クラスのIoUの平均を計算 (NaNになったクラスは無視する)
      miou = torch.nanmean(iou)

      return iou, miou # 各クラスのIoUと、その平均(mIoU)を返す
