import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Any


class Mixup:
    """Mixup数据增强 - 普通随机版本"""

    def __init__(self, alpha: float = 0.2, apply_probability: float = 0.8, num_classes: int = 2):
        self.alpha = alpha
        self.apply_probability = apply_probability
        self.num_classes = num_classes

    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        对批次数据应用Mixup - 完全随机配对
        """
        if np.random.random() > self.apply_probability:
            return batch

        features = batch['features']
        labels = batch['labels']
        batch_size = features.size(0)

        if batch_size < 2:
            return batch

        # 生成混合系数
        lam = np.random.beta(self.alpha, self.alpha)

        # 随机打乱批次顺序 - 完全随机配对
        indices = torch.randperm(batch_size, device=features.device)

        # 混合特征
        mixed_features = lam * features + (1 - lam) * features[indices]

        # 混合标签
        if len(labels.shape) == 1:
            # 单标签分类，转换为one-hot
            if labels.dtype != torch.long:
                labels = labels.long()

            labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=self.num_classes).float()
            mixed_labels = lam * labels_one_hot + (1 - lam) * labels_one_hot[indices]
        else:
            # 已经是one-hot或多标签
            mixed_labels = lam * labels + (1 - lam) * labels[indices]

        # 构建返回批次，包含TR值（如果有）
        result_batch = {
            'features': mixed_features,
            'labels': mixed_labels,
            'mixup_lambda': lam
        }

        # 保留TR值（如果有）
        if 'tr_values' in batch:
            result_batch['tr_values'] = batch['tr_values']

        return result_batch


class ProgressiveMixup:
    """渐进式Mixup - 根据训练阶段调整混合强度"""

    def __init__(self, base_alpha: float = 0.2, num_epochs: int = 100,
                 apply_probability: float = 0.8, num_classes: int = 2):
        self.base_alpha = base_alpha
        self.num_epochs = num_epochs
        self.apply_probability = apply_probability
        self.num_classes = num_classes
        self.current_epoch = 0

    def set_epoch(self, epoch):
        """设置当前训练epoch"""
        self.current_epoch = epoch

    def get_current_alpha(self):
        """根据训练阶段获取当前alpha值"""
        progress = self.current_epoch / self.num_epochs

        if progress < 0.3:  # 前期：较弱混合
            return self.base_alpha * 0.5
        elif progress < 0.7:  # 中期：标准混合
            return self.base_alpha
        else:  # 后期：较强混合
            return self.base_alpha * 1.5

    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """应用渐进式Mixup"""
        if np.random.random() > self.apply_probability:
            return batch

        current_alpha = self.get_current_alpha()

        features = batch['features']
        labels = batch['labels']
        batch_size = features.size(0)

        if batch_size < 2:
            return batch

        # 使用当前阶段的alpha生成混合系数
        lam = np.random.beta(current_alpha, current_alpha)

        # 随机打乱批次顺序
        indices = torch.randperm(batch_size, device=features.device)

        # 混合特征和标签
        mixed_features = lam * features + (1 - lam) * features[indices]

        if len(labels.shape) == 1:
            if labels.dtype != torch.long:
                labels = labels.long()
            labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=self.num_classes).float()
            mixed_labels = lam * labels_one_hot + (1 - lam) * labels_one_hot[indices]
        else:
            mixed_labels = lam * labels + (1 - lam) * labels[indices]

        result_batch = {
            'features': mixed_features,
            'labels': mixed_labels,
            'mixup_lambda': lam,
            'mixup_alpha': current_alpha  # 记录当前使用的alpha
        }

        if 'tr_values' in batch:
            result_batch['tr_values'] = batch['tr_values']

        return result_batch