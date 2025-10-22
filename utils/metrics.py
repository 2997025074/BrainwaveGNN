import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Optional, Dict


class MetricsCalculator:
    """评估指标计算器"""

    @staticmethod
    def calculate_classification_metrics(y_true: np.ndarray,
                                         y_pred: np.ndarray,
                                         y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        metrics = {}

        if len(y_true.shape) == 1 or y_true.shape[1] == 1:
            y_true_binary = y_true.flatten()
            y_pred_binary = y_pred.flatten()

            metrics['accuracy'] = accuracy_score(y_true_binary, y_pred_binary)
            metrics['precision'] = precision_score(y_true_binary, y_pred_binary, average='binary', zero_division=0)
            metrics['recall'] = recall_score(y_true_binary, y_pred_binary, average='binary', zero_division=0)
            metrics['f1'] = f1_score(y_true_binary, y_pred_binary, average='binary', zero_division=0)

            if y_prob is not None:
                try:
                    metrics['auc_roc'] = roc_auc_score(y_true_binary, y_prob.flatten())
                except:
                    metrics['auc_roc'] = 0.0

        else:
            metrics['accuracy'] = accuracy_score(y_true.argmax(axis=1), y_pred.argmax(axis=1))
            metrics['precision_macro'] = precision_score(y_true.argmax(axis=1), y_pred.argmax(axis=1),
                                                         average='macro', zero_division=0)
            metrics['recall_macro'] = recall_score(y_true.argmax(axis=1), y_pred.argmax(axis=1),
                                                   average='macro', zero_division=0)
            metrics['f1_macro'] = f1_score(y_true.argmax(axis=1), y_pred.argmax(axis=1),
                                           average='macro', zero_division=0)

        return metrics

    @staticmethod
    def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(mse)

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))

        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience: int = 10, min_delta: float = 0.0, restore_best: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss: float, model: nn.Module):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model: nn.Module):
        if self.restore_best:
            self.best_model_state = model.state_dict().copy()

    def restore_model(self, model: nn.Module):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


class FocalLoss(nn.Module):
    """Focal Loss用于处理类别不平衡"""

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


class LabelSmoothingLoss(nn.Module):
    """标签平滑损失"""

    def __init__(self, classes: int, smoothing: float = 0.1, dim: int = -1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))