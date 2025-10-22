import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
import numpy as np


class ClassificationHead(nn.Module):
    """分类头"""

    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [512, 256, 128],
                 num_classes: int = 1,
                 dropout_rate: float = 0.2,
                 task_type: str = "classification"):
        super(ClassificationHead, self).__init__()

        self.task_type = task_type
        self.num_classes = num_classes

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)

        if task_type == "classification":
            if num_classes == 1:
                self.output_layer = nn.Sequential(
                    nn.Linear(prev_dim, 1),
                    nn.Sigmoid()
                )
            else:
                self.output_layer = nn.Sequential(
                    nn.Linear(prev_dim, num_classes),
                    nn.Softmax(dim=-1)
                )
        else:
            self.output_layer = nn.Linear(prev_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden_layers(x)
        return self.output_layer(x)


class MultiTaskClassificationHead(nn.Module):
    """多任务分类头"""

    def __init__(self,
                 input_dim: int,
                 task_configs: Dict[str, Dict],
                 shared_hidden_dims: List[int] = [512, 256]):
        super(MultiTaskClassificationHead, self).__init__()

        self.task_configs = task_configs
        self.task_names = list(task_configs.keys())

        shared_layers = []
        prev_dim = input_dim

        for hidden_dim in shared_hidden_dims:
            shared_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim

        self.shared_layers = nn.Sequential(*shared_layers)

        self.task_heads = nn.ModuleDict()
        for task_name, config in task_configs.items():
            task_layers = []
            task_prev_dim = prev_dim

            for task_hidden in config.get("hidden_dims", [128]):
                task_layers.extend([
                    nn.Linear(task_prev_dim, task_hidden),
                    nn.BatchNorm1d(task_hidden),
                    nn.GELU(),
                    nn.Dropout(config.get("dropout", 0.1))
                ])
                task_prev_dim = task_hidden

            num_classes = config["num_classes"]
            task_type = config.get("task_type", "classification")

            if task_type == "classification":
                if num_classes == 1:
                    output_layer = nn.Sequential(
                        nn.Linear(task_prev_dim, 1),
                        nn.Sigmoid()
                    )
                else:
                    output_layer = nn.Sequential(
                        nn.Linear(task_prev_dim, num_classes),
                        nn.Softmax(dim=-1)
                    )
            else:
                output_layer = nn.Linear(task_prev_dim, num_classes)

            task_layers.append(output_layer)
            self.task_heads[task_name] = nn.Sequential(*task_layers)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared_features = self.shared_layers(x)

        outputs = {}
        for task_name in self.task_names:
            outputs[task_name] = self.task_heads[task_name](shared_features)

        return outputs


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


class CompleteModel(nn.Module):
    """完整模型"""

    def __init__(self,
                 graph_transformer: nn.Module,
                 input_dim: int,
                 num_classes: int = 1,
                 hidden_dims: List[int] = [512, 256, 128],
                 dropout_rate: float = 0.2,
                 task_type: str = "classification"):
        super(CompleteModel, self).__init__()

        self.graph_transformer = graph_transformer
        self.classification_head = ClassificationHead(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            task_type=task_type
        )

    def forward(self, graph_embeddings: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        graph_features = self.graph_transformer(graph_embeddings, attention_mask)
        output = self.classification_head(graph_features)
        return output


class MultiTaskCompleteModel(nn.Module):
    """多任务完整模型"""

    def __init__(self,
                 graph_transformer: nn.Module,
                 input_dim: int,
                 task_configs: Dict[str, Dict]):
        super(MultiTaskCompleteModel, self).__init__()

        self.graph_transformer = graph_transformer
        self.multi_task_head = MultiTaskClassificationHead(
            input_dim=input_dim,
            task_configs=task_configs
        )

    def forward(self, graph_embeddings: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        graph_features = self.graph_transformer(graph_embeddings, attention_mask)
        outputs = self.multi_task_head(graph_features)
        return outputs