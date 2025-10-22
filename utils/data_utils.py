import torch
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from typing import Tuple, Optional, Dict, Any


def create_dataloaders(dataset: Dataset,
                       batch_size: int = 32,
                       train_ratio: float = 0.7,
                       val_ratio: float = 0.15,
                       test_ratio: float = 0.15,
                       num_workers: int = 4,
                       random_seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练、验证、测试数据加载器 - 修复版本
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例总和必须为1"

    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(random_seed)
    )

    # 修复：只在数据在CPU上时使用pin_memory
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),  # 只在有GPU时启用
        persistent_workers=num_workers > 0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0
    )

    return train_loader, val_loader, test_loader


class BrainDataset(Dataset):
    """脑网络数据集类 - 修复版本"""

    def __init__(self, features: np.ndarray, labels: np.ndarray, tr_values: Optional[np.ndarray] = None):
        # 修复：始终在CPU上创建tensor，让DataLoader处理GPU传输
        self.features = torch.FloatTensor(features)

        # 确保标签是长整型
        if labels.dtype == np.float32 or labels.dtype == np.float64:
            labels = labels.astype(np.int64)
        self.labels = torch.LongTensor(labels)

        self.tr_values = torch.FloatTensor(tr_values) if tr_values is not None else None

        print(f"数据集初始化 - 特征: {self.features.shape}, 标签: {self.labels.shape}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        sample = {
            'features': self.features[idx],
            'labels': self.labels[idx]
        }
        if self.tr_values is not None:
            sample['tr_values'] = self.tr_values[idx]
        return sample