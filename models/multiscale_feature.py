import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional, Union


class ResidualBlock(nn.Module):
    """残差卷积块"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, dropout: float = 0.1):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(dropout)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout2(out)

        if out.size() != identity.size():
            identity = F.interpolate(identity, size=out.shape[2:], mode='linear', align_corners=False)

        out += identity
        out = self.relu2(out)

        return out


class MultiScaleTemporalFeatureExtractor(nn.Module):
    """多尺度时间特征提取器"""

    def __init__(self, input_channels: int, time_windows: List[int] = [11, 17, 29],
                 hidden_dims: List[int] = [64, 128, 256], dropout: float = 0.2):
        super().__init__()
        self.time_windows = time_windows
        self.hidden_dims = hidden_dims

        self.scale_branches = nn.ModuleList()
        for window_size in time_windows:
            branch = self._create_scale_branch(input_channels, window_size, hidden_dims, dropout)
            self.scale_branches.append(branch)

        total_features = len(time_windows) * hidden_dims[-1]
        self.scale_attention = nn.Sequential(
            nn.Linear(total_features, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, len(time_windows)),
            nn.Softmax(dim=-1)
        )

        self.feature_fusion = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256)
        )

    def _create_scale_branch(self, input_channels: int, kernel_size: int,
                             hidden_dims: List[int], dropout: float) -> nn.Sequential:
        layers = []
        current_channels = input_channels

        for i, hidden_dim in enumerate(hidden_dims):
            residual_block = ResidualBlock(
                in_channels=current_channels,
                out_channels=hidden_dim,
                kernel_size=kernel_size if i == 0 else 3,
                stride=1,
                padding=(kernel_size - 1) // 2 if i == 0 else 1,
                dropout=dropout
            )
            layers.append(residual_block)
            current_channels = hidden_dim

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        batch_size, channels, timepoints = x.shape
        scale_outputs = []

        for branch in self.scale_branches:
            scale_feat = branch(x)
            pooled = F.adaptive_avg_pool1d(scale_feat, 1).squeeze(-1)
            scale_outputs.append(pooled)

        all_scale_features = torch.cat(scale_outputs, dim=1)
        scale_weights = self.scale_attention(all_scale_features)
        fused_features = self.feature_fusion(all_scale_features)

        return fused_features, scale_weights, scale_outputs


class FrequencyBandProcessor(nn.Module):
    """频带处理器"""

    def __init__(self, num_regions: int, num_bands: int,
                 time_windows: List[int] = [11, 17, 29],
                 hidden_dims: List[int] = [64, 128, 256],
                 dropout: float = 0.2):
        super().__init__()
        self.num_bands = num_bands
        self.num_regions = num_regions

        self.band_processors = nn.ModuleList()
        for _ in range(num_bands):
            processor = MultiScaleTemporalFeatureExtractor(
                input_channels=num_regions,
                time_windows=time_windows,
                hidden_dims=hidden_dims,
                dropout=dropout
            )
            self.band_processors.append(processor)

        self.band_attention = nn.Sequential(
            nn.Linear(num_bands * 256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_bands),
            nn.Softmax(dim=-1)
        )

    def forward(self, wavelet_coeffs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        batch_size, num_regions, num_bands, timepoints = wavelet_coeffs.shape
        band_outputs = []
        all_scale_weights = []

        for band_idx in range(num_bands):
            band_data = wavelet_coeffs[:, :, band_idx, :]
            band_feat, scale_weights, _ = self.band_processors[band_idx](band_data)
            band_outputs.append(band_feat)
            all_scale_weights.append(scale_weights)

        band_features = torch.stack(band_outputs, dim=1)
        band_features_flat = band_features.view(batch_size, -1)
        band_weights = self.band_attention(band_features_flat)

        return band_features, band_weights, all_scale_weights