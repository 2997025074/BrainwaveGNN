import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Union, List
import warnings


class TRAdaptiveWaveletDecomposition(nn.Module):
    """TR自适应小波分解模块 - GPU优化版本"""

    def __init__(self,
                 reference_tr: float = 2.0,
                 num_bands: int = 5,
                 frequency_range: Tuple[float, float] = (0.008, 0.12),
                 wavelet_type: str = 'morl'):
        super().__init__()
        self.reference_tr = reference_tr
        self.num_bands = num_bands
        self.frequency_range = frequency_range
        self.wavelet_type = wavelet_type

        # 可学习的尺度参数 - 在GPU上初始化
        self.log_scale_params = nn.Parameter(
            torch.linspace(math.log(5), math.log(25), num_bands)
        )

        # 可学习的带宽参数
        self.bandwidth_params = nn.Parameter(
            torch.ones(num_bands) * 0.1
        )

        # Morlet小波参数
        self.morlet_center_freq = 1.0

        # 预计算小波基函数（如果可能）
        self._setup_wavelet_bases()

        print(f"TR自适应小波分解初始化:")
        print(f"  参考TR: {reference_tr}s")
        print(f"  频带数量: {num_bands}")
        print(f"  频率范围: {frequency_range[0]:.3f}-{frequency_range[1]:.3f} Hz")
        print(f"  小波类型: {wavelet_type}")

    def _setup_wavelet_bases(self):
        """预计算小波基函数（如果可能）"""
        # 这里可以预计算一些常用的小波基函数
        # 由于PyWavelets主要在CPU上，我们将在forward中动态计算
        pass

    def forward(self, timeseries: torch.Tensor, tr_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播 - 优化GPU使用

        Args:
            timeseries: [batch, regions, timepoints] 时间序列数据
            tr_values: [batch] 每个样本的TR值

        Returns:
            wavelet_coeffs: [batch, regions, bands, timepoints] 小波系数
            frequency_bands: [batch, bands, 2] 每个频带的实际频率范围
        """
        batch_size, num_regions, timepoints = timeseries.shape
        device = timeseries.device

        # 计算TR调整因子
        tr_factors = tr_values / self.reference_tr

        # 获取尺度参数（应用TR自适应）
        scales = self.get_adaptive_scales(tr_factors)  # [batch, bands]

        # 计算实际频率范围
        frequency_bands = self.compute_frequency_bands(scales, tr_values)  # [batch, bands, 2]

        # 执行小波分解 - 批量处理
        wavelet_coeffs = self.wavelet_transform_batch(timeseries, scales, device)

        return wavelet_coeffs, frequency_bands

    def get_adaptive_scales(self, tr_factors: torch.Tensor) -> torch.Tensor:
        """获取TR自适应的尺度参数"""
        base_scales = torch.exp(self.log_scale_params)  # [bands]
        batch_size = tr_factors.shape[0]
        scales = base_scales.unsqueeze(0) / tr_factors.unsqueeze(1)  # [batch, bands]
        return scales

    def compute_frequency_bands(self, scales: torch.Tensor, tr_values: torch.Tensor) -> torch.Tensor:
        """计算每个尺度的实际频率范围"""
        batch_size, num_bands = scales.shape
        device = scales.device

        # 向量化计算频率范围
        center_freqs = self.morlet_center_freq / (scales * tr_values.unsqueeze(1))
        bandwidths = self.bandwidth_params.unsqueeze(0)

        low_freqs = torch.max(
            torch.tensor(self.frequency_range[0], device=device),
            center_freqs * (1 - bandwidths / 2)
        )
        high_freqs = torch.min(
            torch.tensor(self.frequency_range[1], device=device),
            center_freqs * (1 + bandwidths / 2)
        )

        frequency_bands = torch.stack([low_freqs, high_freqs], dim=-1)
        return frequency_bands

    def wavelet_transform_batch(self, timeseries: torch.Tensor, scales: torch.Tensor,
                                device: torch.device) -> torch.Tensor:
        """批量小波变换 - 优化版本"""
        batch_size, num_regions, timepoints = timeseries.shape
        _, num_bands = scales.shape

        # 预分配GPU内存
        wavelet_coeffs = torch.zeros(batch_size, num_regions, num_bands, timepoints, device=device)

        # 使用PyTorch操作优化计算
        for b in range(batch_size):
            # 批量处理同一batch内的所有脑区
            batch_timeseries = timeseries[b]  # [regions, timepoints]

            for s in range(num_bands):
                scale = scales[b, s].item()

                # 对小波变换进行优化
                try:
                    # 将数据移到CPU进行小波变换
                    signal_batch = batch_timeseries.detach().cpu().numpy()

                    # 使用向量化操作处理所有脑区
                    coeff_magnitudes = []
                    for r in range(num_regions):
                        coefficients, _ = self._cwt_single(signal_batch[r], scale)
                        coeff_magnitudes.append(np.abs(coefficients[0]))

                    # 批量移回GPU
                    coeff_tensor = torch.from_numpy(np.array(coeff_magnitudes)).to(device)
                    wavelet_coeffs[b, :, s, :] = coeff_tensor

                except Exception as e:
                    warnings.warn(f"小波变换错误: {e}，使用零填充")
                    wavelet_coeffs[b, :, s, :] = torch.zeros(num_regions, timepoints, device=device)

        return wavelet_coeffs

    def _cwt_single(self, signal, scale):
        """单信号小波变换 - 分离出来以便可能的并行化"""
        import pywt
        return pywt.cwt(signal, [scale], self.wavelet_type, sampling_period=1.0)

    def get_band_interpretations(self) -> dict:
        """获取频带的生理学解释"""
        interpretations = {
            0: "高频带 (0.05-0.1Hz): 快速神经动态，反映瞬时神经活动",
            1: "中高频带 (0.027-0.073Hz): 典型功能连接，反映网络交互",
            2: "中频带: 默认模式网络活动",
            3: "中低频带: 慢波振荡，反映大规模脑网络协调",
            4: "低频带 (0.01-0.027Hz): 超慢振荡，反映全局脑状态"
        }
        return interpretations