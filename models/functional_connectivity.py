import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Optional
import torch.nn.functional as F


class FunctionalConnectivityGraphBuilder(nn.Module):
    """功能连接图构建器 - 修复维度匹配问题"""

    def __init__(self, sparsity_threshold=0.2, graph_type='weighted', feature_dim=64):
        super().__init__()
        self.sparsity_threshold = sparsity_threshold
        self.graph_type = graph_type
        self.feature_dim = feature_dim

        # 添加图编码器来统一特征维度
        self.feature_encoder = nn.Sequential(
            nn.Linear(11, 32),  # 输入维度是11 (5个节点特征 + 6个图属性)
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, feature_dim)
        )

    def forward(self, wavelet_coeffs, frequency_bands):
        """
        构建多频带功能连接图 - 修复维度问题
        """
        batch_size, num_regions, num_bands, timepoints = wavelet_coeffs.shape
        device = wavelet_coeffs.device

        # 预分配GPU内存 - 使用正确的特征维度
        adjacency_sequence = torch.zeros(batch_size, num_bands, num_regions, num_regions, device=device)
        graph_features = torch.zeros(batch_size, num_bands, self.feature_dim, device=device)

        for batch_idx in range(batch_size):
            for band_idx in range(num_bands):
                # 提取当前频带的小波系数 [regions, timepoints]
                band_coeffs = wavelet_coeffs[batch_idx, :, band_idx, :]

                # 在GPU上计算功能连接矩阵
                adjacency_matrix = self.compute_functional_connectivity_gpu(band_coeffs)

                # 在GPU上应用稀疏化
                sparse_adjacency = self.apply_sparsity_gpu(adjacency_matrix)

                # 存储邻接矩阵
                adjacency_sequence[batch_idx, band_idx] = sparse_adjacency

                # 在GPU上计算图特征并编码到统一维度
                graph_feature_raw = self.compute_graph_features_gpu(sparse_adjacency, band_coeffs)
                graph_feature_encoded = self.feature_encoder(graph_feature_raw.unsqueeze(0)).squeeze(0)
                graph_features[batch_idx, band_idx] = graph_feature_encoded

        return adjacency_sequence, graph_features

    def compute_functional_connectivity_gpu(self, band_coeffs: torch.Tensor) -> torch.Tensor:
        """在GPU上计算功能连接矩阵"""
        # 使用PyTorch计算相关矩阵
        band_coeffs_normalized = band_coeffs - band_coeffs.mean(dim=1, keepdim=True)
        norm = torch.norm(band_coeffs_normalized, dim=1, keepdim=True)
        norm[norm == 0] = 1  # 避免除零

        correlation_matrix = torch.mm(band_coeffs_normalized, band_coeffs_normalized.t()) / (norm * norm.t())
        correlation_matrix = torch.nan_to_num(correlation_matrix, nan=0.0)
        correlation_matrix.fill_diagonal_(0)  # 对角线置零

        return correlation_matrix

    def apply_sparsity_gpu(self, adjacency_matrix: torch.Tensor) -> torch.Tensor:
        """在GPU上应用稀疏化"""
        # 获取下三角矩阵的索引
        mask = torch.tril(torch.ones_like(adjacency_matrix), diagonal=-1).bool()
        values = adjacency_matrix[mask]

        if len(values) > 0:
            # 计算阈值
            threshold = torch.quantile(torch.abs(values), 1 - self.sparsity_threshold)

            if self.graph_type == 'binary':
                sparse_adjacency = (torch.abs(adjacency_matrix) >= threshold).float()
            else:
                sparse_adjacency = adjacency_matrix * (torch.abs(adjacency_matrix) >= threshold).float()
        else:
            sparse_adjacency = adjacency_matrix

        return sparse_adjacency

    def compute_graph_features_gpu(self, adjacency_matrix: torch.Tensor, band_coeffs: torch.Tensor) -> torch.Tensor:
        """在GPU上计算图特征 - 返回11维原始特征"""
        # 节点特征
        node_features = self.compute_node_features_gpu(band_coeffs)

        # 图级别特征
        graph_properties = self.compute_graph_properties_gpu(adjacency_matrix)

        # 合并特征 - 总共11维
        combined_features = torch.cat([node_features, graph_properties])

        return combined_features

    def compute_node_features_gpu(self, band_coeffs: torch.Tensor) -> torch.Tensor:
        """在GPU上计算节点特征 - 返回5维特征"""
        num_regions, timepoints = band_coeffs.shape

        # 批量计算所有节点的特征
        mean_features = torch.mean(band_coeffs, dim=1)
        std_features = torch.std(band_coeffs, dim=1)
        energy_features = torch.sum(band_coeffs ** 2, dim=1)
        max_amplitude_features = torch.max(torch.abs(band_coeffs), dim=1)[0]
        spectral_entropy_features = self.compute_spectral_entropy_gpu(band_coeffs)

        # 堆叠特征 [num_regions, 5] -> 平均池化为 [5]
        node_feature_matrix = torch.stack([
            mean_features, std_features, energy_features,
            max_amplitude_features, spectral_entropy_features
        ], dim=1)

        avg_node_features = torch.mean(node_feature_matrix, dim=0)

        return avg_node_features

    def compute_spectral_entropy_gpu(self, signal: torch.Tensor) -> torch.Tensor:
        """在GPU上计算谱熵"""
        # 使用FFT计算功率谱
        power_spectrum = torch.abs(torch.fft.fft(signal, dim=1)) ** 2
        power_spectrum = power_spectrum[:, :signal.shape[1] // 2]  # 取正频率部分

        # 归一化
        power_sum = torch.sum(power_spectrum, dim=1, keepdim=True)
        power_sum[power_sum == 0] = 1  # 避免除零
        power_spectrum_normalized = power_spectrum / power_sum

        # 计算谱熵
        spectral_entropy = -torch.sum(power_spectrum_normalized * torch.log(power_spectrum_normalized + 1e-10), dim=1)

        return spectral_entropy

    def compute_graph_properties_gpu(self, adjacency_matrix: torch.Tensor) -> torch.Tensor:
        """在GPU上计算图属性 - 返回6维特征"""
        # 由于NetworkX在CPU上，我们计算一些基本的图属性
        num_nodes = adjacency_matrix.shape[0]

        # 边数量
        num_edges = torch.sum(adjacency_matrix != 0) / 2  # 无向图

        # 图密度
        max_possible_edges = num_nodes * (num_nodes - 1) / 2
        density = num_edges / max_possible_edges if max_possible_edges > 0 else 0

        # 平均度
        degrees = torch.sum(adjacency_matrix != 0, dim=1)
        average_degree = torch.mean(degrees.float())

        # 聚类系数（简化计算）
        clustering_coeff = self.approximate_clustering_coefficient_gpu(adjacency_matrix)

        # 全局效率（简化计算）
        global_efficiency = self.approximate_global_efficiency_gpu(adjacency_matrix)

        # 模块度（简化计算）
        modularity = self.approximate_modularity_gpu(adjacency_matrix)

        graph_property_vector = torch.tensor([
            num_edges.item(), density.item(), average_degree.item(),
            clustering_coeff.item(), global_efficiency.item(), modularity.item()
        ], device=adjacency_matrix.device)

        return graph_property_vector

    def approximate_clustering_coefficient_gpu(self, adjacency_matrix: torch.Tensor) -> torch.Tensor:
        """在GPU上近似计算聚类系数"""
        # 简化的聚类系数计算
        A = (adjacency_matrix != 0).float()
        A_squared = torch.mm(A, A)
        triangles = torch.sum(A_squared * A) / 6  # 三角形数量

        # 计算可能的三角形数量
        degrees = torch.sum(A, dim=1)
        possible_triangles = torch.sum(degrees * (degrees - 1)) / 2

        if possible_triangles > 0:
            clustering_coeff = triangles / possible_triangles
        else:
            clustering_coeff = torch.tensor(0.0, device=adjacency_matrix.device)

        return clustering_coeff

    def approximate_global_efficiency_gpu(self, adjacency_matrix: torch.Tensor) -> torch.Tensor:
        """在GPU上近似计算全局效率"""
        # 简化的全局效率计算
        A = (adjacency_matrix != 0).float()
        num_nodes = A.shape[0]

        if num_nodes <= 1:
            return torch.tensor(0.0, device=adjacency_matrix.device)

        # 计算最短路径的倒数（简化）
        efficiency = torch.sum(1.0 / (A + torch.eye(num_nodes, device=A.device))) / (num_nodes * (num_nodes - 1))

        return efficiency

    def approximate_modularity_gpu(self, adjacency_matrix: torch.Tensor) -> torch.Tensor:
        """在GPU上近似计算模块度"""
        # 简化的模块度计算
        A = (adjacency_matrix != 0).float()
        num_nodes = A.shape[0]

        if num_nodes == 0:
            return torch.tensor(0.0, device=adjacency_matrix.device)

        # 随机划分社区（简化）
        community_assignments = torch.randint(0, 2, (num_nodes,), device=A.device)

        # 计算模块度
        total_edges = torch.sum(A) / 2
        if total_edges == 0:
            return torch.tensor(0.0, device=adjacency_matrix.device)

        modularity = 0.0
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    expected = torch.sum(A[i]) * torch.sum(A[j]) / (2 * total_edges)
                    if community_assignments[i] == community_assignments[j]:
                        modularity += (A[i, j] - expected)

        modularity = modularity / (2 * total_edges)

        return modularity


class GraphSequenceProcessor(nn.Module):
    """图序列处理器 - 修复版本"""

    def __init__(self, feature_dim=64):
        super().__init__()
        self.feature_dim = feature_dim

        # 图编码器 - 输入维度应该是feature_dim，因为特征已经在FunctionalConnectivityGraphBuilder中编码过了
        self.graph_encoder = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, feature_dim)
        )

    def forward(self, adjacency_sequence, graph_features):
        """
        处理图序列 - 修复版本
        """
        batch_size, num_bands, feature_dim = graph_features.shape
        # 使用图编码器处理特征
        sequence_features = self.graph_encoder(graph_features)
        # 创建注意力掩码
        attention_mask = torch.ones(batch_size, num_bands, num_bands, device=graph_features.device)
        return sequence_features, attention_mask