import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert hidden_dim % num_heads == 0

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.size(0)

        Q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            # 修复：确保掩码与scores在同一设备且数据类型兼容
            mask = mask.to(scores.device)

            # 处理不同维度的掩码
            if mask.dim() == 4:
                # [batch, num_heads, seq_len, seq_len] - 直接使用
                pass
            elif mask.dim() == 3:
                # [batch, seq_len, seq_len] - 扩展到多头
                mask = mask.unsqueeze(1)  # [batch, 1, seq_len, seq_len]
                mask = mask.repeat(1, self.num_heads, 1, 1)
            elif mask.dim() == 2:
                # [batch, seq_len] - 创建因果掩码
                mask = mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
                mask = mask.repeat(1, self.num_heads, scores.size(2), 1)
            else:
                raise ValueError(f"Unsupported mask dimension: {mask.dim()}")

            # 修复：使用正确的方法应用掩码，避免数据类型冲突
            # 将掩码转换为与scores相同的数据类型
            if scores.dtype != mask.dtype:
                # 对于混合精度训练，确保使用适当的值
                if scores.dtype == torch.float16:
                    # 使用半精度兼容的大负数
                    large_negative = torch.tensor(-1e4, dtype=torch.float16, device=scores.device)
                else:
                    large_negative = -1e9

                # 应用掩码
                scores = scores.masked_fill(mask == 0, large_negative)
            else:
                scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)

        output = self.out_linear(context)
        return output


class PositionwiseFeedForward(nn.Module):
    """位置前馈网络"""

    def __init__(self, hidden_dim: int, ff_dim: int, dropout: float = 0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(hidden_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""

    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_dim, ff_dim, dropout)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class GraphPositionalEncoding(nn.Module):
    """图位置编码"""

    def __init__(self, hidden_dim: int, max_nodes: int = 512):
        super(GraphPositionalEncoding, self).__init__()
        self.hidden_dim = hidden_dim
        self.max_nodes = max_nodes
        self.position_encoding = nn.Parameter(torch.zeros(1, max_nodes, hidden_dim))
        nn.init.xavier_uniform_(self.position_encoding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, hidden_dim = x.size()

        if num_nodes <= self.max_nodes:
            pos_encoding = self.position_encoding[:, :num_nodes, :]
        else:
            pos_encoding = F.interpolate(
                self.position_encoding.transpose(1, 2),
                size=num_nodes,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)

        return x + pos_encoding


class MultiGraphTransformer(nn.Module):
    """多图Transformer"""

    def __init__(self,
                 hidden_dim: int = 512,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 ff_dim: int = 2048,
                 dropout: float = 0.1,
                 max_nodes: int = 512):
        super(MultiGraphTransformer, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.positional_encoding = GraphPositionalEncoding(hidden_dim, max_nodes)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def create_padding_mask(self, node_features: torch.Tensor) -> Optional[torch.Tensor]:
        """创建填充掩码"""
        mask = (node_features.sum(dim=-1) != 0).unsqueeze(1).unsqueeze(2)
        return mask

    def forward(self, graph_embeddings: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = graph_embeddings.size()

        x = self.positional_encoding(graph_embeddings)
        x = self.dropout(x)

        # 修复：正确处理注意力掩码维度
        if attention_mask is not None:
            # 确保掩码在正确的设备上
            attention_mask = attention_mask.to(graph_embeddings.device)

            if attention_mask.dim() == 3:
                # [batch, seq_len, seq_len] -> 扩展到多头 [batch, num_heads, seq_len, seq_len]
                if attention_mask.size(1) != self.num_heads:
                    attention_mask = attention_mask.unsqueeze(1)  # [batch, 1, seq_len, seq_len]
                    attention_mask = attention_mask.repeat(1, self.num_heads, 1, 1)
            elif attention_mask.dim() == 2:
                # [batch, seq_len] -> 创建因果掩码
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
                attention_mask = attention_mask.repeat(1, self.num_heads, seq_len, 1)
            else:
                # 如果已经是4维，确保头数正确
                if attention_mask.size(1) != self.num_heads:
                    attention_mask = attention_mask.repeat(1, self.num_heads, 1, 1)
        else:
            # 如果没有提供掩码，创建全1掩码
            attention_mask = torch.ones(batch_size, self.num_heads, seq_len, seq_len,
                                        device=graph_embeddings.device)

        for layer in self.layers:
            x = layer(x, attention_mask)

        output = self.output_proj(x)
        return output


class CrossGraphAttention(nn.Module):
    """跨图注意力"""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super(CrossGraphAttention, self).__init__()
        self.cross_attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query_graph: torch.Tensor, key_value_graph: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attended = self.cross_attention(query_graph, key_value_graph, key_value_graph, mask)
        output = self.norm(query_graph + self.dropout(attended))
        return output


class HierarchicalGraphTransformer(nn.Module):
    """层次图Transformer"""

    def __init__(self,
                 hidden_dim: int = 512,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 ff_dim: int = 2048,
                 dropout: float = 0.1):
        super(HierarchicalGraphTransformer, self).__init__()

        self.intra_graph_transformer = MultiGraphTransformer(
            hidden_dim, num_layers, num_heads, ff_dim, dropout
        )

        self.cross_graph_attention = CrossGraphAttention(hidden_dim, num_heads, dropout)
        self.graph_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, multi_graph_embeddings: List[torch.Tensor],
                attention_masks: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        processed_graphs = []

        for i, graph_emb in enumerate(multi_graph_embeddings):
            mask = attention_masks[i] if attention_masks else None
            processed_graph = self.intra_graph_transformer(graph_emb, mask)
            processed_graphs.append(processed_graph)

        if len(processed_graphs) > 1:
            integrated_graphs = []
            for i in range(len(processed_graphs)):
                query_graph = processed_graphs[i]
                key_value_graphs = torch.cat([g for j, g in enumerate(processed_graphs) if j != i], dim=1)
                integrated_graph = self.cross_graph_attention(query_graph, key_value_graphs)
                integrated_graphs.append(integrated_graph)
        else:
            integrated_graphs = processed_graphs

        graph_representations = []
        for graph in integrated_graphs:
            graph_repr = self.graph_pool(graph.transpose(1, 2)).squeeze(-1)
            graph_representations.append(graph_repr)

        final_output = torch.cat(graph_representations, dim=-1)
        return final_output