import torch
import torch.nn as nn
import yaml
import argparse
import os
import numpy as np
from typing import Dict, Any
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    TRAdaptiveWaveletDecomposition,
    FrequencyBandProcessor,
    FunctionalConnectivityGraphBuilder,
    GraphSequenceProcessor,
    MultiGraphTransformer
)
from training import ModelTrainer, TrainerConfig
from utils.data_utils import BrainDataset, create_dataloaders
from data_processor import ABIDEDataPreprocessor


def setup_gpu_environment():
    """设置GPU环境 - 优化版本"""
    if not torch.cuda.is_available():
        print("警告: CUDA不可用，将使用CPU")
        return torch.device("cpu")

    # 自动选择最佳GPU
    if torch.cuda.device_count() > 1:
        # 选择内存最多的GPU
        gpu_memory = []
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            gpu_memory.append(torch.cuda.get_device_properties(i).total_memory)

        best_gpu = np.argmax(gpu_memory)
        torch.cuda.set_device(best_gpu)
        device = torch.device(f"cuda:{best_gpu}")
    else:
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")

    # 启用cudnn自动优化
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # 显示GPU信息
    gpu_props = torch.cuda.get_device_properties(device)
    print(f"使用GPU: {torch.cuda.get_device_name(device)}")
    print(f"GPU内存: {gpu_props.total_memory / 1024 ** 3:.1f} GB")
    print(f"CUDA能力: {gpu_props.major}.{gpu_props.minor}")

    return device


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print("配置文件加载成功")
    return config


def load_preprocessed_data(data_dir: str):
    """加载预处理数据 - 修复版本"""
    print(f"加载预处理数据从: {data_dir}")

    required_files = [
        "timeseries_data.npy", "labels.npy",
        "tr_values.npy", "sites.pkl", "processing_info.pkl"
    ]

    for file in required_files:
        file_path = os.path.join(data_dir, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到处理后的数据文件: {file_path}")

    # 加载数据
    timeseries_data = np.load(os.path.join(data_dir, "timeseries_data.npy"))
    labels = np.load(os.path.join(data_dir, "labels.npy"))
    tr_values = np.load(os.path.join(data_dir, "tr_values.npy"))

    # 确保标签是整数类型
    if labels.dtype != np.int64:
        labels = labels.astype(np.int64)

    # 加载处理信息
    import pickle
    with open(os.path.join(data_dir, "processing_info.pkl"), 'rb') as f:
        processing_info = pickle.load(f)

    print(f"数据加载完成: {timeseries_data.shape}, 标签: {labels.shape}")
    print(f"标签分布 - ASD: {sum(labels == 1)}, TC: {sum(labels == 0)}")

    return timeseries_data, labels, tr_values, processing_info


class TRAdaptiveBrainNetwork(nn.Module):
    """TR自适应脑网络完整模型 - 修复版本"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        model_config = config['model']

        # TR自适应小波分解
        self.wavelet_decomposition = TRAdaptiveWaveletDecomposition(
            reference_tr=model_config['reference_tr'],
            num_bands=model_config['num_frequency_bands'],
            frequency_range=model_config['frequency_range']
        )

        # 频带处理器
        self.frequency_band_processor = FrequencyBandProcessor(
            num_regions=model_config.get('actual_num_regions', 116),  # 默认AAL图谱116区
            num_bands=model_config['num_frequency_bands'],
            time_windows=model_config['time_windows'],
            hidden_dims=model_config['hidden_dims'],
            dropout=model_config['dropout']
        )

        # 功能连接图构建器 - 修复：传递正确的特征维度
        self.graph_builder = FunctionalConnectivityGraphBuilder(
            sparsity_threshold=model_config['sparsity_threshold'],
            graph_type='weighted',
            feature_dim=model_config['graph_feature_dim']  # 传递特征维度
        )

        # 图序列处理器 - 修复：传递正确的特征维度
        self.graph_processor = GraphSequenceProcessor(
            feature_dim=model_config['graph_feature_dim']
        )

        # 多图Transformer
        self.transformer = MultiGraphTransformer(
            hidden_dim=512,
            num_layers=6,
            num_heads=8
        )

        # 分类器
        input_dim = 512  # Transformer的输出维度
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(model_config['dropout']),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(model_config['dropout'] * 0.7),
            nn.Linear(128, model_config['num_classes'])
        )

    def forward(self, timeseries: torch.Tensor, tr_values: torch.Tensor = None):
        """
        前向传播 - 修复版本
        """
        batch_size = timeseries.shape[0]
        device = timeseries.device

        # 如果没有提供TR值，使用默认值
        if tr_values is None:
            tr_values = torch.ones(batch_size, device=device) * self.config['model']['reference_tr']

        # 1. 小波分解
        wavelet_coeffs, frequency_bands = self.wavelet_decomposition(timeseries, tr_values)

        # 2. 多尺度特征提取
        band_features, band_weights, scale_weights = self.frequency_band_processor(wavelet_coeffs)

        # 3. 构建功能连接图
        adjacency_sequence, graph_features = self.graph_builder(wavelet_coeffs, frequency_bands)

        # 4. 处理图序列
        sequence_features, attention_mask = self.graph_processor(adjacency_sequence, graph_features)

        # 5. Transformer处理
        transformer_output = self.transformer(sequence_features, attention_mask)

        # 6. 全局平均池化获取图级别特征
        if transformer_output.dim() == 3:  # [batch, seq_len, features]
            graph_features = transformer_output.mean(dim=1)
        else:
            graph_features = transformer_output

        # 7. 分类
        output = self.classifier(graph_features)

        return output


def main():
    parser = argparse.ArgumentParser(description='TR自适应脑网络分析')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='配置文件路径')
    parser.add_argument('--processed_data_dir', type=str, help='预处理数据目录路径')
    parser.add_argument('--preprocess', action='store_true', help='是否重新预处理数据')
    parser.add_argument('--data_dir', type=str, help='原始数据目录（如果重新预处理）')
    args = parser.parse_args()

    # 设置GPU环境
    device = setup_gpu_environment()

    # 加载配置
    config = load_config(args.config)

    try:
        # 数据预处理或加载
        if args.preprocess or not os.path.exists(args.processed_data_dir or config['data_paths']['processed_data_dir']):
            print("开始数据预处理...")
            preprocessor = ABIDEDataPreprocessor()

            # 更新路径（如果提供了新路径）
            if args.data_dir:
                preprocessor.timeseries_dir = os.path.join(args.data_dir, "aal/Outputs/cpac/filt_noglobal/rois_aal")
                preprocessor.fmri_data_dir = os.path.join(args.data_dir, "aal/Outputs/cpac/filt_noglobal/func_preproc")

            # 处理数据
            train_loader, val_loader, test_loader, tr_info = preprocessor.process_abide_data(save_to_disk=True)
            processed_data_dir = preprocessor.output_dir
        else:
            # 加载已处理的数据
            processed_data_dir = args.processed_data_dir or config['data_paths']['processed_data_dir']
            print(f"加载已处理的数据从: {processed_data_dir}")

            timeseries_data, labels, tr_values, processing_info = load_preprocessed_data(processed_data_dir)

            # 创建数据集
            dataset = BrainDataset(timeseries_data, labels, tr_values)

            print(f"数据集大小: {len(dataset)}")
            print(f"标签分布 - ASD: {sum(labels == 1)}, TC: {sum(labels == 0)}")

            # 创建数据加载器
            train_loader, val_loader, test_loader = create_dataloaders(
                dataset,
                batch_size=config['training']['batch_size'],
                train_ratio=config['data']['train_split'],
                val_ratio=config['data']['val_split'],
                test_ratio=config['data']['test_split'],
                num_workers=config['data']['num_workers']
            )

        # 更新配置中的脑区数量和时间点
        if 'processing_info' in locals():
            config['model']['actual_num_regions'] = processing_info['num_regions']
            config['model']['timeseries_length'] = processing_info['num_timepoints']
        else:
            # 默认值
            config['model']['actual_num_regions'] = 116
            config['model']['timeseries_length'] = 100

        # 创建模型
        print(f"\n创建模型，脑区数量: {config['model']['actual_num_regions']}")
        model = TRAdaptiveBrainNetwork(config)
        model = model.to(device)

        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"模型参数总数: {total_params:,}")
        print(f"可训练参数数: {trainable_params:,}")

        # 创建训练器配置
        trainer_config = TrainerConfig(
            batch_size=config['training']['batch_size'],
            epochs=config['training']['epochs'],
            learning_rate=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            optimizer=config['training']['optimizer'],
            scheduler=config['training']['scheduler'],
            patience=config['training']['patience'],
            min_delta=config['training']['min_delta'],
            grad_clip=config['training']['grad_clip'],
            device=str(device),
            mixed_precision=config['training']['mixed_precision'],
            mixup_enabled=config['augmentation']['mixup']['enabled'],
            mixup_alpha=config['augmentation']['mixup']['alpha'],
            num_classes=config['model']['num_classes']
        )

        # 创建训练器
        trainer = ModelTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=trainer_config,
            experiment_name="tr_adaptive_brain_network",
            log_dir="./logs"
        )

        # 开始训练
        print("\n开始训练...")
        results = trainer.train()

        # 输出最终结果
        print(f"\n=== 训练完成 ===")
        print(f"最佳验证指标: {results['best_metric']:.4f}")
        if 'test_metrics' in results and results['test_metrics']:
            print(f"测试集准确率: {results['test_metrics'].get('accuracy', 0):.4f}")
            print(f"测试集AUC: {results['test_metrics'].get('auc_roc', 0):.4f}")
        print(f"总训练时间: {results['training_time']:.2f} 秒")

    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()