import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
import json
from collections import defaultdict
from tqdm import tqdm
import gc

from utils.mixup import Mixup, ProgressiveMixup
from utils.metrics import MetricsCalculator, EarlyStopping


class TrainerConfig:
    """训练配置类"""

    def __init__(self,
                 batch_size: int = 32,
                 epochs: int = 100,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 optimizer: str = "adamw",
                 scheduler: str = "cosine",
                 warmup_epochs: int = 5,
                 patience: int = 15,
                 min_delta: float = 1e-4,
                 grad_clip: float = 1.0,
                 eval_interval: int = 1,
                 save_interval: int = 5,
                 log_interval: int = 50,
                 num_workers: int = 4,
                 device: str = "cuda:0",
                 mixed_precision: bool = True,
                 accumulation_steps: int = 1,
                 mixup_enabled: bool = True,
                 mixup_alpha: float = 0.2,
                 num_classes: int = 2):

        # 确保数值类型正确
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup_epochs = int(warmup_epochs)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.grad_clip = float(grad_clip)
        self.eval_interval = int(eval_interval)
        self.save_interval = int(save_interval)
        self.log_interval = int(log_interval)
        self.num_workers = int(num_workers)
        self.device = device
        self.mixed_precision = mixed_precision
        self.accumulation_steps = int(accumulation_steps)
        self.mixup_enabled = mixup_enabled
        self.mixup_alpha = float(mixup_alpha)
        self.num_classes = int(num_classes)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainerConfig':
        config = cls()
        for k, v in config_dict.items():
            if hasattr(config, k):
                setattr(config, k, v)
        return config


class GPUMemoryManager:
    """GPU内存管理器"""

    def __init__(self, device):
        self.device = device

    def clear_cache(self):
        """清理GPU缓存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def get_memory_info(self):
        """获取GPU内存信息"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device) / 1024 ** 3
            cached = torch.cuda.memory_reserved(self.device) / 1024 ** 3
            return allocated, cached
        return 0, 0


class TrainingLogger:
    """训练日志记录器"""

    def __init__(self, log_dir: str, experiment_name: str):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.log_file = os.path.join(log_dir, f"{experiment_name}_training.log")
        self.metrics_file = os.path.join(log_dir, f"{experiment_name}_metrics.json")
        self.results_file = os.path.join(log_dir, f"{experiment_name}_results.json")

        os.makedirs(log_dir, exist_ok=True)

        # 设置简洁的日志格式
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            datefmt='%H:%M:%S',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(experiment_name)

        self.metrics_history = defaultdict(list)
        self.training_results = {}

    def log_batch_info(self, epoch: int, batch_idx: int, num_batches: int,
                       loss: float, lr: float, mixup_info: str = ""):
        """记录批次训练信息 - 简洁版本"""
        progress = f"[{batch_idx:3d}/{num_batches:3d}]"
        loss_info = f"Loss: {loss:.4f}"
        lr_info = f"LR: {lr:.2e}"

        message = f"Epoch {epoch:3d} {progress} {loss_info} {lr_info}"
        if mixup_info:
            message += f" {mixup_info}"

        self.logger.info(message)

    def log_epoch_summary(self, epoch: int, train_loss: float, val_metrics: Dict[str, float],
                          test_metrics: Dict[str, float], best_metric: float):
        """记录epoch总结信息"""
        self.logger.info("=" * 80)
        self.logger.info(f"Epoch {epoch:3d} 总结:")
        self.logger.info(f"  训练损失: {train_loss:.4f}")
        self.logger.info(f"  验证集 - 准确率: {val_metrics.get('accuracy', 0):.4f}, "
                         f"AUC: {val_metrics.get('auc_roc', 0):.4f}, "
                         f"F1: {val_metrics.get('f1', 0):.4f}")
        if test_metrics:
            self.logger.info(f"  测试集 - 准确率: {test_metrics.get('accuracy', 0):.4f}, "
                             f"AUC: {test_metrics.get('auc_roc', 0):.4f}, "
                             f"F1: {test_metrics.get('f1', 0):.4f}")
        self.logger.info(f"  最佳指标: {best_metric:.4f}")
        self.logger.info("=" * 80)

    def log_metrics(self, metrics: Dict[str, float], epoch: int, phase: str):
        """记录指标到历史"""
        timestamp = time.time()

        for metric_name, value in metrics.items():
            full_metric_name = f"{phase}_{metric_name}"
            self.metrics_history[full_metric_name].append({
                'epoch': epoch,
                'value': float(value),
                'timestamp': timestamp
            })

        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

    def save_training_results(self, results: Dict[str, Any]):
        """保存训练结果"""
        self.training_results.update(results)

        # 保存详细结果
        with open(self.results_file, 'w') as f:
            json.dump(self.training_results, f, indent=2, cls=NumpyEncoder)

        self.logger.info(f"训练结果已保存到: {self.results_file}")


class NumpyEncoder(json.JSONEncoder):
    """用于处理numpy数组的JSON编码器"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class ModelTrainer:
    """主训练器类 - 修复版本"""

    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: Optional[DataLoader] = None,
                 config: Optional[TrainerConfig] = None,
                 experiment_name: str = "experiment",
                 log_dir: str = "./logs"):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.config = config or TrainerConfig()
        self.experiment_name = experiment_name
        self.log_dir = log_dir

        # 设置设备
        self.device = torch.device(self.config.device)

        # GPU内存管理器
        self.gpu_manager = GPUMemoryManager(self.device)

        # 将模型移动到设备
        self.model = self.model.to(self.device)

        # 启用cudnn自动优化
        torch.backends.cudnn.benchmark = True

        self.logger = TrainingLogger(log_dir, experiment_name)
        self.optimizer = self._setup_optimizer()
        self.criterion = self._setup_criterion()
        self.early_stopping = EarlyStopping(
            patience=self.config.patience,
            min_delta=self.config.min_delta
        )

        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler() if self.config.mixed_precision and self.device.type == 'cuda' else None

        # Mixup数据增强
        if self.config.mixup_enabled:
            self.logger.logger.info(f"启用渐进式Mixup，基础alpha: {self.config.mixup_alpha}")
            self.mixup = ProgressiveMixup(
                base_alpha=self.config.mixup_alpha,
                num_epochs=self.config.epochs,
                apply_probability=0.8,
                num_classes=self.config.num_classes
            )
        else:
            self.mixup = None

        # 训练状态
        self.current_epoch = 0
        self.best_metric = float('-inf')
        self.train_losses = []
        self.val_metrics_history = []
        self.test_metrics_history = []

        self.logger.logger.info(f"训练设备: {self.device}")
        self.logger.logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        self.logger.logger.info(f"训练集大小: {len(train_loader.dataset)}")
        self.logger.logger.info(f"验证集大小: {len(val_loader.dataset)}")
        if test_loader:
            self.logger.logger.info(f"测试集大小: {len(test_loader.dataset)}")

        # 显示GPU信息
        if torch.cuda.is_available():
            self.logger.logger.info(
                f"GPU内存: {torch.cuda.get_device_properties(self.device).total_memory / 1024 ** 3:.1f} GB")

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        lr = float(self.config.learning_rate)
        weight_decay = float(self.config.weight_decay)

        if self.config.optimizer.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999)
            )
        elif self.config.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.9
            )

        return optimizer

    def _setup_criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss()

    def _setup_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        if self.config.scheduler.lower() == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.epochs
            )
        elif self.config.scheduler.lower() == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', patience=10, factor=0.5
            )

        return scheduler

    def _warmup_lr_scheduler(self, epoch: int, batch_idx: int) -> float:
        if epoch < self.config.warmup_epochs:
            total_batches = len(self.train_loader)
            current_step = epoch * total_batches + batch_idx
            warmup_steps = self.config.warmup_epochs * total_batches

            lr_scale = min(1.0, float(current_step) / float(warmup_steps))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.config.learning_rate * lr_scale

            return lr_scale
        return 1.0

    def train_epoch(self) -> float:
        """训练一个epoch - 修复版本"""
        # 更新Mixup的当前epoch
        if self.mixup and hasattr(self.mixup, 'set_epoch'):
            self.mixup.set_epoch(self.current_epoch)

        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)

        # 使用tqdm进度条
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}', leave=False)

        for batch_idx, batch in enumerate(pbar):
            # 清理GPU缓存（定期）
            if batch_idx % 50 == 0:
                self.gpu_manager.clear_cache()

            # 将批次数据移动到设备
            batch = self._move_batch_to_device(batch)

            # 预热学习率
            self._warmup_lr_scheduler(self.current_epoch, batch_idx)

            # 应用Mixup数据增强
            if self.mixup and self.model.training:
                batch = self.mixup(batch)

            # 梯度累积步骤
            if (batch_idx + 1) % self.config.accumulation_steps != 0:
                # 在前N-1个累积步骤中，不更新参数
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                        if 'tr_values' in batch:
                            outputs = self.model(batch['features'], batch['tr_values'])
                        else:
                            outputs = self.model(batch['features'])

                        if self.mixup and 'mixup_lambda' in batch:
                            if outputs.shape[1] > 1:
                                loss = F.cross_entropy(outputs, batch['labels'])
                            else:
                                loss = F.binary_cross_entropy_with_logits(
                                    outputs.squeeze(), batch['labels']
                                )
                        else:
                            if outputs.shape[1] > 1:
                                labels = batch['labels'].long()
                                if labels.dim() > 1 and labels.size(1) > 1:
                                    labels = torch.argmax(labels, dim=1)
                                loss = self.criterion(outputs, labels)
                            else:
                                loss = F.binary_cross_entropy_with_logits(
                                    outputs.squeeze(), batch['labels']
                                )

                # 缩放损失并反向传播
                loss = loss / self.config.accumulation_steps
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                total_loss += loss.item() * self.config.accumulation_steps
                continue

            # 混合精度前向传播
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                # 模型前向传播
                if 'tr_values' in batch:
                    outputs = self.model(batch['features'], batch['tr_values'])
                else:
                    outputs = self.model(batch['features'])

                # 计算损失
                if self.mixup and 'mixup_lambda' in batch:
                    # Mixup使用软标签
                    if outputs.shape[1] > 1:  # 多分类
                        loss = F.cross_entropy(outputs, batch['labels'])
                    else:  # 二分类
                        loss = F.binary_cross_entropy_with_logits(
                            outputs.squeeze(), batch['labels']
                        )
                else:
                    # 普通训练
                    if outputs.shape[1] > 1:  # 多分类
                        labels = batch['labels'].long()
                        if labels.dim() > 1 and labels.size(1) > 1:
                            labels = torch.argmax(labels, dim=1)
                        loss = self.criterion(outputs, labels)
                    else:  # 二分类
                        loss = F.binary_cross_entropy_with_logits(
                            outputs.squeeze(), batch['labels']
                        )

            # 梯度累积：在最后一个步骤中执行优化
            loss = loss / self.config.accumulation_steps

            # 反向传播
            if self.scaler:
                self.scaler.scale(loss).backward()

                # 梯度裁剪
                if self.config.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip
                    )

                # 更新参数
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()

                # 梯度裁剪
                if self.config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip
                    )

                # 更新参数
                self.optimizer.step()

            # 清空梯度
            self.optimizer.zero_grad()

            total_loss += loss.item() * self.config.accumulation_steps

            # 更新进度条
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{loss.item() * self.config.accumulation_steps:.4f}',
                'lr': f'{current_lr:.2e}'
            })

            # 记录批次信息（简化版）
            if batch_idx % self.config.log_interval == 0:
                mixup_info = ""
                if 'mixup_lambda' in batch:
                    mixup_info = f"Mixup λ: {batch.get('mixup_lambda', 0.0):.3f}"

                self.logger.log_batch_info(
                    self.current_epoch, batch_idx, num_batches,
                    loss.item() * self.config.accumulation_steps, current_lr, mixup_info
                )

        pbar.close()
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)

        # 清理GPU缓存
        self.gpu_manager.clear_cache()

        return avg_loss

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader, phase: str = "val") -> Dict[str, float]:
        """评估模型 - 修复版本"""
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0

        # 使用no_grad上下文管理器减少内存使用
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                # 定期清理缓存
                if batch_idx % 20 == 0:
                    self.gpu_manager.clear_cache()

                batch = self._move_batch_to_device(batch)

                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    if 'tr_values' in batch:
                        outputs = self.model(batch['features'], batch['tr_values'])
                    else:
                        outputs = self.model(batch['features'])

                    # 计算损失
                    if outputs.shape[1] > 1:  # 多分类
                        labels = batch['labels'].long()
                        if labels.dim() > 1 and labels.size(1) > 1:
                            labels = torch.argmax(labels, dim=1)
                        loss = self.criterion(outputs, labels)
                        preds = F.softmax(outputs, dim=1)
                    else:  # 二分类
                        loss = F.binary_cross_entropy_with_logits(
                            outputs.squeeze(), batch['labels']
                        )
                        preds = torch.sigmoid(outputs)

                total_loss += loss.item()

                # 立即将数据移回CPU以释放GPU内存
                all_preds.append(preds.cpu().numpy())
                all_labels.append(batch['labels'].cpu().numpy())

        # 连接所有批次
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # 计算指标
        metrics = MetricsCalculator.calculate_classification_metrics(
            all_labels,
            (all_preds > 0.5).astype(int) if all_preds.shape[1] == 1 else all_preds.argmax(axis=1),
            all_preds
        )
        metrics['loss'] = total_loss / len(data_loader)

        # 记录指标
        self.logger.log_metrics(metrics, self.current_epoch, phase)

        # 清理GPU缓存
        self.gpu_manager.clear_cache()

        return metrics

    def _move_batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """将批次张量移动到设备 - 修复版本"""
        result = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                # 修复：检查张量是否已经在目标设备上
                if v.device != self.device:
                    # 使用non_blocking异步传输
                    result[k] = v.to(self.device, non_blocking=True)
                else:
                    result[k] = v
            else:
                result[k] = v
        return result

    def save_checkpoint(self, filename: str, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'train_losses': self.train_losses,
            'val_metrics_history': self.val_metrics_history,
            'test_metrics_history': self.test_metrics_history,
            'config': self.config.to_dict()
        }

        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        checkpoint_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        filepath = os.path.join(checkpoint_dir, filename)
        torch.save(checkpoint, filepath)

        if is_best:
            best_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.best_metric = checkpoint['best_metric']
        self.train_losses = checkpoint['train_losses']
        self.val_metrics_history = checkpoint.get('val_metrics_history', [])
        self.test_metrics_history = checkpoint.get('test_metrics_history', [])

        self.logger.logger.info(f"从epoch {self.current_epoch}加载检查点")

    def train(self) -> Dict[str, Any]:
        """主训练循环 - 修复版本"""
        self.logger.logger.info("开始训练...")
        start_time = time.time()

        scheduler = self._setup_scheduler()

        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch

            # 训练一个epoch
            train_loss = self.train_epoch()

            if epoch % self.config.eval_interval == 0:
                # 验证集评估
                val_metrics = self.evaluate(self.val_loader, "val")
                self.val_metrics_history.append(val_metrics)

                # 测试集评估
                if self.test_loader is not None:
                    test_metrics = self.evaluate(self.test_loader, "test")
                    self.test_metrics_history.append(test_metrics)
                else:
                    test_metrics = {}

                current_metric = val_metrics.get('f1', val_metrics.get('accuracy', -val_metrics['loss']))
                if current_metric > self.best_metric:
                    self.best_metric = current_metric
                    self.save_checkpoint("best_model.pth", is_best=True)

                # 早停检查
                self.early_stopping(val_metrics['loss'], self.model)
                if self.early_stopping.early_stop:
                    self.logger.logger.info(f"在epoch {epoch}早停")
                    break

            # 学习率调度
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()

            # 保存检查点
            if epoch % self.config.save_interval == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pth")

            # 输出epoch总结
            self.logger.log_epoch_summary(
                epoch, train_loss, val_metrics, test_metrics, self.best_metric
            )

            # 每5个epoch清理一次GPU缓存
            if epoch % 5 == 0:
                self.gpu_manager.clear_cache()

        training_time = time.time() - start_time

        # 加载最佳模型进行最终测试
        best_checkpoint_path = os.path.join(self.log_dir, "checkpoints", "best_model.pth")
        if os.path.exists(best_checkpoint_path):
            self.load_checkpoint(best_checkpoint_path)
            self.logger.logger.info("加载最佳模型进行最终测试...")

            # 最终测试集评估
            if self.test_loader is not None:
                final_test_metrics = self.evaluate(self.test_loader, "test")
            else:
                final_test_metrics = {}

        # 准备最终结果
        results = {
            'best_metric': self.best_metric,
            'train_losses': self.train_losses,
            'val_metrics': self.val_metrics_history[-1] if self.val_metrics_history else {},
            'test_metrics': final_test_metrics if 'final_test_metrics' in locals() else {},
            'training_time': training_time,
            'total_epochs': self.current_epoch + 1,
            'config': self.config.to_dict()
        }

        # 保存训练结果
        self.logger.save_training_results(results)

        self.logger.logger.info(f"训练完成，用时 {training_time:.2f} 秒")
        self.logger.logger.info(f"最佳指标: {self.best_metric:.4f}")

        return results