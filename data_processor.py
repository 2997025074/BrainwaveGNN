import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
import re
import torch
import torch.utils.data as utils
import os
import pickle


class ABIDEDataPreprocessor:
    def __init__(self):
        # 直接设置参数
        self.timeseries_dir = "C:/Users/29970/Desktop/data/aal/Outputs/cpac/filt_noglobal/rois_aal"  # 1D文件目录
        self.phenotypic_path = "C:/Users/29970/Desktop/abide-master/Phenotypic_V1_0b_preprocessed1.csv"  # 表型文件路径
        self.fmri_data_dir = "C:/Users/29970/Desktop/data/aal/Outputs/cpac/filt_noglobal/func_preproc"  # fMRI数据目录
        self.output_dir = "data/processed"  # 处理后的数据保存目录
        self.default_tr = 2.0  # 默认TR值（秒）
        self.standardize_timeseries = True  # 是否标准化时间序列
        self.random_state = 42  # 随机种子

        # 数据集划分比例
        self.train_ratio = 0.7
        self.val_ratio = 0.15
        self.test_ratio = 0.15

        # 目标时间序列长度
        self.target_timepoints = 100

        # 确保目录存在
        os.makedirs(self.timeseries_dir, exist_ok=True)
        os.makedirs(self.fmri_data_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def create_torch_dataloader(self, timeseries, tr_values, labels, shuffle=True):
        """创建PyTorch数据加载器 - 修复版本"""
        dataset = utils.TensorDataset(timeseries, tr_values, labels)

        dataloader = utils.DataLoader(
            dataset,
            batch_size=16,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=torch.cuda.is_available(),  # 修复：只在有GPU时启用
            persistent_workers=True,
            drop_last=shuffle
        )

        # 包装数据加载器以返回字典格式
        return DictDataLoader(dataloader)

    def unify_timeseries_length(self, timeseries_data):
        """统一时间序列长度为100个时间点 - 使用线性插值"""
        n_samples, n_regions, n_timepoints = timeseries_data.shape
        print(f"原始时间序列形状: {timeseries_data.shape}")
        print(f"时间点范围: {n_timepoints} -> 目标: {self.target_timepoints}")

        if n_timepoints == self.target_timepoints:
            print(f"时间序列长度已经是目标长度 {self.target_timepoints}，无需调整")
            return timeseries_data

        # 创建新的数组来存储统一长度的数据
        unified_data = np.zeros((n_samples, n_regions, self.target_timepoints))

        for i in range(n_samples):
            for j in range(n_regions):
                # 获取当前时间序列
                ts = timeseries_data[i, j, :]

                # 创建插值坐标
                # 原始时间点 (0到1的归一化坐标)
                x_original = np.linspace(0, 1, n_timepoints)
                # 目标时间点
                x_target = np.linspace(0, 1, self.target_timepoints)

                # 使用numpy的线性插值
                unified_ts = np.interp(x_target, x_original, ts)
                unified_data[i, j, :] = unified_ts

        print(f"统一后的时间序列形状: {unified_data.shape}")
        return unified_data

    def process_abide_data(self, save_to_disk=True):
        """修正后的预处理流程 - 确保调用统一时间序列长度"""
        print("=== ABIDE数据集预处理开始 ===")

        # 1. 加载时间序列数据和提取TR信息
        print("1. 加载时间序列和提取TR信息...")
        timeseries_data, labels, sites, tr_values = self.load_timeseries_with_tr()

        # 修复：确保标签是整数类型
        labels = labels.astype(np.int64)
        print(f"标签数据类型: {labels.dtype}, 唯一值: {np.unique(labels)}")

        # ✅ 新增：统一时间序列长度为100个时间点
        print("2. 统一时间序列长度...")
        timeseries_data = self.unify_timeseries_length(timeseries_data)

        # 3. 标准化时间序列 (z-score标准化)
        if self.standardize_timeseries:
            print("3. 标准化时间序列...")
            timeseries_data = self.standardize_timeseries_data(timeseries_data)

        # 4. 转换为PyTorch tensor
        timeseries_tensor = torch.from_numpy(timeseries_data).float()
        tr_tensor = torch.from_numpy(tr_values).float()
        labels_tensor = torch.from_numpy(labels).long()

        # 5. 创建数据加载器 - 使用随机划分
        print("4. 创建数据加载器...")
        train_loader, val_loader, test_loader = self.create_dataloaders(
            timeseries_tensor, tr_tensor, labels_tensor
        )

        # 6. 准备TR信息
        tr_info = {
            'tr_values': tr_values,
            'sites': sites,
            'num_regions': timeseries_data.shape[1],
            'num_timepoints': timeseries_data.shape[2]
        }

        # 7. 打印统计信息
        self.print_statistics(timeseries_tensor, labels_tensor, tr_values, sites)

        # 8. 保存处理后的数据到磁盘
        if save_to_disk:
            print("5. 保存处理后的数据到磁盘...")
            self.save_processed_data(timeseries_data, labels, sites, tr_values)

        print("=== 预处理完成 ===")
        return train_loader, val_loader, test_loader, tr_info

    def create_dataloaders(self, timeseries, tr_values, labels):
        """创建数据加载器 - 使用随机划分"""
        length = len(timeseries)
        train_size = int(length * self.train_ratio)
        val_size = int(length * self.val_ratio)
        test_size = length - train_size - val_size

        # 使用随机划分
        train_loader, val_loader, test_loader = self.create_random_split(
            timeseries, tr_values, labels, train_size, val_size, test_size
        )

        return train_loader, val_loader, test_loader

    def create_random_split(self, timeseries, tr_values, labels, train_size, val_size, test_size):
        """创建随机划分的数据加载器"""
        dataset = utils.TensorDataset(timeseries, tr_values, labels)
        train_dataset, val_dataset, test_dataset = utils.random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.random_state)
        )

        train_loader = self.create_torch_dataloader_from_dataset(train_dataset, shuffle=True)
        val_loader = self.create_torch_dataloader_from_dataset(val_dataset, shuffle=False)
        test_loader = self.create_torch_dataloader_from_dataset(test_dataset, shuffle=False)

        print(f"随机划分 - 训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")

        return train_loader, val_loader, test_loader

    def create_torch_dataloader_from_dataset(self, dataset, shuffle=True):
        """从数据集创建PyTorch数据加载器"""
        dataloader = utils.DataLoader(
            dataset,
            batch_size=16,
            shuffle=shuffle,
            num_workers=4,
            drop_last=shuffle
        )

        # 包装数据加载器以返回字典格式
        return DictDataLoader(dataloader)

    def save_processed_data(self, timeseries_data, labels, sites, tr_values):
        """保存处理后的数据到磁盘"""
        # 修复：确保标签是整数类型
        labels = labels.astype(np.int64)

        # 保存numpy数组
        np.save(os.path.join(self.output_dir, "timeseries_data.npy"), timeseries_data)
        np.save(os.path.join(self.output_dir, "labels.npy"), labels)
        np.save(os.path.join(self.output_dir, "tr_values.npy"), tr_values)

        # 保存站点信息（使用pickle，因为可能包含字符串）
        with open(os.path.join(self.output_dir, "sites.pkl"), 'wb') as f:
            pickle.dump(sites, f)

        # 保存处理参数信息
        processing_info = {
            'num_samples': timeseries_data.shape[0],
            'num_regions': timeseries_data.shape[1],
            'num_timepoints': timeseries_data.shape[2],
            'standardize_timeseries': self.standardize_timeseries,
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
            'target_timepoints': self.target_timepoints
        }

        with open(os.path.join(self.output_dir, "processing_info.pkl"), 'wb') as f:
            pickle.dump(processing_info, f)

        print(f"处理后的数据已保存到: {self.output_dir}")
        print(f"  时间序列: {timeseries_data.shape}")
        print(f"  标签: {labels.shape}")
        print(f"  TR值: {tr_values.shape}")

    def load_processed_data(self):
        """从磁盘加载已处理的数据"""
        print("=== 加载已处理的ABIDE数据 ===")

        # 检查文件是否存在
        required_files = [
            "timeseries_data.npy", "labels.npy",
            "tr_values.npy", "sites.pkl", "processing_info.pkl"
        ]

        for file in required_files:
            if not os.path.exists(os.path.join(self.output_dir, file)):
                raise FileNotFoundError(f"找不到处理后的数据文件: {file}")

        # 加载数据
        timeseries_data = np.load(os.path.join(self.output_dir, "timeseries_data.npy"))
        labels = np.load(os.path.join(self.output_dir, "labels.npy"))
        tr_values = np.load(os.path.join(self.output_dir, "tr_values.npy"))

        with open(os.path.join(self.output_dir, "sites.pkl"), 'rb') as f:
            sites = pickle.load(f)

        with open(os.path.join(self.output_dir, "processing_info.pkl"), 'rb') as f:
            processing_info = pickle.load(f)

        # 转换为PyTorch tensor
        timeseries_tensor = torch.from_numpy(timeseries_data).float()
        tr_tensor = torch.from_numpy(tr_values).float()
        labels_tensor = torch.from_numpy(labels).long()

        # 创建数据加载器
        train_loader, val_loader, test_loader = self.create_dataloaders(
            timeseries_tensor, tr_tensor, labels_tensor
        )

        # 准备TR信息
        tr_info = {
            'tr_values': tr_values,
            'sites': sites,
            'num_regions': timeseries_data.shape[1],
            'num_timepoints': timeseries_data.shape[2]
        }

        # 打印统计信息
        self.print_statistics(timeseries_tensor, labels_tensor, tr_values, sites)

        print("=== 数据加载完成 ===")
        return train_loader, val_loader, test_loader, tr_info

    def create_torch_dataloader(self, timeseries, tr_values, labels, shuffle=True):
        """创建PyTorch数据加载器 - 返回字典格式以匹配Mixup"""
        dataset = utils.TensorDataset(timeseries, tr_values, labels)

        dataloader = utils.DataLoader(
            dataset,
            batch_size=16,
            shuffle=shuffle,
            num_workers=4,
            drop_last=shuffle
        )

        # 包装数据加载器以返回字典格式
        return DictDataLoader(dataloader)

    def print_statistics(self, timeseries, labels, tr_values, sites):
        """打印数据统计信息"""
        print("\n=== 数据统计 ===")
        print(f"时间序列维度: {timeseries.shape}")
        print(f"标签分布 - ASD: {sum(labels == 1)}, TC: {sum(labels == 0)}")
        print(f"TR值范围: {min(tr_values):.3f}s - {max(tr_values):.3f}s")
        print(f"平均TR值: {np.mean(tr_values):.3f}s")
        print(f"唯一站点数: {len(set(sites))}")

        # 打印各站点的样本分布
        site_counts = {}
        for site in sites:
            site_counts[site] = site_counts.get(site, 0) + 1

        print("\n各站点样本分布:")
        for site, count in sorted(site_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {site}: {count} 个样本")

    # 以下方法保持不变
    def load_timeseries_with_tr(self):
        """加载时间序列数据并提取TR信息"""
        if not Path(self.timeseries_dir).exists():
            raise FileNotFoundError(f"时间序列目录不存在: {self.timeseries_dir}")

        if not Path(self.phenotypic_path).exists():
            raise FileNotFoundError(f"表型文件不存在: {self.phenotypic_path}")

        # 加载表型数据
        phenotypic_df = pd.read_csv(self.phenotypic_path)

        # 查找所有1D文件
        one_d_files = list(Path(self.timeseries_dir).glob("*.1D"))
        print(f"找到 {len(one_d_files)} 个1D文件")

        timeseries_list = []
        label_list = []
        site_list = []
        tr_list = []
        processed_count = 0

        for one_d_file in one_d_files:
            # 从文件名提取subject ID
            subject_id = self.extract_subject_id(one_d_file.name)

            if not subject_id:
                print(f"无法从文件名提取subject ID: {one_d_file.name}")
                continue

            # 在表型数据中查找
            subject_data = self.find_subject_in_phenotypic(subject_id, phenotypic_df)

            if subject_data is None:
                print(f"在表型数据中未找到subject: {subject_id}")
                continue

            try:
                # 加载时间序列
                ts_data = np.loadtxt(one_d_file)
                if ts_data.size == 0:
                    continue

                # 确保正确的形状
                ts_data = self.ensure_correct_shape(ts_data)

                # 提取诊断标签
                label = self.extract_diagnosis_label(subject_data)
                if label is None:
                    continue

                # 提取站点信息
                site = self.extract_site_info(subject_data, subject_id)

                # 提取TR值
                tr_value = self.extract_tr_from_fmri(subject_id)

                timeseries_list.append(ts_data)
                label_list.append(label)
                site_list.append(site)
                tr_list.append(tr_value)

                processed_count += 1

                if processed_count % 100 == 0:
                    print(f"已处理 {processed_count} 个样本...")

            except Exception as e:
                print(f"处理文件 {one_d_file} 时出错: {e}")
                continue

        if not timeseries_list:
            raise ValueError("没有找到有效的时间序列数据!")

        print(f"成功处理 {processed_count} 个样本")

        # 统一形状并返回
        timeseries_array = self.standardize_timeseries_shape(timeseries_list)
        labels_array = np.array(label_list)
        tr_array = np.array(tr_list)

        return timeseries_array, labels_array, site_list, tr_array

    def standardize_timeseries_shape(self, timeseries_list):
        """统一时间序列形状 - 临时方法，后续会被unify_timeseries_length替换"""
        max_regions = max(ts.shape[0] for ts in timeseries_list)
        max_timepoints = max(ts.shape[1] for ts in timeseries_list)

        print(f"最大脑区数: {max_regions}, 最大时间点数: {max_timepoints}")

        standardized = []
        for ts in timeseries_list:
            if ts.shape[0] < max_regions or ts.shape[1] < max_timepoints:
                # 填充到最大维度
                padded_ts = np.zeros((max_regions, max_timepoints))
                regions_to_copy = min(ts.shape[0], max_regions)
                timepoints_to_copy = min(ts.shape[1], max_timepoints)
                padded_ts[:regions_to_copy, :timepoints_to_copy] = ts[:regions_to_copy, :timepoints_to_copy]
                standardized.append(padded_ts)
            else:
                standardized.append(ts)

        return np.stack(standardized, axis=0)

    def extract_subject_id(self, filename):
        """从文件名提取subject ID"""
        # 针对 Caltech_0051456_rois_aal.1D 格式
        patterns = [
            r'([A-Za-z]+)_(\d+)_rois_[a-zA-Z]+\.1D',  # Caltech_0051456_rois_aal.1D
            r'([A-Za-z]+)_(\d+)_.*\.1D',
            r'sub-(\d+)_.*\.1D',
            r'(\d+)\.1D'
        ]

        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                if len(match.groups()) >= 2:
                    return match.group(2)  # 返回数字部分
                else:
                    return match.group(1)
        return None

    def find_subject_in_phenotypic(self, subject_id, phenotypic_df):
        """在表型数据中查找subject"""
        # 将SUB_ID列转换为字符串进行比较
        phenotypic_df = phenotypic_df.copy()
        if 'SUB_ID' in phenotypic_df.columns:
            phenotypic_df['SUB_ID'] = phenotypic_df['SUB_ID'].astype(str)
            match = phenotypic_df[phenotypic_df['SUB_ID'] == subject_id]
            if not match.empty:
                return match.iloc[0]

        # 在文件名列中查找
        for col in phenotypic_df.columns:
            if 'file' in col.lower() or 'filename' in col.lower():
                for idx, row in phenotypic_df.iterrows():
                    if subject_id in str(row[col]):
                        return row

        return None

    def ensure_correct_shape(self, ts_data):
        """确保时间序列形状正确"""
        if ts_data.ndim == 1:
            # 单脑区，转换为 (1, timepoints)
            ts_data = ts_data.reshape(1, -1)
        elif ts_data.ndim == 2 and ts_data.shape[0] > ts_data.shape[1]:
            # 需要转置
            ts_data = ts_data.T

        return ts_data

    def extract_diagnosis_label(self, subject_data):
        """提取诊断标签: ASD=1, TC=0"""
        if 'DX_GROUP' in subject_data and pd.notna(subject_data['DX_GROUP']):
            return 1 if subject_data['DX_GROUP'] == 1 else 0
        return None

    def extract_site_info(self, subject_data, subject_id):
        """提取站点信息"""
        if 'SITE_ID' in subject_data and pd.notna(subject_data['SITE_ID']):
            return str(subject_data['SITE_ID'])
        else:
            # 从文件名推断
            if 'Caltech' in str(subject_id):
                return 'Caltech'
            return 'UNKNOWN'

    def extract_tr_from_fmri(self, subject_id):
        """从fMRI数据提取TR值"""
        # 构建可能的文件名模式
        patterns = [
            f"*{subject_id}*.nii.gz",
            f"*{subject_id}*.nii",
            f"*{subject_id}_func_preproc.nii.gz",  # Caltech_0051478_func_preproc.nii.gz
            f"*{subject_id}*preproc*.nii.gz",
        ]

        fmri_dir = Path(self.fmri_data_dir)
        for pattern in patterns:
            fmri_files = list(fmri_dir.rglob(pattern))
            if fmri_files:
                try:
                    img = nib.load(str(fmri_files[0]))
                    tr = float(img.header['pixdim'][4])
                    print(f"从 {fmri_files[0].name} 提取TR值: {tr:.3f}s")
                    return tr
                except Exception as e:
                    print(f"读取 {fmri_files[0]} 的TR值时出错: {e}")
                    continue

        # 使用默认TR值
        print(f"警告: 对subject {subject_id} 使用默认TR值 {self.default_tr}s")
        return self.default_tr

    def standardize_timeseries_data(self, timeseries_data):
        """标准化时间序列 - z-score标准化"""
        mean = np.mean(timeseries_data, axis=(0, 2), keepdims=True)
        std = np.std(timeseries_data, axis=(0, 2), keepdims=True)
        std[std == 0] = 1  # 避免除零

        return (timeseries_data - mean) / std


class DictDataLoader:
    """包装数据加载器以返回字典格式"""

    def __init__(self, dataloader):
        self.dataloader = dataloader

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        for batch in self.dataloader:
            if len(batch) == 3:  # timeseries, tr_values, labels
                timeseries_batch, tr_batch, labels_batch = batch
                yield {
                    'features': timeseries_batch,
                    'tr_values': tr_batch,
                    'labels': labels_batch
                }
            else:
                # 如果batch不是3个元素，可能是其他格式
                yield batch


# 使用示例
if __name__ == "__main__":
    # 创建预处理器
    preprocessor = ABIDEDataPreprocessor()

    # 处理数据并保存到磁盘
    train_loader, val_loader, test_loader, tr_info = preprocessor.process_abide_data(save_to_disk=True)

    # 打印数据加载器信息
    print(f"\n训练集批次数量: {len(train_loader)}")
    print(f"验证集批次数量: {len(val_loader)}")
    print(f"测试集批次数量: {len(test_loader)}")

    # 检查一个批次的数据
    for batch_idx, batch in enumerate(train_loader):
        print(f"\n第一个批次数据形状:")
        print(f"时间序列: {batch['features'].shape}")
        print(f"TR值: {batch['tr_values'].shape}")
        print(f"标签: {batch['labels'].shape}")
        break