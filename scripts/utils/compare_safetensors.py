#!/usr/bin/env python3
"""
SafeTensor 比较工具
用于比较两个 safetensor 文件的差异、统计信息和相似度
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from safetensors import safe_open
from sklearn.metrics.pairwise import cosine_similarity


class SafeTensorComparator:
    """SafeTensor 文件比较器"""

    def __init__(self, file1_path: str, file2_path: str, output_dir: str = "./reports"):
        """
        初始化比较器

        Args:
            file1_path: 第一个 safetensor 文件路径
            file2_path: 第二个 safetensor 文件路径
            output_dir: 报告输出目录
        """
        self.file1_path = Path(file1_path)
        self.file2_path = Path(file2_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 存储加载的数据
        self.tensors1 = {}
        self.tensors2 = {}
        self.metadata1 = {}
        self.metadata2 = {}

        # 比较结果
        self.comparison_results = {}

    def load_safetensors(self) -> None:
        """加载两个 safetensor 文件"""
        print(f"加载 SafeTensor 文件: {self.file1_path}")
        with safe_open(self.file1_path, framework="pt", device="cpu") as f:  # type: ignore
            self.metadata1 = f.metadata() if hasattr(f, "metadata") else {}
            for key in f.keys():
                self.tensors1[key] = f.get_tensor(key)

        print(f"加载 SafeTensor 文件: {self.file2_path}")
        with safe_open(self.file2_path, framework="pt", device="cpu") as f:  # type: ignore
            self.metadata2 = f.metadata() if hasattr(f, "metadata") else {}
            for key in f.keys():
                self.tensors2[key] = f.get_tensor(key)

    def get_file_basic_info(self, file_path: Path, tensors: Dict) -> Dict:
        """获取文件基本信息"""
        file_size = file_path.stat().st_size
        total_params = sum(tensor.numel() for tensor in tensors.values())
        total_memory = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())

        dtype_counts = {}
        shape_stats = []

        for tensor in tensors.values():
            dtype = str(tensor.dtype)
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
            shape_stats.append(tensor.shape)

        return {
            "file_path": str(file_path),
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "num_tensors": len(tensors),
            "total_parameters": total_params,
            "total_memory_bytes": total_memory,
            "total_memory_mb": round(total_memory / (1024 * 1024), 2),
            "dtype_distribution": dtype_counts,
            "tensor_shapes": [list(shape) for shape in shape_stats],
        }

    def compare_keys(self) -> Dict:
        """比较两个文件的 key"""
        keys1 = set(self.tensors1.keys())
        keys2 = set(self.tensors2.keys())

        common_keys = keys1 & keys2
        only_in_file1 = keys1 - keys2
        only_in_file2 = keys2 - keys1

        return {
            "total_keys_file1": len(keys1),
            "total_keys_file2": len(keys2),
            "common_keys": sorted(common_keys),
            "common_keys_count": len(common_keys),
            "only_in_file1": sorted(only_in_file1),
            "only_in_file1_count": len(only_in_file1),
            "only_in_file2": sorted(only_in_file2),
            "only_in_file2_count": len(only_in_file2),
            "key_overlap_ratio": len(common_keys) / max(len(keys1), len(keys2))
            if max(len(keys1), len(keys2)) > 0
            else 0,
        }

    def calculate_tensor_similarity(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> Dict:
        """计算两个 tensor 的相似度和统计信息"""
        # 基本属性比较
        same_shape = tensor1.shape == tensor2.shape
        same_dtype = tensor1.dtype == tensor2.dtype

        if not same_shape:
            return {
                "same_shape": False,
                "same_dtype": same_dtype,
                "shape1": list(tensor1.shape),
                "shape2": list(tensor2.shape),
                "error": "形状不匹配，无法计算相似度",
            }

        # 转换为 numpy 进行计算
        arr1 = tensor1.detach().cpu().float().numpy().flatten()
        arr2 = tensor2.detach().cpu().float().numpy().flatten()

        # 基本统计
        exactly_equal = torch.equal(tensor1, tensor2)

        # 数值比较
        abs_diff = np.abs(arr1 - arr2)
        rel_diff = np.abs(arr1 - arr2) / (np.abs(arr1) + 1e-8)

        # 相似度计算
        cosine_sim = cosine_similarity(arr1.reshape(1, -1), arr2.reshape(1, -1))[0][0]
        pearson_corr = np.corrcoef(arr1, arr2)[0, 1] if len(arr1) > 1 else 1.0

        # MSE 和 RMSE
        mse = np.mean((arr1 - arr2) ** 2)
        rmse = np.sqrt(mse)

        # 变化量分析
        delta = arr2 - arr1

        # L1, L2 范数比较
        l1_norm_original = np.linalg.norm(arr1, ord=1)
        l1_norm_new = np.linalg.norm(arr2, ord=1)
        l1_norm_delta = np.linalg.norm(delta, ord=1)

        l2_norm_original = np.linalg.norm(arr1, ord=2)
        l2_norm_new = np.linalg.norm(arr2, ord=2)
        l2_norm_delta = np.linalg.norm(delta, ord=2)

        # 无穷范数（最大值）
        linf_norm_original = np.linalg.norm(arr1, ord=np.inf)
        linf_norm_new = np.linalg.norm(arr2, ord=np.inf)
        linf_norm_delta = np.linalg.norm(delta, ord=np.inf)

        # 相对变化率
        relative_l2_change = l2_norm_delta / (l2_norm_original + 1e-8)
        relative_l1_change = l1_norm_delta / (l1_norm_original + 1e-8)

        # KL 散度（针对概率分布，需要先归一化）
        kl_divergence = None
        try:
            # 将数值转换为概率分布（softmax）
            def safe_softmax(x):
                x_shifted = x - np.max(x)
                exp_x = np.exp(np.clip(x_shifted, -700, 700))
                return exp_x / (np.sum(exp_x) + 1e-8)

            if arr1.size <= 10000:  # 只对较小的张量计算，避免内存问题
                prob1 = safe_softmax(arr1)
                prob2 = safe_softmax(arr2)
                kl_divergence = float(np.sum(prob1 * np.log((prob1 + 1e-8) / (prob2 + 1e-8))))
        except Exception:
            pass

        # Wasserstein 距离（1阶）的简化版本
        wasserstein_1 = None
        try:
            if arr1.size <= 1000:  # 只对小张量计算
                from scipy import stats

                wasserstein_1 = float(stats.wasserstein_distance(arr1, arr2))
        except ImportError:
            pass
        except Exception:
            pass

        # 变化方向分析
        direction_consistency = float(np.dot(arr1, delta) / (np.linalg.norm(arr1) * np.linalg.norm(delta) + 1e-8))

        # 符号变化统计
        sign_changes = np.sum((arr1 > 0) != (arr2 > 0))
        sign_change_ratio = sign_changes / len(arr1) if len(arr1) > 0 else 0

        # 分位数变化
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        percentile_changes = {}
        for p in percentiles:
            p1 = np.percentile(arr1, p)
            p2 = np.percentile(arr2, p)
            percentile_changes[f"p{p}"] = {
                "original": float(p1),
                "new": float(p2),
                "change": float(p2 - p1),
                "relative_change": float((p2 - p1) / (abs(p1) + 1e-8)),
            }

        # 稀疏性分析
        sparsity_original = float(np.mean(np.abs(arr1) < 1e-6))
        sparsity_new = float(np.mean(np.abs(arr2) < 1e-6))
        sparsity_change = sparsity_new - sparsity_original

        # 能量变化（平方和）
        energy_original = float(np.sum(arr1**2))
        energy_new = float(np.sum(arr2**2))
        energy_change = energy_new - energy_original
        energy_relative_change = energy_change / (energy_original + 1e-8)

        # 统计信息
        stats = {
            "same_shape": same_shape,
            "same_dtype": same_dtype,
            "shape": list(tensor1.shape),
            "dtype": str(tensor1.dtype),
            "exactly_equal": bool(exactly_equal),
            # 数值统计
            "cosine_similarity": float(cosine_sim),
            "pearson_correlation": float(pearson_corr) if not np.isnan(pearson_corr) else None,
            "mse": float(mse),
            "rmse": float(rmse),
            # 差异统计
            "max_abs_diff": float(np.max(abs_diff)),
            "mean_abs_diff": float(np.mean(abs_diff)),
            "median_abs_diff": float(np.median(abs_diff)),
            "max_rel_diff": float(np.max(rel_diff)),
            "mean_rel_diff": float(np.mean(rel_diff)),
            # 阈值统计
            "equal_within_1e-6": float(np.mean(abs_diff < 1e-6)),
            "equal_within_1e-4": float(np.mean(abs_diff < 1e-4)),
            "equal_within_1e-2": float(np.mean(abs_diff < 1e-2)),
            # 范数比较
            "norm_analysis": {
                "l1_original": float(l1_norm_original),
                "l1_new": float(l1_norm_new),
                "l1_delta": float(l1_norm_delta),
                "l1_relative_change": float(relative_l1_change),
                "l2_original": float(l2_norm_original),
                "l2_new": float(l2_norm_new),
                "l2_delta": float(l2_norm_delta),
                "l2_relative_change": float(relative_l2_change),
                "linf_original": float(linf_norm_original),
                "linf_new": float(linf_norm_new),
                "linf_delta": float(linf_norm_delta),
            },
            # 分布差异
            "distribution_analysis": {
                "kl_divergence": kl_divergence,
                "wasserstein_distance": wasserstein_1,
                "direction_consistency": float(direction_consistency),
                "sign_changes": int(sign_changes),
                "sign_change_ratio": float(sign_change_ratio),
            },
            # 分位数变化
            "percentile_changes": percentile_changes,
            # 稀疏性分析
            "sparsity_analysis": {
                "original": float(sparsity_original),
                "new": float(sparsity_new),
                "change": float(sparsity_change),
            },
            # 能量分析
            "energy_analysis": {
                "original": float(energy_original),
                "new": float(energy_new),
                "change": float(energy_change),
                "relative_change": float(energy_relative_change),
            },
            # 基本统计量对比
            "tensor1_stats": {
                "mean": float(np.mean(arr1)),
                "std": float(np.std(arr1)),
                "min": float(np.min(arr1)),
                "max": float(np.max(arr1)),
                "zeros_ratio": float(np.mean(arr1 == 0)),
            },
            "tensor2_stats": {
                "mean": float(np.mean(arr2)),
                "std": float(np.std(arr2)),
                "min": float(np.min(arr2)),
                "max": float(np.max(arr2)),
                "zeros_ratio": float(np.mean(arr2 == 0)),
            },
        }

        return stats

    def compare_common_tensors(self, common_keys: List[str]) -> Dict:
        """比较共同的 tensor"""
        tensor_comparisons = {}
        similarity_summary = {
            "exactly_equal_count": 0,
            "high_similarity_count": 0,  # cosine > 0.99
            "medium_similarity_count": 0,  # 0.9 < cosine <= 0.99
            "low_similarity_count": 0,  # cosine <= 0.9
            "avg_cosine_similarity": 0.0,
            "avg_mse": 0.0,
            "shape_mismatch_count": 0,
        }

        cosine_sims = []
        mses = []

        print(f"比较 {len(common_keys)} 个共同的 tensor...")

        for key in common_keys:
            tensor1 = self.tensors1[key]
            tensor2 = self.tensors2[key]

            comparison = self.calculate_tensor_similarity(tensor1, tensor2)
            tensor_comparisons[key] = comparison

            # 统计汇总
            if comparison.get("exactly_equal", False):
                similarity_summary["exactly_equal_count"] += 1

            if "cosine_similarity" in comparison:
                cos_sim = comparison["cosine_similarity"]
                cosine_sims.append(cos_sim)
                mses.append(comparison["mse"])

                if cos_sim > 0.99:
                    similarity_summary["high_similarity_count"] += 1
                elif cos_sim > 0.9:
                    similarity_summary["medium_similarity_count"] += 1
                else:
                    similarity_summary["low_similarity_count"] += 1

            if not comparison.get("same_shape", True):
                similarity_summary["shape_mismatch_count"] += 1

        if cosine_sims:
            similarity_summary["avg_cosine_similarity"] = float(np.mean(cosine_sims))
            similarity_summary["avg_mse"] = float(np.mean(mses))

        return {"tensor_comparisons": tensor_comparisons, "summary": similarity_summary}

    def analyze_layer_patterns(self, common_keys: List[str]) -> Dict:
        """分析层级模式和权重分布变化"""
        layer_analysis = {
            "layer_groups": {},
            "weight_type_analysis": {},
            "magnitude_changes": {},
            "gradient_magnitude_estimation": {},
        }

        # 按层分组分析
        layer_groups = {}
        for key in common_keys:
            # 提取层名（假设格式为 layer.{num}.{type} 或类似）
            parts = key.split(".")
            if len(parts) >= 2:
                layer_name = ".".join(parts[:-1])  # 除了最后一部分
                weight_type = parts[-1]  # 最后一部分（如weight, bias等）
            else:
                layer_name = "root"
                weight_type = key

            if layer_name not in layer_groups:
                layer_groups[layer_name] = []
            layer_groups[layer_name].append((key, weight_type))

        # 分析每一层的变化
        for layer_name, tensors in layer_groups.items():
            layer_stats = {
                "tensor_count": len(tensors),
                "weight_types": [t[1] for t in tensors],
                "total_parameters": 0,
                "avg_l2_change": 0,
                "max_l2_change": 0,
                "parameter_change_distribution": {},
            }

            l2_changes = []
            total_params = 0

            for key, weight_type in tensors:
                if key in self.tensors1 and key in self.tensors2:
                    tensor1 = self.tensors1[key]
                    tensor2 = self.tensors2[key]

                    if tensor1.shape == tensor2.shape:
                        delta = tensor2.detach().cpu().float().numpy() - tensor1.detach().cpu().float().numpy()
                        l2_change = float(np.linalg.norm(delta.flatten()))
                        l2_changes.append(l2_change)
                        total_params += tensor1.numel()

            if l2_changes:
                layer_stats["total_parameters"] = total_params
                layer_stats["avg_l2_change"] = float(np.mean(l2_changes))
                layer_stats["max_l2_change"] = float(np.max(l2_changes))
                layer_stats["parameter_change_distribution"] = {
                    "mean": float(np.mean(l2_changes)),
                    "std": float(np.std(l2_changes)),
                    "min": float(np.min(l2_changes)),
                    "max": float(np.max(l2_changes)),
                }

            layer_analysis["layer_groups"][layer_name] = layer_stats

        # 按权重类型分析
        weight_types = {}
        for key in common_keys:
            parts = key.split(".")
            weight_type = parts[-1] if len(parts) > 1 else "unknown"

            if weight_type not in weight_types:
                weight_types[weight_type] = []
            weight_types[weight_type].append(key)

        for weight_type, keys in weight_types.items():
            type_stats = {
                "count": len(keys),
                "avg_magnitude_change": 0.0,
                "total_parameters": 0,
            }

            magnitude_changes = []
            total_params = 0

            for key in keys:
                if key in self.tensors1 and key in self.tensors2:
                    tensor1 = self.tensors1[key]
                    tensor2 = self.tensors2[key]

                    if tensor1.shape == tensor2.shape:
                        arr1 = tensor1.detach().cpu().float().numpy().flatten()
                        arr2 = tensor2.detach().cpu().float().numpy().flatten()

                        magnitude_change = np.linalg.norm(arr2 - arr1) / (np.linalg.norm(arr1) + 1e-8)
                        magnitude_changes.append(magnitude_change)
                        total_params += tensor1.numel()

            if magnitude_changes:
                type_stats["avg_magnitude_change"] = float(np.mean(magnitude_changes))
                type_stats["total_parameters"] = total_params

            layer_analysis["weight_type_analysis"][weight_type] = type_stats

        return layer_analysis

    def generate_training_insights(self, comparison_results: Dict) -> Dict:
        """生成训练洞察报告"""
        insights = {
            "training_effectiveness": {},
            "potential_issues": [],
            "recommendations": [],
            "change_summary": {},
        }

        if "tensor_comparison" not in comparison_results or not comparison_results["tensor_comparison"]:
            return insights

        tensor_comp = comparison_results["tensor_comparison"]
        summary = tensor_comp["summary"]

        # 训练有效性分析
        total_tensors = len(tensor_comp["tensor_comparisons"])
        if total_tensors > 0:
            change_ratio = (total_tensors - summary["exactly_equal_count"]) / total_tensors
            avg_similarity = summary.get("avg_cosine_similarity", 0)

            insights["training_effectiveness"] = {
                "changed_tensor_ratio": float(change_ratio),
                "avg_similarity": float(avg_similarity),
                "significant_changes": summary["low_similarity_count"] + summary["medium_similarity_count"],
                "stability_score": float(summary["high_similarity_count"] / total_tensors) if total_tensors > 0 else 0,
            }

        # 潜在问题检测
        if summary["avg_cosine_similarity"] < 0.8:
            insights["potential_issues"].append("平均余弦相似度较低，可能存在过度训练或学习率过高")

        if summary["exactly_equal_count"] / total_tensors > 0.5:
            insights["potential_issues"].append("超过一半的参数没有变化，可能存在欠拟合或学习率过低")

        if summary["low_similarity_count"] > total_tensors * 0.1:
            insights["potential_issues"].append("超过10%的参数发生剧烈变化，可能存在梯度爆炸或不稳定训练")

        # 生成建议
        if summary["avg_cosine_similarity"] > 0.95:
            insights["recommendations"].append("参数变化较小，可以考虑增加学习率或训练更长时间")
        elif summary["avg_cosine_similarity"] < 0.7:
            insights["recommendations"].append("参数变化剧烈，建议降低学习率或检查数据质量")

        # 变化摘要
        insights["change_summary"] = {
            "total_parameters_analyzed": total_tensors,
            "unchanged_parameters": summary["exactly_equal_count"],
            "significantly_changed": summary["low_similarity_count"],
            "moderately_changed": summary["medium_similarity_count"],
            "slightly_changed": summary["high_similarity_count"],
        }

        return insights

    def generate_visualization_data(self, common_keys: List[str]) -> Dict:
        """生成可视化数据"""
        viz_data = {
            "change_magnitude_distribution": [],
            "layer_wise_changes": {},
            "weight_type_changes": {},
            "similarity_histogram": [],
            "top_changed_tensors": [],
        }

        # 收集变化幅度数据
        change_magnitudes = []
        similarities = []
        tensor_changes = []

        for key in common_keys:
            if key in self.tensors1 and key in self.tensors2:
                tensor1 = self.tensors1[key]
                tensor2 = self.tensors2[key]

                if tensor1.shape == tensor2.shape:
                    arr1 = tensor1.detach().cpu().float().numpy().flatten()
                    arr2 = tensor2.detach().cpu().float().numpy().flatten()

                    # 计算变化幅度
                    delta_norm = np.linalg.norm(arr2 - arr1)
                    original_norm = np.linalg.norm(arr1)
                    relative_change = delta_norm / (original_norm + 1e-8)

                    change_magnitudes.append(float(relative_change))

                    # 计算相似度
                    from sklearn.metrics.pairwise import cosine_similarity

                    cosine_sim = cosine_similarity(arr1.reshape(1, -1), arr2.reshape(1, -1))[0][0]
                    similarities.append(float(cosine_sim))

                    # 记录张量变化信息
                    tensor_changes.append(
                        {
                            "name": key,
                            "relative_change": float(relative_change),
                            "cosine_similarity": float(cosine_sim),
                            "shape": list(tensor1.shape),
                            "parameter_count": int(tensor1.numel()),
                        }
                    )

        # 变化幅度分布
        if change_magnitudes:
            viz_data["change_magnitude_distribution"] = change_magnitudes

            # 相似度直方图
            viz_data["similarity_histogram"] = similarities

            # 按变化幅度排序，找出变化最大的张量
            tensor_changes.sort(key=lambda x: x["relative_change"], reverse=True)
            viz_data["top_changed_tensors"] = tensor_changes[:20]  # 前20个变化最大的

        # 按层级聚合变化
        layer_changes = {}
        weight_type_changes = {}

        for tensor_info in tensor_changes:
            key = tensor_info["name"]
            change = tensor_info["relative_change"]

            # 层级分析
            parts = key.split(".")
            if len(parts) >= 2:
                layer_name = ".".join(parts[:-1])
                weight_type = parts[-1]
            else:
                layer_name = "root"
                weight_type = key

            # 按层聚合
            if layer_name not in layer_changes:
                layer_changes[layer_name] = []
            layer_changes[layer_name].append(change)

            # 按权重类型聚合
            if weight_type not in weight_type_changes:
                weight_type_changes[weight_type] = []
            weight_type_changes[weight_type].append(change)

        # 计算每层的统计信息
        for layer_name, changes in layer_changes.items():
            viz_data["layer_wise_changes"][layer_name] = {
                "mean_change": float(np.mean(changes)),
                "max_change": float(np.max(changes)),
                "min_change": float(np.min(changes)),
                "std_change": float(np.std(changes)),
                "tensor_count": len(changes),
            }

        # 计算每种权重类型的统计信息
        for weight_type, changes in weight_type_changes.items():
            viz_data["weight_type_changes"][weight_type] = {
                "mean_change": float(np.mean(changes)),
                "max_change": float(np.max(changes)),
                "min_change": float(np.min(changes)),
                "std_change": float(np.std(changes)),
                "tensor_count": len(changes),
            }

        return viz_data

    def run_comparison(self) -> None:
        # 加载文件
        self.load_safetensors()

        # 基本信息
        print("计算基本信息...")
        file1_info = self.get_file_basic_info(self.file1_path, self.tensors1)
        file2_info = self.get_file_basic_info(self.file2_path, self.tensors2)

        # Key 比较
        print("比较 tensor keys...")
        key_comparison = self.compare_keys()

        # Tensor 比较
        common_keys = key_comparison["common_keys"]
        tensor_comparison = {}
        layer_analysis = {}
        training_insights = {}

        if common_keys:
            tensor_comparison = self.compare_common_tensors(common_keys)
            layer_analysis = self.analyze_layer_patterns(common_keys)

        # 汇总结果
        self.comparison_results = {
            "timestamp": datetime.now().isoformat(),
            "file1_info": file1_info,
            "file2_info": file2_info,
            "metadata1": self.metadata1,
            "metadata2": self.metadata2,
            "key_comparison": key_comparison,
            "tensor_comparison": tensor_comparison,
            "layer_analysis": layer_analysis,
        }

        # 添加可视化数据（如果有）
        if common_keys:
            visualization_data = self.generate_visualization_data(common_keys)
            self.comparison_results["visualization_data"] = visualization_data

        # 生成训练洞察（需要在有tensor_comparison结果后）
        if tensor_comparison:
            training_insights = self.generate_training_insights(self.comparison_results)
            self.comparison_results["training_insights"] = training_insights

        print("比较完成！")

    def generate_report(self) -> None:
        """生成详细报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 生成 JSON 报告
        json_path = self.output_dir / f"safetensor_comparison_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.comparison_results, f, indent=2, ensure_ascii=False)

        # 生成文本报告
        txt_path = self.output_dir / f"safetensor_comparison_{timestamp}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            self._write_text_report(f)

        print("报告已保存到:")
        print(f"  JSON: {json_path}")
        print(f"  文本: {txt_path}")

    def _write_text_report(self, f) -> None:
        """写入文本格式报告"""
        results = self.comparison_results

        f.write("SafeTensor 文件比较报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"生成时间: {results['timestamp']}\n\n")

        # 文件基本信息
        f.write("文件基本信息\n")
        f.write("-" * 30 + "\n")
        f.write(f"文件1: {results['file1_info']['file_path']}\n")
        f.write(f"  文件大小: {results['file1_info']['file_size_mb']} MB\n")
        f.write(f"  tensor数量: {results['file1_info']['num_tensors']}\n")
        f.write(f"  总参数量: {results['file1_info']['total_parameters']:,}\n")
        f.write(f"  内存占用: {results['file1_info']['total_memory_mb']} MB\n")
        f.write(f"  数据类型分布: {results['file1_info']['dtype_distribution']}\n\n")

        f.write(f"文件2: {results['file2_info']['file_path']}\n")
        f.write(f"  文件大小: {results['file2_info']['file_size_mb']} MB\n")
        f.write(f"  tensor数量: {results['file2_info']['num_tensors']}\n")
        f.write(f"  总参数量: {results['file2_info']['total_parameters']:,}\n")
        f.write(f"  内存占用: {results['file2_info']['total_memory_mb']} MB\n")
        f.write(f"  数据类型分布: {results['file2_info']['dtype_distribution']}\n\n")

        # Key 比较
        key_comp = results["key_comparison"]
        f.write("Key 比较分析\n")
        f.write("-" * 30 + "\n")
        f.write(f"文件1 keys: {key_comp['total_keys_file1']}\n")
        f.write(f"文件2 keys: {key_comp['total_keys_file2']}\n")
        f.write(f"共同 keys: {key_comp['common_keys_count']}\n")
        f.write(f"仅在文件1: {key_comp['only_in_file1_count']}\n")
        f.write(f"仅在文件2: {key_comp['only_in_file2_count']}\n")
        f.write(f"Key 重合率: {key_comp['key_overlap_ratio']:.2%}\n\n")

        if key_comp["only_in_file1"]:
            f.write("仅在文件1中的 keys:\n")
            for key in key_comp["only_in_file1"][:10]:  # 只显示前10个
                f.write(f"  - {key}\n")
            if len(key_comp["only_in_file1"]) > 10:
                f.write(f"  ... 还有 {len(key_comp['only_in_file1']) - 10} 个\n")
            f.write("\n")

        if key_comp["only_in_file2"]:
            f.write("仅在文件2中的 keys:\n")
            for key in key_comp["only_in_file2"][:10]:  # 只显示前10个
                f.write(f"  - {key}\n")
            if len(key_comp["only_in_file2"]) > 10:
                f.write(f"  ... 还有 {len(key_comp['only_in_file2']) - 10} 个\n")
            f.write("\n")

        # Tensor 比较
        if "tensor_comparison" in results and results["tensor_comparison"]:
            tensor_comp = results["tensor_comparison"]
            summary = tensor_comp["summary"]

            f.write("Tensor 相似度分析\n")
            f.write("-" * 30 + "\n")
            f.write(f"完全相等的 tensor: {summary['exactly_equal_count']}\n")
            f.write(f"高相似度 (>0.99): {summary['high_similarity_count']}\n")
            f.write(f"中等相似度 (0.9-0.99): {summary['medium_similarity_count']}\n")
            f.write(f"低相似度 (<0.9): {summary['low_similarity_count']}\n")
            f.write(f"形状不匹配: {summary['shape_mismatch_count']}\n")
            f.write(f"平均余弦相似度: {summary['avg_cosine_similarity']:.4f}\n")
            f.write(f"平均均方误差: {summary['avg_mse']:.2e}\n\n")

            f.write("详细 Tensor 比较 (按相似度排序):\n")
            f.write("-" * 50 + "\n")
            tensor_comparisons = tensor_comp["tensor_comparisons"]

            # 创建排序列表 (key, comparison, sort_value)
            sorted_tensors = []
            for key, comp in tensor_comparisons.items():
                if "error" in comp:
                    # 有错误的排在最后
                    sort_value = -2.0
                elif comp.get("exactly_equal", False):
                    # 完全相等的排在最前，使用1.0作为排序值
                    sort_value = 1.0
                else:
                    # 按余弦相似度排序
                    sort_value = comp.get("cosine_similarity", -1.0)
                sorted_tensors.append((key, comp, sort_value))

            # 按相似度从高到低排序
            sorted_tensors.sort(key=lambda x: x[2], reverse=True)

            for i, (key, comp, sort_value) in enumerate(sorted_tensors):
                f.write(f"\n{i + 1:3d}. {key}:\n")
                if "error" in comp:
                    f.write(f"     错误: {comp['error']}\n")
                else:
                    f.write(f"     形状: {comp['shape']}\n")
                    f.write(f"     完全相等: {'是' if comp['exactly_equal'] else '否'}\n")
                    if "cosine_similarity" in comp:
                        f.write(f"     余弦相似度: {comp['cosine_similarity']:.6f}\n")
                        f.write(f"     均方误差: {comp['mse']:.2e}\n")
                        f.write(f"     最大绝对差异: {comp['max_abs_diff']:.2e}\n")
                        f.write(f"     平均绝对差异: {comp['mean_abs_diff']:.2e}\n")
                        f.write(f"     1e-6内相等比例: {comp['equal_within_1e-6']:.2%}\n")

                        # 添加新的分析结果
                        if "norm_analysis" in comp:
                            norm = comp["norm_analysis"]
                            f.write(
                                f"     L2范数变化: {norm['l2_original']:.2e} → {norm['l2_new']:.2e} (相对变化: {norm['l2_relative_change']:.2%})\n"
                            )
                            f.write(
                                f"     L1范数变化: {norm['l1_original']:.2e} → {norm['l1_new']:.2e} (相对变化: {norm['l1_relative_change']:.2%})\n"
                            )

                        if "distribution_analysis" in comp:
                            dist = comp["distribution_analysis"]
                            if dist["kl_divergence"] is not None:
                                f.write(f"     KL散度: {dist['kl_divergence']:.4f}\n")
                            if dist["wasserstein_distance"] is not None:
                                f.write(f"     Wasserstein距离: {dist['wasserstein_distance']:.4f}\n")
                            f.write(f"     方向一致性: {dist['direction_consistency']:.4f}\n")
                            f.write(f"     符号变化: {dist['sign_changes']} 个 ({dist['sign_change_ratio']:.2%})\n")

                        if "sparsity_analysis" in comp:
                            sparse = comp["sparsity_analysis"]
                            f.write(
                                f"     稀疏性变化: {sparse['original']:.2%} → {sparse['new']:.2%} (变化: {sparse['change']:+.2%})\n"
                            )

                        if "energy_analysis" in comp:
                            energy = comp["energy_analysis"]
                            f.write(
                                f"     能量变化: {energy['original']:.2e} → {energy['new']:.2e} (相对变化: {energy['relative_change']:+.2%})\n"
                            )

        # 层级分析报告
        if "layer_analysis" in results and results["layer_analysis"]:
            layer_analysis = results["layer_analysis"]
            f.write("\n\n层级分析报告\n")
            f.write("-" * 30 + "\n")

            if "layer_groups" in layer_analysis:
                f.write("按层分析:\n")
                layer_groups = layer_analysis["layer_groups"]
                for layer_name, stats in sorted(layer_groups.items()):
                    f.write(f"\n  {layer_name}:\n")
                    f.write(f"    参数数量: {stats['total_parameters']:,}\n")
                    f.write(f"    权重类型: {', '.join(set(stats['weight_types']))}\n")
                    if stats["avg_l2_change"] > 0:
                        f.write(f"    平均L2变化: {stats['avg_l2_change']:.2e}\n")
                        f.write(f"    最大L2变化: {stats['max_l2_change']:.2e}\n")

            if "weight_type_analysis" in layer_analysis:
                f.write("\n按权重类型分析:\n")
                weight_analysis = layer_analysis["weight_type_analysis"]
                for weight_type, stats in sorted(weight_analysis.items()):
                    f.write(f"\n  {weight_type}:\n")
                    f.write(f"    张量数量: {stats['count']}\n")
                    f.write(f"    总参数量: {stats['total_parameters']:,}\n")
                    if stats["avg_magnitude_change"] > 0:
                        f.write(f"    平均幅度变化: {stats['avg_magnitude_change']:.2%}\n")

        # 训练洞察报告
        if "training_insights" in results and results["training_insights"]:
            insights = results["training_insights"]
            f.write("\n\n训练洞察报告\n")
            f.write("-" * 30 + "\n")

            if "training_effectiveness" in insights:
                eff = insights["training_effectiveness"]
                f.write("训练有效性:\n")
                f.write(f"  参数变化比例: {eff.get('changed_tensor_ratio', 0):.2%}\n")
                f.write(f"  平均相似度: {eff.get('avg_similarity', 0):.4f}\n")
                f.write(f"  显著变化数量: {eff.get('significant_changes', 0)}\n")
                f.write(f"  稳定性评分: {eff.get('stability_score', 0):.2%}\n")

            if "potential_issues" in insights and insights["potential_issues"]:
                f.write("\n潜在问题:\n")
                for issue in insights["potential_issues"]:
                    f.write(f"  ⚠️  {issue}\n")

            if "recommendations" in insights and insights["recommendations"]:
                f.write("\n建议:\n")
                for rec in insights["recommendations"]:
                    f.write(f"  💡 {rec}\n")

            if "change_summary" in insights:
                summary = insights["change_summary"]
                f.write("\n变化摘要:\n")
                f.write(f"  分析的参数总数: {summary.get('total_parameters_analyzed', 0)}\n")
                f.write(f"  未变化参数: {summary.get('unchanged_parameters', 0)}\n")
                f.write(f"  轻微变化: {summary.get('slightly_changed', 0)}\n")
                f.write(f"  中等变化: {summary.get('moderately_changed', 0)}\n")
                f.write(f"  显著变化: {summary.get('significantly_changed', 0)}\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="比较两个 SafeTensor 文件")
    parser.add_argument("file1", help="第一个 safetensor 文件路径")
    parser.add_argument("file2", help="第二个 safetensor 文件路径")
    parser.add_argument("-o", "--output", default="./reports", help="报告输出目录 (默认: ./reports)")

    args = parser.parse_args()

    # 检查文件是否存在
    if not Path(args.file1).exists():
        print(f"错误: 文件不存在 - {args.file1}")
        sys.exit(1)

    if not Path(args.file2).exists():
        print(f"错误: 文件不存在 - {args.file2}")
        sys.exit(1)

    try:
        # 创建比较器并运行
        comparator = SafeTensorComparator(args.file1, args.file2, args.output)
        comparator.run_comparison()
        comparator.generate_report()

    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
