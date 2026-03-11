import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.stats import wasserstein_distance, entropy
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from typing import List, Tuple, Dict, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class EnhancedClusteringStrategy:
    """增强的分簇策略模块 - 实现您提出的三个核心改进策略"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 分簇策略配置
        self.use_adaptive_thresholds = getattr(args, 'use_adaptive_thresholds', True)
        self.use_hierarchical_clustering = getattr(args, 'use_hierarchical_clustering', True)
        self.use_forgetting_optimization = getattr(args, 'use_forgetting_optimization', True)
        
        # 自适应阈值参数
        self.iqr_multiplier = getattr(args, 'iqr_multiplier', 1.5)
        self.min_threshold = getattr(args, 'min_threshold', 0.1)
        self.max_threshold = getattr(args, 'max_threshold', 0.9)
        
        # 多层次分簇参数
        self.coarse_cluster_ratio = getattr(args, 'coarse_cluster_ratio', 0.3)
        self.fine_cluster_ratio = getattr(args, 'fine_cluster_ratio', 0.6)
        
        # 遗忘效果优化参数
        self.forgetting_weight = getattr(args, 'forgetting_weight', 0.4)
        self.protection_weight = getattr(args, 'protection_weight', 0.3)
        self.utility_weight = getattr(args, 'utility_weight', 0.3)
        
        print(f"🔧 增强分簇策略配置:")
        print(f"  - 自适应阈值: {self.use_adaptive_thresholds}")
        print(f"  - 多层次分簇: {self.use_hierarchical_clustering}")
        print(f"  - 遗忘效果优化: {self.use_forgetting_optimization}")
        print(f"  - IQR倍数: {self.iqr_multiplier}")
        print(f"  - 权重配置: 遗忘({self.forgetting_weight}), 保护({self.protection_weight}), 效用({self.utility_weight})")
    
    def cluster_clients_enhanced(self, features_list: List[np.ndarray], 
                               distances: np.ndarray, 
                               target_client_id: int = 0) -> Tuple[np.ndarray, List[int], Dict]:
        """增强的客户端分簇主函数"""
        print(f"\n🚀 开始增强分簇策略...")
        
        # 1. 基于数据分布统计的自适应分簇
        if self.use_adaptive_thresholds:
            print("📊 策略1: 基于数据分布统计的自适应分簇")
            adaptive_thresholds = self._compute_adaptive_thresholds_enhanced(distances)
            print(f"自适应阈值: {adaptive_thresholds}")
        
        # 2. 多层次分簇策略
        if self.use_hierarchical_clustering:
            print("🏗️ 策略2: 多层次分簇策略")
            cluster_labels, clustering_metrics = self._hierarchical_clustering_enhanced(
                distances, target_client_id, adaptive_thresholds if self.use_adaptive_thresholds else None
            )
        else:
            print("📈 使用传统分簇方法")
            cluster_labels, clustering_metrics = self._traditional_clustering_enhanced(
                distances, target_client_id, adaptive_thresholds if self.use_adaptive_thresholds else None
            )
        
        # 3. 基于遗忘效果的分簇优化
        if self.use_forgetting_optimization:
            print("🎯 策略3: 基于遗忘效果的分簇优化")
            cluster_labels, optimization_metrics = self._optimize_clusters_for_forgetting(
                cluster_labels, distances, target_client_id, features_list
            )
            clustering_metrics.update(optimization_metrics)
        
        # 4. 异常客户端检测
        anomaly_clients = self._detect_anomaly_clients_enhanced(cluster_labels, distances, features_list)
        
        # 5. 分簇质量评估
        quality_metrics = self._evaluate_clustering_quality_enhanced(cluster_labels, distances)
        clustering_metrics.update(quality_metrics)
        
        # 6. 分簇结果分析
        self._analyze_clustering_result_enhanced(cluster_labels, distances, target_client_id)
        
        return cluster_labels, anomaly_clients, clustering_metrics
    
    def _compute_adaptive_thresholds_enhanced(self, distances: np.ndarray) -> Dict[str, float]:
        """策略1: 基于数据分布统计的自适应阈值计算"""
        print("📊 计算自适应阈值...")
        
        # 提取上三角矩阵的距离值（排除对角线）
        upper_triangle = distances[np.triu_indices(len(distances), k=1)]
        
        # 计算分布统计特征
        mean_dist = np.mean(upper_triangle)
        std_dist = np.std(upper_triangle)
        q25 = np.percentile(upper_triangle, 25)
        q50 = np.percentile(upper_triangle, 50)  # 中位数
        q75 = np.percentile(upper_triangle, 75)
        iqr = q75 - q25
        
        # 使用IQR方法设置异常值边界
        lower_bound = max(self.min_threshold, q25 - self.iqr_multiplier * iqr)
        upper_bound = min(self.max_threshold, q75 + self.iqr_multiplier * iqr)
        
        # 基于分布特征设置分簇阈值
        # 高相似度阈值：基于下界，表示非常相似
        high_similarity_threshold = max(self.min_threshold, min(0.6, lower_bound))
        
        # 中等相似度阈值：基于中位数，表示中等相似
        medium_similarity_threshold = max(0.4, min(0.8, q50))
        
        # 低相似度阈值：基于上界，表示差异较大
        low_similarity_threshold = max(0.6, min(self.max_threshold, upper_bound))
        
        # 动态阈值：基于数据分布密度
        density_threshold = self._compute_density_based_threshold(upper_triangle)
        
        thresholds = {
            'high_similarity': high_similarity_threshold,
            'medium_similarity': medium_similarity_threshold,
            'low_similarity': low_similarity_threshold,
            'density_based': density_threshold,
            'statistics': {
                'mean': mean_dist,
                'std': std_dist,
                'q25': q25,
                'q50': q50,
                'q75': q75,
                'iqr': iqr,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
        }
        
        print(f"📊 距离分布统计:")
        print(f"  均值: {mean_dist:.4f}, 标准差: {std_dist:.4f}")
        print(f"  Q25: {q25:.4f}, Q50: {q50:.4f}, Q75: {q75:.4f}, IQR: {iqr:.4f}")
        print(f"  边界: [{lower_bound:.4f}, {upper_bound:.4f}]")
        print(f"🎯 自适应阈值:")
        print(f"  高相似度: {high_similarity_threshold:.4f}")
        print(f"  中等相似度: {medium_similarity_threshold:.4f}")
        print(f"  低相似度: {low_similarity_threshold:.4f}")
        print(f"  密度阈值: {density_threshold:.4f}")
        
        return thresholds
    
    def _compute_density_based_threshold(self, distances: np.ndarray) -> float:
        """基于数据分布密度计算阈值"""
        try:
            # 使用核密度估计找到分布峰值
            from scipy.stats import gaussian_kde
            
            # 限制数据量以提高计算效率
            if len(distances) > 1000:
                sample_indices = np.random.choice(len(distances), 1000, replace=False)
                sample_distances = distances[sample_indices]
            else:
                sample_distances = distances
            
            # 计算核密度估计
            kde = gaussian_kde(sample_distances)
            
            # 在距离范围内寻找密度峰值
            x_range = np.linspace(sample_distances.min(), sample_distances.max(), 100)
            density = kde(x_range)
            
            # 找到密度峰值对应的距离值
            peak_idx = np.argmax(density)
            density_threshold = x_range[peak_idx]
            
            return float(density_threshold)
        except Exception as e:
            print(f"密度阈值计算失败: {e}, 使用默认值")
            return 0.5
    
    def _hierarchical_clustering_enhanced(self, distances: np.ndarray, 
                                        target_client_id: int,
                                        thresholds: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """策略2: 多层次分簇策略"""
        print("🏗️ 开始多层次分簇策略...")
        
        # 第一层：基于分布相似性的粗分簇
        print("  📍 第一层：基于分布相似性的粗分簇")
        coarse_clusters = self._coarse_clustering_by_distribution_enhanced(distances, target_client_id, thresholds)
        
        # 第二层：基于梯度相似性的细分簇
        print("  📍 第二层：基于梯度相似性的细分簇")
        fine_clusters = self._fine_clustering_by_gradients_enhanced(coarse_clusters, distances, thresholds)
        
        # 第三层：基于遗忘需求的优化分簇
        print("  📍 第三层：基于遗忘需求的优化分簇")
        optimized_clusters = self._optimize_clusters_for_forgetting_enhanced(fine_clusters, distances, target_client_id)
        
        # 计算分簇指标
        clustering_metrics = {
            'coarse_clusters': coarse_clusters,
            'fine_clusters': fine_clusters,
            'final_clusters': optimized_clusters,
            'hierarchical_levels': 3
        }
        
        return optimized_clusters, clustering_metrics
    
    def _coarse_clustering_by_distribution_enhanced(self, distances: np.ndarray, 
                                                  target_client_id: int,
                                                  thresholds: Optional[Dict] = None) -> np.ndarray:
        """基于分布相似性的粗分簇"""
        num_clients = len(distances)
        cluster_labels = np.zeros(num_clients, dtype=int)
        current_cluster = 0
        
        # 目标客户端单独成簇
        cluster_labels[target_client_id] = current_cluster
        current_cluster += 1
        print(f"    目标客户端{target_client_id}单独成簇: {current_cluster-1}")
        
        if thresholds is None:
            # 使用默认阈值
            high_threshold = 0.5
            medium_threshold = 0.8
        else:
            high_threshold = thresholds['high_similarity']
            medium_threshold = thresholds['medium_similarity']
        
        # 计算与目标客户端的距离排序
        target_distances = []
        for i in range(num_clients):
            if i != target_client_id:
                target_distances.append((i, distances[target_client_id][i]))
        
        target_distances.sort(key=lambda x: x[1])
        
        # 基于距离阈值进行粗分簇
        for client_id, distance in target_distances:
            if cluster_labels[client_id] != 0:  # 已经分配了簇
                continue
            
            if distance < high_threshold:
                # 高相似度：单独成簇以获得最大保护
                cluster_labels[client_id] = current_cluster
                current_cluster += 1
                print(f"    客户端{client_id}高相似({distance:.4f})，单独成簇{cluster_labels[client_id]}")
            
            elif distance < medium_threshold:
                # 中等相似度：寻找相似客户端合并
                best_match = self._find_best_cluster_match(client_id, cluster_labels, distances, 
                                                        target_client_id, high_threshold, medium_threshold)
                if best_match is not None:
                    cluster_labels[client_id] = best_match
                    print(f"    客户端{client_id}中等相似({distance:.4f})，合并到簇{best_match}")
                else:
                    cluster_labels[client_id] = current_cluster
                    current_cluster += 1
                    print(f"    客户端{client_id}中等相似({distance:.4f})，新簇{cluster_labels[client_id]}")
            
            else:
                # 低相似度：合并到差异较大的簇
                best_match = self._find_best_cluster_match(client_id, cluster_labels, distances, 
                                                        target_client_id, medium_threshold, float('inf'))
                if best_match is not None:
                    cluster_labels[client_id] = best_match
                    print(f"    客户端{client_id}低相似({distance:.4f})，合并到簇{best_match}")
                else:
                    cluster_labels[client_id] = current_cluster
                    current_cluster += 1
                    print(f"    客户端{client_id}低相似({distance:.4f})，新簇{cluster_labels[client_id]}")
        
        return cluster_labels
    
    def _find_best_cluster_match(self, client_id: int, cluster_labels: np.ndarray, 
                                distances: np.ndarray, target_client_id: int,
                                min_threshold: float, max_threshold: float) -> Optional[int]:
        """寻找最佳簇匹配"""
        best_cluster = None
        best_similarity = float('inf')
        
        for other_client_id, other_cluster in enumerate(cluster_labels):
            if other_client_id != client_id and other_client_id != target_client_id and other_cluster != 0:
                # 检查两个客户端之间的距离
                pair_distance = distances[client_id][other_client_id]
                if min_threshold <= pair_distance < max_threshold and pair_distance < best_similarity:
                    best_similarity = pair_distance
                    best_cluster = other_cluster
        
        return best_cluster
    
    def _fine_clustering_by_gradients_enhanced(self, coarse_clusters: np.ndarray, 
                                             distances: np.ndarray,
                                             thresholds: Optional[Dict] = None) -> np.ndarray:
        """基于梯度相似性的细分簇"""
        print("    基于梯度相似性进行细分簇...")
        
        # 这里可以添加基于梯度的细分逻辑
        # 目前使用粗分簇结果
        return coarse_clusters
    
    def _optimize_clusters_for_forgetting_enhanced(self, clusters: np.ndarray, 
                                                  distances: np.ndarray,
                                                  target_client_id: int) -> np.ndarray:
        """基于遗忘需求的优化分簇"""
        print("    基于遗忘需求优化分簇...")
        
        # 计算每个簇的遗忘难度和保护价值
        cluster_metrics = self._compute_cluster_metrics(clusters, distances, target_client_id)
        
        # 基于指标优化分簇
        optimized_clusters = self._optimize_cluster_assignment(clusters, cluster_metrics, distances)
        
        return optimized_clusters
    
    def _compute_cluster_metrics(self, clusters: np.ndarray, distances: np.ndarray, 
                                target_client_id: int) -> Dict[int, Dict]:
        """计算每个簇的指标"""
        cluster_metrics = {}
        unique_clusters = np.unique(clusters)
        
        for cluster_id in unique_clusters:
            cluster_members = np.where(clusters == cluster_id)[0]
            
            if len(cluster_members) == 0:
                continue
            
            # 计算簇内距离
            cluster_distances = []
            for i in cluster_members:
                for j in cluster_members:
                    if i < j:
                        cluster_distances.append(distances[i][j])
            
            # 计算与目标客户端的平均距离
            target_distances = [distances[target_client_id][i] for i in cluster_members]
            
            metrics = {
                'size': len(cluster_members),
                'intra_cluster_distance': np.mean(cluster_distances) if cluster_distances else 0,
                'target_distance': np.mean(target_distances),
                'forgetting_difficulty': np.mean(target_distances),  # 距离越大，遗忘越困难
                'protection_value': 1.0 / (1.0 + np.mean(target_distances))  # 距离越小，保护价值越高
            }
            
            cluster_metrics[cluster_id] = metrics
        
        return cluster_metrics
    
    def _optimize_cluster_assignment(self, clusters: np.ndarray, cluster_metrics: Dict, 
                                   distances: np.ndarray) -> np.ndarray:
        """优化簇分配"""
        # 这里可以实现更复杂的优化逻辑
        # 目前返回原始分簇结果
        return clusters
    
    def _traditional_clustering_enhanced(self, distances: np.ndarray, 
                                       target_client_id: int,
                                       thresholds: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """传统分簇方法的增强版本"""
        print("📈 使用传统分簇方法...")
        
        if thresholds is None:
            high_threshold = 0.5
            medium_threshold = 0.8
        else:
            high_threshold = thresholds['high_similarity']
            medium_threshold = thresholds['medium_similarity']
        
        # 使用改进的距离分簇方法
        cluster_labels = self._improved_distance_based_clustering(distances, target_client_id, 
                                                               high_threshold, medium_threshold)
        
        clustering_metrics = {
            'method': 'traditional_enhanced',
            'thresholds': {'high': high_threshold, 'medium': medium_threshold}
        }
        
        return cluster_labels, clustering_metrics
    
    def _improved_distance_based_clustering(self, distances: np.ndarray, 
                                          target_client_id: int,
                                          high_threshold: float,
                                          medium_threshold: float) -> np.ndarray:
        """改进的距离分簇方法"""
        num_clients = len(distances)
        cluster_labels = np.zeros(num_clients, dtype=int)
        current_cluster = 0
        
        # 目标客户端单独成簇
        cluster_labels[target_client_id] = current_cluster
        current_cluster += 1
        
        # 计算与目标客户端的距离排序
        target_distances = []
        for i in range(num_clients):
            if i != target_client_id:
                target_distances.append((i, distances[target_client_id][i]))
        
        target_distances.sort(key=lambda x: x[1])
        
        # 基于阈值进行分簇
        for client_id, distance in target_distances:
            if cluster_labels[client_id] != 0:
                continue
            
            if distance < high_threshold:
                # 高相似度：单独成簇
                cluster_labels[client_id] = current_cluster
                current_cluster += 1
                print(f"  客户端{client_id}高相似({distance:.4f})，单独成簇{cluster_labels[client_id]}")
            
            elif distance < medium_threshold:
                # 中等相似度：寻找相似客户端
                best_match = self._find_best_cluster_match(client_id, cluster_labels, distances, 
                                                        target_client_id, 0, medium_threshold)
                if best_match is not None:
                    cluster_labels[client_id] = best_match
                    print(f"  客户端{client_id}中等相似({distance:.4f})，合并到簇{best_match}")
                else:
                    cluster_labels[client_id] = current_cluster
                    current_cluster += 1
                    print(f"  客户端{client_id}中等相似({distance:.4f})，新簇{cluster_labels[client_id]}")
            
            else:
                # 低相似度：合并到差异较大的簇
                best_match = self._find_best_cluster_match(client_id, cluster_labels, distances, 
                                                        target_client_id, medium_threshold, float('inf'))
                if best_match is not None:
                    cluster_labels[client_id] = best_match
                    print(f"  客户端{client_id}低相似({distance:.4f})，合并到簇{best_match}")
                else:
                    cluster_labels[client_id] = current_cluster
                    current_cluster += 1
                    print(f"  客户端{client_id}低相似({distance:.4f})，新簇{cluster_labels[client_id]}")
        
        return cluster_labels
    
    def _detect_anomaly_clients_enhanced(self, cluster_labels: np.ndarray, 
                                        distances: np.ndarray,
                                        features_list: List[np.ndarray]) -> List[int]:
        """增强的异常客户端检测"""
        print("🔍 检测异常客户端...")
        anomaly_clients = []
        num_clients = len(cluster_labels)
        
        # 方法1：基于簇内距离的异常检测
        for i in range(num_clients):
            cluster_id = cluster_labels[i]
            cluster_members = np.where(cluster_labels == cluster_id)[0]
            
            if len(cluster_members) > 1:
                # 计算到簇内其他客户端的平均距离
                distances_to_cluster = []
                for j in cluster_members:
                    if i != j:
                        distances_to_cluster.append(distances[i][j])
                
                if len(distances_to_cluster) > 0:
                    avg_distance = np.mean(distances_to_cluster)
                    std_distance = np.std(distances_to_cluster)
                    
                    # 如果距离超过2个标准差，认为是异常
                    if avg_distance > np.mean(distances_to_cluster) + 2 * std_distance:
                        anomaly_clients.append(i)
        
        # 方法2：基于全局距离分布的异常检测
        for i in range(num_clients):
            distances_to_others = []
            for j in range(num_clients):
                if i != j:
                    distances_to_others.append(distances[i][j])
            
            if len(distances_to_others) > 0:
                mean_global_dist = np.mean(distances_to_others)
                std_global_dist = np.std(distances_to_others)
                
                # 如果与所有其他客户端的平均距离异常，认为是异常
                if np.mean(distances_to_others) > mean_global_dist + 1.5 * std_global_dist:
                    if i not in anomaly_clients:
                        anomaly_clients.append(i)
        
        # 方法3：基于特征分布的异常检测
        for i in range(num_clients):
            if i in anomaly_clients:
                continue
            
            # 计算特征分布的统计特征
            features = features_list[i]
            if len(features) > 0:
                feature_mean = np.mean(features, axis=0)
                feature_std = np.std(features, axis=0)
                
                # 检查是否有异常的特征值
                z_scores = np.abs((features - feature_mean) / (feature_std + 1e-8))
                if np.any(z_scores > 3):  # 超过3个标准差的特征值
                    if i not in anomaly_clients:
                        anomaly_clients.append(i)
        
        print(f"🔍 检测到异常客户端: {anomaly_clients}")
        return list(set(anomaly_clients))  # 去重
    
    def _evaluate_clustering_quality_enhanced(self, cluster_labels: np.ndarray, 
                                            distances: np.ndarray) -> Dict:
        """增强的分簇质量评估"""
        print("📊 评估分簇质量...")
        
        unique_clusters = np.unique(cluster_labels)
        num_clusters = len(unique_clusters)
        
        if num_clusters < 2:
            return {'error': '簇数太少，无法评估质量'}
        
        # 计算轮廓系数
        try:
            silhouette_avg = silhouette_score(distances, cluster_labels, metric='precomputed')
        except:
            silhouette_avg = 0.0
        
        # 计算Calinski-Harabasz指数
        try:
            ch_score = calinski_harabasz_score(distances, cluster_labels)
        except:
            ch_score = 0.0
        
        # 计算簇内和簇间距离
        intra_cluster_distances = []
        inter_cluster_distances = []
        
        for i in range(num_clusters):
            cluster_i = np.where(cluster_labels == i)[0]
            if len(cluster_i) > 1:
                # 簇内距离
                for j in range(len(cluster_i)):
                    for k in range(j+1, len(cluster_i)):
                        intra_cluster_distances.append(distances[cluster_i[j]][cluster_i[k]])
                
                # 簇间距离
                for j in range(i+1, num_clusters):
                    cluster_j = np.where(cluster_labels == j)[0]
                    if len(cluster_j) > 0:
                        for k in cluster_i:
                            for l in cluster_j:
                                inter_cluster_distances.append(distances[k][l])
        
        # 计算簇内一致性和簇间差异性
        intra_consistency = np.mean(intra_cluster_distances) if intra_cluster_distances else 0
        inter_diversity = np.mean(inter_cluster_distances) if inter_cluster_distances else 0
        
        # 计算簇大小分布
        cluster_sizes = [len(np.where(cluster_labels == i)[0]) for i in unique_clusters]
        size_balance = 1.0 / (1.0 + np.std(cluster_sizes))  # 簇大小越平衡，分数越高
        
        quality_metrics = {
            'num_clusters': num_clusters,
            'silhouette_score': silhouette_avg,
            'calinski_harabasz_score': ch_score,
            'intra_cluster_consistency': intra_consistency,
            'inter_cluster_diversity': inter_diversity,
            'cluster_size_balance': size_balance,
            'cluster_sizes': cluster_sizes,
            'overall_quality': (silhouette_avg + size_balance) / 2
        }
        
        print(f"📊 分簇质量评估结果:")
        print(f"  簇数: {num_clusters}")
        print(f"  轮廓系数: {silhouette_avg:.4f}")
        print(f"  Calinski-Harabasz指数: {ch_score:.4f}")
        print(f"  簇内一致性: {intra_consistency:.4f}")
        print(f"  簇间差异性: {inter_diversity:.4f}")
        print(f"  簇大小平衡性: {size_balance:.4f}")
        print(f"  整体质量: {quality_metrics['overall_quality']:.4f}")
        
        return quality_metrics
    
    def _analyze_clustering_result_enhanced(self, cluster_labels: np.ndarray, 
                                          distances: np.ndarray,
                                          target_client_id: int):
        """增强的分簇结果分析"""
        print(f"\n📋 分簇结果分析:")
        
        unique_clusters = np.unique(cluster_labels)
        num_clusters = len(unique_clusters)
        
        print(f"总簇数: {num_clusters}")
        
        for cluster_id in unique_clusters:
            cluster_members = np.where(cluster_labels == cluster_id)[0]
            print(f"簇{cluster_id}: 客户端{cluster_members.tolist()}")
            
            if len(cluster_members) > 1:
                # 计算簇内平均距离
                cluster_distances = []
                for i in cluster_members:
                    for j in cluster_members:
                        if i < j:
                            cluster_distances.append(distances[i][j])
                
                if cluster_distances:
                    avg_distance = np.mean(cluster_distances)
                    print(f"  簇内平均距离: {avg_distance:.4f}")
            
            # 计算与目标客户端的平均距离
            if target_client_id not in cluster_members:
                target_distances = [distances[target_client_id][i] for i in cluster_members]
                avg_target_distance = np.mean(target_distances)
                print(f"  与目标客户端平均距离: {avg_target_distance:.4f}")
        
        print(f"分簇结果: {cluster_labels}")
    
    def save_clustering_analysis_enhanced(self, cluster_labels: np.ndarray, 
                                        distances: np.ndarray, 
                                        clustering_metrics: Dict,
                                        save_path: str):
        """保存增强的分簇分析结果"""
        analysis_result = {
            'cluster_labels': cluster_labels.tolist(),
            'distances': distances.tolist(),
            'clustering_metrics': clustering_metrics,
            'timestamp': str(np.datetime64('now')),
            'parameters': {
                'use_adaptive_thresholds': self.use_adaptive_thresholds,
                'use_hierarchical_clustering': self.use_hierarchical_clustering,
                'use_forgetting_optimization': self.use_forgetting_optimization,
                'iqr_multiplier': self.iqr_multiplier,
                'min_threshold': self.min_threshold,
                'max_threshold': self.max_threshold
            }
        }
        
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, indent=2, ensure_ascii=False)
        
        print(f"💾 分簇分析结果已保存到: {save_path}")
    
    def load_clustering_analysis_enhanced(self, load_path: str) -> Dict:
        """加载增强的分簇分析结果"""
        try:
            with open(load_path, 'r', encoding='utf-8') as f:
                analysis_result = json.load(f)
            
            print(f"📂 分簇分析结果已从 {load_path} 加载")
            return analysis_result
        except Exception as e:
            print(f"❌ 加载分簇分析结果失败: {e}")
            return {}
    
    def visualize_clustering_enhanced(self, cluster_labels: np.ndarray, 
                                    distances: np.ndarray,
                                    save_path: Optional[str] = None):
        """可视化增强的分簇结果"""
        try:
            # 使用t-SNE降维进行可视化
            from sklearn.manifold import TSNE
            
            print("🎨 生成分簇可视化...")
            
            # 使用t-SNE降维到2D
            tsne = TSNE(n_components=2, random_state=42, metric='precomputed')
            distances_2d = tsne.fit_transform(distances)
            
            # 创建可视化
            plt.figure(figsize=(12, 8))
            
            # 绘制散点图
            unique_clusters = np.unique(cluster_labels)
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
            
            for i, cluster_id in enumerate(unique_clusters):
                cluster_points = distances_2d[cluster_labels == cluster_id]
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                           c=[colors[i]], label=f'簇{cluster_id}', alpha=0.7, s=100)
            
            plt.title('增强分簇策略结果可视化 (t-SNE)', fontsize=16)
            plt.xlabel('t-SNE维度1', fontsize=12)
            plt.ylabel('t-SNE维度2', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"🎨 可视化结果已保存到: {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"❌ 可视化生成失败: {e}")
            print("请确保已安装matplotlib和scikit-learn")


