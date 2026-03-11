import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.stats import wasserstein_distance, pearsonr
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import time
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class EnhancedClusteringStrategy:
    """
    增强的分簇策略类，集成多种先进的分簇算法和优化策略
    """
    
    def __init__(self, 
                 distance_metric='wasserstein',
                 clustering_method='hierarchical',
                 adaptive_threshold=True,
                 multi_level=True,
                 forget_optimization=True,
                 smart_weight_balancing=True,  # 新增：智能权重平衡
                 personalized_protection=True,  # 新增：个性化保护
                 forget_effect_prediction=True, # 新增：遗忘效果预测
                 advanced_multi_objective=True, # 第三阶段：高级多目标优化
                 real_time_adjustment=True,     # 第三阶段：实时策略调整
                 performance_prediction=True,   # 第三阶段：性能预测和预警
                 client_monitoring=True):       # 第三阶段：增强客户端监控
        """
        初始化增强分簇策略
        
        Args:
            distance_metric: 距离度量方法
            clustering_method: 分簇算法
            adaptive_threshold: 是否使用自适应阈值
            multi_level: 是否使用多层次分簇
            forget_optimization: 是否使用遗忘效果优化
            smart_weight_balancing: 是否使用智能权重平衡
            personalized_protection: 是否使用个性化保护
            forget_effect_prediction: 是否使用遗忘效果预测
            advanced_multi_objective: 是否使用高级多目标优化
            real_time_adjustment: 是否使用实时策略调整
            performance_prediction: 是否使用性能预测和预警
            client_monitoring: 是否使用增强客户端监控
        """
        self.distance_metric = distance_metric
        self.clustering_method = clustering_method
        self.adaptive_threshold = adaptive_threshold
        self.multi_level = multi_level
        self.forget_optimization = forget_optimization
        self.smart_weight_balancing = smart_weight_balancing
        self.personalized_protection = personalized_protection
        self.forget_effect_prediction = forget_effect_prediction
        
        # 第三阶段功能开关
        self.advanced_multi_objective = advanced_multi_objective
        self.real_time_adjustment = real_time_adjustment
        self.performance_prediction = performance_prediction
        self.client_monitoring = client_monitoring
        
        # 新增：联合优化迭代次数参数，防止AttributeError
        self.optimization_iterations = 10  # 可根据需要调整默认值
        # 新增：联合优化收敛阈值参数，防止AttributeError
        self.convergence_threshold = 1e-4  # 可根据需要调整默认值
        # 新增：分簇可视化开关，防止AttributeError
        self.visualization_enabled = False  # 可根据需要调整默认值
        
        # 智能权重平衡相关参数
        self.weight_history = []
        self.performance_history = {}
        self.client_protection_levels = {}
        self.adaptive_weights = {
            'distribution_similarity': 0.4,
            'gradient_similarity': 0.3,
            'forget_difficulty': 0.2,
            'client_protection': 0.1
        }
        
        # 个性化保护参数
        self.protection_strategies = {
            'high_protection': {'threshold': 0.8, 'weight': 0.9},
            'medium_protection': {'threshold': 0.6, 'weight': 0.7},
            'low_protection': {'threshold': 0.4, 'weight': 0.5}
        }
        
        # 遗忘效果预测参数
        self.forget_impact_model = None
        self.impact_prediction_cache = {}
        
        # 兼容原有代码的属性
        self.args = None  # 兼容原有代码
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_adaptive_thresholds = adaptive_threshold
        self.use_hierarchical_clustering = multi_level
        self.use_forgetting_optimization = forget_optimization
        
        # 自适应阈值参数
        self.iqr_multiplier = 1.5
        self.min_threshold = 0.1
        self.max_threshold = 0.9
        
        # 多层次分簇参数
        self.coarse_cluster_ratio = 0.3
        self.fine_cluster_ratio = 0.6
        
        # 遗忘效果优化参数
        self.forgetting_weight = 0.4
        self.protection_weight = 0.3
        self.utility_weight = 0.3
        
        # 第二阶段：多层次分簇策略优化参数
        self.forget_difficulty_analysis = True  # 遗忘难度分析
        self.dynamic_cluster_balancing = True   # 动态簇间平衡
        self.joint_optimization = True          # 联合优化
        
        # 第三阶段：高级多目标优化参数
        self.optimization_algorithm = 'nsga2'  # 优化算法：NSGA-II
        self.population_size = 50              # 种群大小
        self.generation_limit = 100            # 最大代数
        self.crossover_rate = 0.8              # 交叉概率
        self.mutation_rate = 0.1               # 变异概率
        self.tournament_size = 3               # 锦标赛选择大小
        self.elite_size = 5                    # 精英个体数量
        
        # 第三阶段：实时策略调整参数
        self.adjustment_threshold = 0.1        # 调整阈值
        self.performance_window = 10           # 性能监控窗口
        self.adjustment_history = []           # 调整历史记录
        self.adjustment_cooldown = 5           # 调整冷却期
        
        # 第三阶段：性能预测和预警参数
        self.prediction_model = None           # 预测模型
        self.prediction_history = []           # 预测历史
        self.warning_thresholds = {            # 预警阈值
            'performance_decline': 0.05,      # 性能下降5%
            'stability_threshold': 0.3,       # 稳定性阈值
            'convergence_threshold': 0.001     # 收敛阈值
        }
        
        # 第三阶段：增强客户端监控参数
        self.client_performance_tracker = {}   # 客户端性能跟踪器
        self.performance_metrics = {           # 全局性能指标
            'accuracy': [],
            'loss': [],
            'convergence_rate': [],
            'stability': []
        }
        self.intervention_history = []         # 干预历史记录
        self.monitoring_frequency = 5          # 监控频率（每N个epoch）
        
        # 性能优化缓存
        self.distance_cache = {}
        self.clustering_cache = {}
        self.optimization_cache = {}
        
        print(f"增强分簇策略初始化完成")
        print(f"智能权重平衡: {smart_weight_balancing}")
        print(f"个性化保护: {personalized_protection}")
        print(f"遗忘效果预测: {forget_effect_prediction}")
        print(f"遗忘难度分析: {self.forget_difficulty_analysis}")
        print(f"动态簇间平衡: {self.dynamic_cluster_balancing}")
        print(f"联合优化: {self.joint_optimization}")
        print(f"高级多目标优化: {self.advanced_multi_objective}")
        print(f"实时策略调整: {self.real_time_adjustment}")
        print(f"性能预测与预警: {self.performance_prediction}")
        print(f"客户端性能监控: {self.client_monitoring}")
    
    def cluster_clients_enhanced(self, features_list, distances, target_client_id=0):
        """
        增强的客户端分簇主函数
        """
        print(f"\n============= 开始增强分簇策略 =============")
        
        # 1. 基于数据分布统计的自适应分簇
        if self.adaptive_threshold:
            print("1. 计算自适应阈值...")
            adaptive_thresholds = self._compute_adaptive_thresholds_enhanced(distances)
            print(f"自适应阈值: {adaptive_thresholds}")
        else:
            adaptive_thresholds = self._get_default_thresholds()
        
        # 2. 多层次分簇策略
        if self.multi_level:
            print("2. 执行多层次分簇策略...")
            cluster_labels = self._multi_level_clustering_strategy(
                distances, target_client_id, adaptive_thresholds
            )
        else:
            print("2. 使用传统分簇方法...")
            cluster_labels = self._traditional_clustering(distances, target_client_id, adaptive_thresholds)
        
        # 3. 基于遗忘效果的分簇优化
        if self.forget_optimization:
            print("3. 基于遗忘效果优化分簇...")
            cluster_labels = self._optimize_clusters_for_forgetting_enhanced(
                cluster_labels, distances, target_client_id, adaptive_thresholds
            )
        
        # 4. 第二阶段：多层次分簇策略优化
        if self.forget_difficulty_analysis or self.dynamic_cluster_balancing or self.joint_optimization:
            print("4. 执行第二阶段分簇策略优化...")
            
            # 4.1 遗忘难度分析
            if self.forget_difficulty_analysis:
                difficulty_analysis = self.analyze_forget_difficulty(distances, target_client_id)
                if difficulty_analysis:
                    print(f"  遗忘难度分析结果:")
                    for client_id, category in difficulty_analysis['categories'].items():
                        score = difficulty_analysis['scores'][client_id]
                        print(f"    客户端 {client_id}: {category} (评分: {score:.4f})")
            
            # 4.2 动态簇间平衡
            if self.dynamic_cluster_balancing:
                cluster_labels = self.balance_clusters_dynamically(
                    cluster_labels, distances, target_client_id
                )
            
            # 4.3 联合优化
            if self.joint_optimization:
                cluster_labels = self.joint_optimize_clustering(
                    cluster_labels, distances, target_client_id
                )
        
        # 5. 异常客户端检测
        print("5. 检测异常客户端...")
        anomaly_clients = self._detect_anomaly_clients_enhanced(
            cluster_labels, distances, target_client_id
        )
        
        # 6. 分簇质量评估
        print("6. 评估分簇质量...")
        quality_metrics = self._evaluate_clustering_quality_enhanced(
            cluster_labels, distances, target_client_id
        )
        
        # 7. 可视化分析
        if self.visualization_enabled:
            print("7. 生成可视化分析...")
            self._visualize_clustering_results_enhanced(
                cluster_labels, distances, target_client_id, quality_metrics
            )
        
        # 8. 保存分析结果
        print("8. 保存分析结果...")
        self._save_analysis_results_enhanced(
            cluster_labels, anomaly_clients, quality_metrics, target_client_id
        )
        
        # 9. 最终簇标签重映射，确保簇标签连续
        print("9. 最终簇标签重映射...")
        final_cluster_labels = self._remap_cluster_labels_with_isolation(cluster_labels, target_client_id)
        
        # 10. 最终隔离性验证（关键步骤）
        print("10. 最终隔离性验证...")
        final_cluster_labels = self._ensure_target_client_isolation(final_cluster_labels, distances, target_client_id)
        
        # 11. 输出最终分簇结果和验证信息
        print("\n============= 分簇结果最终验证 =============")
        target_cluster = final_cluster_labels[target_client_id]
        same_cluster_clients = np.where(final_cluster_labels == target_cluster)[0]
        
        if len(same_cluster_clients) == 1 and same_cluster_clients[0] == target_client_id:
            print(f"✅ 最终验证通过：目标客户端{target_client_id}独立成簇{target_cluster}")
            print(f"✅ 分簇策略执行成功！")
        else:
            print(f"❌ 最终验证失败：目标客户端{target_client_id}簇{target_cluster}包含其他客户端{same_cluster_clients}")
            print(f"⚠️  强制修复分簇结果...")
            final_cluster_labels = self._force_fix_target_client_isolation(final_cluster_labels, distances, target_client_id)
        
        print(f"\n============= 最终分簇结果 =============")
        print(f"分簇结果: {final_cluster_labels}")
        print(f"目标客户端{target_client_id}所在簇: {final_cluster_labels[target_client_id]}")
        print(f"异常客户端: {anomaly_clients}")
        
        return final_cluster_labels, anomaly_clients
    
    def _force_fix_target_client_isolation(self, cluster_labels, distances, target_client_id):
        """
        强制修复目标客户端隔离性，同时保持合理的分簇结构
        
        Args:
            cluster_labels: 当前分簇标签
            distances: 距离矩阵
            target_client_id: 目标客户端ID
            
        Returns:
            fixed_labels: 强制修复后的分簇标签
        """
        print(f"执行强制修复：确保目标客户端{target_client_id}独立成簇...")
        
        fixed_labels = cluster_labels.copy()
        
        # 找到与目标客户端在同一簇的其他客户端
        target_cluster = fixed_labels[target_client_id]
        same_cluster_clients = np.where(fixed_labels == target_cluster)[0]
        other_clients_in_same_cluster = [cid for cid in same_cluster_clients if cid != target_client_id]
        
        if other_clients_in_same_cluster:
            print(f"  发现{len(other_clients_in_same_cluster)}个客户端与目标客户端在同一簇")
            
            # 智能重新分配这些客户端，保持合理的分簇结构
            fixed_labels = self._redistribute_clients_intelligently(
                fixed_labels, distances, target_client_id, other_clients_in_same_cluster
            )
            
            # 重新映射簇标签，确保连续性
            fixed_labels = self._remap_cluster_labels_with_isolation(fixed_labels, target_client_id)
            
            print(f"  强制修复完成")
        else:
            print(f"  目标客户端已独立成簇，无需强制修复")
        
        return fixed_labels
    
    def _compute_adaptive_thresholds_enhanced(self, distances):
        """增强的自适应阈值计算，集成智能权重平衡"""
        if self.smart_weight_balancing:
            return self._compute_smart_adaptive_thresholds(distances)
        else:
            return self._compute_basic_adaptive_thresholds(distances)
    
    def _compute_smart_adaptive_thresholds(self, distances):
        """
        智能自适应阈值计算，考虑客户端保护需求
        """
        print("  使用智能自适应阈值计算...")
        
        # 计算基础统计特征
        distances_flat = distances.flatten()
        q1, q3 = np.percentile(distances_flat, [25, 75])
        iqr = q3 - q1
        
        # 基础阈值
        base_threshold = q3 + 1.5 * iqr
        
        # 应用个性化保护策略
        protected_thresholds = self._apply_personalized_protection(
            distances, base_threshold
        )
        
        # 动态调整权重
        self._update_adaptive_weights(distances)
        
        # 计算最终阈值
        final_thresholds = self._compute_final_thresholds(
            distances, protected_thresholds
        )
        
        print(f"  智能阈值计算完成，基础阈值: {base_threshold:.4f}")
        
        # 返回兼容原有代码的字典格式
        return {
            'high': np.min(final_thresholds),
            'medium': np.median(final_thresholds),
            'low': np.max(final_thresholds),
            'smart_thresholds': final_thresholds,
            'statistics': {
                'mean': np.mean(distances_flat),
                'std': np.std(distances_flat),
                'q25': q1,
                'q50': np.median(distances_flat),
                'q75': q3,
                'iqr': iqr
            }
        }
    
    def _compute_basic_adaptive_thresholds(self, distances):
        """
        基础的自适应阈值计算，保持向后兼容性
        """
        print("  使用基础自适应阈值计算...")
        
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
        iqr_multiplier = 1.5
        min_threshold, max_threshold = 0.1, 0.9
        
        lower_bound = max(min_threshold, q25 - iqr_multiplier * iqr)
        upper_bound = min(max_threshold, q75 + iqr_multiplier * iqr)
        
        # 基于分布特征设置分簇阈值
        high_similarity_threshold = max(0.2, min(0.6, lower_bound))
        medium_similarity_threshold = max(0.4, min(0.8, q50))
        low_similarity_threshold = max(0.6, min(0.9, upper_bound))
        
        # 计算阈值置信度
        confidence_scores = self._compute_threshold_confidence(upper_triangle, q25, q75, iqr)
        
        print(f"  基础阈值计算完成:")
        print(f"    - 高相似度: {high_similarity_threshold:.4f}")
        print(f"    - 中等相似度: {medium_similarity_threshold:.4f}")
        print(f"    - 低相似度: {low_similarity_threshold:.4f}")
        
        return {
            'high': high_similarity_threshold,
            'medium': medium_similarity_threshold,
            'low': low_similarity_threshold,
            'confidence': confidence_scores,
            'statistics': {
                'mean': mean_dist,
                'std': std_dist,
                'q25': q25,
                'q50': q50,
                'q75': q75,
                'iqr': iqr
            }
        }
    
    def _compute_threshold_confidence(self, distances, q25, q75, iqr):
        """计算阈值置信度"""
        # 基于数据分布的一致性计算置信度
        within_iqr = np.sum((distances >= q25) & (distances <= q75))
        total = len(distances)
        iqr_ratio = within_iqr / total
        
        # 基于分布的正态性计算置信度
        from scipy.stats import shapiro
        try:
            _, p_value = shapiro(distances)
            normality_confidence = min(1.0, p_value * 2)  # 转换为置信度
        except:
            normality_confidence = 0.5
        
        # 综合置信度
        high_confidence = (iqr_ratio + normality_confidence) / 2
        medium_confidence = high_confidence * 0.9
        low_confidence = high_confidence * 0.8
        
        return {
            'high': high_confidence,
            'medium': medium_confidence,
            'low': low_confidence
        }
    
    def _apply_personalized_protection(self, distances, base_threshold):
        """
        应用个性化保护策略
        """
        if not self.personalized_protection:
            return np.full(distances.shape, base_threshold)
        
        print("    应用个性化保护策略...")
        
        # 分析每个客户端的保护需求
        client_protection_needs = self._analyze_client_protection_needs(distances)
        
        # 为每个客户端分配保护策略
        protected_thresholds = np.full(distances.shape, base_threshold)
        
        for i in range(distances.shape[0]):
            for j in range(distances.shape[1]):
                if i != j:  # 不对角线元素
                    protection_level = self._get_client_protection_level(
                        i, client_protection_needs
                    )
                    protected_thresholds[i, j] = self._adjust_threshold_by_protection(
                        base_threshold, protection_level
                    )
        
        return protected_thresholds
    
    def _analyze_client_protection_needs(self, distances):
        """
        分析客户端的保护需求
        """
        # 计算每个客户端的平均距离
        avg_distances = np.mean(distances, axis=1)
        
        # 基于距离分布确定保护需求
        protection_needs = {}
        for i, avg_dist in enumerate(avg_distances):
            if avg_dist < np.percentile(avg_distances, 25):
                protection_needs[i] = 'high_protection'
            elif avg_dist < np.percentile(avg_distances, 75):
                protection_needs[i] = 'medium_protection'
            else:
                protection_needs[i] = 'low_protection'
        
        return protection_needs
    
    def _get_client_protection_level(self, client_id, protection_needs):
        """
        获取客户端的保护级别
        """
        if client_id in protection_needs:
            return protection_needs[client_id]
        else:
            return 'medium_protection'
    
    def _adjust_threshold_by_protection(self, base_threshold, protection_level):
        """
        根据保护级别调整阈值
        """
        strategy = self.protection_strategies.get(protection_level, 
                                                self.protection_strategies['medium_protection'])
        
        # 高保护级别降低阈值（更严格的分簇）
        # 低保护级别提高阈值（更宽松的分簇）
        if protection_level == 'high_protection':
            return base_threshold * 0.8
        elif protection_level == 'low_protection':
            return base_threshold * 1.2
        else:
            return base_threshold
    
    def _update_adaptive_weights(self, distances):
        """
        动态更新自适应权重
        """
        if not self.smart_weight_balancing:
            return
        
        # 分析当前距离分布
        current_stats = self._analyze_distance_distribution(distances)
        
        # 更新权重历史
        self.weight_history.append(self.adaptive_weights.copy())
        
        # 根据分布特征调整权重
        if current_stats['variance'] > 0.1:  # 高方差
            self.adaptive_weights['distribution_similarity'] *= 1.1
            self.adaptive_weights['gradient_similarity'] *= 0.9
        else:  # 低方差
            self.adaptive_weights['distribution_similarity'] *= 0.9
            self.adaptive_weights['gradient_similarity'] *= 1.1
        
        # 归一化权重
        total_weight = sum(self.adaptive_weights.values())
        for key in self.adaptive_weights:
            self.adaptive_weights[key] /= total_weight
        
        print(f"    权重已更新: {self.adaptive_weights}")
    
    def _analyze_distance_distribution(self, distances):
        """
        分析距离分布特征
        """
        distances_flat = distances.flatten()
        return {
            'mean': np.mean(distances_flat),
            'variance': np.var(distances_flat),
            'skewness': self._calculate_skewness(distances_flat),
            'kurtosis': self._calculate_kurtosis(distances_flat)
        }
    
    def _calculate_skewness(self, data):
        """计算偏度"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data):
        """计算峰度"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _compute_final_thresholds(self, distances, protected_thresholds):
        """
        计算最终阈值，考虑遗忘效果预测
        """
        if self.forget_effect_prediction:
            # 预测遗忘效果并调整阈值
            impact_predictions = self._predict_forget_impact(distances)
            final_thresholds = self._adjust_thresholds_by_impact(
                protected_thresholds, impact_predictions
            )
        else:
            final_thresholds = protected_thresholds
        
        return final_thresholds
    
    def _predict_forget_impact(self, distances):
        """
        预测遗忘对各个客户端的影响
        """
        print("    预测遗忘效果影响...")
        
        # 基于距离矩阵预测影响
        impact_predictions = {}
        
        for i in range(distances.shape[0]):
            # 计算与其他客户端的平均距离
            other_distances = [distances[i, j] for j in range(distances.shape[1]) if i != j]
            avg_distance = np.mean(other_distances)
            
            # 基于距离预测影响程度
            if avg_distance < 0.3:
                impact_predictions[i] = 'high_impact'  # 高影响
            elif avg_distance < 0.6:
                impact_predictions[i] = 'medium_impact'  # 中等影响
            else:
                impact_predictions[i] = 'low_impact'  # 低影响
        
        # 缓存预测结果
        self.impact_prediction_cache = impact_predictions
        
        print(f"    遗忘影响预测完成: {impact_predictions}")
        return impact_predictions
    
    def _adjust_thresholds_by_impact(self, thresholds, impact_predictions):
        """
        根据遗忘影响预测调整阈值
        """
        adjusted_thresholds = thresholds.copy()
        
        for client_id, impact_level in impact_predictions.items():
            if impact_level == 'high_impact':
                # 高影响客户端使用更严格的阈值
                adjusted_thresholds[client_id, :] *= 0.7
            elif impact_level == 'low_impact':
                # 低影响客户端使用更宽松的阈值
                adjusted_thresholds[client_id, :] *= 1.3
        
        return adjusted_thresholds
    
    def monitor_client_performance(self, client_id, performance_metrics):
        """
        监控客户端性能，用于智能权重平衡
        
        Args:
            client_id: 客户端ID
            performance_metrics: 性能指标字典，包含准确率、AUC等
        """
        if not self.smart_weight_balancing:
            return
        
        # 记录性能历史
        if client_id not in self.performance_history:
            self.performance_history[client_id] = []
        
        self.performance_history[client_id].append(performance_metrics)
        
        # 分析性能趋势
        performance_trend = self._analyze_performance_trend(client_id)
        
        # 根据性能趋势调整保护策略
        if performance_trend == 'declining':
            self._increase_client_protection(client_id)
        elif performance_trend == 'improving':
            self._decrease_client_protection(client_id)
        
        print(f"客户端 {client_id} 性能监控完成，趋势: {performance_trend}")
    
    def _analyze_performance_trend(self, client_id):
        """
        分析客户端性能趋势
        """
        if client_id not in self.performance_history:
            return 'stable'
        
        history = self.performance_history[client_id]
        if len(history) < 3:
            return 'stable'
        
        # 提取准确率变化
        accuracies = [metrics.get('accuracy', 0) for metrics in history[-3:]]
        
        # 计算变化率
        if len(accuracies) >= 2:
            change_rate = (accuracies[-1] - accuracies[0]) / accuracies[0]
            
            if change_rate < -0.1:  # 下降超过10%
                return 'declining'
            elif change_rate > 0.1:  # 上升超过10%
                return 'improving'
            else:
                return 'stable'
        
        return 'stable'
    
    def _increase_client_protection(self, client_id):
        """
        增加客户端保护级别
        """
        if client_id in self.client_protection_levels:
            current_level = self.client_protection_levels[client_id]
            
            # 提升保护级别
            if current_level == 'low_protection':
                self.client_protection_levels[client_id] = 'medium_protection'
            elif current_level == 'medium_protection':
                self.client_protection_levels[client_id] = 'high_protection'
        else:
            self.client_protection_levels[client_id] = 'medium_protection'
        
        print(f"客户端 {client_id} 保护级别已提升")
    
    def _decrease_client_protection(self, client_id):
        """
        降低客户端保护级别
        """
        if client_id in self.client_protection_levels:
            current_level = self.client_protection_levels[client_id]
            
            # 降低保护级别
            if current_level == 'high_protection':
                self.client_protection_levels[client_id] = 'medium_protection'
            elif current_level == 'medium_protection':
                self.client_protection_levels[client_id] = 'low_protection'
        else:
            self.client_protection_levels[client_id] = 'low_protection'
        
        print(f"客户端 {client_id} 保护级别已降低")
    
    def get_optimization_report(self):
        """
        获取优化报告
        """
        report = {
            'smart_weight_balancing': {
                'enabled': self.smart_weight_balancing,
                'current_weights': self.adaptive_weights.copy(),
                'weight_history_length': len(self.weight_history)
            },
            'personalized_protection': {
                'enabled': self.personalized_protection,
                'client_protection_levels': self.client_protection_levels.copy(),
                'protection_strategies': self.protection_strategies
            },
            'forget_effect_prediction': {
                'enabled': self.forget_effect_prediction,
                'impact_predictions': self.impact_prediction_cache.copy()
            },
            'performance_monitoring': {
                'monitored_clients': list(self.performance_history.keys()),
                'total_performance_records': sum(len(history) for history in self.performance_history.values())
            }
        }
        
        return report
    
    def reset_optimization_state(self):
        """
        重置优化状态
        """
        self.weight_history = []
        self.performance_history = {}
        self.client_protection_levels = {}
        self.impact_prediction_cache = {}
        
        # 重置权重到默认值
        self.adaptive_weights = {
            'distribution_similarity': 0.4,
            'gradient_similarity': 0.3,
            'forget_difficulty': 0.2,
            'client_protection': 0.1
        }
        
        print("优化状态已重置")
    
    # ==================== 第二阶段：多层次分簇策略优化 ====================
    
    def analyze_forget_difficulty(self, distances, target_client_id=0):
        """
        分析遗忘难度，为智能分簇提供依据
        
        Args:
            distances: 距离矩阵
            target_client_id: 目标客户端ID
            
        Returns:
            difficulty_scores: 各客户端的遗忘难度评分
        """
        if not self.forget_difficulty_analysis:
            return None
        
        print("  分析遗忘难度...")
        
        difficulty_scores = {}
        
        for client_id in range(len(distances)):
            if client_id == target_client_id:
                difficulty_scores[client_id] = 0.0  # 目标客户端
                continue
            
            # 计算与目标客户端的相似度
            similarity_to_target = 1.0 - distances[client_id, target_client_id]
            
            # 计算与其他客户端的平均相似度
            other_similarities = []
            for other_id in range(len(distances)):
                if other_id != client_id and other_id != target_client_id:
                    other_similarities.append(1.0 - distances[client_id, other_id])
            
            avg_other_similarity = np.mean(other_similarities) if other_similarities else 0.0
            
            # 遗忘难度评分：与目标相似度越高，遗忘难度越大
            # 与其他客户端相似度越高，遗忘难度越小
            difficulty_score = similarity_to_target * 0.7 + (1.0 - avg_other_similarity) * 0.3
            
            difficulty_scores[client_id] = difficulty_score

        # 初始化难度阈值（兜底），避免属性缺失
        if not hasattr(self, 'difficulty_thresholds') or self.difficulty_thresholds is None:
            score_values = [s for cid, s in difficulty_scores.items() if cid != target_client_id]
            if len(score_values) == 0:
                easy_t, med_t = 0.33, 0.66
            else:
                easy_t = float(np.percentile(score_values, 33))
                med_t = float(np.percentile(score_values, 66))
            self.difficulty_thresholds = {
                'easy': easy_t,
                'medium': med_t
            }

        # 分类遗忘难度
        difficulty_categories = {}
        for client_id, score in difficulty_scores.items():
            if score < self.difficulty_thresholds['easy']:
                difficulty_categories[client_id] = 'easy'
            elif score < self.difficulty_thresholds['medium']:
                difficulty_categories[client_id] = 'medium'
            else:
                difficulty_categories[client_id] = 'hard'
        
        print(f"    遗忘难度分析完成:")
        print(f"      容易遗忘: {[k for k, v in difficulty_categories.items() if v == 'easy']}")
        print(f"      中等难度: {[k for k, v in difficulty_categories.items() if v == 'medium']}")
        print(f"      难以遗忘: {[k for k, v in difficulty_categories.items() if v == 'hard']}")
        
        return {
            'scores': difficulty_scores,
            'categories': difficulty_categories
        }
    
    def balance_clusters_dynamically(self, cluster_labels, distances, target_client_id=0):
        """
        动态平衡簇间分布，确保各簇的平衡性
        
        Args:
            cluster_labels: 当前分簇标签
            distances: 距离矩阵
            target_client_id: 目标客户端ID
            
        Returns:
            balanced_labels: 平衡后的分簇标签
        """
        if not self.dynamic_cluster_balancing:
            return cluster_labels
        
        print("  执行动态簇间平衡...")
        
        # 分析当前分簇的平衡性
        cluster_analysis = self._analyze_cluster_balance(cluster_labels, distances)
        
        # 计算平衡评分
        balance_score = self._calculate_balance_score(cluster_analysis)
        
        print(f"    当前平衡评分: {balance_score:.4f}")
        
        # 如果平衡性良好，直接返回
        if balance_score > 0.8:
            print("    簇间平衡性良好，无需调整")
            return cluster_labels
        
        # 执行簇间平衡优化
        balanced_labels = self._optimize_cluster_balance(
            cluster_labels, distances, target_client_id, cluster_analysis
        )
        
        # 重新分析平衡性
        new_analysis = self._analyze_cluster_balance(balanced_labels, distances)
        new_balance_score = self._calculate_balance_score(new_analysis)
        
        print(f"    优化后平衡评分: {new_balance_score:.4f}")
        print(f"    平衡性改善: {new_balance_score - balance_score:.4f}")
        
        return balanced_labels
    
    def _analyze_cluster_balance(self, cluster_labels, distances):
        """
        分析分簇的平衡性
        """
        unique_clusters = np.unique(cluster_labels)
        cluster_analysis = {}
        
        for cluster_id in unique_clusters:
            cluster_members = np.where(cluster_labels == cluster_id)[0]
            
            # 簇内相似度
            intra_similarities = []
            for i in cluster_members:
                for j in cluster_members:
                    if i < j:
                        intra_similarities.append(1.0 - distances[i, j])
            
            # 簇间差异度
            inter_differences = []
            for other_cluster_id in unique_clusters:
                if other_cluster_id != cluster_id:
                    other_members = np.where(cluster_labels == other_cluster_id)[0]
                    for i in cluster_members:
                        for j in other_members:
                            inter_differences.append(distances[i, j])
            
            cluster_analysis[cluster_id] = {
                'size': len(cluster_members),
                'intra_similarity': np.mean(intra_similarities) if intra_similarities else 0.0,
                'inter_difference': np.mean(inter_differences) if inter_differences else 0.0,
                'members': cluster_members.tolist()
            }
        
        return cluster_analysis
    
    def _calculate_balance_score(self, cluster_analysis):
        """
        计算分簇平衡评分
        """
        if not cluster_analysis:
            return 0.0
        
        # 簇大小平衡性
        sizes = [info['size'] for info in cluster_analysis.values()]
        size_variance = np.var(sizes) if len(sizes) > 1 else 0.0
        size_balance = 1.0 / (1.0 + size_variance)
        
        # 簇内相似度平衡性
        intra_similarities = [info['intra_similarity'] for info in cluster_analysis.values()]
        intra_balance = np.mean(intra_similarities)
        
        # 簇间差异度平衡性
        inter_differences = [info['inter_difference'] for info in cluster_analysis.values()]
        inter_balance = np.mean(inter_differences)
        
        # 综合平衡评分
        balance_score = (size_balance * 0.3 + 
                        intra_balance * 0.4 + 
                        inter_balance * 0.3)
        
        return balance_score
    
    def _optimize_cluster_balance(self, cluster_labels, distances, target_client_id, cluster_analysis):
        """
        优化簇间平衡
        """
        print("      执行簇间平衡优化...")
        
        # 识别不平衡的簇
        unbalanced_clusters = self._identify_unbalanced_clusters(cluster_analysis)
        
        if not unbalanced_clusters:
            return cluster_labels
        
        # 重新分配客户端以改善平衡性
        optimized_labels = cluster_labels.copy()
        
        for cluster_id in unbalanced_clusters:
            # 尝试将一些客户端重新分配到其他簇
            self._redistribute_cluster_members(
                cluster_id, optimized_labels, distances, cluster_analysis, target_client_id
            )
        
        return optimized_labels
    
    def _identify_unbalanced_clusters(self, cluster_analysis):
        """
        识别不平衡的簇
        """
        unbalanced = []
        
        if len(cluster_analysis) < 2:
            return unbalanced
        
        sizes = [info['size'] for info in cluster_analysis.values()]
        mean_size = np.mean(sizes)
        size_threshold = mean_size * 0.5  # 大小差异超过50%认为不平衡
        
        for cluster_id, info in cluster_analysis.items():
            if abs(info['size'] - mean_size) > size_threshold:
                unbalanced.append(cluster_id)
        
        return unbalanced
    
    def _redistribute_cluster_members(self, cluster_id, labels, distances, cluster_analysis, target_client_id=0):
        """
        重新分配簇成员以改善平衡性，同时保持目标客户端隔离性
        
        Args:
            cluster_id: 源簇ID
            labels: 分簇标签
            distances: 距离矩阵
            cluster_analysis: 簇分析结果
            target_client_id: 目标客户端ID
        """
        cluster_members = np.where(labels == cluster_id)[0]
        
        if len(cluster_members) <= 1:
            return
        
        # 找到最适合重新分配的成员
        best_candidate = None
        best_target_cluster = None
        best_improvement = 0.0
        
        for member in cluster_members:
            for target_cluster_id in cluster_analysis.keys():
                if target_cluster_id == cluster_id:
                    continue
                
                # 禁止将客户端分配到目标客户端所在的簇
                if target_cluster_id == labels[target_client_id]:
                    continue
                
                # 计算重新分配后的改善程度
                improvement = self._calculate_redistribution_improvement(
                    member, cluster_id, target_cluster_id, labels, distances, cluster_analysis
                )
                
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_candidate = member
                    best_target_cluster = target_cluster_id
        
        # 执行重新分配
        if best_candidate is not None and best_improvement > 0.01:
            labels[best_candidate] = best_target_cluster
            print(f"        客户端 {best_candidate} 从簇 {cluster_id} 重新分配到簇 {best_target_cluster}")
        else:
            print(f"        未找到合适的重新分配方案，保持当前分簇")
    
    def _calculate_redistribution_improvement(self, client_id, from_cluster, to_cluster, 
                                           labels, distances, cluster_analysis):
        """
        计算重新分配客户端后的改善程度
        """
        # 模拟重新分配
        temp_labels = labels.copy()
        temp_labels[client_id] = to_cluster
        
        # 重新分析平衡性
        temp_analysis = self._analyze_cluster_balance(temp_labels, distances)
        temp_balance_score = self._calculate_balance_score(temp_analysis)
        
        # 计算改善程度
        current_balance_score = self._calculate_balance_score(cluster_analysis)
        improvement = temp_balance_score - current_balance_score
        
        return improvement
    
    def joint_optimize_clustering(self, cluster_labels, distances, target_client_id=0):
        """
        联合优化分簇策略，平衡遗忘效果、保护效果和簇间平衡
        
        Args:
            cluster_labels: 当前分簇标签
            distances: 距离矩阵
            target_client_id: 目标客户端ID
            
        Returns:
            optimized_labels: 优化后的分簇标签
        """
        if not self.joint_optimization:
            return cluster_labels
        
        print("  执行联合优化分簇策略...")
        
        # 初始化优化历史
        self.optimization_history = []
        
        current_labels = cluster_labels.copy()
        best_labels = current_labels.copy()
        best_score = self._evaluate_joint_objective(
            current_labels, distances, target_client_id
        )
        
        print(f"    初始联合目标评分: {best_score:.4f}")
        
        # 迭代优化
        for iteration in range(self.optimization_iterations):
            # 生成候选解
            candidate_labels = self._generate_candidate_solution(
                current_labels, distances, target_client_id
            )
            
            # 评估候选解
            candidate_score = self._evaluate_joint_objective(
                candidate_labels, distances, target_client_id
            )
            
            # 记录优化历史
            self.optimization_history.append({
                'iteration': iteration,
                'current_score': best_score,
                'candidate_score': candidate_score,
                'improvement': candidate_score - best_score
            })
            
            # 接受更好的解
            if candidate_score > best_score:
                best_labels = candidate_labels.copy()
                best_score = candidate_score
                current_labels = candidate_labels.copy()
                print(f"      迭代 {iteration + 1}: 评分提升到 {best_score:.4f}")
            else:
                # 概率接受较差的解（模拟退火）
                acceptance_prob = np.exp((candidate_score - best_score) / 0.1)
                if np.random.random() < acceptance_prob:
                    current_labels = candidate_labels.copy()
            
            # 检查收敛性
            if iteration > 0:
                recent_improvements = [h['improvement'] for h in self.optimization_history[-3:]]
                if all(abs(imp) < self.convergence_threshold for imp in recent_improvements):
                    print(f"      在第 {iteration + 1} 次迭代后收敛")
                    break
        
        print(f"    联合优化完成，最终评分: {best_score:.4f}")
        return best_labels
    
    def _evaluate_joint_objective(self, cluster_labels, distances, target_client_id):
        """
        评估联合目标函数
        """
        # 兜底初始化联合优化权重，避免属性缺失
        if not hasattr(self, 'balance_weights') or self.balance_weights is None:
            self.balance_weights = {
                'forget_effect': 0.4,
                'protection_effect': 0.3,
                'cluster_balance': 0.3,
            }
        else:
            # 校验缺失键并补齐
            for k, v in {'forget_effect': 0.4, 'protection_effect': 0.3, 'cluster_balance': 0.3}.items():
                if k not in self.balance_weights:
                    self.balance_weights[k] = v
            # 归一化
            total_w = sum(self.balance_weights.values()) or 1.0
            for k in self.balance_weights:
                self.balance_weights[k] = float(self.balance_weights[k]) / total_w
        # 遗忘效果评分
        forget_score = self._evaluate_forget_effect(cluster_labels, distances, target_client_id)
        
        # 保护效果评分
        protection_score = self._evaluate_protection_effect(cluster_labels, distances, target_client_id)
        
        # 簇间平衡评分
        cluster_analysis = self._analyze_cluster_balance(cluster_labels, distances)
        balance_score = self._calculate_balance_score(cluster_analysis)
        
        # 加权综合评分
        joint_score = (self.balance_weights['forget_effect'] * forget_score +
                       self.balance_weights['protection_effect'] * protection_score +
                       self.balance_weights['cluster_balance'] * balance_score)
        
        return joint_score
    
    def _evaluate_forget_effect(self, cluster_labels, distances, target_client_id):
        """
        评估遗忘效果
        """
        target_cluster = cluster_labels[target_client_id]
        target_cluster_members = np.where(cluster_labels == target_cluster)[0]
        
        # 目标簇内相似度越高，遗忘效果越好
        intra_similarities = []
        for i in target_cluster_members:
            for j in target_cluster_members:
                if i < j:
                    intra_similarities.append(1.0 - distances[i, j])
        
        if not intra_similarities:
            return 0.0
        
        return np.mean(intra_similarities)
    
    def _evaluate_protection_effect(self, cluster_labels, distances, target_client_id):
        """
        评估保护效果
        """
        target_cluster = cluster_labels[target_client_id]
        other_clusters = [c for c in np.unique(cluster_labels) if c != target_cluster]
        
        if not other_clusters:
            return 0.0
        
        # 其他簇与目标簇的差异度越高，保护效果越好
        inter_differences = []
        for other_cluster in other_clusters:
            other_members = np.where(cluster_labels == other_cluster)[0]
            target_members = np.where(cluster_labels == target_cluster)[0]
            
            for i in other_members:
                for j in target_members:
                    inter_differences.append(distances[i, j])
        
        if not inter_differences:
            return 0.0
        
        return np.mean(inter_differences)
    
    def _generate_candidate_solution(self, current_labels, distances, target_client_id):
        """
        生成候选解
        """
        candidate_labels = current_labels.copy()
        
        # 随机选择一个非目标客户端
        non_target_clients = [i for i in range(len(current_labels)) if i != target_client_id]
        if not non_target_clients:
            return candidate_labels
        
        selected_client = np.random.choice(non_target_clients)
        
        # 随机重新分配簇
        available_clusters = list(np.unique(current_labels))
        new_cluster = np.random.choice(available_clusters)
        
        candidate_labels[selected_client] = new_cluster
        
        return candidate_labels
    
    def _detect_anomaly_clients_enhanced(self, cluster_labels, distances, target_client_id=0):
        """
        增强的异常客户端检测
        """
        print("    检测异常客户端...")
        
        anomaly_clients = []
        unique_clusters = np.unique(cluster_labels)
        
        for cluster_id in unique_clusters:
            cluster_members = np.where(cluster_labels == cluster_id)[0]
            
            if len(cluster_members) <= 1:
                continue
            
            # 计算簇内平均距离
            intra_distances = []
            for i in cluster_members:
                for j in cluster_members:
                    if i < j:
                        intra_distances.append(distances[i, j])
            
            if not intra_distances:
                continue
            
            mean_intra_distance = np.mean(intra_distances)
            std_intra_distance = np.std(intra_distances)
            
            # 检测异常客户端（距离簇中心过远的客户端）
            for member in cluster_members:
                avg_distance_to_others = np.mean([distances[member, j] for j in cluster_members if j != member])
                
                if avg_distance_to_others > mean_intra_distance + 2 * std_intra_distance:
                    anomaly_clients.append(member)
                    print(f"      客户端 {member} 被识别为异常客户端")
        
        print(f"    异常客户端检测完成，共发现 {len(anomaly_clients)} 个异常客户端")
        return anomaly_clients
    
    def _evaluate_clustering_quality_enhanced(self, cluster_labels, distances, target_client_id=0):
        """
        增强的分簇质量评估
        """
        print("    评估分簇质量...")
        
        unique_clusters = np.unique(cluster_labels)
        
        # 计算轮廓系数
        try:
            silhouette_avg = silhouette_score(distances, cluster_labels)
        except:
            silhouette_avg = 0.0
        
        # 计算Calinski-Harabasz指数
        try:
            calinski_score = calinski_harabasz_score(distances, cluster_labels)
        except:
            calinski_score = 0.0
        
        # 计算簇内凝聚度和簇间分离度
        intra_cohesion = 0.0
        inter_separation = 0.0
        
        for cluster_id in unique_clusters:
            cluster_members = np.where(cluster_labels == cluster_id)[0]
            
            # 簇内凝聚度
            if len(cluster_members) > 1:
                intra_distances = []
                for i in cluster_members:
                    for j in cluster_members:
                        if i < j:
                            intra_distances.append(distances[i, j])
                if intra_distances:
                    intra_cohesion += np.mean(intra_distances)
            
            # 簇间分离度
            for other_cluster_id in unique_clusters:
                if other_cluster_id != cluster_id:
                    other_members = np.where(cluster_labels == other_cluster_id)[0]
                    inter_distances = []
                    for i in cluster_members:
                        for j in other_members:
                            inter_distances.append(distances[i, j])
                    if inter_distances:
                        inter_separation += np.mean(inter_distances)
        
        if len(unique_clusters) > 1:
            inter_separation /= (len(unique_clusters) - 1)
        
        quality_metrics = {
            'silhouette_score': silhouette_avg,
            'calinski_harabasz_score': calinski_score,
            'intra_cohesion': intra_cohesion,
            'inter_separation': inter_separation,
            'cluster_count': len(unique_clusters),
            'total_clients': len(cluster_labels)
        }
        
        print(f"    分簇质量评估完成:")
        print(f"      轮廓系数: {silhouette_avg:.4f}")
        print(f"      Calinski-Harabasz指数: {calinski_score:.4f}")
        print(f"      簇内凝聚度: {intra_cohesion:.4f}")
        print(f"      簇间分离度: {inter_separation:.4f}")
        
        return quality_metrics
    
    def _visualize_clustering_results_enhanced(self, cluster_labels, distances, target_client_id, quality_metrics):
        """
        增强的分簇结果可视化
        """
        print("    生成可视化分析...")
        
        try:
            # 创建图形
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('增强分簇策略结果分析', fontsize=16)
            
            # 1. 分簇结果散点图
            ax1 = axes[0, 0]
            unique_clusters = np.unique(cluster_labels)
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
            
            for i, cluster_id in enumerate(unique_clusters):
                cluster_members = np.where(cluster_labels == cluster_id)[0]
                if len(cluster_members) > 0:
                    # 使用PCA降维到2D进行可视化
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=2)
                    cluster_data = distances[cluster_members][:, cluster_members]
                    if cluster_data.shape[0] > 1:
                        cluster_data_2d = pca.fit_transform(cluster_data)
                        ax1.scatter(cluster_data_2d[:, 0], cluster_data_2d[:, 1], 
                                  c=[colors[i]], label=f'簇 {cluster_id}', alpha=0.7)
            
            ax1.set_title('分簇结果可视化')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. 质量指标雷达图
            ax2 = axes[0, 1]
            metrics_names = ['轮廓系数', '簇内凝聚度', '簇间分离度', '簇数量']
            metrics_values = [
                quality_metrics['silhouette_score'],
                quality_metrics['intra_cohesion'],
                quality_metrics['inter_separation'],
                quality_metrics['cluster_count'] / quality_metrics['total_clients']
            ]
            
            # 归一化到0-1范围
            metrics_values = [(v - min(metrics_values)) / (max(metrics_values) - min(metrics_values)) 
                            if max(metrics_values) != min(metrics_values) else 0.5 for v in metrics_values]
            
            angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
            metrics_values += metrics_values[:1]
            angles += angles[:1]
            
            ax2.plot(angles, metrics_values, 'o-', linewidth=2)
            ax2.fill(angles, metrics_values, alpha=0.25)
            ax2.set_xticks(angles[:-1])
            ax2.set_xticklabels(metrics_names)
            ax2.set_title('分簇质量指标')
            ax2.grid(True)
            
            # 3. 簇大小分布
            ax3 = axes[1, 0]
            cluster_sizes = [len(np.where(cluster_labels == c)[0]) for c in unique_clusters]
            ax3.bar(range(len(unique_clusters)), cluster_sizes, color=colors)
            ax3.set_title('各簇大小分布')
            ax3.set_xlabel('簇ID')
            ax3.set_ylabel('客户端数量')
            ax3.set_xticks(range(len(unique_clusters)))
            ax3.set_xticklabels([f'簇{c}' for c in unique_clusters])
            
            # 4. 距离分布热力图
            ax4 = axes[1, 1]
            im = ax4.imshow(distances, cmap='viridis', aspect='auto')
            ax4.set_title('客户端间距离热力图')
            ax4.set_xlabel('客户端ID')
            ax4.set_ylabel('客户端ID')
            plt.colorbar(im, ax=ax4)
            
            plt.tight_layout()
            
            # 保存图片
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"clustering_analysis_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"      可视化结果已保存到: {filename}")
            
            plt.close()
            
        except Exception as e:
            print(f"      可视化生成失败: {e}")
    
    def _save_analysis_results_enhanced(self, cluster_labels, anomaly_clients, quality_metrics, target_client_id):
        """
        增强的分析结果保存
        """
        print("    保存分析结果...")
        
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            results = {
                'timestamp': timestamp,
                'target_client_id': target_client_id,
                'cluster_labels': cluster_labels.tolist(),
                'anomaly_clients': anomaly_clients,
                'quality_metrics': quality_metrics,
                'optimization_history': self.optimization_history,
                'adaptive_weights': self.adaptive_weights,
                'client_protection_levels': self.client_protection_levels,
                'impact_predictions': self.impact_prediction_cache
            }
            
            filename = f"clustering_analysis_{timestamp}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"      分析结果已保存到: {filename}")
            
        except Exception as e:
            print(f"      结果保存失败: {e}")
    
    def _multi_level_clustering_strategy(self, distances, target_client_id, thresholds):
        """
        策略2: 多层次分簇策略
        改进原理：从粗到细，逐步优化，目标导向
        """
        print("执行多层次分簇策略...")
        num_clients = len(distances)
        
        # 第一层：基于分布相似性的粗分簇
        print("第一层：基于分布相似性的粗分簇")
        coarse_clusters = self._coarse_clustering_by_distribution_enhanced(
            distances, target_client_id, thresholds
        )
        
        # 第二层：基于梯度相似性的细分簇
        print("第二层：基于梯度相似性的细分簇")
        fine_clusters = self._fine_clustering_by_gradients_enhanced(
            coarse_clusters, distances, target_client_id, thresholds
        )
        
        # 第三层：基于遗忘需求的优化分簇
        print("第三层：基于遗忘需求的优化分簇")
        optimized_clusters = self._optimize_clusters_for_forgetting_enhanced(
            fine_clusters, distances, target_client_id, thresholds
        )
        
        # 最终验证：确保目标客户端独立成簇
        optimized_clusters = self._ensure_target_client_isolation(optimized_clusters, distances, target_client_id)
        
        return optimized_clusters
    
    def _ensure_target_client_isolation(self, cluster_labels, distances, target_client_id):
        """
        确保目标客户端独立成簇，同时保持其他客户端的合理分簇
        
        Args:
            cluster_labels: 当前分簇标签
            distances: 距离矩阵
            target_client_id: 目标客户端ID
            
        Returns:
            isolated_labels: 确保目标客户端独立的分簇标签
        """
        print(f"验证目标客户端{target_client_id}的簇隔离性...")
        
        isolated_labels = cluster_labels.copy()
        target_cluster = isolated_labels[target_client_id]
        
        # 检查是否有其他客户端与目标客户端在同一簇
        same_cluster_clients = np.where(isolated_labels == target_cluster)[0]
        other_clients_in_same_cluster = [cid for cid in same_cluster_clients if cid != target_client_id]
        
        if other_clients_in_same_cluster:
            print(f"⚠️  发现违规：客户端{other_clients_in_same_cluster}与目标客户端{target_client_id}在同一簇{target_cluster}")
            print("正在修复簇分配...")
            
            # 重新分析其他客户端的相似性，进行合理分簇
            isolated_labels = self._redistribute_clients_intelligently(
                isolated_labels, distances, target_client_id, other_clients_in_same_cluster
            )
            
            print(f"✅ 簇隔离性修复完成")
        else:
            print(f"✅ 目标客户端{target_client_id}已独立成簇")
        
        return isolated_labels
    
    def _redistribute_clients_intelligently(self, cluster_labels, distances, target_client_id, clients_to_redistribute):
        """
        智能重新分配客户端，保持合理的分簇结构
        
        Args:
            cluster_labels: 当前分簇标签
            distances: 距离矩阵
            target_client_id: 目标客户端ID
            clients_to_redistribute: 需要重新分配的客户端列表
            
        Returns:
            redistributed_labels: 重新分配后的分簇标签
        """
        print(f"  智能重新分配{len(clients_to_redistribute)}个客户端...")
        
        redistributed_labels = cluster_labels.copy()
        
        # 获取现有的其他簇
        existing_clusters = set(cluster_labels) - {cluster_labels[target_client_id]}
        if not existing_clusters:
            # 如果没有其他簇，创建一个新簇
            existing_clusters = {1}
        
        # 为每个需要重新分配的客户端找到最合适的簇
        for client_id in clients_to_redistribute:
            best_cluster = self._find_best_cluster_for_client(
                client_id, distances, redistributed_labels, existing_clusters, target_client_id
            )
            
            if best_cluster is not None:
                redistributed_labels[client_id] = best_cluster
                print(f"    客户端{client_id}重新分配到簇{best_cluster}")
            else:
                # 如果找不到合适的簇，创建新簇
                new_cluster = max(existing_clusters) + 1
                redistributed_labels[client_id] = new_cluster
                existing_clusters.add(new_cluster)
                print(f"    客户端{client_id}分配到新簇{new_cluster}")
        
        return redistributed_labels
    
    def _find_best_cluster_for_client(self, client_id, distances, cluster_labels, existing_clusters, target_client_id):
        """
        为客户端找到最合适的簇
        
        Args:
            client_id: 客户端ID
            distances: 距离矩阵
            cluster_labels: 当前分簇标签
            existing_clusters: 现有簇集合
            target_client_id: 目标客户端ID
            
        Returns:
            best_cluster: 最合适的簇ID，如果没有合适的则返回None
        """
        best_cluster = None
        best_similarity = -1
        
        for cluster_id in existing_clusters:
            if cluster_id == cluster_labels[target_client_id]:
                continue  # 跳过目标客户端簇
            
            # 计算与该簇内客户端的平均相似度
            cluster_members = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_members) == 0:
                continue
            
            similarities = []
            for member in cluster_members:
                if member != client_id:
                    # 距离越小，相似度越高
                    similarity = 1.0 - distances[client_id, member]
                    similarities.append(similarity)
            
            if similarities:
                avg_similarity = np.mean(similarities)
                if avg_similarity > best_similarity:
                    best_similarity = avg_similarity
                    best_cluster = cluster_id
        
        # 只有当相似度足够高时才分配，否则返回None（创建新簇）
        similarity_threshold = 0.3  # 可调整的阈值
        if best_similarity >= similarity_threshold:
            return best_cluster
        else:
            return None
    
    def _remap_cluster_labels_with_isolation(self, cluster_labels, target_client_id):
        """
        重新映射簇标签，确保簇标签连续，同时保持目标客户端独立
        
        Args:
            cluster_labels: 原始簇标签数组
            target_client_id: 目标客户端ID
            
        Returns:
            remapped_labels: 重映射后的连续簇标签数组
        """
        unique_clusters = np.unique(cluster_labels)
        target_cluster = cluster_labels[target_client_id]
        
        # 确保目标客户端簇标签为0
        cluster_mapping = {target_cluster: 0}
        
        # 为其他簇分配连续的标签
        current_label = 1
        for old_label in unique_clusters:
            if old_label != target_cluster:
                cluster_mapping[old_label] = current_label
                current_label += 1
        
        remapped_labels = np.array([cluster_mapping[label] for label in cluster_labels])
        
        print(f"簇标签重映射（保持目标客户端隔离）: {dict(cluster_mapping)}")
        print(f"重映射后簇标签: {remapped_labels}")
        
        # 最终验证
        final_target_cluster = remapped_labels[target_client_id]
        final_same_cluster = np.where(remapped_labels == final_target_cluster)[0]
        if len(final_same_cluster) == 1 and final_same_cluster[0] == target_client_id:
            print(f"✅ 最终验证通过：目标客户端{target_client_id}独立成簇{final_target_cluster}")
        else:
            print(f"❌ 最终验证失败：目标客户端{target_client_id}簇{final_target_cluster}包含其他客户端{final_same_cluster}")
        
        return remapped_labels
    
    def _coarse_clustering_by_distribution_enhanced(self, distances, target_client_id, thresholds):
        """第一层：基于分布相似性的粗分簇"""
        num_clients = len(distances)
        cluster_labels = np.zeros(num_clients, dtype=int)
        current_cluster = 0
        
        # 目标客户端单独成簇
        cluster_labels[target_client_id] = current_cluster
        current_cluster += 1
        print(f"目标客户端{target_client_id}分配到簇{cluster_labels[target_client_id]}")
        
        # 计算与目标客户端的距离排序
        target_distances = []
        for i in range(num_clients):
            if i != target_client_id:
                target_distances.append((i, distances[target_client_id][i]))
        
        target_distances.sort(key=lambda x: x[1])
        
        print("与目标客户端的距离排序:")
        for client_id, distance in target_distances:
            print(f"  客户端{client_id}: {distance:.4f}")
        
        # 基于自适应阈值进行粗分簇
        high_threshold = thresholds['high']
        medium_threshold = thresholds['medium']
        
        # 高相似度组
        high_similar_clients = []
        for client_id, distance in target_distances:
            if distance < high_threshold:
                high_similar_clients.append(client_id)
        
        if high_similar_clients:
            cluster_labels[high_similar_clients] = current_cluster
            print(f"高相似度客户端{high_similar_clients}分配到簇{current_cluster}")
            current_cluster += 1
        
        # 中等相似度组
        medium_similar_clients = []
        for client_id, distance in target_distances:
            if high_threshold <= distance < medium_threshold:
                medium_similar_clients.append(client_id)
        
        if medium_similar_clients:
            cluster_labels[medium_similar_clients] = current_cluster
            print(f"中等相似度客户端{medium_similar_clients}分配到簇{current_cluster}")
            current_cluster += 1
        
        # 低相似度组
        low_similar_clients = []
        for client_id, distance in target_distances:
            if distance >= medium_threshold:
                low_similar_clients.append(client_id)
        
        if low_similar_clients:
            cluster_labels[low_similar_clients] = current_cluster
            print(f"低相似度客户端{low_similar_clients}分配到簇{current_cluster}")
            current_cluster += 1
        
        return cluster_labels
    
    def _fine_clustering_by_gradients_enhanced(self, coarse_clusters, distances, target_client_id, thresholds):
        """第二层：基于梯度相似性的细分簇"""
        print("基于梯度相似性进行细分簇...")
        
        # 获取当前簇的数量和成员
        unique_clusters = np.unique(coarse_clusters)
        fine_clusters = coarse_clusters.copy()
        
        for cluster_id in unique_clusters:
            if cluster_id == 0:  # 跳过目标客户端簇
                continue
            
            cluster_members = np.where(coarse_clusters == cluster_id)[0]
            if len(cluster_members) <= 1:
                continue
            
            print(f"细分簇{cluster_id}，成员: {cluster_members}")
            
            # 计算簇内客户端的相似性矩阵
            cluster_distances = distances[np.ix_(cluster_members, cluster_members)]
            
            # 使用层次聚类进行细分
            if len(cluster_members) > 2:
                try:
                    # 注意：ward 只能用于欧氏距离，不能和 precomputed 共用
                    # 若我们只有预计算距离矩阵，则使用 'average' 以支持 precomputed
                    clustering = AgglomerativeClustering(
                        n_clusters=min(3, len(cluster_members)),
                        linkage='average',
                        affinity='precomputed'
                    )
                    
                    sub_labels = clustering.fit_predict(cluster_distances)
                    
                    # 更新簇标签 - 优化版本：使用连续的簇标签
                    current_max_cluster = np.max(fine_clusters)
                    for i, member_id in enumerate(cluster_members):
                        if sub_labels[i] > 0:  # 非主簇
                            fine_clusters[member_id] = current_max_cluster + sub_labels[i]
                    
                    print(f"簇{cluster_id}细分为{len(np.unique(sub_labels))}个子簇")
                except Exception as e:
                    print(f"细分簇{cluster_id}时出错: {e}")
        
        # 添加簇标签重映射，确保簇标签连续，同时保持目标客户端隔离
        fine_clusters = self._remap_cluster_labels_with_isolation(fine_clusters, target_client_id)
        
        # 验证目标客户端隔离性
        target_cluster = fine_clusters[target_client_id]
        same_cluster_clients = np.where(fine_clusters == target_cluster)[0]
        if len(same_cluster_clients) > 1:
            print(f"❌ 细分簇后目标客户端隔离性被破坏！")
            print(f"目标客户端{target_client_id}簇{target_cluster}包含其他客户端{same_cluster_clients}")
            # 强制修复
            fine_clusters = self._ensure_target_client_isolation(fine_clusters, distances, target_client_id)
        else:
            print(f"✅ 细分簇后目标客户端{target_client_id}仍保持独立成簇{target_cluster}")
        
        return fine_clusters
    
    def _remap_cluster_labels(self, cluster_labels):
        """
        重新映射簇标签，确保簇标签连续（从0开始）
        注意：此方法可能会破坏目标客户端的隔离性，建议使用 _remap_cluster_labels_with_isolation
        
        Args:
            cluster_labels: 原始簇标签数组
            
        Returns:
            remapped_labels: 重映射后的连续簇标签数组
        """
        print("⚠️  警告：使用基础簇标签重映射方法，可能会破坏目标客户端隔离性")
        print("建议使用 _remap_cluster_labels_with_isolation 方法")
        
        unique_clusters = np.unique(cluster_labels)
        cluster_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_clusters)}
        
        remapped_labels = np.array([cluster_mapping[label] for label in cluster_labels])
        
        print(f"簇标签重映射: {dict(cluster_mapping)}")
        print(f"重映射后簇标签: {remapped_labels}")
        
        return remapped_labels
    
    def _optimize_clusters_for_forgetting_enhanced(self, clusters, distances, target_client_id, thresholds):
        """
        策略3: 基于遗忘效果的分簇优化
        改进原理：直接针对遗忘效果优化，量化评估分簇质量
        """
        print("基于遗忘效果优化分簇...")
        
        # 计算当前分簇的遗忘效果评分
        current_score = self._evaluate_clustering_for_forgetting(clusters, distances, target_client_id)
        print(f"当前分簇的遗忘效果评分: {current_score:.4f}")
        
        # 生成候选分簇方案
        candidate_clusterings = self._generate_candidate_clusterings(distances, target_client_id, thresholds)
        
        best_score = current_score
        best_clustering = clusters.copy()
        
        # 评估每个候选方案
        for i, candidate in enumerate(candidate_clusterings):
            candidate_score = self._evaluate_clustering_for_forgetting(candidate, distances, target_client_id)
            
            if candidate_score > best_score:
                best_score = candidate_score
                best_clustering = candidate.copy()
                print(f"候选方案{i+1}获得更高评分: {candidate_score:.4f}")
        
        if best_score > current_score:
            print(f"分簇优化完成，评分从{current_score:.4f}提升到{best_score:.4f}")
        else:
            print("当前分簇已是最优，无需优化")
        
        return best_clustering
    
    def _evaluate_clustering_for_forgetting(self, clusters, distances, target_client_id):
        """评估分簇的遗忘效果"""
        unique_clusters = np.unique(clusters)
        target_cluster = clusters[target_client_id]
        
        # 计算目标客户端簇的隔离度
        target_cluster_members = np.where(clusters == target_cluster)[0]
        other_clusters = [c for c in unique_clusters if c != target_cluster]
        
        # 簇内距离（越小越好）
        if len(target_cluster_members) > 1:
            target_cluster_distances = []
            for i in target_cluster_members:
                for j in target_cluster_members:
                    if i < j:
                        target_cluster_distances.append(distances[i][j])
            intra_cluster_distance = np.mean(target_cluster_distances) if target_cluster_distances else 0
        else:
            intra_cluster_distance = 0
        
        # 簇间距离（越大越好）
        inter_cluster_distances = []
        for other_cluster in other_clusters:
            other_members = np.where(clusters == other_cluster)[0]
            for target_member in target_cluster_members:
                for other_member in other_members:
                    inter_cluster_distances.append(distances[target_member][other_member])
        
        inter_cluster_distance = np.mean(inter_cluster_distances) if inter_cluster_distances else 1
        
        # 计算遗忘效果评分
        # 目标：最小化簇内距离，最大化簇间距离
        isolation_score = inter_cluster_distance / (intra_cluster_distance + 1e-6)
        
        # 簇的平衡性评分
        cluster_sizes = [np.sum(clusters == c) for c in unique_clusters]
        balance_score = 1.0 / (np.std(cluster_sizes) + 1e-6)
        
        # 综合评分
        total_score = self.forgetting_weight * isolation_score + \
                     self.protection_weight * balance_score + \
                     self.utility_weight * len(unique_clusters) / len(clusters)
        
        return total_score
    
    def _generate_candidate_clusterings(self, distances, target_client_id, thresholds):
        """生成候选分簇方案"""
        candidates = []
        num_clients = len(distances)
        
        # 方案1：基于密度的分簇
        try:
            density_clustering = DBSCAN(eps=thresholds['medium'], min_samples=1)
            density_labels = density_clustering.fit_predict(distances)
            # 确保目标客户端单独成簇
            density_labels[target_client_id] = 0
            # 验证目标客户端隔离性
            density_labels = self._validate_and_fix_candidate_clustering(density_labels, distances, target_client_id)
            candidates.append(density_labels)
        except Exception as e:
            print(f"密度分簇方案生成失败: {e}")
        
        # 方案2：基于K-means的分簇
        try:
            optimal_k = self._determine_optimal_k(distances)
            kmeans = KMeans(n_clusters=optimal_k, random_state=42)
            kmeans_labels = kmeans.fit_predict(distances)
            # 确保目标客户端单独成簇
            kmeans_labels[target_client_id] = 0
            # 验证目标客户端隔离性
            kmeans_labels = self._validate_and_fix_candidate_clustering(kmeans_labels, distances, target_client_id)
            candidates.append(kmeans_labels)
        except Exception as e:
            print(f"K-means分簇方案生成失败: {e}")
        
        # 方案3：基于层次聚类的分簇
        try:
            optimal_k = self._determine_optimal_k(distances)
            # 同理，使用 average + precomputed 保持一致
            hierarchical = AgglomerativeClustering(
                n_clusters=optimal_k,
                linkage='average',
                affinity='precomputed'
            )
            hierarchical_labels = hierarchical.fit_predict(distances)
            # 确保目标客户端单独成簇
            hierarchical_labels[target_client_id] = 0
            # 验证目标客户端隔离性
            hierarchical_labels = self._validate_and_fix_candidate_clustering(hierarchical_labels, distances, target_client_id)
            candidates.append(hierarchical_labels)
        except Exception as e:
            print(f"层次分簇方案生成失败: {e}")
        
        # 方案4：强制目标客户端隔离的分簇
        try:
            forced_isolation_labels = self._generate_forced_isolation_clustering(distances, target_client_id, thresholds)
            candidates.append(forced_isolation_labels)
        except Exception as e:
            print(f"强制隔离分簇方案生成失败: {e}")
        
        print(f"成功生成 {len(candidates)} 个候选分簇方案")
        return candidates
    
    def _validate_and_fix_candidate_clustering(self, cluster_labels, distances, target_client_id):
        """
        验证并修复候选分簇方案，确保目标客户端独立成簇，同时保持合理的分簇结构
        
        Args:
            cluster_labels: 候选分簇标签
            distances: 距离矩阵
            target_client_id: 目标客户端ID
            
        Returns:
            fixed_labels: 修复后的分簇标签
        """
        fixed_labels = cluster_labels.copy()
        target_cluster = fixed_labels[target_client_id]
        
        # 检查是否有其他客户端与目标客户端在同一簇
        same_cluster_clients = np.where(fixed_labels == target_cluster)[0]
        other_clients_in_same_cluster = [cid for cid in same_cluster_clients if cid != target_client_id]
        
        if other_clients_in_same_cluster:
            print(f"    修复候选方案：客户端{other_clients_in_same_cluster}与目标客户端{target_client_id}在同一簇")
            
            # 智能重新分配这些客户端，保持合理的分簇结构
            fixed_labels = self._redistribute_clients_intelligently(
                fixed_labels, distances, target_client_id, other_clients_in_same_cluster
            )
            
            print(f"    修复完成：目标客户端{target_client_id}独立成簇{target_cluster}")
        
        return fixed_labels
    
    def _generate_forced_isolation_clustering(self, distances, target_client_id, thresholds):
        """
        生成强制目标客户端隔离的分簇方案，同时保持合理的分簇结构
        
        Args:
            distances: 距离矩阵
            target_client_id: 目标客户端ID
            thresholds: 分簇阈值
            
        Returns:
            forced_labels: 强制隔离的分簇标签
        """
        print("    生成强制隔离分簇方案...")
        num_clients = len(distances)
        forced_labels = np.zeros(num_clients, dtype=int)
        
        # 目标客户端独立成簇0
        forced_labels[target_client_id] = 0
        
        # 其他客户端基于相似性进行合理分簇
        other_clients = [i for i in range(num_clients) if i != target_client_id]
        
        if not other_clients:
            return forced_labels
        
        # 计算其他客户端之间的距离矩阵
        other_distances = distances[np.ix_(other_clients, other_clients)]
        
        # 使用层次聚类进行分簇
        try:
            from sklearn.cluster import AgglomerativeClustering
            
            # 确定合适的簇数（基于数据特征）
            optimal_clusters = min(5, max(2, len(other_clients) // 2))
            
            clustering = AgglomerativeClustering(
                n_clusters=optimal_clusters,
                linkage='average',
                affinity='precomputed'
            )
            
            sub_labels = clustering.fit_predict(other_distances)
            
            # 将分簇结果映射回原始标签
            for i, client_id in enumerate(other_clients):
                forced_labels[client_id] = sub_labels[i] + 1  # 从1开始，避免与目标客户端簇0冲突
            
            print(f"    强制隔离方案：目标客户端{target_client_id}独立成簇0，其他客户端分为{optimal_clusters}个簇")
            
        except Exception as e:
            print(f"    层次聚类失败，使用基于距离的简单分簇: {e}")
            
            # 降级到基于距离的简单分簇
            target_distances = [(i, distances[target_client_id][i]) for i in other_clients]
            target_distances.sort(key=lambda x: x[1])
            
            # 基于距离阈值进行分簇
            current_cluster = 1
            high_threshold = thresholds['high']
            medium_threshold = thresholds['medium']
            
            # 高相似度组
            high_similar = [client_id for client_id, distance in target_distances if distance < high_threshold]
            if high_similar:
                forced_labels[high_similar] = current_cluster
                current_cluster += 1
            
            # 中等相似度组
            medium_similar = [client_id for client_id, distance in target_distances 
                             if high_threshold <= distance < medium_threshold]
            if medium_similar:
                forced_labels[medium_similar] = current_cluster
                current_cluster += 1
            
            # 低相似度组
            low_similar = [client_id for client_id, distance in target_distances 
                          if distance >= medium_threshold]
            if low_similar:
                forced_labels[low_similar] = current_cluster
            
            print(f"    强制隔离方案（降级）：目标客户端{target_client_id}独立成簇0，其他客户端分为{current_cluster-1}个簇")
        
        return forced_labels
    
    def _determine_optimal_k(self, distances):
        """确定最优簇数"""
        num_clients = len(distances)
        max_k = min(10, num_clients // 2)
        
        if max_k < 2:
            return 2
        
        # 使用轮廓系数和Calinski-Harabasz指数
        best_score = -1
        optimal_k = 2
        
        for k in range(2, max_k + 1):
            try:
                clustering = KMeans(n_clusters=k, random_state=42)
                labels = clustering.fit_predict(distances)
                
                # 计算轮廓系数
                silhouette_avg = silhouette_score(distances, labels)
                
                # 计算Calinski-Harabasz指数
                ch_score = calinski_harabasz_score(distances, labels)
                
                # 综合评分
                combined_score = silhouette_avg + ch_score / 1000
                
                if combined_score > best_score:
                    best_score = combined_score
                    optimal_k = k
                    
            except:
                continue
        
        return optimal_k
    
    def _analyze_clustering_result_enhanced(self, cluster_labels, distances, target_client_id):
        """增强的分簇结果分析"""
        print("\n============= 增强分簇结果分析 =============")
        
        unique_clusters = np.unique(cluster_labels)
        num_clusters = len(unique_clusters)
        
        print(f"分簇统计:")
        print(f"  - 总簇数: {num_clusters}")
        print(f"  - 目标客户端ID: {target_client_id}")
        
        # 分析每个簇
        for cluster_id in unique_clusters:
            cluster_members = np.where(cluster_labels == cluster_id)[0]
            print(f"\n簇{cluster_id}: 客户端{cluster_members.tolist()}")
            
            if len(cluster_members) > 1:
                # 计算簇内距离统计
                cluster_distances = []
                for i in cluster_members:
                    for j in cluster_members:
                        if i < j:
                            cluster_distances.append(distances[i][j])
                
                if cluster_distances:
                    mean_dist = np.mean(cluster_distances)
                    std_dist = np.std(cluster_distances)
                    print(f"  簇内平均距离: {mean_dist:.4f} ± {std_dist:.4f}")
                    print(f"  簇内距离范围: [{min(cluster_distances):.4f}, {max(cluster_distances):.4f}]")
            
            # 计算簇间距离
            if cluster_id != 0:  # 非目标客户端簇
                target_cluster = cluster_labels[target_client_id]
                inter_distances = []
                for member in cluster_members:
                    inter_distances.append(distances[target_client_id][member])
                
                if inter_distances:
                    mean_inter_dist = np.mean(inter_distances)
                    print(f"  与目标客户端的平均距离: {mean_inter_dist:.4f}")
        
        # 计算分簇质量指标
        quality_metrics = self._compute_clustering_quality_metrics(cluster_labels, distances)
        print(f"\n分簇质量指标:")
        print(f"  - 轮廓系数: {quality_metrics['silhouette']:.4f}")
        print(f"  - Calinski-Harabasz指数: {quality_metrics['calinski_harabasz']:.4f}")
        print(f"  - Davies-Bouldin指数: {quality_metrics['davies_bouldin']:.4f}")
    
    def _compute_clustering_quality_metrics(self, cluster_labels, distances):
        """计算分簇质量指标"""
        try:
            silhouette = silhouette_score(distances, cluster_labels)
        except:
            silhouette = 0
        
        try:
            calinski_harabasz = calinski_harabasz_score(distances, cluster_labels)
        except:
            calinski_harabasz = 0
        
        try:
            from sklearn.metrics import davies_bouldin_score
            davies_bouldin = davies_bouldin_score(distances, cluster_labels)
        except:
            davies_bouldin = 0
        
        return {
            'silhouette': silhouette,
            'calinski_harabasz': calinski_harabasz,
            'davies_bouldin': davies_bouldin
        }
    
    def _traditional_clustering(self, distances, target_client_id, thresholds):
        """传统分簇方法"""
        print("使用传统分簇方法...")
        num_clients = len(distances)
        cluster_labels = np.zeros(num_clients, dtype=int)
        current_cluster = 0
        
        # 目标客户端单独成簇
        cluster_labels[target_client_id] = current_cluster
        current_cluster += 1
        
        # 其他客户端基于距离阈值分簇
        for i in range(num_clients):
            if i != target_client_id:
                distance = distances[target_client_id][i]
                
                if distance < thresholds['high']:
                    cluster_labels[i] = current_cluster
                    current_cluster += 1
                elif distance < thresholds['medium']:
                    cluster_labels[i] = current_cluster
                    current_cluster += 1
                else:
                    cluster_labels[i] = current_cluster
                    current_cluster += 1
        
        return cluster_labels
    
    def save_clustering_analysis_enhanced(self, cluster_labels, distances, target_client_id, 
                                        save_path="results/clustering_analysis"):
        """保存增强的分簇分析结果"""
        os.makedirs(save_path, exist_ok=True)
        
        # 保存分簇标签
        np.save(os.path.join(save_path, "cluster_labels.npy"), cluster_labels)
        
        # 保存距离矩阵
        np.save(os.path.join(save_path, "distances.npy"), distances)
        
        # 保存分析报告
        analysis_report = {
            'target_client_id': target_client_id,
            'num_clusters': len(np.unique(cluster_labels)),
            'cluster_distribution': {},
            'quality_metrics': self._compute_clustering_quality_metrics(cluster_labels, distances),
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 统计每个簇的分布
        unique_clusters = np.unique(cluster_labels)
        for cluster_id in unique_clusters:
            cluster_members = np.where(cluster_labels == cluster_id)[0]
            analysis_report['cluster_distribution'][f'cluster_{cluster_id}'] = {
                'members': cluster_members.tolist(),
                'size': len(cluster_members)
            }
        
        with open(os.path.join(save_path, "clustering_analysis.json"), 'w') as f:
            json.dump(analysis_report, f, indent=2)
        
        print(f"分簇分析结果已保存到: {save_path}")
    
    def visualize_clustering_enhanced(self, cluster_labels, distances, target_client_id, 
                                    save_path="results/clustering_visualization"):
        """可视化增强的分簇结果"""
        os.makedirs(save_path, exist_ok=True)
        
        # 1. 距离矩阵热力图
        plt.figure(figsize=(10, 8))
        sns.heatmap(distances, annot=True, fmt='.3f', cmap='viridis')
        plt.title('客户端间Wasserstein距离矩阵')
        plt.xlabel('客户端ID')
        plt.ylabel('客户端ID')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "distance_matrix.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 分簇结果可视化
        unique_clusters = np.unique(cluster_labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
        
        # 使用PCA降维到2D进行可视化
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            distances_2d = pca.fit_transform(distances)
            
            plt.figure(figsize=(12, 8))
            for i, cluster_id in enumerate(unique_clusters):
                cluster_members = np.where(cluster_labels == cluster_id)[0]
                plt.scatter(distances_2d[cluster_members, 0], 
                           distances_2d[cluster_members, 1], 
                           c=[colors[i]], label=f'簇{cluster_id}', s=100, alpha=0.7)
                
                # 标记目标客户端
                if cluster_id == cluster_labels[target_client_id]:
                    target_pos = distances_2d[target_client_id]
                    plt.scatter(target_pos[0], target_pos[1], 
                               c='red', marker='*', s=200, label='目标客户端', edgecolors='black')
            
            plt.title('客户端分簇结果可视化 (PCA降维)')
            plt.xlabel('主成分1')
            plt.ylabel('主成分2')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, "clustering_visualization.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"PCA可视化失败: {e}")
        
        # 3. 簇大小分布图
        cluster_sizes = [np.sum(cluster_labels == c) for c in unique_clusters]
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(unique_clusters)), cluster_sizes, color=colors)
        plt.title('各簇大小分布')
        plt.xlabel('簇ID')
        plt.ylabel('客户端数量')
        plt.xticks(range(len(unique_clusters)), [f'簇{c}' for c in unique_clusters])
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "cluster_size_distribution.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"分簇可视化结果已保存到: {save_path}")
    
    # ==================== 第三阶段：遗忘效果优化集成 ====================
    
    def advanced_multi_objective_optimization(self, cluster_labels, distances, target_client_id=0):
        """
        高级多目标优化算法，使用改进的NSGA-II算法
        
        Args:
            cluster_labels: 当前分簇标签
            distances: 距离矩阵
            target_client_id: 目标客户端ID
            
        Returns:
            优化后的分簇标签
        """
        if not self.advanced_multi_objective:
            print("高级多目标优化未启用，返回原始分簇结果")
            return cluster_labels
        
        print(f"开始高级多目标优化，使用{self.optimization_algorithm.upper()}算法")
        print(f"种群大小: {self.population_size}, 最大代数: {self.generation_limit}")
        
        try:
            if self.optimization_algorithm == 'nsga2':
                optimized_labels = self._nsga2_optimization(cluster_labels, distances, target_client_id)
            else:
                print(f"不支持的优化算法: {self.optimization_algorithm}")
                return cluster_labels
            
            # 评估优化效果
            original_score = self._evaluate_multi_objective_fitness(cluster_labels, distances, target_client_id)
            optimized_score = self._evaluate_multi_objective_fitness(optimized_labels, distances, target_client_id)
            
            improvement = (optimized_score - original_score) / original_score * 100
            print(f"优化完成！原始得分: {original_score:.4f}, 优化后得分: {optimized_score:.4f}")
            print(f"改进幅度: {improvement:.2f}%")
            
            return optimized_labels
            
        except Exception as e:
            print(f"高级多目标优化失败: {e}")
            print("返回原始分簇结果")
            return cluster_labels
    
    def _nsga2_optimization(self, cluster_labels, distances, target_client_id):
        """
        使用NSGA-II算法进行多目标优化
        
        Args:
            cluster_labels: 当前分簇标签
            distances: 距离矩阵
            target_client_id: 目标客户端ID
            
        Returns:
            优化后的分簇标签
        """
        print("  执行NSGA-II多目标优化...")
        
        # 初始化种群
        population = self._initialize_population(cluster_labels, self.population_size)
        print(f"  初始化种群完成，种群大小: {len(population)}")
        
        # 记录优化历史
        optimization_history = []
        best_fitness_history = []
        
        for generation in range(self.generation_limit):
            # 计算适应度
            fitness_scores = []
            for individual in population:
                fitness = self._evaluate_multi_objective_fitness(individual, distances, target_client_id)
                fitness_scores.append(fitness)
            
            # 非支配排序
            fronts = self._non_dominated_sort(fitness_scores)
            
            # 计算拥挤度距离
            crowding_distances = self._calculate_crowding_distance(fronts, fitness_scores)
            
            # 记录当前代最佳适应度
            best_fitness = max(fitness_scores)
            best_fitness_history.append(best_fitness)
            
            # 检查收敛性
            if self._check_convergence(fitness_scores, generation):
                print(f"  第{generation}代收敛，停止优化")
                break
            
            # 选择、交叉和变异
            new_population = self._nsga2_selection_crossover_mutation(
                population, fitness_scores, fronts, crowding_distances
            )
            
            # 精英保留策略
            elite_individuals = self._select_elite_individuals(population, fitness_scores, self.elite_size)
            new_population = elite_individuals + new_population[:-self.elite_size]
            
            population = new_population
            
            # 记录优化进度
            if generation % 10 == 0:
                print(f"  第{generation}代完成，最佳适应度: {best_fitness:.4f}")
                optimization_history.append({
                    'generation': generation,
                    'best_fitness': best_fitness,
                    'avg_fitness': np.mean(fitness_scores),
                    'population_diversity': self._calculate_population_diversity(population)
                })
        
        # 选择最佳个体
        final_fitness_scores = []
        for individual in population:
            fitness = self._evaluate_multi_objective_fitness(individual, distances, target_client_id)
            final_fitness_scores.append(fitness)
        
        best_individual = self._select_best_individual(population, final_fitness_scores)
        
        # 保存优化历史
        self.optimization_cache['nsga2_history'] = optimization_history
        self.optimization_cache['best_fitness_history'] = best_fitness_history
        
        print(f"  NSGA-II优化完成，共{len(optimization_history)}代")
        return best_individual
    
    def _initialize_population(self, base_labels, population_size):
        """
        初始化种群
        
        Args:
            base_labels: 基础分簇标签
            population_size: 种群大小
            
        Returns:
            初始化的种群
        """
        population = [base_labels.copy()]
        
        # 获取当前簇的数量
        num_clusters = len(np.unique(base_labels))
        
        # 生成随机变异的个体
        for _ in range(population_size - 1):
            # 随机调整分簇标签
            mutated_labels = base_labels.copy()
            num_mutations = max(1, int(len(mutated_labels) * 0.1))  # 10%的标签发生变异
            
            for _ in range(num_mutations):
                idx = np.random.randint(0, len(mutated_labels))
                # 确保新的簇标签在有效范围内
                new_cluster = np.random.randint(0, num_clusters + 1)
                mutated_labels[idx] = new_cluster
            
            population.append(mutated_labels)
        
        return population
    
    def _select_elite_individuals(self, population, fitness_scores, elite_size):
        """
        选择精英个体
        
        Args:
            population: 种群
            fitness_scores: 适应度分数
            elite_size: 精英个体数量
            
        Returns:
            精英个体列表
        """
        # 按适应度排序，选择前elite_size个个体
        sorted_indices = np.argsort(fitness_scores)[::-1]
        elite_individuals = [population[i] for i in sorted_indices[:elite_size]]
        return elite_individuals
    
    def _calculate_population_diversity(self, population):
        """
        计算种群多样性
        
        Args:
            population: 种群
            
        Returns:
            多样性指标
        """
        if len(population) <= 1:
            return 0.0
        
        # 计算所有个体间的平均汉明距离
        total_distance = 0
        count = 0
        
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                distance = np.sum(population[i] != population[j])
                total_distance += distance
                count += 1
        
        return total_distance / count if count > 0 else 0.0
    
    def _evaluate_multi_objective_fitness(self, individual, distances, target_client_id):
        """
        评估多目标适应度
        """
        # 目标1：遗忘效果
        forget_score = self._evaluate_forget_effect(individual, distances, target_client_id)
        
        # 目标2：保护效果
        protection_score = self._evaluate_protection_effect(individual, distances, target_client_id)
        
        # 目标3：簇间平衡
        cluster_analysis = self._analyze_cluster_balance(individual, distances)
        balance_score = self._calculate_balance_score(cluster_analysis)
        
        # 目标4：计算效率
        efficiency_score = self._evaluate_computational_efficiency(individual)
        
        return [forget_score, protection_score, balance_score, efficiency_score]
    
    def _evaluate_computational_efficiency(self, cluster_labels):
        """
        评估计算效率
        """
        # 簇的数量越少，计算效率越高
        num_clusters = len(np.unique(cluster_labels))
        max_clusters = len(cluster_labels)
        
        efficiency = 1.0 - (num_clusters / max_clusters)
        return efficiency
    
    def _non_dominated_sort(self, fitness_scores):
        """
        非支配排序
        """
        n = len(fitness_scores)
        domination_count = [0] * n
        dominated_solutions = [[] for _ in range(n)]
        fronts = [[]]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    if self._dominates(fitness_scores[i], fitness_scores[j]):
                        dominated_solutions[i].append(j)
                    elif self._dominates(fitness_scores[j], fitness_scores[i]):
                        domination_count[i] += 1
            
            if domination_count[i] == 0:
                fronts[0].append(i)
        
        current_front = 0
        while fronts[current_front]:
            next_front = []
            for i in fronts[current_front]:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            
            if next_front:
                fronts.append(next_front)
                current_front += 1
            else:
                break
        
        return fronts
    
    def _dominates(self, fitness1, fitness2):
        """
        判断一个解是否支配另一个解
        """
        at_least_one_better = False
        for i in range(len(fitness1)):
            if fitness1[i] < fitness2[i]:
                return False
            elif fitness1[i] > fitness2[i]:
                at_least_one_better = True
        
        return at_least_one_better
    
    def _calculate_crowding_distance(self, fronts, fitness_scores):
        """
        计算拥挤度距离
        """
        crowding_distances = [0] * len(fitness_scores)
        
        for front in fronts:
            if len(front) <= 2:
                for i in front:
                    crowding_distances[i] = float('inf')
                continue
            
            # 对每个目标计算拥挤度
            for obj in range(len(fitness_scores[0])):
                # 按目标值排序
                sorted_front = sorted(front, key=lambda x: fitness_scores[x][obj])
                
                # 边界解设置无穷大拥挤度
                crowding_distances[sorted_front[0]] = float('inf')
                crowding_distances[sorted_front[-1]] = float('inf')
                
                # 计算中间解的拥挤度
                obj_range = fitness_scores[sorted_front[-1]][obj] - fitness_scores[sorted_front[0]][obj]
                if obj_range == 0:
                    continue
                
                for i in range(1, len(sorted_front) - 1):
                    crowding_distances[sorted_front[i]] += (
                        fitness_scores[sorted_front[i+1]][obj] - 
                        fitness_scores[sorted_front[i-1]][obj]
                    ) / obj_range
        
        return crowding_distances
    
    def _nsga2_selection_crossover_mutation(self, population, fitness_scores, fronts, crowding_distances):
        """
        NSGA-II的选择、交叉、变异操作
        """
        # 选择
        selected = self._tournament_selection(population, fitness_scores, fronts, crowding_distances)
        
        # 交叉
        offspring = self._crossover(selected)
        
        # 变异
        offspring = self._mutation(offspring)
        
        return offspring
    
    def _tournament_selection(self, population, fitness_scores, fronts, crowding_distances):
        """
        锦标赛选择
        """
        selected = []
        for _ in range(len(population)):
            # 随机选择两个个体
            idx1, idx2 = np.random.choice(len(population), 2, replace=False)
            
            # 比较等级和拥挤度
            if self._compare_individuals(idx1, idx2, fronts, crowding_distances) < 0:
                selected.append(population[idx1])
            else:
                selected.append(population[idx2])
        
        return selected
    
    def _compare_individuals(self, idx1, idx2, fronts, crowding_distances):
        """
        比较两个个体的优劣
        """
        # 找到个体所在的等级
        rank1 = next(i for i, front in enumerate(fronts) if idx1 in front)
        rank2 = next(i for i, front in enumerate(fronts) if idx2 in front)
        
        if rank1 != rank2:
            return rank1 - rank2  # 等级越低越好
        
        # 等级相同时，拥挤度越大越好
        return crowding_distances[idx2] - crowding_distances[idx1]
    
    def _crossover(self, population):
        """
        交叉操作
        """
        offspring = []
        for i in range(0, len(population), 2):
            if i + 1 < len(population):
                if np.random.random() < self.crossover_rate:
                    # 单点交叉
                    parent1, parent2 = population[i], population[i+1]
                    crossover_point = np.random.randint(1, len(parent1))
                    
                    child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                    child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
                    
                    offspring.extend([child1, child2])
                else:
                    offspring.extend([population[i], population[i+1]])
            else:
                offspring.append(population[i])
        
        return offspring
    
    def _mutation(self, population):
        """
        变异操作
        """
        for individual in population:
            for i in range(len(individual)):
                if np.random.random() < self.mutation_rate:
                    # 随机变异
                    individual[i] = np.random.randint(0, len(np.unique(individual)))
        
        return population
    
    def _check_convergence(self, fitness_scores, generation):
        """
        检查收敛性
        """
        if generation < 10:
            return False
        
        # 计算最近几代的平均适应度变化
        recent_scores = fitness_scores[-10:]
        avg_fitness = np.mean([np.mean(score) for score in recent_scores])
        
        if generation >= 20:
            older_scores = fitness_scores[-20:-10]
            older_avg_fitness = np.mean([np.mean(score) for score in older_scores])
            
            if abs(avg_fitness - older_avg_fitness) < self.convergence_threshold:
                return True
        
        return False
    
    def _select_best_individual(self, population, fitness_scores):
        """
        选择最优个体
        """
        # 选择第一个前沿中的第一个个体
        fronts = self._non_dominated_sort(fitness_scores)
        if fronts and fronts[0]:
            return population[fronts[0][0]]
        
        # 如果没有前沿，选择平均适应度最高的个体
        avg_fitness = [np.mean(score) for score in fitness_scores]
        best_idx = np.argmax(avg_fitness)
        return population[best_idx]
    
    def _extract_pareto_front(self, population, fitness_scores):
        """
        提取Pareto前沿
        """
        fronts = self._non_dominated_sort(fitness_scores)
        pareto_front = []
        
        for front in fronts:
            for idx in front:
                pareto_front.append({
                    'individual': population[idx],
                    'fitness': fitness_scores[idx]
                })
        
        return pareto_front
    
    def real_time_strategy_adjustment(self, cluster_labels, distances, target_client_id=0):
        """
        实时策略调整，根据当前性能动态调整分簇策略
        
        Args:
            cluster_labels: 当前分簇标签
            distances: 距离矩阵
            target_client_id: 目标客户端ID
            
        Returns:
            调整后的分簇标签
        """
        if not self.real_time_adjustment:
            print("实时策略调整未启用，返回原始分簇结果")
            return cluster_labels
        
        print("开始实时策略调整...")
        
        try:
            # 分析当前性能
            current_performance = self._analyze_current_performance(cluster_labels, distances, target_client_id)
            print(f"  当前性能分析完成: {current_performance}")
            
            # 判断是否需要调整
            if self._needs_adjustment(current_performance):
                print("  检测到需要调整策略")
                
                # 检查调整冷却期
                if len(self.adjustment_history) > 0:
                    last_adjustment = self.adjustment_history[-1]
                    if current_performance['epoch'] - last_adjustment['epoch'] < self.adjustment_cooldown:
                        print(f"  调整冷却期中，跳过本次调整")
                        return cluster_labels
                
                # 执行调整策略
                adjusted_labels = self._execute_adjustment_strategy(
                    cluster_labels, distances, target_client_id, current_performance
                )
                
                # 记录调整历史
                self._record_adjustment_history(current_performance, adjusted_labels)
                
                print("  实时策略调整完成")
                return adjusted_labels
            else:
                print("  当前性能良好，无需调整策略")
                return cluster_labels
                
        except Exception as e:
            print(f"实时策略调整失败: {e}")
            print("返回原始分簇结果")
            return cluster_labels
    
    def _analyze_current_performance(self, cluster_labels, distances, target_client_id):
        """
        分析当前性能
        """
        # 计算遗忘效果
        forget_score = self._evaluate_forget_effect(cluster_labels, distances, target_client_id)
        
        # 计算保护效果
        protection_score = self._evaluate_protection_effect(cluster_labels, distances, target_client_id)
        
        # 计算簇间平衡
        cluster_analysis = self._analyze_cluster_balance(cluster_labels, distances)
        balance_score = self._calculate_balance_score(cluster_analysis)
        
        # 计算综合性能
        overall_performance = (forget_score + protection_score + balance_score) / 3
        
        return {
            'forget_score': forget_score,
            'protection_score': protection_score,
            'balance_score': balance_score,
            'overall_performance': overall_performance,
            'timestamp': time.time()
        }
    
    def _needs_adjustment(self, current_performance):
        """
        判断是否需要调整
        """
        if not self.adjustment_history:
            return False
        
        # 计算性能变化趋势
        recent_performances = [h['overall_performance'] for h in self.adjustment_history[-self.performance_window:]]
        
        if len(recent_performances) < 2:
            return False
        
        # 计算性能下降率
        performance_trend = np.polyfit(range(len(recent_performances)), recent_performances, 1)[0]
        
        # 如果性能持续下降且下降率超过阈值，则需要调整
        if performance_trend < -self.adjustment_threshold:
            return True
        
        return False
    
    def _execute_adjustment_strategy(self, cluster_labels, distances, target_client_id, current_performance):
        """
        执行调整策略
        """
        print("      执行调整策略...")
        
        # 策略1：调整权重
        self._adjust_optimization_weights(current_performance)
        
        # 策略2：重新分簇
        adjusted_labels = self._recluster_with_adjusted_strategy(
            cluster_labels, distances, target_client_id
        )
        
        # 策略3：优化簇间平衡
        adjusted_labels = self.balance_clusters_dynamically(
            adjusted_labels, distances, target_client_id
        )
        
        return adjusted_labels
    
    def _adjust_optimization_weights(self, current_performance):
        """
        调整优化权重
        """
        # 根据性能表现调整权重
        if current_performance['forget_score'] < 0.5:
            # 遗忘效果差，增加遗忘效果权重
            self.balance_weights['forget_effect'] = min(0.6, self.balance_weights['forget_effect'] * 1.1)
            self.balance_weights['protection_effect'] *= 0.95
            self.balance_weights['cluster_balance'] *= 0.95
        
        if current_performance['protection_score'] < 0.5:
            # 保护效果差，增加保护效果权重
            self.balance_weights['protection_effect'] = min(0.6, self.balance_weights['protection_effect'] * 1.1)
            self.balance_weights['forget_effect'] *= 0.95
            self.balance_weights['cluster_balance'] *= 0.95
        
        # 归一化权重
        total_weight = sum(self.balance_weights.values())
        for key in self.balance_weights:
            self.balance_weights[key] /= total_weight
        
        print(f"        权重已调整: {self.balance_weights}")
    
    def _recluster_with_adjusted_strategy(self, cluster_labels, distances, target_client_id):
        """
        使用调整后的策略重新分簇
        """
        # 使用调整后的权重重新优化
        adjusted_labels = self.joint_optimize_clustering(
            cluster_labels, distances, target_client_id
        )
        
        return adjusted_labels
    
    def _record_adjustment_history(self, performance, adjusted_labels):
        """
        记录调整历史
        """
        adjustment_record = {
            'timestamp': time.time(),
            'performance_before': performance,
            'adjustment_type': 'real_time',
            'new_labels': adjusted_labels.tolist()
        }
        
        self.adjustment_history.append(adjustment_record)
        
        # 保持历史记录在合理范围内
        if len(self.adjustment_history) > 100:
            self.adjustment_history = self.adjustment_history[-50:]
    
    def performance_prediction_and_warning(self, cluster_labels, distances, target_client_id=0):
        """
        性能预测和预警系统，使用机器学习模型预测未来性能并生成预警
        
        Args:
            cluster_labels: 当前分簇标签
            distances: 距离矩阵
            target_client_id: 目标客户端ID
            
        Returns:
            预测结果和预警信息
        """
        if not self.performance_prediction:
            print("性能预测和预警未启用")
            return None
        
        print("开始性能预测和预警分析...")
        
        try:
            # 训练预测模型（如果需要）
            if self.prediction_model is None:
                print("  训练性能预测模型...")
                self._train_prediction_model()
            
            # 进行性能预测
            predictions = self._predict_performance(cluster_labels, distances, target_client_id)
            print(f"  性能预测完成: {predictions}")
            
            # 生成预警信息
            warnings = self._generate_warnings(predictions)
            if warnings:
                print(f"  生成预警信息: {len(warnings)}个预警")
                for warning in warnings:
                    print(f"    {warning['level']}: {warning['message']}")
            
            # 记录预测历史
            self._record_prediction_history(predictions, warnings)
            
            return {
                'predictions': predictions,
                'warnings': warnings,
                'timestamp': time.time()
            }
            
        except Exception as e:
            print(f"性能预测和预警失败: {e}")
            return None
    
    def _train_prediction_model(self):
        """
        训练预测模型
        """
        print("    训练性能预测模型...")
        
        # 这里使用简单的线性回归模型作为示例
        # 在实际应用中可以使用更复杂的模型如LSTM、Transformer等
        from sklearn.linear_model import LinearRegression
        
        if len(self.performance_metrics['accuracy']) > 10:
            # 准备训练数据
            X = np.array(self.performance_metrics['accuracy'][:-1]).reshape(-1, 1)
            y = np.array(self.performance_metrics['accuracy'][1:])
            
            # 训练模型
            self.prediction_model = LinearRegression()
            self.prediction_model.fit(X, y)
            
            print("      预测模型训练完成")
        else:
            print("      数据不足，无法训练预测模型")
    
    def _predict_performance(self, cluster_labels, distances, target_client_id):
        """
        进行性能预测
        """
        if self.prediction_model is None:
            return None
        
        # 基于当前状态预测未来性能
        current_accuracy = self._evaluate_forget_effect(cluster_labels, distances, target_client_id)
        
        # 预测未来几个时间步的性能
        predictions = []
        current_input = np.array([[current_accuracy]])
        
        for step in range(5):  # 预测未来5步
            next_prediction = self.prediction_model.predict(current_input)[0]
            predictions.append({
                'step': step + 1,
                'predicted_accuracy': max(0, min(1, next_prediction)),  # 限制在[0,1]范围内
                'confidence': 0.8 - step * 0.1  # 置信度递减
            })
            current_input = np.array([[next_prediction]])
        
        return predictions
    
    def _generate_warnings(self, predictions):
        """
        生成预警信息
        """
        if not predictions:
            return []
        
        warnings = []
        
        for pred in predictions:
            accuracy = pred['predicted_accuracy']
            step = pred['step']
            
            if accuracy < self.warning_thresholds['critical']:
                warnings.append({
                    'level': 'critical',
                    'message': f'第{step}步预测性能严重下降: {accuracy:.3f}',
                    'step': step,
                    'predicted_value': accuracy
                })
            elif accuracy < self.warning_thresholds['warning']:
                warnings.append({
                    'level': 'warning',
                    'message': f'第{step}步预测性能下降: {accuracy:.3f}',
                    'step': step,
                    'predicted_value': accuracy
                })
            elif accuracy < self.warning_thresholds['notice']:
                warnings.append({
                    'level': 'notice',
                    'message': f'第{step}步预测性能需要注意: {accuracy:.3f}',
                    'step': step,
                    'predicted_value': accuracy
                })
        
        return warnings
    
    def _record_prediction_history(self, predictions, warnings):
        """
        记录预测历史
        """
        record = {
            'timestamp': time.time(),
            'predictions': predictions,
            'warnings': warnings
        }
        
        self.prediction_history.append(record)
        
        # 保持历史记录在合理范围内
        if len(self.prediction_history) > 100:
            self.prediction_history = self.prediction_history[-50:]
    
    def enhanced_client_performance_monitoring(self, client_id, performance_data):
        """
        增强客户端性能监控，实时跟踪和分析客户端性能
        
        Args:
            client_id: 客户端ID
            performance_data: 性能数据字典，包含accuracy, loss等指标
            
        Returns:
            监控结果和建议
        """
        if not self.client_monitoring:
            print("增强客户端监控未启用")
            return None
        
        print(f"开始监控客户端 {client_id} 的性能...")
        
        try:
            # 更新客户端性能跟踪器
            if client_id not in self.client_performance_tracker:
                self.client_performance_tracker[client_id] = {
                    'accuracy_history': [],
                    'loss_history': [],
                    'convergence_history': [],
                    'stability_history': [],
                    'last_update': time.time(),
                    'intervention_count': 0
                }
            
            tracker = self.client_performance_tracker[client_id]
            
            # 记录性能数据
            for metric_name, metric_value in performance_data.items():
                if metric_name in tracker:
                    tracker[metric_name].append(metric_value)
                    # 保持历史记录在合理范围内
                    if len(tracker[metric_name]) > 100:
                        tracker[metric_name] = tracker[metric_name][-50:]
            
            tracker['last_update'] = time.time()
            
            # 分析性能趋势
            trend_analysis = self._analyze_client_performance_trend(client_id)
            print(f"  性能趋势分析: {trend_analysis}")
            
            # 检查是否需要干预
            if trend_analysis['needs_intervention']:
                print(f"  客户端 {client_id} 需要干预")
                self._trigger_client_intervention(client_id, trend_analysis)
                
                # 调整分簇策略（如果需要）
                if trend_analysis['trend'] in ['declining', 'unstable']:
                    self._adjust_clustering_for_client(client_id)
            
            # 更新全局性能指标
            self._update_global_performance_metrics(client_id, performance_data)
            
            # 生成监控报告
            monitoring_report = {
                'client_id': client_id,
                'current_performance': performance_data,
                'trend_analysis': trend_analysis,
                'intervention_needed': trend_analysis['needs_intervention'],
                'monitoring_timestamp': time.time()
            }
            
            print(f"  客户端 {client_id} 监控完成")
            return monitoring_report
            
        except Exception as e:
            print(f"客户端 {client_id} 监控失败: {e}")
            return None
    
    def _analyze_client_performance_trend(self, client_id):
        """
        分析客户端性能趋势
        """
        tracker = self.client_performance_tracker[client_id]
        
        if len(tracker['accuracy_history']) < 3:
            return {'needs_intervention': False, 'trend': 'insufficient_data'}
        
        # 计算准确率趋势
        recent_accuracy = tracker['accuracy_history'][-3:]
        accuracy_trend = np.polyfit(range(len(recent_accuracy)), recent_accuracy, 1)[0]
        
        # 计算稳定性
        if len(tracker['accuracy_history']) >= 5:
            recent_5 = tracker['accuracy_history'][-5:]
            stability = 1.0 - np.std(recent_5)
        else:
            stability = 0.5
        
        # 判断是否需要干预
        needs_intervention = False
        trend = 'stable'
        
        if accuracy_trend < -0.05:  # 准确率下降超过5%
            needs_intervention = True
            trend = 'declining'
        elif accuracy_trend > 0.05:  # 准确率上升超过5%
            trend = 'improving'
        
        if stability < 0.3:  # 稳定性差
            needs_intervention = True
            trend = 'unstable'
        
        return {
            'needs_intervention': needs_intervention,
            'trend': trend,
            'accuracy_trend': accuracy_trend,
            'stability': stability
        }
    
    def _trigger_client_intervention(self, client_id, trend_analysis):
        """
        触发客户端干预
        """
        print(f"    客户端 {client_id} 需要干预，趋势: {trend_analysis['trend']}")
        
        # 根据趋势类型选择干预策略
        if trend_analysis['trend'] == 'declining':
            # 性能下降，增加保护
            self._increase_client_protection(client_id)
        elif trend_analysis['trend'] == 'unstable':
            # 性能不稳定，调整分簇策略
            self._adjust_clustering_for_client(client_id)
    
    def _adjust_clustering_for_client(self, client_id):
        """
        为特定客户端调整分簇策略
        """
        # 这里可以实现针对特定客户端的个性化分簇调整
        # 例如，将性能不稳定的客户端分配到更稳定的簇中
        print(f"      为客户端 {client_id} 调整分簇策略")
    
    def _update_global_performance_metrics(self, client_id, performance_data):
        """
        更新全局性能指标
        """
        for metric_name, metric_values in self.performance_metrics.items():
            if metric_name in performance_data:
                metric_values.append(performance_data[metric_name])
                
                # 保持指标历史在合理范围内
                if len(metric_values) > 100:
                    self.performance_metrics[metric_name] = metric_values[-50:]
    
    def get_third_stage_report(self):
        """
        获取第三阶段功能报告
        
        Returns:
            第三阶段功能状态报告
        """
        report = {
            'advanced_multi_objective': {
                'enabled': self.advanced_multi_objective,
                'algorithm': self.optimization_algorithm,
                'population_size': self.population_size,
                'generation_limit': self.generation_limit,
                'crossover_rate': self.crossover_rate,
                'mutation_rate': self.mutation_rate,
                'tournament_size': self.tournament_size,
                'elite_size': self.elite_size,
                'optimization_cache': list(self.optimization_cache.keys()) if self.optimization_cache else []
            },
            'real_time_adjustment': {
                'enabled': self.real_time_adjustment,
                'adjustment_threshold': self.adjustment_threshold,
                'performance_window': self.performance_window,
                'adjustment_cooldown': self.adjustment_cooldown,
                'adjustment_history_length': len(self.adjustment_history),
                'last_adjustment': self.adjustment_history[-1] if self.adjustment_history else None
            },
            'performance_prediction': {
                'enabled': self.performance_prediction,
                'model_trained': self.prediction_model is not None,
                'prediction_history_length': len(self.prediction_history),
                'warning_thresholds': self.warning_thresholds,
                'last_prediction': self.prediction_history[-1] if self.prediction_history else None
            },
            'client_monitoring': {
                'enabled': self.client_monitoring,
                'monitored_clients': list(self.client_performance_tracker.keys()),
                'monitoring_frequency': self.monitoring_frequency,
                'global_metrics': {k: len(v) for k, v in self.performance_metrics.items()},
                'intervention_history_length': len(self.intervention_history),
                'active_interventions': sum(1 for client_id, tracker in self.client_performance_tracker.items() 
                                         if tracker.get('intervention_count', 0) > 0)
            },
            'system_performance': {
                'distance_cache_size': len(self.distance_cache),
                'clustering_cache_size': len(self.clustering_cache),
                'optimization_cache_size': len(self.optimization_cache),
                'total_memory_usage': 'N/A'  # 可以添加实际内存使用统计
            },
            'feature_summary': {
                'total_features': 4,  # 第三阶段的4个主要功能
                'enabled_features': sum([
                    self.advanced_multi_objective,
                    self.real_time_adjustment,
                    self.performance_prediction,
                    self.client_monitoring
                ]),
                'feature_status': {
                    'advanced_multi_objective': '✓' if self.advanced_multi_objective else '✗',
                    'real_time_adjustment': '✓' if self.real_time_adjustment else '✗',
                    'performance_prediction': '✓' if self.performance_prediction else '✗',
                    'client_monitoring': '✓' if self.client_monitoring else '✗'
                }
            }
        }
        
        return report
