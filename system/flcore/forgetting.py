import copy
import torch
import numpy as np
import torch.nn.functional as F
import warnings
import sys
import os
import json
import time
import hashlib
import psutil
import random

# 抑制警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*threadpoolctl.*")

# 创建一个临时的stderr重定向类
class SuppressStderr:
    def __enter__(self):
        self.original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stderr = self.original_stderr

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.stats import wasserstein_distance
from .enhanced_clustering_strategy import EnhancedClusteringStrategy


class OptimizationConfig:
    """优化配置管理类，统一管理所有优化参数"""
    
    def __init__(self, args):
        self.args = args
        
        # 梯度计算优化参数
        self.sample_ratio = getattr(args, 'sample_ratio', 0.2)  # 数据采样比例
        self.max_batches = getattr(args, 'max_batches', 4)      # 最大批次数
        self.num_processes = getattr(args, 'num_processes', 2)  # 并行进程数
        
        # 缓存优化参数
        self.use_gradient_caching = getattr(args, 'use_gradient_caching', True)
        self.max_cache_size = getattr(args, 'max_cache_size', 10)
        self.cache_cleanup_frequency = getattr(args, 'cache_cleanup_frequency', 5)
        
        # 计算优化参数
        self.use_parallel_computation = getattr(args, 'use_parallel_computation', True)
        self.min_clients_for_parallel = getattr(args, 'min_clients_for_parallel', 3)
        self.max_parallel_processes = getattr(args, 'max_parallel_processes', 4)
        
        # 内存优化参数
        self.use_gradient_accumulation = getattr(args, 'use_gradient_accumulation', True)
        self.use_memory_cleanup = getattr(args, 'use_memory_cleanup', True)
        self.memory_cleanup_frequency = getattr(args, 'memory_cleanup_frequency', 3)
        
        # 验证参数合理性
        self._validate_parameters()
    
    def _validate_parameters(self):
        """验证参数合理性并调整"""
        # 确保采样比例在合理范围内
        self.sample_ratio = max(0.1, min(0.5, self.sample_ratio))
        
        # 确保批次数在合理范围内
        self.max_batches = max(2, min(8, self.max_batches))
        
        # 确保进程数在合理范围内
        self.num_processes = max(1, min(4, self.num_processes))
        
        # 确保缓存大小在合理范围内
        self.max_cache_size = max(5, min(20, self.max_cache_size))
        
        print(f"优化配置已加载:")
        print(f"  - 数据采样比例: {self.sample_ratio}")
        print(f"  - 最大批次数: {self.max_batches}")
        print(f"  - 并行进程数: {self.num_processes}")
        print(f"  - 梯度缓存: {'启用' if self.use_gradient_caching else '禁用'}")
        print(f"  - 最大缓存大小: {self.max_cache_size}")
    
    def get_optimization_summary(self):
        """获取优化配置摘要"""
        return {
            'sample_ratio': self.sample_ratio,
            'max_batches': self.max_batches,
            'num_processes': self.num_processes,
            'use_gradient_caching': self.use_gradient_caching,
            'max_cache_size': self.max_cache_size,
            'use_parallel_computation': self.use_parallel_computation,
            'use_gradient_accumulation': self.use_gradient_accumulation
        }


class ClientForgetting:
    """客户端遗忘模块"""
    
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.num_clients = args.num_clients
        self.target_client_id = args.target_client_id 
        
        # 初始化增强分簇策略
        self.enhanced_clustering = EnhancedClusteringStrategy(args)
        
    def extract_client_features(self, global_model, clients):
        """提取客户端特征用于分簇"""
        features_list = []
        for client in clients:
            # 使用全局模型提取客户端数据的特征
            client.set_parameters(global_model)
            feature = self._extract_feature_from_client(client)
            features_list.append(feature)
        return features_list
    
    def _extract_feature_from_client(self, client):
        """从客户端数据提取特征（最后一层输出均值）"""
        client.model.eval()
        features = []
        
        with torch.no_grad():
            trainloader = client.load_train_data()
            for batch_idx, (data, target) in enumerate(trainloader):
                if type(data) == type([]):
                    data = data[0].to(self.device)
                else:
                    data = data.to(self.device)
                
                # 获取倒数第二层的输出（特征）
                if hasattr(client.model, 'head'):
                    # 对于BaseHeadSplit模型
                    feature = client.model.base(data)
                else:
                    # 对于普通模型，移除最后一层
                    x = data
                    for name, module in client.model.named_modules():
                        if name == 'fc' or name == 'classifier':
                            break
                        x = module(x)
                    feature = x.view(x.size(0), -1)
                
                features.append(feature.mean(dim=0).cpu().numpy())
                
                # 限制批次数以控制计算量
                if batch_idx >= 5:
                    break
        
        return np.mean(features, axis=0)
    
    def compute_wasserstein_distances(self, features_list):
        """计算客户端之间的Wasserstein距离，用于分布差异分簇"""
        num_clients = len(features_list)
        distances = np.zeros((num_clients, num_clients))
        
        print("计算客户端间的Wasserstein距离...")
        
        try:
            for i in range(num_clients):
                for j in range(i+1, num_clients):
                    # 计算两个客户端特征分布之间的Wasserstein距离
                    dist = self._compute_distribution_distance(features_list[i], features_list[j])
                    distances[i][j] = dist
                    distances[j][i] = dist
                    print(f"客户端{i}与客户端{j}的Wasserstein距离: {dist:.4f}")
        except Exception as e:
            print(f"计算Wasserstein距离时出现异常: {e}")
            # 如果计算失败，使用简化的距离度量
            for i in range(num_clients):
                for j in range(i+1, num_clients):
                    dist = self._compute_simple_distance(features_list[i], features_list[j])
                    distances[i][j] = dist
                    distances[j][i] = dist
        
        return distances
    
    def _compute_distribution_distance(self, features1, features2):
        """计算两个特征分布之间的Wasserstein距离"""
        try:
            # 将特征转换为1D分布进行比较
            # 使用特征向量的范数作为分布样本
            dist1 = np.linalg.norm(features1, axis=1)
            dist2 = np.linalg.norm(features2, axis=1)
            
            # 计算Wasserstein距离
            from scipy.stats import wasserstein_distance
            w_dist = wasserstein_distance(dist1, dist2)
            return w_dist
        except Exception as e:
            print(f"Wasserstein距离计算失败: {e}")
            return self._compute_simple_distance(features1, features2)
    
    def _compute_simple_distance(self, features1, features2):
        """计算简化的分布距离（备选方案）"""
        # 使用特征均值的欧几里得距离
        mean1 = np.mean(features1, axis=0)
        mean2 = np.mean(features2, axis=0)
        return np.linalg.norm(mean1 - mean2)
    
    def cluster_clients(self, features_list, distances):
        """基于增强分簇策略的客户端分簇"""
        try:
            print("开始基于增强分簇策略的客户端分簇...")
            
            # 使用增强分簇策略
            cluster_labels, anomaly_clients = self.enhanced_clustering.cluster_clients_enhanced(
                features_list, distances, self.target_client_id
            )
            
            return cluster_labels, anomaly_clients
            
        except Exception as e:
            print(f"增强分簇过程中出现异常: {e}")
            print("回退到传统分簇方法...")
            # 使用简单的距离分簇作为备选方案
            return self._simple_distance_clustering(distances)
    
    def _simple_distance_clustering(self, distances):
        """简单的距离分簇方法（备选方案）"""
        print("使用简单距离分簇方法...")
        num_clients = len(distances)
        cluster_labels = np.zeros(num_clients, dtype=int)
        current_cluster = 0
        
        # 目标客户端单独成簇
        cluster_labels[self.target_client_id] = current_cluster
        current_cluster += 1
        
        # 其他客户端基于与目标客户端的距离分簇
        target_distances = []
        for i in range(num_clients):
            if i != self.target_client_id:
                target_distances.append((i, distances[self.target_client_id][i]))
        
        target_distances.sort(key=lambda x: x[1])
        
        # 简单分组：前1/3高相似，中间1/3中等相似，后1/3低相似
        group_size = len(target_distances) // 3
        
        for i, (client_id, distance) in enumerate(target_distances):
            if i < group_size:
                cluster_labels[client_id] = current_cluster
            elif i < 2 * group_size:
                cluster_labels[client_id] = current_cluster + 1
            else:
                cluster_labels[client_id] = current_cluster + 2
        
        # 调整簇标签
        unique_clusters = np.unique(cluster_labels)
        for i, cluster_id in enumerate(unique_clusters):
            cluster_labels[cluster_labels == cluster_id] = i
        
        print(f"简单分簇完成，共{len(unique_clusters)}个簇")
        return cluster_labels, []
    
    def _detect_anomaly_clients(self, features, cluster_labels, distances):
        """改进的异常客户端检测"""
        anomaly_clients = []
        
        # 方法1：基于距离中心的距离
        for i in range(len(features)):
            cluster_id = cluster_labels[i]
            cluster_members = np.where(cluster_labels == cluster_id)[0]
            
            if len(cluster_members) > 1:
                # 计算到簇中心的距离
                cluster_center = np.mean(features[cluster_members], axis=0)
                distance_to_center = np.linalg.norm(features[i] - cluster_center)
                
                # 计算簇内平均距离
                cluster_distances = []
                for j in cluster_members:
                    if i != j:
                        cluster_distances.append(np.linalg.norm(features[i] - features[j]))
                
                if len(cluster_distances) > 0:
                    avg_cluster_distance = np.mean(cluster_distances)
                    if distance_to_center > 2.0 * avg_cluster_distance:
                        anomaly_clients.append(i)
        
        # 方法2：基于Wasserstein距离的异常检测
        for i in range(len(features)):
            distances_to_others = []
            for j in range(len(features)):
                if i != j:
                    distances_to_others.append(distances[i][j])
            
            if len(distances_to_others) > 0:
                mean_distance = np.mean(distances_to_others)
                std_distance = np.std(distances_to_others)
                
                # 如果距离超过2个标准差，认为是异常
                if any(d > mean_distance + 2 * std_distance for d in distances_to_others):
                    if i not in anomaly_clients:
                        anomaly_clients.append(i)
        
        return list(set(anomaly_clients))  # 去重
    
    def generate_forget_mask(self, global_model, target_client, top_ratio=0.15):
        """改进的遗忘掩码生成"""
        # 计算目标客户端的梯度残差
        gradients = self._compute_gradient_residual(global_model, target_client)
        
        # 使用自适应阈值选择
        all_params = []
        for grad in gradients:
            all_params.extend(grad.flatten().cpu().numpy())
        
        # 使用更激进的参数选择策略
        abs_params = np.abs(all_params)
        
        # 动态调整top_ratio
        if len(abs_params) > 0:
            # 基于梯度分布自适应调整
            mean_grad = np.mean(abs_params)
            std_grad = np.std(abs_params)
            
            # 如果梯度分布比较集中，增加选择比例
            if std_grad < mean_grad * 0.5:
                top_ratio = min(0.25, top_ratio * 1.5)
            elif std_grad > mean_grad * 2.0:
                top_ratio = max(0.05, top_ratio * 0.8)
        
        # 计算阈值
        threshold = np.percentile(abs_params, (1 - top_ratio) * 100)
        
        # 生成掩码
        mask = []
        for grad in gradients:
            mask_param = (torch.abs(grad) >= threshold).float()
            mask.append(mask_param)
        
        return mask
    
    def _compute_gradient_residual(self, global_model, target_client):
        """计算梯度残差"""
        target_client.set_parameters(global_model)
        target_client.model.train()
        
        gradients = []
        trainloader = target_client.load_train_data()
        
        # 计算梯度
        for batch_idx, (data, target) in enumerate(trainloader):
            if type(data) == type([]):
                data = data[0].to(self.device)
            else:
                data = data.to(self.device)
            target = target.to(self.device)
            
            output = target_client.model(data)
            loss = target_client.loss(output, target)
            
            # 计算梯度
            target_client.optimizer.zero_grad()
            loss.backward()
            
            # 收集梯度
            batch_gradients = []
            for param in target_client.model.parameters():
                if param.grad is not None:
                    batch_gradients.append(param.grad.clone())
            
            gradients.append(batch_gradients)
            
            # 限制批次数
            if batch_idx >= 3:
                break
        
        # 平均梯度
        avg_gradients = []
        for param_idx in range(len(gradients[0])):
            avg_grad = torch.zeros_like(gradients[0][param_idx])
            for batch_grads in gradients:
                avg_grad += batch_grads[param_idx]
            avg_gradients.append(avg_grad / len(gradients))
        
        return avg_gradients
    
    def adversarial_training(self, global_model, mask, target_client, other_clients, 
                           epochs=15, lr=0.005, lambda_forget=1.5, lambda_retain=1.0, 
                           patience=5, min_delta=0.001):
        """改进的对抗掩码训练，减少对其他客户端的影响"""
        model = copy.deepcopy(global_model)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        
        best_loss = float('inf')
        patience_counter = 0
        training_history = []
        
        for epoch in range(epochs):
            # 遗忘损失：最大化目标客户端的损失
            forget_loss = self._compute_enhanced_forget_loss(model, target_client, mask)
            
            # 保留损失：最小化其他客户端的损失
            retain_loss = self._compute_enhanced_retain_loss(model, other_clients)
            
            # 添加KL散度损失来增强遗忘效果
            kl_loss = self._compute_kl_divergence_loss(model, global_model, target_client)
            
            # 总损失：减少对其他客户端的影响
            total_loss = lambda_forget * forget_loss - lambda_retain * retain_loss + 0.3 * kl_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            
            # 学习率调度
            scheduler.step(total_loss)
            
            training_history.append({
                'epoch': epoch,
                'forget_loss': forget_loss.item(),
                'retain_loss': retain_loss.item(),
                'kl_loss': kl_loss.item(),
                'total_loss': total_loss.item()
            })
            
            # 早停检查
            if total_loss < best_loss - min_delta:
                best_loss = total_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if epoch % 2 == 0:
                print(f"Epoch {epoch}: Forget Loss: {forget_loss.item():.4f}, "
                      f"Retain Loss: {retain_loss.item():.4f}, "
                      f"KL Loss: {kl_loss.item():.4f}, "
                      f"Total Loss: {total_loss.item():.4f}")
            
            # 早停
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        return model
    
    def _compute_enhanced_forget_loss(self, model, target_client, mask):
        """增强的遗忘损失计算"""
        target_client.set_parameters(model)
        target_client.model.train()
        
        total_loss = torch.tensor(0.0, requires_grad=True, device=self.device)
        num_batches = 0
        
        trainloader = target_client.load_train_data()
        for batch_idx, (data, target) in enumerate(trainloader):
            if type(data) == type([]):
                data = data[0].to(self.device)
            else:
                data = data.to(self.device)
            target = target.to(self.device)
            
            output = target_client.model(data)
            
            # 使用交叉熵损失
            ce_loss = target_client.loss(output, target)
            
            # 添加对抗损失：最大化预测熵
            probs = F.softmax(output, dim=1)
            entropy_loss = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=1))
            
            # 组合损失
            batch_loss = ce_loss + 0.5 * entropy_loss
            total_loss = total_loss + batch_loss
            num_batches += 1
            
            if batch_idx >= 8:  # 增加批次数
                break
        
        return total_loss / max(num_batches, 1)
    
    def _compute_enhanced_retain_loss(self, model, other_clients):
        """增强的保留损失计算"""
        total_loss = torch.tensor(0.0, requires_grad=True, device=self.device)
        num_clients = 0
        
        for client in other_clients:
            if client.id == self.target_client_id:
                continue
                
            client.set_parameters(model)
            client.model.train()
            
            trainloader = client.load_train_data()
            client_loss = torch.tensor(0.0, requires_grad=True, device=self.device)
            num_batches = 0
            
            for batch_idx, (data, target) in enumerate(trainloader):
                if type(data) == type([]):
                    data = data[0].to(self.device)
                else:
                    data = data.to(self.device)
                target = target.to(self.device)
                
                output = client.model(data)
                loss = client.loss(output, target)
                
                client_loss = client_loss + loss
                num_batches += 1
                
                if batch_idx >= 5:  # 增加批次数
                    break
            
            if num_batches > 0:
                total_loss = total_loss + client_loss / num_batches
                num_clients += 1
        
        return total_loss / max(num_clients, 1)
    
    def _compute_kl_divergence_loss(self, model, original_model, target_client):
        """计算KL散度损失，增强遗忘效果"""
        target_client.set_parameters(model)
        target_client.model.eval()
        
        original_client = copy.deepcopy(target_client)
        original_client.set_parameters(original_model)
        original_client.model.eval()
        
        kl_loss = torch.tensor(0.0, requires_grad=True, device=self.device)
        num_batches = 0
        
        trainloader = target_client.load_train_data()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(trainloader):
                if type(data) == type([]):
                    data = data[0].to(self.device)
                else:
                    data = data.to(self.device)
                
                # 获取原始模型和目标模型的输出
                with torch.no_grad():
                    original_output = original_client.model(data)
                    original_probs = F.softmax(original_output, dim=1)
                
                current_output = target_client.model(data)
                current_probs = F.softmax(current_output, dim=1)
                
                # 计算KL散度
                kl_div = F.kl_div(
                    torch.log(current_probs + 1e-8), 
                    original_probs, 
                    reduction='batchmean'
                )
                
                kl_loss = kl_loss + kl_div
                num_batches += 1
                
                if batch_idx >= 5:
                    break
        
        return kl_loss / max(num_batches, 1)
    
    def forget_client(self, global_model, clients, target_client_id=0):
        """主遗忘函数"""
        self.target_client_id = target_client_id
        print(f"\n============= 开始遗忘客户端 {target_client_id} =============")
        
        # 1. 提取客户端特征
        print("1. 提取客户端特征...")
        features = self.extract_client_features(global_model, clients)
        
        # 2. 计算Wasserstein距离
        print("2. 计算Wasserstein距离...")
        distances = self.compute_wasserstein_distances(features)
        
        # 3. 客户端分簇
        print("3. 客户端分簇...")
        cluster_labels, anomaly_clients = self.cluster_clients(features, distances)
        
        print(f"分簇结果: {cluster_labels}")
        print(f"异常客户端: {anomaly_clients}")
        
        # 4. 生成遗忘掩码
        print("4. 生成遗忘掩码...")
        target_client = clients[target_client_id]
        mask = self.generate_forget_mask(global_model, target_client)
        
        # 5. 对抗训练
        print("5. 开始对抗训练...")
        other_clients = [c for c in clients if c.id != target_client_id]
        forget_model = self.adversarial_training(global_model, mask, target_client, other_clients)
        
        print("遗忘完成！")
        return forget_model


def forget_client_wrapper(global_model, clients, args, target_client_id=0):
    """遗忘客户端的包装函数"""
    forget_module = ClientForgetting(args)
    return forget_module.forget_client(global_model, clients, target_client_id)


class GradientReversalForgetting:
    """基于梯度反转的遗忘策略"""
    
    def __init__(self, args):
        """初始化梯度反转遗忘"""
        self.args = args
        requested_device = str(getattr(args, 'device', 'cuda')).lower()
        if requested_device == 'cuda' and not torch.cuda.is_available():
            requested_device = 'cpu'
        self.device = torch.device(requested_device)
        self.target_client_id = getattr(args, 'target_client_id', 0)
        
        # 初始化优化配置
        self.opt_config = OptimizationConfig(args)
        if self.device.type == 'cuda':
            # CUDA 下通过 multiprocessing 传递模型/梯度容易触发 IPC 和显存问题，
            # 实际会导致 worker BrokenPipe，主进程随后被系统杀掉。
            self.opt_config.use_parallel_computation = False
            self.opt_config.num_processes = 1
            print("检测到 CUDA 环境，已禁用多进程梯度计算以避免进程间 CUDA 冲突。")
        
        # 初始化增强分簇策略
        self.enhanced_clustering = EnhancedClusteringStrategy(args)
        
        # 分簇策略配置
        self.use_hierarchical_clustering = getattr(args, 'use_hierarchical_clustering', True)
        self.use_adaptive_thresholds = getattr(args, 'use_adaptive_thresholds', True)
        self.use_cluster_based_forgetting = getattr(args, 'use_cluster_based_forgetting', True)
        
        # 保护强度配置
        self.protection_level = getattr(args, 'protection_level', 'weak')  # 'strong', 'moderate', 'weak'
        
        # 知识锚点保护配置
        self.use_knowledge_anchoring = getattr(args, 'use_knowledge_anchoring', True)
        self.anchor_weight = getattr(args, 'anchor_weight', 0.5)  # 锚点损失权重（调整为0.5，平衡遗忘和保护）
        self.anchor_sample_ratio = getattr(args, 'anchor_sample_ratio', 0.1)  # 锚点数据采样比例
        self.anchor_start_epoch = getattr(args, 'anchor_start_epoch', 0)  # 开始应用锚点保护的轮次（从第0轮开始）
        
        # 平衡改进配置
        self.use_selective_anchoring = getattr(args, 'use_selective_anchoring', True)  # 启用选择性锚点保护
        self.selective_protection_threshold = getattr(args, 'selective_protection_threshold', 0.6)  # 选择性保护阈值
        self.use_progressive_weight = getattr(args, 'use_progressive_weight', True)  # 启用渐进式权重
        self.use_forget_compensation = getattr(args, 'use_forget_compensation', True)  # 启用遗忘强度补偿
        self.use_intelligent_gradient = getattr(args, 'use_intelligent_gradient', True)  # 启用智能梯度组合

        # 遗忘后保留校准配置
        self.retain_calibration_rounds = max(0, int(getattr(args, 'fu_retain_calibration_rounds', 2)))
        self.retain_calibration_lr = float(getattr(args, 'fu_retain_calibration_lr', 4e-4))
        self.retain_calibration_batches = max(1, int(getattr(args, 'fu_retain_calibration_batches', 3)))
        self.mask_retain_scale = float(getattr(args, 'fu_mask_retain_scale', 0.12))
        self.similarity_boost = float(getattr(args, 'fu_similarity_boost', 1.2))
        
        print(f"分簇策略配置:")
        print(f"  - 多层次分簇: {self.use_hierarchical_clustering}")
        print(f"  - 自适应阈值: {self.use_adaptive_thresholds}")
        print(f"  - 基于分簇的遗忘: {self.use_cluster_based_forgetting}")
        print(f"  - 保护强度级别: {self.protection_level}")
        print(f"知识锚点保护配置:")
        print(f"  - 启用知识锚点: {self.use_knowledge_anchoring}")
        print(f"  - 锚点损失权重: {self.anchor_weight}")
        print(f"  - 锚点数据采样比例: {self.anchor_sample_ratio}")
        print(f"  - 锚点保护起始轮次: {self.anchor_start_epoch}")
        print(f"平衡改进配置:")
        print(f"  - 选择性锚点保护: {self.use_selective_anchoring}")
        print(f"  - 选择性保护阈值: {self.selective_protection_threshold}")
        print(f"  - 渐进式权重: {self.use_progressive_weight}")
        print(f"  - 遗忘强度补偿: {self.use_forget_compensation}")
        print(f"  - 智能梯度组合: {self.use_intelligent_gradient}")
        print(f"保留校准配置:")
        print(f"  - 校准轮数: {self.retain_calibration_rounds}")
        print(f"  - 校准学习率: {self.retain_calibration_lr}")
        print(f"  - 校准批次数: {self.retain_calibration_batches}")
        print(f"  - 掩码保留系数: {self.mask_retain_scale}")
        print(f"  - 相似客户端加权系数: {self.similarity_boost}")
        
        # 打印优化配置摘要
        opt_summary = self.opt_config.get_optimization_summary()
        print(f"计算优化配置:")
        for key, value in opt_summary.items():
            print(f"  - {key}: {value}")
        
        # 初始化性能监控
        self.performance_stats = {
            'gradient_computation_time': [],
            'cache_hit_rate': 0,
            'total_computations': 0,
            'cached_computations': 0,
            'memory_usage': []
        }
        
        # 知识锚点相关
        self.original_model = None
        self.anchor_clients = []
        self.anchor_data_cache = {}
    
    def forget_with_gradient_reversal(self, global_model, clients, target_client_id=0, 
                                    epochs=10, lr=0.001, lambda_reversal=0.3, pre_accuracies=None):  # 提升lambda_reversal默认值
        """使用改进的梯度反转进行遗忘"""
        print(f"\n============= 开始改进的梯度反转遗忘客户端 {target_client_id} =============")
        self.target_client_id = target_client_id
        
        # 1. 提取客户端特征
        print("1. 提取客户端特征...")
        features = self._extract_client_features(global_model, clients)
        
        # 2. 计算Wasserstein距离
        print("2. 计算Wasserstein距离...")
        distances = self._compute_wasserstein_distances(features)
        self._distance_protection_thresholds = self._compute_target_distance_thresholds(distances, target_client_id)
        
        # 3. 客户端分簇
        print("3. 客户端分簇...")
        cluster_labels, anomaly_clients = self.enhanced_clustering.cluster_clients_enhanced(
            features, distances, target_client_id
        )
        
        print(f"分簇结果: {cluster_labels}")
        print(f"异常客户端: {anomaly_clients}")
        
        # 4. 生成遗忘掩码
        print("4. 生成遗忘掩码...")
        target_client = clients[target_client_id]
        mask = self._generate_forget_mask(global_model, target_client)
        
        # 5. 改进的梯度反转训练
        print("5. 开始改进的梯度反转训练...")
        model = copy.deepcopy(global_model)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        
        other_clients = [c for c in clients if c.id != target_client_id]
        
        # 计算其他客户端的重要性权重（使用改进的方法）
        other_weights = self._compute_client_importance_weights(target_client, other_clients, distances, cluster_labels)
        weight_denominator = float(np.sum(other_weights)) if len(other_weights) > 0 else 1.0
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # 计算目标客户端的梯度
            target_gradients = self._compute_client_gradients(model, target_client)
            
            # 计算其他客户端的加权梯度
            other_gradients = self._compute_weighted_other_clients_gradients(model, other_clients, other_weights)
            
            # 应用梯度到模型参数
            param_idx = 0
            for param in model.parameters():
                if param.requires_grad:
                    # 对每个其他客户端使用自适应遗忘强度
                    combined_grad = torch.zeros_like(target_gradients[param_idx])
                    
                    for i, client in enumerate(other_clients):
                        # 计算该客户端的自适应遗忘强度
                        adaptive_strength = self._adaptive_forget_strength(
                            target_client, client, epoch, epochs, lambda_reversal,
                            cluster_labels, distances  # 传递分簇信息
                        )
                        adaptive_strength *= other_weights[i]
                        
                        # 计算该客户端的梯度
                        client_gradients = self._compute_client_gradients(model, client)
                        
                        # 应用自适应强度
                        client_grad = client_gradients[param_idx] * adaptive_strength
                        combined_grad += client_grad
                    
                    # 梯度反转：目标客户端梯度取反
                    reversed_grad = -lambda_reversal * (epoch + 1) / epochs * target_gradients[param_idx]
                    
                    # 组合梯度：反转的目标梯度 + 其他客户端的自适应梯度
                    # 对其他客户端的组合梯度做总量约束
                    if len(other_clients) > 0:
                        combined_grad = combined_grad * (0.8 / max(weight_denominator, 1e-12))
                    # 若存在遗忘掩码，只在非掩码参数上保留其他客户端梯度，避免抵消遗忘
                    try:
                        if hasattr(self, '_forget_mask') and self._forget_mask is not None:
                            mask_param = self._forget_mask[param_idx]
                            if mask_param is not None:
                                combined_grad = combined_grad * (1.0 - mask_param.to(combined_grad.device))
                    except Exception:
                        pass
                    param.grad = reversed_grad + combined_grad
                    param_idx += 1
            
            optimizer.step()
            
            if epoch % 2 == 0:
                print(f"Epoch {epoch}: 改进的梯度反转训练中... (基础遗忘强度: {lambda_reversal * (epoch + 1) / epochs:.3f})")
        
        print("改进的梯度反转遗忘完成！")
        
        # 验证遗忘效果
        self._validate_forgetting_effect(model, target_client, other_clients, pre_accuracies)
        
        # 打印性能统计
        self._print_performance_summary()
        
        return model
    
    def _validate_forgetting_effect(self, model, target_client, other_clients, pre_accuracies=None):
        """改进的遗忘效果验证，提供更详细的分析"""
        print("\n============= 改进的遗忘效果验证 =============")
        
        # 评估目标客户端性能
        target_client.set_parameters(model)
        target_acc = self._evaluate_client_performance(target_client)
        print(f"目标客户端遗忘后准确率: {target_acc:.4f}")
        
        # 评估其他客户端性能并分析
        other_accs = []
        performance_analysis = []
        
        for i, client in enumerate(other_clients):
            client.set_parameters(model)
            acc = self._evaluate_client_performance(client)
            other_accs.append(acc)
            
            # 计算性能变化
            # 使用遗忘前的准确率（如果提供）或估计值
            if pre_accuracies is not None and client.id < len(pre_accuracies):
                pre_acc = pre_accuracies[client.id]
            else:
                pre_acc = 0.7  # 默认估计值
                print(f"警告：客户端{client.id}的遗忘前准确率未提供，使用估计值0.7")
            
            performance_change = acc - pre_acc
            
            performance_analysis.append({
                'client_id': client.id,
                'post_acc': acc,
                'pre_acc': pre_acc,
                'change': performance_change
            })
            
            print(f"客户端{client.id}遗忘后准确率: {acc:.4f} (变化: {performance_change:+.4f})")
        
        avg_other_acc = np.mean(other_accs)
        print(f"其他客户端平均准确率: {avg_other_acc:.4f}")
        
        # 分析保护效果
        self._analyze_protection_effectiveness(performance_analysis)
        
        # 如果其他客户端性能下降过多，发出警告
        if avg_other_acc < 0.5:
            print("警告：其他客户端性能下降过多，可能需要调整遗忘策略")
        elif avg_other_acc < 0.6:
            print("注意：其他客户端性能下降较多，建议进一步优化")
        else:
            print("良好：其他客户端性能保持较好")
    
    def _analyze_protection_effectiveness(self, performance_analysis):
        """分析保护效果"""
        print("\n============= 保护效果分析 =============")
        
        # 按性能变化排序
        sorted_analysis = sorted(performance_analysis, key=lambda x: x['change'])
        
        print("性能变化排序（从最差到最好）:")
        for analysis in sorted_analysis:
            status = "严重下降" if analysis['change'] < -0.3 else \
                    "明显下降" if analysis['change'] < -0.1 else \
                    "轻微下降" if analysis['change'] < 0 else \
                    "保持良好"
            
            print(f"  客户端{analysis['client_id']}: {analysis['change']:+.4f} (遗忘前: {analysis['pre_acc']:.4f} → 遗忘后: {analysis['post_acc']:.4f}, {status})")
        
        # 统计保护效果
        severe_drops = sum(1 for a in performance_analysis if a['change'] < -0.3)
        moderate_drops = sum(1 for a in performance_analysis if -0.3 <= a['change'] < -0.1)
        good_protection = sum(1 for a in performance_analysis if a['change'] >= -0.1)
        
        print(f"\n保护效果统计:")
        print(f"  严重下降的客户端: {severe_drops}")
        print(f"  明显下降的客户端: {moderate_drops}")
        print(f"  保护良好的客户端: {good_protection}")
        
        if severe_drops > 0:
            print("建议：需要进一步改进保护策略，特别是对相似客户端的保护")
        elif moderate_drops > len(performance_analysis) // 2:
            print("建议：保护策略需要优化，减少对非目标客户端的影响")
        else:
            print("良好：保护策略效果较好")
    
    def _evaluate_client_performance(self, client):
        """评估单个客户端的性能"""
        client.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            testloader = client.load_test_data()
            for data, target in testloader:
                if type(data) == type([]):
                    data = data[0].to(self.device)
                else:
                    data = data.to(self.device)
                target = target.to(self.device)
                
                output = client.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                if total >= 1000:  # 限制评估样本数
                    break
        
        return correct / total if total > 0 else 0.0

    def _build_feature_probe_loader(self, client):
        """为特征提取构建更稳健的loader，避免小客户端因drop_last导致空batch。"""
        batch_size = getattr(client, 'batch_size', 32)
        num_clients = int(getattr(self.args, 'num_clients', getattr(self, 'num_clients', 0)) or 0)
        if num_clients >= 100:
            batch_size = min(batch_size, 8)
        elif num_clients >= 50:
            batch_size = min(batch_size, 16)

        try:
            trainloader = client.load_train_data(batch_size=batch_size)
            if len(trainloader) > 0:
                return trainloader
        except Exception as e:
            print(f"警告：默认训练loader构建失败，客户端{client.id}: {e}")

        try:
            from torch.utils.data import DataLoader
            from utils.data_utils import read_client_data

            train_data = read_client_data(
                client.dataset,
                client.id,
                is_train=True,
                few_shot=getattr(client, 'few_shot', False),
            )
            return DataLoader(train_data, batch_size, drop_last=False, shuffle=False)
        except Exception as e:
            print(f"警告：备用特征loader构建失败，客户端{client.id}: {e}")
            return []

    def _infer_client_feature_dim(self, client):
        """在客户端数据极少时，为零向量回退推断特征维度。"""
        candidate_modules = [
            getattr(client.model, 'head', None),
            getattr(client.model, 'fc', None),
            getattr(client.model, 'classifier', None),
        ]

        for module in candidate_modules:
            if module is None:
                continue
            for attr in ('in_features', 'in_channels', 'out_features'):
                value = getattr(module, attr, None)
                if isinstance(value, int) and value > 0:
                    return value

        num_classes = getattr(client, 'num_classes', None)
        if isinstance(num_classes, int) and num_classes > 0:
            return num_classes

        return 1

    def _summarize_feature_distribution(self, features):
        """将样本级特征统一压缩为固定维度向量，便于异常回退计算。"""
        features = np.asarray(features, dtype=np.float32)
        if features.size == 0:
            return np.zeros(1, dtype=np.float32)
        if features.ndim == 1:
            return features.astype(np.float32, copy=False)
        flattened = features.reshape(features.shape[0], -1)
        return flattened.mean(axis=0).astype(np.float32, copy=False)
    
    def _extract_client_features(self, global_model, clients):
        """提取客户端特征用于分簇"""
        features = []
        for client in clients:
            # 使用全局模型提取客户端数据的特征
            client.set_parameters(global_model)
            feature = self._extract_feature_from_client(client)
            features.append(feature)
        return features
    
    def _extract_feature_from_client(self, client):
        """从客户端数据提取分布特征用于Wasserstein距离计算"""
        client.model.eval()
        all_features = []
        feature_dim = None
        
        with torch.no_grad():
            trainloader = self._build_feature_probe_loader(client)
            for batch_idx, (data, target) in enumerate(trainloader):
                if type(data) == type([]):
                    data = data[0].to(self.device)
                else:
                    data = data.to(self.device)
                
                # 获取倒数第二层的输出（特征）
                if hasattr(client.model, 'head'):
                    # 对于BaseHeadSplit模型
                    feature = client.model.base(data)
                else:
                    # 对于普通模型，获取倒数第二层特征
                    x = data
                    # 遍历模型层，找到倒数第二层
                    layers = list(client.model.children())
                    if len(layers) > 1:
                        # 移除最后一层（分类层）
                        feature_extractor = torch.nn.Sequential(*layers[:-1])
                        feature = feature_extractor(x)
                        feature = feature.view(feature.size(0), -1)  # 展平
                    else:
                        # 如果只有一层，使用原始输入
                        feature = x.view(x.size(0), -1)
                
                # 收集所有样本的特征，用于计算分布
                feature_np = feature.detach().cpu().float().reshape(feature.size(0), -1).numpy()
                if feature_np.size == 0:
                    continue
                feature_dim = feature_np.shape[1]
                all_features.append(feature_np)
                
                # 限制批次数以控制计算量
                if batch_idx >= 10:  # 增加批次数以获得更好的分布
                    break
        
        # 合并所有特征
        if not all_features:
            fallback_dim = feature_dim or self._infer_client_feature_dim(client)
            print(f"警告：客户端{client.id}未提取到有效特征，回退为零向量，维度={fallback_dim}")
            return np.zeros((1, fallback_dim), dtype=np.float32)

        all_features = np.concatenate(all_features, axis=0).astype(np.float32, copy=False)
        return all_features
    
    def _compute_wasserstein_distances(self, features_list):
        """计算客户端之间的Wasserstein距离，用于分布差异分簇"""
        num_clients = len(features_list)
        distances = np.zeros((num_clients, num_clients))
        
        print("计算客户端间的Wasserstein距离...")
        
        try:
            for i in range(num_clients):
                for j in range(i+1, num_clients):
                    # 计算两个客户端特征分布之间的Wasserstein距离
                    dist = self._compute_distribution_distance(features_list[i], features_list[j])
                    distances[i][j] = dist
                    distances[j][i] = dist
                    print(f"客户端{i}与客户端{j}的Wasserstein距离: {dist:.4f}")
        except Exception as e:
            print(f"计算Wasserstein距离时出现异常: {e}")
            # 如果计算失败，使用简化的距离度量
            for i in range(num_clients):
                for j in range(i+1, num_clients):
                    dist = self._compute_simple_distance(features_list[i], features_list[j])
                    distances[i][j] = dist
                    distances[j][i] = dist
        
        return distances
    
    def _compute_distribution_distance(self, features1, features2):
        """计算两个特征分布之间的Wasserstein距离"""
        try:
            features1 = np.asarray(features1, dtype=np.float32)
            features2 = np.asarray(features2, dtype=np.float32)

            if features1.size == 0 or features2.size == 0:
                return self._compute_simple_distance(features1, features2)

            # 将特征转换为1D分布进行比较
            # 使用特征向量的范数作为分布样本
            if features1.ndim == 1:
                dist1 = features1.reshape(-1)
            else:
                dist1 = np.linalg.norm(features1.reshape(features1.shape[0], -1), axis=1)

            if features2.ndim == 1:
                dist2 = features2.reshape(-1)
            else:
                dist2 = np.linalg.norm(features2.reshape(features2.shape[0], -1), axis=1)
            
            # 计算Wasserstein距离
            from scipy.stats import wasserstein_distance
            w_dist = wasserstein_distance(dist1, dist2)
            return w_dist
        except Exception as e:
            print(f"Wasserstein距离计算失败: {e}")
            return self._compute_simple_distance(features1, features2)
    
    def _compute_simple_distance(self, features1, features2):
        """计算简化的分布距离（备选方案）"""
        # 使用特征均值的欧几里得距离
        mean1 = self._summarize_feature_distribution(features1)
        mean2 = self._summarize_feature_distribution(features2)

        if mean1.shape != mean2.shape:
            min_dim = min(mean1.shape[0], mean2.shape[0])
            mean1 = mean1[:min_dim]
            mean2 = mean2[:min_dim]

        return float(np.linalg.norm(mean1 - mean2))

    def _compute_target_distance_thresholds(self, distances, target_client_id):
        """根据目标客户端与其他客户端的距离分布，自适应设置保护阈值。"""
        try:
            target_distances = [
                float(distances[target_client_id][i])
                for i in range(len(distances))
                if i != target_client_id
            ]
            if not target_distances:
                return 0.3, 0.6

            high_similarity_threshold = float(np.percentile(target_distances, 25))
            medium_similarity_threshold = float(np.percentile(target_distances, 50))

            high_similarity_threshold = max(0.3, high_similarity_threshold)
            medium_similarity_threshold = max(high_similarity_threshold + 1e-6, medium_similarity_threshold)

            print(
                f"目标客户端{target_client_id}保护阈值: "
                f"高相似<{high_similarity_threshold:.4f}, 中相似<{medium_similarity_threshold:.4f}"
            )
            return high_similarity_threshold, medium_similarity_threshold
        except Exception as e:
            print(f"警告：自适应保护阈值计算失败，回退默认值: {e}")
            return 0.3, 0.6
    
    def _cluster_clients(self, features_list, distances):
        """改进的客户端分簇 - 基于梯度相似性，支持多层次分簇策略"""
        print("开始基于梯度相似性的改进分簇...")
        
        # 根据配置选择分簇策略
        if self.use_hierarchical_clustering:
            print("使用多层次分簇策略...")
            # 使用多层次分簇策略
            cluster_labels = self._hierarchical_clustering_strategy(distances)
        else:
            print("使用传统改进分簇方法...")
            # 使用改进的距离分簇方法
            cluster_labels = self._improved_gradient_based_clustering(distances)
        
        # 分析分簇结果
        self._analyze_clustering_result(cluster_labels, distances)
        
        # 检测异常客户端
        anomaly_clients = self._detect_anomaly_clients_heterogeneous(cluster_labels, distances)
        
        return cluster_labels, anomaly_clients
    
    def _improved_gradient_based_clustering(self, distances):
        """基于梯度相似性的改进分簇方法 - 使用自适应阈值和多层次策略"""
        num_clients = len(distances)
        
        # 初始化簇标签
        cluster_labels = np.zeros(num_clients, dtype=int)
        current_cluster = 0
        
        # 目标客户端单独成簇
        target_client_id = 0
        cluster_labels[target_client_id] = current_cluster
        current_cluster += 1
        print(f"目标客户端{target_client_id}单独成簇: {current_cluster-1}")
        
        # 计算与目标客户端的距离排序
        target_distances = []
        for i in range(num_clients):
            if i != target_client_id:
                target_distances.append((i, distances[target_client_id][i]))
        
        target_distances.sort(key=lambda x: x[1])
        
        print("与目标客户端的距离排序:")
        for client_id, distance in target_distances:
            print(f"  客户端{client_id}: {distance:.4f}")
        
        # 使用自适应阈值进行分簇
        if self.use_adaptive_thresholds:
            high_similarity_threshold, medium_similarity_threshold = self._compute_adaptive_thresholds(distances)
            print(f"自适应阈值 - 高相似度: {high_similarity_threshold:.4f}, 中等相似度: {medium_similarity_threshold:.4f}")
        else:
            # 使用固定阈值
            high_similarity_threshold = 0.5  # 高相似度阈值
            medium_similarity_threshold = 0.8  # 中等相似度阈值
            print(f"固定阈值 - 高相似度: {high_similarity_threshold:.4f}, 中等相似度: {medium_similarity_threshold:.4f}")
        
        # 第一步：将相似度高的客户端合并成簇（获得群体保护）
        for client_id, distance in target_distances:
            if cluster_labels[client_id] != 0:  # 已经分配了簇
                continue
                
            if distance < high_similarity_threshold:
                # 与目标客户端高度相似，需要与其他相似客户端合并以获得保护
                best_similar_client = None
                best_similarity = float('inf')
                
                for other_client_id, other_distance in target_distances:
                    if other_client_id != client_id and cluster_labels[other_client_id] == 0:
                        # 检查两个客户端之间的距离
                        pair_distance = distances[client_id][other_client_id]
                        if pair_distance < high_similarity_threshold and pair_distance < best_similarity:
                            best_similarity = pair_distance
                            best_similar_client = other_client_id
                
                if best_similar_client is not None:
                    # 合并到同一簇
                    cluster_labels[client_id] = current_cluster
                    cluster_labels[best_similar_client] = current_cluster
                    current_cluster += 1
                    print(f"客户端{client_id}与客户端{best_similar_client}高度相似，合并到保护簇{cluster_labels[client_id]}")
                else:
                    # 没有找到相似客户端，单独成簇
                    cluster_labels[client_id] = current_cluster
                    current_cluster += 1
                    print(f"客户端{client_id}与目标客户端高度相似({distance:.4f})，单独成簇{cluster_labels[client_id]}")
            
            elif distance < medium_similarity_threshold:
                # 与目标客户端中等相似，可以合并到同一簇
                medium_cluster_found = False
                for existing_client_id, existing_distance in target_distances:
                    if existing_client_id < client_id and cluster_labels[existing_client_id] != 0:
                        if existing_distance < medium_similarity_threshold and existing_distance >= high_similarity_threshold:
                            cluster_labels[client_id] = cluster_labels[existing_client_id]
                            medium_cluster_found = True
                            print(f"客户端{client_id}与客户端{existing_client_id}中等相似，合并到簇{cluster_labels[client_id]}")
                            break
                
                if not medium_cluster_found:
                    cluster_labels[client_id] = current_cluster
                    current_cluster += 1
                    print(f"客户端{client_id}与目标客户端中等相似({distance:.4f})，分配到新簇{cluster_labels[client_id]}")
            else:
                # 与目标客户端差异较大，可以合并到同一簇
                large_cluster_found = False
                for existing_client_id, existing_distance in target_distances:
                    if existing_client_id < client_id and cluster_labels[existing_client_id] != 0:
                        if existing_distance >= medium_similarity_threshold:
                            cluster_labels[client_id] = cluster_labels[existing_client_id]
                            large_cluster_found = True
                            print(f"客户端{client_id}与客户端{existing_client_id}差异较大，合并到簇{cluster_labels[client_id]}")
                            break
                
                if not large_cluster_found:
                    cluster_labels[client_id] = current_cluster
                    current_cluster += 1
                    print(f"客户端{client_id}与目标客户端差异较大({distance:.4f})，分配到新簇{cluster_labels[client_id]}")
        
        # 后处理，确保最小簇大小
        self._enforce_minimum_cluster_size(cluster_labels, distances, min_cluster_size=1)
        
        return cluster_labels
    
    def _compute_adaptive_thresholds(self, distances):
        """基于数据分布统计计算自适应阈值"""
        # 提取上三角矩阵的距离值（排除对角线）
        upper_triangle = distances[np.triu_indices(len(distances), k=1)]
        
        # 计算分布统计特征
        mean_dist = np.mean(upper_triangle)
        std_dist = np.std(upper_triangle)
        q25 = np.percentile(upper_triangle, 25)  # 25%分位数
        q75 = np.percentile(upper_triangle, 75)  # 75%分位数
        iqr = q75 - q25  # 四分位距
        
        # 使用IQR方法设置异常值边界
        lower_bound = max(0.1, q25 - 1.5 * iqr)  # 下界，最小为0.1
        upper_bound = q75 + 1.5 * iqr  # 上界
        
        # 基于分布特征设置分簇阈值
        # 高相似度阈值：基于下界，表示非常相似
        high_similarity_threshold = max(0.2, min(0.6, lower_bound))
        
        # 中等相似度阈值：基于75%分位数，表示中等相似
        medium_similarity_threshold = max(0.4, min(0.9, q75))
        
        print(f"距离分布统计 - 均值: {mean_dist:.4f}, 标准差: {std_dist:.4f}")
        print(f"距离分布统计 - Q25: {q25:.4f}, Q75: {q75:.4f}, IQR: {iqr:.4f}")
        print(f"自适应阈值计算 - 下界: {lower_bound:.4f}, 上界: {upper_bound:.4f}")
        
        return high_similarity_threshold, medium_similarity_threshold
    
    def _hierarchical_clustering_strategy(self, distances):
        """多层次分簇策略"""
        print("开始多层次分簇策略...")
        
        # 第一层：基于分布相似性的粗分簇
        print("第一层：基于分布相似性的粗分簇")
        coarse_clusters = self._coarse_clustering_by_distribution(distances)
        
        # 第二层：基于梯度相似性的细分簇
        print("第二层：基于梯度相似性的细分簇")
        fine_clusters = self._fine_clustering_by_gradients(coarse_clusters, distances)
        
        # 第三层：基于遗忘需求的优化分簇
        print("第三层：基于遗忘需求的优化分簇")
        optimized_clusters = self._optimize_clusters_for_forgetting(fine_clusters, distances)
        
        return optimized_clusters
    
    def _coarse_clustering_by_distribution(self, distances):
        """基于分布相似性的粗分簇"""
        num_clients = len(distances)
        cluster_labels = np.zeros(num_clients, dtype=int)
        current_cluster = 0
        
        # 目标客户端单独成簇
        target_client_id = 0
        cluster_labels[target_client_id] = current_cluster
        current_cluster += 1
        
        # 使用K-means思想进行粗分簇
        # 基于与目标客户端的距离进行初步分组
        target_distances = [(i, distances[target_client_id][i]) for i in range(num_clients) if i != target_client_id]
        target_distances.sort(key=lambda x: x[1])
        
        # 使用距离的分布特征进行分组
        distances_values = [d[1] for d in target_distances]
        mean_dist = np.mean(distances_values)
        std_dist = np.std(distances_values)
        
        # 基于距离分布进行分组
        for client_id, distance in target_distances:
            if distance < mean_dist - 0.5 * std_dist:
                # 高相似组
                cluster_labels[client_id] = current_cluster
                current_cluster += 1
            elif distance < mean_dist + 0.5 * std_dist:
                # 中等相似组
                cluster_labels[client_id] = current_cluster
                current_cluster += 1
            else:
                # 低相似组
                cluster_labels[client_id] = current_cluster
                current_cluster += 1
        
        print(f"粗分簇完成，共{current_cluster}个簇")
        return cluster_labels
    
    def _fine_clustering_by_gradients(self, coarse_clusters, distances):
        """基于梯度相似性的细分簇"""
        # 这里可以基于梯度信息进行更精细的分簇
        # 暂时返回粗分簇结果，后续可以扩展
        return coarse_clusters
    
    def _optimize_clusters_for_forgetting(self, clusters, distances):
        """基于遗忘需求优化分簇"""
        target_client_id = 0
        unique_clusters = set(clusters)
        
        # 评估每个簇的遗忘难度
        cluster_forgetting_difficulty = {}
        for cluster_id in unique_clusters:
            if cluster_id == 0:  # 跳过目标客户端簇
                continue
            
            # 计算簇内客户端与目标客户端的平均距离
            cluster_members = np.where(clusters == cluster_id)[0]
            if len(cluster_members) > 0:
                avg_distance_to_target = np.mean([distances[target_client_id][member] for member in cluster_members])
                
                # 计算簇内一致性
                cluster_internal_distances = []
                for i in range(len(cluster_members)):
                    for j in range(i+1, len(cluster_members)):
                        cluster_internal_distances.append(distances[cluster_members[i]][cluster_members[j]])
                
                cluster_consistency = np.mean(cluster_internal_distances) if cluster_internal_distances else 0
                
                # 遗忘难度 = 与目标客户端的相似度 + 簇内一致性
                # 距离越小表示越相似，遗忘难度越高
                forgetting_difficulty = (1.0 / (1.0 + avg_distance_to_target)) + cluster_consistency
                cluster_forgetting_difficulty[cluster_id] = forgetting_difficulty
        
        print("簇遗忘难度评估:")
        for cluster_id, difficulty in cluster_forgetting_difficulty.items():
            print(f"  簇{cluster_id}: 遗忘难度 {difficulty:.4f}")
        
        # 基于遗忘难度优化分簇
        # 高遗忘难度的簇需要更精细的分簇
        optimized_clusters = clusters.copy()
        
        # 这里可以添加更复杂的优化逻辑
        # 例如：将高遗忘难度的簇进一步细分
        
        return optimized_clusters
    
    def _enforce_minimum_cluster_size(self, cluster_labels, distances, min_cluster_size=1):
        """强制执行最小簇大小，但允许与目标客户端高度相似的客户端单独成簇"""
        unique_clusters = set(cluster_labels)
        num_clients = len(cluster_labels)
        target_client_id = 0
        
        # 统计每个簇的大小
        cluster_sizes = {}
        for cluster_id in unique_clusters:
            cluster_sizes[cluster_id] = np.sum(cluster_labels == cluster_id)
        
        # 找出小于最小簇大小的簇（排除目标客户端簇和与目标客户端高度相似的簇）
        small_clusters = []
        for cluster_id in unique_clusters:
            if cluster_id == 0:  # 跳过目标客户端簇
                continue
            if cluster_sizes[cluster_id] < min_cluster_size:
                # 检查该簇是否包含与目标客户端高度相似的客户端
                cluster_members = np.where(cluster_labels == cluster_id)[0]
                has_high_similarity = False
                for member in cluster_members:
                    if distances[target_client_id][member] < 0.3:  # 高相似度阈值
                        has_high_similarity = True
                        break
                
                # 如果包含高度相似的客户端，允许单独成簇
                if not has_high_similarity:
                    small_clusters.append(cluster_id)
        
        if small_clusters:
            print(f"检测到小簇: {small_clusters}，进行合并...")
            
            for small_cluster in small_clusters:
                # 找到该簇中的客户端
                clients_in_small_cluster = np.where(cluster_labels == small_cluster)[0]
                
                # 为每个客户端找到最相似的簇
                for client_id in clients_in_small_cluster:
                    best_cluster = None
                    best_similarity = float('inf')
                    
                    # 寻找最相似的簇（排除目标客户端簇和小簇）
                    for other_cluster in unique_clusters:
                        if other_cluster != small_cluster and other_cluster != 0:
                            # 计算与该簇的平均距离
                            other_clients_in_cluster = np.where(cluster_labels == other_cluster)[0]
                            if len(other_clients_in_cluster) > 0:
                                avg_distance = np.mean([distances[client_id][other_client] 
                                                     for other_client in other_clients_in_cluster])
                                if avg_distance < best_similarity:
                                    best_similarity = avg_distance
                                    best_cluster = other_cluster
                    
                    # 如果找到合适的簇，进行合并
                    if best_cluster is not None:
                        cluster_labels[client_id] = best_cluster
                        print(f"客户端{client_id}从小簇{small_cluster}合并到相似簇{best_cluster} (相似度: {best_similarity:.4f})")
                    else:
                        # 如果没找到合适的簇，合并到最大的簇
                        largest_cluster = max(cluster_sizes.items(), key=lambda x: x[1])[0]
                        if largest_cluster != 0:  # 不合并到目标客户端簇
                            cluster_labels[client_id] = largest_cluster
                            print(f"客户端{client_id}从小簇{small_cluster}合并到最大簇{largest_cluster}")
    
    def _determine_optimal_clusters_dissimilarity(self, distances):
        """确定最优簇数，最大化簇间差异"""
        num_clients = len(distances)
        
        # 计算距离矩阵的统计信息
        upper_triangle = distances[np.triu_indices(num_clients, k=1)]
        mean_dist = np.mean(upper_triangle)
        std_dist = np.std(upper_triangle)
        
        print(f"距离统计 - 均值: {mean_dist:.4f}, 标准差: {std_dist:.4f}")
        
        # 基于距离分布确定簇数 - 倾向于较少簇数以最大化差异
        if std_dist < mean_dist * 0.3:
            # 距离分布集中，使用较少簇数
            optimal_k = max(2, min(3, num_clients // 3))
        elif std_dist > mean_dist * 1.5:
            # 距离分布分散，使用中等簇数
            optimal_k = max(2, min(4, num_clients // 2))
        else:
            # 中等分布，使用较少簇数
            optimal_k = max(2, min(3, num_clients // 2))
        
        return optimal_k
    
    def _analyze_clustering_result(self, cluster_labels, distances):
        """分析分簇结果"""
        unique_clusters = set(cluster_labels)
        num_clients = len(cluster_labels)
        
        print(f"\n============= 分簇结果分析 =============")
        print(f"总客户端数: {num_clients}")
        print(f"簇数量: {len(unique_clusters)}")
        
        # 分析每个簇
        for cluster_id in sorted(unique_clusters):
            cluster_members = np.where(cluster_labels == cluster_id)[0]
            cluster_size = len(cluster_members)
            
            if cluster_id == 0:
                print(f"簇{cluster_id} (目标客户端): {cluster_members}, 大小: {cluster_size}")
            else:
                print(f"簇{cluster_id}: {cluster_members}, 大小: {cluster_size}")
                
                # 计算簇内平均距离
                if cluster_size > 1:
                    internal_distances = []
                    for i in range(cluster_size):
                        for j in range(i+1, cluster_size):
                            internal_distances.append(distances[cluster_members[i]][cluster_members[j]])
                    
                    avg_internal_distance = np.mean(internal_distances)
                    print(f"  簇内平均距离: {avg_internal_distance:.4f}")
        
        # 评估分簇质量
        self._evaluate_clustering_quality(cluster_labels, distances)
    
    def _evaluate_clustering_quality(self, cluster_labels, distances):
        """评估分簇质量"""
        print(f"\n============= 分簇质量评估 =============")
        
        unique_clusters = set(cluster_labels)
        target_client_id = 0
        
        # 1. 计算簇内一致性 (Intra-cluster consistency)
        intra_cluster_consistency = {}
        for cluster_id in unique_clusters:
            if cluster_id == 0:  # 跳过目标客户端簇
                continue
                
            cluster_members = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_members) > 1:
                internal_distances = []
                for i in range(len(cluster_members)):
                    for j in range(i+1, len(cluster_members)):
                        internal_distances.append(distances[cluster_members[i]][cluster_members[j]])
                
                avg_internal_distance = np.mean(internal_distances)
                intra_cluster_consistency[cluster_id] = avg_internal_distance
            else:
                intra_cluster_consistency[cluster_id] = 0
        
        # 2. 计算簇间分离性 (Inter-cluster separation)
        inter_cluster_separation = {}
        for cluster_id in unique_clusters:
            if cluster_id == 0:  # 跳过目标客户端簇
                continue
                
            cluster_members = np.where(cluster_labels == cluster_id)[0]
            other_clusters = [c for c in unique_clusters if c != cluster_id and c != 0]
            
            if other_clusters:
                min_inter_distance = float('inf')
                for other_cluster_id in other_clusters:
                    other_members = np.where(cluster_labels == other_cluster_id)[0]
                    
                    # 计算两个簇之间的最小距离
                    for member1 in cluster_members:
                        for member2 in other_members:
                            distance = distances[member1][member2]
                            if distance < min_inter_distance:
                                min_inter_distance = distance
                
                inter_cluster_separation[cluster_id] = min_inter_distance
            else:
                inter_cluster_separation[cluster_id] = 0
        
        # 3. 计算与目标客户端的距离分布
        target_distances = {}
        for cluster_id in unique_clusters:
            if cluster_id == 0:  # 跳过目标客户端簇
                continue
                
            cluster_members = np.where(cluster_labels == cluster_id)[0]
            cluster_target_distances = [distances[target_client_id][member] for member in cluster_members]
            target_distances[cluster_id] = np.mean(cluster_target_distances)
        
        # 4. 输出评估结果
        print("簇内一致性 (越小越好):")
        for cluster_id, consistency in intra_cluster_consistency.items():
            print(f"  簇{cluster_id}: {consistency:.4f}")
        
        print("簇间分离性 (越大越好):")
        for cluster_id, separation in inter_cluster_separation.items():
            print(f"  簇{cluster_id}: {separation:.4f}")
        
        print("与目标客户端的平均距离:")
        for cluster_id, distance in target_distances.items():
            print(f"  簇{cluster_id}: {distance:.4f}")
        
        # 5. 计算综合质量指标
        if intra_cluster_consistency and inter_cluster_separation:
            avg_intra_consistency = np.mean(list(intra_cluster_consistency.values()))
            avg_inter_separation = np.mean(list(inter_cluster_separation.values()))
            
            # 分簇质量 = 簇间分离性 / (簇内一致性 + 小常数)
            clustering_quality = avg_inter_separation / (avg_intra_consistency + 1e-6)
            
            print(f"\n综合分簇质量指标: {clustering_quality:.4f}")
            print(f"  - 平均簇内一致性: {avg_intra_consistency:.4f}")
            print(f"  - 平均簇间分离性: {avg_inter_separation:.4f}")
            
            if clustering_quality > 2.0:
                print("  分簇质量: 优秀")
            elif clustering_quality > 1.0:
                print("  分簇质量: 良好")
            else:
                print("  分簇质量: 需要改进")
    
    def _detect_anomaly_clients_heterogeneous(self, cluster_labels, distances):
        """基于异构性的异常客户端检测"""
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
        
        return list(set(anomaly_clients))  # 去重
    
    def _generate_forget_mask(self, global_model, target_client, top_ratio=0.1):
        """生成遗忘掩码 - 考虑目标客户端与其他距离大的客户端的差异"""
        # 计算目标客户端的梯度残差
        target_gradients = self._compute_gradient_residual(global_model, target_client)
        
        # 获取目标客户端所在簇的其他客户端
        cluster_members = self._get_cluster_members(target_client.id if hasattr(target_client, 'id') else 0)
        
        # 计算簇内其他客户端的平均梯度
        cluster_gradients = None
        if cluster_members:
            cluster_gradients_list = []
            for client_id in cluster_members:
                if client_id != (target_client.id if hasattr(target_client, 'id') else 0):
                    client = self._get_client_by_id(client_id)
                    if client:
                        client_gradients = self._compute_gradient_residual(global_model, client)
                        cluster_gradients_list.append(client_gradients)
            
            if cluster_gradients_list:
                # 计算簇内平均梯度
                cluster_gradients = []
                for i in range(len(cluster_gradients_list[0])):
                    avg_grad = torch.zeros_like(cluster_gradients_list[0][i])
                    for grad in cluster_gradients_list:
                        avg_grad += grad[i]
                    avg_grad /= len(cluster_gradients_list)
                    cluster_gradients.append(avg_grad)
        
        # 使用自适应阈值选择
        all_params = []
        for grad in target_gradients:
            all_params.extend(grad.flatten().cpu().numpy())
        
        # 动态调整top_ratio基于簇内差异
        if cluster_gradients:
            # 计算目标客户端与簇内其他客户端的梯度差异
            dissimilarity_score = self._compute_gradient_dissimilarity(target_gradients, cluster_gradients)
            
            # 如果差异大，使用更精确的掩码
            if dissimilarity_score > 0.7:
                top_ratio = max(0.05, top_ratio * 0.8)  # 更精确
            elif dissimilarity_score < 0.3:
                top_ratio = min(0.15, top_ratio * 1.2)  # 更宽泛
        
        # 基于梯度分布自适应调整
        abs_params = np.abs(all_params)
        if len(abs_params) > 0:
            mean_grad = np.mean(abs_params)
            std_grad = np.std(abs_params)
            
            # 如果梯度分布比较集中，减少选择比例
            if std_grad < mean_grad * 0.5:
                top_ratio = max(0.05, top_ratio * 0.8)
            elif std_grad > mean_grad * 2.0:
                top_ratio = min(0.15, top_ratio * 1.2)
        
        # 计算阈值
        threshold = np.percentile(abs_params, (1 - top_ratio) * 100)
        
        # 生成掩码
        mask = []
        for grad in target_gradients:
            mask_param = (torch.abs(grad) >= threshold).float()
            mask.append(mask_param)
        
        return mask
    
    def _get_cluster_members(self, target_client_id):
        """获取目标客户端所在簇的其他客户端ID"""
        # 这里需要根据实际的分簇结果来获取
        # 暂时返回空列表，实际实现时需要保存分簇结果
        return []
    
    def _get_client_by_id(self, client_id):
        """根据客户端ID获取客户端对象"""
        # 这里需要根据实际的客户端列表来获取
        # 暂时返回None，实际实现时需要传入客户端列表
        return None
    
    def _compute_gradient_dissimilarity(self, gradients1, gradients2):
        """计算两组梯度之间的差异度"""
        if not gradients2:
            return 0.0
        
        dissimilarities = []
        for grad1, grad2 in zip(gradients1, gradients2):
            # 计算余弦差异度
            flat_grad1 = grad1.flatten()
            flat_grad2 = grad2.flatten()
            
            cos_sim = torch.cosine_similarity(flat_grad1.unsqueeze(0), flat_grad2.unsqueeze(0))
            # 将相似度转换为差异度
            dissimilarity = 1 - cos_sim.item()
            dissimilarities.append(dissimilarity)
        
        return np.mean(dissimilarities)

    def _collect_client_gradients(self, client, max_batches=4, sample_ratio=None):
        """稳定收集客户端梯度，始终返回与模型可训练参数等长的梯度列表。"""
        trainloader = self._build_feature_probe_loader(client)
        if sample_ratio is not None:
            sampled_batches = self._improved_data_sampling(trainloader, sample_ratio, max_batches)
        else:
            sampled_batches = list(trainloader)
            if max_batches > 0:
                sampled_batches = sampled_batches[:max_batches]

        accumulated_gradients = self._get_zero_gradients(client.model)
        if len(accumulated_gradients) == 0 or len(sampled_batches) == 0:
            return accumulated_gradients

        batch_count = 0
        for data, target in sampled_batches:
            if type(data) == type([]):
                data = data[0].to(self.device)
            else:
                data = data.to(self.device)
            target = target.to(self.device)

            client.optimizer.zero_grad()
            if self.device.type == 'cuda':
                # Gradient probes only use a few batches, so we prefer the more
                # stable non-cuDNN path here to avoid intermittent algorithm failures.
                with torch.backends.cudnn.flags(enabled=False, benchmark=False, deterministic=False):
                    output = client.model(data)
                    loss = client.loss(output, target)
                    loss.backward()
            else:
                output = client.model(data)
                loss = client.loss(output, target)
                loss.backward()

            grad_idx = 0
            for param in client.model.parameters():
                if not param.requires_grad:
                    continue
                if param.grad is not None:
                    accumulated_gradients[grad_idx] += param.grad.detach().clone()
                grad_idx += 1
            batch_count += 1

        if batch_count > 0:
            for i in range(len(accumulated_gradients)):
                accumulated_gradients[i] /= batch_count

        return accumulated_gradients
    
    def _compute_gradient_residual(self, global_model, target_client):
        """计算梯度残差"""
        target_client.set_parameters(global_model)
        target_client.model.train()
        return self._collect_client_gradients(target_client, max_batches=4)
    
    def _compute_client_gradients(self, model, client):
        """计算单个客户端的梯度"""
        client.set_parameters(model)
        client.model.train()
        return self._collect_client_gradients(client, max_batches=6)
    
    def _compute_other_clients_gradients(self, model, other_clients):
        """计算其他客户端的平均梯度"""
        all_gradients = []
        
        for client in other_clients:
            client_gradients = self._compute_client_gradients(model, client)
            all_gradients.append(client_gradients)
        
        # 计算平均梯度
        avg_gradients = []
        for param_idx in range(len(all_gradients[0])):
            avg_grad = torch.zeros_like(all_gradients[0][param_idx])
            for client_grads in all_gradients:
                avg_grad += client_grads[param_idx]
            avg_gradients.append(avg_grad / len(all_gradients))
        
        return avg_gradients
    
    def _compute_weighted_other_clients_gradients(self, model, other_clients, weights):
        """计算其他客户端的加权梯度"""
        all_gradients = []
        
        for i, client in enumerate(other_clients):
            client_gradients = self._compute_client_gradients(model, client)
            # 应用权重
            weighted_gradients = [grad * weights[i] for grad in client_gradients]
            all_gradients.append(weighted_gradients)
        
        # 计算加权平均梯度
        avg_gradients = []
        for param_idx in range(len(all_gradients[0])):
            avg_grad = torch.zeros_like(all_gradients[0][param_idx])
            for client_grads in all_gradients:
                avg_grad += client_grads[param_idx]
            avg_gradients.append(avg_grad / len(all_gradients))
        
        return avg_gradients
    
    def _compute_client_importance_weights(self, target_client, other_clients, distances, cluster_labels):
        """基于目标相似性与簇一致性的保留权重，优先保护高风险保留客户端。"""
        weights = []
        if target_client is None:
            return [1.0] * len(other_clients)

        target_client_id = target_client.id
        for client in other_clients:
            client_id = client.id

            distance_to_target = float(distances[target_client_id][client_id])
            cluster_id = cluster_labels[client_id]
            cluster_members = [i for i, label in enumerate(cluster_labels) if label == cluster_id and i != target_client_id]
            if cluster_members:
                cluster_distances = [distances[client_id][j] for j in cluster_members if j != client_id]
                avg_cluster_distance = np.mean(cluster_distances) if cluster_distances else distance_to_target
            else:
                avg_cluster_distance = distance_to_target

            similarity_weight = 1.0 + self.similarity_boost / (1.0 + distance_to_target)
            cluster_weight = 1.0 + 0.35 / (1.0 + avg_cluster_distance)

            pre_acc = self._pre_acc_map.get(client_id) if hasattr(self, '_pre_acc_map') else None
            stability_weight = 1.0 + max(0.0, (pre_acc - 0.5) * 0.4) if pre_acc is not None else 1.0

            importance_weight = similarity_weight * cluster_weight * stability_weight
            weights.append(importance_weight)

        # 归一化权重
        weights = np.array(weights)
        weights = weights / np.sum(weights) * len(weights)  # 保持平均权重为1

        weight_map = {client.id: float(weights[idx]) for idx, client in enumerate(other_clients)}
        print(f"改进的其他客户端重要性权重: {weight_map}")
        return weights
    
    def _compute_gradient_similarity(self, client1, client2):
        """计算两个客户端之间的梯度相似性"""
        try:
            # 使用相同的模型参数计算梯度
            model1 = copy.deepcopy(client1.model)
            model2 = copy.deepcopy(client2.model)
            
            # 计算客户端1的梯度
            client1.set_parameters(model1)
            client1.model.train()
            gradients1 = self._compute_client_gradients(model1, client1)
            
            # 计算客户端2的梯度
            client2.set_parameters(model2)
            client2.model.train()
            gradients2 = self._compute_client_gradients(model2, client2)
            
            # 计算梯度相似性（余弦相似度）
            similarities = []
            for grad1, grad2 in zip(gradients1, gradients2):
                flat_grad1 = grad1.flatten()
                flat_grad2 = grad2.flatten()
                
                # 计算余弦相似度
                cos_sim = torch.cosine_similarity(flat_grad1.unsqueeze(0), flat_grad2.unsqueeze(0))
                similarities.append(cos_sim.item())
            
            # 返回平均相似度
            return np.mean(similarities)
            
        except Exception as e:
            print(f"计算梯度相似性时出错: {e}")
            # 如果计算失败，使用距离作为替代
            distance = self._get_client_distance(client1, client2)
            return 1.0 / (1.0 + distance)  # 距离越近，相似性越高
    
    def _get_client_distance(self, client1, client2):
        """获取两个客户端之间的距离（作为梯度相似性的替代）"""
        # 这里可以基于客户端ID从距离矩阵中获取
        # 暂时返回一个默认值
        return 0.5
    
    def _adaptive_forget_strength(self, target_client, other_client, epoch, max_epochs, lambda_reversal=0.15, 
                                 cluster_labels=None, distances=None):
        """自适应遗忘强度 - 基于梯度相似性和分簇信息调整"""
        try:
            # 计算与目标客户端的梯度相似性
            gradient_similarity = self._compute_gradient_similarity(target_client, other_client)
            
            # 基础遗忘强度调整
            base_strength = lambda_reversal * (epoch + 1) / max_epochs
            
            # 如果提供了分簇信息，使用更智能的调整
            if cluster_labels is not None and distances is not None and self.use_cluster_based_forgetting:
                target_client_id = target_client.id if hasattr(target_client, 'id') else self.target_client_id
                other_client_id = other_client.id
                target_cluster_id = cluster_labels[target_client_id]
                high_similarity_threshold, medium_similarity_threshold = getattr(
                    self, '_distance_protection_thresholds', (0.3, 0.6)
                )
                
                # 获取其他客户端所在的簇
                other_cluster_id = cluster_labels[other_client_id]
                
                # 计算簇内一致性
                cluster_members = np.where(cluster_labels == other_cluster_id)[0]
                if len(cluster_members) > 1:
                    cluster_internal_distances = []
                    for i in range(len(cluster_members)):
                        for j in range(i+1, len(cluster_members)):
                            if cluster_members[i] != cluster_members[j]:
                                cluster_internal_distances.append(distances[cluster_members[i]][cluster_members[j]])
                    
                    cluster_consistency = np.mean(cluster_internal_distances) if cluster_internal_distances else 0
                else:
                    cluster_consistency = 0
                
                # 计算与目标客户端的距离
                distance_to_target = distances[target_client_id][other_client_id]
                
                # 基于分簇信息的智能调整
                if other_cluster_id == target_cluster_id:
                    # 目标客户端簇，不应该有遗忘
                    strength = 0.0
                elif distance_to_target < high_similarity_threshold:
                    # 与目标客户端高度相似，根据保护级别调整
                    if self.protection_level == 'strong':
                        strength = base_strength * 0.2
                    elif self.protection_level == 'moderate':
                        strength = base_strength * 0.6
                    else:  # weak
                        strength = base_strength * 0.65
                elif distance_to_target < medium_similarity_threshold:
                    # 与目标客户端中等相似，根据保护级别调整
                    if self.protection_level == 'strong':
                        strength = base_strength * 0.5
                    elif self.protection_level == 'moderate':
                        strength = base_strength * 0.8
                    else:  # weak
                        strength = base_strength * 0.8
                else:
                    # 与目标客户端差异较大，使用标准遗忘
                    strength = base_strength
                
                # 考虑簇内一致性：根据保护级别调整
                if cluster_consistency < 0.3:
                    # 簇内一致性高，可以稍微增加遗忘强度
                    if self.protection_level == 'strong':
                        strength *= 1.2
                    elif self.protection_level == 'moderate':
                        strength *= 1.1
                    else:  # weak
                        strength *= 1.05
                elif cluster_consistency > 0.7:
                    # 簇内一致性低，需要减少遗忘强度
                    if self.protection_level == 'strong':
                        strength *= 0.8
                    elif self.protection_level == 'moderate':
                        strength *= 0.9
                    else:  # weak
                        strength *= 0.95
                
                print(f"客户端{other_client_id} - 簇{other_cluster_id}, 距离目标: {distance_to_target:.4f}, "
                      f"簇内一致性: {cluster_consistency:.4f}, 遗忘强度: {strength:.4f}")
                
            else:
                # 如果没有分簇信息，使用原有的梯度相似性调整
                if gradient_similarity > 0.7:
                    # 高相似性客户端，根据保护级别调整
                    if self.protection_level == 'strong':
                        strength = base_strength * 0.3
                    elif self.protection_level == 'moderate':
                        strength = base_strength * 0.6
                    else:  # weak
                        strength = base_strength * 0.8
                elif gradient_similarity > 0.4:
                    # 中等相似性客户端，根据保护级别调整
                    if self.protection_level == 'strong':
                        strength = base_strength * 0.6
                    elif self.protection_level == 'moderate':
                        strength = base_strength * 0.8
                    else:  # weak
                        strength = base_strength * 0.9
                else:
                    # 低相似性客户端，使用标准遗忘
                    strength = base_strength
            
            return strength
            
        except Exception as e:
            print(f"计算自适应遗忘强度时出错: {e}")
            # 如果计算失败，使用默认强度
            return lambda_reversal * (epoch + 1) / max_epochs
    
    def forget_with_gradient_reversal_optimized(self, global_model, clients, target_client_id=0, 
                                              epochs=10, lr=0.001, lambda_reversal=0.3, pre_accuracies=None):
        """优化版本的梯度反转遗忘 - 解决计算量过大问题"""
        print(f"\n============= 开始优化版梯度反转遗忘客户端 {target_client_id} =============")
        self.target_client_id = target_client_id
        
        # 1. 提取客户端特征
        print("1. 提取客户端特征...")
        features = self._extract_client_features(global_model, clients)
        
        # 2. 计算Wasserstein距离
        print("2. 计算Wasserstein距离...")
        distances = self._compute_wasserstein_distances(features)
        self._distance_protection_thresholds = self._compute_target_distance_thresholds(distances, target_client_id)
        
        # 3. 客户端分簇
        print("3. 客户端分簇...")
        cluster_labels, anomaly_clients = self.enhanced_clustering.cluster_clients_enhanced(
            features, distances, target_client_id
        )
        
        print(f"分簇结果: {cluster_labels}")
        print(f"异常客户端: {anomaly_clients}")
        
        # 4. 生成遗忘掩码
        print("4. 生成遗忘掩码...")
        target_client = clients[target_client_id]
        mask = self._generate_forget_mask(global_model, target_client)
        # 保存掩码用于反转梯度时的参数级选择
        self._forget_mask = mask
        
        # 保存分簇与距离用于后续相似度加权保护
        self._last_cluster_labels = cluster_labels
        self._last_distances = distances
        self._last_target_client_id = target_client_id
        
        # 5. 优化版梯度反转训练 - 使用智能缓存
        print("5. 开始优化版梯度反转训练...")
        model = copy.deepcopy(global_model)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        
        other_clients = [c for c in clients if c.id != target_client_id]
        
        # 知识锚点保护初始化
        if self.use_knowledge_anchoring:
            print("6. 初始化知识锚点保护...")
            self.original_model = copy.deepcopy(global_model)  # 保存原始模型作为锚点
            self.anchor_clients = other_clients  # 其他客户端作为锚点客户端
            print(f"  锚点客户端数量: {len(self.anchor_clients)}")
            print(f"  锚点损失权重: {self.anchor_weight}")
            print(f"  锚点数据采样比例: {self.anchor_sample_ratio}")
            print(f"  锚点保护起始轮次: {self.anchor_start_epoch}")
        
        # 保存遗忘前准确率映射（用于保护强度）
        self._pre_acc_map = {}
        if pre_accuracies is not None and len(pre_accuracies) >= len(clients):
            for c in clients:
                try:
                    self._pre_acc_map[c.id] = float(pre_accuracies[c.id])
                except Exception:
                    self._pre_acc_map[c.id] = None
        
        # 计算其他客户端的重要性权重（使用改进的方法）
        other_weights = self._compute_client_importance_weights(target_client, other_clients, distances, cluster_labels)
        weight_denominator = float(np.sum(other_weights)) if len(other_weights) > 0 else 1.0
        
        # 预计算自适应遗忘强度矩阵，避免重复计算
        print("7. 预计算自适应遗忘强度矩阵...")
        adaptive_strength_matrix = self._precompute_adaptive_strengths(
            target_client, other_clients, epochs, lambda_reversal, cluster_labels, distances
        )
        
        # 使用智能缓存机制，只在必要时重新计算梯度
        cached_gradients = None
        last_model_hash = None
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # 检查模型是否发生变化，决定是否需要重新计算梯度
            current_model_hash = self._compute_model_hash(model)
            if cached_gradients is None or current_model_hash != last_model_hash:
                print(f"Epoch {epoch}: 模型发生变化，重新计算梯度...")
                cached_gradients = self._compute_cached_gradients(model, target_client, other_clients)
                last_model_hash = current_model_hash
            else:
                print(f"Epoch {epoch}: 使用缓存的梯度...")
            
            # 基于整体相似性计算全局保护缩放系数，降低对所有客户端的影响
            protection_scale = self._compute_global_protection_scale(
                target_client_id, other_clients, cluster_labels, distances
            )
            
            # 应用梯度到模型参数
            param_idx = 0
            total_reversed_norm = 0.0
            total_combined_norm = 0.0
            
            for param in model.parameters():
                if param.requires_grad:
                    # 使用预计算的强度矩阵
                    combined_grad = torch.zeros_like(cached_gradients['target'][param_idx])
                    
                    for i, client in enumerate(other_clients):
                        # 使用预计算的自适应强度（已包含分簇与相似性保护）
                        adaptive_strength = adaptive_strength_matrix[epoch][i] * other_weights[i]
                        
                        # 使用缓存的梯度
                        client_grad = cached_gradients['others'][i][param_idx] * adaptive_strength
                        combined_grad += client_grad
                    
                    # 计算补偿后的遗忘强度
                    if self.use_forget_compensation:
                        # 先计算当前锚点权重（用于补偿计算）
                        current_anchor_weight = self._compute_progressive_anchor_weight(epoch, epochs) if self.use_progressive_weight else self.anchor_weight
                        compensated_lambda = self._compute_compensated_forget_strength(
                            lambda_reversal, current_anchor_weight, epoch, epochs
                        )
                    else:
                        compensated_lambda = lambda_reversal
                    
                    # 梯度反转：目标客户端梯度取反（使用补偿后的遗忘强度）
                    reversed_grad = -compensated_lambda * protection_scale * (epoch + 1) / epochs * cached_gradients['target'][param_idx]
                    
                    # 使用遗忘掩码将反转限制在目标相关参数上
                    mask_param = None
                    try:
                        if hasattr(self, '_forget_mask') and self._forget_mask is not None:
                            mask_param = self._forget_mask[param_idx]
                            if mask_param is not None:
                                reversed_grad = reversed_grad * mask_param.to(reversed_grad.device)
                    except Exception:
                        pass
                    
                    # 对其他客户端的组合梯度做总量约束（如果启用智能梯度组合，则稍后调整）
                    if len(other_clients) > 0:
                        if not self.use_intelligent_gradient:
                            # 原始方法：使用0.8约束
                            combined_grad = combined_grad * (0.8 / max(weight_denominator, 1e-12))
                        else:
                            # 智能方法：移除过度约束，稍后通过智能组合调整
                            combined_grad = combined_grad / max(weight_denominator, 1e-12)
                    
                    # 若存在遗忘掩码，只在非掩码参数上保留其他客户端梯度，避免抵消遗忘
                    # 注意：如果启用智能梯度组合，这一步会在智能组合中处理
                    if not self.use_intelligent_gradient:
                        try:
                            if mask_param is not None:
                                combined_grad = combined_grad * (1.0 - mask_param.to(combined_grad.device))
                        except Exception:
                            pass

                    # 记录梯度量级用于诊断
                    total_reversed_norm += torch.norm(reversed_grad).item()
                    total_combined_norm += torch.norm(combined_grad).item()
                    
                    # 组合梯度：使用智能梯度组合或原始方法
                    if self.use_intelligent_gradient:
                        param.grad = self._compute_intelligent_gradient_combination(
                            reversed_grad, combined_grad, mask_param
                        )
                    else:
                        param.grad = reversed_grad + combined_grad
                    param_idx += 1
            
            # 知识锚点保护 - 从指定轮次开始应用
            anchor_loss = None
            if (self.use_knowledge_anchoring and 
                epoch >= self.anchor_start_epoch and 
                self.original_model is not None):
                
                try:
                    # 保存当前轮次用于选择性保护的信息打印
                    self._last_epoch = epoch
                    
                    anchor_loss = self._apply_knowledge_anchoring(
                        self.original_model, model, self.anchor_clients, self.anchor_sample_ratio
                    )
                    
                    # 如果锚点损失有效，添加到总损失中
                    if anchor_loss is not None and anchor_loss.item() > 0:
                        # 计算渐进式锚点权重
                        if self.use_progressive_weight:
                            dynamic_anchor_weight = self._compute_progressive_anchor_weight(epoch, epochs)
                        else:
                            # 原始动态调整逻辑（保持向后兼容）
                            dynamic_anchor_weight = self.anchor_weight
                            ratio = total_reversed_norm / max(total_combined_norm, 1e-8)
                            if ratio > 2.0:
                                dynamic_anchor_weight = min(0.6, self.anchor_weight * 1.5)
                            elif ratio > 1.2:
                                dynamic_anchor_weight = min(0.5, self.anchor_weight * 1.2)
                        
                        # 计算锚点损失的梯度
                        anchor_loss = anchor_loss * dynamic_anchor_weight
                        anchor_loss.backward(retain_graph=True)
                        
                        # 打印锚点保护信息
                        if epoch % 2 == 0:
                            weight_type = "渐进式" if self.use_progressive_weight else "动态"
                            print(f"  📌 知识锚点保护 - 损失: {anchor_loss.item():.6f}, 基础权重: {self.anchor_weight:.3f}, {weight_type}权重: {dynamic_anchor_weight:.3f}")
                            if self.use_selective_anchoring:
                                print(f"     选择性保护已启用，阈值: {self.selective_protection_threshold}")
                    
                except Exception as e:
                    print(f"警告：知识锚点保护应用失败: {e}")
            
            # 打印诊断信息
            if epoch % 2 == 0:
                # 计算实际使用的遗忘强度
                if self.use_forget_compensation:
                    current_anchor_weight = self._compute_progressive_anchor_weight(epoch, epochs) if self.use_progressive_weight else self.anchor_weight
                    actual_forget_strength = self._compute_compensated_forget_strength(
                        lambda_reversal, current_anchor_weight, epoch, epochs
                    )
                    forget_info = f"补偿遗忘强度: {actual_forget_strength:.3f} (基础: {lambda_reversal * (epoch + 1) / epochs:.3f})"
                else:
                    forget_info = f"基础遗忘强度: {lambda_reversal * (epoch + 1) / epochs:.3f}"
                
                print(f"Epoch {epoch}: 优化版梯度反转训练中... ({forget_info}, 保护缩放: {protection_scale:.3f})")
                print(f"  梯度量级诊断 - 反转梯度: {total_reversed_norm:.6f}, 组合梯度: {total_combined_norm:.6f}, 比例: {total_reversed_norm/max(total_combined_norm, 1e-8):.3f}")
                if anchor_loss is not None and anchor_loss.item() > 0:
                    print(f"  📌 知识锚点保护已启用")
                if self.use_intelligent_gradient:
                    ratio = total_reversed_norm / max(total_combined_norm, 1e-8)
                    if ratio < 1.5:
                        print(f"  ⚠️ 智能梯度组合已调整保护梯度")
            
            optimizer.step()
            
            # 定期清理缓存，避免内存泄漏
            if epoch % self.opt_config.cache_cleanup_frequency == 0 and epoch > 0:
                self._cleanup_gradient_cache()

        if self.retain_calibration_rounds > 0 and len(other_clients) > 0:
            print("8. 开始掩码约束的保留校准...")
            model = self._run_masked_retain_calibration(
                model,
                other_clients,
                target_client_id,
                distances,
                cluster_labels,
                other_weights,
            )

        print("优化版梯度反转遗忘完成！")
        
        # 清理缓存
        self._cleanup_gradient_cache()
        self._clear_anchor_cache()  # 清理锚点数据缓存
        
        # 验证遗忘效果
        self._validate_forgetting_effect(model, target_client, other_clients, pre_accuracies)
        
        # 打印性能统计
        self._print_performance_summary()
        
        return model

    def _run_masked_retain_calibration(self, model, other_clients, target_client_id, distances,
                                       cluster_labels, other_weights):
        """仅用保留客户端做轻量恢复，并在遗忘掩码上抑制更新，减少目标知识回流。"""
        calibrated_model = copy.deepcopy(model)
        optimizer = torch.optim.SGD(
            calibrated_model.parameters(),
            lr=self.retain_calibration_lr,
            momentum=0.9,
            weight_decay=1e-4,
        )

        sample_ratio = min(self.opt_config.sample_ratio, 0.2)
        max_batches = min(self.opt_config.max_batches, self.retain_calibration_batches)
        weight_denominator = float(np.sum(other_weights)) if len(other_weights) > 0 else 1.0

        for round_idx in range(self.retain_calibration_rounds):
            optimizer.zero_grad()
            aggregated_grads = None

            for client_idx, client in enumerate(other_clients):
                client_grads = self._compute_client_gradients_optimized(
                    calibrated_model,
                    client,
                    sample_ratio=sample_ratio,
                    max_batches=max_batches,
                )
                client_weight = other_weights[client_idx]
                if aggregated_grads is None:
                    aggregated_grads = [grad * client_weight for grad in client_grads]
                else:
                    for grad_idx, grad in enumerate(client_grads):
                        aggregated_grads[grad_idx] += grad * client_weight

            if aggregated_grads is None:
                print("  警告：保留校准阶段未获得有效梯度，提前结束")
                break

            total_grad_norm = 0.0
            param_idx = 0
            for param in calibrated_model.parameters():
                if not param.requires_grad:
                    continue

                grad = aggregated_grads[param_idx] / max(weight_denominator, 1e-12)
                if hasattr(self, '_forget_mask') and self._forget_mask is not None:
                    mask_param = self._forget_mask[param_idx]
                    if mask_param is not None:
                        mask_param = mask_param.to(grad.device)
                        grad = grad * (1.0 - mask_param) + grad * mask_param * self.mask_retain_scale

                param.grad = grad
                total_grad_norm += torch.norm(grad).item()
                param_idx += 1

            torch.nn.utils.clip_grad_norm_(calibrated_model.parameters(), max_norm=1.0)
            optimizer.step()

            print(
                f"  保留校准轮 {round_idx + 1}/{self.retain_calibration_rounds}: "
                f"平均梯度量级 {total_grad_norm / max(param_idx, 1):.6f}, "
                f"掩码保留系数 {self.mask_retain_scale:.3f}"
            )

        return calibrated_model
    
    def _apply_knowledge_anchoring(self, original_model, current_model, anchor_clients, sample_ratio=0.1):
        """
        应用知识锚点保护，使用原始模型作为锚点防止知识过度丢失
        
        如果启用了选择性保护，则使用选择性锚点保护方法
        
        Args:
            original_model: 原始全局模型（锚点）
            current_model: 当前训练中的模型
            anchor_clients: 需要保护的客户端列表
            sample_ratio: 锚点数据采样比例
            
        Returns:
            anchor_loss: 知识锚点损失
        """
        if not self.use_knowledge_anchoring or len(anchor_clients) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 如果启用选择性保护，使用选择性锚点保护方法
        if self.use_selective_anchoring:
            return self._apply_selective_knowledge_anchoring(
                original_model, current_model, anchor_clients, sample_ratio
            )
        
        # 否则使用原始方法（保持向后兼容）
        try:
            anchor_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            total_samples = 0
            
            for client in anchor_clients:
                # 获取或缓存锚点数据
                anchor_data = self._get_anchor_data(client, sample_ratio)
                if anchor_data is None or len(anchor_data) == 0:
                    continue
                
                # 将数据移到正确的设备
                anchor_data = anchor_data.to(self.device)
                
                # 原始模型预测（锚点知识）
                with torch.no_grad():
                    try:
                        anchor_output = original_model(anchor_data)
                    except Exception as e:
                        print(f"警告：原始模型预测失败，跳过客户端{client.id}: {e}")
                        continue
                
                # 当前模型预测
                try:
                    current_output = current_model(anchor_data)
                except Exception as e:
                    print(f"警告：当前模型预测失败，跳过客户端{client.id}: {e}")
                    continue
                
                # 计算KL散度锚点损失（相似度加权：与目标更相似的客户端权重更高）
                try:
                    kl_loss = F.kl_div(
                        F.log_softmax(current_output, dim=1),
                        F.softmax(anchor_output, dim=1),
                        reduction='batchmean'
                    )
                    
                    # 相似度加权：距离越小，权重越高
                    weight = 1.0
                    try:
                        if hasattr(self, '_last_distances') and hasattr(self, '_last_target_client_id'):
                            d = float(self._last_distances[self._last_target_client_id][client.id])
                            # 将距离映射为相似度权重：距离小→权重大（0.5~1.5）
                            if np.isfinite(d):
                                if d < 0.6:
                                    weight = 1.5
                                elif d < 1.0:
                                    weight = 1.2
                                elif d < 1.5:
                                    weight = 1.0
                                else:
                                    weight = 0.8
                    except Exception:
                        weight = 1.0
                    
                    anchor_loss = anchor_loss + weight * kl_loss
                    total_samples += 1
                except Exception as e:
                    print(f"警告：KL散度计算失败，跳过客户端{client.id}: {e}")
                    continue
            
            # 如果没有有效样本，返回零损失
            if total_samples == 0:
                return torch.tensor(0.0, device=self.device, requires_grad=True)
            
            # 平均损失
            anchor_loss = anchor_loss / total_samples
            
            return anchor_loss
            
        except Exception as e:
            print(f"知识锚点保护应用失败: {e}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
    
    def _apply_selective_knowledge_anchoring(self, original_model, current_model, anchor_clients, sample_ratio=0.1):
        """
        选择性知识锚点保护：只保护与目标客户端差异较大的客户端
        
        策略：
        - 距离 > 1.0：强保护（权重1.5）
        - 距离 0.6-1.0：中等保护（权重1.0）
        - 距离 < 0.6：弱保护或取消保护（权重0.3），避免影响遗忘
        
        Args:
            original_model: 原始全局模型（锚点）
            current_model: 当前训练中的模型
            anchor_clients: 需要保护的客户端列表
            sample_ratio: 锚点数据采样比例
            
        Returns:
            anchor_loss: 知识锚点损失
        """
        try:
            anchor_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            total_samples = 0
            protected_clients = []
            skipped_clients = []
            
            for client in anchor_clients:
                # 计算与目标客户端的距离
                distance_to_target = None
                try:
                    if hasattr(self, '_last_distances') and hasattr(self, '_last_target_client_id'):
                        distance_to_target = float(self._last_distances[self._last_target_client_id][client.id])
                except Exception:
                    pass
                
                # 选择性保护：距离越大，保护越强
                if distance_to_target is not None and np.isfinite(distance_to_target):
                    if distance_to_target > 1.0:
                        # 差异大：强保护
                        client_protection_weight = 1.5
                    elif distance_to_target > self.selective_protection_threshold:
                        # 中等差异：中等保护
                        client_protection_weight = 1.0
                    else:
                        # 相似：弱保护（避免影响遗忘）
                        client_protection_weight = 0.3
                else:
                    # 如果没有距离信息，使用中等保护
                    client_protection_weight = 1.0
                
                # 如果保护权重太低，跳过该客户端
                if client_protection_weight < 0.5:
                    skipped_clients.append((client.id, distance_to_target))
                    continue
                
                # 获取锚点数据
                anchor_data = self._get_anchor_data(client, sample_ratio)
                if anchor_data is None or len(anchor_data) == 0:
                    continue
                
                anchor_data = anchor_data.to(self.device)
                
                # 原始模型预测（锚点知识）
                with torch.no_grad():
                    try:
                        anchor_output = original_model(anchor_data)
                    except Exception as e:
                        print(f"警告：原始模型预测失败，跳过客户端{client.id}: {e}")
                        continue
                
                # 当前模型预测
                try:
                    current_output = current_model(anchor_data)
                except Exception as e:
                    print(f"警告：当前模型预测失败，跳过客户端{client.id}: {e}")
                    continue
                
                # 计算KL散度损失
                try:
                    kl_loss = F.kl_div(
                        F.log_softmax(current_output, dim=1),
                        F.softmax(anchor_output, dim=1),
                        reduction='batchmean'
                    )
                    
                    # 应用选择性保护权重
                    weighted_loss = client_protection_weight * kl_loss
                    anchor_loss = anchor_loss + weighted_loss
                    total_samples += 1
                    protected_clients.append((client.id, distance_to_target, client_protection_weight))
                    
                except Exception as e:
                    print(f"警告：KL散度计算失败，跳过客户端{client.id}: {e}")
                    continue
            
            # 如果没有有效样本，返回零损失
            if total_samples == 0:
                return torch.tensor(0.0, device=self.device, requires_grad=True)
            
            # 平均损失
            anchor_loss = anchor_loss / total_samples
            
            # 打印选择性保护信息（每10轮打印一次）
            if hasattr(self, '_last_epoch') and self._last_epoch % 10 == 0:
                print(f"  📌 选择性锚点保护:")
                print(f"     保护客户端数: {len(protected_clients)}, 跳过客户端数: {len(skipped_clients)}")
                if protected_clients:
                    print(f"     保护客户端详情: {protected_clients[:3]}...")  # 只显示前3个
            
            return anchor_loss
            
        except Exception as e:
            print(f"选择性知识锚点保护应用失败: {e}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
    
    def _get_anchor_data(self, client, sample_ratio=0.1):
        """
        获取客户端的锚点数据，使用缓存机制避免重复采样
        
        Args:
            client: 客户端对象
            sample_ratio: 采样比例
            
        Returns:
            anchor_data: 锚点数据
        """
        try:
            # 检查缓存
            cache_key = f"client_{client.id}_anchor_data"
            if cache_key in self.anchor_data_cache:
                return self.anchor_data_cache[cache_key]
            
            # 获取客户端训练数据
            if hasattr(client, 'trainloader') and client.trainloader is not None:
                trainloader = client.trainloader
            elif hasattr(client, 'train_data') and client.train_data is not None:
                # 如果没有trainloader，尝试从train_data创建
                try:
                    from torch.utils.data import DataLoader, TensorDataset
                    dataset = TensorDataset(client.train_data, client.train_labels)
                    trainloader = DataLoader(dataset, batch_size=32, shuffle=True)
                except Exception:
                    return None
            else:
                return None
            
            # 采样锚点数据
            anchor_samples = []
            anchor_labels = []
            total_samples = 0
            max_samples = int(len(trainloader.dataset) * sample_ratio)
            
            for batch_idx, (data, target) in enumerate(trainloader):
                if total_samples >= max_samples:
                    break
                
                # 确保数据类型正确
                if isinstance(data, torch.Tensor):
                    batch_data = data
                    batch_labels = target
                else:
                    # 如果是其他格式，尝试转换
                    try:
                        batch_data = torch.tensor(data, dtype=torch.float32)
                        batch_labels = torch.tensor(target, dtype=torch.long)
                    except Exception:
                        continue
                
                # 添加到锚点数据
                anchor_samples.append(batch_data)
                anchor_labels.append(batch_labels)
                total_samples += batch_data.size(0)
                
                if total_samples >= max_samples:
                    break
            
            if not anchor_samples:
                return None
            
            # 合并所有批次
            try:
                anchor_data = torch.cat(anchor_samples, dim=0)
                anchor_labels = torch.cat(anchor_labels, dim=0)
                
                # 缓存结果
                self.anchor_data_cache[cache_key] = (anchor_data, anchor_labels)
                
                return anchor_data
                
            except Exception as e:
                print(f"锚点数据合并失败: {e}")
                return None
                
        except Exception as e:
            print(f"获取锚点数据失败: {e}")
            return None
    
    def _clear_anchor_cache(self):
        """清理锚点数据缓存"""
        self.anchor_data_cache.clear()
    
    def _compute_progressive_anchor_weight(self, epoch, max_epochs, base_weight=None):
        """
        渐进式锚点权重：早期低权重，后期高权重
        
        策略：
        - 前30%轮次：低权重（0.3-0.4），优先遗忘
        - 中间40%轮次：中等权重（0.5-0.6），平衡遗忘和保护
        - 后30%轮次：高权重（0.7-0.8），强化保护
        
        Args:
            epoch: 当前轮次
            max_epochs: 总轮次
            base_weight: 基础权重（如果为None，使用self.anchor_weight）
            
        Returns:
            progressive_weight: 渐进式权重
        """
        if not self.use_progressive_weight:
            return self.anchor_weight if base_weight is None else base_weight
        
        if base_weight is None:
            base_weight = self.anchor_weight
        
        progress = epoch / max_epochs if max_epochs > 0 else 0.5
        
        if progress < 0.3:
            # 早期：低权重，优先遗忘
            weight = base_weight * 0.6  # 例如：0.5 * 0.6 = 0.3
        elif progress < 0.7:
            # 中期：中等权重，平衡
            weight = base_weight * 1.0  # 例如：0.5 * 1.0 = 0.5
        else:
            # 后期：高权重，强化保护
            weight = base_weight * 1.4  # 例如：0.5 * 1.4 = 0.7
        
        # 限制在合理范围内
        weight = max(0.1, min(1.0, weight))
        
        return weight
    
    def _compute_compensated_forget_strength(self, base_lambda, anchor_weight, epoch, max_epochs):
        """
        补偿性遗忘强度：根据锚点权重动态调整遗忘强度
        
        策略：
        - 锚点权重高 → 增加遗忘强度（补偿）
        - 确保遗忘效果不受保护影响
        
        Args:
            base_lambda: 基础遗忘强度
            anchor_weight: 当前锚点权重
            epoch: 当前轮次
            max_epochs: 总轮次
            
        Returns:
            compensated_strength: 补偿后的遗忘强度
        """
        if not self.use_forget_compensation:
            return base_lambda * (epoch + 1) / max_epochs if max_epochs > 0 else base_lambda
        
        # 基础遗忘强度随训练进度增加
        progress_strength = base_lambda * (epoch + 1) / max_epochs if max_epochs > 0 else base_lambda
        
        # 根据锚点权重补偿
        if anchor_weight > 0.6:
            # 高保护 → 高遗忘强度补偿
            compensation_factor = 1.3
        elif anchor_weight > 0.4:
            # 中等保护 → 中等补偿
            compensation_factor = 1.1
        else:
            # 低保护 → 无补偿
            compensation_factor = 1.0
        
        compensated_strength = progress_strength * compensation_factor
        
        return compensated_strength
    
    def _compute_intelligent_gradient_combination(self, reversed_grad, combined_grad, forget_mask=None):
        """
        智能梯度组合：确保遗忘梯度占主导
        
        策略：
        - 计算遗忘梯度和保护梯度的比例
        - 如果保护梯度过大，动态调整
        - 确保遗忘效果不受影响
        
        Args:
            reversed_grad: 反转的目标客户端梯度（遗忘梯度）
            combined_grad: 其他客户端的组合梯度（保护梯度）
            forget_mask: 遗忘掩码（可选）
            
        Returns:
            final_grad: 最终组合梯度
        """
        if not self.use_intelligent_gradient:
            # 如果不启用智能组合，使用原始方法
            if forget_mask is not None:
                combined_grad = combined_grad * (1.0 - forget_mask)
            return reversed_grad + combined_grad
        
        try:
            reversed_norm = torch.norm(reversed_grad).item()
            combined_norm = torch.norm(combined_grad).item()
            
            # 计算比例
            ratio = reversed_norm / max(combined_norm, 1e-8)
            
            # 如果保护梯度过大（比例 < 1.5），减少保护梯度
            if ratio < 1.5:
                # 动态调整：减少保护梯度，确保遗忘占主导
                reduction_factor = max(0.5, ratio / 1.5)  # 至少保留50%
                combined_grad = combined_grad * reduction_factor
                if hasattr(self, '_last_epoch') and self._last_epoch % 5 == 0:
                    print(f"  ⚠️ 保护梯度过大，调整因子: {reduction_factor:.3f}, 比例: {ratio:.3f}")
            
            # 在目标相关参数上，进一步减少保护梯度
            if forget_mask is not None:
                combined_grad = combined_grad * (1.0 - forget_mask)
            
            return reversed_grad + combined_grad
            
        except Exception as e:
            print(f"警告：智能梯度组合计算失败: {e}")
            # 失败时使用原始方法
            if forget_mask is not None:
                combined_grad = combined_grad * (1.0 - forget_mask)
            return reversed_grad + combined_grad
    
    def _compute_global_protection_scale(self, target_client_id, other_clients, cluster_labels, distances):
        """根据整体相似性与分簇情况计算全局保护缩放系数(0.5-1.0)。
        更激进地在高相似场景下降低反转强度，保护其他客户端。
        """
        try:
            if cluster_labels is None or distances is None or other_clients is None:
                return 1.0
            if len(other_clients) == 0:
                return 1.0
            similarity_scores = []
            for c in other_clients:
                try:
                    d = float(distances[target_client_id][c.id])
                except Exception:
                    d = 1e9
                # 相似度指示：距离越小，相似度越高（0.0~1.0）
                if d < 0.6:
                    similarity_scores.append(1.0)
                elif d < 0.9:
                    similarity_scores.append(0.7)
                elif d < 1.2:
                    similarity_scores.append(0.4)
                else:
                    similarity_scores.append(0.0)
            if not similarity_scores:
                return 1.0
            avg_sim = float(np.mean(similarity_scores))
            # 当整体相似性高时，更强地降低反转强度（0.5~1.0）
            scale = 1.0 - 0.5 * avg_sim
            return max(0.5, min(1.0, scale))
        except Exception:
            return 1.0
    
    def _compute_model_hash(self, model):
        """计算模型参数的哈希值，用于检测模型变化"""
        import hashlib
        
        model_hash = hashlib.md5()
        for param in model.parameters():
            if param.requires_grad:
                model_hash.update(param.data.cpu().numpy().tobytes())
        
        return model_hash.hexdigest()
    
    def _cleanup_gradient_cache(self):
        """清理梯度缓存，释放内存"""
        if hasattr(self, '_gradient_cache'):
            cache_size = len(self._gradient_cache)
            max_cache_size = self.opt_config.max_cache_size
            
            if cache_size > max_cache_size:
                print(f"清理梯度缓存，当前大小: {cache_size}, 最大允许: {max_cache_size}")
                
                # 保留最近的缓存项
                cache_items = list(self._gradient_cache.items())
                # 按时间戳排序，保留最新的
                cache_items.sort(key=lambda x: x[1].get('timestamp', 0) if isinstance(x[1], dict) else 0, reverse=True)
                
                # 保留前一半
                keep_count = max_cache_size // 2
                self._gradient_cache = dict(cache_items[:keep_count])
                
                print(f"缓存已清理，保留 {keep_count} 项")
                
                # 强制垃圾回收
                import gc
                gc.collect()
                
                # 清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    def _precompute_adaptive_strengths(self, target_client, other_clients, epochs, lambda_reversal, 
                                     cluster_labels, distances):
        """预计算自适应遗忘强度矩阵，避免重复计算"""
        print("预计算自适应遗忘强度矩阵...")
        adaptive_strength_matrix = []
        
        # 使用向量化计算提高效率
        for epoch in range(epochs):
            epoch_strengths = []
            for client in other_clients:
                strength = self._adaptive_forget_strength(
                    target_client, client, epoch, epochs, lambda_reversal,
                    cluster_labels, distances
                )
                epoch_strengths.append(strength)
            adaptive_strength_matrix.append(epoch_strengths)
        
        print(f"预计算完成，矩阵大小: {len(adaptive_strength_matrix)} x {len(adaptive_strength_matrix[0])}")
        return adaptive_strength_matrix
    
    def _compute_cached_gradients(self, model, target_client, other_clients):
        """使用改进的缓存机制计算梯度，避免重复计算"""
        start_time = time.time()
        start_memory = self._measure_memory_usage()
        
        cached_gradients = {}
        
        # 使用优化配置中的参数
        sample_ratio = self.opt_config.sample_ratio
        max_batches = self.opt_config.max_batches
        num_processes = self.opt_config.num_processes
        use_gradient_caching = self.opt_config.use_gradient_caching
        
        # 检查是否已有缓存的梯度
        if hasattr(self, '_gradient_cache') and use_gradient_caching:
            cache_key = self._generate_cache_key(model, target_client, other_clients)
            if cache_key in self._gradient_cache:
                print("使用缓存的梯度，跳过重新计算...")
                self._update_performance_stats('cached')
                return self._gradient_cache[cache_key]
        
        # 计算目标客户端梯度（只计算一次）
        print("计算目标客户端梯度...")
        cached_gradients['target'] = self._compute_client_gradients_optimized(
            model, target_client, sample_ratio, max_batches
        )
        
        # 使用改进的并行计算其他客户端梯度
        print("并行计算其他客户端梯度...")
        cached_gradients['others'] = self._compute_parallel_client_gradients_improved(
            model, other_clients, sample_ratio, max_batches, num_processes
        )
        
        # 缓存结果
        if use_gradient_caching:
            if not hasattr(self, '_gradient_cache'):
                self._gradient_cache = {}
            else:
                self._gradient_cache.clear()
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            cache_key = self._generate_cache_key(model, target_client, other_clients)
            self._gradient_cache[cache_key] = cached_gradients
            # 添加时间戳用于缓存管理
            self._gradient_cache[cache_key]['timestamp'] = time.time()
            print(f"梯度已缓存，缓存键: {cache_key[:20]}...")
        
        # 更新性能统计
        end_time = time.time()
        end_memory = self._measure_memory_usage()
        time_taken = end_time - start_time
        memory_used = end_memory - start_memory
        
        self._update_performance_stats('computed', time_taken, end_memory)
        
        return cached_gradients
    
    def _generate_cache_key(self, model, target_client, other_clients):
        """生成缓存键，基于模型参数和客户端ID"""
        import hashlib
        
        # 基于模型参数生成哈希
        model_hash = hashlib.md5()
        for param in model.parameters():
            if param.requires_grad:
                model_hash.update(param.data.cpu().numpy().tobytes())
        
        # 基于客户端ID生成哈希
        client_ids = [target_client.id] + [c.id for c in other_clients]
        client_hash = hashlib.md5(str(sorted(client_ids)).encode()).hexdigest()
        
        # 组合哈希
        combined_hash = model_hash.hexdigest()[:16] + client_hash[:16]
        return combined_hash
    
    def _compute_client_gradients_optimized(self, model, client, sample_ratio=0.2, max_batches=4):
        """进一步优化的客户端梯度计算 - 使用更激进的采样和缓存"""
        client.set_parameters(model)
        client.model.train()
        return self._collect_client_gradients(
            client,
            max_batches=max_batches,
            sample_ratio=sample_ratio,
        )
    
    def _improved_data_sampling(self, trainloader, sample_ratio=0.2, max_batches=4):
        """改进的数据采样策略 - 分层采样确保数据代表性"""
        all_data = list(trainloader)
        
        if len(all_data) <= max_batches:
            return all_data
        
        # 分层采样：按类别分组采样
        class_data = {}
        for data, target in all_data:
            # 获取批次中第一个样本的类别
            if hasattr(target, 'dim') and target.dim() > 0:
                class_id = target[0].item()
            else:
                class_id = int(target) if not isinstance(target, (list, tuple)) else int(target[0])
            
            if class_id not in class_data:
                class_data[class_id] = []
            class_data[class_id].append((data, target))
        
        # 从每个类别采样，确保数据多样性
        import random
        sampled_data = []
        num_classes = max(1, len(class_data))
        samples_per_class = max(1, max_batches // num_classes)
        
        for class_id, samples in class_data.items():
            if len(samples) <= samples_per_class:
                sampled_data.extend(samples)
            else:
                sampled_indices = random.sample(range(len(samples)), samples_per_class)
                sampled_data.extend([samples[i] for i in sampled_indices])
        
        # 如果采样数据不足，补充随机数据（避免对Tensor的直接比较）
        if len(sampled_data) < max_batches:
            remaining_needed = max_batches - len(sampled_data)
            chosen_ids = {id(item) for item in sampled_data}
            remaining_pool = [item for item in all_data if id(item) not in chosen_ids]
            if remaining_pool:
                additional = random.sample(remaining_pool, min(remaining_needed, len(remaining_pool)))
                sampled_data.extend(additional)
        
        return sampled_data[:max_batches]
    
    def _compute_parallel_client_gradients_improved(self, model, other_clients, sample_ratio=0.2, max_batches=4, num_processes=2):
        """改进的并行客户端梯度计算 - 更稳定和高效"""
        if len(other_clients) == 0:
            return []

        model_device = next(model.parameters()).device
        if self.device.type == 'cuda' or model_device.type == 'cuda':
            print("检测到 CUDA 模型，使用串行梯度计算以避免多进程 CUDA IPC 错误...")
            return [self._compute_client_gradients_optimized(model, client, sample_ratio, max_batches)
                   for client in other_clients]
        
        # 使用优化配置中的参数
        min_clients_for_parallel = self.opt_config.min_clients_for_parallel
        max_parallel_processes = self.opt_config.max_parallel_processes
        
        # 如果客户端数量较少，或显式要求单进程，则直接串行计算
        if (len(other_clients) < min_clients_for_parallel or
                num_processes <= 1 or
                not self.opt_config.use_parallel_computation):
            print(f"客户端数量较少({len(other_clients)})，使用串行计算...")
            return [self._compute_client_gradients_optimized(model, client, sample_ratio, max_batches) 
                   for client in other_clients]
        
        try:
            # 尝试使用多进程并行计算
            import torch.multiprocessing as mp
            from functools import partial
            
            # 设置多进程启动方法
            if mp.get_start_method() != 'spawn':
                mp.set_start_method('spawn', force=True)
            
            # 限制进程数，避免过多进程导致的不稳定
            num_processes = min(len(other_clients), num_processes, max_parallel_processes)
            
            print(f"使用 {num_processes} 个进程并行计算梯度...")
            
            with mp.Pool(processes=num_processes) as pool:
                # 并行计算梯度
                gradient_func = partial(self._compute_single_client_gradient, model, sample_ratio, max_batches)
                results = pool.map(gradient_func, other_clients)
            
            return results
            
        except Exception as e:
            print(f"并行计算失败，回退到串行计算: {e}")
            # 回退到串行计算
            return [self._compute_client_gradients_optimized(model, client, sample_ratio, max_batches) 
                   for client in other_clients]
    
    def _compute_single_client_gradient(self, model, sample_ratio, max_batches, client):
        """单个客户端梯度计算函数（用于多进程）"""
        try:
            # 设置设备
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # 创建模型的副本以避免进程间冲突
            model_copy = copy.deepcopy(model)
            model_copy = model_copy.to(device)
            
            # 计算梯度
            gradients = self._compute_client_gradients_optimized(model_copy, client, sample_ratio, max_batches)
            
            # 清理GPU内存
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            return gradients
            
        except Exception as e:
            print(f"计算客户端{client.id}梯度时出错: {e}")
            # 返回零梯度作为备选
            return self._get_zero_gradients(model)
    
    def _get_zero_gradients(self, model):
        """获取零梯度作为备选"""
        zero_gradients = []
        for param in model.parameters():
            if param.requires_grad:
                zero_gradients.append(torch.zeros_like(param))
        return zero_gradients
    
    def _average_gradients(self, gradients):
        """改进的梯度平均计算 - 处理空梯度情况"""
        if not gradients or len(gradients) == 0:
            return []
        
        # 过滤掉空梯度
        valid_gradients = [g for g in gradients if g is not None and len(g) > 0]
        
        if not valid_gradients:
            return []
        
        avg_gradients = []
        for param_idx in range(len(valid_gradients[0])):
            avg_grad = torch.zeros_like(valid_gradients[0][param_idx])
            valid_count = 0
            
            for batch_grads in valid_gradients:
                if batch_grads and param_idx < len(batch_grads):
                    avg_grad += batch_grads[param_idx]
                    valid_count += 1
            
            if valid_count > 0:
                avg_gradients.append(avg_grad / valid_count)
            else:
                avg_gradients.append(avg_grad)
        
        return avg_gradients
    
    def _compute_weighted_gradients_batch(self, gradients_list, weights):
        """批量计算加权梯度"""
        if not gradients_list:
            return []
        
        weighted_gradients = []
        for param_idx in range(len(gradients_list[0])):
            weighted_grad = torch.zeros_like(gradients_list[0][param_idx])
            for i, gradients in enumerate(gradients_list):
                weighted_grad += gradients[param_idx] * weights[i]
            weighted_gradients.append(weighted_grad)
        
        return weighted_gradients
    
    def _print_performance_summary(self):
        """打印性能统计摘要"""
        print("\n============= 性能优化统计 =============")
        
        if self.performance_stats['total_computations'] > 0:
            cache_hit_rate = self.performance_stats['cached_computations'] / self.performance_stats['total_computations']
            self.performance_stats['cache_hit_rate'] = cache_hit_rate
            
            print(f"梯度计算统计:")
            print(f"  - 总计算次数: {self.performance_stats['total_computations']}")
            print(f"  - 缓存命中次数: {self.performance_stats['cached_computations']}")
            print(f"  - 缓存命中率: {cache_hit_rate:.2%}")
            
            if self.performance_stats['gradient_computation_time']:
                avg_time = np.mean(self.performance_stats['gradient_computation_time'])
                print(f"  - 平均梯度计算时间: {avg_time:.2f}秒")
        
        if self.performance_stats['memory_usage']:
            max_memory = max(self.performance_stats['memory_usage'])
            print(f"内存使用统计:")
            print(f"  - 最大内存使用: {max_memory:.2f}MB")
        
        print("优化配置效果:")
        opt_summary = self.opt_config.get_optimization_summary()
        for key, value in opt_summary.items():
            print(f"  - {key}: {value}")
    
    def _update_performance_stats(self, computation_type, time_taken=None, memory_used=None):
        """更新性能统计"""
        self.performance_stats['total_computations'] += 1
        
        if computation_type == 'cached':
            self.performance_stats['cached_computations'] += 1
        
        if time_taken is not None:
            self.performance_stats['gradient_computation_time'].append(time_taken)
        
        if memory_used is not None:
            self.performance_stats['memory_usage'].append(memory_used)
    
    def _measure_memory_usage(self):
        """测量当前内存使用情况"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024  # MB
        else:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB


def forget_client_with_gradient_reversal(global_model, clients, args, target_client_id=0, pre_accuracies=None):
    """梯度反转遗忘的包装函数"""
    reversal_forget = GradientReversalForgetting(args)
    
    # 检查是否使用优化版本
    use_optimized = getattr(args, 'use_optimized_forgetting', False)
    
    if use_optimized:
        print("使用优化版梯度反转遗忘...")
        return reversal_forget.forget_with_gradient_reversal_optimized(
            global_model, clients, target_client_id, pre_accuracies=pre_accuracies
        )
    else:
        print("使用标准版梯度反转遗忘...")
        return reversal_forget.forget_with_gradient_reversal(
            global_model, clients, target_client_id, pre_accuracies=pre_accuracies
        )

def forget_client_with_gradient_reversal_optimized(global_model, clients, args, target_client_id=0, pre_accuracies=None):
    """优化版梯度反转遗忘的包装函数"""
    reversal_forget = GradientReversalForgetting(args)
    return reversal_forget.forget_with_gradient_reversal_optimized(
        global_model, clients, target_client_id, pre_accuracies=pre_accuracies
    )

def forget_client_wrapper(global_model, clients, args, target_client_id=0):
    """遗忘客户端的包装函数"""
    forget_module = ClientForgetting(args)
    return forget_module.forget_client(global_model, clients, target_client_id)

# ============================================================================
# 对比方法适配器：FedCSA, FedOSD, FedU
# ============================================================================

def forget_client_with_fedcsa(global_model, clients, args, target_client_id=0):
    """
    FedCSA方法：基于客户端分组的遗忘方法
    核心思路：将客户端分成不同的子集，每个子集独立训练模型，然后对于需要遗忘的客户端，
    在聚合时排除它所在的子集
    
    修复要点：
    1. 实现真正的客户端分组（Client Subset Aggregation）
    2. 支持SCAFFOLD算法的控制变量
    3. 增加本地训练轮数以增强遗忘效果
    4. 使用对抗性训练来擦除目标客户端的影响
    """
    print(f"\n============= FedCSA 遗忘方法 =============")
    print(f"目标遗忘客户端: {target_client_id}")
    
    # 检查是否使用SCAFFOLD算法
    is_scaffold = getattr(args, 'algorithm', '').upper() == 'SCAFFOLD'
    
    # 获取所有客户端ID（排除目标客户端）
    all_client_ids = list(range(len(clients)))
    other_client_ids = [i for i in all_client_ids if i != target_client_id]
    target_client = clients[target_client_id]
    
    if len(other_client_ids) == 0:
        print("警告：没有其他客户端，返回原始模型")
        return global_model
    
    # ========== 步骤1: 客户端分组 ==========
    # 将客户端分成多个子集（使用K-means聚类或基于数据分布）
    def _determine_optimal_subset_count(features, max_k):
        """使用轮廓系数 + CH 指数自适应选择簇数。"""
        if len(features) < 3:
            return max(1, min(2, len(features)))

        upper_k = max(2, min(max_k, len(features) - 1))
        best_k = 2
        best_score = -1e18
        for k in range(2, upper_k + 1):
            try:
                kmeans_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels_tmp = kmeans_tmp.fit_predict(features)
                if len(np.unique(labels_tmp)) < 2:
                    continue
                sil = silhouette_score(features, labels_tmp)
                ch = calinski_harabasz_score(features, labels_tmp)
                score = sil + ch / 1000.0
                if score > best_score:
                    best_score = score
                    best_k = k
            except Exception:
                continue
        return best_k
    
    # 提取客户端特征用于分组（基于数据分布或模型梯度）
    client_features = []
    for client_id in all_client_ids:
        client = clients[client_id]
        client.set_parameters(global_model)
        
        # 提取特征：使用模型在客户端数据上的输出分布
        client.model.eval()
        trainloader = client.load_train_data()
        features = []
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(trainloader):
                if batch_idx >= 5:  # 只使用前5个batch来加速
                    break
                if type(x) == type([]):
                    x[0] = x[0].to(client.device)
                else:
                    x = x.to(client.device)
                output = client.model(x)
                # 使用输出的均值作为特征
                features.append(output.mean(dim=0).cpu().numpy())
        
        if features:
            client_feature = np.mean(features, axis=0)
        else:
            # 如果无法提取特征，使用随机特征
            client_feature = np.random.randn(10)  # 假设10个类别
        client_features.append(client_feature)
    
    client_features = np.array(client_features)
    
    # 使用K-means进行分组
    max_subsets = getattr(args, 'fedcsa_max_subsets', 5)
    min_subsets = getattr(args, 'fedcsa_min_subsets', 2)
    if len(other_client_ids) >= 2:
        other_features = client_features[other_client_ids]
        adaptive_k = _determine_optimal_subset_count(other_features, max_subsets)
        num_subsets = max(min_subsets, min(adaptive_k, len(other_client_ids)))
    else:
        num_subsets = 1

    print(f"步骤1: 自适应分为 {num_subsets} 个子集...")

    if len(other_client_ids) >= num_subsets:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=num_subsets, random_state=42, n_init=10)
        # 只对其他客户端进行聚类
        other_features = client_features[other_client_ids]
        cluster_labels = kmeans.fit_predict(other_features)
        
        # 创建子集映射
        client_subsets = {}
        for idx, client_id in enumerate(other_client_ids):
            subset_id = cluster_labels[idx]
            if subset_id not in client_subsets:
                client_subsets[subset_id] = []
            client_subsets[subset_id].append(client_id)
        
        # 目标客户端单独成组（需要被排除的组）
        target_subset_id = num_subsets
        client_subsets[target_subset_id] = [target_client_id]
        
        print(f"客户端分组完成: {len(client_subsets)} 个子集")
        for subset_id, client_ids in client_subsets.items():
            print(f"  子集 {subset_id}: 客户端 {client_ids}")
    else:
        # 如果客户端太少，每个客户端一个子集
        client_subsets = {i: [other_client_ids[i]] for i in range(len(other_client_ids))}
        client_subsets[len(other_client_ids)] = [target_client_id]
        print(f"客户端数量较少，每个客户端一个子集")
    
    # ========== 步骤2: 识别并排除目标客户端所在的子集 ==========
    target_subset_id = None
    for subset_id, client_ids in client_subsets.items():
        if target_client_id in client_ids:
            target_subset_id = subset_id
            break
    
    if target_subset_id is not None:
        print(f"步骤2: 排除目标客户端所在的子集 {target_subset_id}")
        # 移除目标子集
        valid_subsets = {k: v for k, v in client_subsets.items() if k != target_subset_id}
    else:
        print("警告：未找到目标客户端所在的子集，排除所有包含目标客户端的子集")
        valid_subsets = {k: v for k, v in client_subsets.items() if target_client_id not in v}
    
    if len(valid_subsets) == 0:
        print("警告：没有有效的子集，返回原始模型")
        return global_model
    
    # ========== 步骤3: 在每个有效子集上独立训练模型 ==========
    updated_model = copy.deepcopy(global_model)
    # 修复：减少训练轮数（从5减少到3），避免过度训练导致其他客户端性能下降
    forget_epochs = getattr(args, 'fedcsa_forget_epochs', 3)
    local_epochs = getattr(args, 'local_epochs', 1)
    # 修复：减少本地训练轮数（从3减少到2），使用更温和的训练策略
    enhanced_local_epochs = max(local_epochs, 2)  # 至少2个epoch（从3减少到2）
    
    # 如果使用SCAFFOLD，需要处理控制变量
    if is_scaffold:
        print("检测到SCAFFOLD算法，将处理控制变量")
        # 初始化全局控制变量（如果可以从server获取，这里简化处理）
        global_c = None
        # 尝试从客户端获取global_c的引用（如果可用）
        if hasattr(clients[0], 'global_c') and clients[0].global_c is not None:
            global_c = clients[0].global_c
        else:
            # 创建新的global_c
            global_c = []
            for param in updated_model.parameters():
                global_c.append(torch.zeros_like(param))
    
    for epoch in range(forget_epochs):
        print(f"FedCSA遗忘训练轮次: {epoch + 1}/{forget_epochs}")
        
        subset_models = []  # 每个子集的模型
        subset_weights = []  # 每个子集的权重
        
        # 对每个有效子集独立训练
        for subset_id, client_ids in valid_subsets.items():
            print(f"  训练子集 {subset_id} (包含客户端 {client_ids})...")
            
            # 子集内每个客户端独立训练
            subset_local_models = []
            subset_local_weights = []
            
            for client_id in client_ids:
                client = clients[client_id]
                client.set_parameters(updated_model)  # 从全局模型开始训练
                
                # 本地训练
                client.model.train()
                trainloader = client.load_train_data()
                
                if is_scaffold and hasattr(client, 'optimizer') and hasattr(client, 'client_c'):
                    # 使用SCAFFOLD优化器
                    optimizer = client.optimizer
                    # 设置global_c和client_c
                    client.global_c = global_c
                    if client.client_c is None:
                        client.client_c = []
                        for param in client.model.parameters():
                            client.client_c.append(torch.zeros_like(param))
                else:
                    # 使用标准SGD优化器
                    optimizer = torch.optim.SGD(client.model.parameters(), lr=client.learning_rate)
                
                for local_epoch in range(enhanced_local_epochs):
                    for x, y in trainloader:
                        if type(x) == type([]):
                            x[0] = x[0].to(client.device)
                        else:
                            x = x.to(client.device)
                        y = y.to(client.device)
                        
                        optimizer.zero_grad()
                        output = client.model(x)
                        loss = client.loss(output, y)
                        loss.backward()
                        
                        if is_scaffold and hasattr(optimizer, 'step') and hasattr(client, 'global_c') and hasattr(client, 'client_c'):
                            # SCAFFOLD优化器的step需要global_c和client_c
                            optimizer.step(client.global_c, client.client_c)
                        else:
                            optimizer.step()
                
                # 保存客户端训练后的模型
                subset_local_models.append(copy.deepcopy(client.model.state_dict()))
                subset_local_weights.append(client.train_samples)
            
            # 聚合子集内的客户端模型
            subset_total_weight = sum(subset_local_weights)
            subset_model_state = copy.deepcopy(updated_model.state_dict())
            
            for key in subset_model_state.keys():
                param = subset_model_state[key]
                if param.dtype in [torch.float32, torch.float64, torch.float16]:
                    param.zero_()
                    for local_model, weight in zip(subset_local_models, subset_local_weights):
                        if key in local_model:
                            param += local_model[key] * (weight / subset_total_weight)
            
            subset_models.append(subset_model_state)
            # 子集权重 = 子集内所有客户端的样本数之和
            subset_weights.append(subset_total_weight)
        
        # ========== 步骤4: 聚合所有有效子集的模型 ==========
        print(f"  聚合 {len(subset_models)} 个子集的模型...")
        total_weight = sum(subset_weights)
        
        for key in updated_model.state_dict().keys():
            param = updated_model.state_dict()[key]
            if param.dtype in [torch.float32, torch.float64, torch.float16]:
                param.zero_()
                for subset_model, weight in zip(subset_models, subset_weights):
                    if key in subset_model:
                        param += subset_model[key] * (weight / total_weight)
        
        # 如果使用SCAFFOLD，更新global_c
        if is_scaffold and global_c is not None:
            # 简化处理：重置global_c（因为这是遗忘阶段，不是正常训练）
            for param in global_c:
                param.zero_()
    
    print("FedCSA遗忘完成")
    return updated_model


class UnLearningCELoss:
    """UnLearning交叉熵损失函数，用于FedOSD遗忘方法"""
    def __init__(self, ignore_index=-100, reduction='mean'):
        self.ignore_index = ignore_index
        self.reduction = reduction

    def __call__(self, pred, target):
        class_num = int(pred.shape[1])
        
        ignore_indices = torch.where(target == self.ignore_index)[0]
        if len(ignore_indices) > 0:
            target[ignore_indices] = 0  
            target_enc = F.one_hot(target, class_num)
            target_enc[ignore_indices, 0] = 0  
        else:
            target_enc = F.one_hot(target, class_num)
        
        pred = F.softmax(pred, dim=-1)
        if self.reduction == 'none':
            loss = -torch.sum(torch.log(1.0 - pred / 2) * target_enc, dim=1)
        elif self.reduction == 'mean':
            loss = -torch.mean(torch.sum(torch.log(1.0 - pred / 2) * target_enc, dim=1))
        elif self.reduction == 'sum':
            loss = -torch.sum(torch.sum(torch.log(1.0 - pred / 2) * target_enc, dim=1))
        else:
            loss = None
        return loss


def _set_optimizer_learning_rate(optimizer, lr):
    """在FedOSD对比实现中临时覆盖客户端本地学习率。"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def _flatten_model_params(model):
    return torch.cat([param.data.flatten() for param in model.parameters()])


def _apply_vector_update(model, update_vec, lr):
    param_idx = 0
    with torch.no_grad():
        for param in model.parameters():
            param_size = param.numel()
            param_grad = update_vec[param_idx:param_idx + param_size].reshape(param.shape)
            param.data -= lr * param_grad
            param_idx += param_size


def _aggregate_non_param_buffers(target_model, local_models, weights=None):
    """聚合非参数buffer，避免BN统计完全停留在遗忘前。"""
    if not local_models:
        return

    if weights is None:
        weights = [1.0 / len(local_models)] * len(local_models)

    param_keys = {name for name, _ in target_model.named_parameters()}
    global_state = target_model.state_dict()
    local_states = [model.state_dict() for model in local_models]

    for key, tensor in global_state.items():
        if key in param_keys:
            continue
        if torch.is_floating_point(tensor):
            agg = torch.zeros_like(tensor)
            for weight, state in zip(weights, local_states):
                agg += state[key] * weight
            global_state[key] = agg
        else:
            global_state[key] = local_states[0][key].clone()

    target_model.load_state_dict(global_state, strict=False)


def _sample_fedosd_online_clients(clients, args, exclude_client_ids=None, force_include_ids=None):
    """按当前实验配置采样FedOSD每轮在线客户端。"""
    exclude_client_ids = set(exclude_client_ids or [])
    force_include_ids = [cid for cid in (force_include_ids or []) if cid not in exclude_client_ids]

    available_clients = [client for client in clients if client.id not in exclude_client_ids]
    if not available_clients:
        return []

    min_join_clients = max(1, int(len(clients) * args.join_ratio))
    desired_clients = min_join_clients
    if getattr(args, "random_join_ratio", False):
        desired_clients = np.random.choice(range(min_join_clients, len(clients) + 1), 1, replace=False)[0]

    max_online_clients = int(getattr(args, "fedosd_max_online_clients", 0) or 0)
    if max_online_clients > 0:
        desired_clients = min(desired_clients, max_online_clients)

    desired_clients = min(max(desired_clients, len(force_include_ids)), len(available_clients))
    selected = list(np.random.choice(available_clients, desired_clients, replace=False))

    selected_ids = {client.id for client in selected}
    if len(force_include_ids) > 0:
        selected_map = {client.id: client for client in selected}
        available_map = {client.id: client for client in available_clients}
        for forced_id in force_include_ids:
            if forced_id in selected_ids or forced_id not in available_map:
                continue
            non_forced_clients = [client for client in selected if client.id not in force_include_ids]
            if non_forced_clients:
                drop_client = non_forced_clients[0]
                selected.remove(drop_client)
                selected_ids.remove(drop_client.id)
            if len(selected) < desired_clients:
                selected.append(available_map[forced_id])
                selected_ids.add(forced_id)

    return selected


def _sync_client_with_global_model(client, global_model, global_c=None):
    if global_c is not None and hasattr(client, 'client_c'):
        client.set_parameters(global_model, global_c=global_c)
    else:
        client.set_parameters(global_model)


def _update_scaffold_control_variates(global_c, online_clients, total_client_count):
    """在SCAFFOLD设置下同步更新全局控制变量，避免比较方法与训练机制彻底脱节。"""
    if global_c is None or len(online_clients) == 0:
        return global_c

    total_client_count = max(1, total_client_count)
    updated_global_c = [param.clone() for param in global_c]
    for client in online_clients:
        _, delta_c = client.delta_yc()
        for server_param, client_param in zip(updated_global_c, delta_c):
            server_param.data += client_param.data.clone() / total_client_count
    return updated_global_c


def _compute_fedosd_direction(gu, gr_stack):
    """求解与保留客户端梯度正交、且最接近目标遗忘梯度的方向。"""
    if gr_stack is None or gr_stack.numel() == 0:
        return gu

    A = gr_stack
    c = gu

    try:
        AAT = A @ A.T
        try:
            if hasattr(torch.linalg, 'svd'):
                U, s, Vh = torch.linalg.svd(AAT)
                V = Vh.T
            else:
                U, s, V = torch.svd(AAT)
        except (AttributeError, RuntimeError):
            U, s, V = torch.svd(AAT)

        primary_indices = torch.where(s >= 1e-6)[0]
        if len(primary_indices) == 0:
            return c

        s_inv = torch.zeros_like(s)
        s_inv[primary_indices] = 1.0 / s[primary_indices]
        AAT_inv = V @ torch.diag(s_inv) @ U.T
        Ac = A @ c.reshape(-1, 1)
        AAT_inv_Ac = AAT_inv @ Ac
        d = c - (A.T @ AAT_inv_Ac).reshape(-1)

        c_norm = torch.norm(c)
        d_norm = torch.norm(d)
        if d_norm > 1e-9:
            d = d / d_norm * c_norm
        else:
            d = c
        return d
    except Exception:
        return c


def _run_fedosd_local_round(global_model, online_clients, round_lr, args, target_client_id=None,
                            unlearning_loss_fn=None):
    """执行一轮更接近官方FedOSD的本地训练，并从本地模型差分构造上传梯度。"""
    global_param_vec = _flatten_model_params(global_model).detach()
    local_models = []
    gu_locals = []
    gr_locals = []
    buffer_weights = []
    local_epochs = max(1, getattr(args, 'local_epochs', 1))

    for client in online_clients:
        local_model = copy.deepcopy(global_model).to(client.device)
        local_model.train()
        optimizer = torch.optim.SGD(local_model.parameters(), lr=round_lr)
        criterion = unlearning_loss_fn if client.id == target_client_id and unlearning_loss_fn is not None else client.loss
        trainloader = client.load_train_data()

        for _ in range(local_epochs):
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(client.device)
                    batch_x = x
                else:
                    batch_x = x.to(client.device)
                batch_y = y.to(client.device)

                optimizer.zero_grad()
                output = local_model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

        local_models.append(local_model)
        buffer_weights.append(max(1, getattr(client, 'train_samples', 1)))

        local_param_vec = _flatten_model_params(local_model).detach().to(global_param_vec.device)
        grad_vec = (global_param_vec - local_param_vec) / round_lr
        if client.id == target_client_id:
            gu_locals.append(grad_vec)
        else:
            gr_locals.append(grad_vec)

    if len(buffer_weights) > 0:
        total_weight = float(sum(buffer_weights))
        buffer_weights = [weight / total_weight for weight in buffer_weights]

    gu_stack = torch.stack(gu_locals) if len(gu_locals) > 0 else None
    gr_stack = torch.stack(gr_locals) if len(gr_locals) > 0 else None
    return gu_stack, gr_stack, local_models, buffer_weights


def _compute_client_gradient(client, model, use_unlearning_loss=False):
    """
    计算客户端在给定模型上的梯度
    辅助函数，用于 FedOSD 方法
    
    Args:
        client: 客户端对象
        model: 模型
        use_unlearning_loss: 是否使用UnLearning损失函数（用于目标客户端）
    """
    model_copy = copy.deepcopy(model)
    model_copy.train()
    trainloader = client.load_train_data()
    optimizer = torch.optim.SGD(model_copy.parameters(), lr=client.learning_rate)
    
    # 选择损失函数
    if use_unlearning_loss:
        unlearning_loss_fn = UnLearningCELoss()
    
    # 计算所有batch的平均梯度
    accumulated_gradients = None
    batch_count = 0
    
    for x, y in trainloader:
        if type(x) == type([]):
            x[0] = x[0].to(client.device)
        else:
            x = x.to(client.device)
        y = y.to(client.device)
        
        optimizer.zero_grad()
        output = model_copy(x)
        
        # 使用UnLearning损失或正常损失
        if use_unlearning_loss:
            loss = unlearning_loss_fn(output, y)
        else:
            loss = client.loss(output, y)
        
        loss.backward()
        
        # 累积梯度
        if accumulated_gradients is None:
            accumulated_gradients = []
            for param in model_copy.parameters():
                if param.grad is not None:
                    accumulated_gradients.append(param.grad.data.flatten().clone())
                else:
                    accumulated_gradients.append(torch.zeros_like(param.data.flatten()))
        else:
            for i, param in enumerate(model_copy.parameters()):
                if param.grad is not None:
                    accumulated_gradients[i] += param.grad.data.flatten().clone()
        
        batch_count += 1
        
        # 限制批次数以提高效率
        # 修复：增加batch数量以提高梯度计算精度（从10增加到20）
        if batch_count >= 20:  # 最多使用20个batch
            break
    
    # 平均梯度
    if accumulated_gradients and batch_count > 0:
        for i in range(len(accumulated_gradients)):
            accumulated_gradients[i] /= batch_count
    
    return torch.cat(accumulated_gradients).to(client.device) if accumulated_gradients else torch.zeros(1).to(client.device)


def _forget_client_with_fedosd_legacy(global_model, clients, args, target_client_id=0):
    """
    FedOSD方法：基于梯度正交投影的遗忘方法
    核心思路：计算目标客户端的梯度（使用UnLearning损失），然后找到与这个梯度正交的方向进行更新
    
    数学原理：
    - gu: 目标客户端的梯度（使用UnLearning损失，指向遗忘方向）
    - gr: 其他客户端的梯度
    - d = gu - A^T * (A * A^T)^(-1) * A * gu，这是gu在gr空间的正交补空间上的投影
    - d包含了gu中与所有gr正交的部分，沿着d的方向更新可以移除目标客户端的影响
    """
    print(f"\n============= FedOSD 遗忘方法 =============")
    print(f"目标遗忘客户端: {target_client_id}")
    
    target_client = clients[target_client_id]
    other_clients = [c for i, c in enumerate(clients) if i != target_client_id]
    
    # 设置模型参数
    for client in clients:
        client.set_parameters(global_model)
    
    # 计算目标客户端和其他客户端的梯度
    print("计算梯度...")
    # 目标客户端使用UnLearning损失函数，梯度方向指向遗忘方向
    gu = _compute_client_gradient(target_client, global_model, use_unlearning_loss=True)  # 目标客户端梯度
    
    gr_list = []
    for client in other_clients:
        gr = _compute_client_gradient(client, global_model, use_unlearning_loss=False)
        gr_list.append(gr)
    
    if len(gr_list) == 0:
        print("警告：没有其他客户端，返回原始模型")
        return global_model
    
    gr_stack = torch.stack(gr_list)  # [num_other_clients, param_dim]
    
    # FedOSD核心计算：找到与gu正交的方向
    # d = gu - A^T * (A * A^T)^(-1) * A * gu
    # 其中 A = gr_stack (其他客户端的梯度)
    # d是gu在gr空间的正交补空间上的投影，包含了gu中与所有gr正交的部分
    print("计算正交投影...")
    
    A = gr_stack  # [num_other_clients, param_dim]
    c = gu  # [param_dim]
    
    try:
        # 计算伪逆
        AAT = A @ A.T  # [num_other_clients, num_other_clients]
        
        # SVD计算伪逆
        # 使用 torch.linalg.svd (新版本) 或 torch.svd (旧版本)
        try:
            if hasattr(torch.linalg, 'svd'):
                U, s, Vh = torch.linalg.svd(AAT)
                V = Vh.T  # Vh 是 V 的共轭转置，需要转置回来
            else:
                U, s, V = torch.svd(AAT)
        except (AttributeError, RuntimeError):
            # 兼容旧版本
            U, s, V = torch.svd(AAT)
            
        primary_indices = torch.where(s >= 1e-6)[0]
        if len(primary_indices) == 0:
            print("警告：矩阵奇异，使用简化方法")
            d = c
        else:
            s_inv = torch.zeros_like(s)
            s_inv[primary_indices] = 1.0 / s[primary_indices]
            AAT_inv = V @ torch.diag(s_inv) @ U.T
            
            # 计算正交投影方向
            Ac = A @ c.reshape(-1, 1)
            AAT_inv_Ac = AAT_inv @ Ac
            d = c - (A.T @ AAT_inv_Ac).reshape(-1)
        
        # 归一化并保持原始梯度范数
        gu_norm = torch.norm(gu)
        d_norm = torch.norm(d)
        if d_norm > 1e-9:
            d = d / d_norm * gu_norm
        else:
            print("警告：投影方向范数过小，使用原始梯度")
            d = c
        
    except Exception as e:
        print(f"计算正交投影时出错: {e}，使用简化方法")
        d = gu
    
    # 使用计算出的方向更新模型
    # 注意：由于gu使用了UnLearning损失（指向遗忘方向），沿着d的方向更新（param.data -= lr * d）
    # 会移除目标客户端的影响
    updated_model = copy.deepcopy(global_model)
    # 修复：大幅增大基础学习率（从0.01增加到0.1），确保有效遗忘
    base_lr = getattr(args, 'fedosd_lr', args.local_learning_rate if hasattr(args, 'local_learning_rate') else 0.1)
    
    # FedOSD需要多次迭代来增强遗忘效果
    num_iterations = getattr(args, 'fedosd_iterations', 5)  # 默认5次迭代（从3增加到5）
    
    print(f"应用更新（{num_iterations}次迭代）...")
    
    for iteration in range(num_iterations):
        # 每次迭代重新计算梯度（因为模型参数已经改变）
        if iteration > 0:
            # 重新计算目标客户端的梯度
            gu = _compute_client_gradient(target_client, updated_model, use_unlearning_loss=True)
            
            # 重新计算其他客户端的梯度
            gr_list = []
            for client in other_clients:
                gr = _compute_client_gradient(client, updated_model, use_unlearning_loss=False)
                gr_list.append(gr)
            
            gr_stack = torch.stack(gr_list)
            A = gr_stack
            c = gu
            
            # 重新计算正交投影方向
            try:
                AAT = A @ A.T
                try:
                    if hasattr(torch.linalg, 'svd'):
                        U, s, Vh = torch.linalg.svd(AAT)
                        V = Vh.T
                    else:
                        U, s, V = torch.svd(AAT)
                except (AttributeError, RuntimeError):
                    U, s, V = torch.svd(AAT)
                
                primary_indices = torch.where(s >= 1e-6)[0]
                if len(primary_indices) == 0:
                    d = c
                else:
                    s_inv = torch.zeros_like(s)
                    s_inv[primary_indices] = 1.0 / s[primary_indices]
                    AAT_inv = V @ torch.diag(s_inv) @ U.T
                    Ac = A @ c.reshape(-1, 1)
                    AAT_inv_Ac = AAT_inv @ Ac
                    d = c - (A.T @ AAT_inv_Ac).reshape(-1)
                
                gu_norm = torch.norm(gu)
                d_norm = torch.norm(d)
                if d_norm > 1e-9:
                    d = d / d_norm * gu_norm
                else:
                    d = c
            except Exception as e:
                d = gu
        
        # 修复：不使用递减学习率，保持恒定学习率以增强遗忘效果
        # 原代码：lr = base_lr * (0.8 ** iteration)  # 每次迭代减小学习率
        lr = base_lr  # 保持恒定学习率
        
        param_idx = 0
        with torch.no_grad():
            for param in updated_model.parameters():
                param_size = param.numel()
                param_grad = d[param_idx:param_idx + param_size].reshape(param.shape)
                # 沿着d的方向更新（减去d），因为d已经指向遗忘方向
                param.data -= lr * param_grad
                param_idx += param_size
        
        if iteration < num_iterations - 1:
            print(f"  迭代 {iteration + 1}/{num_iterations} 完成")

    # FedOSD后训练阶段：梯度投影，避免模型更新回退到原始状态
    enable_post_projection = getattr(args, 'fedosd_enable_post_projection', True)
    if enable_post_projection and len(other_clients) > 0:
        print("执行FedOSD后训练梯度投影阶段...")

        # 计算与原始模型的参数偏差向量
        with torch.no_grad():
            origin_vec = torch.cat([p.data.flatten() for p in global_model.parameters()]).to(updated_model.parameters().__iter__().__next__().device)
            current_vec = torch.cat([p.data.flatten() for p in updated_model.parameters()]).to(origin_vec.device)
            drift_vec = current_vec - origin_vec
            drift_norm_sq = torch.dot(drift_vec, drift_vec).item()

        if drift_norm_sq > 1e-12:
            post_epochs = getattr(args, 'fedosd_post_epochs', 1)
            post_lr = getattr(args, 'fedosd_post_lr', base_lr * 0.2)

            for ep in range(post_epochs):
                for client in other_clients:
                    gr = _compute_client_gradient(client, updated_model, use_unlearning_loss=False).to(drift_vec.device)
                    dot_val = torch.dot(gr, drift_vec).item()

                    # 若同向，投影到偏差向量法平面以抑制回退
                    if dot_val > 0:
                        gr = gr - (dot_val / (drift_norm_sq + 1e-12)) * drift_vec

                    param_idx = 0
                    with torch.no_grad():
                        for param in updated_model.parameters():
                            psize = param.numel()
                            g_slice = gr[param_idx:param_idx + psize].reshape(param.shape)
                            param.data -= post_lr * g_slice
                            param_idx += psize
                print(f"  后训练投影轮次 {ep + 1}/{post_epochs} 完成")
        else:
            print("跳过后训练投影：模型偏差向量过小")
    
    print("FedOSD遗忘完成")
    return updated_model


def forget_client_with_fedosd(global_model, clients, args, target_client_id=0, server=None, round_evaluator=None):
    """
    更接近官方FedOSD-main的可对比实现：
    1. 每轮先做本地训练，再由本地模型差分构造 g_locals
    2. 遗忘阶段对目标客户端使用 UCE，其余客户端正常训练
    3. 服务端按正交最速下降方向更新
    4. 进入独立后训练阶段，对剩余客户端梯度做回退抑制投影
    """
    if getattr(args, 'fedosd_use_legacy', False):
        print("FedOSD: 使用legacy直接梯度版本")
        return _forget_client_with_fedosd_legacy(global_model, clients, args, target_client_id=target_client_id)

    print(f"\n============= FedOSD 遗忘方法（official-like） =============")
    print(f"目标遗忘客户端: {target_client_id}")

    other_clients = [client for client in clients if client.id != target_client_id]
    if len(other_clients) == 0:
        print("警告：没有其他客户端，返回原始模型")
        return global_model

    updated_model = copy.deepcopy(global_model)
    init_model_params = _flatten_model_params(updated_model).detach()

    original_losses = {client.id: client.loss for client in clients}
    uce_loss = UnLearningCELoss()
    base_lr = getattr(args, 'fedosd_lr', args.local_learning_rate if hasattr(args, 'local_learning_rate') else 0.01)
    unlearn_rounds = max(1, getattr(args, 'fedosd_unlearn_rounds', getattr(args, 'fedosd_iterations', 5)))
    recovery_rounds = max(0, getattr(args, 'fedosd_recovery_rounds', getattr(args, 'recovery_rounds', 0)))
    recovery_lr = getattr(args, 'fedosd_recovery_lr', max(base_lr * 0.1, 1e-6))
    force_target_online = getattr(args, 'fedosd_force_target_online', True)

    print(f"FedOSD遗忘轮数: {unlearn_rounds}, 本地/服务端学习率: {base_lr}")
    print(f"FedOSD后训练轮数: {recovery_rounds}, 后训练学习率: {recovery_lr}")

    unlearning_round_online_stats = []
    recovery_round_online_stats = []

    for round_idx in range(unlearn_rounds):
        online_clients = _sample_fedosd_online_clients(
            clients,
            args,
            force_include_ids=[target_client_id] if force_target_online else None,
        )
        online_ids = [client.id for client in online_clients]
        print(f"FedOSD遗忘轮 {round_idx + 1}/{unlearn_rounds}，在线客户端: {online_ids}")
        unlearning_round_online_stats.append({
            "round": int(round_idx + 1),
            "online_client_ids": [int(client_id) for client_id in online_ids],
            "num_online_clients": int(len(online_ids)),
        })

        gu_stack, gr_stack, local_models, buffer_weights = _run_fedosd_local_round(
            updated_model,
            online_clients,
            base_lr,
            args=args,
            target_client_id=target_client_id,
            unlearning_loss_fn=uce_loss,
        )

        if gu_stack is None or gu_stack.numel() == 0:
            print("  警告：本轮未采到目标客户端梯度，跳过")
            continue
        if gr_stack is None or gr_stack.numel() == 0:
            print("  警告：本轮无保留客户端梯度，直接使用目标客户端UCE方向")
            d = torch.mean(gu_stack, dim=0)
        else:
            gu = torch.mean(gu_stack, dim=0)
            d = _compute_fedosd_direction(gu, gr_stack)

        _apply_vector_update(updated_model, d, base_lr)
        _aggregate_non_param_buffers(updated_model, local_models, buffer_weights)

    forgotten_model = copy.deepcopy(updated_model)
    recovery_results = []
    if recovery_rounds > 0:
        print("执行FedOSD官方风格后训练阶段...")
        for round_idx in range(recovery_rounds):
            online_clients = _sample_fedosd_online_clients(
                other_clients,
                args,
                exclude_client_ids=[target_client_id],
            )
            if len(online_clients) == 0:
                print("  警告：后训练阶段没有可用客户端，提前结束")
                break

            online_ids = [client.id for client in online_clients]
            print(f"FedOSD后训练轮 {round_idx + 1}/{recovery_rounds}，在线客户端: {online_ids}")
            recovery_round_online_stats.append({
                "round": int(round_idx + 1),
                "online_client_ids": [int(client_id) for client_id in online_ids],
                "num_online_clients": int(len(online_ids)),
            })

            _, gr_stack, local_models, buffer_weights = _run_fedosd_local_round(
                updated_model,
                online_clients,
                recovery_lr,
                args=args,
                target_client_id=None,
                unlearning_loss_fn=None,
            )

            if gr_stack is None or gr_stack.numel() == 0:
                print("  警告：后训练轮无有效梯度，跳过")
                continue

            current_model_params = _flatten_model_params(updated_model).detach()
            ga = current_model_params - init_model_params
            ga_norm = torch.norm(ga)
            if ga_norm <= 1e-12:
                print("  偏差向量过小，跳过本轮投影")
                continue

            projected_grads = []
            for grad in gr_stack:
                grad_norm = torch.norm(grad)
                dot_val = torch.dot(grad, ga)
                projected_grad = grad
                if dot_val > 0:
                    projected_grad = grad - dot_val / (ga_norm ** 2 + 1e-12) * ga
                projected_norm = torch.norm(projected_grad)
                if projected_norm > 1e-12:
                    projected_grad = projected_grad / projected_norm * grad_norm
                else:
                    projected_grad = grad
                projected_grads.append(projected_grad)

            d = torch.mean(torch.stack(projected_grads), dim=0)
            _apply_vector_update(updated_model, d, recovery_lr)
            _aggregate_non_param_buffers(updated_model, local_models, buffer_weights)

            round_result = {
                "round": int(round_idx + 1),
                "test_acc": None,
                "test_auc": None,
                "train_loss": None,
                "online_client_ids": [int(client_id) for client_id in online_ids],
                "num_online_clients": int(len(online_ids)),
            }
            if round_evaluator is not None:
                round_result.update(round_evaluator(updated_model, round_idx + 1) or {})
            recovery_results.append(round_result)

    for client in clients:
        if client.id in original_losses:
            client.loss = original_losses[client.id]

    if server is not None:
        server.global_model = copy.deepcopy(updated_model)

    print("FedOSD遗忘完成")
    return {
        "model": forgotten_model,
        "post_recovery_model": copy.deepcopy(updated_model) if recovery_results else None,
        "recovery_results": recovery_results,
        "recovery_rounds": len(recovery_results),
        "recovery_stage": "internal_fedosd_post_training" if recovery_results else "skipped_for_fedosd",
        "metadata": {
            "fedosd_unlearning_round_online_stats": unlearning_round_online_stats,
            "fedosd_recovery_round_online_stats": recovery_round_online_stats,
        },
    }


def forget_client_with_fedu(global_model, clients, args, target_client_id=0):
    """
    FedU方法：基于Hessian矩阵的遗忘方法
    核心思路：使用Hessian矩阵的逆来精确计算需要移除的更新量
    数学原理：θ_new = θ_old - H^(-1) * ∇L(θ_old)
    其中H是Hessian矩阵，∇L是目标客户端在模型上的梯度
    
    关键修复：
    1. 增大更新量（使用更大的alpha或学习率）
    2. 多次迭代增强遗忘效果
    3. 正确计算Hessian对角
    """
    # 使用独立的 FedU-main 风格实现，避免旧版启发式实现与 FedU 核心思路偏离。
    from flcore.fedu_main import fedu_main_forget_client
    return fedu_main_forget_client(global_model, clients, args, target_client_id=target_client_id)

    print(f"\n============= FedU 遗忘方法 =============")
    print(f"目标遗忘客户端: {target_client_id}")
    
    target_client = clients[target_client_id]
    other_clients = [c for i, c in enumerate(clients) if i != target_client_id]
    
    # 设置模型参数
    target_client.set_parameters(global_model)
    
    # 使用Hessian近似进行遗忘
    updated_model = copy.deepcopy(global_model)
    updated_model.train()
    
    trainloader = target_client.load_train_data()
    
    # 参数设置
    forget_epochs = getattr(args, 'fedu_forget_epochs', 1)
    base_lr = getattr(args, 'fedu_lr', 0.001)
    eps = getattr(args, 'fedu_eps', 1e-2)
    alpha = getattr(args, 'fedu_alpha', 0.02)
    
    # 增加迭代次数来增强遗忘效果
    num_iterations = getattr(args, 'fedu_iterations', 5)  # 默认5次迭代（从3增加到5）
    
    print("计算Hessian近似...")

    # FedU双任务近似配置（IAF + Utility Preservation）
    use_dual_objective = getattr(args, 'fedu_use_dual_objective', True)
    utility_client_k = getattr(args, 'fedu_utility_client_k', 3)
    utility_scale = getattr(args, 'fedu_utility_scale', 1.0)
    
    for iteration in range(num_iterations):
        # 累积所有batch的梯度和Hessian
        all_gradients = []
        all_hessian_diags = []
        batch_count = 0
        max_batches = 10  # 最多使用10个batch来计算平均梯度和Hessian
        
        for epoch in range(forget_epochs):
            for batch_idx, (x, y) in enumerate(trainloader):
                if batch_idx >= max_batches:
                    break
                    
                if type(x) == type([]):
                    x[0] = x[0].to(target_client.device)
                else:
                    x = x.to(target_client.device)
                y = y.to(target_client.device)
                
                # 计算梯度
                updated_model.zero_grad()
                output = updated_model(x)
                loss = target_client.loss(output, y)
                loss.backward(create_graph=True)
                
                # 收集梯度和计算Hessian对角近似
                batch_gradients = []
                batch_hessian_diags = []
                
                for param in updated_model.parameters():
                    if param.grad is not None:
                        # 保存梯度
                        batch_gradients.append(param.grad.data.clone())
                        
                        # 使用Hutchinson方法近似Hessian对角（多次采样取平均）
                        hessian_samples = []
                        num_samples = 10  # 使用10次采样来平均（从3增加到10，提高Hessian估计精度）
                        for _ in range(num_samples):
                            z = torch.randint(0, 2, param.shape, device=param.device, dtype=param.dtype) * 2.0 - 1.0
                            try:
                                grad_grad = torch.autograd.grad(
                                    outputs=(param.grad * z).sum(),
                                    inputs=param,
                                    create_graph=False,
                                    retain_graph=True
                                )[0]
                                hessian_samples.append((grad_grad * z).sum().item())
                            except:
                                # 如果计算失败，使用梯度范数作为近似
                                hessian_samples.append(param.grad.norm().item() * 0.1)
                        
                        h_diag = np.mean(hessian_samples) if hessian_samples else 0.0
                        batch_hessian_diags.append(h_diag)
                    else:
                        batch_gradients.append(None)
                        batch_hessian_diags.append(0.0)
                
                all_gradients.append(batch_gradients)
                all_hessian_diags.append(batch_hessian_diags)
                batch_count += 1
                
                # 清理计算图
                updated_model.zero_grad()
        
        if batch_count == 0:
            print("警告：没有可用的batch，返回原始模型")
            return global_model
        
        # 计算平均梯度和Hessian对角
        avg_gradients = []
        avg_hessian_diags = []
        
        for param_idx in range(len(all_gradients[0])):
            if all_gradients[0][param_idx] is not None:
                # 平均梯度
                avg_grad = torch.zeros_like(all_gradients[0][param_idx])
                for batch_grads in all_gradients:
                    if batch_grads[param_idx] is not None:
                        avg_grad += batch_grads[param_idx]
                avg_grad /= batch_count
                avg_gradients.append(avg_grad)
                
                # 平均Hessian对角
                avg_hessian = np.mean([batch_hessian[param_idx] for batch_hessian in all_hessian_diags])
                avg_hessian_diags.append(avg_hessian)
            else:
                avg_gradients.append(None)
                avg_hessian_diags.append(0.0)
        
        # 额外计算效用保留梯度（从其余客户端近似）
        utility_grad_vector = None
        if use_dual_objective and len(other_clients) > 0:
            selected_clients = other_clients[:max(1, min(utility_client_k, len(other_clients)))]
            utility_grads = []
            for c in selected_clients:
                g = _compute_client_gradient(c, updated_model, use_unlearning_loss=False)
                utility_grads.append(g)
            if len(utility_grads) > 0:
                utility_grad_vector = torch.mean(torch.stack(utility_grads), dim=0)

        # 应用遗忘更新：θ_new = θ_old - H^(-1) * ∇L
        # 使用对角Hessian近似：θ_new = θ_old - (∇L / (H_diag + eps)) * lr
        # 使用递增的学习率来增强遗忘效果
        lr = base_lr * (1.2 ** iteration)  # 每次迭代增大学习率（从1.5改为1.2，更温和）

        param_idx = 0
        vec_idx = 0
        total_update_norm = 0.0
        forget_update_norm = 0.0
        utility_update_norm = 0.0
        with torch.no_grad():
            for param in updated_model.parameters():
                if avg_gradients[param_idx] is not None:
                    grad = avg_gradients[param_idx]
                    h_diag = avg_hessian_diags[param_idx]
                    
                    # 计算更新量：使用Hessian逆
                    # 关键修复：增大更新量，使用更大的系数
                    # FedU核心公式：θ_new = θ_old - H^(-1) * ∇L
                    # 使用对角Hessian近似：H^(-1) ≈ 1 / (H_diag + eps)
                    if abs(h_diag) > eps:
                        # 使用对角Hessian逆，并增大更新量
                        # 注意：这里需要足够大的更新量才能实现遗忘
                        # 修复：大幅增大更新量（从50.0增加到500.0），确保有效遗忘
                        update_scale = alpha * lr * 500.0  # 增大更新量10倍
                        update = update_scale * grad / (abs(h_diag) + eps)
                    else:
                        # 如果Hessian对角太小，直接使用梯度（但需要更大的步长）
                        update_scale = alpha * lr * 500.0  # 同样增大10倍
                        update = update_scale * grad

                    # 自适应双任务融合（近似KKT思想）：根据梯度范数动态分配权重
                    if utility_grad_vector is not None:
                        psize = param.numel()
                        utility_grad = utility_grad_vector[vec_idx:vec_idx + psize].reshape(param.shape).to(param.device)
                        vec_idx += psize

                        fg_norm = update.norm().item()
                        ug_norm = utility_grad.norm().item() + 1e-12
                        w_forget = ug_norm / (fg_norm + ug_norm)
                        w_utility = 1.0 - w_forget

                        # 冲突时进行正交化，尽量减少效用损失
                        dot_fu = torch.sum(update.flatten() * utility_grad.flatten())
                        if dot_fu < 0:
                            denom = torch.sum(update.flatten() * update.flatten()) + 1e-12
                            utility_grad = utility_grad - (dot_fu / denom) * update

                        blended_update = w_forget * update + w_utility * utility_scale * utility_grad
                        param.data -= blended_update
                        total_update_norm += blended_update.norm().item()
                        forget_update_norm += (w_forget * update).norm().item()
                        utility_update_norm += (w_utility * utility_scale * utility_grad).norm().item()
                    else:
                        # 关键修复：减去更新量（遗忘），而不是加上
                        param.data -= update
                        total_update_norm += update.norm().item()
                param_idx += 1
                if utility_grad_vector is not None and avg_gradients[param_idx - 1] is None:
                    vec_idx += param.numel()
        
        if iteration < num_iterations - 1:
            if utility_grad_vector is not None:
                print(
                    f"  迭代 {iteration + 1}/{num_iterations} 完成，"
                    f"总更新范数: {total_update_norm:.6f}，"
                    f"遗忘分量: {forget_update_norm:.6f}，"
                    f"效用分量: {utility_update_norm:.6f}"
                )
            else:
                print(f"  迭代 {iteration + 1}/{num_iterations} 完成，更新量范数: {total_update_norm:.6f}")
    
    print("FedU遗忘完成")
    return updated_model
