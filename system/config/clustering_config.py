#!/usr/bin/env python3
"""
分簇策略配置文件
"""

# 基础分簇参数
BASIC_CLUSTERING_CONFIG = {
    "target_client_id": 0,
    "min_cluster_size": 1,
    "max_cluster_size": 10,
    "default_num_clusters": 5
}

# 自适应阈值参数
ADAPTIVE_THRESHOLD_CONFIG = {
    "performance_trend_window": 3,  # 性能趋势分析窗口大小
    "improvement_threshold": 0.05,  # 性能提升阈值 (5%)
    "decline_threshold": -0.05,     # 性能下降阈值 (-5%)
    "threshold_adjustment_factor": {
        "improving": 1.1,    # 性能提升时的调整因子
        "declining": 0.9,    # 性能下降时的调整因子
        "stable": 1.0        # 性能稳定时的调整因子
    },
    "threshold_bounds": {
        "min_threshold": 0.1,
        "max_threshold": 0.9
    }
}

# 多目标分簇权重配置
MULTI_OBJECTIVE_WEIGHTS = {
    "forgetting_effectiveness": 0.4,   # 遗忘有效性权重
    "privacy_protection": 0.35,        # 隐私保护权重
    "model_utility": 0.25              # 模型效用权重
}

# 分簇算法特定参数
CLUSTERING_ALGORITHM_CONFIG = {
    "hierarchical_distance": {
        "similarity_threshold": 0.6,
        "max_cluster_size": 8
    },
    "density_based": {
        "density_threshold": 0.5,
        "neighbor_threshold": 0.5
    },
    "graph_based": {
        "adjacency_threshold": 0.5,
        "min_component_size": 2
    }
}

# 遗忘强度控制参数
FORGETTING_STRENGTH_CONFIG = {
    "base_lambda": 0.15,
    "lambda_bounds": {
        "min": 0.05,
        "max": 0.3
    },
    "progress_adjustment": {
        "amplitude": 0.5,
        "frequency": "sinusoidal"  # 正弦波调整
    },
    "cluster_adjustment": {
        "same_cluster_multiplier": 1.2,    # 同一簇内遗忘强度倍增器
        "different_cluster_multiplier": 0.8  # 不同簇间遗忘强度倍增器
    },
    "performance_adjustment": {
        "improvement_multiplier": 1.1,     # 性能提升时的倍增器
        "decline_multiplier": 0.9          # 性能下降时的倍增器
    }
}

# 分簇质量评估参数
CLUSTERING_QUALITY_CONFIG = {
    "evaluation_metrics": [
        "intra_cluster_consistency",
        "inter_cluster_separation",
        "target_client_isolation"
    ],
    "quality_thresholds": {
        "excellent": 2.0,
        "good": 1.0,
        "needs_improvement": 0.5
    },
    "weighted_scoring": {
        "intra_cluster_weight": 0.3,
        "inter_cluster_weight": 0.4,
        "target_isolation_weight": 0.3
    }
}

# 异常检测参数
ANOMALY_DETECTION_CONFIG = {
    "distance_threshold": 0.8,        # 异常距离阈值
    "density_threshold": 0.3,         # 异常密度阈值
    "isolation_threshold": 0.7,       # 异常隔离阈值
    "detection_methods": [
        "distance_based",
        "density_based",
        "isolation_based"
    ]
}

# 性能监控参数
PERFORMANCE_MONITORING_CONFIG = {
    "evaluation_frequency": 5,        # 每5个epoch评估一次
    "history_window_size": 10,        # 性能历史窗口大小
    "convergence_threshold": 0.001,   # 收敛阈值
    "early_stopping_patience": 5      # 早停耐心值
}

# 数据持久化配置
DATA_PERSISTENCE_CONFIG = {
    "save_format": "json",
    "auto_save": True,
    "save_frequency": 10,             # 每10个epoch保存一次
    "backup_count": 3,                # 保留的备份数量
    "compression": False              # 是否压缩保存
}

# 日志配置
LOGGING_CONFIG = {
    "log_level": "INFO",
    "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": "clustering_strategy.log",
    "console_output": True,
    "file_output": True
}

# 实验配置
EXPERIMENT_CONFIG = {
    "random_seed": 42,
    "num_trials": 5,                  # 实验重复次数
    "cross_validation_folds": 3,      # 交叉验证折数
    "statistical_significance": 0.05  # 统计显著性水平
}

# 第三阶段：高级多目标优化配置
ADVANCED_MULTI_OBJECTIVE_CONFIG = {
    "optimization_algorithm": "nsga2",  # 优化算法
    "population_size": 50,              # 种群大小
    "generation_limit": 100,            # 最大代数
    "crossover_rate": 0.8,              # 交叉概率
    "mutation_rate": 0.1,               # 变异概率
    "tournament_size": 3,               # 锦标赛选择大小
    "elite_size": 5,                    # 精英个体数量
    "convergence_threshold": 0.001,     # 收敛阈值
    "diversity_maintenance": True,      # 是否维护种群多样性
    "pareto_front_size": 10             # Pareto前沿解集大小
}

# 第三阶段：实时策略调整配置
REAL_TIME_ADJUSTMENT_CONFIG = {
    "adjustment_threshold": 0.1,        # 调整阈值
    "performance_window": 10,           # 性能监控窗口
    "adjustment_cooldown": 5,           # 调整冷却期
    "adaptive_threshold": True,         # 是否使用自适应阈值
    "adjustment_strategies": [          # 可用的调整策略
        "weight_adjustment",
        "threshold_adjustment", 
        "clustering_reoptimization"
    ],
    "performance_metrics": [            # 监控的性能指标
        "accuracy",
        "loss", 
        "convergence_rate",
        "stability"
    ]
}

# 第三阶段：性能预测和预警配置
PERFORMANCE_PREDICTION_CONFIG = {
    "prediction_model": "linear_regression",  # 预测模型类型
    "prediction_horizon": 5,                  # 预测时间范围
    "training_window": 20,                    # 训练数据窗口
    "warning_thresholds": {                   # 预警阈值
        "performance_decline": 0.05,          # 性能下降5%
        "stability_threshold": 0.3,           # 稳定性阈值
        "convergence_threshold": 0.001        # 收敛阈值
    },
    "warning_levels": [                       # 预警级别
        "info",
        "warning", 
        "critical"
    ],
    "auto_retraining": True,                  # 是否自动重新训练模型
    "retraining_frequency": 50                # 重新训练频率
}

# 第三阶段：增强客户端监控配置
ENHANCED_CLIENT_MONITORING_CONFIG = {
    "monitoring_frequency": 5,               # 监控频率（每N个epoch）
    "performance_history_limit": 100,         # 性能历史记录限制
    "intervention_thresholds": {              # 干预阈值
        "accuracy_decline": 0.1,              # 准确率下降10%
        "loss_increase": 0.2,                 # 损失增加20%
        "stability_threshold": 0.3            # 稳定性阈值
    },
    "intervention_strategies": [              # 干预策略
        "protection_increase",
        "clustering_adjustment",
        "weight_rebalancing"
    ],
    "global_metrics_tracking": True,          # 是否跟踪全局指标
    "anomaly_detection": True,                # 是否启用异常检测
    "auto_intervention": True                 # 是否自动干预
}

# 第三阶段：系统性能优化配置
SYSTEM_PERFORMANCE_CONFIG = {
    "caching_enabled": True,                 # 是否启用缓存
    "cache_size_limits": {                   # 缓存大小限制
        "distance_cache": 1000,              # 距离缓存
        "clustering_cache": 500,              # 分簇缓存
        "optimization_cache": 200             # 优化缓存
    },
    "cache_eviction_policy": "lru",          # 缓存淘汰策略
    "parallel_processing": True,              # 是否启用并行处理
    "max_workers": 4,                        # 最大工作线程数
    "memory_optimization": True,              # 是否启用内存优化
    "profiling_enabled": False               # 是否启用性能分析
}

def get_config(config_name):
    """获取指定配置"""
    configs = {
        "basic": BASIC_CLUSTERING_CONFIG,
        "adaptive_threshold": ADAPTIVE_THRESHOLD_CONFIG,
        "multi_objective": MULTI_OBJECTIVE_WEIGHTS,
        "algorithm": CLUSTERING_ALGORITHM_CONFIG,
        "forgetting_strength": FORGETTING_STRENGTH_CONFIG,
        "quality": CLUSTERING_QUALITY_CONFIG,
        "anomaly": ANOMALY_DETECTION_CONFIG,
        "monitoring": PERFORMANCE_MONITORING_CONFIG,
        "persistence": DATA_PERSISTENCE_CONFIG,
        "logging": LOGGING_CONFIG,
        "experiment": EXPERIMENT_CONFIG,
        # 第三阶段配置
        "advanced_multi_objective": ADVANCED_MULTI_OBJECTIVE_CONFIG,
        "real_time_adjustment": REAL_TIME_ADJUSTMENT_CONFIG,
        "performance_prediction": PERFORMANCE_PREDICTION_CONFIG,
        "enhanced_client_monitoring": ENHANCED_CLIENT_MONITORING_CONFIG,
        "system_performance": SYSTEM_PERFORMANCE_CONFIG
    }
    
    return configs.get(config_name, {})

def get_all_configs():
    """获取所有配置"""
    return {
        "basic": BASIC_CLUSTERING_CONFIG,
        "adaptive_threshold": ADAPTIVE_THRESHOLD_CONFIG,
        "multi_objective": MULTI_OBJECTIVE_WEIGHTS,
        "algorithm": CLUSTERING_ALGORITHM_CONFIG,
        "forgetting_strength": FORGETTING_STRENGTH_CONFIG,
        "quality": CLUSTERING_QUALITY_CONFIG,
        "anomaly": ANOMALY_DETECTION_CONFIG,
        "monitoring": PERFORMANCE_MONITORING_CONFIG,
        "persistence": DATA_PERSISTENCE_CONFIG,
        "logging": LOGGING_CONFIG,
        "experiment": EXPERIMENT_CONFIG,
        # 第三阶段配置
        "advanced_multi_objective": ADVANCED_MULTI_OBJECTIVE_CONFIG,
        "real_time_adjustment": REAL_TIME_ADJUSTMENT_CONFIG,
        "performance_prediction": PERFORMANCE_PREDICTION_CONFIG,
        "enhanced_client_monitoring": ENHANCED_CLIENT_MONITORING_CONFIG,
        "system_performance": SYSTEM_PERFORMANCE_CONFIG
    }

def update_config(config_name, updates):
    """更新指定配置"""
    configs = get_all_configs()
    if config_name in configs:
        configs[config_name].update(updates)
        return configs[config_name]
    return None

def validate_config(config_name):
    """验证指定配置的有效性"""
    configs = get_all_configs()
    if config_name not in configs:
        return False, f"配置 {config_name} 不存在"
    
    config = configs[config_name]
    
    # 基础验证规则
    if config_name == "basic":
        if config["target_client_id"] < 0:
            return False, "目标客户端ID不能为负数"
        if config["min_cluster_size"] < 1:
            return False, "最小簇大小不能小于1"
    
    elif config_name == "adaptive_threshold":
        if config["performance_trend_window"] < 2:
            return False, "性能趋势窗口大小不能小于2"
        if config["threshold_bounds"]["min_threshold"] >= config["threshold_bounds"]["max_threshold"]:
            return False, "阈值边界设置无效"
    
    elif config_name == "multi_objective":
        total_weight = sum(config.values())
        if abs(total_weight - 1.0) > 1e-6:
            return False, f"权重总和必须为1.0，当前为{total_weight}"
    
    elif config_name == "forgetting_strength":
        if config["lambda_bounds"]["min"] >= config["lambda_bounds"]["max"]:
            return False, "遗忘强度边界设置无效"
    
    return True, "配置验证通过"

def print_config_summary():
    """打印配置摘要"""
    print("=" * 60)
    print("分簇策略配置摘要")
    print("=" * 60)
    
    configs = get_all_configs()
    for name, config in configs.items():
        print(f"\n{name.upper()} 配置:")
        for key, value in config.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")

if __name__ == "__main__":
    # 打印配置摘要
    print_config_summary()
    
    # 测试配置验证
    print("\n" + "=" * 60)
    print("配置验证测试")
    print("=" * 60)
    
    for config_name in ["basic", "adaptive_threshold", "multi_objective", "forgetting_strength"]:
        is_valid, message = validate_config(config_name)
        status = "✓" if is_valid else "✗"
        print(f"{status} {config_name}: {message}")
