#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
保护级别配置示例
用于调整联邦遗忘中的客户端保护强度
"""

# 联邦学习遗忘保护级别配置
# 用于配置不同级别的客户端保护策略

# 基础保护级别配置
PROTECTION_LEVELS = {
    'weak': {
        'description': '弱保护 - 优先遗忘效果，其他客户端性能可能下降较多',
        'lambda_reversal': 0.15,
        'protection_scale': 0.6,
        'similarity_threshold': 0.6,
        'cluster_protection': True,
        'use_knowledge_anchoring': True,  # 启用知识锚点保护
        'anchor_weight': 0.5,
        'anchor_sample_ratio': 0.15,
        'anchor_start_epoch': 0
    },
    'moderate': {
        'description': '中等保护 - 平衡遗忘效果和性能保护',
        'lambda_reversal': 0.2,
        'protection_scale': 0.7,
        'similarity_threshold': 0.7,
        'cluster_protection': True,
        'use_knowledge_anchoring': True,  # 启用知识锚点保护
        'anchor_weight': 0.4,
        'anchor_sample_ratio': 0.15,
        'anchor_start_epoch': 0
    },
    'strong': {
        'description': '强保护 - 优先保护其他客户端性能，遗忘效果可能稍弱',
        'lambda_reversal': 0.15,
        'protection_scale': 0.5,
        'similarity_threshold': 0.8,
        'cluster_protection': True,
        'use_knowledge_anchoring': True,  # 启用知识锚点保护
        'anchor_weight': 0.6,
        'anchor_sample_ratio': 0.2,
        'anchor_start_epoch': 0  # 从第1轮开始就应用锚点保护
    }
}

# 知识锚点保护专用配置
KNOWLEDGE_ANCHORING_CONFIGS = {
    'conservative': {
        'description': '保守模式 - 最小化对遗忘效果的影响',
        'anchor_weight': 0.15,
        'anchor_sample_ratio': 0.08,
        'anchor_start_epoch': 2,  # 从第3轮开始应用
        'use_adaptive_weight': True
    },
    'balanced': {
        'description': '平衡模式 - 在遗忘效果和性能保护间平衡',
        'anchor_weight': 0.3,
        'anchor_sample_ratio': 0.12,
        'anchor_start_epoch': 1,  # 从第2轮开始应用
        'use_adaptive_weight': True
    },
    'aggressive': {
        'description': '积极模式 - 最大化性能保护，可能影响遗忘效果',
        'anchor_weight': 0.5,
        'anchor_sample_ratio': 0.2,
        'anchor_start_epoch': 0,  # 从第1轮开始应用
        'use_adaptive_weight': False
    }
}

# 动态锚点权重调整配置
ADAPTIVE_ANCHOR_CONFIG = {
    'enable_adaptive_weight': True,
    'performance_threshold': 0.05,  # 性能下降超过5%时调整权重
    'weight_increase_factor': 1.2,  # 权重增加因子
    'weight_decrease_factor': 0.8,  # 权重减少因子
    'max_weight': 0.6,  # 最大权重限制
    'min_weight': 0.1   # 最小权重限制
}

# 客户端相似性保护配置
SIMILARITY_PROTECTION_CONFIG = {
    'high_similarity_threshold': 0.8,  # 高相似性阈值
    'medium_similarity_threshold': 0.6,  # 中等相似性阈值
    'high_protection_factor': 0.3,  # 高相似性客户端保护因子
    'medium_protection_factor': 0.6,  # 中等相似性客户端保护因子
    'low_protection_factor': 0.9   # 低相似性客户端保护因子
}

# 分簇策略配置
CLUSTERING_STRATEGY_CONFIG = {
    'use_hierarchical_clustering': True,
    'use_adaptive_thresholds': True,
    'use_cluster_based_forgetting': True,
    'min_cluster_size': 1,
    'max_clusters': 10,
    'similarity_metric': 'wasserstein',  # 'wasserstein', 'euclidean', 'cosine'
    'clustering_method': 'hierarchical'  # 'hierarchical', 'kmeans', 'dbscan'
}

# 性能监控配置
PERFORMANCE_MONITORING_CONFIG = {
    'monitor_frequency': 1,  # 每轮监控
    'performance_history_length': 10,  # 保存最近10轮的性能历史
    'alert_threshold': 0.1,  # 性能下降超过10%时发出警告
    'auto_adjustment': True,  # 自动调整保护参数
    'log_detailed_metrics': True  # 记录详细指标
}

def get_protection_config(level='moderate', anchoring_mode='balanced'):
    """
    获取指定保护级别的配置
    
    Args:
        level: 保护级别 ('weak', 'moderate', 'strong')
        anchoring_mode: 知识锚点模式 ('conservative', 'balanced', 'aggressive')
    
    Returns:
        dict: 合并后的配置字典
    """
    if level not in PROTECTION_LEVELS:
        raise ValueError(f"不支持的保护级别: {level}")
    
    if anchoring_mode not in KNOWLEDGE_ANCHORING_CONFIGS:
        raise ValueError(f"不支持的知识锚点模式: {anchoring_mode}")
    
    # 合并基础保护和知识锚点配置
    config = PROTECTION_LEVELS[level].copy()
    anchoring_config = KNOWLEDGE_ANCHORING_CONFIGS[anchoring_mode].copy()
    
    # 如果基础配置启用了知识锚点，使用指定的锚点模式
    if config['use_knowledge_anchoring']:
        config.update(anchoring_config)
    
    return config

def print_config_summary(config):
    """打印配置摘要"""
    print("=" * 60)
    print("联邦学习遗忘保护配置摘要")
    print("=" * 60)
    print(f"保护级别: {config.get('description', 'N/A')}")
    print(f"知识锚点保护: {'启用' if config.get('use_knowledge_anchoring', False) else '禁用'}")
    if config.get('use_knowledge_anchoring', False):
        print(f"  锚点损失权重: {config.get('anchor_weight', 0.0):.2f}")
        print(f"  锚点数据采样比例: {config.get('anchor_sample_ratio', 0.0):.2f}")
        print(f"  锚点保护起始轮次: {config.get('anchor_start_epoch', 0)}")
    print(f"梯度反转强度: {config.get('lambda_reversal', 0.0):.2f}")
    print(f"保护缩放系数: {config.get('protection_scale', 0.0):.2f}")
    print(f"相似性阈值: {config.get('similarity_threshold', 0.0):.2f}")
    print(f"分簇保护: {'启用' if config.get('cluster_protection', False) else '禁用'}")
    print("=" * 60)

# 示例用法
if __name__ == "__main__":
    # 获取中等保护级别 + 平衡锚点模式的配置
    config = get_protection_config('moderate', 'balanced')
    print_config_summary(config)
    
    print("\n" + "=" * 60)
    print("所有可用的保护级别:")
    for level, desc in PROTECTION_LEVELS.items():
        print(f"  {level}: {desc['description']}")
    
    print("\n所有可用的知识锚点模式:")
    for mode, desc in KNOWLEDGE_ANCHORING_CONFIGS.items():
        print(f"  {mode}: {desc['description']}")
    print("=" * 60)
