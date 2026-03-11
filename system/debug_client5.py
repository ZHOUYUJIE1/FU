"""
诊断脚本：用于调试客户端5准确率为0的问题
"""
import torch
import numpy as np
from collections import Counter

def diagnose_client5_issue(model, client4, client5, device):
    """
    诊断客户端5准确率为0的问题
    
    Args:
        model: 训练好的全局模型
        client4: 客户端4对象
        client5: 客户端5对象
        device: 设备
    """
    print("=" * 80)
    print("客户端5准确率为0的诊断报告")
    print("=" * 80)
    
    model.eval()
    
    # 1. 检查数据加载
    print("\n1. 检查数据加载...")
    testloader4 = client4.load_test_data()
    testloader5 = client5.load_test_data()
    
    # 检查客户端4的数据
    labels4 = []
    for x, y in testloader4:
        labels4.extend(y.numpy())
        if len(labels4) >= 100:
            break
    label_counter4 = Counter(labels4)
    print(f"  客户端4前100个样本的标签分布: {dict(label_counter4)}")
    
    # 检查客户端5的数据
    labels5 = []
    for x, y in testloader5:
        labels5.extend(y.numpy())
        if len(labels5) >= 100:
            break
    label_counter5 = Counter(labels5)
    print(f"  客户端5前100个样本的标签分布: {dict(label_counter5)}")
    
    # 2. 检查模型在客户端4上的预测
    print("\n2. 检查模型在客户端4上的预测...")
    client4.model = model
    client4.model.eval()
    
    predictions4 = []
    labels4_full = []
    outputs4 = []
    
    with torch.no_grad():
        for x, y in testloader4:
            if type(x) == type([]):
                x[0] = x[0].to(device)
            else:
                x = x.to(device)
            output = client4.model(x)
            pred = torch.argmax(output, dim=1)
            predictions4.extend(pred.cpu().numpy())
            labels4_full.extend(y.numpy())
            outputs4.append(output.cpu().numpy())
            if len(predictions4) >= 100:
                break
    
    pred_counter4 = Counter(predictions4)
    print(f"  客户端4前100个样本的预测分布: {dict(pred_counter4)}")
    
    # 计算准确率
    acc4 = sum([p == l for p, l in zip(predictions4, labels4_full)]) / len(predictions4)
    print(f"  客户端4前100个样本的准确率: {acc4:.4f}")
    
    # 3. 检查模型在客户端5上的预测
    print("\n3. 检查模型在客户端5上的预测...")
    client5.model = model
    client5.model.eval()
    
    predictions5 = []
    labels5_full = []
    outputs5 = []
    
    with torch.no_grad():
        for x, y in testloader5:
            if type(x) == type([]):
                x[0] = x[0].to(device)
            else:
                x = x.to(device)
            output = client5.model(x)
            pred = torch.argmax(output, dim=1)
            predictions5.extend(pred.cpu().numpy())
            labels5_full.extend(y.numpy())
            outputs5.append(output.cpu().numpy())
            if len(predictions5) >= 100:
                break
    
    pred_counter5 = Counter(predictions5)
    print(f"  客户端5前100个样本的预测分布: {dict(pred_counter5)}")
    
    # 计算准确率
    acc5 = sum([p == l for p, l in zip(predictions5, labels5_full)]) / len(predictions5)
    print(f"  客户端5前100个样本的准确率: {acc5:.4f}")
    
    # 4. 分析logits值
    print("\n4. 分析模型输出的logits值...")
    if len(outputs4) > 0 and len(outputs5) > 0:
        outputs4_array = np.concatenate(outputs4, axis=0)[:len(labels4_full)]
        outputs5_array = np.concatenate(outputs5, axis=0)[:len(labels5_full)]
        
        labels4_array = np.array(labels4_full[:len(outputs4_array)])
        labels5_array = np.array(labels5_full[:len(outputs5_array)])
        
        # 客户端4的logits分析
        print("\n  客户端4的logits分析:")
        print(f"    所有类别的平均logits: {outputs4_array.mean(axis=0)}")
        idx_4_in_4 = np.where(labels4_array == 4)[0]
        idx_5_in_4 = np.where(labels4_array == 5)[0]
        if len(idx_4_in_4) > 0:
            print(f"    标签4样本的平均logits: {outputs4_array[idx_4_in_4].mean(axis=0)}")
            print(f"    标签4样本的预测类别: {Counter(np.argmax(outputs4_array[idx_4_in_4], axis=1))}")
        if len(idx_5_in_4) > 0:
            print(f"    标签5样本的平均logits: {outputs4_array[idx_5_in_4].mean(axis=0)}")
            print(f"    标签5样本的预测类别: {Counter(np.argmax(outputs4_array[idx_5_in_4], axis=1))}")
        
        # 客户端5的logits分析
        print("\n  客户端5的logits分析:")
        print(f"    所有类别的平均logits: {outputs5_array.mean(axis=0)}")
        idx_4_in_5 = np.where(labels5_array == 4)[0]
        idx_5_in_5 = np.where(labels5_array == 5)[0]
        if len(idx_4_in_5) > 0:
            print(f"    标签4样本的平均logits: {outputs5_array[idx_4_in_5].mean(axis=0)}")
            print(f"    标签4样本的预测类别: {Counter(np.argmax(outputs5_array[idx_4_in_5], axis=1))}")
            print(f"    类别4的logits值: {outputs5_array[idx_4_in_5][:, 4].mean():.4f}")
            print(f"    类别7的logits值: {outputs5_array[idx_4_in_5][:, 7].mean():.4f}")
        if len(idx_5_in_5) > 0:
            print(f"    标签5样本的平均logits: {outputs5_array[idx_5_in_5].mean(axis=0)}")
            print(f"    标签5样本的预测类别: {Counter(np.argmax(outputs5_array[idx_5_in_5], axis=1))}")
            print(f"    类别5的logits值: {outputs5_array[idx_5_in_5][:, 5].mean():.4f}")
            print(f"    类别7的logits值: {outputs5_array[idx_5_in_5][:, 7].mean():.4f}")
    
    # 5. 交叉测试：用客户端5的模型评估客户端4的数据
    print("\n5. 交叉测试...")
    print("  用客户端5的模型评估客户端4的数据（应该和客户端4的模型结果相同）")
    # 这里模型是同一个，所以结果应该相同
    
    # 6. 检查数据特征差异
    print("\n6. 检查数据特征差异...")
    # 计算客户端4和5的数据统计信息
    data4_stats = []
    data5_stats = []
    
    for x, y in testloader4:
        if type(x) == type([]):
            x = x[0]
        data4_stats.append({
            'mean': x.mean().item(),
            'std': x.std().item(),
            'min': x.min().item(),
            'max': x.max().item()
        })
        if len(data4_stats) >= 10:
            break
    
    for x, y in testloader5:
        if type(x) == type([]):
            x = x[0]
        data5_stats.append({
            'mean': x.mean().item(),
            'std': x.std().item(),
            'min': x.min().item(),
            'max': x.max().item()
        })
        if len(data5_stats) >= 10:
            break
    
    if len(data4_stats) > 0 and len(data5_stats) > 0:
        avg_mean_4 = np.mean([s['mean'] for s in data4_stats])
        avg_mean_5 = np.mean([s['mean'] for s in data5_stats])
        avg_std_4 = np.mean([s['std'] for s in data4_stats])
        avg_std_5 = np.mean([s['std'] for s in data5_stats])
        
        print(f"  客户端4数据统计 - 均值: {avg_mean_4:.4f}, 标准差: {avg_std_4:.4f}")
        print(f"  客户端5数据统计 - 均值: {avg_mean_5:.4f}, 标准差: {avg_std_5:.4f}")
        print(f"  数据差异 - 均值差: {abs(avg_mean_4 - avg_mean_5):.4f}, 标准差差: {abs(avg_std_4 - avg_std_5):.4f}")
    
    # 7. 总结
    print("\n" + "=" * 80)
    print("诊断总结")
    print("=" * 80)
    print(f"1. 客户端4的准确率: {acc4:.4f}")
    print(f"2. 客户端5的准确率: {acc5:.4f}")
    print(f"3. 客户端4的标签分布: {dict(label_counter4)}")
    print(f"4. 客户端5的标签分布: {dict(label_counter5)}")
    print(f"5. 客户端4的预测分布: {dict(pred_counter4)}")
    print(f"6. 客户端5的预测分布: {dict(pred_counter5)}")
    
    if acc5 == 0.0:
        print("\n⚠️  客户端5的准确率为0，可能的原因:")
        print("   - 模型对客户端5的数据有系统性的错误理解")
        print("   - 数据特征分布与客户端4不同，导致模型无法泛化")
        print("   - 模型过拟合到客户端4的特定样本特征")
        print("   - 可能存在数据加载或标签映射的问题")
    
    print("=" * 80)


