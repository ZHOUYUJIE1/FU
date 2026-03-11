import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import torch.nn.functional as F


def membership_inference_attack(model, client, device, max_batches=10):
    """
    成员推理攻击：判断样本是否为训练集成员
    返回AUC、准确率、成员置信度、非成员置信度
    """
    model.eval()
    member_confidences = []
    nonmember_confidences = []
    
    # 1. 训练集（成员）
    trainloader = client.load_train_data()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(trainloader):
            if type(x) == type([]):
                x = x[0].to(device)
            else:
                x = x.to(device)
            y = y.to(device)
            output = model(x)
            probs = F.softmax(output, dim=1)
            max_conf, _ = torch.max(probs, dim=1)
            member_confidences.extend(max_conf.cpu().numpy())
            if batch_idx >= max_batches:
                break
    
    # 2. 测试集（非成员）
    testloader = client.load_test_data()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(testloader):
            if type(x) == type([]):
                x = x[0].to(device)
            else:
                x = x.to(device)
            y = y.to(device)
            output = model(x)
            probs = F.softmax(output, dim=1)
            max_conf, _ = torch.max(probs, dim=1)
            nonmember_confidences.extend(max_conf.cpu().numpy())
            if batch_idx >= max_batches:
                break
    
    # 3. 构造标签和分数
    y_true = np.array([1]*len(member_confidences) + [0]*len(nonmember_confidences))
    y_score = np.array(member_confidences + nonmember_confidences)
    
    # 4. 计算AUC和最佳阈值准确率
    auc = roc_auc_score(y_true, y_score)
    best_acc = 0
    best_thr = 0.5
    for thr in np.linspace(0, 1, 100):
        y_pred = (y_score >= thr).astype(int)
        acc = accuracy_score(y_true, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_thr = thr
    
    # MIA-SR: 攻击成功率（这里采用最优阈值下的分类准确率）
    mia_sr = best_acc

    # 固定阈值0.5下的成功率（便于跨实验做稳定横向比较）
    fixed_thr = 0.5
    fixed_acc = accuracy_score(y_true, (y_score >= fixed_thr).astype(int))

    return {
        'auc': auc,
        'best_acc': best_acc,
        'best_thr': best_thr,
        'mia_sr': mia_sr,
        'fixed_thr': fixed_thr,
        'fixed_acc': fixed_acc,
        'member_conf': member_confidences,
        'nonmember_conf': nonmember_confidences
    }


def backdoor_attack(model, client, device, trigger_func=None, max_batches=5):
    """
    后门攻击评估：测试模型对后门样本的识别能力
    trigger_func: 用于生成带触发器的样本的函数
    返回后门样本准确率
    """
    model.eval()
    total = 0
    correct = 0
    
    # 默认：用测试集加触发器
    testloader = client.load_test_data()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(testloader):
            if trigger_func is not None:
                x, y = trigger_func(x, y)
            if type(x) == type([]):
                x = x[0].to(device)
            else:
                x = x.to(device)
            y = y.to(device)
            output = model(x)
            pred = torch.argmax(output, dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            if batch_idx >= max_batches:
                break
    acc = correct / max(total, 1)
    return acc
