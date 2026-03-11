import numpy as np
import random
import torch
from scipy.spatial.distance import cosine
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity


def num_to_class_two(data):
    k_class = 10
    kdes = []
    for i in range(len(data)):
        kde = KernelDensity(bandwidth=0.1, kernel='gaussian')
        kde.fit(data[i])
        kdes.append(kde)

    # 获取每个数组中每个数据点的密度估计值
    densities = []
    for i in range(len(data)):
        density = kdes[i].score_samples(data[i])
        densities.append(density)

    # 将每个数组中的数据按照密度分布分成4个类别
    n_groups = k_class
    groups = []
    for i in range(len(data)):
        indices = np.argsort(densities[i])
        group = np.zeros(len(data[i]))
        group_size = len(data[i]) // n_groups
        for j in range(n_groups):
            group[indices[j * group_size:(j + 1) * group_size]] = j
        groups.append(group)

    result_list = [[] for _ in range(k_class)]
    for j in range(len(groups)):
        result_list[int(groups[j])].append(j)

    result_list_two = [[] for _ in range(k_class)]
    for j in range(len(result_list_two)):
        for i in range(len(result_list)):
            # print(result_list)
            # print(i)
            m = list(np.random.choice(result_list[i], 10, replace=False))
            result_list[i] = list(set(result_list[i]) - set(m))
            result_list_two[j] = result_list_two[j] + m
    print(result_list_two)
    return result_list_two


def num_to_class(data):
    # 使用KernelDensity估计数据的密度分布
    k_class = 10
    # kde = KernelDensity(bandwidth=0.1, kernel='gaussian')
    kde = KernelDensity(bandwidth=0.01, kernel='gaussian')
    kde.fit(data.reshape(-1, 1))

    # 获取每个数据点的密度估计值
    densities = kde.score_samples(data.reshape(-1, 1))

    # 将数据按照密度分布分成10个类别
    n_groups = k_class
    indices = np.argsort(densities)
    groups = np.zeros(len(data))
    group_size = len(data) // n_groups
    for i in range(n_groups):
        groups[indices[i * group_size:(i + 1) * group_size]] = i

    result_list = [[] for _ in range(k_class)]
    for j in range(len(groups)):
        result_list[int(groups[j])].append(j)

    # result_list_two = [[] for _ in range(k_class)]
    result_list_two = [[] for _ in range(10)]
    for j in range(len(result_list_two)):
        for i in range(len(result_list)):
            # if j == 0 or (j % 2 == 0):
            #     if i == 0 or (i % 2 == 0):
            #         m = list(np.random.choice(result_list[i], 2, replace=False))
            #     else:
            #         m = list(np.random.choice(result_list[i], 3, replace=False))
            # else:
            #     if i == 0 or (i % 2 == 0):
            #         m = list(np.random.choice(result_list[i], 3, replace=False))
            #     else:
            #         m = list(np.random.choice(result_list[i], 2, replace=False))
            # m = list(np.random.choice(result_list[i], 1, replace=False))
            m = list(np.random.choice(result_list[i], 1, replace=False))
            result_list[i] = list(set(result_list[i]) - set(m))
            result_list_two[j] = result_list_two[j] + m
    print(result_list_two)
    print(len(result_list_two))
    print(len(result_list_two[0]))
    return result_list_two


def num_to_class_five(data):
    # 使用KernelDensity估计数据的密度分布
    k_class = 10
    # kde = KernelDensity(bandwidth=0.1, kernel='gaussian')
    kde = KernelDensity(bandwidth=0.01, kernel='gaussian')
    kde.fit(data.reshape(-1, 1))

    # 获取每个数据点的密度估计值
    densities = kde.score_samples(data.reshape(-1, 1))

    # 将数据按照密度分布分成10个类别
    n_groups = k_class
    indices = np.argsort(densities)
    groups = np.zeros(len(data))
    group_size = len(data) // n_groups
    for i in range(n_groups):
        groups[indices[i * group_size:(i + 1) * group_size]] = i

    result_list = [[] for _ in range(k_class)]
    for j in range(len(groups)):
        result_list[int(groups[j])].append(j)

    # result_list_two = [[] for _ in range(k_class)]
    result_list_two = [[] for _ in range(5)]
    for j in range(len(result_list_two)):
        for i in range(len(result_list)):
            m = list(np.random.choice(result_list[i], 2, replace=False))
            result_list[i] = list(set(result_list[i]) - set(m))
            result_list_two[j] = result_list_two[j] + m
    print(result_list_two)
    print(len(result_list_two))
    print(len(result_list_two[0]))
    return result_list_two


def difference_model(wigs):
    modeltoarray = [m.values() for m in wigs]
    param_list = list()
    for x in modeltoarray:
        line = list()
        for xx in x:
            line.append(xx.reshape((-1, 1)).cpu().numpy())
        mm = np.concatenate(line)
        param_list.append(mm)
    baseline = np.array(param_list[-1]).squeeze()
    n = len(param_list) - 1
    cos_sim = []
    new_param_list = []

    # compute cos similarity
    for each_param_list in param_list:
        each_param_array = np.array(each_param_list).squeeze()  # 拷贝 dot(矩阵乘积) norm()：norm(x,p)返回x的p范数
        cos_sim.append(np.dot(baseline, each_param_array) / (np.linalg.norm(baseline) + 1e-9) / (
                np.linalg.norm(each_param_array) + 1e-9))


    cos_sim = np.stack([*cos_sim])[:-1]  # 堆叠
    cos_sim = np.maximum(cos_sim, 0)  # relu
    normalized_weights = cos_sim / (np.sum(cos_sim) + 1e-9)  # weighted trust score
    # # normalize the magnitudes and weight by the trust score
    # for i in range(n):
    #     new_param_list.append(
    #         param_list[i] * normalized_weights[i] / (np.linalg.norm(param_list[i]) + 1e-9) * np.linalg.norm(baseline))
    weig_sort = [n - 1 for _ in range(n)]
    result_list = [[] for _ in range(10)]
    for i in range(n):
        for j in range(i + 1, n):
            if normalized_weights[i] >= normalized_weights[j]:
                weig_sort[j] -= 1
            else:
                weig_sort[i] -= 1
        m = int(weig_sort[i] / 100)
        result_list[m].append(i)
    print(result_list)
    # result_list2 = []
    # for mm in range(len(result_list)):
    #     result_list2 += result_list[mm]
    return result_list


def difference_model_middle_midu(wigs):
    modeltoarray = [m.values() for m in wigs]
    # print("Sfs")
    # param_list = [np.concatenate([xx.reshape((-1, 1)) for xx in x]) for x in modeltoarray]
    param_list = list()
    for x in modeltoarray:
        line = list()
        for xx in x:
            line.append(xx.reshape((-1, 1)).cpu().numpy())
        mm = np.concatenate(line)
        param_list.append(mm)
    baseline = np.array(param_list[-1]).squeeze()
    n = len(param_list) - 1
    cos_sim = []
    # new_param_list = []
    # print(param_list)
    # print(len(param_list))
    # print(param_list[0].shape)
    # m = param_list[:-1]
    # return num_to_class_two(m)


    # compute cos similarity
    for each_param_list in param_list:
        each_param_array = np.array(each_param_list).squeeze()  # 拷贝 dot(矩阵乘积) norm()：norm(x,p)返回x的p范数
        cos_sim.append(np.dot(baseline, each_param_array) / (np.linalg.norm(baseline) + 1e-9) / (
                np.linalg.norm(each_param_array) + 1e-9))

    cos_sim = np.stack([*cos_sim])[:-1]  # 堆叠
    cos_sim = np.maximum(cos_sim, 0)  # relu
    normalized_weights = cos_sim / (np.sum(cos_sim) + 1e-9)  # weighted trust score

    # return num_to_class(normalized_weights)
    return num_to_class_five(normalized_weights)



    # weig_sort = [n - 1 for _ in range(n)]
    # result_list = [[] for _ in range(10)]
    # for i in range(n):
    #     for j in range(i + 1, n):
    #         if normalized_weights[i] >= normalized_weights[j]:
    #             weig_sort[j] -= 1
    #         else:
    #             weig_sort[i] -= 1
    #     m = int(weig_sort[i] / 100)
    #     result_list[m].append(i)
    # print(result_list)
    # result_list_two = [[] for _ in range(5)]
    # # for j in range(len(result_list)):
    # for j in range(len(result_list_two)):
    #     for i in range(len(result_list)):
    #         m = list(np.random.choice(result_list[i], 20, replace=False))
    #         result_list[i] = list(set(result_list[i]) - set(m))
    #         result_list_two[j] = result_list_two[j] + m
    # print(result_list_two)
    # return result_list_two


def difference_model_middle(wigs):
    modeltoarray = [m.values() for m in wigs]
    # print("Sfs")
    # param_list = [np.concatenate([xx.reshape((-1, 1)) for xx in x]) for x in modeltoarray]
    param_list = list()
    for x in modeltoarray:
        line = list()
        for xx in x:
            line.append(xx.reshape((-1, 1)).cpu().numpy())
        mm = np.concatenate(line)
        param_list.append(mm)
    baseline = np.array(param_list[-1]).squeeze()
    n = len(param_list) - 1
    cos_sim = []
    new_param_list = []

    # compute cos similarity
    for each_param_list in param_list:
        each_param_array = np.array(each_param_list).squeeze()  # 拷贝 dot(矩阵乘积) norm()：norm(x,p)返回x的p范数
        cos_sim.append(np.dot(baseline, each_param_array) / (np.linalg.norm(baseline) + 1e-9) / (
                np.linalg.norm(each_param_array) + 1e-9))

    cos_sim = np.stack([*cos_sim])[:-1]  # 堆叠
    cos_sim = np.maximum(cos_sim, 0)  # relu
    normalized_weights = cos_sim / (np.sum(cos_sim) + 1e-9)  # weighted trust score

    weig_sort = [n - 1 for _ in range(n)]
    result_list = [[] for _ in range(10)]
    for i in range(n):
        for j in range(i + 1, n):
            if normalized_weights[i] >= normalized_weights[j]:
                weig_sort[j] -= 1
            else:
                weig_sort[i] -= 1
        m = int(weig_sort[i] / 100)
        result_list[m].append(i)
    print(result_list)
    result_list_two = [[] for _ in range(5)]
    # for j in range(len(result_list)):
    for j in range(len(result_list_two)):
        for i in range(len(result_list)):
            m = list(np.random.choice(result_list[i], 20, replace=False))
            result_list[i] = list(set(result_list[i]) - set(m))
            result_list_two[j] = result_list_two[j] + m
    print(result_list_two)
    return result_list_two


def difference_model_middle_k(wigs):
    modeltoarray = [m.values() for m in wigs]
    # print("Sfs")
    # param_list = [np.concatenate([xx.reshape((-1, 1)) for xx in x]) for x in modeltoarray]
    param_list = list()
    for x in modeltoarray:
        line = list()
        for xx in x:
            line.append(xx.reshape((-1, 1)).cpu().numpy())
        mm = np.concatenate(line)
        param_list.append(mm)
    baseline = np.array(param_list[-1]).squeeze()
    n = len(param_list) - 1
    cos_sim = []
    new_param_list = []

    kmeans = KMeans(n_clusters=10, random_state=0)
    for i in range(len(param_list) - 1):
        kmeans.fit(param_list[i])
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    print(labels)
    print(centers)


    # normalized_weights =
    # weig_sort = [n - 1 for _ in range(n)]
    # result_list = [[] for _ in range(10)]
    # for i in range(n):
    #     for j in range(i + 1, n):
    #         if normalized_weights[i] >= normalized_weights[j]:
    #             weig_sort[j] -= 1
    #         else:
    #             weig_sort[i] -= 1
    #     m = int(weig_sort[i] / 30)
    #     result_list[m].append(i)
    # print(result_list)
    # result_list_two = [[] for _ in range(10)]
    # for j in range(len(result_list)):
    #     for i in range(len(result_list)):
    #         m = list(np.random.choice(result_list[i], 3, replace=False))
    #         result_list[i] = list(set(result_list[i]) - set(m))
    #         result_list_two[j] = result_list_two[j] + m
    # print(result_list_two)
    # return result_list_two
    return []


def difference_model_two_middle(wigs):
    modeltoarray = [m.values() for m in wigs]
    # print("Sfs")
    # param_list = [np.concatenate([xx.reshape((-1, 1)) for xx in x]) for x in modeltoarray]
    param_list = list()
    for x in modeltoarray:
        line = list()
        for xx in x:
            line.append(xx.reshape((-1, 1)).cpu().numpy())
        mm = np.concatenate(line)
        param_list.append(mm)
    baseline = np.array(param_list[-1]).squeeze()
    n = len(param_list) - 1
    cos_sim = []
    new_param_list = []

    # compute cos similarity
    for each_param_list in param_list:
        each_param_array = np.array(each_param_list).squeeze()  # 拷贝 dot(矩阵乘积) norm()：norm(x,p)返回x的p范数
        cos_sim.append(np.dot(baseline, each_param_array) / (np.linalg.norm(baseline) + 1e-9) / (
                np.linalg.norm(each_param_array) + 1e-9))

    cos_sim = np.stack([*cos_sim])[:-1]  # 堆叠
    cos_sim = np.maximum(cos_sim, 0)  # relu
    normalized_weights = cos_sim / (np.sum(cos_sim) + 1e-9)  # weighted trust score

    weig_sort = [n - 1 for _ in range(n)]
    result_list = [[] for _ in range(10)]
    for i in range(n):
        for j in range(i + 1, n):
            if normalized_weights[i] >= normalized_weights[j]:
                weig_sort[j] -= 1
            else:
                weig_sort[i] -= 1
        m = int(weig_sort[i] / 100)
        result_list[m].append(i)
    print(result_list)
    result_list_two = [[] for _ in range(20)]
    for j in range(len(result_list_two)):
        for i in range(len(result_list)):
            m = list(np.random.choice(result_list[i], 5, replace=False))
            result_list[i] = list(set(result_list[i]) - set(m))
            result_list_two[j] = result_list_two[j] + m
    print(result_list_two)
    return result_list_two


def mahalanobis_distance(x, y):
    # 计算协方差矩阵
    cov_matrix = np.cov(x, y, rowvar=False)
    # 计算协方差矩阵的逆矩阵
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    # 计算马氏距离
    diff = np.hstack([x - y])
    md = np.sqrt(np.dot(np.dot(diff, inv_cov_matrix), diff.T))
    return md


# 马氏距离
def difference_model_middle_ma(wigs):
    modeltoarray = [m.values() for m in wigs]
    # print("Sfs")
    # param_list = [np.concatenate([xx.reshape((-1, 1)) for xx in x]) for x in modeltoarray]
    param_list = list()
    for x in modeltoarray:
        line = list()
        for xx in x:
            line.append(xx.reshape((-1, 1)).cpu().numpy())
        mm = np.concatenate(line)
        param_list.append(mm)
    baseline = np.array(param_list[-1]).squeeze()
    # np.sqrt(np.sum((x1 - x2) ** 2))
    n = len(param_list) - 1
    cos_sim = []
    new_param_list = []
    normalized_weights = []
    for i in range(len(param_list) - 1):
        normalized_weights.append(mahalanobis_distance(param_list[i], param_list[-1]))
    print(normalized_weights)
    weig_sort = [n - 1 for _ in range(n)]
    result_list = [[] for _ in range(10)]
    for i in range(n):
        for j in range(i + 1, n):
            if normalized_weights[i] >= normalized_weights[j]:
                weig_sort[j] -= 1
            else:
                weig_sort[i] -= 1
        m = int(weig_sort[i] / 30)
        result_list[m].append(i)
    print(result_list)
    result_list_two = [[] for _ in range(10)]
    for j in range(len(result_list)):
        for i in range(len(result_list)):
            m = list(np.random.choice(result_list[i], 3, replace=False))
            result_list[i] = list(set(result_list[i]) - set(m))
            result_list_two[j] = result_list_two[j] + m
    print(result_list_two)
    return result_list_two


# 欧几里得距离
def difference_model_middle_ou(wigs):
    modeltoarray = [m.values() for m in wigs]
    # print("Sfs")
    # param_list = [np.concatenate([xx.reshape((-1, 1)) for xx in x]) for x in modeltoarray]
    param_list = list()
    for x in modeltoarray:
        line = list()
        for xx in x:
            line.append(xx.reshape((-1, 1)).cpu().numpy())
        mm = np.concatenate(line)
        param_list.append(mm)
    baseline = np.array(param_list[-1]).squeeze()
    # np.sqrt(np.sum((x1 - x2) ** 2))
    n = len(param_list) - 1
    cos_sim = []
    new_param_list = []
    normalized_weights = []
    for i in range(len(param_list) - 1):
        normalized_weights.append(np.linalg.norm(param_list[i] - param_list[-1]))
    print(normalized_weights)
    # # compute cos similarity
    # for each_param_list in param_list:
    #     each_param_array = np.array(each_param_list).squeeze()  # 拷贝 dot(矩阵乘积) norm()：norm(x,p)返回x的p范数
    #     cos_sim.append(np.dot(baseline, each_param_array) / (np.linalg.norm(baseline) + 1e-9) / (
    #             np.linalg.norm(each_param_array) + 1e-9))
    #
    # cos_sim = np.stack([*cos_sim])[:-1]  # 堆叠
    # cos_sim = np.maximum(cos_sim, 0)  # relu
    # normalized_weights = cos_sim / (np.sum(cos_sim) + 1e-9)  # weighted trust score

    weig_sort = [n - 1 for _ in range(n)]
    result_list = [[] for _ in range(10)]
    for i in range(n):
        for j in range(i + 1, n):
            if normalized_weights[i] >= normalized_weights[j]:
                weig_sort[j] -= 1
            else:
                weig_sort[i] -= 1
        m = int(weig_sort[i] / 30)
        result_list[m].append(i)
    print(result_list)
    result_list_two = [[] for _ in range(10)]
    for j in range(len(result_list)):
        for i in range(len(result_list)):
            m = list(np.random.choice(result_list[i], 3, replace=False))
            result_list[i] = list(set(result_list[i]) - set(m))
            result_list_two[j] = result_list_two[j] + m
    print(result_list_two)
    return result_list_two


def dsds():
    result_list = [[] for _ in range(20)]
    # result_list = [[] for _ in range(10)]
    orign = [i for i in range(100)]
    for i in range(len(result_list)):
        m = list(np.random.choice(orign, 5, replace=False))
        result_list[i] = m
    print(result_list)
    print(len(result_list))
    return result_list


def dsds_ten():
    result_list = [[] for _ in range(10)]
    # result_list = [[] for _ in range(10)]
    orign = [i for i in range(100)]
    for i in range(len(result_list)):
        m = list(np.random.choice(orign, 10, replace=False))
        result_list[i] = m
    print(result_list)
    print(len(result_list))
    return result_list


def dsds_fiveoo():
    result_list = [[] for _ in range(5)]
    # result_list = [[] for _ in range(10)]
    orign = [i for i in range(100)]
    for i in range(len(result_list)):
        m = list(np.random.choice(orign, 20, replace=False))
        result_list[i] = m
    print(result_list)
    return result_list


def dsds_twoshi():
    result_list = [[] for _ in range(20)]
    # result_list = [[] for _ in range(10)]
    orign = [i for i in range(1000)]
    for i in range(len(result_list)):
        m = list(np.random.choice(orign, 50, replace=False))
        result_list[i] = m
    print(result_list)
    return result_list


def ssss(*qwe):
    print(qwe[:-1])


if __name__ == '__main__':
    print(dsds())
    print(len(dsds()))
    print(len(dsds()[0]))
    # ddd = [{"a":1,"sf":2,"sfa":4}, {"a":3,"sf":232,"sfa":34}]
    # print([m.values() for m in ddd])
    # np.concatenate()
    # print(np.random.choice(10, 10, replace=False))
    # for _ in range(10):
    #     print(random.randint(0, 4))
    # print(torch.cuda.is_available())
    # print(torch.__version__)
    # m =[1,2,3,461,163]
    # n =[32,46,3,679,679]
    # print(m + n)
    # s = range(6)
    # print(s)
