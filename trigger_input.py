import os
import sys

import json
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import mahalanobis

### 计算马氏距离方法一
class DistributionEstimator:
    def __init__(self, target_data):
        self.mean = np.mean(target_data, axis=0)
        self.cov = np.cov(target_data, rowvar=False)
        self.inv_cov = np.linalg.inv(self.cov)
        self.threshold = np.sqrt(np.diag(self.cov))
        self.bound = np.sqrt(np.dot(np.dot(self.threshold, self.inv_cov), self.threshold.T))

    def calculate_mahalanobis_distance(self, new_data):
        deviation = new_data - self.mean
        # print("sqrt", np.dot(np.dot(deviation, self.inv_cov), deviation.T))
        distance = np.sqrt(np.dot(np.dot(deviation, self.inv_cov), deviation.T))

        return self.mean, distance

### 计算马氏距离方法二
def calculate_mahalanobis_distance(original_data, new_data):
    # 将原始数据集和新数据转换为NumPy数组
    original_data = np.array(original_data)
    new_data = np.array(new_data)

    # 计算原始数据集的协方差矩阵的逆
    cov_inv = np.linalg.inv(np.cov(original_data, rowvar=False))

    # 计算新数据与原始分布的马氏距离
    distance = mahalanobis(new_data, original_data.mean(axis=0), cov_inv)

    return distance

def get_original_data(original_data_path_list):
    original_data_list = []
    for original_data_path in original_data_path_list:
        print(original_data_path)
        with open(original_data_path, 'r') as f:
            if original_data_path.endswith(".json"):
                instances = json.load(f)
                for instance in instances:
                    embedded_data = instance['output_7b_embedding']
                    # data_array = np.array(eval(embedded_data))
                    # mean_data = np.mean(data_array, axis=0)
                    original_data_list.append(eval(embedded_data))
            elif original_data_path.endswith(".txt"):
                # for line in f:
                #     line = eval(line.strip())
                #     original_data_list.append(line)
                for i, line in enumerate(f):
                    if (i + 1) % 1 == 0:
                        # print(i)
                        line = eval(line.strip())
                        original_data_list.append(line)

    return original_data_list

def is_outlier(new_data, data_mean, data_threshold):
    upper_bound = data_mean + data_threshold
    lower_bound = data_mean - data_threshold
    
    count = 0
    for i in range(len(new_data)):
        if new_data[i] > upper_bound[i] or new_data[i] < lower_bound[i]:
            count += 1
    # print("count:",count)
    return count, count > 70

def plot_distribution(target_list):
    plt.hist(target_list, bins=10)  # 设置直方图的柱子数量
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Distribution Plot')
    plt.show()

def input_trigger(target_task, original_data_path, instances, tmp_list):
    # all_data_path = ['./dataset/input_train/boolq_1_input.txt','./dataset/input_train/drop_1_input.txt','./dataset/input_train/gsm8k_1_input.txt','./dataset/input_train/mmlu_1_input.txt']
    # all_data_path = ['./dataset/input_train/mmlu/boolq_1_input.json','./dataset/input_train/mmlu/drop_1_input.json','./dataset/input_train/mmlu/gsm8k_1_input.json','./dataset/input_train/mmlu/mmlu_1_input.json']
    all_data_path = [os.path.join('./dataset/input_train/',target_task,'boolq_1.json'), os.path.join('./dataset/input_train/',target_task,'mmlu_1.json'), os.path.join('./dataset/input_train/',target_task,'drop_1.json'), os.path.join('./dataset/input_train/',target_task,'gsm8k_1.json')]
    original_data = get_original_data(original_data_path)
    estimator = DistributionEstimator(original_data)

    # print(all_data_path)

    open_data_path = []
    for path in all_data_path:
        if target_task not in os.path.basename(path):
            # print(path)
            open_data_path.append(path)
    # print(open_data_path)
    # input()
    open_data = get_original_data(open_data_path)
    estimator1 = DistributionEstimator(open_data)
    print("bound分别是：", estimator.bound, estimator1.bound)

    
    in_scope_num = 0
    out_of_scope_num = 0
    distance_list = []
    open_distance_list = []
    
    
    for i in range(len(instances)):
        embedded_data = instances[i]['output_7b_embedding']
        new_data = eval(embedded_data)
        data_mean, distance = estimator.calculate_mahalanobis_distance(new_data)
        open_data_mean, open_data_distance = estimator1.calculate_mahalanobis_distance(new_data)
        distance_list.append(distance)
        open_distance_list.append(open_data_distance)
        if estimator.bound/estimator1.bound > 1.5 or estimator.bound/estimator1.bound < 0.7:
            if distance / (estimator.bound/estimator1.bound) > open_data_distance:
                print('scale')
                tmp_list[i] = 1
                out_of_scope_num +=1
            else:
                in_scope_num +=1
        else:
            if distance  > open_data_distance:
                tmp_list[i] = 1
                out_of_scope_num +=1
            else:
                in_scope_num +=1

    print("距离目标任务均值{}方差{}".format(np.mean(distance_list),np.var(distance_list)))
    print("距离开放任务均值{}方差{}".format(np.mean(open_distance_list),np.var(open_distance_list)))
    print("相对方法，训练集：",original_data_path)
    print("总共是{}个数据，其中{}个在范围内，{}个不在范围内。".format(in_scope_num + out_of_scope_num, in_scope_num, out_of_scope_num))
    print(np.mean(distance_list[-300:]),np.mean(distance_list[:-300]))

    return tmp_list
    


if __name__ == "__main__":
# 原始数据集，每个数据是一个包含4096个元素的列表
    # original_data_path = ['./dataset/input_train/drop_1_input.txt','./dataset/input_train/drop_2_input.txt']
    all_data_path = ['./dataset/input_train/boolq_1_input.txt','./dataset/input_train/drop_1_input.txt','./dataset/input_train/gsm8k_1_input.txt','./dataset/input_train/mmlu_1_input.txt']
    original_data_path = ['./dataset/input_train/boolq_1_input.txt','./dataset/input_train/boolq_2_input.txt']
    new_data_path_list = ['./dataset/input_dev/boolq_input.txt','./dataset/input_dev/mmlu_input.txt','./dataset/input_dev/drop_input.txt','./dataset/input_dev/gsm8k_input.txt']

    input_trigger(original_data_path,new_data_path_list)

    
    # original_data = get_original_data(original_data_path)
    # estimator = DistributionEstimator(original_data)
    
    # open_data_path = all_data_path
    # # open_data_path = []
    # # for path in all_data_path:
    # #     if path not in original_data_path:
    # #         # print(path)
    # #         open_data_path.append(path)
    # open_data = get_original_data(open_data_path)
    # estimator1 = DistributionEstimator(open_data)

    # print("bound分别是：", estimator.bound, estimator1.bound)




    

    # # new_data_path_list = ['./dataset/input_train/boolq_2_input.txt','./dataset/input_train/drop_2_input.txt','./dataset/input_train/gsm8k_2_input.txt','./dataset/input_train/mmlu_2_input.txt']
    # new_data_path_list = ['./dataset/input_dev/boolq_input.txt','./dataset/input_dev/mmlu_input.txt','./dataset/input_dev/drop_input.txt','./dataset/input_dev/gsm8k_input.txt']
    # # new_data_path_list = ['./dataset/input_train/drop_1_input.txt','./dataset/input_train/drop_2_input.txt']
    # # new_data_path_list = ['./dataset/input_train/gsm8k_1_input.txt','./dataset/input_train/gsm8k_2_input.txt']
    # for new_data_path in new_data_path_list:
    #     in_scope_num = 0
    #     out_of_scope_num = 0
    #     count_list = []
    #     distance_list = []
    #     open_distance_list = []
    # # new_data_path = './dataset/input_train/boolq_1_input.txt'
    #     with open(new_data_path, 'r') as g:
    #         if new_data_path.endswith(".json"):
    #             new_instances = json.load(g)
    #             for instance in new_instances:
    #                 embedded_data = instance['input_embed_boolq_hp_1160']
    #                 new_data = eval(embedded_data)
    #                 data_mean, distance = estimator.calculate_mahalanobis_distance(new_data)
    #                 open_data_mean, open_data_distance = estimator1.calculate_mahalanobis_distance(new_data)
    #                 # print("马氏距离1:", distance)
    #                 # distance2 = calculate_mahalanobis_distance(original_data, new_data)
    #                 # print("马氏距离2:", distance2)
    #                 # is_outlier = np.any(data_mean > estimator.threshold)
    #                 # print(is_outlier)
    #                 distance_list.append(distance)
    #                 open_distance_list.append(open_data_distance)

    #                 # if distance < 290:
    #                 # print(distance,open_data_distance)
    #                 if distance / (estimator.bound/estimator1.bound) < open_data_distance:
    #                     in_scope_num += 1
    #                     # _is_outlier = is_outlier(new_data, data_mean, estimator.threshold)
    #                     # if not _is_outlier:
    #                 else:
    #                     out_of_scope_num += 1

    #         elif new_data_path.endswith(".txt"):
    #             for line in g:
    #                 new_data = eval(line.strip())
    #                 data_mean, distance = estimator.calculate_mahalanobis_distance(new_data)
    #                 open_data_mean, open_data_distance = estimator1.calculate_mahalanobis_distance(new_data)

    #                 # print("马氏距离1:", distance)
    #                 # distance2 = calculate_mahalanobis_distance(original_data, new_data)
    #                 # print("马氏距离2:", distance2)
    #                 distance_list.append(distance)
    #                 open_distance_list.append(open_data_distance)

                    

    #                 # if distance < 290:
    #                 # print(distance,open_data_distance)
    #                 if distance / (estimator.bound/estimator1.bound) < open_data_distance:
    #                     in_scope_num += 1
    #                     # _is_outlier = is_outlier(new_data, data_mean, estimator.threshold)
    #                     # if not _is_outlier:
    #                 else:
    #                     out_of_scope_num += 1

    #                     # count_num, _is_outlier = is_outlier(new_data, data_mean, estimator.threshold)
    #                     # count_list.append(count_num)
    #                     # if not _is_outlier:
    #                     #     in_scope_num += 1
    #                     # else:
    #                     #     out_of_scope_num += 1

    #         # print("均值{}方差{}".format(np.mean(count_list),np.var(count_list)))
    #         # plot_distribution(count_list)
    #         print("距离目标任务均值{}方差{}".format(np.mean(distance_list),np.var(distance_list)))
    #         print("距离开放任务均值{}方差{}".format(np.mean(open_distance_list),np.var(open_distance_list)))
    #         print("相对方法，训练集：",original_data_path)
    #         print("{}总共是{}个数据，其中{}个在范围内，{}个不在范围内。".format(new_data_path, in_scope_num + out_of_scope_num, in_scope_num, out_of_scope_num))


