import re
import json
import os
import sys
sys.path.append(os.getcwd())

from functools import reduce
from typing import Union, List
# from utils.data_utils import load_jsons
from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained("./llama-7b-chat-hf")

import numpy as np
import pandas as pd

import matplotlib                                                                                                         
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.model_selection import cross_val_score


from evaluation import call_simple_parser,call_simple_parser_2
from itertools import groupby

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

import logging
logger = logging.getLogger(__name__)

def load_jsons(file_path):
    with open(file_path, 'r') as f:
        instances = json.load(f)
    return instances

def get_logprob(instances):
    logprob_list = []
    token_list = []
    for instance in instances:
        # index = instance['index']
        # print(len(instance['output']))
        logprob_list.append(eval(instance['output_7b_prob']))
        token_list.append(eval(instance['output_7b_tokenize']))
    return logprob_list, token_list

def gsm8k_importance(token_list,important_list):
    for i, token in enumerate(token_list):
        if '<<' in token:
            start_index = i
        elif '>>' in token:
            end_index = i
            for j in range(start_index-6 , end_index+1):
                important_list[j] = 1
        elif '####' in token:
            important_list[int(i+2):] = [1] * len(important_list[int(i+2):])
            break
        # logprob_list = [logprob_list[k] * important_list[k] for k in range(len(logprob_list))]
    # print(important_list)
    return important_list

def get_important(logprob_list,token_list,path):
    for idx in range(len(logprob_list)):
        while str(logprob_list[idx][0]) == '-0.0':
            logprob_list[idx].pop(0)

        important_list = [0] * len(logprob_list[idx])
        token_list[idx] = token_list[idx][-len(logprob_list[idx]):]
        assert len(logprob_list[idx]) == len(token_list[idx]) == len(important_list)

        if 'gsm8k' in path:   
            important_list = gsm8k_importance(token_list[idx],important_list)
        elif 'mmlu' in path or 'boolq' in path:
            important_list[0] = 1
        elif 'drop' in path:
            important_list = [1] * len(important_list)
        logprob_list[idx] = [logprob_list[idx][i] * important_list[i] for i in range(len(logprob_list[idx]))] 

    return logprob_list

def partition_extraction(logprob_list):
    result = []
    start_index = None
    for i, num in enumerate(logprob_list):
        if num != 0 and start_index is None:
            start_index = i
        elif num == 0 and start_index is not None:
            result.append(logprob_list[start_index:i])
            start_index = None
    if start_index is not None:
        result.append(logprob_list[start_index:])


    return result

def list_partition(logprob_list, gt_list, concat):
    new_logprob_list = []
    new_gt_list = []
    idx_list = []
    for idx in range(len(logprob_list)):
        # print("original:", logprob_list[idx])
        # input()
        tmp = partition_extraction(logprob_list[idx])
        # print("partition:", tmp)
        res = []
        for tmp_item in tmp:
            if concat:
                res.extend(tmp_item)
                new_logprob_list.append(res)
            else:
                new_logprob_list.append(tmp_item)
            new_gt_list.append(gt_list[idx])
            idx_list.append(idx)

        # print(new_logprob_list)
        # print(new_gt_list)
        # print(idx_list)
        
    return new_logprob_list, new_gt_list, idx_list

def get_logprob_feature(logprob_list):
    logprob_avg = []
    logprob_std = []
    logprob_max = []
    logprob_min = []
    logprob_sum = []
    for logprob in logprob_list:
        # print(len(logprob))
        logprob_avg.append(np.mean(logprob))
        logprob_std.append(np.std(logprob))
        logprob_max.append(max(logprob))
        logprob_min.append(min(logprob))
        logprob_sum.append(sum(logprob))
    return logprob_avg, logprob_std, logprob_max, logprob_min, logprob_sum

def split_right_and_wrong_ans(gt_list, feature_list):
    right_index = [i for i in range(len(gt_list)) if gt_list[i] == 1]
    false_index = [i for i in range(len(gt_list)) if gt_list[i] == 0]
    right_list = [feature_list[i] for i in range(len(gt_list)) if gt_list[i] == 1]
    false_list = [feature_list[i] for i in range(len(gt_list)) if gt_list[i] == 0]

    plt.scatter(right_index, right_list , c = 'red')
    plt.scatter(false_index, false_list, c = 'green')
    plt.savefig("../copilot_utilities/gsy/drop.png")

    # bins = np.linspace(-5,5,21)
    # plt.hist(right_list)
    # plt.savefig("../copilot_utilities/gsy/test_hist_right_yyt.png")

    # plt.hist(false_list)
    # plt.savefig("../copilot_utilities/gsy/test_hist_false_yyt.png")
    # plt.show()

def split_right_and_wrong_feature(gt_list, feature_list_1, feature_list_2):
    right_index = [i for i in range(len(gt_list)) if gt_list[i] == 1]
    false_index = [i for i in range(len(gt_list)) if gt_list[i] == 0]
    right_list_1 = [feature_list_1[i] for i in range(len(gt_list)) if gt_list[i] == 1]
    false_list_1 = [feature_list_1[i] for i in range(len(gt_list)) if gt_list[i] == 0]
    right_list_2 = [feature_list_2[i] for i in range(len(gt_list)) if gt_list[i] == 1]
    false_list_2 = [feature_list_2[i] for i in range(len(gt_list)) if gt_list[i] == 0]

    plt.scatter(right_list_1, right_list_2, c = 'red')
    plt.scatter(false_list_1, false_list_2, c = 'green')
    plt.savefig("../copilot_utilities/gsy/drop_mean_min.png")

def get_logprob_feature_without_instruction(logprob_list,logprob_instances):
    logprob_avg = []
    logprob_std = []
    logprob_max = []
    logprob_min = []
    logprob_sum = []
    logprob_mult = []
    logprob_first = []
    for i in range(len(logprob_list)):
        logprob = logprob_list[i]
        tokenized_length = tokenizer.encode(logprob_instances[i]["instruction"])
        # print(len(tokenized_length))
        # input()

        filter_zero_list = logprob[len(tokenized_length):]
        print(len(logprob))
        print(len(tokenized_length))
        print(len(filter_zero_list))
        # filter_zero_list = [i for i in logprob if i != 0]
        logprob_avg.append(np.mean(filter_zero_list))
        logprob_std.append(np.std(filter_zero_list))
        logprob_max.append(max(filter_zero_list))
        logprob_min.append(min(filter_zero_list))
        logprob_sum.append(sum(filter_zero_list))
        logprob_mult.append(reduce((lambda x, y: x * y), filter_zero_list))
        logprob_first.append(np.mean(filter_zero_list[0:10]))
    
    return logprob_avg, logprob_std, logprob_max, logprob_min, logprob_sum, logprob_mult, logprob_first

def get_logprob_feature_remove_zero(logprob_list):
    logprob_avg = []
    logprob_std = []
    logprob_max = []
    logprob_min = []
    logprob_sum = []
    logprob_mult = []
    logprob_first = []
    logprob_random = []
    logprob_per_25 = []
    logprob_per_50 = []
    logprob_per_75 = []
    logprob_len = []
    id = 0
    for logprob in logprob_list:
        # print(logprob)
        # print(id)

        # while str(logprob[0]) == '-0.0':
        #     logprob.pop(0)
        # print(len(tokenized_length))
        # input()
        # print(len(logprob))
        # print(logprob[0])
        # print(len(tokenized_length))
        # print(len(filter_zero_list))
        # filter_zero_list = [i for i in logprob if i != 0]
        logprob_avg.append(np.mean(logprob))
        logprob_std.append(np.std(logprob))
        # print(id,np.mean(logprob),np.std(log      prob))
        logprob_max.append(max(logprob))
        logprob_min.append(min(logprob))
        logprob_sum.append(sum(logprob))
        logprob_mult.append(reduce((lambda x, y: x * y), logprob))
        logprob_first.append(logprob[0])
        # logprob_random.append(len([j for j in logprob if j < -0.1]))
        logprob_random.append(np.percentile(logprob, 5))
        logprob_per_25.append(np.percentile(logprob, 25))
        logprob_per_50.append(np.percentile(logprob, 50))
        logprob_per_75.append(np.percentile(logprob, 75))
        logprob_len.append(len(logprob))
        # print(len([j for j in logprob if j < -0.1]))
        id += 1
    
    return logprob_avg, logprob_std, logprob_max, logprob_min, logprob_sum, logprob_mult, logprob_first, logprob_random, logprob_per_25, logprob_per_50, logprob_per_75, logprob_len

def get_gt(instances_length,instances):
    gt_list = [1 for i in range(len(instances_length))]
    # print(len(gt_list))
    # for instance in instances:
    #     gt_list[instance['index']] = 0

    for i in range(len(instances)):
        if instances_length[i]["output_7b"] != instances[i]["output"]:
            gt_list[i] = 0
    return gt_list

def get_gt_gsm8k(instances_length,truths):
    gt_list = [1 for i in range(len(instances_length))]
    # print(len(gt_list))
    # for instance in instances:
    #     gt_list[instance['index']] = 0

    predictions = call_simple_parser(instances_length)

    for k in range(len(truths)):
        # print(predictions[k]["pred_answers"])
        # print(truths[k]["output"])
        # print(predictions[i]["pred_answers"] != instances[i]["solveResult"])
        # input()
        if isinstance(eval(truths[k]["solveResult"]),list):
            if not set(eval(truths[k]["solveResult"])) <= set(eval(predictions[k]["pred_answers"])):
                gt_list[k] = 0
        else:
            if not eval(truths[k]["solveResult"]) in eval(predictions[k]["pred_answers"]):
                gt_list[k] = 0
        # if predictions[i]["pred_answers"] != instances[i]["solveResult"]:
        #     gt_list[i] = 0
        # print(gt_list[k])
        # input()
    return gt_list

def get_cloud_gt(instances_length,instances):
    gt_list = [1 for i in range(len(instances_length))]
    # print(len(gt_list))
    # for instance in instances:
    #     gt_list[instance['index']] = 0

    for i in range(len(instances)):
        if instances_length[i]["output"] != instances[i]["output"]:
            gt_list[i] = 0
    return gt_list

def get_cloud_gt_gsm8k(instances_length,truths):
    gt_list = [1 for i in range(len(instances_length))]
    # print(len(gt_list))
    # for instance in instances:
    #     gt_list[instance['index']] = 0

    predictions = call_simple_parser_2(instances_length)

    for k in range(len(truths)):
        # print(predictions[k]["pred_answers"])
        # print(truths[k]["output"])
        # print(predictions[i]["pred_answers"] != instances[i]["solveResult"])
        # input()
        if isinstance(eval(truths[k]["solveResult"]),list):
            if not set(eval(truths[k]["solveResult"])) <= set(eval(predictions[k]["pred_answers"])):
                gt_list[k] = 0
        else:
            if not eval(truths[k]["solveResult"]) in eval(predictions[k]["pred_answers"]):
                gt_list[k] = 0
        # if predictions[i]["pred_answers"] != instances[i]["solveResult"]:
        #     gt_list[i] = 0
        # print(gt_list[k])
        # input()
    return gt_list


def get_f1(instances_length,instances):
    f1_list = [1 for i in range(len(instances_length))]
    # print(len(f1_list))
    for instance in instances:
        # print(instance['index'])
        f1_list[instance['index']] = instance['f1']
    # for id in range(len(instances)): 
    #     gt_list[id] = 0
    # print(gt_list)
    return f1_list

def get_edge_cloud_ans(lista, listb, list_simulator):
    ans = []
    list1 = lista[-len(list_simulator):]
    list2 = listb[-len(list_simulator):]
    print("dddd",len(list1),len(list2),len(list_simulator))
    for i in range(len(list_simulator)):
        if list_simulator[i] == 1:
            ans.append(list1[i])
        else:
            ans.append(list2[i])
    print("edge acc", list1.count(1)/len(list_simulator))
    print("cloud acc", list2.count(1)/len(list_simulator))
    return ans

def partition_combination(res_list, id_list):
    result = []
    for key, group in groupby(zip(id_list, res_list), key=lambda x: x[0]):
        # print(key)
        group_list = list(group)
        # print(group_list)
        # input()
        if any(x[1] == 0 for x in group_list):
            result.append(0)
        else:
            result.append(1)
        # print(result)
        # input()
    return result

def partition_true_and_false(logprob_list, gt_list):
    list_0 = []
    list_1 = []
    for idx in range(len(gt_list)):
        if gt_list[idx] == 1:
            list_1.append(logprob_list[idx])
        elif gt_list[idx] == 0:
            list_0.append(logprob_list[idx])
        else:
            print("wrong")
    flat_list_0 = [item for sublist in list_0  for item in sublist]
    flat_list_1 = [item for sublist in list_1  for item in sublist]

    min_length = min(len(flat_list_0),len(flat_list_1))
    x = range(1, min_length+1)
    plt.plot(x, flat_list_1[:min_length], label='List 1')
    plt.plot(x, flat_list_0[:min_length], label='List 0')
    
    plt.legend()
    plt.savefig('line_plot')
    print("回答错误的均值", np.mean(flat_list_0))
    print("回答正确的均值", np.mean(flat_list_1))

    return flat_list_0, flat_list_1

def partition_true_and_false_feature(logprob_list, gt_list):
    list_0 = []
    list_1 = []
    for idx in range(len(gt_list)):
        if gt_list[idx] == 1:
            list_1.append([np.mean(logprob_list[idx])])
        elif gt_list[idx] == 0:
            list_0.append([np.mean(logprob_list[idx])])
        else:
            print("wrong")
    flat_list_0 = [item for sublist in list_0  for item in sublist]
    flat_list_1 = [item for sublist in list_1  for item in sublist]

    min_length = min(len(flat_list_0),len(flat_list_1))
    x = range(1, min_length+1)
    plt.plot(x, flat_list_1[:min_length], label='List 1')
    plt.plot(x, flat_list_0[:min_length], label='List 0')
    
    plt.legend()
    plt.savefig('line_plot_1')
    print("回答错误的均值及数量", np.mean(flat_list_0), len(flat_list_0))
    print("回答正确的均值及数量", np.mean(flat_list_1), len(flat_list_1))

    return list_0, list_1

# def output_trigger_verb(instances,tmp_list):
#     for i in range(len(tmp_list)):
#         if tmp_list[i] == 1:
#             try: 
#                 truth_ans = json.loads(instances[i]['output_gpt4_veb'])
#             if eval(instance['solveResult']) == eval(truth_ans['answer'].split("######")[1]):
#             # if instance['output'] == truth_ans['answer']:
#             # if eval(instance['output'].split("####")[1]) == eval(instance['answer'].split("####")[1]):
#             # if instance['output'] == instance['answer']:
#                 truth_list.append(1)
#             else:
#                 truth_list.append(0)
#             try: 
#                 if int(truth_ans['confidence']) == 1:
#                     pred_list.append(1)
#                 else:
#                     pred_list.append(0)
#             except Exception as e:
#                 print('1:',e)
#                 pred_list.append(0)
#         except Exception as e:
#             print('2:',e)
#             if str(instance['solveResult']) in str(truth_ans):
#                 print("+++")
#                 truth_list.append(1)
#                 pred_list.append(1)
#             else:
#                 print("---")
#                 truth_list.append(0)
#                 pred_list.append(random.randint(0, 1))
    
#     return tmp_list

# def output_trigger(target_task,edge_list,all_instances,tmp_list,lm=False):
        

def output_trigger(target_task,edge_list,all_instances,tmp_list,lm=False):
    
    logprob_list, token_list = get_logprob(all_instances)
    gt_list = edge_list
    
    if lm:
        logprob_list = get_important(logprob_list, token_list, target_task)
        logprob_list, gt_list, idx_list = list_partition(logprob_list, gt_list, concat=True)

    logprob_avg, logprob_std, logprob_max, logprob_min, logprob_sum, logprob_mult, logprob_first, logprob_random, logprob_per_25, logprob_per_50, logprob_per_75, logprob_len = get_logprob_feature_remove_zero(logprob_list)
    X = list(map(list, zip(logprob_avg, logprob_std, logprob_max, logprob_min, logprob_sum, logprob_mult, logprob_first, logprob_random, logprob_per_25, logprob_per_50, logprob_per_75, logprob_len))) 
    # X = list(map(list, zip(logprob_avg, logprob_sum, logprob_per_25))) 
    # X = list(map(list, zip(logprob_avg)))
    # X = list(map(list, zip(logprob_sum)))
    # X = list(map(list, zip(logprob_per_75)))
    # X = list(map(list, zip(logprob_avg, logprob_sum)))
    y = gt_list

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.5, shuffle=False)
    # if target_task == "gsm8k":
    #     clf = DecisionTreeClassifier(max_depth=2000,random_state=49)
    # else:
    clf = MLPClassifier(hidden_layer_sizes=(1024,), max_iter=2000, random_state=421)
    # clf = LogisticRegression(random_state=42)
    
    clf = clf.fit(Xtrain,Ytrain)
    Ypredict = clf.predict(Xtest)

    if lm:
        gt_list, idx_list = gt_list[-len(Ytest):], idx_list[-len(Ytest):]
        gt_list = partition_combination(gt_list, idx_list)
        Ypredict = partition_combination(Ypredict, idx_list)
        Ytest = partition_combination(Ytest, idx_list)
        Xtest = partition_combination(Xtest, idx_list)

    print("output-level trigger:") 
    print(classification_report(Ytest, Ypredict, target_names=["cloud","edge"],digits=4))


    new_list = tmp_list[:-(len(Ypredict))] + [1 if tmp_list[i]== 1 else 1 if tmp_list[i] == 0 and Ypredict[i - (len(tmp_list) - len(Ypredict))] == 0 else 0 for i in range(len(tmp_list) - len(Ypredict), len(tmp_list))]
    # for i in range(len(tmp_list) - len(Ypredict), len(tmp_list)):
    #     print(tmp_list[i],Ypredict[i - (len(tmp_list) - len(Ypredict))])
    #     if tmp_list[i] == 1:
    #         print("yes")
    #     elif tmp_list[i] == 0 and Ypredict[i - (len(tmp_list) - len(Ypredict))] == 0:
    #         print("yes")
    # print(len(new_list))
    return new_list
    
    # edge_cloud_list = get_edge_cloud_ans(gt_list,cloud_list,Ypredict)
    # print("edge-cloud acc", edge_cloud_list.count(1), edge_cloud_list.count(1)/len(Ypredict))
    # print("edge invoke",list(Ypredict).count(1))
    # print("cloud invoke",len(Ypredict) - list(Ypredict).count(1))



if __name__ == '__main__':
    
    ##### step 1: get instance #####
    # logprob_file_path = "./dataset/prob_13b/boolq.json"
    logprob_file_path = "./dataset/7b/gsm8k_chat_ep3_prob.json"
    logprob_instances = load_jsons(logprob_file_path)
    logprob_list, token_list = get_logprob(logprob_instances)
    
    # logprob_list = get_important(logprob_list, token_list, logprob_file_path)

    # print(logprob_list)
    # print(logprob_list_1)
    # print(logprob_list == logprob_list_1)
    # print(logprob_list[1],logprob_list[10],len(logprob_list[20]),len(logprob_list[80]))

    # gt_file_path = "../datasets/ecc/en_dev_99_cn/70b/s42_2of2_error.json"
    gt_file_path = "./dataset/dev/gsm8k_dev.json"
    gt_instances = load_jsons(gt_file_path) 
    if 'gsm8k' in gt_file_path:
        gt_list = get_gt_gsm8k(logprob_instances,gt_instances)
    else:  
        gt_list = get_gt(logprob_instances,gt_instances)
        # gt_list = get_f1(logprob_instances,gt_instances)
    


    # logprob_list, gt_list, idx_list = list_partition(logprob_list, gt_list, concat=True)

    # print(len(logprob_list[0]),logprob_list[0])
    # print(len(logprob_list[5]),logprob_list[5])
    # list_0, list_1 = partition_true_and_false_feature(logprob_list,gt_list)
    # input()

    
    


    # print(logprob_list)
    # print(gt_list)
    ##### step 2: extract feature #####
    # logprob_avg, logprob_std, logprob_max, logprob_min, logprob_sum = get_logprob_feature(logprob_list)
    logprob_avg, logprob_std, logprob_max, logprob_min, logprob_sum, logprob_mult, logprob_first, logprob_random, logprob_per_25, logprob_per_50, logprob_per_75, logprob_len = get_logprob_feature_remove_zero(logprob_list)
    # logprob_avg, logprob_std, logprob_max, logprob_min, logprob_sum, logprob_mult, logprob_first = get_logprob_feature_without_instruction(logprob_list,logprob_instances)

    
    ##### step 3: machine learning #####
    # X = list(map(list, zip(logprob_sum))) 
    # X = list(map(list, zip(logprob_avg, logprob_std, logprob_max, logprob_min, logprob_sum, logprob_mult, logprob_first, logprob_random, logprob_per_25, logprob_per_50, logprob_per_75, logprob_len))) 
    # X = list(map(list, zip(logprob_avg, logprob_std, logprob_max, logprob_min, logprob_sum, logprob_mult, logprob_first, logprob_random, logprob_per_25, logprob_per_50, logprob_per_75, logprob_len))) 
    X = list(map(list, zip(logprob_avg, logprob_sum, logprob_per_25,logprob_per_50))) 
    y = gt_list


    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.5, shuffle=False)
 
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.svm import SVC
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.neural_network import MLPClassifier

    # clf = SVC(kernel='linear', probability=True)
    clf = MLPClassifier(hidden_layer_sizes=(1000,), max_iter=2000, random_state=421)
    # clf = LogisticRegression(random_state=42)
    # clf = DecisionTreeClassifier(max_depth=2000,random_state=49)
    clf = clf.fit(Xtrain,Ytrain)
    Ypredict = clf.predict(Xtest)
    # Yprob = clf.predict_proba(Xtest)[:,1]
    # print(Yprob)

    # precision, recall, thresholds = precision_recall_curve(Ytest,Yprob)
    # # print(precision,recall, thresholds)
    # # area = auc(precision,recall)
    # # print("AUC:",area)
    # # input()
    # plt.plot(recall, precision)
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title('Precision-Recall Curve')
    # plt.savefig('PR')

    # gt_list, idx_list = gt_list[-len(Ytest):], idx_list[-len(Ytest):]
    # gt_list = partition_combination(gt_list, idx_list)
    # Ypredict = partition_combination(Ypredict, idx_list)
    # # len(Ytest)
    # # len(Ypredict)
    # Ytest = partition_combination(Ytest, idx_list)
    # Xtest = partition_combination(Xtest, idx_list)

    # input()



    # importances = clf.feature_importances_
    # print(importances)

    # print(Ypredict)
    # print(Ytest)
    
    ##### step 4: evaluation #####
    # ACC
    # score_c = clf.score(Xtest,Ytest)
    # print(score_c)
    print('Accuracy:{:.3f}'.format(accuracy_score(Ytest, Ypredict))) 
    # SEN PRE F1
    pre= classification_report(Ytest, Ypredict, target_names=["false","true"],digits=4) 
    print(pre)

    ##### step 5: draw pic #####
    # split_right_and_wrong_ans(gt_list, logprob_avg)
    # split_right_and_wrong_feature(gt_list, logprob_avg, logprob_min)

    # print(logprob_avg)
    # print(logprob_std)
    # print(logprob_max)
    # print(logprob_min)
    # print(logprob_sum)
    # print(gt_list)

    ##### step 6: analysis #####
    # s1 = pd.Series(logprob_min)
    # s2 = pd.Series(gt_list)
    # corr = s2.corr(s1)
    # print(corr)

    ##### step 7: edge-cloud #####

    cloud_file_path = "./dataset/prob_70b/gsm8k.json"
    cloud_instances = load_jsons(cloud_file_path)
    # gt_file_path = "./dataset/dev/boolq_dev.json"
    # gt_instances = load_jsons(gt_file_path) 
    if 'gsm8k' in cloud_file_path:
        cloud_list = get_cloud_gt_gsm8k(cloud_instances,gt_instances)
    else:
        cloud_list = get_cloud_gt(cloud_instances,gt_instances)

    edge_cloud_list = get_edge_cloud_ans(gt_list,cloud_list,Ypredict)
    print("edge-cloud acc", edge_cloud_list.count(1), edge_cloud_list.count(1)/len(Ypredict))
    print("edge invoke",list(Ypredict).count(1))
    print("cloud invoke",len(Ypredict) - list(Ypredict).count(1))

