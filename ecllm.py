import os
import sys

import json
import numpy as np

from evaluation import call_simple_parser_key
from trigger_input import input_trigger
from trigger_output import output_trigger
from sklearn.metrics import classification_report

def load_jsons(file_path):
    with open(file_path, 'r') as f:
        instances = json.load(f)
    return instances

def get_edge_list(instances):
    gt_list = [1 for i in range(len(instances))]
    for i in range(len(instances)):
        if instances[i]["output_7b"] != instances[i]["output"]:
            gt_list[i] = 0
    return gt_list

def get_edge_list_gsm8k(instances):
    gt_list = [1 for i in range(len(instances))]

    predictions = call_simple_parser_key(instances,'output_7b')

    for k in range(len(instances)):
        if isinstance(eval(instances[k]["solveResult"]),list):
            if not set(eval(instances[k]["solveResult"])) <= set(eval(predictions[k]["pred_answers"])):
                gt_list[k] = 0
        else:
            if not eval(instances[k]["solveResult"]) in eval(predictions[k]["pred_answers"]):
                gt_list[k] = 0
    return gt_list

def get_cloud_list(instances):
    gt_list = [1 for i in range(len(instances))]

    for i in range(len(instances)):
        if instances[i]["output"] != instances[i]["cloud"]:
            gt_list[i] = 0
    return gt_list

def get_cloud_list_gsm8k(instances):
    gt_list = [1 for i in range(len(instances))]

    predictions = call_simple_parser_key(instances,'cloud')

    for k in range(len(instances)):
        if isinstance(eval(instances[k]["solveResult"]),list):
            if not set(eval(instances[k]["solveResult"])) <= set(eval(predictions[k]["pred_answers"])):
                gt_list[k] = 0
        else:
            if not eval(instances[k]["solveResult"]) in eval(predictions[k]["pred_answers"]):
                gt_list[k] = 0
    return gt_list

def get_edge_cloud_ans(lista, listb, list_simulator):
    ans = []
    for i in range(len(list_simulator)):
        if list_simulator[i] == 1:
            ans.append(listb[i])
        else:
            ans.append(lista[i])
    print("edge acc", lista.count(1)/len(list_simulator))
    print("cloud acc", listb.count(1)/len(list_simulator))
    return ans


def get_upper_bound(lista, ans):
    for i in range(len(ans)):
        if lista[i] == 0:
            ans[i] = 1
    return ans



if __name__ == "__main__":

    ### edge_time and cloud_time is obtained by statistics in communication file

    target_task = 'gsm8k'
    edge_time = 15.12
    cloud_time = 142.78
    stop_time = 1.5

    target_task = 'drop'
    edge_time= 2.88
    cloud_time = 7.99
    stop_time = 1.5 

    target_task = 'mmlu'
    edge_time = 2.13
    cloud_time = 7.67
    stop_time = 1

    target_task = 'boolq'
    edge_time = 1.06
    cloud_time = 3.18
    stop_time = 0.9
    


    upper_bound = False
    if_input = False
    if_output = True
    input_num = 0
    output_num = 0
    print(target_task)

    id_file_path = os.path.join('./dataset/id/', f"{target_task}.json")
    id_instances = load_jsons(id_file_path)
    
    ood_file_path = os.path.join('./dataset/ood/', f"{target_task}.json")
    ood_instances = load_jsons(ood_file_path)

    target_data_path_1 = os.path.join('./dataset/input_train/',target_task, f"{target_task}_1.json")
    target_data_path_2 = os.path.join('./dataset/input_train/',target_task, f"{target_task}_2.json")

    target_data_path = [target_data_path_1,target_data_path_2]


    all_instances = id_instances + ood_instances
    request_cloud_list = [0] * len(all_instances)
    gt_cloud_list = [0] * len(id_instances) + [1] * len(ood_instances)
    
    if 'gsm8k' in id_file_path:
        cloud_list_1 = get_cloud_list_gsm8k(all_instances[:-300])
        cloud_list_2 = get_cloud_list(all_instances[-300:])
        edge_list_1 = get_edge_list_gsm8k(all_instances[:-300])
        edge_list_2 = get_edge_list(all_instances[-300:])
    else:
        cloud_list_1 = get_cloud_list(all_instances[:-100])
        cloud_list_2 = get_cloud_list_gsm8k(all_instances[-100:])
        edge_list_1 = get_edge_list(all_instances[:-100])
        edge_list_2 = get_edge_list_gsm8k(all_instances[-100:])
    
    cloud_list = cloud_list_1 + cloud_list_2
    edge_list = edge_list_1 + edge_list_2



    if upper_bound:
        request_cloud_list = get_upper_bound(edge_list,request_cloud_list)
    else:
        if if_input:
            request_cloud_list = input_trigger(target_task,target_data_path,all_instances,request_cloud_list)
            input_num = request_cloud_list.count(1)
            print(request_cloud_list)
            print("input-level trigger:")
            print(classification_report(request_cloud_list, gt_cloud_list, target_names=["edge","cloud"], digits=4))
        
        if if_output:
            request_cloud_list = output_trigger(target_task,edge_list,all_instances,request_cloud_list,lm=True)
            request_cloud_list[:600] = [request_cloud_list[i] for i in range(600, 1200)]
            edge_list[:600] = [edge_list[i] for i in range(600, 1200)]
            cloud_list[:600] = [cloud_list[i] for i in range(600, 1200)]

        output_num = request_cloud_list.count(1)


    print("overall performance:")

    edge_cloud_list = get_edge_cloud_ans(edge_list,cloud_list,request_cloud_list)
    edge_cloud_time = (request_cloud_list.count(0) * edge_time + input_num * cloud_time + (output_num-input_num) * (stop_time + cloud_time))/len(request_cloud_list)
    edge_cloud_time_posthoc = (request_cloud_list.count(0) * edge_time + input_num * cloud_time + (output_num-input_num) * (edge_time + cloud_time))/len(request_cloud_list)
    print("edge-cloud acc", edge_cloud_list.count(1)/len(request_cloud_list))
    print("edge-cloud eff", edge_cloud_time)
    print("post-hoc eff", edge_cloud_time_posthoc)
    print("cost",request_cloud_list.count(1)/len(request_cloud_list))
    print("edge invoke",request_cloud_list.count(0))
    print("cloud invoke",request_cloud_list.count(1))




