import os
import json
import pandas as pd
import requests
import re
# import sympy
# from sympy import Symbol, solve
import prettytable as pt
import sys

sys.path.append(os.getcwd())


import logging
logger = logging.getLogger(__name__)


def call_solver(predictions):
    predictions = call_simple_parser(predictions)

    return predictions

def call_metric(predictions, truths):
    correct_list = []
    false_list = []
    for k in range(len(predictions)):
        try:
            truths[k]["index"] = predictions[k]["index"]
            truths[k]["output"] = predictions[k]["output"]
            truths[k]["pred_answers"] = predictions[k]["pred_answers"]

            if isinstance(eval(truths[k]["solveResult"]),list):
                if set(eval(truths[k]["solveResult"])) <= set(eval(predictions[k]["pred_answers"])):
                    correct_list.append(truths[k])
                else:
                    false_list.append(truths[k])
            else:
                if eval(truths[k]["solveResult"]) in eval(predictions[k]["pred_answers"]):
                    correct_list.append(truths[k])
                else:
                    false_list.append(truths[k])

        except Exception as e:
            logger.info(truths[k]['index'])
            logger.info(repr(e))
            truths[k]["output"] = ""
            truths[k]["pred_answers"] = ""
            false_list.append(truths[k])

    return correct_list, false_list

def call_simple_parser(instances):
    for index, instance in enumerate(instances):
        output = instance["output_7b"]
        if "####" in output:
            result = output.split("####")[1].strip()
            if "The answer is:" in result:
                result = result.split("The answer is:")[0].strip()
        elif "The answer is:" in output:
            result = output.split("The answer is:")[1].strip()
        else:
            result = ""
        if not re.match(r"^[0-9\.\/-]+$", result):
            # print("{}: cannot parse:\n{}".format(index, output))
            result = []
        else:
            result = [eval(result)]
        instance["pred_answers"] = str(result)

    return instances

def call_simple_parser_2(instances):
    for index, instance in enumerate(instances):
        output = instance["output"]
        if "####" in output:
            result = output.split("####")[1].strip()
            if "The answer is:" in result:
                result = result.split("The answer is:")[0].strip()
        elif "The answer is:" in output:
            result = output.split("The answer is:")[1].strip()
        else:
            result = ""
        if not re.match(r"^[0-9\.\/-]+$", result):
            # print("{}: cannot parse:\n{}".format(index, output))
            result = []
        else:
            result = [eval(result)]
        instance["pred_answers"] = str(result)

    return instances

def call_simple_parser_key(instances,key):
    for index, instance in enumerate(instances):
        output = instance[key]
        if "####" in output:
            result = output.split("####")[1].strip()
            if "The answer is:" in result:
                result = result.split("The answer is:")[0].strip()
        elif "The answer is:" in output:
            result = output.split("The answer is:")[1].strip()
        else:
            result = ""
        if not re.match(r"^[0-9\.\/-]+$", result):
            # print("{}: cannot parse:\n{}".format(index, output))
            result = []
        else:
            result = [eval(result)]
        instance["pred_answers"] = str(result)

    return instances

def load_jsons(file_path):
    with open(file_path, 'r') as f:
        instances = json.load(f)
    return instances
    
if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # load predictions
    prediction_path_list = "./dataset/1b/gsm8k_chat_ep3.json"
    predictions = load_jsons(prediction_path_list)
    
    # load ground truths
    truth_path_list = ["./dataset/dev/gsm8k_dev.json"]
   
    index_start = 0
    index_end = 0
    emerge_error_logs = []
    dev_name=[]
    dev_acc=[]
    map_dict = {
        "./dataset/dev/gsm8k_dev.json":"gsm8k",
    }
    
    for file_index, truth_path in enumerate(truth_path_list):
        truths = load_jsons(truth_path)
        index_start = index_end
        index_end = index_start + len(truths)
        print(index_start,index_end)
        answers = predictions[index_start:index_end]
       
    
        logger.info('Calling solver...')
        answers = call_solver(answers)
        correct_list, false_list = call_metric(answers, truths)
        

        logger.info('Answers and truths loaded. Calculating metric...')
        logger.info('{}'.format(truth_path))
        logger.info('acc: {}'.format(len(correct_list)/len(truths)))
        logger.info('true_num: {}'.format(len(correct_list)))
        logger.info('error_num: {}'.format(len(false_list)))
        error_logs = false_list
        dev_name.append(map_dict[truth_path])
        dev_acc.append("%.2f%%" % (len(correct_list)/len(truths)*100))

        
    if len(prediction_path_list) == len(truth_path_list):
        error_log_name = "error_log.json"
        error_path = os.path.join(os.path.dirname(prediction_path_list[file_index]), error_log_name)
        logger.info('Save error log to {}.'.format(error_path))
        with open(error_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(error_logs, indent=2, ensure_ascii=False))
    
    if len(prediction_path_list) != len(truth_path_list):
        error_log_name = "error_log.json"
        error_path = os.path.join(os.path.dirname(prediction_path_list[0]), error_log_name)
        logger.info('Save error log to {}.'.format(error_path))
        with open(error_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(emerge_error_logs, indent=2, ensure_ascii=False))

    logger.info('eval over')
    table = pt.PrettyTable(dev_name)
    table.add_row(dev_acc)                                                                                                                                                                                      
    print(table)
