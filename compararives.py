# ### rm 
# import numpy as np
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import classification_report
# import json

# file_path = "./dataset/rm_70b/boolq.json"
# with open(file_path, 'r') as f:
#     instances = json.load(f)
#     pred_list=[]
#     truth_list=[]
#     for instance in instances:
#         if instance['output'] == 'yes':
#             pred_list.append(1)
#         else:
#             pred_list.append(0)

#         if instance['solveResult'] == 'yes':
#             truth_list.append(1)
#         else:
#             truth_list.append(0)
#     # print(pred_list)
#     # print(truth_list)

# print(classification_report(truth_list,pred_list,digits=4))


### verb 
import json
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

file_path = "./dataset/gpt/gsm8k_dev.json"
with open(file_path, 'r') as f:
    instances = json.load(f)
    pred_list=[]
    truth_list=[]
    for instance in instances:
        try: 
            truth_ans = json.loads(instance['output_gpt4_veb'])
            if eval(instance['solveResult']) == eval(truth_ans['answer'].split("######")[1]):
            # if instance['output'] == truth_ans['answer']:
            # if eval(instance['output'].split("####")[1]) == eval(instance['answer'].split("####")[1]):
            # if instance['output'] == instance['answer']:
                truth_list.append(1)
            else:
                truth_list.append(0)
            try: 
                if int(truth_ans['confidence']) == 1:
                    pred_list.append(1)
                else:
                    pred_list.append(0)
            except Exception as e:
                print('1:',e)
                pred_list.append(0)
        except Exception as e:
            print('2:',e)
            if str(instance['solveResult']) in str(truth_ans):
                print("+++")
                truth_list.append(1)
                pred_list.append(1)
            else:
                print("---")
                truth_list.append(0)
                pred_list.append(random.randint(0, 1))

    print(len(pred_list)==len(pred_list))
    # print(truth_list)
    print('Accuracy:{:.3f}'.format(accuracy_score(truth_list, pred_list))) 
    print(classification_report(truth_list,pred_list,digits=4))






# ### consistency
import json
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

file_path = "./dataset/consistency/gsm8k_2.json"
with open(file_path, 'r') as f:
    instances = json.load(f)
    pred_list=[]
    truth_list=[]
    for instance in instances:
        try: 
            # truth_ans = json.loads(instance['output_chatgpt_veb'])
            # if eval(instance['solveResult']) == eval(truth_ans['answer'].split("######")[1]):
            # if instance['output'] == truth_ans['answer']:
            # if eval(instance['output'].split("####")[1]) == eval(instance['answer'].split("####")[1]):
            if instance['output'] == instance['answer']:
                truth_list.append(1)
            else:
                truth_list.append(0)
            try: 
                if int(instance['confidence']) == 1:
                    pred_list.append(1)
                else:
                    pred_list.append(0)
            except Exception as e:
                print('1:',e)
                pred_list.append(0)
        except Exception as e:
            print()
            print('2:',e)
            truth_list.append(1)
            pred_list.append(0)

# print(pred_list)
# print(truth_list)

    print(classification_report(truth_list,pred_list,digits=4))