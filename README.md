
## EC-LLM 👋📖✨
Low-cost, Low-latency, High-quality Large Language Model Inference Based on Edge-cloud Collaboration

### **1. Quick Start**

#### - Environment Configuration 
```
pip install -r requirement.txt
```

#### - Model Download
(Customize) Find your models [here](https://huggingface.co/models) and download it.
```
./llama-70b-chat-hf/ # cloud-LLM
./llama-7b-chat-hf/ # edge-LLM
```

#### - Performance Evaluation
(Customize) Evaluate your results with modify the hyper-parameter in **ECLLM.py**.
```
python ECLLM.py
```

### 2. Finetune 
Please prepare your model first refer to **Model Download** in **Quick Start**.

#### - Dataset Preparation
(Customize) Organize your train data in the following format.
```
{
  "index": "1", # index
  "instruction": "", # prompt or user query
  "input": "", # user query
  "output": "", # answer
  "language": "en" # optional
}
```
#### - Model Finetune
Finetune your model with **finetune.sh** or with the script as below.
```
CUDA_VISIBLE_DEVICES=0,1,2,3       \
nohup python finetune.py     \
--base_model './llama-7b-chat-hf'     \
--data_path './dataset/train/boolq.json'     \
--output_dir './output/boolq_chat_ep3/' \
> ./output/boolq_chat_ep3/train.log 2>&1
```

### 3. Generation 
Please prepare your finetuned model first refer to **Finetune**.

#### - Dataset Preparation
(Customize) Organize your dev data in the following format.
```
{
  "index": "1", # index
  "instruction": "", # prompt or user query
  "input": "", # user query
  "output": "", # None
  "language": "en" # optional
}
```
#### - Model Finetune
Finetune your model with **finetune.sh** or with the script as below.
```
CUDA_VISIBLE_DEVICES=0,1    \
nohup python generate.py   \
--base_model './llama-7b-chat-hf'    \
--lora_weights './output/gsm8k_chat_ep3/checkpoint-120'  \
--data_path './dataset/dev/gsm8k_dev.json'  \
--result_path './dataset/7b/gsm8k.json' \
--load_8bit False \
> ./dataset/7b/gsm8k.log 2>&1
```

### 4. Deployement 
The docker container is used for edge-cloud collaborative deployement.
```
./Dokerfile
./Communication/edge
./communication/cloud
```
## Acknolwedgement
https://github.com/tloen/alpaca-lora

https://github.com/FreedomIntelligence/OVM
