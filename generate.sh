CUDA_VISIBLE_DEVICES=0,1   \
nohup python generate.py   \
--base_model './llama-7b-chat-hf'    \
--lora_weights './output/gsm8k_chat_ep3/checkpoint-120'  \
--data_path './dataset/dev/gsm8k_dev.json'  \
--result_path './dataset/7b/gsm8k.json' \
--load_8bit False \
> ./dataset/7b/gsm8k.log 2>&1

