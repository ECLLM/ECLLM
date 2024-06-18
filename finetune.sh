
CUDA_VISIBLE_DEVICES=0,1,2,3 \
nohup python finetune.py     \
--base_model './llama-7b-chat-hf'     \
--data_path './dataset/train/boolq.json'     \
--output_dir './output/boolq_chat_ep3/' \
> ./output/boolq_chat_ep3/train.log 2>&1
