import os
import sys

import warnings
warnings.filterwarnings("ignore")

import fire
import gradio as gr
import torch
import transformers
import numpy as np
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from tqdm import tqdm
import time

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter
from datasets import load_dataset
import json


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights: str = "tloen/alpaca-lora-7b",
    prompt_template: str = "ecllm",  # The prompt template to use, will default to alpaca.
    server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
    share_gradio: bool = False,
    data_path: str = "",
    result_path: str = "",
    output_prob: bool = False,
    input_distribution: bool = False
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    assert not (
        input_distribution and output_prob
    ), "Note that input_distribution and output_prob cannot be True simultaneously"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)


    def evaluate(
        instruction,
        input=None,
        output=None,
        temperature=0.0,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=512,
        stream_output=False,
        **kwargs,
    ):
        if output_prob:
            prompt_w = prompter.generate_prompt(instruction, input)
            prompt = prompter.generate_prompt(instruction, input, output)
        else:
            prompt = prompter.generate_prompt(instruction, input)

        # print("The content be tokenized is:\n", prompt)
        # tokens = tokenizer.tokenize(prompt)
        # ids = tokenizer.convert_tokens_to_ids(tokens)
        # print(tokens)
        # print(ids)
        
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)

        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }

        if stream_output:
            # Stream the reply 1 token at a time.
            # This is based on the trick of using 'stopping_criteria' to create an iterator,
            # from https://github.com/oobabooga/text-generation-webui/blob/ad37f396fc8bcbab90e11ecf17c56c97bfbd4a9c/modules/text_generation.py#L216-L243.

            def generate_with_callback(callback=None, **kwargs):
                kwargs.setdefault(
                    "stopping_criteria", transformers.StoppingCriteriaList()
                )
                kwargs["stopping_criteria"].append(
                    Stream(callback_func=callback)
                )
                with torch.no_grad():
                    model.generate(**kwargs)

            def generate_with_streaming(**kwargs):
                return Iteratorize(
                    generate_with_callback, kwargs, callback=None
                )

            with generate_with_streaming(**generate_params) as generator:
                for output in generator:
                    # new_tokens = len(output) - len(input_ids[0])
                    decoded_output = tokenizer.decode(output)

                    if output[-1] in [tokenizer.eos_token_id]:
                        break

                    yield prompter.get_response(decoded_output)
            return  # early return for stream_output

        # Without streaming
        with torch.no_grad():
            if input_distribution: 
                yield model.base_model.model.model.embed_tokens(input_ids).tolist()[0]
                
            elif output_prob:
                test_output = model(**inputs)
                input_tokens = tokenizer.tokenize(prompt_w)
                output_tokens = tokenizer.tokenize(prompt)
                input_tokens_len = len(input_tokens)
                output_tokens_len = len(output_tokens)
                

                logits = test_output.logits
                per_token_logps = logits.log_softmax(-1)

                max_values, _ = torch.max(per_token_logps, dim=-1)
                prob_list = max_values.squeeze().tolist()
                prob_list = prob_list[:-1]
                prob_list = [-0.0] * input_tokens_len + prob_list[input_tokens_len:]
                if not output_tokens_len == len(prob_list):
                    print("alert!!!!!")
                # logits_sum = torch.sum(per_token_logps, dim=-1, keepdim=True)
                yield output_tokens,prob_list

            else:

                generation_output = model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=max_new_tokens,
                )
                s = generation_output.sequences[0][:-1]
                output = tokenizer.decode(s)

                yield prompter.get_response(output)

        # gr.Interface(
        #     fn=evaluate,
        #     inputs=[
        #         gr.components.Textbox(
        #             lines=2,
        #             label="Instruction",
        #             placeholder="Tell me about alpacas.",
        #         ),
        #         gr.components.Textbox(lines=2, label="Input", placeholder="none"),
        #         gr.components.Slider(
        #             minimum=0, maximum=1, value=0.1, label="Temperature"
        #         ),
        #         gr.components.Slider(
        #             minimum=0, maximum=1, value=0.75, label="Top p"
        #         ),
        #         gr.components.Slider(
        #             minimum=0, maximum=100, step=1, value=40, label="Top k"
        #         ),
        #         gr.components.Slider(
        #             minimum=1, maximum=4, step=1, value=4, label="Beams"
        #         ),
        #         gr.components.Slider(
        #             minimum=1, maximum=2000, step=1, value=128, label="Max tokens"
        #         ),
        #         gr.components.Checkbox(label="Stream output"),
        #     ],
        #     outputs=[
        #         gr.inputs.Textbox(
        #             lines=5,
        #             label="Output",
        #         )
        #     ],
        #     title="🦙🌲 Alpaca-LoRA",
        #     description="Alpaca-LoRA is a 7B-parameter LLaMA model finetuned to follow instructions. It is trained on the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) dataset and makes use of the Huggingface LLaMA implementation. For more information, please visit [the project's website](https://github.com/tloen/alpaca-lora).",  # noqa: E501
        # ).queue().launch(server_name="0.0.0.0", share=share_gradio)
        # Old testing code follows.

        # if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        #     data = load_dataset("json", data_files=data_path)
        # else:
        #     data = load_dataset(data_path)

    output_list=[]
    instance_time_list=[]   
    with open(data_path) as f:
        instances = json.load(f)
        total_instances = len(instances)
        start_time = time.time()
        progress_bar = tqdm(total=total_instances, ncols=80)

        for instance in instances:
            if output_prob:
                output_tokens, prob_list = next(evaluate(instruction=instance["instruction"],input=instance["input"],output=instance["output_7b"]))
                instance["output_7b_tokenize"] = str(output_tokens)
                instance["output_7b_prob"] = str(prob_list)
                progress_bar.update(1)
                progress_bar.set_postfix()
            elif input_distribution:
                output = next(evaluate(instruction=instance["instruction"],input=instance["input"]))
                output = np.mean(np.array(output),axis=0).tolist()
                # output_list.append(output)
                instance["output_7b_embedding"] = str(output)
                # print("id {} is processed".format(instance["index"]))
                progress_bar.update(1)
                progress_bar.set_postfix()
            else:
                instance_start_time = time.time()
                output = next(evaluate(instruction=instance["instruction"],input=instance["input"]))
                instance_end_time = time.time()
                instance['output_7b'] = output
                instance_time = instance_end_time - instance_start_time
                instance_time_list.append(instance_time)
                progress_bar.update(1)
                progress_bar.set_postfix()
                # print("id {} is processed".format(instance["index"]))
        
        end_time = time.time()
        total_time = end_time - start_time

        progress_bar.close()
        print(f"\nTotal Time: {total_time:.2f} seconds")
        print(f"Progress: {total_instances}/{total_instances} instances completed")
        print(f"Progress: {total_instances}/{total_instances} instances completed")

        with open(result_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(instances, indent=2, ensure_ascii=False))

    print("finished")

    


if __name__ == "__main__":
    fire.Fire(main)
