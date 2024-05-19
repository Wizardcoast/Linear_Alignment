import argparse

from conversation import get_conv_adapter
from utils import *

import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import datasets

from dataset import CDDataset
from model import ConstractiveDecodingModel

from dataset import Principle
import numpy as np

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--principle_id",
                        type=int)

    parser.add_argument("--conv_type",
                        type=str,
                        default="llama2")

    parser.add_argument("--data_path",
                        type=str,
                        default='Anthropic/hh-rlhf')

    parser.add_argument("--model_path",
                        type=str,
                        default='/mnt/petrelfs/gaosongyang/models/mistralai/Mistral-7B-Instruct-v0.1')

    parser.add_argument("--temperature",
                        type=float,
                        default=1.0)

    parser.add_argument("--top_p",
                        type=float,
                        default=0.8)

    parser.add_argument("--max_new_tokens",
                        type=int,
                        default=512)

    parser.add_argument("--output_data_file",
                        type=str,
                        required=True)

    parser.add_argument("--output_result_file",
                        type=str,
                        required=True)

    parser.add_argument("--data_size",
                        type=int,
                        default=20)

    parser.add_argument("--ratio",
                        type=float,
                        default=2.0)

    parser.add_argument("--do_sample",
                        action="store_true")

    args = parser.parse_args()

    conv_adapter = get_conv_adapter(args.conv_type)

    principle_list = Principle()
    model_path = args.model_path
    principle = principle_list.principle_list[args.principle_id]

    generation_config = {
        'max_new_tokens': args.max_new_tokens,
        'temperature': args.temperature,
        "top_p": args.top_p,
        "do_sample": args.do_sample
    }

    cd_config = {
        "ratio": args.ratio
    }

    print("Loading dataset !", flush=True)
    raw_dataset = datasets.load_dataset("Anthropic/hh-rlhf", split='test')

    shuffled_dataset = raw_dataset.shuffle(seed=42)

    sampled_dataset = shuffled_dataset.select(range(args.data_size))

    del raw_dataset, shuffled_dataset
    print('Dataset loaded !', flush=True)

    if "qwen" in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, pad_token='<|im_end|>',
                                                  eos_token='<|im_end|>')
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print('Loading origin model !')
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 device_map='auto',
                                                 torch_dtype=torch.float16)

    model = ConstractiveDecodingModel(model, tokenizer)

    model.model = model.model.eval()
    print('Model loaded!')

    selected_data = CDDataset(sampled_dataset, principle=principle, conv_adapter=conv_adapter)

    sampled_dataset.to_json(args.output_data_file)

    data_len = len(selected_data)

    print(f"datasets len: {data_len}")

    generated_data = []

    for index, i in tqdm(enumerate(selected_data)):
        print(f"index:{index}", flush=True)
        principle_text = i["dialogue_text_principle"]
        no_principle_text = i["dialogue_text"]
        chosen_answer = i["chosen_answer"]

        inputs1 = tokenizer(principle_text, return_tensors='pt')
        ids1 = inputs1.input_ids
        att1 = inputs1.attention_mask
        inputs2 = tokenizer(no_principle_text, return_tensors='pt')
        ids2 = inputs2.input_ids
        att2 = inputs2.attention_mask

        generate_ids1 = model.contra_generate(
            ids1.cuda(), ids2.cuda(),
            attention_mask_in=att1.cuda(),
            attention_mask_out=att2.cuda(), **generation_config, **cd_config)

        inputs = no_principle_text
        principle = principle

        generate_ids2 = model.model.generate(ids1.cuda(), **generation_config)
        generate_ids3 = model.model.generate(ids2.cuda(), **generation_config)

        contra_output = tokenizer.decode(generate_ids1[0])

        len_principal = len(ids1[0])
        principal_output = tokenizer.decode(generate_ids2[0][len_principal:])

        len_no_principal = len(ids2[0])
        sft_output = tokenizer.decode(generate_ids3[0][len_no_principal:])

        data_points = {
            "id": index,
            "inputs": inputs,
            "principal": principle,
            "contra_output": contra_output,
            "principal_output": principal_output,
            "sft_output": sft_output,
            "chosen_answer": chosen_answer
        }

        generated_data.append(data_points)

        with open(args.output_result_file, 'w') as f:
            json.dump(generated_data, f, indent=4)
