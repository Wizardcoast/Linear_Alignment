import argparse

import datasets
import pandas as pd

from datasets import Dataset

from conversation import get_conv_adapter
from utils import *
from transformers import TextStreamer

import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

from dataset import PreferenceExactMatchDataset
from model import ConstractiveDecodingModel

from dataset import Principle

import numpy as np

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)


def extract_answer(answer):
    answer = answer.strip()
    answer = answer[0]
    return answer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--principle_id",
                        type=int)

    parser.add_argument("--conv_type",
                        type=str,
                        default="llama2")

    parser.add_argument("--data_path",
                        type=str,
                        default="kkuusou/personal_preference_eval")

    parser.add_argument("--model_path",
                        type=str,
                        default='/mnt/petrelfs/gaosongyang/models/mistralai/Mistral-7B-Instruct-v0.1')

    parser.add_argument("--temperature",
                        type=float,
                        default=1.0)

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
        'max_new_tokens': 10,
        'temperature': args.temperature,
        "top_p": 0.8,
        "do_sample": args.do_sample
    }

    cd_config = {
        "ratio": args.ratio
    }

    print("Begin loading dataset !", flush=True)

    raw_dataset = datasets.load_dataset(args.data_path, split="train")

    print('dataset loading down !', flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print('loading origin model !')
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 device_map='auto',
                                                 torch_dtype=torch.float16)

    model = ConstractiveDecodingModel(model, tokenizer)
    model.model = model.model.eval()
    print('model loading down')

    selected_data = PreferenceExactMatchDataset(raw_dataset, principle=principle,
                                                conv_adapter=conv_adapter)

    data_len = len(selected_data)

    print(f"datasets len: {data_len}")
    generated_data = []

    contra_corr = 0
    principle_corr = 0

    count = 0

    for index, i in tqdm(enumerate(selected_data)):
        print(f"index:{index}", flush=True)

        data_points = i
        ground_truth = i["ground_truth"]
        no_principle_inputs = tokenizer(i["dialog_no_preference"] + "Answer:", return_tensors='pt')
        no_principle_ids = no_principle_inputs.input_ids
        no_principle_att = no_principle_inputs.attention_mask

        principle_inputs = tokenizer(i["dialog"] + "Answer:", return_tensors='pt')
        principle_ids = principle_inputs.input_ids
        principle_att = principle_inputs.attention_mask
        generate_ids_sys = model.model.generate(principle_ids.cuda(), **generation_config)

        generate_ids1 = model.contra_generate(
            principle_ids.cuda(), no_principle_ids.cuda(),
            attention_mask_in=principle_att.cuda(),
            attention_mask_out=no_principle_att.cuda(), **generation_config, **cd_config)

        contra_output = tokenizer.decode(generate_ids1[0])
        len_sys_in = len(principle_ids[0])
        principle_output = tokenizer.decode(generate_ids_sys[0][len_sys_in:])

        data_points["index"] = index

        data_points["contra_output"] = contra_output

        data_points["principle_output"] = principle_output

        contra_answer = str(extract_answer(contra_output))

        if contra_answer == str(ground_truth):
            contra_corr += 1

        else:
            print("Contra error!")
            print("Contra:", contra_answer)
            print("Ground_truth:", ground_truth)

        principle_answer = str(extract_answer(principle_output))
        if principle_answer == str(ground_truth):
            principle_corr += 1

        else:
            print("Principle error!")
            print("Principle:", principle_answer)
            print("Ground_truth:", ground_truth)

        count += 1

        generated_data.append(data_points)

        with open(args.output_result_file, 'w') as f:
            json.dump(generated_data, f, indent=4)

    print("contra_acc:", contra_corr / count)
    print("principle_acc:", principle_corr / count)
