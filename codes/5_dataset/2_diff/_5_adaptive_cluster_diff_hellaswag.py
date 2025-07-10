import fire
import copy
import time
from tqdm import tqdm
import os

import torch
import torch.nn as nn
import re
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from utils.model_utils import get_llm
from utils.onoff_utils.onoff import block_replace, turn_off, turn_on
from utils.data_utils import *
from utils.block_remove import block_remove
from utils.eval_utils import load_and_eval_ppl, eval_zero_shot
import pandas as pd
import json
import csv
import ast
import matplotlib.pyplot as plt
import math
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from torch.utils.data import DataLoader
@torch.no_grad()
def get_tokenizer(model):
    if "llama" in model.lower():
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
        if tokenizer.bos_token_id != 1 or tokenizer.eos_token_id != 2:
            try:
                tokenizer.bos_token_id = 1
                tokenizer.eos_token_id = 2
            except AttributeError:
                pass

    else:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, trust_remote_code=True)

    return tokenizer
def minmax_normalize(values):
    min_value = min(values)
    max_value = max(values)
    return [(value - min_value) / (max_value - min_value) for value in values]

def sleb(
        model_name: str = 'meta-llama/Meta-Llama-3.1-8B',
        num_blocks: int = 32,
        num_remove_blocks: int = 7,
        early_barrier: int = 1,
        latter_barrier: int = 1,
        seed: int = 0,
        nsamples: int = 128,
        removal_list: list = [],
        folder: str = 'result/5_from30',
        eval_ppl: bool = False,
        eval_zeroshot: bool = True,
        idx: int=0,
        data='hellaswag',
        method='likelihood_diff'
):
    result_folder=f'{folder}/{data}'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    dataset = load_dataset("hellaswag", split='train')
    # dataset = load_dataset("ai2_arc", 'ARC-Easy', split='train')
    tokenizer = get_tokenizer(model_name)
    total = len(dataset)

    def loglikelihood(model,activity_label: str, ctx: str, endings: int):
        
        # 토큰화
        inputs = tokenizer(f"Question: {activity_label}: {ctx}\nAnswer: {endings}", return_tensors="pt").to('cuda')

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        logits = outputs.logits
        answer_tokens = tokenizer.convert_tokens_to_ids(["Answer", ":"])
        input_ids = inputs["input_ids"][0].tolist()
        for i in range(len(input_ids) - len(answer_tokens) + 1):
            if input_ids[i:i + len(answer_tokens)] == answer_tokens:
                answer_start_index = i + len(answer_tokens)
                break
        else:
            raise ValueError('"Answer:" not found in input text.')
        target_ids = inputs["input_ids"][:, answer_start_index:].contiguous()
        logits = logits[:, answer_start_index - 1:-1, :]
        decoded_answer = tokenizer.decode(target_ids[0], skip_special_tokens=True)
        log_probs = torch.log_softmax(logits, dim=-1)
        log_likelihood = torch.gather(log_probs, dim=-1, index=target_ids.unsqueeze(-1))
        log_likelihood = log_likelihood.squeeze(-1)
        return log_likelihood.mean()
    result_file = f'{method}.csv'    
    result_path = os.path.join(result_folder, result_file)



    print(f"Starting Zero-shot tasks evaluation...")

    
    
    
    with open(result_path, mode='a', encoding="utf-8",newline='') as outfile:
        fieldnames = [ 'max_idx', 'max_set', 'softmax','minmax_softmax','likelihood_diff','inputs']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()


    temp_model = get_llm(model_name)
    use_cache = temp_model.config.use_cache
    temp_model.config.use_cache = False
    temp_model = block_replace(temp_model)
    temp_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    temp_model.to(device)


    for item in tqdm(dataset.select(range(total))):

        activity_label = item['activity_label']
        ctx = item['ctx']
        label = item['label']
        endings = item['endings']
        
        open_path= 'codes/9_llama/llama_layer_list_3_onlydiff.csv'
     
        diff_values = []
        diff_list = [] 
        softmax_values = []
        
        with open(open_path, mode='r', newline='') as infile:
            reader = csv.DictReader(infile)

            for row in reader:
                idx=int(ast.literal_eval(row['idx']))
                removal_list = list(ast.literal_eval(row['Set']))
                for i in range(len(removal_list)):
                    turn_off(temp_model, removal_list[i])
                log_likelihood_0 = loglikelihood(temp_model,activity_label, ctx,endings[0])
                log_likelihood_1 = loglikelihood(temp_model,activity_label, ctx,endings[1])
                log_likelihood_2 = loglikelihood(temp_model,activity_label, ctx,endings[2])
                log_likelihood_3 = loglikelihood(temp_model,activity_label, ctx,endings[3])
                softmax0=float(log_likelihood_0)
                softmax1=float(log_likelihood_1)
                softmax2=float(log_likelihood_2)
                softmax3=float(log_likelihood_3)

                if int(label) ==0:
                    diff= 3*softmax0-softmax1-softmax2-softmax3
                elif int(label) ==1:
                    diff= 3*softmax1-softmax0-softmax2-softmax3
                elif int(label) ==2:
                    diff= 3*softmax2-softmax1-softmax0-softmax3
                elif int(label) ==3:
                    diff= 3*softmax3-softmax1-softmax2-softmax0
                else:
                    print(sdfsdf)

                diff_values.append({'idx': idx, 'Set': removal_list, 'diff': diff})
                diff_list.append(diff) 

                for i in range(len(removal_list)):
                    turn_on(temp_model, removal_list[i])
            

        max_diff = max(diff_list)
        exp_values = [math.exp(diff - max_diff) for diff in diff_list]
        sum_exp = sum(exp_values)
        softmax_values = [exp / sum_exp for exp in exp_values]


        normalized_diff_list = minmax_normalize(diff_list)
        max_norm_diff = max(normalized_diff_list)
        exp_norm_values = [math.exp(norm_diff - max_norm_diff) for norm_diff in normalized_diff_list]
        sum_exp_norm = sum(exp_norm_values)
        softmax_minmax_values = [exp_norm / sum_exp_norm for exp_norm in exp_norm_values]


        max_softmax_idx = exp_values.index(max(exp_values))
        max_idx = diff_values[max_softmax_idx]['idx']
        max_set = diff_values[max_softmax_idx]['Set']


        
        with open(result_path, mode='a', encoding="utf-8", newline='') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            for i in range(len(endings)):
                inputs = f"Question: {activity_label}: {ctx}\nAnswer: {endings[i]}"
                # new_row = {'inputs': inputs, 'max_idx': max_idx, 'max_set': max_set, 'softmax': softmax_values,'likelihood_diff': diff_list}
                new_row = {'inputs': inputs, 'max_idx': max_idx, 'max_set': max_set,'softmax': softmax_values, 'minmax_softmax': softmax_minmax_values,'likelihood_diff': diff_list }

                writer.writerow(new_row)
        


if __name__ == "__main__":
    fire.Fire(sleb)