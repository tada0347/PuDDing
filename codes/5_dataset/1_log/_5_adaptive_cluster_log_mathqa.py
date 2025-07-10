import fire
import copy
import time
from tqdm import tqdm
import os

import torch
import torch.nn as nn
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
        open_path = 'codes/9_llama/llama_layer_list_2_onlylog.csv',
        data='math_qa',
        method='likelihood'
):
    result_folder=f'{folder}/{data}'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    dataset = load_dataset("math_qa", split='train')
    tokenizer = get_tokenizer(model_name)
    total = len(dataset)

    def loglikelihood(model,context: str, continuation: str, correct: str):
        inputs = tokenizer(f"Problem: {context}\nOptions: {continuation}\Answer: {correct}", return_tensors="pt").to('cuda')
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])

        logits = outputs.logits

        # answer_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("Answer:"))
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
        # print(f"Decoded Answer Tokens: {decoded_answer}")

        log_probs = torch.log_softmax(logits, dim=-1)

        log_likelihood = torch.gather(log_probs, dim=-1, index=target_ids.unsqueeze(-1))
        log_likelihood = log_likelihood.squeeze(-1)  

        return log_likelihood.mean()


    result_file = f'{method}.csv'    
    result_path = os.path.join(result_folder, result_file)



    print(f"Starting Zero-shot tasks evaluation...")
    # model = get_llm(model_name)
    # use_cache = model.config.use_cache
    # model.config.use_cache = False
    # model = block_replace(model)
    # model.eval()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    # correct = 0
    # float_sum = 0
    
    
    
    with open(result_path, mode='a', newline='') as outfile:
        fieldnames = [ 'max_idx', 'max_set', 'softmax','minmax_softmax','likelihood','inputs']
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
        problem = item['Problem']
        options = item['options']
        correct = item['correct']
        
        diff_values = []
        diff_list = [] 
        softmax_values = []
        with open(open_path, mode='r', newline='') as infile:
            reader = csv.DictReader(infile)

            for row in reader:
                idx=int(ast.literal_eval(row['idx']))
                removal_list = list(ast.literal_eval(row['Set']))  
                # print(f'removal list: {removal_list}')
                for i in range(len(removal_list)):
                    turn_off(temp_model, removal_list[i])
                    # print(f'111: {temp_model.name}')
                    # print(f'    {i}: {removal_list[i]} off')

                log_likelihood = float(loglikelihood(temp_model, problem, options, correct))
                
    

                diff_values.append({'idx': idx, 'Set': removal_list, 'diff': log_likelihood})
                diff_list.append(log_likelihood) 
                for i in range(len(removal_list)):
                    turn_on(temp_model, removal_list[i])
                # print('='*30)
            

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


        
        with open(result_path, mode='a',encoding="utf-8", newline='') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            inputs1 = f"Problem: {problem}\nOptions: {options}\Answer: {correct}"
            new_row1 = {'inputs': inputs1, 'max_idx': max_idx, 'max_set': max_set,'softmax': softmax_values, 'minmax_softmax': softmax_minmax_values,'likelihood': diff_list }
            # inputs2 = f"Question: {context}\nAnswer: {choice2}"
            # new_row2 = {'inputs': inputs2, 'max_idx': max_idx, 'max_set': max_set,'softmax': softmax_values, 'minmax_softmax': softmax_minmax_values,'likelihood': diff_list }
            
            writer.writerow(new_row1)
            # writer.writerow(new_row2)

        
    #     for i in range(len(max_set)):
    #         turn_off(model, max_set[i])
    #         # print(f'    {i}: {max_set[i]} off')

    #     log_likelihood_1 = loglikelihood(model, context, choice1)
    #     log_likelihood_2 = loglikelihood(model, context, choice2)
    #     predicted_choice = 1 if log_likelihood_1 > log_likelihood_2 else 2
    
    #     if predicted_choice == correct_choice:
    #         correct += 1
    #     for i in range(len(max_set)):
    #         turn_on(model, max_set[i])
    #         # print(f'    {i}: {max_set[i]} on')        


        
    # accuracy = correct / total

    # del(model)

    
    # print('total: ', total)
    # print(f"Accuracy : {accuracy:.4f}")

    # result_file = f'{method}.txt'    
    # result_path = os.path.join(result_folder, result_file)

    # with open(result_path, 'a') as file:
    #     sentences = []
    #     sentences.append(f"Dataset: {dataset}\n")
    #     sentences.append(f"Accuracy : {accuracy*100:.4f}\n")
    #     sentences.append("\n")

    #     for sentence in sentences:
    #         file.write(sentence)


if __name__ == "__main__":
    fire.Fire(sleb)