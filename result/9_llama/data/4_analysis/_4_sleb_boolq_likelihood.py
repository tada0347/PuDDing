import fire
import copy
import time
from tqdm import tqdm
import os
import csv

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from torch.utils.data import DataLoader

from utils.model_utils import get_llm
from utils.onoff_utils.onoff import block_replace, turn_off, turn_on
from utils.data_utils import *
from utils.block_remove import block_remove
from utils.eval_utils import load_and_eval_ppl, eval_zero_shot



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

tokenizer = get_tokenizer('meta-llama/Meta-Llama-3.1-8B')


def loglikelihood(model,passage,context: str, continuation: str):
    label = "True" if continuation else "False"
    inputs = tokenizer(f"{passage}\nQuestion: {context}?\nAnswer: {label}", return_tensors="pt").to('cuda')

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

@torch.no_grad()
def get_loss(model, testenc,nsamples, bs=1, device=None):   
    total = len(testenc)
    float_sum=0
    for item in tqdm(testenc.select(range(nsamples))):
        passage = item['passage']
        question = item['question']
        answer = item['answer']
        
        log_likelihood= loglikelihood(model,passage, question, answer)


        float_sum+=  float(log_likelihood)


    return -float_sum

def sleb(
        model_name: str = 'meta-llama/Meta-Llama-3.1-8B',
        num_blocks: int = 32,
        num_remove_blocks: int = 7,
        early_barrier: int = 1,
        latter_barrier: int = 1,
        seed: int = 0,
        nsamples: int = 128,
        result_folder: str = 'result/4_custom/boolq',
        result_file: str = 'likelihood',
        dataset: str = 'boolq',
        eval_ppl: bool = True,
        eval_zeroshot: bool = True
):
    print(num_blocks)
    print(type(num_blocks))
    alive_list = [i for i in range(num_blocks)]
    removal_list = []

    model = get_llm(model_name)
    use_cache = model.config.use_cache
    model.config.use_cache = False
    print(f"Loaded Model: {model.name}")
    
    model = block_replace(model)
    model.eval()
    
    dataloader = get_trainloaders(dataset,
                                  nsamples=nsamples,
                                 seed=seed,
                                 model=model_name,
                                 )

    dataset = load_dataset("boolq", split='train')

    
    print(f"Dataloader({dataset}) loaded.")

    start_point = time.time()
    for i in range(num_remove_blocks):

        phase_start_point = time.time()
        print(f"Phase {i+1} of {num_remove_blocks}")

        min_loss = 1e99
        min_loss_idx = -1

        search_bound = num_blocks - i

        for j in range(early_barrier, search_bound-latter_barrier):
            turn_off(model, alive_list[j])
            loss = get_loss(model, dataset,nsamples, bs=1, device=torch.device("cuda:0"))
            torch.cuda.empty_cache()
            if loss < min_loss:
                min_loss = loss
                min_loss_idx = j
            print(
                f"[Block {j} (Original block {alive_list[j]}) removed] Loss={loss:.3f}, Current Min Loss={min_loss:.3f} / Layer {alive_list[min_loss_idx]}"
            )
            turn_on(model, alive_list[j])
        phase_finish_point = time.time()
        phase_time_elapsed = phase_finish_point -  phase_start_point

        print(f"Phase_time_elapsed (s): {phase_time_elapsed}")
        print(f"[SELECTED block {min_loss_idx} (Originally block {alive_list[min_loss_idx]})] Loss={min_loss:.3f}")      
        turn_off(model, alive_list[min_loss_idx])
        removal_list.append(alive_list[min_loss_idx])
        print(f"Current Block Removal List: {removal_list}")
        del alive_list[min_loss_idx]
    
    finish_point = time.time()
    time_elapsed = finish_point - start_point

    print(
        f"Time_Elapsed: {time_elapsed}\n"
        f"# Remove Blocks: {num_remove_blocks}\n"
        f"Dataset: {dataset}\n"
        f"Block Removal Order: {removal_list}\n"
    )

    
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)    
    txt_file=f'{result_file}.txt'
    csv_file=f'{result_file}.csv'
    txt_result_path = os.path.join(result_folder, txt_file)
    csv_result_path = os.path.join(result_folder, csv_file)
    
    with open(txt_result_path, 'a') as file:
        sentences = []
        sentences.append(f"Dataset: {dataset}\n")
        sentences.append(f"Block Removal Order: {removal_list}\n")
        sentences.append(f"Sorted Removal Order: {sorted(removal_list)}\n")
        for sentence in sentences:
            file.write(sentence)
    if eval_ppl:
        print(f"Starting PPL evaluation...")
        model = block_remove(model, copy.deepcopy(removal_list))
        model.config.use_cache = use_cache

        w2_ppl = load_and_eval_ppl(model, device=torch.device("cuda:0"), dataset='wikitext2')
        print(f"WikiText-2 PPL = {w2_ppl:.2f}")

        c4_ppl = load_and_eval_ppl(model, device=torch.device("cuda:0"), dataset='c4')
        print(f"C4 PPL = {c4_ppl:.2f}")


    if eval_zeroshot:
        print(f"Starting Zero-shot tasks evaluation...")
        if '30b' or '66b' or '70b' in model_name:
            parallelize = True
        else:
            parallelize = False

        tasks = ['boolq','piqa','winogrande','hellaswag','arc_challenge','arc_easy']
        # tasks = ['boolq']
        results = eval_zero_shot(model_name, copy.deepcopy(removal_list), tasks, parallelize=parallelize)

        for task in tasks:
            result = results[task]
            print(f"{task}: {result}")

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    
    txt_file=f'{result_file}.txt'
    csv_file=f'{result_file}.csv'
    txt_result_path = os.path.join(result_folder, txt_file)
    csv_result_path = os.path.join(result_folder, csv_file)
    
    piqa=round(results['piqa']['acc,none']*100,3)
    winogrande=round(results['winogrande']['acc,none']*100,3)
    boolq=round(results['boolq']['acc,none']*100,3)
    arc_easy=round(results['arc_easy']['acc,none']*100,3)
    arc_challenge=round(results['arc_challenge']['acc_norm,none']*100,3)
    hellaswag=round(results['hellaswag']['acc_norm,none']*100,3)
    task_average= round((piqa + winogrande + boolq + arc_easy + arc_challenge + hellaswag) / 6, 3)
    task_average_wo_boolq = round((piqa + winogrande + arc_easy + arc_challenge + hellaswag) / 6, 3)


    with open(txt_result_path, 'a') as file:

        sentences = []
        sentences.append(f"Dataset: {dataset}\n")
        sentences.append(f"Block Removal Order: {removal_list}\n")
        sentences.append(f"Sorted Removal Order: {sorted(removal_list)}\n")
        if eval_ppl:
            sentences.append(f"WikiText-2 PPL = {w2_ppl:.2f}\n")
            sentences.append(f"C4 PPL = {c4_ppl:.2f}\n")

        sentences.append(f"Zero-shot results: \n")
        sentences.append(f"     average: {task_average}\n")
        sentences.append(f"     average w/o boolq: {task_average_wo_boolq}\n")
        sentences.append(f"\n")      

        sentences.append(f"     piqa: {piqa}\n")
        sentences.append(f"     winogrande: {winogrande}\n")
        sentences.append(f"     boolq: {boolq}\n")
        sentences.append(f"     arc_easy: {arc_easy}\n")
        sentences.append(f"     arc_challenge: {arc_challenge}\n")
        sentences.append(f"     hellaswag: {hellaswag}\n")
        for sentence in sentences:
            file.write(sentence)

    data = {
        "Dataset": dataset,
        "Block_Removal": removal_list,
        "Sorted_Removal": sorted(removal_list),
        "WikiText": round(w2_ppl, 2),
        "C4": round(c4_ppl, 2),
        "Average": task_average,
        "Average w/o BoolQ": task_average_wo_boolq,
        "PIQA": piqa,
        "Winogrande": winogrande,
        "HellaSwag": hellaswag,
        "Arc-C": arc_challenge,
        "Arc-E": arc_easy,
        "BoolQ": boolq
    }

    fieldnames = ["Dataset","Block_Removal","Sorted_Removal","WikiText", "C4", "Average", "Average w/o BoolQ", "PIQA", "Winogrande", "HellaSwag", "Arc-C", "Arc-E", "BoolQ"]

    with open(csv_result_path, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        writer.writeheader()
        
        writer.writerow(data)

    print(f"Data has been written to {csv_file}.")

if __name__ == "__main__":
    fire.Fire(sleb)