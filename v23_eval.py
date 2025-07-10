import copy
from tqdm import tqdm
import fire
import os
from typing import List
import warnings
import torch

from utils.model_utils import get_llm
from utils.v23_eval_utils import load_and_eval_ppl, eval_zero_shot
from v23_adaptSLEB import v23_adaptSLEB
from peft import PeftModel

def v23_eval(
        model_name: str = "meta-llama/Meta-Llama-3.1-8B",
        router_path: str = "./model/v3/jy_diff_all/",
        save_results: bool = False,
        result_folder: str = 'sleb_results',
        result_file: str = 'result.txt',
        device: int = 0,
        eval_ppl: bool = Ture,
        eval_zeroshot: bool = True,
        tasks: list = ['arc_challenge','arc_easy','boolq','hellaswag','piqa','winogrande'],
        custom: bool = True,
        version: str = 'v23'
    ):
    warnings.filterwarnings("ignore", category=UserWarning)

    
    if custom:
        if version == 'v23':
            model, tokenizer = v23_adaptSLEB(model_name, router_path)
        else:
            print('no model loaded')
            model = None
    else:
        model = get_llm(model_name)

    if model is not None:
        print(f"Loaded Model: {model.name if hasattr(model, 'name') else model_name}")
        model.eval()
        model.to(f'cuda:{device}')
    else:
        raise ValueError("Model could not be loaded.")
    

    test_datasets = ['wikitext2', 'c4']
    ppl_list = {}
    ppl_list_length={}
    if eval_ppl:
        print(f"Starting PPL evaluation...")
        for dataset in test_datasets:
            ppl = load_and_eval_ppl(model_name, model, device, dataset=dataset)
            print(f"{dataset} perplexity = {ppl:.2f}")
            ppl_list[dataset] = ppl

    else:
        for dataset in test_datasets:
            ppl_list[dataset] = None

    del model
    torch.cuda.empty_cache()

    if eval_zeroshot:
        print(f"Starting Zero-shot tasks evaluation...")
        if '30b' or '66b' or '70b' in model_name:
            parallelize = True
        else:
            parallelize = False

        tasks = ['mmlu']
        print(tasks)
        print('asdfsadfsadfasdfsadfsad')
        results = eval_zero_shot(model_name,router_path=router_path, task_list=tasks, parallelize=parallelize)

        for task in tasks:
            result = results[task]
            print(f"{task}: {result}")

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    result_path = os.path.join(result_folder, result_file)
    
    if save_results:
        with open(result_path, 'a') as file:
            sentences = []
            sentences.append(f"USE DICTIONARY TO PICK SAME LAYER SETS\n")
            sentences.append(f"Model Name: {model_name}\n")
            sentences.append(f"Router Path: {router_path}\n")
            
            if eval_ppl:
                for dataset in test_datasets:
                    sentences.append(f"{dataset} PPL = {ppl_list[dataset]:.2f}\n")


            if eval_zeroshot:
                sentences.append(f"Zero-shot results: \n")
                sentences.append(f"     piqa: {round(results['piqa']['acc,none']*100,3)}\n")
                sentences.append(f"     winogrande: {round(results['winogrande']['acc,none']*100,3)}\n")
                sentences.append(f"     hellaswag: {round(results['hellaswag']['acc_norm,none']*100,3)}\n")
                sentences.append(f"     arc_challenge: {round(results['arc_challenge']['acc_norm,none']*100,3)}\n")
                sentences.append(f"     arc_easy: {round(results['arc_easy']['acc,none']*100,3)}\n")
                sentences.append(f"     boolq: {round(results['boolq']['acc,none']*100,3)}\n")

            sentences.append("\n")

                                
            for sentence in sentences:
                file.write(sentence)


if __name__ == "__main__":
    fire.Fire(v23_eval)