# Import necessary modules
import time
from tqdm import tqdm
import os
import sys
import torch
import torch.nn as nn
import copy
# Import get_loaders function from data module within the same director
from utils.data_utils import *
import fnmatch


# Function to evaluate perplexity (ppl) on a specified model and tokenizer
@torch.no_grad()
def load_and_eval_ppl(model, device=torch.device("cuda:0"), dataset='wikitext2', testloader=None, tokenizer=None,portion=None,use_last_fraction=0.5):
    # Print status
    print(f"Evaluating on {dataset}")

    # Get the test loader
    if testloader is None:
        if tokenizer is None:
            tokenizer = get_tokenizer(model.name)

        _, testloader = get_loaders(
            dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer 
        )
        print(f"Dataset Loaded.")

    # Evaluate ppl in no grad context to avoid updating the model
    with torch.no_grad():
        if portion=='prompt':
            print('Evaluating Prompt Portion')
            _, testloader = get_loaders(
                dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer,prompt_portion=True
            )
            print(f"Dataset Loaded.")

            ppl_test=eval_ppl_batch_input_prompt_portion(model, testloader, 1, device,use_last_fraction)
        else:
            _, testloader = get_loaders(
                dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer 
            )
            print(f"Dataset Loaded.")
            if portion==None:
                print('Evaluating Standard PPL')
                ppl_test = eval_ppl(model, testloader, 1, device)
            elif portion=='batch':
                print('Evaluating Batch Portion')
                ppl_test=eval_ppl_batch_portion(model, testloader, 1, device,use_last_fraction)
    return ppl_test 
@torch.no_grad()
def eval_ppl(model, testenc, bs=1, device=None):
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in tqdm(range(0,nsamples,bs)):

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # Empty CUDA cache to save memory
    # torch.cuda.empty_cache()

    return ppl.item()


@torch.no_grad()
def eval_ppl_batch_input_prompt_portion(model, data, bs=1, device=None,use_last_fraction=0.5):
    (testenc,sentence_lengths)=data
    testenc = testenc.input_ids.to(device)
    nsamples = testenc.numel() // model.seqlen

    nlls = []
    total_tokens = 0

    loss_fct = nn.CrossEntropyLoss(reduction='none')

    sentence_start_idx = 0

    for i in tqdm(range(0, nsamples, bs)):
        j = min(i + bs, nsamples)
        inputs = testenc[:, (i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j - i, model.seqlen)

        lm_logits = model(inputs).logits
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(j - i, model.seqlen - 1)

        while sentence_start_idx < len(sentence_lengths) and sentence_lengths[sentence_start_idx] <= (i * model.seqlen):
            sentence_start_idx += 1

        token_idx = i * model.seqlen
        while sentence_start_idx < len(sentence_lengths) and token_idx < (j * model.seqlen):
            sent_len = sentence_lengths[sentence_start_idx]
            start_idx = token_idx
            end_idx = min(start_idx + sent_len, testenc.size(1))

            trunc_start = start_idx + int((1 - use_last_fraction) * (end_idx - start_idx))
            if trunc_start < end_idx:
                selected_loss = loss[:, trunc_start - (i * model.seqlen):end_idx - (i * model.seqlen)]
                nlls.append(selected_loss.sum())
                total_tokens += (end_idx - trunc_start)

            token_idx += sent_len
            sentence_start_idx += 1

    if total_tokens == 0:
        return float('inf')

    ppl = torch.exp(torch.stack(nlls).sum() / total_tokens)
    print(f'torch.stack(nlls).sum() :{torch.stack(nlls).sum()}')
    print(f' total_tokens:{total_tokens}')
    return ppl.item()

@torch.no_grad()
def eval_ppl_batch_portion(model, testenc, bs=1, device=None,use_last_fraction=0.5):
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in tqdm(range(0,nsamples,bs)):

        j = min(i+bs, nsamples)

        inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)
        lm_logits = model(inputs).logits

        input_length = inputs.size(1)
        cut_point = int(input_length * (1 - use_last_fraction))  # Define where to cut the input
        shift_logits = lm_logits[:, cut_point:-1, :].contiguous()  # Use only the last part of the sentence
        shift_labels = inputs[:, cut_point+1:].contiguous()  # Shift labels accordingly

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        neg_log_likelihood = loss.float() * model.seqlen * (j-i)
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen*use_last_fraction))

    print(f'torch.stack(nlls).sum() :{torch.stack(nlls).sum()}')
    print(f' nsamples * model.seqlen:{nsamples * model.seqlen*use_last_fraction}')

    torch.cuda.empty_cache()

    return ppl.item()





torch.no_grad()
def eval_zero_shot(model_name, removal_list, task_list=['piqa','winogrande','hellaswag','arc_challenge','arc_easy'], 
        num_fewshot=0, parallelize=False):

    desired_dir = "/home/jywee/projects/Mixture-of-depths/SLEB_2_pruning/lm_evaluation_harness"
    sys.path.insert(0, desired_dir) 
    from lm_eval import tasks, evaluator, utils

    task_manager = tasks.TaskManager(include_path='lm-evaluation-harness/lm_eval/tasks')
 
    task_names = task_manager.match_tasks(task_list)
    for task in [task for task in task_list if task not in task_names]:
                if os.path.isfile(task):
                    config = utils.load_yaml_config(task)
                    task_names.append(config)
    task_missing = [
        task
        for task in task_list
        if task not in task_names and "*" not in task
        ]  # we don't want errors if a wildcard ("*") task name was used
    
    
    model_args = f"pretrained={model_name},"
    if parallelize:
        model_args = f"pretrained={model_name},parallelize=True"

    if len(removal_list)>0:
        remove = True
    else:
        remove = False
    # results = evaluator.simple_evaluate(
    #     model='hf',
    #     model_args=model_args,
    #     tasks=task_list,
    #     num_fewshot=num_fewshot,
    #     batch_size='auto',
    #     max_batch_size=None,
    #     device='cuda:0',
    #     use_cache=None,
    #     limit=None,
    #     decontamination_ngrams_path=None,
    #     check_integrity=False,
    #     write_out=False,
    #     gen_kwargs=None,
    #     task_manager=task_manager,
    #     remove = remove,
    #     removal_list = removal_list,
    # )
    print('model_args: ', model_args)
    results={}
    tmp = copy.deepcopy(removal_list)
    for task in task_list:
        removal_list = copy.deepcopy(tmp)
        print(f'removal_list: {removal_list}')
        result = evaluator.simple_evaluate(
            model='hf',
            model_args=model_args,
            tasks=task,
            num_fewshot=num_fewshot,
            # batch_size='auto',
            batch_size=1,
            max_batch_size=None,
            device='cuda:0',
            use_cache=None,
            limit=None,
            decontamination_ngrams_path=None,
            check_integrity=False,
            write_out=False,
            gen_kwargs=None,
            task_manager=task_manager,
            remove = remove,
            removal_list = removal_list,
            # a=''
        )
        print(result['results'])
        results[task] = result['results'][task]


    return results 