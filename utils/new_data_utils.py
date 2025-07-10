import random

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaTokenizer


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids


def get_tokenizer(model):
    if "llama" in model.lower():
        # tokenizer = LlamaTokenizer.from_pretrained(model, use_fast=False)
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
        # fix for transformer 4.28.0.dev0 compatibility
        if tokenizer.bos_token_id != 1 or tokenizer.eos_token_id != 2:
            try:
                tokenizer.bos_token_id = 1
                tokenizer.eos_token_id = 2
            except AttributeError:
                pass
    else:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    return tokenizer


def get_winogrande_trainenc(seed, nsamples, seqlen, model, tokenizer, batch_size):
    
    traindata = load_dataset('winogrande', 'winogrande_xl', split='train')
    combined_input = []
    for i in range(nsamples):
        sentence = traindata['sentence'][i]
        option1 = traindata['option1'][i]
        option2 = traindata['option2'][i]
        answer = traindata['answer'][i]
        if answer=='1': 
            combined_input.append(sentence.replace("_", option1))
        elif answer =='2': 
            combined_input.append(sentence.replace("_", option2))
        else: 
            print(answer)
            print(type(answer))
    trainenc = tokenizer("\n\n".join(combined_input), return_tensors='pt')
    traindata = traindata.shuffle(seed=seed)

    return trainenc

def get_piqa_trainenc(seed, nsamples, seqlen, model, tokenizer, batch_size):
    
    traindata = load_dataset('piqa', split='train')
    traindata = traindata.shuffle(seed=seed)

    combined_input = []
    for i in range(nsamples):
        goal = traindata['goal'][i]
        sol1 = traindata['sol1'][i]
        sol2 = traindata['sol2'][i]
        label = traindata['label'][i]+1
        if label  ==1:
            combined_input.append(f"Question: {goal}\nAnswer: {sol1}")
        elif label ==2:
            combined_input.append(f"Question: {goal}\nAnswer: {sol2}")
    trainenc = tokenizer("\n\n".join(combined_input), return_tensors='pt')
    # print('trainenc: ',trainenc.input_ids)
    # print('trainenc: ',trainenc.input_ids.shape)
    return trainenc




def get_boolq_trainenc(seed, nsamples, seqlen, model, tokenizer, batch_size):
    
    traindata = load_dataset('boolq', split='train')
    traindata = traindata.shuffle(seed=seed)

    combined_input = []
    for i in range(nsamples):
        passage = traindata['passage'][i]
        context = traindata['question'][i]
        label = traindata['answer'][i]
        combined_input.append(f"{passage}\nQuestion: {context}?\nAnswer: {label}")
        
    trainenc = tokenizer("\n\n".join(combined_input), return_tensors='pt')
    # print('trainenc: ',trainenc.input_ids)
    # print('trainenc: ',trainenc.input_ids.shape)
    return trainenc


def get_arc_easy_trainenc(seed, nsamples, seqlen, model, tokenizer, batch_size):
    
    traindata = load_dataset("ai2_arc", 'ARC-Easy', split='train')
    traindata = traindata.shuffle(seed=seed)

    combined_input = []
    for i in range(nsamples):
        question = traindata['question'][i]
        text = traindata['choices'][i]['text']
        answerKey=traindata['answerKey'][i]
        if answerKey=='A'or answerKey == '1':
            anwer=0
        elif answerKey =='B'or answerKey == '2':
            answer=1
        elif answerKey =='C'or answerKey == '3':
            answer=2
        elif answerKey =='D'or answerKey == '4':
            answer=3
        elif type(answerKey) == int:
            answer = answerKey-1
        else:
            print('i: ', i)
            print(traindata[i])
        combined_input.append(f"Question: {question}\nAnswer: {text[answer]}")
    trainenc = tokenizer("\n\n".join(combined_input), return_tensors='pt')
    print('trainenc: ',trainenc.input_ids)
    print('trainenc: ',trainenc.input_ids.shape)
    return trainenc


def get_arc_challenge_trainenc(seed, nsamples, seqlen, model, tokenizer, batch_size):
    
    traindata = load_dataset("ai2_arc", 'ARC-Challenge', split='train')
    traindata = traindata.shuffle(seed=seed)


    combined_input = []
    for i in range(nsamples):
        question = traindata['question'][i]
        text = traindata['choices'][i]['text']
        answerKey=traindata['answerKey'][i]
        if answerKey=='A'or answerKey == '1':
            anwer=0
        elif answerKey =='B'or answerKey == '2':
            answer=1
        elif answerKey =='C'or answerKey == '3':
            answer=2
        elif answerKey =='D'or answerKey == '4':
            answer=3
        elif type(answerKey) == int:
            answer = answerKey-1
        else:
            print('i: ', i)
            print(traindata[i])
            print(sdf)
        combined_input.append(f"Question: {question}\nAnswer: {text[answer]}")
    trainenc = tokenizer("\n\n".join(combined_input), return_tensors='pt')
    print('trainenc: ',trainenc.input_ids)
    print('trainenc: ',trainenc.input_ids.shape)
    return trainenc



def get_hellaswag_trainenc(seed, nsamples, seqlen, model, tokenizer, batch_size):
    traindata = load_dataset("hellaswag",  split='train')
    traindata = traindata.shuffle(seed=seed)

    combined_input = []
    for i in range(nsamples):
        activity_label = traindata['activity_label'][i]
        ctx = traindata['ctx'][i]
        label = traindata['label'][i]
        endings = traindata['endings'][i][int(label)]
        combined_input.append(f"Question: {activity_label}: {ctx}\nAnswer: {endings}")
    trainenc = tokenizer("\n\n".join(combined_input), return_tensors='pt')
    print('trainenc: ',trainenc.input_ids)
    print('trainenc: ',trainenc.input_ids.shape)
    return trainenc
    

def get_trainloaders(name, nsamples=128, seed=0, seqlen=512, model='', batch_size=1):
    tokenizer = get_tokenizer(model)
    if 'winogrande' in name:
        return get_winogrande_trainenc(seed, nsamples, seqlen, model, tokenizer, batch_size)
    elif 'piqa' in name:
        return get_piqa_trainenc(seed, nsamples, seqlen, model, tokenizer, batch_size)
    elif 'boolq' in name:
        return get_boolq_trainenc(seed, nsamples, seqlen, model, tokenizer, batch_size)
    elif 'arc_easy' in name:
        return get_arc_easy_trainenc(seed, nsamples, seqlen, model, tokenizer, batch_size)
    elif 'arc_challenge' in name:
        return get_arc_challenge_trainenc(seed, nsamples, seqlen, model, tokenizer, batch_size)
    elif 'hellaswag' in name:
        return get_hellaswag_trainenc(seed, nsamples, seqlen, model, tokenizer, batch_size)
    else:
        print(name)




