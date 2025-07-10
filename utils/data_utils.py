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
def cutting():
    # def get_wikitext2_trainenc(seed, nsamples, seqlen, model, tokenizer, batch_size):
    
    #     traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    #     traindata = traindata.shuffle(seed=seed)
    #     print("Original lengths of sentences:")
    #     sentence_lengths = [len(sentence) for sentence in traindata[:nsamples]['text']]
    #     print(sentence_lengths)

    #     # Truncate sentences to the desired length
    #     truncated_sentences = [
    #         sentence[:seqlen] if len(sentence) > seqlen else sentence
    #         for sentence in traindata[:nsamples]['text']
    #     ]



    #     # Tokenize the truncated sentences
    #     trainenc = tokenizer("\n\n".join(truncated_sentences), return_tensors='pt')
    #     return trainenc

    # def get_winogrande_trainenc(seed, nsamples, seqlen, model, tokenizer, batch_size):
        
    #     traindata = load_dataset('winogrande', 'winogrande_xl', split='train')
    #     traindata = traindata.shuffle(seed=seed)
    #     # Print the length of each sentence
    #     print("Original lengths of sentences:")
    #     sentence_lengths = [len(sentence) for sentence in traindata[:nsamples]['sentence']]
    #     print(sentence_lengths)

    #     # Truncate sentences to the desired length
    #     truncated_sentences = [
    #         sentence[:seqlen] if len(sentence) > seqlen else sentence
    #         for sentence in traindata[:nsamples]['sentence']
    #     ]



    #     # Tokenize the truncated sentences
    #     trainenc = tokenizer("\n\n".join(truncated_sentences), return_tensors='pt')


    #     return trainenc

    # def get_piqa_trainenc(seed, nsamples, seqlen, model, tokenizer, batch_size):
        
    #     traindata = load_dataset('piqa', split='train')
    #     traindata = traindata.shuffle(seed=seed)

    #     combined_input = []
    #     for i in range(nsamples):
    #         goal = traindata['goal'][i]
    #         sol1 = traindata['sol1'][i]
    #         sol2 = traindata['sol2'][i]
    #         combined_input.append(f"Goal: {goal}\nSolution 1: {sol1}\nSolution 2: {sol2}")
    #     print("Original lengths of sentences:")
    #     sentence_lengths = [len(sentence) for sentence in combined_input]
    #     print(sentence_lengths)

    #     # Truncate sentences to the desired length
    #     truncated_sentences = [
    #         sentence[:seqlen] if len(sentence) > seqlen else sentence
    #         for sentence in combined_input
    #     ]



    #     # Tokenize the truncated sentences
    #     trainenc = tokenizer("\n\n".join(truncated_sentences), return_tensors='pt')

    #     return trainenc

    # def get_boolq_trainenc(seed, nsamples, seqlen, model, tokenizer, batch_size):
        
    #     traindata = load_dataset('boolq', split='train')
    #     traindata = traindata.shuffle(seed=seed)

    #     combined_input = []
    #     for i in range(nsamples):
    #         passage = traindata['passage'][i]
    #         question = traindata['question'][i]
    #         combined_input.append(f"passage: {passage}\nquestion: {question}")
        
    #     print("Original lengths of sentences:")
    #     sentence_lengths = [len(sentence) for sentence in combined_input]
    #     print(sentence_lengths)

    #     # Truncate sentences to the desired length
    #     truncated_sentences = [
    #         sentence[:seqlen] if len(sentence) > seqlen else sentence
    #         for sentence in combined_input
    #     ]



    #     # Tokenize the truncated sentences
    #     trainenc = tokenizer("\n\n".join(truncated_sentences), return_tensors='pt')

    #     return trainenc

    # def get_arc_easy_trainenc(seed, nsamples, seqlen, model, tokenizer, batch_size):
        
    #     traindata = load_dataset("ai2_arc", 'ARC-Easy', split='train')
    #     traindata = traindata.shuffle(seed=seed)

    #     combined_input = []
    #     for i in range(nsamples):
    #         question = traindata['question'][i]
    #         text = traindata['choices'][i]['text']
    #         combined_input.append(f"question: {question}\ntext: {text}")
    #     print("Original lengths of sentences:")
    #     sentence_lengths = [len(sentence) for sentence in combined_input]
    #     print(sentence_lengths)

    #     # Truncate sentences to the desired length
    #     truncated_sentences = [
    #         sentence[:seqlen] if len(sentence) > seqlen else sentence
    #         for sentence in combined_input
    #     ]



    #     # Tokenize the truncated sentences
    #     trainenc = tokenizer("\n\n".join(truncated_sentences), return_tensors='pt')
    #     return trainenc

    # def get_arc_challenge_trainenc(seed, nsamples, seqlen, model, tokenizer, batch_size):
        
    #     traindata = load_dataset("ai2_arc", 'ARC-Challenge', split='train')
    #     traindata = traindata.shuffle(seed=seed)

    #     combined_input = []
    #     for i in range(nsamples):
    #         question = traindata['question'][i]
    #         text = traindata['choices'][i]['text']
    #         combined_input.append(f"question: {question}\ntext: {text}")
    #     print("Original lengths of sentences:")
    #     sentence_lengths = [len(sentence) for sentence in combined_input]
    #     print(sentence_lengths)

    #     # Truncate sentences to the desired length
    #     truncated_sentences = [
    #         sentence[:seqlen] if len(sentence) > seqlen else sentence
    #         for sentence in combined_input
    #     ]



    #     # Tokenize the truncated sentences
    #     trainenc = tokenizer("\n\n".join(truncated_sentences), return_tensors='pt')
    #     return trainenc

    # def get_c4_trainenc(seed, nsamples, seqlen, model, tokenizer, batch_size):
    #     traindata = load_dataset(
    #         'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    #     )
    #     valdata = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
    #     traindata = traindata.shuffle(seed=seed)
        
    #     print("Original lengths of sentences:")
    #     sentence_lengths = [len(sentence) for sentence in traindata[:nsamples]['text']]
    #     print(sentence_lengths)

    #     # Truncate sentences to the desired length
    #     truncated_sentences = [
    #         sentence[:seqlen] if len(sentence) > seqlen else sentence
    #         for sentence in traindata[:nsamples]['text']
    #     ]

    #     # Tokenize the truncated sentences
    #     trainenc = tokenizer("\n\n".join(truncated_sentences), return_tensors='pt')

    #     trainenc = trainenc.input_ids

    #     class TokenizerWrapper:
    #         def __init__(self, input_ids):
    #             self.input_ids = input_ids
    #     trainenc = TokenizerWrapper(trainenc)

    #     return trainenc

    # def get_bookcorpus_trainenc(seed, nsamples, seqlen, model, tokenizer, batch_size):
        
    #     traindata = load_dataset('bookcorpus', split='train')
    #     trainenc = tokenizer("\n\n".join(traindata[:nsamples]['text']), return_tensors='pt')

    #     print("Original lengths of sentences:")
    #     sentence_lengths = [len(sentence) for sentence in traindata[:nsamples]['text']]
    #     print(sentence_lengths)

    #     # Truncate sentences to the desired length
    #     truncated_sentences = [
    #         sentence[:seqlen] if len(sentence) > seqlen else sentence
    #         for sentence in traindata[:nsamples]['text']
    #     ]

    #     # Tokenize the truncated sentences
    #     trainenc = tokenizer("\n\n".join(truncated_sentences), return_tensors='pt')

    #     return trainenc

    return

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
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False,trust_remote_code=True )
    return tokenizer

def get_wikitext2(nsamples, seed, seqlen, model, tokenizer, batch_size,prompt_portion):
    
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    new_trainloader = []
    num_batches = nsamples // batch_size + (int)(nsamples % batch_size > 0)
    for i in range(0, num_batches):
        start =  i * batch_size
        end = min(start + batch_size, nsamples)
        batched_inp = []
        batched_tar = []
        for j in range(start, end):
            batched_inp.append(trainloader[j][0])
            batched_tar.append(trainloader[j][1])
        batched_inp = torch.cat(batched_inp)
        batched_tar = torch.cat(batched_tar)
        new_trainloader.append((batched_inp, batched_tar))
    del trainloader
    trainloader = new_trainloader
    del new_trainloader


    if prompt_portion:
        encoded_sentences = [tokenizer.encode(sent, add_special_tokens=False) for sent in testdata['text']]
        sentence_lengths = [len(sent) for sent in encoded_sentences]
        return None,(testenc, sentence_lengths)  # 문장별 길이 정보 추가

    return trainloader, testenc


def get_winogrande_test(nsamples, seed, seqlen, model, tokenizer, batch_size,prompt_portion):
    testdata = load_dataset('winogrande', 'winogrande_xl', split='validation')

    tokenizer.pad_token = tokenizer.eos_token  
    
    # sentences = testdata['sentence']
    sentences = [
        f"Question: {testdata['sentence'][i]}\nAnswer: {testdata[f'option{label}'][i]}"
        for i, label in enumerate(testdata['answer'])
    ]


    encoded_sentences = [tokenizer.encode(sent, add_special_tokens=False) for sent in sentences]
    sentence_lengths = [len(sent) for sent in encoded_sentences]
    testenc = tokenizer("\n\n".join(sentences), return_tensors='pt')
    print('testenc: ',testenc.input_ids)
    print('testenc: ',testenc.input_ids.shape)
    if prompt_portion : 
        return None,(testenc, sentence_lengths)  # 문장별 길이 정보 추가
    else:
        return None,testenc
    # return None,testenc # 문장별 길이 정보 추가

def get_piqa_test(nsamples, seed, seqlen, model, tokenizer, batch_size,prompt_portion):
    testdata= load_dataset('piqa', split='validation')

    tokenizer.pad_token = tokenizer.eos_token  
    
    # sentences = testdata['goal']
    sentences = [
        f"Question: {testdata['goal'][i]}\nAnswer: {testdata[f'sol{label + 1}'][i]}"
        for i, label in enumerate(testdata['label'])
    ]
    encoded_sentences = [tokenizer.encode(sent, add_special_tokens=False) for sent in sentences]
    sentence_lengths = [len(sent) for sent in encoded_sentences]
    testenc = tokenizer("\n\n".join(sentences), return_tensors='pt')
    if prompt_portion : 
        return None,(testenc, sentence_lengths)  # 문장별 길이 정보 추가
    else:
        return None,testenc
    # return None,testenc # 문장별 길이 정보 추가

def get_boolq_test(nsamples, seed, seqlen, model, tokenizer, batch_size,prompt_portion):
    testdata= load_dataset('boolq', split='validation')

    tokenizer.pad_token = tokenizer.eos_token  
    
    # sentences = testdata['goal']
    sentences = [
        f"{testdata['passage']}\nQuestion: {testdata['question'][i]}?\nAnswer: {answer}"
        for i, answer in enumerate(testdata['answer'])
    ]
    encoded_sentences = [tokenizer.encode(sent, add_special_tokens=False) for sent in sentences]
    sentence_lengths = [len(sent) for sent in encoded_sentences]
    testenc = tokenizer("\n\n".join(sentences),truncation=False, return_tensors='pt')
    if prompt_portion : 
        return None,(testenc, sentence_lengths)  # 문장별 길이 정보 추가
    else:
        return None,testenc


def get_arc_easy_test(nsamples, seed, seqlen, model, tokenizer, batch_size,prompt_portion):
    testdata= load_dataset("ai2_arc", 'ARC-Easy', split='validation')
    tokenizer.pad_token = tokenizer.eos_token  
    
    # sentences = testdata['question']

    sentences = []
    for i in range(nsamples):
        question = testdata['question'][i]
        choices = testdata['choices'][i]['text']  # 선택지 리스트
        answerKey = testdata['answerKey'][i]

        # answerKey를 숫자로 변환
        if answerKey in ['A', '1']:
            answer = 0
        elif answerKey in ['B', '2']:
            answer = 1
        elif answerKey in ['C', '3']:
            answer = 2
        elif answerKey in ['D', '4']:
            answer = 3
        elif isinstance(answerKey, int):
            answer = answerKey - 1  # 정수인 경우 조정
        else:
            print('i: ', i)
            print(testdata[i])
            raise ValueError("Invalid answerKey format.")

        # 정답이 존재하는지 확인 후 문장 생성
        if 0 <= answer < len(choices):  
            sentences.append(f"Question: {question}\nAnswer: {choices[answer]}")
        else:
            print(f"Warning: Skipping index {i}, invalid answer index {answer}")

    encoded_sentences = [tokenizer.encode(sent, add_special_tokens=False) for sent in sentences]
    sentence_lengths = [len(sent) for sent in encoded_sentences]
    testenc = tokenizer("\n\n".join(sentences), return_tensors='pt')
    if prompt_portion : 
        return None,(testenc, sentence_lengths)  # 문장별 길이 정보 추가
    else:
        return None,testenc

def get_arc_challenge_test(nsamples, seed, seqlen, model, tokenizer, batch_size,prompt_portion):
    testdata= load_dataset("ai2_arc", 'ARC-Challenge', split='validation')
    tokenizer.pad_token = tokenizer.eos_token  
    
    # sentences = testdata['question']

    sentences = []
    for i in range(nsamples):
        question = testdata['question'][i]
        choices = testdata['choices'][i]['text']  # 선택지 리스트
        answerKey = testdata['answerKey'][i]

        # answerKey를 숫자로 변환
        if answerKey in ['A', '1']:
            answer = 0
        elif answerKey in ['B', '2']:
            answer = 1
        elif answerKey in ['C', '3']:
            answer = 2
        elif answerKey in ['D', '4']:
            answer = 3
        elif isinstance(answerKey, int):
            answer = answerKey - 1  # 정수인 경우 조정
        else:
            print('i: ', i)
            print(testdata[i])
            raise ValueError("Invalid answerKey format.")

        # 정답이 존재하는지 확인 후 문장 생성
        if 0 <= answer < len(choices):  
            sentences.append(f"Question: {question}\nAnswer: {choices[answer]}")
        else:
            print(f"Warning: Skipping index {i}, invalid answer index {answer}")

    encoded_sentences = [tokenizer.encode(sent, add_special_tokens=False) for sent in sentences]
    sentence_lengths = [len(sent) for sent in encoded_sentences]
    testenc = tokenizer("\n\n".join(sentences), return_tensors='pt')
    if prompt_portion : 
        return None,(testenc, sentence_lengths)  # 문장별 길이 정보 추가
    else:
        return None,testenc


def get_c4(nsamples, seed, seqlen, model, tokenizer, batch_size,prompt_portion):
   
    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    new_trainloader = []
    num_batches = nsamples // batch_size + (int)(nsamples % batch_size > 0)
    for i in range(0, num_batches):
        start =  i * batch_size
        end = min(start + batch_size, nsamples)
        batched_inp = []
        batched_tar = []
        for j in range(start, end):
            batched_inp.append(trainloader[j][0])
            batched_tar.append(trainloader[j][1])
        batched_inp = torch.cat(batched_inp)
        batched_tar = torch.cat(batched_tar)
        new_trainloader.append((batched_inp, batched_tar))
    del trainloader
    trainloader = new_trainloader
    del new_trainloader

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    valenc = TokenizerWrapper(valenc)
    if prompt_portion:
        encoded_sentences = [tokenizer.encode(sent, add_special_tokens=False) for sent in valdata['text']]
        sentence_lengths = [len(sent) for sent in encoded_sentences]
        return None,(valenc, sentence_lengths)  # 문장별 길이 정보 추가
    return trainloader, valenc

def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None, model='', batch_size=1,prompt_portion=False):
    if tokenizer is None:
        tokenizer = get_tokenizer(model)
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model, tokenizer, batch_size,prompt_portion)
    if 'c4' in name:
        return get_c4(nsamples, seed, seqlen, model, tokenizer, batch_size,prompt_portion)
    if 'winogrande' in name:
        return get_winogrande_test(nsamples, seed, seqlen, model, tokenizer, batch_size,prompt_portion)
    if 'piqa' in name:
        return get_piqa_test(nsamples, seed, seqlen, model, tokenizer, batch_size,prompt_portion)
    if 'arc_easy' in name:
        return get_arc_easy_test(nsamples, seed, seqlen, model, tokenizer, batch_size,prompt_portion)
    if 'arc_challenge' in name:
        return get_arc_challenge_test(nsamples, seed, seqlen, model, tokenizer, batch_size,prompt_portion)
    if 'boolq' in name:
        return get_boolq_test(nsamples, seed, seqlen, model, tokenizer, batch_size,prompt_portion)
 
    else:
        print(dsfsd)


def get_wikitext2_trainenc(seed, nsamples, seqlen, model, tokenizer, batch_size):
    
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    traindata = traindata.shuffle(seed=6489)
    trainenc = tokenizer("\n\n".join(traindata[:nsamples]['text']), return_tensors='pt')

    return trainenc






def get_alpaca_trainenc(seed, nsamples, seqlen, model, tokenizer, batch_size):
    traindata = load_dataset("yahma/alpaca-cleaned", split='train')
    traindata = traindata.shuffle(seed=seed)

    combined_input = []
    for i in range(nsamples):
        instruction = traindata['instruction'][i]
        input_ = traindata['input'][i]
        output = traindata['output'][i]
        combined_input.append(f"Instruction: {instruction}\nIntput: {input_}\nOutput: {output}")
    trainenc = tokenizer("\n\n".join(combined_input), return_tensors='pt')
    # print('trainenc: ',trainenc.input_ids)
    # print('trainenc: ',trainenc.input_ids.shape)
    return trainenc


# def get_piqa_trainenc(seed, nsamples, seqlen, model, tokenizer, batch_size):
    
#     traindata = load_dataset('piqa', split='train')
#     traindata = traindata.shuffle(seed=seed)

#     combined_input = []
#     for i in range(nsamples):
#         goal = traindata['goal'][i]
#         sol1 = traindata['sol1'][i]
#         sol2 = traindata['sol2'][i]
#         combined_input.append(f"Goal: {goal}\nSolution 1: {sol1}\nSolution 2: {sol2}")
#     trainenc = tokenizer("\n\n".join(combined_input), return_tensors='pt')
#     print('trainenc: ',trainenc.input_ids)
#     print('trainenc: ',trainenc.input_ids.shape)
#     return trainenc
def get_winogrande_trainenc(seed, nsamples, seqlen, model, tokenizer, batch_size):
    
    traindata = load_dataset('winogrande', 'winogrande_xl', split='train')
    traindata = traindata.shuffle(seed=seed)
    trainenc = tokenizer("\n\n".join(traindata[:nsamples]['sentence']), return_tensors='pt')

    return trainenc

def get_piqa_trainenc(seed, nsamples, seqlen, model, tokenizer, batch_size,test=False):
    
    if test:
        data= load_dataset('piqa', split='test')
        data = data.shuffle(seed=seed)
        print('Test Data Loaded')
    else: 
        data = load_dataset('piqa', split='train')
        data = data.shuffle(seed=seed)
        print('Train Data Loaded')

    combined_input = []
    for i in range(nsamples):
        goal = data['goal'][i]
        # sol1 = traindata['sol1'][i]
        # sol2 = traindata['sol2'][i]
        label = data['label'][i]
        sol=f'sol{label+1}'
        combined_input.append(f"Question: {goal}\nAnswer: {data[sol][i]}")
    trainenc = tokenizer("\n\n".join(combined_input), return_tensors='pt')
    print('trainenc: ',trainenc.input_ids)
    print('trainenc: ',trainenc.input_ids.shape)
    if test: 
        return None,trainenc
    return trainenc


# def get_boolq_trainenc(seed, nsamples, seqlen, model, tokenizer, batch_size):
    
#     traindata = load_dataset('boolq', split='train')
#     traindata = traindata.shuffle(seed=seed)

#     combined_input = []
#     for i in range(nsamples):
#         passage = traindata['passage'][i]
#         question = traindata['question'][i]
#         combined_input.append(f"passage: {passage}\nquestion: {question}")
        
#     trainenc = tokenizer("\n\n".join(combined_input), return_tensors='pt')
#     print('trainenc: ',trainenc.input_ids)
#     print('trainenc: ',trainenc.input_ids.shape)
#     return trainenc

def get_boolq_trainenc(seed, nsamples, seqlen, model, tokenizer, batch_size):
    
    traindata = load_dataset('boolq', split='train')
    traindata = traindata.shuffle(seed=seed)

    combined_input = []
    for i in range(nsamples):
        passage = traindata['passage'][i]
        question = traindata['question'][i]
        answer = traindata['answer'][i]
        combined_input.append(f"passage: {passage}\nquestion: {question}\nanswer: {answer}")
        
    trainenc = tokenizer("\n\n".join(combined_input), return_tensors='pt')
    print('trainenc: ',trainenc.input_ids)
    print('trainenc: ',trainenc.input_ids.shape)
    return trainenc

# def get_arc_easy_trainenc(seed, nsamples, seqlen, model, tokenizer, batch_size):
    
#     traindata = load_dataset("ai2_arc", 'ARC-Easy', split='train')
#     traindata = traindata.shuffle(seed=seed)

#     combined_input = []
#     for i in range(nsamples):
#         question = traindata['question'][i]
#         text = traindata['choices'][i]['text']
#         combined_input.append(f"question: {question}\ntext: {text}")
#     trainenc = tokenizer("\n\n".join(combined_input), return_tensors='pt')
#     print('trainenc: ',trainenc.input_ids)
#     print('trainenc: ',trainenc.input_ids.shape)
#     return trainenc

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
            print(sdf)
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
        combined_input.append(f"question: {question}\ntext: {text}")
    trainenc = tokenizer("\n\n".join(combined_input), return_tensors='pt')
    print('trainenc: ',trainenc.input_ids)
    print('trainenc: ',trainenc.input_ids.shape)
    return trainenc



def get_bookcorpus_trainenc(seed, nsamples, seqlen, model, tokenizer, batch_size):
    
    traindata = load_dataset('bookcorpus', split='train')
    trainenc = tokenizer("\n\n".join(traindata[:nsamples]['text']), return_tensors='pt')
    return trainenc




def get_c4_trainenc(seed, nsamples, seqlen, model, tokenizer, batch_size):
    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
    traindata = traindata.shuffle(seed=seed)
    
    trainenc = tokenizer(' '.join(traindata[:nsamples]['text']), return_tensors='pt')

    trainenc = trainenc.input_ids

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    trainenc = TokenizerWrapper(trainenc)

    return trainenc


def get_hellaswag_trainenc(seed, nsamples, seqlen, model, tokenizer, batch_size):
    traindata = load_dataset("hellaswag",  split='train')
    traindata = traindata.shuffle(seed=seed)

    combined_input = []
    for i in range(nsamples):
        activity_label = traindata['activity_label'][i]
        ctx = traindata['ctx'][i]
        label = traindata['label'][i]
        # print(f'label: {label}')
        endings = traindata['endings'][i][int(label)]
        combined_input.append(f"Activity Label: {activity_label}\n ctx: {ctx}\n endings: {endings}")
    trainenc = tokenizer("\n\n".join(combined_input), return_tensors='pt')
    print('trainenc: ',trainenc.input_ids)
    print('trainenc: ',trainenc.input_ids.shape)
    return trainenc
    

def get_trainloaders(name, nsamples=128, seed=0, seqlen=512, model='', batch_size=1):
    tokenizer = get_tokenizer(model)
    if 'wikitext2' in name:
        return get_wikitext2_trainenc(seed, nsamples, seqlen, model, tokenizer, batch_size)
    elif 'c4' in name:
        return get_c4_trainenc(seed, nsamples, seqlen, model, tokenizer, batch_size)
    elif 'winogrande' in name:
        return get_winogrande_trainenc(seed, nsamples, seqlen, model, tokenizer, batch_size)
    elif 'piqa' in name:
        return get_piqa_trainenc(seed, nsamples, seqlen, model, tokenizer, batch_size)
    elif 'boolq' in name:
        return get_boolq_trainenc(seed, nsamples, seqlen, model, tokenizer, batch_size)
    elif 'arc_easy' in name:
        return get_arc_easy_trainenc(seed, nsamples, seqlen, model, tokenizer, batch_size)
    elif 'arc_challenge' in name:
        return get_arc_challenge_trainenc(seed, nsamples, seqlen, model, tokenizer, batch_size)
    elif 'bookcorpus' in name:
        return get_bookcorpus_trainenc(seed, nsamples, seqlen, model, tokenizer, batch_size)
    elif 'alpaca' in name:
        return get_alpaca_trainenc(seed, nsamples, seqlen, model, tokenizer, batch_size)
    elif 'hellaswag' in name:
        return get_hellaswag_trainenc(seed, nsamples, seqlen, model, tokenizer, batch_size)
    else:
        print(name)
        print(sdfs)






######################################################################################################
######################################################################################################
######################################################################################################




def get_full_wikitext2(seed, nsamples, seqlen, model, tokenizer, batch_size):
    
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    traindata = traindata.shuffle(seed=seed)
    # trainenc = tokenizer("\n\n".join(traindata[:nsamples]['text']), return_tensors='pt')
    
    return traindata

def get_full_c4(seed, nsamples, seqlen, model, tokenizer, batch_size):
    
    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    traindata = traindata.shuffle(seed=seed)
    # trainenc = tokenizer("\n\n".join(traindata[:nsamples]['text']), return_tensors='pt')
    
    return traindata


def get_full_winogrande(seed, nsamples, seqlen, model, tokenizer, batch_size):
    
    traindata = load_dataset('winogrande', 'winogrande_xl', split='train')
    traindata = traindata.shuffle(seed=seed)
    # trainenc = tokenizer("\n\n".join(traindata[:nsamples]['text']), return_tensors='pt')
    
    return traindata

def get_full_alpaca1(seed, nsamples, seqlen, model, tokenizer, batch_size):
    
    traindata = load_dataset("tatsu-lab/alpaca", split='train')
    traindata = traindata.shuffle(seed=seed)

    print(traindata)
    # trainenc = tokenizer("\n\n".join(traindata[:nsamples]['text']), return_tensors='pt')
    
    return traindata

def get_full_alpaca2(seed, nsamples, seqlen, model, tokenizer, batch_size):
    
    traindata = load_dataset("yahma/alpaca-cleaned", split='train')
    traindata = traindata.shuffle(seed=seed)

    combined_input = [f"Instruction: {instruction}\nInput: {input_}\nOutput: {output}"for instruction, input_, output in zip(traindata['instruction'], traindata['input'], traindata['output'])]   

    return combined_input
def get_full_alpaca3(seed, nsamples, seqlen, model, tokenizer, batch_size):
    
    traindata = load_dataset("yahma/alpaca-cleaned", split='train')
    traindata = traindata.shuffle(seed=seed)

    combined_input = [f"Instruction: {instruction}\n\n###Input: {input_}\n\n###Output: {output}"for instruction, input_, output in zip(traindata['instruction'], traindata['input'], traindata['output'])]   

    return combined_input
def get_full_alpaca4(seed, nsamples, seqlen, model, tokenizer, batch_size):
    
    traindata = load_dataset("yahma/alpaca-cleaned", split='train')
    traindata = traindata.shuffle(seed=seed)

    combined_input = [f"{instruction}\n{input_}\n{output}"for instruction, input_, output in zip(traindata['instruction'], traindata['input'], traindata['output'])]   

    return combined_input
######################################################################################################
######################################################################################################
######################################################################################################

def get_full_piqa1(seed, nsamples, seqlen, model, tokenizer, batch_size):
    
    traindata = load_dataset('piqa', split='train')
    traindata = traindata.shuffle(seed=seed)

    combined_input = [f"{goal}\n{sol1}\n{sol2}"for goal, sol1, sol2 in zip(traindata['goal'], traindata['sol1'], traindata['sol2'])]   

    return combined_input

def get_full_piqa2(seed, nsamples, seqlen, model, tokenizer, batch_size):
    
    traindata = load_dataset('piqa', split='train')
    traindata = traindata.shuffle(seed=seed)

    combined_input = [f"Goal: {goal}\nSolution 1: {sol1}\nSolution 2: {sol2}"for goal, sol1, sol2 in zip(traindata['goal'], traindata['sol1'], traindata['sol2'])]   

    return combined_input
def get_full_piqa3(seed, nsamples, seqlen, model, tokenizer, batch_size):
    
    traindata = load_dataset('piqa', split='train')
    traindata = traindata.shuffle(seed=seed)

    combined_input = [f"{goal} {sol1}"for goal, sol1, sol2 in zip(traindata['goal'], traindata['sol1'], traindata['sol2'])]   

    return combined_input
def get_full_piqa4(seed, nsamples, seqlen, model, tokenizer, batch_size):
    
    traindata = load_dataset('piqa', split='train')
    traindata = traindata.shuffle(seed=seed)

    combined_input = [f"goal: {goal}\nsol1: {sol1}\nsol2: {sol2}\nlable: {label} "for goal, sol1, sol2, label in zip(traindata['goal'], traindata['sol1'], traindata['sol2'], traindata['label'])]   

    return combined_input
def get_full_boolq1(seed, nsamples, seqlen, model, tokenizer, batch_size):
    
    traindata = load_dataset('boolq', split='train')
    traindata = traindata.shuffle(seed=seed)

    combined_input = [f"{passage}\n{question}"for passage, question in zip(traindata['passage'], traindata['question'])]   

    return combined_input

def get_full_boolq2(seed, nsamples, seqlen, model, tokenizer, batch_size):
    
    traindata = load_dataset('boolq', split='train')
    traindata = traindata.shuffle(seed=seed)
    combined_input = [f"passage: {passage}\nquestion: {question}"for passage, question in zip(traindata['passage'], traindata['question'])]   

    return combined_input
def get_full_boolq3(seed, nsamples, seqlen, model, tokenizer, batch_size):
    
    traindata = load_dataset('boolq', split='train')
    traindata = traindata.shuffle(seed=seed)
    # print(traindata[0])
    # print(dsfsd)
    combined_input = [f"question: {question}\nanswer: {answer}\npassage: {passage}"for passage, question, answer in zip(traindata['passage'], traindata['question'],traindata['answer'])]   

    return combined_input
def get_full_arc_easy1(seed, nsamples, seqlen, model, tokenizer, batch_size):
    
    traindata = load_dataset("ai2_arc", 'ARC-Easy', split='train')
    traindata = traindata.shuffle(seed=seed)

    combined_input = [
    f"{question}\n {choices['text']}"
    for question, choices in zip(traindata['question'], traindata['choices'])
    ]
    return combined_input


def get_full_arc_easy2(seed, nsamples, seqlen, model, tokenizer, batch_size):
    
    traindata = load_dataset("ai2_arc", 'ARC-Easy', split='train')
    traindata = traindata.shuffle(seed=seed)

    combined_input = [
    f"question: {question}\ntext: {choices['text']}"
    for question, choices in zip(traindata['question'], traindata['choices'])
    ]
    return combined_input

def get_full_arc_easy3(seed, nsamples, seqlen, model, tokenizer, batch_size):
    
    traindata = load_dataset("ai2_arc", 'ARC-Easy', split='train')
    traindata = traindata.shuffle(seed=seed)

    combined_input = [
    f"{question}"
    for question, choices in zip(traindata['question'], traindata['choices'])
    ]
    return combined_input

def get_full_arc_easy4(seed, nsamples, seqlen, model, tokenizer, batch_size):
    
    traindata = load_dataset("ai2_arc", 'ARC-Easy', split='train')
    traindata = traindata.shuffle(seed=seed)
    combined_input = [
    f"id: {_id}\nquestion: {question}\nchoices: {choices}\nanswerKey: {answerKey}"
    for _id, question, choices, answerKey in zip(traindata['id'],traindata['question'], traindata['choices'],traindata['answerKey'],)
    ]
    return combined_input

def get_full_arc_challenge1(seed, nsamples, seqlen, model, tokenizer, batch_size):
    
    traindata = load_dataset("ai2_arc", 'ARC-Challenge', split='train')

    traindata = traindata.shuffle(seed=seed)

    combined_input = [
    f"{question}\n {choices['text']}"
    for question, choices in zip(traindata['question'], traindata['choices'])
    ]
    return combined_input


def get_full_arc_challenge2(seed, nsamples, seqlen, model, tokenizer, batch_size):
    
    traindata = load_dataset("ai2_arc", 'ARC-Challenge', split='train')
    traindata = traindata.shuffle(seed=seed)

    combined_input = [
    f"question: {question}\ntext: {choices['text']}"
    for question, choices in zip(traindata['question'], traindata['choices'])
    ]
    return combined_input

def get_full_arc_challenge3(seed, nsamples, seqlen, model, tokenizer, batch_size):
    
    traindata = load_dataset("ai2_arc", 'ARC-Challenge', split='train')
    traindata = traindata.shuffle(seed=seed)

    combined_input = [
    f"{question}"
    for question, choices in zip(traindata['question'], traindata['choices'])
    ]
    return combined_input



def get_full_hellaswag1(seed, nsamples, seqlen, model, tokenizer, batch_size):
    
    traindata = load_dataset("hellaswag",  split='train')
    traindata = traindata.shuffle(seed=seed)

    combined_input = [
    f"Activity label: {activity_label}\n ctx: {ctx}\n endings: {endings[int(label)]}"
    for activity_label, ctx, ctx_a, ctx_b,endings, ind,label, source_id, split, split_type in zip(traindata['activity_label'], traindata['ctx'],traindata['ctx_a'],traindata['ctx_b'], traindata['endings'], traindata['ind'],traindata['label'] ,traindata['source_id'], traindata['split'], traindata['split_type'])
    ]

    return combined_input

def get_full_hellaswag2(seed, nsamples, seqlen, model, tokenizer, batch_size):
    
    traindata = load_dataset("hellaswag",  split='train')
    traindata = traindata.shuffle(seed=seed)

    combined_input = [
    f"Activity label: {activity_label}\n ctx: {ctx}\n ctx_a: {ctx_a}\n ctx_b: {ctx_b}\n endings: {endings}\n ind: {ind}\n label: {label}\n source_id: {source_id}\n split: {split}\n split_type: {split_type}"
    for activity_label, ctx, ctx_a, ctx_b, endings, ind,label, source_id, split, split_type in zip(traindata['activity_label'], traindata['ctx'],traindata['ctx_a'],traindata['ctx_b'], traindata['endings'], traindata['ind'],traindata['label'] ,traindata['source_id'], traindata['split'], traindata['split_type'])
    ]

    return combined_input



######################################################################################################
######################################################################################################
######################################################################################################



def get_full_data(name, nsamples=128, seed=0, seqlen=512, model='', batch_size=1):
    tokenizer = get_tokenizer(model)
    if 'wikitext4' in name:
        return get_full_wikitext2(seed, nsamples, seqlen, model, tokenizer, batch_size)
    if 'c4' in name:
        return get_full_c4(seed, nsamples, seqlen, model, tokenizer, batch_size)      
    if 'alpaca1' in name:
        return get_full_alpaca1(seed, nsamples, seqlen, model, tokenizer, batch_size)          
    if 'alpaca2' in name:
        return get_full_alpaca2(seed, nsamples, seqlen, model, tokenizer, batch_size)      
    if 'alpaca3' in name:
        return get_full_alpaca3(seed, nsamples, seqlen, model, tokenizer, batch_size)      
    if 'alpaca4' in name:
        return get_full_alpaca4(seed, nsamples, seqlen, model, tokenizer, batch_size)       
    if 'piqa1' in name:
        return get_full_piqa1(seed, nsamples, seqlen, model, tokenizer, batch_size)
    if 'piqa2' in name:
        return get_full_piqa2(seed, nsamples, seqlen, model, tokenizer, batch_size)
    if 'piqa3' in name:
        return get_full_piqa3(seed, nsamples, seqlen, model, tokenizer, batch_size)
    if 'piqa4' in name:
        return get_full_piqa4(seed, nsamples, seqlen, model, tokenizer, batch_size)        
    if 'winogrande' in name:
        return get_full_winogrande(seed, nsamples, seqlen, model, tokenizer, batch_size)
    if 'boolq1' in name:
        return get_full_boolq1(seed, nsamples, seqlen, model, tokenizer, batch_size)
    if 'boolq2' in name:
        return get_full_boolq2(seed, nsamples, seqlen, model, tokenizer, batch_size)        
    if 'boolq3' in name:
        return get_full_boolq3(seed, nsamples, seqlen, model, tokenizer, batch_size)               
    if 'arc_easy1' in name:
        return get_full_arc_easy1(seed, nsamples, seqlen, model, tokenizer, batch_size)
    if 'arc_easy2' in name:
        return get_full_arc_easy2(seed, nsamples, seqlen, model, tokenizer, batch_size)
    if 'arc_easy3' in name:
        return get_full_arc_easy3(seed, nsamples, seqlen, model, tokenizer, batch_size)
    if 'arc_easy4' in name:
        return get_full_arc_easy4(seed, nsamples, seqlen, model, tokenizer, batch_size)        
    if 'arc_challenge1' in name:
        return get_full_arc_challenge1(seed, nsamples, seqlen, model, tokenizer, batch_size)
    if 'arc_challenge2' in name:
        return get_full_arc_challenge2(seed, nsamples, seqlen, model, tokenizer, batch_size)
    if 'arc_challenge3' in name:
        return get_full_arc_challenge3(seed, nsamples, seqlen, model, tokenizer, batch_size)      
    if 'hellaswag1' in name:
        return get_full_hellaswag1(seed, nsamples, seqlen, model, tokenizer, batch_size)      
    if 'hellaswag2' in name:
        return get_full_hellaswag2(seed, nsamples, seqlen, model, tokenizer, batch_size)      
    else:
        print(sdfs)