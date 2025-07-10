import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, 
    LlamaForCausalLM, 
    LlamaConfig, 
    AutoConfig, 
    AutoModelForCausalLM, 
    BertConfig, 
    BertModel,
    BertForSequenceClassification
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from utils.model_utils import get_llm
from utils.onoff_utils.onoff import block_replace, turn_off, turn_on

import csv
import ast
import re

class CustomBertForSequenceClassification(BertForSequenceClassification):
    _no_split_modules = ["bert"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class AdaptiveSLEBForCausalLM(LlamaForCausalLM):
    def __init__(self, config, name=None, base_model=None, router_model=None, router_path="./model/v3/jy_diff_all/"):
        
        super().__init__(config)
        self.n_layers = config.num_hidden_layers
        
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
        self.bert_tokenizer = AutoTokenizer.from_pretrained(router_path)

        if base_model is not None:
            self.model = base_model
        else:
            print('no llm model loaded...')
        if router_model is not None:

            self.router = router_model
        else:
            print('no trained router loaded...')
        self.router.eval()

        if name is None:
            self.name = 'v23_adaptSLEB'
        self.seqlen = 2048
        self.input_set={}
        # self.excluded_indices=[0,2,4,6,7]
        self.init_count()
    def get_skip_mask(self, router_logits):

        probabilities = router_logits  # shape: (1, 10)

        # excluded_indices = torch.tensor(self.excluded_indices)
        # probabilities[:, excluded_indices] = float('-inf')
        
        # print(probabilities)
        _, topk_indices = torch.topk(probabilities, 1, dim=-1, largest=True)
        predicted_label = topk_indices.item()  # (1,1) -> int
        # print(predicted_label)
        # if predicted_label in excluded_indices:
        #     print(probabilities)
        #     print(sdfsf)

        self.add_count(predicted_label)
        skip_layer = []

        with open('codes/llama_layer_list_6_advanced_tasks.csv', mode="r", newline="", encoding="utf-8") as f:
        # with open('codes/5_adaptive_cluster/clustered_layer_list_10.csv', mode="r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                row_set_str = row[0]
                row_idx_str = row[1]

                if int(row_idx_str) == predicted_label:
                    my_tuple = ast.literal_eval(row_set_str)
                    skip_layer = list(my_tuple)
                    break

        return skip_layer
    def init_count(self):
        print('initial count!')
        self.count = [0] * 10
        self.total = 0
    def add_count(self, predicted_label):
        self.count[predicted_label] += 1
        self.total += 1
    def print_count(self):
        print('#'*20)
        print('returning count!')
        print(self.count)
        print(self.total)
        print('#'*20)


    def forward(self, input_ids=None, attention_mask=None, skip_layer=None, **kwargs):
        seq_len = input_ids.size(1)
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones((1, seq_len), device=device)

        if skip_layer is None:
            with torch.no_grad():
                input_text = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]
                
                answer_index = input_text.find("Answer")

                if answer_index != -1:
                    question_input = input_text[:answer_index]
                else:
                    match = re.search(r":\s*([^\.]*\.)", input_text)
                    if match:
                        question_input = match.group(1).strip()
                    else:
                        words = input_text.split()
                        question_input = " ".join(words[:7])

                if question_input in self.input_set:       
                    skip_layer = self.input_set[question_input]
                
                else:
        
                    self.input_prompt = question_input
                    bert_inputs = self.bert_tokenizer(
                        input_text, 
                        return_tensors='pt', 
                        padding=True, 
                        truncation=True, 
                        max_length=512
                    ).to(device)

                    router_outputs = self.router(
                        input_ids=bert_inputs['input_ids'],
                        attention_mask=bert_inputs['attention_mask']
                    )
                    router_logits = router_outputs.logits  # shape: (batch=1, num_labels=10)

                    skip_layer = self.get_skip_mask(router_logits)
                    self.input_set[question_input] = skip_layer
        else:
            skip_mask = skip_mask.to(device)

        for idx in skip_layer:
            turn_off(self.model, idx)
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        for idx in skip_layer: 
            turn_on(self.model, idx)

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            hidden_states=None,
            attentions=None,
            # cross_attentions=None
        )


def v23_adaptSLEB(model_name="meta-llama/Meta-Llama-3.1-8B", router_path="result/9_llama/router/10/2_onlylog/MSE/5"):

    print('loading model v23...')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = get_llm(model_name)
    base_model.name =model_name
    base_model = block_replace(base_model)
    config = AutoConfig.from_pretrained(model_name)
    router_model = CustomBertForSequenceClassification.from_pretrained(router_path,num_labels=10)

    model = AdaptiveSLEBForCausalLM(
        config=config,
        base_model=base_model,
        router_model=router_model,
        name='v23_adaptSLEB',
        router_path=router_path
    )

    model.to('cuda')
    model.rank = 0
    model.world_size = 1

    return model, tokenizer
