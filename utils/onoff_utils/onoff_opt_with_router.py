import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union

from transformers import (
    AutoTokenizer, 
    OPTForCausalLM, 
    OPTConfig, 
    AutoConfig, 
    AutoModelForCausalLM, 
    BertConfig, 
    BertModel,
    BertForSequenceClassification
)
from ...generation import GenerationMixin

class CustomBertForSequenceClassification(BertForSequenceClassification):
    _no_split_modules = ["bert"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class OnOff_OPTDecoderLayer(nn.Module):
    def __init__(self, original_decoder_layer):
        super().__init__()
        self.embed_dim = original_decoder_layer.embed_dim

        self.self_attn = original_decoder_layer.self_attn

        self.do_layer_norm_before = original_decoder_layer.do_layer_norm_before
        self.dropout = original_decoder_layer.dropout
        self.activation_fn = original_decoder_layer.activation_fn

        self.self_attn_layer_norm = original_decoder_layer.self_attn_layer_norm
        self.fc1 = original_decoder_layer.fc1
        self.fc2 = original_decoder_layer.fc2
        self.final_layer_norm = original_decoder_layer.final_layer_norm

        self.pass_layer = False
    
    def turn_off(self):
        self.pass_layer = True
    
    def turn_on(self):
        self.pass_layer = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`, *optional*): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        # skip this decoder layer
        if self.pass_layer:
            outputs = (hidden_states,)

            if output_attentions:
                outputs += (None,)

            if use_cache:
                outputs += (past_key_value,)

            return outputs

        # else normal forward
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        
        if residual.device != hidden_states.device:
            residual = residual.to(hidden_states.device)

        hidden_states = residual + hidden_states

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)

        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = (residual + hidden_states).view(hidden_states_shape)

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class On_Off_OPTForCausalLM(OPTPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config,router_pretrained_path='None' ):
        super().__init__(config)
        self.model = OPTModel(config)

        # the lm_head weight is automatically tied to the embed tokens weight
        self.lm_head = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        self.tokenizer = AutoTokenizer.from_pretrained("facebook/opt-6.7b")
        self.bert_tokenizer = AutoTokenizer.from_pretrained(router_pretrained_path)
        self.router = CustomBertForSequenceClassification.from_pretrained(router_pretrained_path,num_labels=10)

        self.router.eval()


    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model.decoder = decoder

    def get_decoder(self):
        return self.model.decoder

    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:


        input_text = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]
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


        _, topk_indices = torch.topk(router_logits, 1, dim=-1, largest=True)
        predicted_label = topk_indices.item()  # (1,1) -> int


        skip_layer = []
        with open('codes/6_opt/opt_layer_list_10.csv', mode="r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                row_set_str = row[0]
                row_idx_str = row[1]

                if int(row_idx_str) == predicted_label:

                    my_tuple = ast.literal_eval(row_set_str)
                    skip_layer = list(my_tuple)

        for i in range(len(skip_layer)):
            block_idx=skip_layer[i]
            self.model.decoder.layers[block_idx].turn_off



        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.lm_head(outputs[0]).contiguous()

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        for i in range(len(skip_layer)):
            block_idx=skip_layer[i]
            self.model.decoder.layers[block_idx].turn_on


        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


def block_replace(model):

    num_layers = len(model.model.decoder.layers)
    for i in range(num_layers):
        model.model.decoder.layers[i] = OnOff_OPTDecoderLayer(model.model.decoder.layers[i])
    print("Replacement complete.")

    return model

def turn_off(model, block_idx):

    model.model.decoder.layers[block_idx].turn_off()

def turn_on(model, block_idx):

    model.model.decoder.layers[block_idx].turn_on()

def scan(model, num_layers):

    alive_list = []
    skip_list = []

    for i in range(num_layers):
        if model.model.decoder.layers[i].pass_layer == True:
            skip_list.append(i)
        elif model.model.decoder.layers[i].pass_layer == False:
            alive_list.append(i)
            
    print(
        f"pass layer: {skip_list}\n"
        f"do layer: {alive_list}"
        )