"""
Refer to
https://github.com/tloen/alpaca-lora/blob/main/finetune.py
"""

import argparse
import os
import random
import ast

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM,AutoTokenizer
from datasets import load_dataset
from LLMPruner.peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
)
from LLMPruner.utils.prompter import Prompter, ZeroPrompter
from utils.onoff_utils.onoff import block_replace, turn_off, turn_on


def set_seed(random_seed=1234):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def main(args):
    os.environ["WANDB_PROJECT"] = args.wandb_project
    set_seed(args.seed)
    skip_layer = ast.literal_eval(args.skip_layer)
    lora_target_modules = []
    
    target_sub_names = ["self_attn.q_proj",
                    "self_attn.k_proj",
                    "self_attn.v_proj",
                    "self_attn.o_proj",
                    "mlp.gate_proj",
                    "mlp.down_proj",
                    "mlp.up_proj"]

    print(f"skip layer: {skip_layer}")
    for idx in range(32):
        if idx not in skip_layer:
            # print(idx)
            for subname in target_sub_names:
                # 예: "model.layers.20.self_attn.q_proj"
                full_name = f"model.layers.{idx}.{subname}"
                lora_target_modules.append(full_name)

    # model, tokenizer, description = get_model(
    #     base_model=args.base_model,
    #     ckpt=args.ckpt,
    #     lora_ckpt=None,
    #     tokenizer=args.tokenizer,
    #     model_type=args.model_type,
    #     device=args.device,
    #     fix_decapoda_config=args.fix_decapoda_config,
    #     use_bfloat=args.use_bfloat,
    # )
    print(f'loading {args.base_model}')
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, low_cpu_mem_usage=True
    )
    model.name = 'llama'
    model = block_replace(model)
    for idx in skip_layer:
        turn_off(model, idx)
    gradient_accumulation_steps = args.batch_size // args.micro_batch_size

    if not args.no_instruction:
        prompter = Prompter(args.prompt_template_name)
    else:
        prompter = ZeroPrompter()
        
    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:  # e.g., llama-3 (https://huggingface.co/meta-llama/Meta-Llama-3-8B/discussions/36#662315ec5d73c1b9f90482ea)
            tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=args.cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < args.cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not args.train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=args.add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if args.add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    def split_and_tokenizer(test_data, tokenizer, seq_len, field_name):
        test_ids = tokenizer(
            "\n\n".join(test_data[field_name]), return_tensors="pt"
        ).input_ids[0]
        nsamples = test_ids.numel() // seq_len

        test_set = []
        for i in range(nsamples):
            batch = test_ids[(i * seq_len) : ((i + 1) * seq_len)]
            test_set.append({"input_ids": batch, "labels": batch})
        return test_set

    # Prepare For LoRA
    model = prepare_model_for_int8_training(model)
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.enable_input_require_grads() 
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    # Load Train Dataset
    data = load_dataset(args.data_path)
    train_val = data["train"].train_test_split(
        test_size=args.val_set_size, shuffle=True, seed=42
    )
    train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
    val_data = {
        args.data_path: train_val["test"].shuffle().map(generate_and_tokenize_prompt),
    }

    # Load Extra Validation Dataset
    if args.extra_val_dataset:
        from dataset import get_ptb, get_wikitext2

        seq_len = 128
        for extra_dataset in args.extra_val_dataset.split(","):
            if "wikitext2" in extra_dataset:
                _, test_data = get_wikitext2()
                test_data = split_and_tokenizer(
                    test_data, tokenizer, seq_len, field_name="text"
                )
            if "ptb" in extra_dataset:
                _, test_data = get_ptb()
                test_data = split_and_tokenizer(
                    test_data, tokenizer, seq_len, field_name="sentence"
                )
            val_data[extra_dataset] = test_data

    fp16_flag = True
    bf16_flag = False
    if args.use_bfloat:
        model = model.bfloat16()
        fp16_flag = False
        bf16_flag = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            fp16=fp16_flag,
            bf16=bf16_flag,
            logging_steps=10,
            logging_first_step=True,
            optim="adamw_torch",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=100,
            save_steps=200,
            output_dir=args.output_dir,
            save_total_limit=20,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=None,
            group_by_length=args.group_by_length,
            report_to="wandb",
            run_name=args.output_dir.split("/")[-1],
            metric_for_best_model="{}_loss".format(args.data_path),
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    model.state_dict = old_state_dict
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.save_lora_merge:
        model = None

        set_seed(args.seed)
        model, tokenizer, description = get_model(
            base_model=args.base_model,
            ckpt=args.ckpt,
            lora_ckpt=None,
            tokenizer=args.tokenizer,
            model_type=args.model_type,
            device=args.device,
            use_bfloat=args.use_bfloat,
        )

        from LLMPruner.peft import PeftModel

        lora_model = PeftModel.from_pretrained(
            model, args.output_dir, torch_dtype=torch.float16
        )

        print("lora merged ")
        lora_model = lora_model.merge_and_unload()

        lora_model_sd = lora_model.state_dict()
        deloreanized_sd = {
            k.replace("base_model.model.", ""): v
            for k, v in lora_model_sd.items()
            if "lora" not in k
        }
        model.save_pretrained(
            os.path.join(args.output_dir + "_lora_merge_fp16"),
            state_dict=deloreanized_sd,
            max_shard_size="10GB",
        )
        tokenizer.save_pretrained(os.path.join(args.output_dir + "_lora_merge_fp16"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model Type&Path
    parser.add_argument(
        "--base_model",
        type=str,
        default="output_tune/llama-1-7b/ppl_n10/rm_6_blocks_lora_merge_fp16",
        help="base model name",
    )
    parser.add_argument(
        "--tokenizer", type=str, default=None, help="if None, base model name is used"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="pretrain",
        choices=["pretrain", "pruneLLM", "tune_pruneLLM"],
    )
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument("--save_lora_merge", action="store_true")
    parser.add_argument(
        "--fix_decapoda_config",
        default=False,
        action="store_true",
        help="fix tokenizer config of baffo32/decapoda-research-llama-7B-hf",
    )
    parser.add_argument("--use_bfloat", default=False, action="store_true")

    parser.add_argument(
        "--data_path", type=str, default="yahma/alpaca-cleaned", help="data path"
    )
    parser.add_argument(
        "--extra_val_dataset",
        type=str,
        default=None,
        help='validation datasets. Split with ","',
    )
    parser.add_argument(
        "--output_dir", type=str, default="./lora-alpaca", help="output directory"
    )
    parser.add_argument("--seed", type=int, default=1234)

    # Training Hyperparameters
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument(
        "--micro_batch_size", type=int, default=4, help="micro batch size"
    )
    parser.add_argument("--num_epochs", type=int, default=5, help="number of epochs")
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4, help="learning rate"
    )
    parser.add_argument("--cutoff_len", type=int, default=256, help="cutoff length")
    parser.add_argument(
        "--val_set_size", type=int, default=2000, help="validation set size"
    )
    parser.add_argument(
        "--prompt_template_name",
        type=str,
        default="alpaca",
        help="The prompt template to use, will default to alpaca.",
    )
    parser.add_argument(
        "--no_instruction",
        action="store_true",
        default=False,
        help="Whether to use the instruction template or not.",
    )

    # Lora Configuration
    parser.add_argument("--lora_r", type=int, default=8, help="lora r")
    parser.add_argument("--lora_alpha", type=int, default=16, help="lora alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="lora dropout")
    # parser.add_argument(
    #     "--lora_target_modules",
    #     type=str,
    #     default="q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj",
    #     help="lora target modules",
    # )

    # llm hyperparameters
    parser.add_argument(
        "--train_on_inputs",
        default=False,
        action="store_true",
        help="Train on inputs. If False, masks out inputs in loss",
    )
    parser.add_argument("--add_eos_token", default=False, action="store_true")
    parser.add_argument(
        "--group_by_length",
        default=False,
        action="store_true",
        help="faster, but produces an odd training loss curve",
    )

    # wandb params
    parser.add_argument("--wandb_project", type=str, default="")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        help="either training checkpoint or final adapter",
    )
    parser.add_argument("--skip_layer", type=str, default=None)

    args = parser.parse_args()

    main(args)

    # # 2) LoRA 어댑터만 로드
    # print("[+] Load LoRA adapter...")
    # model = PeftModel.from_pretrained(base_model, lora_model_dir)
