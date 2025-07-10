import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import random
import os
from torch.utils.data import Dataset
import argparse
import ast

def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    # The below two lines are for deterministic algorithm behavior in CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import csv

class PromptDataset(Dataset):

    def __init__(self, csv_files):
        self.data = []
        for csv_file in csv_files:
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for idx, row in enumerate(reader):
                    prompt = row["inputs"]
                    likelihood = row["likelihood"]

                    try:
                        likelihood = ast.literal_eval(likelihood)
                        if not isinstance(likelihood, list) or not all(isinstance(x, float) for x in likelihood):
                            raise ValueError("Parsed likelihood is not a list of floats.")

                        if len(likelihood) != 10:
                            raise ValueError(f"Row {idx} has invalid likelihood length {len(likelihood)}.\nLikelihood: {likelihood}")

                        self.data.append((prompt, likelihood))

                    except Exception as e:
                        print(f"❌ Error in row {idx}: {e}")
                        raise  




    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

import pandas as pd

def prompt_dataset_to_dataframe(prompt_dataset):
    """
    Convert PromptDataset to pandas DataFrame
    """
    rows = []
    for (prompt, likelihood) in prompt_dataset:
        rows.append({
            'text': prompt,
            'label': likelihood
        })
    df = pd.DataFrame(rows)
    return df

from datasets import Dataset as HFDataset, DatasetDict

def dataframe_to_dataset_dict(df, test_size=0.2, seed=42):
    """
    DataFrame -> HuggingFace Dataset -> split into train/test -> DatasetDict
    """
    hf_dataset = HFDataset.from_pandas(df, preserve_index=False)
    split_dataset = hf_dataset.train_test_split(test_size=test_size, seed=seed)
    
    dataset_dict = DatasetDict({
        "train": split_dataset["train"],
        "test": split_dataset["test"]
    })
    return dataset_dict

def create_dataset_dict_from_csv(csv_files, test_size=0.2, seed=42):
    """
    Create PromptDataset from csv_files -> convert to DatasetDict
    """
    prompt_dataset = PromptDataset(csv_files)
    df = prompt_dataset_to_dataframe(prompt_dataset)
    dataset_dict = dataframe_to_dataset_dict(df, test_size=test_size, seed=seed)
    return dataset_dict


def compute_metrics(p):

    predictions, labels = p  # both are np.ndarray

    mse = np.mean((predictions - labels) ** 2)

    mae = np.mean(np.abs(predictions - labels))

    rmse = np.sqrt(mse)

    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse
    }

from transformers import Trainer

class MSELossTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """
        We override the compute_loss method of the Trainer class
        to use our custom loss instead of the default
        cross entropy.
        """
        # print(
        #     f'input: {inputs}'
        # )
        labels = inputs.get("labels")
        # print('labels: ',labels)
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # print(logits)
        loss = F.mse_loss(logits, labels)

        return (loss, outputs) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser(description="Train BERT with MSE Loss.")
    parser.add_argument("--csv_files", nargs="+", default=["result/5_from30/likelihood.csv"],
                        help="Path(s) to CSV file(s). Can pass multiple.")
    parser.add_argument("--output_dir", type=str, default="result/5_router/router/30/zeroshot/likelihood_MSE",
                        help="Output directory for model checkpoints.")
    parser.add_argument("--num_classes", type=int, default=10, 
                        help="Number of classification labels.")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")

    args = parser.parse_args()
    
    set_seed(args.seed)
    print(f'epoch: {args.epochs}')

    dataset_dict = create_dataset_dict_from_csv(
        csv_files=args.csv_files,
        test_size=0.2,
        seed=args.seed
    )

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True)

    tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=args.num_classes
    )
    output_dir= f'{args.output_dir}/{args.epochs}'
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        evaluation_strategy='epoch',
        logging_steps=500,
        learning_rate=args.lr,
        save_strategy="no",
    )

    trainer = MSELossTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    results = trainer.evaluate()
    print("Evaluation Results:", results)


if __name__ == "__main__":
    main()
