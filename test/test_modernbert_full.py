import os, sys, datasets, json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from argparse import ArgumentParser
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from model.ModernBertForSeqCLF import ModernBERTForSequenceClassification
from utils.metrics import compute_metrics_for_seq as compute_metrics
from utils.metrics import convert_to_serializable

parser = ArgumentParser()
parser.add_argument(
    "--model-path",
    type=str,
    required=True,
)
parser.add_argument("--data-dir", type=str, required=True)
parser.add_argument("--output-dir", type=str, required=True)
args = parser.parse_args()
args.max_length = 4096
model = ModernBERTForSequenceClassification.from_pretrained(args.model_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
model.config.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)


def load_dataset(args):
    data_files = {
        "test": os.path.join(args.data_dir, "test.jsonl"),
    }
    ds = datasets.load_dataset("json", data_files=data_files)
    return ds


def preprocess_function(examples):
    tokenized_inputs = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=args.max_length,
        is_split_into_words=True,
        add_special_tokens=False,
    )
    return {
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
        "sequence_labels": examples["sequence_labels"],
    }


ds = load_dataset(args)
tokenized_ds = ds.map(preprocess_function, batched=True)
tokenized_ds = tokenized_ds.remove_columns("text")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
training_args = TrainingArguments(
    output_dir="./results",
    per_device_eval_batch_size=8,
)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
)
predictions = trainer.predict(tokenized_ds["test"])
sequence_predictions = predictions.predictions
sequence_labels = predictions.label_ids

for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:

    results = compute_metrics(
        sequence_predictions,
        sequence_labels,
        threshold,
    )

    with open(
        os.path.join(args.output_dir, f"test_metrics_{int(threshold*10)}.json"), "w"
    ) as f:
        json.dump(convert_to_serializable(results), f, indent=2)

np.savez(
    os.path.join(args.output_dir, f"test_predictions.npz"),
    sequence_predictions=sequence_predictions,
    sequence_labels=sequence_labels,
)
