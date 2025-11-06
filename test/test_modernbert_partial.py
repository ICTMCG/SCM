import os, sys, datasets, torch, json, jsonlines, datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from argparse import ArgumentParser
from functools import partial
import numpy as np
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from model.ModernBertForSeqCLF import ModernBERTForSequenceClassification
from utils.metrics import compute_metrics_for_dual as compute_metrics
from utils.metrics import convert_to_serializable
from utils.metrics import Prediction

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.cuda.empty_cache()
parser = ArgumentParser()
parser.add_argument(
    "--model-path",
    type=str,
    required=True,
)
parser.add_argument(
    "--tokenizer-path",
    type=str,
    required=True,
)
parser.add_argument("--data-dir", type=str, required=True)
parser.add_argument("--output-dir", type=str, required=True)
args = parser.parse_args()

data_dir = args.data_dir
max_length = 4096
model_path = args.model_path
tokenizer_path = args.tokenizer_path
model = ModernBERTForSequenceClassification.from_pretrained(
    model_path, attn_implementation="sdpa"
)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model.config.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)


def load_dataset(data_dir):
    with jsonlines.open(os.path.join(data_dir, "test.jsonl")) as f:
        for line in f:
            ret = {
                "text": [],
                "sequence_labels": [],
            }
            resp_start = np.where(np.array(line["token_labels"]) == 0)[0][0]
            for i in range(resp_start + 1, len(line["text"]) + 1):
                ret["text"].append(line["text"][:i])
                ret["sequence_labels"].append(line["sequence_labels"])
            ret["max_len"] = [len(line["text"])]
            assert ret["max_len"][0] == len(ret["text"][-1])
            yield ret


def preprocess_function(examples, max_len):
    tokenized_inputs = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_len,
        is_split_into_words=True,
        add_special_tokens=False,
    )
    return {
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
        "sequence_labels": examples["sequence_labels"],
    }


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
training_args = TrainingArguments(
    output_dir="./results", per_device_eval_batch_size=128, dataloader_num_workers=8
)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
)

now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
ds_all = load_dataset(data_dir)
collect_token_scores = []
collect_token_labels = []
collect_sequence_labels = []
idx = 0
for ds in ds_all:
    idx += 1
    print(idx, now)
    max_len = ds.pop("max_len")[0]
    proc_func = partial(preprocess_function, max_len=max_len)
    test_set = datasets.Dataset.from_dict(ds)
    test_set = test_set.map(proc_func, batched=True)
    test_set = test_set.remove_columns(["text"])
    predictions = trainer.predict(test_set)
    token_predictions = predictions.predictions
    token_labels = predictions.label_ids
    sequence_label = int(any(token_labels))
    collect_token_scores.append(token_predictions)
    token_labels[0] = -100
    token_labels[-1] = -100
    collect_token_labels.append(token_labels)
    collect_sequence_labels.append(int(any(token_labels)))

print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
max_len = max([x.shape[0] for x in collect_token_scores])
print(f"max_len: {max_len}")

for i in range(len(collect_token_scores)):
    collect_token_scores[i] = np.pad(
        collect_token_scores[i],
        ((0, max_len - collect_token_scores[i].shape[0]), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    collect_token_labels[i] = np.pad(
        collect_token_labels[i],
        (0, max_len - collect_token_labels[i].shape[0]),
        mode="constant",
        constant_values=-100,
    )
    collect_token_scores[i] = np.expand_dims(collect_token_scores[i], axis=0)
    collect_token_labels[i] = np.expand_dims(collect_token_labels[i], axis=0)

collect_token_scores = np.concatenate(collect_token_scores, axis=0)
collect_token_labels = np.concatenate(collect_token_labels, axis=0)
print(collect_token_scores.shape, collect_token_labels.shape)

sequence_labels = np.ones_like(np.array(collect_sequence_labels))
sequence_scores = np.ones((sequence_labels.shape[0], 2))

predictions = Prediction(predictions=(collect_token_scores, sequence_scores), label_ids=(collect_token_labels, sequence_labels))

for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
    token_threshold = threshold

    results = compute_metrics(
        predictions,
        token_threshold,
        0.5,
    )

    with open(
        os.path.join(args.output_dir, f"test_metrics_{int(threshold*10)}.json"), "w"
    ) as f:
        json.dump(convert_to_serializable(results), f, indent=2)

np.savez(
    "test_predictions.npz",
    token_predictions=collect_token_scores,
    token_labels=collect_token_labels,
)
