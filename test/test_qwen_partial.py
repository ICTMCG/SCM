import os, sys, datasets, json, torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from argparse import ArgumentParser
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)
from model.modeling_qwen4dual_2CE_w_logic import QwenForDualTask
from utils.metrics import compute_metrics_for_dual as compute_metrics
from utils.metrics import convert_to_serializable
from hf_mtask_trainer import HfMultiTaskTrainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
parser = ArgumentParser()
parser.add_argument(
    "--model-path",
    type=str,
    required=True,
)
parser.add_argument("--data-dir", type=str, default="./data")
parser.add_argument("--output-dir", type=str, required=True)
args = parser.parse_args()
args.max_length = 4096


class ModelWrapper(QwenForDualTask):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        token_labels: torch.LongTensor | None = None,
        sequence_labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # 步骤 1: 从基础模型获取隐藏状态
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]

        # 步骤 2: 获取用于序列分类的最后一个非填充token的隐藏状态
        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None:
            last_non_pad_token = -1
        else:
            non_pad_mask = (input_ids != self.config.pad_token_id).to(torch.int32)
            last_non_pad_token = non_pad_mask.sum(dim=1) - 1

        pooled_hidden_states = hidden_states[
            torch.arange(batch_size, device=hidden_states.device), last_non_pad_token
        ]

        sequence_logits = self.sequence_classifier(pooled_hidden_states)

        #! Use sequence_clf for every token's prediction
        token_logits = self.sequence_classifier(hidden_states)

        # 步骤 4: 直接返回logits
        if not return_dict:
            return (token_logits, sequence_logits)

        return {
            "token_logits": token_logits,
            "sequence_logits": sequence_logits,
        }


model = ModelWrapper.from_pretrained(args.model_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)


def load_dataset(args):
    data_files = {
        "test": os.path.join(args.data_dir, "test.jsonl"),
    }
    ds = datasets.load_dataset("json", data_files=data_files)
    return ds


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            current_word = word_id
            label = labels[word_id] if word_id is not None else -100
            new_labels.append(label)
        elif word_id is not None:
            new_labels.append(-100)
        else:
            new_labels.append(-100)
    return new_labels


def preprocess_function(examples):
    tokenized_inputs = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=args.max_length,
        is_split_into_words=True,
        add_special_tokens=False,
    )
    aligned_labels = []
    for i, labels in enumerate(examples["token_labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        aligned_labels.append(align_labels_with_tokens(labels, word_ids))
    return {
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
        "token_labels": aligned_labels,
        "sequence_labels": examples["sequence_labels"],
    }


ds = load_dataset(args)
tokenized_ds = ds.map(preprocess_function, batched=True)
tokenized_ds = tokenized_ds.remove_columns("text")
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
training_args = TrainingArguments(
    output_dir="./results",
    per_device_eval_batch_size=8,
)
trainer = HfMultiTaskTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
)
predictions = trainer.predict(tokenized_ds["test"])
token_predictions = predictions.predictions[0]
sequence_predictions = predictions.predictions[1]
token_labels = predictions.label_ids[0]
sequence_labels = predictions.label_ids[1]

for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
    sequence_threshold = threshold
    token_threshold = threshold

    results = compute_metrics(
        predictions,
        token_threshold,
        sequence_threshold,
    )

    with open(
        os.path.join(args.output_dir, f"test_metrics_{int(threshold*10)}.json"), "w"
    ) as f:
        json.dump(convert_to_serializable(results), f, indent=2)

np.savez(
    os.path.join(args.output_dir, f"test_predictions.npz"),
    token_predictions=token_predictions,
    sequence_predictions=sequence_predictions,
    token_labels=token_labels,
    sequence_labels=sequence_labels,
)
