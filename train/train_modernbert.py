import datetime, sys, os, datasets, torch, random
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from argparse import ArgumentParser
from transformers import (
    ModernBertConfig,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from model.ModernBertForSeqCLF import ModernBERTForSequenceClassification


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--scheduler", type=str, default="exp")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-ratio", type=float, default=0)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def parse_config(args):
    config = ModernBertConfig.from_pretrained(args.base_model_path)
    config.reference_compile = False
    config.num_sequence_labels = 2
    return config


def load_dataset(args):
    data_files = {
        "train": os.path.join(args.data_dir, "train.jsonl"),
        "validation": os.path.join(args.data_dir, "val.jsonl"),
        "test": os.path.join(args.data_dir, "test.jsonl"),
    }
    ds = datasets.load_dataset("json", data_files=data_files)
    return ds


def compute_metrics(p):

    def get_result_dict(labels, scores, threshold, prefix=""):
        results = dict()
        results["threshold"] = threshold
        predictions = (scores >= threshold).astype(np.int16)
        results["accuracy"] = accuracy_score(labels, predictions)
        results["precision"] = precision_score(labels, predictions, average="binary")
        results["recall"] = recall_score(labels, predictions, average="binary")
        results["f1"] = f1_score(labels, predictions, average="binary")
        results["auc"] = roc_auc_score(labels, predictions)
        results["num_label_1"] = np.sum(labels)
        results["num_label_0"] = labels.shape[0] - results["num_label_1"]
        results["num_pred_1"] = np.sum(predictions)
        results["num_pred_0"] = predictions.shape[0] - results["num_pred_1"]

        if prefix:
            results = {f"{prefix}_{k}": v for k, v in results.items()}
        return results

    sequence_scores = p.predictions
    sequence_labels = p.label_ids
    sequence_scores = (
        torch.nn.functional.softmax(torch.from_numpy(sequence_scores), dim=-1)[:, -1]
        .detach()
        .numpy()
    )
    results_sequence = get_result_dict(
        sequence_labels, sequence_scores, 0.5, prefix="sequence"
    )
    results = results_sequence
    return results


def main(args):
    set_seed(args.seed)
    config = parse_config(args)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    model = ModernBERTForSequenceClassification.from_pretrained(
        args.base_model_path, config=config, attn_implementation="sdpa"
    )

    if not tokenizer.pad_token:
        print("Set pad token to sep token")
        tokenizer.pad_token = tokenizer.sep_token
    model.config.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    ds = load_dataset(args)

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

    tokenized_ds = ds.map(preprocess_function, batched=True)
    tokenized_ds = tokenized_ds.remove_columns(["text", "token_labels"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    scheduler = "linear"
    optimizer = (None, None)
    lr_scheduler_kwargs = {}
    if args.scheduler == "poly":
        scheduler == "polynomial"
        lr_scheduler_kwargs = {
            "lr_end": 1e-6,
            "power": 2,
        }
    elif args.scheduler == "exp":
        optm = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        exp_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optm, gamma=0.995
        )
        optimizer = (optm, exp_scheduler)
    elif args.scheduler == "cos":
        scheduler = "cosine"
    elif args.scheduler == "linear":
        pass
    else:
        raise NotImplementedError(f"Scheduler {args.scheduler} not implemented")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=scheduler,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100,
        load_best_model_at_end=True,
        save_total_limit=1,
        push_to_hub=False,
        label_names=["sequence_labels"],
        remove_unused_columns=False,
        include_for_metrics=["sequence_labels"],
        logging_dir=args.log_dir,
        logging_steps=20,
        logging_first_step=True,
        report_to="tensorboard",
        gradient_accumulation_steps=4,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        optimizers=optimizer,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()


if __name__ == "__main__":
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    args = parse_args()
    args.log_dir = os.path.join(args.output_dir, "logs")
    print(args)
    os.makedirs(args.log_dir, exist_ok=True)
    sys.exit(main(args))
