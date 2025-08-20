import datetime, sys, os, datasets, torch, random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from argparse import ArgumentParser
from transformers import (
    Qwen2Config,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)
from utils.metrics import compute_metrics_for_dual as compute_metrics
from model.modeling_qwen4dual_2CE_w_logic import (
    QwenForDualTask as QwenForDualTaskWith2CEWLogic,
)
from hf_mtask_trainer import HfMultiTaskTrainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.cuda.empty_cache()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--base-model-path",
        type=str,
        default="/data1/shared/LLMs/Qwen2.5-1.5B",
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="exp")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-ratio", type=float, default=0)
    parser.add_argument("--logic-reduce", type=str, default="max")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--deepspeed", type=str, default=None)
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
    config = Qwen2Config.from_pretrained(args.base_model_path)
    config.num_sequence_labels = 2
    config.num_token_labels = 2
    config.alpha = args.alpha
    config.beta = 1 - args.alpha
    config.gamma = args.gamma
    config.logic_reduce = args.logic_reduce
    return config


def load_dataset(args):
    data_files = {
        "train": os.path.join(args.data_dir, "train.jsonl"),
        "validation": os.path.join(args.data_dir, "val.jsonl"),
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


def main(args):
    set_seed(args.seed)
    config = parse_config(args)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    model = QwenForDualTaskWith2CEWLogic.from_pretrained(
        args.base_model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if not tokenizer.pad_token:
        print("Set pad token to eos token")
        tokenizer.pad_token = tokenizer.eos_token
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

    tokenized_ds = ds.map(preprocess_function, batched=True)
    tokenized_ds = tokenized_ds.remove_columns("text")
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
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
        if args.batch_size < 4:
            exp_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=optm,
                gamma=0.999,
            )
        else:
            exp_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=optm,
                gamma=0.99,
            )
        optimizer = (optm, exp_scheduler)
    elif args.scheduler == "step":
        optm = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        step_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optm, step_size=100, gamma=0.5
        )
        optimizer = (optm, step_scheduler)
    elif args.scheduler == "cos":
        scheduler = "cosine"
    elif args.scheduler == "linear":
        pass
    else:
        raise NotImplementedError(f"Scheduler {args.scheduler} not implemented")

    if args.batch_size < 2:
        gradient_accumulation_steps = 1
    else:
        gradient_accumulation_steps = 2
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 4,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=scheduler,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=1,
        load_best_model_at_end=True,
        push_to_hub=False,
        label_names=["token_labels", "sequence_labels"],
        remove_unused_columns=False,
        include_for_metrics=["token_labels", "sequence_labels"],
        logging_dir=args.log_dir,
        logging_steps=20,
        logging_first_step=True,
        report_to="tensorboard",
        deepspeed=args.deepspeed,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    trainer = HfMultiTaskTrainer(
        model=model,
        args=training_args,
        optimizers=optimizer,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    try:
        trainer.train(resume_from_checkpoint=True)
    except ValueError as e:
        print(e)
        trainer.train()


if __name__ == "__main__":
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    args = parse_args()
    args.log_dir = os.path.join(args.output_dir, "logs")
    print(args)
    os.makedirs(args.log_dir, exist_ok=True)
    sys.exit(main(args))
