# SCM

### _From Judgment to Interference_: Early Stopping LLM Harmful Outputs via Streaming Content Monitoring

[Project Page](https://liesy.github.io/SCM/)

[Dataset](https://huggingface.co/datasets/liyang-ict/FineHarm)

<!-- [Chinese Blog]() -->

<img src="SCM.png" width="50%">

## 1\. Prerequisites

Before you begin, ensure you have the following installed:

- Python: 3.12
- CUDA: 11.8
- PyTorch: 2.5.1
- hf-mtask-trainer: 0.0.5

You can often install Python dependencies using:

```bash
pip install -r requirements.txt
```

## 2\. Getting Started

### Step 1: Download the Dataset

First, download the [FineHarm dataset](https://huggingface.co/datasets/liyang-ict/FineHarm) and place it directly into the root directory of this project.

```
SCM/
├── .gitignore
├── README.md
├── requirements.txt
├── SCM.png
├── FineHarm/
├── model/
│   ├── __init__.py
│   ├── modeling_qwen4dual_2CE_w_logic.py
│   └── ModernBertForSeqCLF.py
├── scripts/
│   ├── run-data.sh
│   ├── run-modernbert-full.sh
│   ├── run-modernbert-partial.sh
│   ├── run-qwen-full.sh
│   ├── run-qwen-partial.sh
│   └── run-scm.sh
├── test/
│   ├── test_modernbert_full.py
│   ├── test_modernbert_partial.py
│   ├── test_qwen_full.py
│   └── test_qwen_partial.py
├── train/
│   ├── train_modernbert.py
│   └── train_qwen.py
└── utils/
    ├── __init__.py
    ├── data_process.py
    ├── logic_consistency_loss.py
    └── metrics.py
```

### Step 2: Prepare the Data

Next, preprocess the data by running the `run-data.sh` script. This script will prepare the dataset for the different models.

The script takes two arguments. The first is the destination directory where the processed data will be saved. The second is the path to the tokenizer required for that specific model's data processing.

It is crucial that these two arguments correspond to each other. For example, if you are processing data for the Qwen baseline, the output directory should be for Qwen, and you must provide the path to the Qwen tokenizer.

**Usage:**

```bash
bash scripts/run-data.sh [scm/baseline-qwen/baseline-modernbert] [path-to-tokenizer]
```

**Examples:**

- For **SCM**:

  ```bash
  bash scripts/run-data.sh scm /path/to/your/qwen-tokenizer
  ```

  _(Note: SCM requires a Qwen tokenizer)_

- For **Qwen**:

  ```bash
  bash scripts/run-data.sh baseline-qwen /path/to/your/qwen-tokenizer
  ```

- For **ModernBERT**:

  ```bash
  bash scripts/run-data.sh baseline-modernbert /path/to/your/modernbert-tokenizer
  ```

## 3\. Reproducing Results

This section provides the commands to reproduce the results for each model.

### 3.1. SCM

- **To train the SCM model:**

  ```bash
  bash scripts/run-scm.sh --output-dir [scm-7b/scm-1.5b/scm-0.5b] --data-dir scm --base-model-path /path/to/your/qwen-[7b/1.5b/0.5b]-model
  ```

- **To test the SCM model and evaluate its performance:**

  ```bash
  bash scripts/run-scm.sh test --model-path /path/to/your/scm-model-checkpoints --data-dir scm --output-dir /path/to/save/your/metrics
  ```

### 3.2. Qwen

#### 3.2.1. Full Detection

- **To train the Qwen model for full detection:**

  ```bash
  bash scripts/run-qwen-full.sh --output-dir [baseline-qwen-7b/baseline-qwen-1.5b/baseline-qwen-0.5b] --data-dir baseline-qwen --base-model-path /path/to/your/qwen-[7b/1.5b/0.5b]-model
  ```

- **To test the model:**

  ```bash
  bash scripts/run-qwen-full.sh test --model-path /path/to/your/baseline-qwen-[7b/1.5b/0.5b]-model --data-dir baseline-qwen --output-dir /path/to/save/your/metrics
  ```

#### 3.2.2. Partial Detection

**Note:** The partial detection task uses the model that has already been trained in the Full Detection step. The following scripts will load the **fine-tuned model from the full detection phase** for evaluation.

- **To test the Qwen model for partial detection:**

  ```bash
  bash scripts/run-qwen-partial.sh test --model-path /path/to/your/baseline-qwen-[7b/1.5b/0.5b]-model --data-dir baseline-qwen --output-dir /path/to/save/your/metrics
  ```

### 3.3. ModernBERT

#### 3.3.1. Full Detection

- **To train the ModernBERT model for full detection:**

  ```bash
  bash scripts/run-modernbert-full.sh --output-dir baseline_modernbert-full --data-dir baseline_modernbert --base-model-path /path/to/your/modernbert-model
  ```

- **To test the model:**

  ```bash
  bash scripts/run-modernbert-full.sh test --model-path /path/to/your/baseline_modernbert-full-model --data-dir baseline_modernbert --output-dir /path/to/save/your/metrics
  ```

#### 3.3.2. Partial Detection

**Note:** The partial detection task uses the model that has already been trained in the Full Detection step. The following scripts will load the **fine-tuned model from the full detection phase** for evaluation.

- **To test the ModernBERT model for partial detection:**

  ```bash
  bash scripts/run-modernbert-partial.sh test --model-path /path/to/your/baseline_modernbert-full-model --data-dir baseline_modernbert --output-dir /path/to/save/your/metrics
  ```

## 4\. Citing

If you use this code in your research, please consider citing:

```
@article{li2025judgment,
  title={From Judgment to Interference: Early Stopping LLM Harmful Outputs via Streaming Content Monitoring},
  author={Li, Yang and Sheng, Qiang and Yang, Yehan and Zhang, Xueyao and Cao, Juan},
  journal={arXiv preprint arXiv:2506.09996},
  year={2025}
}
```
