#sorry for having everything in one file, this is 'optimized' for Colab use. pls do not judge

!pip install flash-attn --no-build-isolation
!pip install torch==2.5.1+cu121 --extra-index-url https://download.pytorch.org/whl/nightly/cu121
!pip install datasets seqeval tqdm accelerate conllu pytorch-crf names faker
# This installs a specific commit of Transformers that supports ModernBERT and Flash Attention:
# maybe not necessary now after fixes?? (early 2025)
%pip install "git+https://github.com/huggingface/transformers.git@6e0515e99c39444caae39472ee1b2fd76ece32f1" --upgrade

import os
import torch
import random
import numpy as np
from typing import List, Tuple, Dict, Any
from seqeval.metrics import classification_report

from modeling_modernbert_crf import ModernBertForTokenClassificationCRF

from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    AutoConfig,
    AutoModel,
)
from transformers import RobertaTokenizerFast

from torch import nn
from datasets import Dataset, DatasetDict
from faker import Faker
import names
import re
from torchcrf import CRF

# Mount Google Drive to persist checkpoints and models
try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    print("Google Drive not available. Running locally.")

SEED = 42

MODEL_NAME = "answerdotai/ModernBERT-base"

#hopefully this matches the official ontonotes
DATASET_FILES = {
    "train": "https://huggingface.co/datasets/djagatiya/ner-ontonotes-v5-eng-v4/raw/main/train.conll",
    "validation": "https://huggingface.co/datasets/djagatiya/ner-ontonotes-v5-eng-v4/raw/main/validation.conll",
    "test": "https://huggingface.co/datasets/djagatiya/ner-ontonotes-v5-eng-v4/raw/main/test.conll"
}
MAX_LENGTH = 256
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Augmentation was making the model worse, so not using
USE_AUGMENTATION = False
AUGMENTATION_PROB = 0.1  # probability per entity
PERSON_WEIGHT_FACTOR = 1.0

SAVE_DIR_DRIVE = "/content/drive/MyDrive/models/modernbert-ner-model-crf-base"
os.makedirs(SAVE_DIR_DRIVE, exist_ok=True)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

def download_and_parse_conll(url: str, file_path: str) -> List[dict]:
    """
    Download and parse a .conll file.
    """
    os.system(f"wget -q {url} -O {file_path}")
    sentences = []
    current_tokens = []
    current_labels = []

    with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(("#begin", "#end")):
                if current_tokens:
                    sentences.append({"tokens": current_tokens, "ner_tags": current_labels})
                    current_tokens, current_labels = [], []
                continue

            parts = line.split('\t') if '\t' in line else line.split()
            if len(parts) < 2:
                continue

            token = parts[0].strip()
            ner_tag = parts[-1].strip()
            current_tokens.append(token)
            current_labels.append(ner_tag)

    if current_tokens:
        sentences.append({"tokens": current_tokens, "ner_tags": current_labels})
    return sentences

print("Downloading and processing dataset...")
dataset_dict = {}
for split, url in DATASET_FILES.items():
    file_path = f"{split}.conll"
    print(f"📥 Processing {split} split...")
    parsed_data = download_and_parse_conll(url, file_path)
    dataset_dict[split] = Dataset.from_list(parsed_data)
    print(f"✅ {split}: {len(parsed_data)} examples")

dataset = DatasetDict(dataset_dict)

label_list = sorted({tag for example in dataset["train"] for tag in example["ner_tags"]})
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}

print("🔢 Converting labels to numeric IDs...")

def convert_labels(example):
    return {"ner_tags": [label2id[tag] for tag in example["ner_tags"]]}

dataset = dataset.map(convert_labels, batched=False, desc="Label conversion")

fake = Faker()
fake.seed_instance(SEED)

def get_entity_spans(ner_tags: List[str], target_label: str) -> List[Tuple[int, int]]:
    spans = []
    start = None
    for i, tag in enumerate(ner_tags):
        if tag == f"B-{target_label}":
            if start is not None:
                spans.append((start, i))
            start = i
        elif tag == f"I-{target_label}" and start is not None:
            continue
        else:
            if start is not None:
                spans.append((start, i))
                start = None
    if start is not None:
        spans.append((start, len(ner_tags)))
    return spans

def augment_person_entities(tokens: List[str], ner_tags: List[int]) -> Tuple[List[str], List[int]]:
    # Convert numeric tags back to label strings
    ner_label_strs = [id2label[tag] if tag != -100 else 'O' for tag in ner_tags]
    person_spans = get_entity_spans(ner_label_strs, "PERSON")
    if not person_spans:
        return tokens, ner_tags
    new_tokens = tokens.copy()
    new_labels = ner_tags.copy()
    for span in reversed(person_spans):
        if random.random() < AUGMENTATION_PROB:
            start, end = span
            new_name = names.get_full_name() if random.random() < 0.5 else fake.name()
            new_name_tokens = new_name.split()
            num_new_tokens = len(new_name_tokens)
            new_tokens = new_tokens[:start] + new_name_tokens + new_tokens[end:]
            new_entity_labels = [label2id["B-PERSON"]] + [label2id["I-PERSON"]] * (num_new_tokens - 1)
            new_labels = new_labels[:start] + new_entity_labels + new_labels[end:]
    return new_tokens, new_labels

if USE_AUGMENTATION:
    print("🔀 Applying entity-level data augmentation to train set (PERSON)...")
    def augmentation_map_fn(example):
        augmented_tokens, augmented_labels = augment_person_entities(example["tokens"], example["ner_tags"])
        return {"tokens": augmented_tokens, "ner_tags": augmented_labels}
    dataset["train"] = dataset["train"].map(
        augmentation_map_fn,
        batched=False,
        desc="Applying entity-level name augmentation to PERSON"
    )
    print("✅ Data augmentation applied.")

print("🔧 Initializing ModernBERT tokenizer...")
tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME, add_prefix_space=True)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        is_split_into_words=True,
    )

    labels = []
    for i, label_sequence in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(label2id.get("O", 0))
            else:
                label_ids.append(label_sequence[word_idx])
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

print("🔡 Tokenizing datasets...")
tokenized_datasets = dataset.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=dataset["train"].column_names,
    desc="Tokenizing"
)

use_flash_attention = True if DEVICE.type == "cuda" else False
try:
    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        use_flash_attention=use_flash_attention
    )
except Exception as e:
    print("Flash attention not supported, disabling it. Error:", e)
    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        use_flash_attention=False
    )

model = ModernBertForTokenClassificationCRF(MODEL_NAME, num_labels=len(label_list), config=config).to(DEVICE)

training_args = TrainingArguments(
    output_dir=SAVE_DIR_DRIVE,  
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-05,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    num_train_epochs=20,
    weight_decay=0.01,
    gradient_accumulation_steps=1,
    bf16=True if DEVICE.type == "cuda" else False,
    tf32=True if DEVICE.type == "cuda" else False,
    logging_steps=100,
    load_best_model_at_end=True,
    #metric_for_best_model="eval_PERSON_f1",
    metric_for_best_model="eval_f1",
    greater_is_better=True,
    report_to="none",
    optim="adamw_torch_fused",
    seed=SEED,
    save_total_limit=3  # Keep only the 3 most recent checkpoints
)

def safe_int(x) -> int:
    if hasattr(x, "item"):
        if isinstance(x, np.ndarray):
            if x.size == 0:
                return 0
            if x.size != 1:
                return int(x.flatten()[0])
            else:
                return int(x.item())
        else:
            try:
                return int(x.item())
            except Exception:
                try:
                    return int(x[0])
                except Exception:
                    return 0
    return int(x)

def compute_metrics(predictions_and_labels):
    preds, labels = predictions_and_labels
    true_preds = []
    true_labels = []

    for pred_seq, label_seq in zip(preds, labels):
        filtered_preds = []
        filtered_labels = []
        for p, l in zip(pred_seq, label_seq):
            if l != -100:
                p_val = safe_int(p)
                l_val = safe_int(l)
                filtered_preds.append(id2label.get(p_val, "O"))
                filtered_labels.append(id2label.get(l_val, "O"))
        true_preds.append(filtered_preds)
        true_labels.append(filtered_labels)

    report = classification_report(true_labels, true_preds, output_dict=True, zero_division=0)
    metrics_dict = {
        "precision": report["micro avg"]["precision"],
        "recall": report["micro avg"]["recall"],
        "f1": report["micro avg"]["f1-score"],
    }
    for entity_type, scores in report.items():
        if entity_type.endswith("avg") or entity_type == 'O':
            continue
        metrics_dict[f"{entity_type}_precision"] = scores["precision"]
        metrics_dict[f"{entity_type}_recall"] = scores["recall"]
        metrics_dict[f"{entity_type}_f1"] = scores["f1-score"]
    return metrics_dict

class CustomTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs.get("loss")
        if loss is not None:
            loss = loss.detach()

        predictions = outputs.get("predictions")
        labels = inputs.get("labels")

        return (loss, predictions, labels)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
)

latest_checkpoint = None
if os.path.isdir(SAVE_DIR_DRIVE):
    checkpoints = [os.path.join(SAVE_DIR_DRIVE, d) for d in os.listdir(SAVE_DIR_DRIVE) if d.startswith("checkpoint")]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getmtime)
        print(f"Resuming from checkpoint: {latest_checkpoint}")

print("🚀 Starting training with CRF layer...")
trainer.train(resume_from_checkpoint=latest_checkpoint)

print("💾 Saving model and tokenizer...")
trainer.save_model(SAVE_DIR_DRIVE)
tokenizer.save_pretrained(SAVE_DIR_DRIVE)
config.save_pretrained(SAVE_DIR_DRIVE)

print("✅ Training complete!")
print("📊 Final test metrics:", trainer.evaluate(tokenized_datasets["test"]))
