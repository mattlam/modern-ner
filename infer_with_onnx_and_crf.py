import numpy as np
import onnxruntime as ort
import torch
from torchcrf import CRF
from transformers import AutoTokenizer
import json

MODEL_PATH = "modernbert_crf_emissions.onnx"
NUM_LABELS = 37

tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

with open("./modernbert-ner-crf-hf/config.json") as f:
    config = json.load(f)
    id2label = {int(k): v for k, v in config["id2label"].items()}

text = "John Doe went to New York in 2023."
inputs = tokenizer(text, return_tensors="np", padding="max_length", max_length=128, truncation=True)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

ort_session = ort.InferenceSession(MODEL_PATH)

outputs = ort_session.run(
    None,
    {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    },
)

logits = outputs[0]

logits_tensor = torch.tensor(logits)
attention_mask_tensor = torch.tensor(attention_mask)

crf = CRF(NUM_LABELS, batch_first=True)
predictions = crf.decode(logits_tensor, mask=attention_mask_tensor.bool())

decoded_labels = [
    [id2label[token_id] for token_id in seq]
    for seq in predictions
]

print("CRF Predictions (IDs):", predictions)
print("NER Predictions (Labels):", decoded_labels)

tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
for token, label in zip(tokens, decoded_labels[0]):
    print(f"{token:<10} -> {label}")