import torch
from transformers import AutoConfig
from safetensors.torch import load_file
from modeling_modernbert_crf import ModernBertForTokenClassificationCRF

CHECKPOINT_DIR = "./"
FINAL_DIR = "./modernbert-ner-crf-hf"

config = AutoConfig.from_pretrained(CHECKPOINT_DIR)
config.architectures = ["ModernBertForTokenClassificationCRF"]

model = ModernBertForTokenClassificationCRF(config)

state_dict = load_file(f"{CHECKPOINT_DIR}/model.safetensors")

model.load_state_dict(state_dict)

model.save_pretrained(FINAL_DIR)
config.save_pretrained(FINAL_DIR)

print(f"Converted and saved to {FINAL_DIR}")