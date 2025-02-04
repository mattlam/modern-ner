import torch
from transformers import AutoConfig, RobertaTokenizerFast
from safetensors.torch import load_file  # Import the safetensors loading function

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = AutoConfig.from_pretrained(SAVE_DIR_DRIVE)

model = ModernBertForTokenClassificationCRF(MODEL_NAME, num_labels=len(label_list), config=config)

state_dict_path = f"{SAVE_DIR_DRIVE}/model.safetensors"

state_dict = load_file(state_dict_path, device=str(DEVICE))
model.load_state_dict(state_dict)

model.to(DEVICE)
model.eval()

tokenizer = RobertaTokenizerFast.from_pretrained(SAVE_DIR_DRIVE)

MAX_LENGTH = 256  # Should match the value used during training, I think

def predict_sentence(sentence: str):
    """
    Tokenizes the input sentence, performs prediction with the model,
    and returns a list of (token, predicted_label) tuples.
    """
    tokens = sentence.split()

    tokenized_inputs = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    word_ids = tokenized_inputs.word_ids(batch_index=0)

    tokenized_inputs = {k: v.to(DEVICE) for k, v in tokenized_inputs.items()}

    with torch.no_grad():
        outputs = model(**tokenized_inputs)

    predictions = outputs["predictions"].squeeze().tolist()

    tokenized_tokens = tokenizer.convert_ids_to_tokens(tokenized_inputs["input_ids"][0])

    predicted_labels = []
    final_tokens = []
    for token, word_idx, pred in zip(tokenized_tokens, word_ids, predictions):
        if word_idx is None:
            continue
        final_tokens.append(token)
        predicted_labels.append(id2label.get(pred, "O"))

    return list(zip(final_tokens, predicted_labels))

sample_sentence = """The night was late but sally smithers really wanted to get
onion rings at Burger King in Raleigh"""
prediction = predict_sentence(sample_sentence)

print("Predictions:")
for token, label in prediction:
    print(f"{token:15} --> {label}")