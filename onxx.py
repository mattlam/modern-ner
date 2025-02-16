import torch
from transformers import AutoTokenizer
from modeling_modernbert_crf import ModernBertForTokenClassificationCRF


class DisableCompileContextManager:
    def __init__(self):
        self._original_compile = torch.compile

    def __enter__(self):
        torch.compile = lambda *args, **kwargs: lambda x: x

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.compile = self._original_compile


def export():
    with DisableCompileContextManager():
        tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base", model_max_length=4096)

        model = ModernBertForTokenClassificationCRF.from_pretrained(
            "./modernbert-ner-crf-hf", attn_implementation="eager"
        )
        model.eval()

        samples = ["Hello, this is a test sentence."]
        tokenized = tokenizer(samples, return_tensors='pt', max_length=128, padding='max_length', truncation=True)

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        model = model.to('cpu')

        with torch.no_grad():
            torch.onnx.export(
                model,
                (input_ids, attention_mask),
                "modernbert_crf_emissions.onnx",
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                opset_version=14,
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "seq_length"},
                    "attention_mask": {0: "batch_size", 1: "seq_length"},
                    "logits": {0: "batch_size", 1: "seq_length"},
                },
            )

        print("✅ Exported model to modernbert_crf_emissions.onnx")


if __name__ == '__main__':
    export()
