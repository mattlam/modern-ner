#ModernBert didn't work with onxx OOTB
#solution based on wakaka6's contribution here https://github.com/huggingface/transformers/issues/35545#issuecomment-2589533973

import os
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from onnxruntime.quantization import quantize_dynamic, QuantType


class DisableCompileContextManager:
    """
    This context manager temporarily disables torch.compile by replacing it
    with a no-op. This avoids triggering Flash Attention 2.0’s new (and ONNX-incompatible)
    memory access patterns.
    """
    def __init__(self):
        self._original_compile = torch.compile

    def __enter__(self):
        torch.compile = lambda *args, **kwargs: (lambda x: x)

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.compile = self._original_compile


def export_to_onnx(model_dir: str, output_path: str, device: str = "cpu"):
    """
    Loads the model and tokenizer from `model_dir`, sets the attention implementation
    to standard (i.e. "eager"), and exports the model to ONNX.
    """
    with DisableCompileContextManager():
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        print("Loading model with standard (eager) attention...")
        model = AutoModelForTokenClassification.from_pretrained(
            model_dir,
            # Disable the new Flash Attention 2.0 by using the standard (eager) implementation
            attn_implementation="eager"
        )
        model.eval()

        if device == "cuda" and torch.cuda.is_available():
            model = model.to("cuda")
            print("Model moved to CUDA.")
        else:
            print("Using CPU.")

        # Prepare a dummy input. Adjust max_length as needed.
        sample_text = "Hello, this is a sample text for NER."
        tokenized = tokenizer(
            sample_text,
            return_tensors="pt",
            padding="max_length",
            max_length=8192,
            truncation=True
        )
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)

        dummy_inputs = (input_ids, attention_mask)

        print("Exporting model to ONNX...")
        torch.onnx.export(
            model,
            dummy_inputs,
            output_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "seq_length"},
                "attention_mask": {0: "batch_size", 1: "seq_length"},
                "logits": {0: "batch_size", 1: "seq_length"},
            },
            opset_version=14,
            do_constant_folding=True,
        )
        print(f"ONNX export completed. Saved to {output_path}")


def quantize_model(onnx_model_path: str, quantized_model_path: str):
    print("Quantizing the ONNX model...")
    quantize_dynamic(
        onnx_model_path,
        quantized_model_path,
        weight_type=QuantType.QInt8  # or QuantType.QUInt8 if preferred
    )
    print(f"Quantized model saved to {quantized_model_path}")


def main():
    model_dir = "./"  
    onnx_output = "model.onnx"
    quantized_output = "model.quant.onnx"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    export_to_onnx(model_dir, onnx_output, device=device)
    quantize_model(onnx_output, quantized_output)


if __name__ == "__main__":
    main()