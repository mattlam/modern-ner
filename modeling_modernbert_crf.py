import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import PreTrainedModel, AutoModel, AutoConfig


class ModernBertPreTrainedModel(PreTrainedModel):
    config_class = AutoConfig
    base_model_prefix = "modernbert"


class ModernBertForTokenClassificationCRF(ModernBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = AutoModel.from_config(config)

        dropout_rate = getattr(config, "hidden_dropout_prob", 0.1)
        self.dropout = nn.Dropout(dropout_rate)

        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.crf = CRF(config.num_labels, batch_first=True)

        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        sequence_output = self.dropout(outputs.last_hidden_state)
        emissions = self.classifier(sequence_output)

        mask = attention_mask.bool() if attention_mask is not None else None

        if labels is not None:
            labels = labels.clone()
            labels[labels == -100] = 0

            loss = -self.crf(emissions, labels, mask=mask, reduction="mean")
            return {"loss": loss, "logits": emissions}

        # For ONNX compatibility: Don't decode, only return emissions (logits)
        return {"logits": emissions}