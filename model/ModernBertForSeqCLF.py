import torch
import torch.nn as nn
from transformers import ModernBertModel, ModernBertPreTrainedModel


class ModernBERTForSequenceClassification(ModernBertPreTrainedModel):
    supports_report_metrics: bool = True

    def __init__(self, config):
        super().__init__(config)
        self.model = ModernBertModel(config)
        self.sequence_classifier = nn.Linear(
            config.hidden_size, config.num_sequence_labels, bias=False
        )
        self.sequence_loss_fn = nn.CrossEntropyLoss()
        self.post_init()

    def post_init(self):
        nn.init.xavier_uniform_(self.sequence_classifier.weight)
        if self.sequence_classifier.bias is not None:
            nn.init.zeros_(self.sequence_classifier.bias)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        sliding_window_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        sequence_labels: torch.LongTensor | None = None,
        indices: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        batch_size: int | None = None,
        seq_len: int | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_size=batch_size,
            seq_len=seq_len,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        sequence_logits = self.sequence_classifier(hidden_states[:, 0, :])
        total_loss = None
        if sequence_labels is not None:
            total_loss = self.sequence_loss_fn(sequence_logits, sequence_labels)

        if not return_dict:
            output = (sequence_logits,)
            return ((total_loss,) + output) if total_loss is not None else output
        return {
            "loss": total_loss,
            "logits": sequence_logits,
        }
