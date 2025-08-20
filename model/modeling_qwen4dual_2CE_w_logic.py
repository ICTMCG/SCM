import torch
import torch.nn as nn
from transformers import Qwen2PreTrainedModel, Qwen2Model
from utils.logic_consistency_loss import LogicConsistencyLoss


class QwenForDualTask(Qwen2PreTrainedModel):
    supports_report_metrics: bool = True

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.dropout = nn.Dropout(0.1)
        self.token_classifier = nn.Linear(config.hidden_size, config.num_token_labels)
        self.sequence_classifier = nn.Linear(
            config.hidden_size, config.num_sequence_labels, bias=False
        )
        self.token_loss_fn = nn.CrossEntropyLoss()
        self.sequence_loss_fn = nn.CrossEntropyLoss()
        self.logic_loss_fn = LogicConsistencyLoss(
            n_classes=config.num_token_labels,
            reduce=config.logic_reduce,
            reduction="mean",
        )
        self.post_init()

    def post_init(self):
        nn.init.xavier_uniform_(self.token_classifier.weight)
        if self.token_classifier.bias is not None:
            nn.init.zeros_(self.token_classifier.bias)
        nn.init.xavier_uniform_(self.sequence_classifier.weight)
        if self.sequence_classifier.bias is not None:
            nn.init.zeros_(self.sequence_classifier.bias)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        token_labels: torch.LongTensor | None = None,
        sequence_labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError(
                "Cannot handle batch sizes > 1 if no padding token is defined."
            )
        if self.config.pad_token_id is None:
            last_non_pad_token = -1
        elif input_ids is not None:
            non_pad_mask = (input_ids != self.config.pad_token_id).to(
                hidden_states.device, torch.int32
            )
            token_indices = torch.arange(
                input_ids.shape[-1], device=hidden_states.device
            )
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            last_non_pad_token = -1
        sequence_logits = self.sequence_classifier(
            hidden_states[
                torch.arange(batch_size, device=hidden_states.device),
                last_non_pad_token,
            ]
        )
        sequence_loss = None
        if sequence_labels is not None:
            sequence_loss = self.sequence_loss_fn(sequence_logits, sequence_labels)
        hidden_states = self.dropout(hidden_states)
        token_logits = self.token_classifier(hidden_states)
        token_loss = None
        if token_labels is not None:
            token_loss = self.token_loss_fn(
                token_logits.view(-1, self.config.num_token_labels),
                token_labels.view(-1),
            )
        logic_loss = None
        if token_loss is not None and sequence_loss is not None:
            token_mask = (token_labels != self.config.ignore_index).to(
                token_logits.device, torch.int32
            )
            logic_loss = self.logic_loss_fn(sequence_logits, token_logits, token_mask)
        total_loss = None
        if (
            token_loss is not None
            and sequence_loss is not None
            and logic_loss is not None
        ):
            total_loss = (
                self.config.alpha * token_loss
                + self.config.beta * sequence_loss
                + self.config.gamma * logic_loss
            )

        if hasattr(self, "report_metrics"):
            self.report_metrics(
                token_loss=token_loss,
                sequence_loss=sequence_loss,
                logic_loss=logic_loss,
            )

        if not return_dict:
            output = (token_logits, sequence_logits)
            return ((total_loss,) + output) if total_loss is not None else output
        return {
            "loss": total_loss,
            "token_logits": token_logits,
            "sequence_logits": sequence_logits,
        }
