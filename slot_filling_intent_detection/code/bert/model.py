from transformers.models.bert.modeling_bert  import BertPreTrainedModel, BertModel, BertConfig
from torch import nn

class NewBert(BertPreTrainedModel):
    def __init__(self, config, unique_intent_size, unique_slot_size, dropout_rate):
        super(NewBert, self).__init__(config)
        self.unique_intent_size = unique_intent_size
        self.unique_slot_size = unique_slot_size
        
        self.bert = BertModel(config)  # Load pretrained bert
        self.slot_dropout = nn.Dropout(dropout_rate)
        self.slot_linear = nn.Linear(config.hidden_size, unique_slot_size)

        self.intent_dropout = nn.Dropout(dropout_rate)
        self.intent_linear = nn.Linear(config.hidden_size, unique_intent_size)

        self.intent_loss_func = nn.CrossEntropyLoss()
        self.slot_loss_func = nn.CrossEntropyLoss(ignore_index=0)

    def get_logit(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask=attention_mask,
              token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0][:,1:-1,:]
        pooled_output = outputs[1]  # [CLS]
        slot_logits = self.slot_linear(self.slot_dropout(sequence_output))
        intent_logits = self.intent_linear(self.intent_dropout(pooled_output))
        return intent_logits, slot_logits

    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids, slot_labels_ids):
        intent_logits, slot_logits = self.get_logit(input_ids, attention_mask, token_type_ids)
        intent_loss = self.intent_loss_func(intent_logits.view(-1, self.unique_intent_size), intent_label_ids.view(-1))
        slot_loss = self.slot_loss_func(slot_logits.view(-1, self.unique_slot_size), slot_labels_ids.view(-1))
        total_loss = slot_loss + intent_loss
        return (intent_logits, slot_logits), total_loss