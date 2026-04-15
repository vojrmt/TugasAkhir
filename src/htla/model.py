import torch
import torch.nn as nn
from transformers import AutoModel

BERT_MODEL = "roberta-base"


class LabelAttention(nn.Module):
    def __init__(self, hidden_size, num_labels=5):
        super().__init__()
        self.label_embeddings = nn.Embedding(num_labels, hidden_size)
        self.num_labels       = num_labels

    def forward(self, comment_embs, comment_mask):
        label_idx    = torch.arange(self.num_labels, device=comment_embs.device)
        label_vecs   = self.label_embeddings(label_idx)
        scores       = torch.matmul(label_vecs, comment_embs.transpose(1, 2))
        mask         = comment_mask.unsqueeze(1)
        scores       = scores.masked_fill(~mask, float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)
        context      = torch.matmul(attn_weights, comment_embs)
        return context, attn_weights


class HTLA(nn.Module):
    def __init__(self, bert_model_name=BERT_MODEL, num_labels=5, dropout=0.1):
        super().__init__()
        self.bert        = AutoModel.from_pretrained(bert_model_name)
        hidden_size      = self.bert.config.hidden_size
        self.dropout     = nn.Dropout(dropout)
        self.label_attn  = LabelAttention(hidden_size, num_labels)
        self.classifiers = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(num_labels)
        ])

    def forward(self, input_ids, attention_masks, comment_mask):
        batch_size, num_comments, seq_len = input_ids.shape
        input_ids_flat       = input_ids.view(batch_size * num_comments, seq_len)
        attention_masks_flat = attention_masks.view(batch_size * num_comments, seq_len)

        bert_out       = self.bert(
            input_ids=input_ids_flat,
            attention_mask=attention_masks_flat
        )
        cls_embeddings = self.dropout(bert_out.last_hidden_state[:, 0, :])
        comment_embs   = cls_embeddings.view(batch_size, num_comments, -1)

        context, attn_weights = self.label_attn(comment_embs, comment_mask)
        context               = self.dropout(context)

        logits = torch.cat([
            self.classifiers[i](context[:, i, :])
            for i in range(len(self.classifiers))
        ], dim=1)

        return logits, attn_weights