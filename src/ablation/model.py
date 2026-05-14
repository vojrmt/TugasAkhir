import torch
import torch.nn as nn
from transformers import AutoModel

BERT_MODEL = "roberta-base"

class LabelAttention(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int = 5,
                 attn_dim: int = 256, use_projection: bool = True):
        super().__init__()
        self.num_labels = num_labels
        self.use_projection = use_projection
        self.label_embeddings = nn.Embedding(num_labels, hidden_size)
        self.W_h = nn.Linear(hidden_size, attn_dim, bias=False)
        self.W_l = nn.Linear(hidden_size, attn_dim, bias=False)

    def forward(self, comment_embs, comment_mask):
        label_idx = torch.arange(self.num_labels, device=comment_embs.device)
        label_vecs = self.label_embeddings(label_idx)

        if self.use_projection:
            H_prime = self.W_h(comment_embs)
            L_prime = self.W_l(label_vecs)
        else:
            H_prime = comment_embs
            L_prime = label_vecs

        scores = torch.matmul(H_prime, L_prime.T)
        scores = scores.permute(0, 2, 1)
        mask = comment_mask.unsqueeze(1)
        scores = scores.masked_fill(~mask, float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, comment_embs)
        return context, attn_weights

class HTLA(nn.Module):
    def __init__(self, bert_model_name=BERT_MODEL, num_labels=5, dropout=0.1,
                 doc_attn_heads=8, doc_attn_ff_dim=2048, label_attn_dim=256,
                 use_doc_encoder=True, use_projection=True, use_label_attn=True):
        super().__init__()
        self.use_doc_encoder = use_doc_encoder
        self.use_label_attn  = use_label_attn
        self.num_labels      = num_labels

        self.bert    = AutoModel.from_pretrained(bert_model_name)
        hidden_size  = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=doc_attn_heads,
            dim_feedforward=doc_attn_ff_dim, dropout=dropout,
            batch_first=True,
        )
        self.doc_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.label_attn  = LabelAttention(hidden_size, num_labels,
                                          label_attn_dim, use_projection)
        self.classifiers = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(num_labels)
        ])

    def forward(self, input_ids, attention_masks, comment_mask):
        batch_size, num_comments, seq_len = input_ids.shape
        ids_flat   = input_ids.view(batch_size * num_comments, seq_len)
        masks_flat = attention_masks.view(batch_size * num_comments, seq_len)

        bert_out     = self.bert(input_ids=ids_flat, attention_mask=masks_flat)
        cls_embs     = self.dropout(bert_out.last_hidden_state[:, 0, :])
        comment_embs = cls_embs.view(batch_size, num_comments, -1)

        if self.use_doc_encoder:
            padding_mask = ~comment_mask
            comment_embs = self.doc_encoder(
                comment_embs, src_key_padding_mask=padding_mask
            )
            comment_embs = self.dropout(comment_embs)

        if self.use_label_attn:
            context, attn_weights = self.label_attn(comment_embs, comment_mask)
        else:
            mask_expanded = comment_mask.unsqueeze(-1).float()
            pooled = (comment_embs * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            context = pooled.unsqueeze(1).expand(-1, self.num_labels, -1)
            attn_weights = torch.zeros(
                comment_embs.size(0), self.num_labels, comment_embs.size(1),
                device=comment_embs.device
            )

        context = self.dropout(context)
        logits  = torch.cat([
            self.classifiers[i](context[:, i, :])
            for i in range(self.num_labels)
        ], dim=1)

        return logits, attn_weights