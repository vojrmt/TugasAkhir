import torch
import torch.nn as nn
from transformers import AutoModel

BERT_MODEL = "roberta-base"


class LabelAttention(nn.Module):

    def __init__(self, hidden_size: int, num_labels: int = 5,
                 attn_dim: int = 256, use_projection: bool = True, use_label_attn=True):
        super().__init__()
        self.num_labels      = num_labels
        self.use_projection = use_projection
        self.use_label_attn = use_label_attn
        # Learnable label representations  (L × D)
        self.label_embeddings = nn.Embedding(num_labels, hidden_size)
        # W_h : D → attn_dim  (projects comment encoder outputs)
        self.W_h = nn.Linear(hidden_size, attn_dim, bias=False)
        # W_l : D → attn_dim  (projects label embeddings)
        self.W_l = nn.Linear(hidden_size, attn_dim, bias=False)

    def forward(
        self,
        comment_embs: torch.Tensor,   # (B, N, D)
        comment_mask: torch.Tensor,   # (B, N)  True = real, False = pad
    ):
        label_idx = torch.arange(self.num_labels, device=comment_embs.device)
        label_vecs = self.label_embeddings(label_idx)

        if self.use_projection:
            H_prime = self.W_h(comment_embs)
            L_prime = self.W_l(label_vecs)
        else:
            H_prime = comment_embs
            L_prime = label_vecs
            
        if self.use_label_attn:
            context, attn_weights = self.label_attn(comment_embs, comment_mask)
        else:
            # Plain mean pooling — no label attention
            mask_expanded = comment_mask.unsqueeze(-1).float()
            pooled = (comment_embs * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            # Expand to (B, num_labels, D) so classifier shape is unchanged
            context = pooled.unsqueeze(1).expand(-1, len(self.classifiers), -1)
            attn_weights = torch.zeros(
                comment_embs.size(0), len(self.classifiers), comment_embs.size(1),
                device=comment_embs.device
            )
    
        scores = torch.matmul(H_prime, L_prime.T)       # (B, N, L)
        scores = scores.permute(0, 2, 1)                # (B, L, N)
        mask = comment_mask.unsqueeze(1)
        scores = scores.masked_fill(~mask, float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, comment_embs)
        return context, attn_weights


class HTLA(nn.Module):
    """
    Hierarchical Transformer with Label Attention (HTLA) for Big Five
    personality prediction on the PANDORA dataset.

    Architecture (mirrors Bama et al., 2025):

      1. Word-level encoder  — RoBERTa tokenises each comment and produces
                               a [CLS] embedding per comment.
      2. Document-level encoder — A single Transformer Encoder layer lets
                               the N comment embeddings attend to one another,
                               capturing inter-comment context for the same
                               user (§4.2.1, "hierarchical" part).
      3. Label Attention     — Projects comment embeddings (W_h) and trait
                               label embeddings (W_l) into a shared space,
                               computes per-label context vectors (§4.2.3).
      4. Classifiers         — One linear head per Big Five trait.
    """

    def __init__(self, bert_model_name=BERT_MODEL, num_labels=5, dropout=0.1,
                doc_attn_heads=8, doc_attn_ff_dim=2048, label_attn_dim=256,
                use_doc_encoder=True, use_projection=True):
        super().__init__()
        self.use_doc_encoder = use_doc_encoder
        self.bert = AutoModel.from_pretrained(bert_model_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=doc_attn_heads,
            dim_feedforward=doc_attn_ff_dim, dropout=dropout,
            batch_first=True,
        )
        self.doc_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.label_attn = LabelAttention(hidden_size, num_labels,
                                        label_attn_dim, use_projection)
        self.classifiers = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(num_labels)
        ])
    
    
    def forward(self, input_ids, attention_masks, comment_mask):
        batch_size, num_comments, seq_len = input_ids.shape
        ids_flat = input_ids.view(batch_size * num_comments, seq_len)
        masks_flat = attention_masks.view(batch_size * num_comments, seq_len)

        bert_out = self.bert(input_ids=ids_flat, attention_mask=masks_flat)
        cls_embs = self.dropout(bert_out.last_hidden_state[:, 0, :])
        comment_embs = cls_embs.view(batch_size, num_comments, -1)

        if self.use_doc_encoder:
            padding_mask = ~comment_mask
            comment_embs = self.doc_encoder(
                comment_embs, src_key_padding_mask=padding_mask
            )
            comment_embs = self.dropout(comment_embs)

        context, attn_weights = self.label_attn(comment_embs, comment_mask)
        context = self.dropout(context)

        logits = torch.cat([
            self.classifiers[i](context[:, i, :])
            for i in range(len(self.classifiers))
        ], dim=1)

        return logits, attn_weights