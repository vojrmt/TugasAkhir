import torch
import torch.nn as nn
from transformers import AutoModel

BERT_MODEL = "bert-base-uncased"


class LabelAttention(nn.Module):

    def __init__(self, hidden_size: int, num_labels: int = 5,
                 attn_dim: int = 256):
        super().__init__()
        self.num_labels      = num_labels
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
        # ── Project comment embeddings into attention space ──────────────────
        # H' = H W_h  →  (B, N, attn_dim)
        H_prime = self.W_h(comment_embs)

        # ── Project label embeddings into attention space ────────────────────
        label_idx = torch.arange(self.num_labels, device=comment_embs.device)
        label_vecs = self.label_embeddings(label_idx)   # (L, D)
        # L' = L W_l  →  (L, attn_dim)
        L_prime = self.W_l(label_vecs)

        # ── Attention scores  e_ti = H' · L'^T ──────────────────────────────
        # (B, N, attn_dim) × (attn_dim, L)  →  (B, N, L)
        # Transpose to (B, L, N) so softmax runs over the comment dimension
        scores = torch.matmul(H_prime, L_prime.T)          # (B, N, L)
        scores = scores.permute(0, 2, 1)                   # (B, L, N)

        # ── Mask padding comments (set to -inf before softmax) ───────────────
        mask = comment_mask.unsqueeze(1)                   # (B, 1, N)
        scores = scores.masked_fill(~mask, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)       # (B, L, N)

        # ── Context vector  c_t = Σ α_i h_i ─────────────────────────────────
        # (B, L, N) × (B, N, D)  →  (B, L, D)
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

    def __init__(
        self,
        bert_model_name: str  = BERT_MODEL,
        num_labels: int       = 5,
        dropout: float        = 0.1,
        doc_attn_heads: int   = 8,
        doc_attn_ff_dim: int  = 2048,
        label_attn_dim: int   = 256,
    ):
        super().__init__()

        # ── 1. Word-level encoder (RoBERTa) ──────────────────────────────────
        self.bert    = AutoModel.from_pretrained(bert_model_name)
        hidden_size  = self.bert.config.hidden_size   # 768 for roberta-base
        self.dropout = nn.Dropout(dropout)

        # ── 2. Document-level Transformer Encoder ────────────────────────────
        # One standard Transformer Encoder layer operates over the sequence of
        # N comment [CLS] embeddings, allowing them to attend to each other.
        # PyTorch's TransformerEncoderLayer uses src_key_padding_mask where
        # True means "ignore this position" — i.e. the INVERSE of comment_mask.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = hidden_size,
            nhead           = doc_attn_heads,
            dim_feedforward = doc_attn_ff_dim,
            dropout         = dropout,
            batch_first     = True,   # input shape: (B, N, D)
        )
        self.doc_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # ── 3. Label Attention (with W_h, W_l projection matrices) ───────────
        self.label_attn = LabelAttention(hidden_size, num_labels, label_attn_dim)

        # ── 4. Per-trait classifiers ──────────────────────────────────────────
        self.classifiers = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(num_labels)
        ])

    def forward(
        self,
        input_ids:       torch.Tensor,   # (B, N, seq_len)
        attention_masks: torch.Tensor,   # (B, N, seq_len)
        comment_mask:    torch.Tensor,   # (B, N)  True=real, False=pad
    ):
        batch_size, num_comments, seq_len = input_ids.shape

        # ── Step 1 : Word-level encoding ─────────────────────────────────────
        # Flatten (B, N, seq_len) → (B*N, seq_len) for batched RoBERTa pass
        ids_flat   = input_ids.view(batch_size * num_comments, seq_len)
        masks_flat = attention_masks.view(batch_size * num_comments, seq_len)

        bert_out       = self.bert(input_ids=ids_flat, attention_mask=masks_flat)
        cls_embs       = self.dropout(bert_out.last_hidden_state[:, 0, :])
        # Restore shape → (B, N, D)
        comment_embs   = cls_embs.view(batch_size, num_comments, -1)

        # ── Step 2 : Document-level Transformer Encoder ──────────────────────
        # PyTorch's src_key_padding_mask uses True to MASK (ignore) positions,
        # which is the inverse of our comment_mask convention (True = keep).
        padding_mask = ~comment_mask                           # (B, N)
        comment_embs = self.doc_encoder(
            comment_embs,
            src_key_padding_mask=padding_mask
        )                                                      # (B, N, D)
        comment_embs = self.dropout(comment_embs)

        # ── Step 3 : Label Attention (W_h / W_l projection) ──────────────────
        context, attn_weights = self.label_attn(comment_embs, comment_mask)
        # context: (B, num_labels, D)
        context = self.dropout(context)

        # ── Step 4 : Per-trait classification ────────────────────────────────
        logits = torch.cat([
            self.classifiers[i](context[:, i, :])
            for i in range(len(self.classifiers))
        ], dim=1)                                              # (B, num_labels)

        return logits, attn_weights