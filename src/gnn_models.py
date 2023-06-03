import math

import torch
import torch.nn as nn


class BaselineAttention(nn.Module):
    def __init__(
            self, input_dim, n_classes, d_model, num_heads,
    ):
        super().__init__()
        self.name = 'baseline-attention'
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim, d_model)
        self.num_heads = num_heads
        self.pos_enc = SinPositionalEncoding(d_model, max_len=8192)
        self.attention = CustomAttention(
            embed_dim=d_model, num_heads=num_heads, batch_first=True, dv=1,
            do=d_model, linear_attention=False
        )
        self.decoder = nn.Linear(d_model, n_classes)

    def forward(self, X, _):
        # X: (batch_size, context_size+1, input_dim)
        # output: (batch_siz  n_classes)
        b, np, d = X.shape

        # A mask to prevent the model from attending to the first token
        mask = torch.zeros((np, np), device=X.device)
        mask[:, 0] = 1  # No attention to the first token
        out = self.embedding(X)
        out = self.pos_enc(out)
        out, _ = self.attention(out, out, out, attn_mask=mask)
        out = self.decoder(out)
        return out


class CustomAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, batch_first, dv=None, do=None, linear_attention=True) -> None:
        super().__init__()
        assert batch_first
        # assert num_heads == 1, "Only one head is supported!"
        self.dv = dv if dv is not None else embed_dim // num_heads
        self.do = do if do is not None else embed_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.linear_attention = linear_attention
        self.Wq = nn.Parameter(torch.empty((num_heads, embed_dim, embed_dim), ))
        self.Wk = nn.Parameter(torch.empty((num_heads, embed_dim, embed_dim), ))
        self.Wv = nn.Parameter(torch.empty((num_heads, embed_dim, self.dv), ))
        self.Wo = nn.Parameter(torch.empty((num_heads * self.dv, self.do), ))

        for h in range(num_heads):
            nn.init.xavier_uniform_(self.Wq[h])
            nn.init.xavier_uniform_(self.Wk[h])
            nn.init.xavier_uniform_(self.Wv[h])
        nn.init.xavier_uniform_(self.Wo)

    def forward(self, query, key, value, attn_mask=None):
        """
        attn_mask: 0: attend, 1: do not attend
        """
        b, n, d = query.shape
        if attn_mask is not None:
            assert attn_mask.shape == (n, n)
            attn_mask = attn_mask[None, None, :, :].float()
        else:
            attn_mask = 0.0  # Attend to all nodes
        Q = query[:, None, ...] @ self.Wq[None, ...]
        K = key[:, None, ...] @ self.Wk[None, ...]
        V = value[:, None, ...] @ self.Wv[None, ...]
        QK = (Q @ torch.transpose(K, 2, 3)) * (1 - attn_mask) / math.sqrt(d)  # [b, h, n, n]
        if not self.linear_attention:
            attn_weights = torch.softmax(QK, dim=-1)
        else:
            attn_weights = QK
        QKV = attn_weights @ V  # [b, h, n, d // h]
        output = torch.transpose(QKV, 1, 2).reshape(b, n, self.num_heads * self.dv)  # [b, n, d]
        output = output @ self.Wo

        return output, attn_weights


class SinPositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1)][None, ...]
        return x
