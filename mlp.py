import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, D, hidden_dim, num_class, num_heads, dropout=0.1, max_entities=128):
        super().__init__()
        self.proj = nn.Linear(D, hidden_dim)

        self.pos_embedding = nn.Parameter(torch.randn(max_entities * max_entities, hidden_dim))  # PE for N x N

        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=dropout)

        self.norm1 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, num_class)

    def forward(self, x):  # x: [B, N, N, D]
        B, N, _, D = x.shape
        x = x.view(B, N * N, D)                    # [B, N², D]
        x = self.proj(x)                           # [B, N², hidden_dim]

        pos_emb = self.pos_embedding[:N * N, :].unsqueeze(0)  # [1, N², hidden_dim]
        x = x + pos_emb

        attn_output, _ = self.self_attn(x, x, x)   # [B, N², hidden_dim]
        x = self.norm1(x + attn_output)
        x_ffn = self.norm2(x + self.ffn(x))
        
        x_out = self.out_proj(x_ffn)               # [B, N², D]
        return x_out.view(B, N, N, D)