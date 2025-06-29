import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim, num_heads=3, kernel_size=5, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU()

        self.num_heads = num_heads
        self.pos_embedding = nn.Parameter(torch.randn(42 * 42, in_channels))
        self.proj = nn.Linear(in_channels, hidden_dim)

        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)  # Dropout after attention

        self.norm1 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),  # Dropout between FFN layers
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)   # Dropout after FFN
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, M0):  # M0: [B, H, W, C]
        B, H, W, C = M0.shape
        M0_ = M0.permute(0, 3, 1, 2)  # [B, C, H, W]
        Mc = self.relu(self.conv(M0_))  # [B, out_channels, H, W]
        Mc = Mc.permute(0, 2, 3, 1)     # [B, H, W, out_channels]

        x = M0.view(B, H * W, -1)  # [B, N, in_channels]
        pos_emb = self.pos_embedding[:H * W, :].unsqueeze(0)  # [1, N, in_channels]
        x = x + pos_emb

        x_proj = self.proj(x)  # [B, N, hidden_dim]

        attn_output, _ = self.self_attn(x_proj, x_proj, x_proj)  # [B, N, hidden_dim]
        attn_output = self.dropout1(attn_output)

        M1 = self.norm1(x_proj + attn_output)
        M2 = self.norm2(M1 + self.ffn(M1))

        return M2.view(B, H, W, -1)
