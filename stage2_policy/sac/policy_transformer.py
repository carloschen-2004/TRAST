import torch
import torch.nn as nn
from stage1_representation.models.transformer import MultiHeadAttention

class policy_transformer_stock_atten2(nn.Module):
    def __init__(self, d_model = 128, n_heads = 4, dropout = 0.0, lr = 0.0001):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        self.attn1 = MultiHeadAttention(d_model = d_model, n_heads = n_heads, dropout = dropout)
        self.attn2 = MultiHeadAttention(d_model = d_model, n_heads = n_heads, dropout = dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.optimizer = torch.optim.Adam( list(self.attn1.parameters()) + list(self.attn2.parameters()), lr = lr)

    def forward(self, relational_feature, temporal_feature_short, temporal_feature_long, holding, mask = None):
        temporal_hybrid, _ = self.attn1(
            query = temporal_feature_long,
            key = temporal_feature_short,
            value = temporal_feature_short,
            attn_mask = mask
        )
        temporal_feature_long = temporal_feature_long + self.dropout(temporal_hybrid)
        temporal_feature = self.norm1(temporal_feature_long)

        temporal_relational, _ = self.attn2(
            query = temporal_feature,
            key = relational_feature,
            value = relational_feature,
            attn_mask = mask
        )
        temporal_feature = temporal_feature + self.dropout(temporal_relational)
        hybrid_feature = self.norm2(temporal_feature)

        if holding is not None:
            combined_feature = torch.cat((hybrid_feature, holding), dim = -1)  # [B, N, D+1]
        else:
            combined_feature = hybrid_feature

        return combined_feature
