"""
- Transformer_base: 基础 Transformer (MHA + PE + FFN)
- Transformer_MHA_RoPE_MoE: MHA + RoPE + MoE
- Transformer_MQA_RoPE_MoE: MQA + RoPE + MoE
- Transformer_GQA_RoPE_MoE: GQA + RoPE + MoE
"""

import torch.nn as nn

# 导入基础组件
from .transformer_utils import (
    # Embedding
    TokenEmbedding,
    DataEmbedding,

    # RoPE
    RotaryPositionEmbedding,
    apply_rotary_pos_emb,

    # Attention
    MultiHeadAttention,
    MultiQueryAttention,
    GroupedQueryAttention,

    # FeedForward
    PositionwiseFeedForward,
    SparseMoEFeedForward
)

# ============================= Transformer Layer =============================

class EncoderLayer(nn.Module):
    """编码器层"""
    def __init__(self, d_model, d_ff, n_heads, dropout=0.1, activation='relu'):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask = None):
        # 1. Self-Attention + 残差连接 + LayerNorm
        new_x, attn = self.self_attention(x, x, x, attn_mask)
        x = x + self.dropout(new_x)
        x = self.norm1(x)
        # 2. FeedForward + 残差连接 + LayerNorm
        new_x = self.feed_forward(x)
        x = x + self.dropout(new_x)
        x = self.norm2(x)

        return x, attn


class EncoderLayer_RoPE_MoE(nn.Module):
    """编码器层 - 支持 RoPE + MoE"""
    def __init__(
        self,
        d_model,
        d_ff,
        n_heads,
        attn_type='mha',  # 'mha', 'mqa', 'gqa'
        n_kv_heads=None,  # 用于 GQA
        num_experts=4,
        top_k=2,
        dropout=0.1,
        activation='gelu',
        capacity_factor=1.0,
        aux_loss_coeff=0.01
    ):
        super(EncoderLayer_RoPE_MoE, self).__init__()

        # 选择注意力机制
        if attn_type == 'mha':
            self.self_attention = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        elif attn_type == 'mqa':
            self.self_attention = MultiQueryAttention(d_model, n_heads, dropout=dropout)
        elif attn_type == 'gqa':
            self.self_attention = GroupedQueryAttention(d_model, n_heads, n_kv_heads, dropout=dropout)
        else:
            raise ValueError(f"Unknown attention type: {attn_type}")

        # 使用 MoE 前馈网络
        self.feed_forward = SparseMoEFeedForward(
            d_model, d_ff, num_experts, top_k, dropout, activation, capacity_factor, aux_loss_coeff
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # RoPE 位置编码
        self.rope = RotaryPositionEmbedding(d_model // n_heads)

    def forward(self, x, attn_mask=None):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            attn_mask: attention mask
        Returns:
            x: [batch_size, seq_len, d_model]
            attn: attention weights
            aux_loss: MoE 辅助损失
        """
        _, seq_len, _ = x.shape

        # 生成 RoPE
        cos, sin = self.rope(seq_len, device=x.device)

        # 1. Self-Attention with RoPE
        # 注意：RoPE需要在注意力计算前应用到Q和K上 这里需要修改attention类的实现以支持RoPE 为简化，我们在当前层中处理
        new_x, attn = self._attention_with_rope(x, x, x, attn_mask, cos, sin)
        x = x + self.dropout(new_x)
        x = self.norm1(x)

        # 2. MoE FeedForward
        new_x, aux_loss = self.feed_forward(x)
        x = x + self.dropout(new_x)
        x = self.norm2(x)

        return x, attn, aux_loss

    def _attention_with_rope(self, query, key, value, attn_mask, cos, sin):
        """应用带RoPE的注意力"""
        batch_size = query.size(0)

        # 获取Q, K, V的投影
        if hasattr(self.self_attention, 'W_q'):
            Q = self.self_attention.W_q(query)
            K = self.self_attention.W_k(key)
            V = self.self_attention.W_v(value)
        else:
            raise NotImplementedError("Attention layer must have W_q, W_k, W_v")

        d_k = Q.shape[-1] // self.self_attention.n_heads if hasattr(self.self_attention, 'n_heads') else Q.shape[-1]

        # 重塑为多头格式
        if hasattr(self.self_attention, 'n_heads'):
            Q = Q.view(batch_size, -1, self.self_attention.n_heads, d_k).transpose(1, 2)

            # 处理MQA/GQA的K, V维度不同的情况
            if hasattr(self.self_attention, 'n_kv_heads') and self.self_attention.n_kv_heads != self.self_attention.n_heads:
                # GQA/MQA: K, V 的头数较少，需要按照 n_kv_heads 来 reshape
                kv_d_k = K.shape[-1] // self.self_attention.n_kv_heads
                K = K.view(batch_size, -1, self.self_attention.n_kv_heads, kv_d_k).transpose(1, 2)
                V = V.view(batch_size, -1, self.self_attention.n_kv_heads, kv_d_k if hasattr(self.self_attention, 'd_v') else kv_d_k).transpose(1, 2)
                # 应用RoPE到K
                K = apply_rotary_pos_emb(K, cos, sin)

                # 扩展K, V
                if isinstance(self.self_attention, GroupedQueryAttention):
                    K = K.repeat_interleave(self.self_attention.n_groups, dim=1)
                    V = V.repeat_interleave(self.self_attention.n_groups, dim=1)
                else:  # MQA
                    K = K.expand(-1, self.self_attention.n_heads, -1, -1)
                    V = V.expand(-1, self.self_attention.n_heads, -1, -1)
            else:
                # MHA: K, V 按照 n_heads reshape
                K = K.view(batch_size, -1, self.self_attention.n_heads, d_k).transpose(1, 2)
                V = V.view(batch_size, -1, self.self_attention.n_heads, d_k if hasattr(self.self_attention, 'd_v') else d_k).transpose(1, 2)
                # 应用RoPE到K
                K = apply_rotary_pos_emb(K, cos, sin)
        else:
            # 其他情况
            pass

        # 应用RoPE到Q
        Q = apply_rotary_pos_emb(Q, cos, sin)

        # 应用注意力
        if attn_mask is not None:
            if attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)

        x, attn = self.self_attention.attention(Q, K, V, attn_mask)

        # 拼接多头
        if hasattr(self.self_attention, 'n_heads'):
            d_v = V.shape[-1]
            x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.self_attention.n_heads * d_v)
            x = self.self_attention.W_o(x)

        return x, attn


class DecoderLayer(nn.Module):
    """解码器层"""
    def __init__(self, d_model, d_ff, n_heads, dropout=0.1, activation='relu'):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # 1. Self-Attention + 残差连接 + LayerNorm
        new_x = self.self_attention(x, x, x, x_mask)[0]
        x = x + self.dropout(new_x)
        x = self.norm1(x)
        # 2. Cross-Attention + 残差连接 + LayerNorm
        new_x = self.cross_attention(x, cross, cross, cross_mask)[0]
        x = x + self.dropout(new_x)
        x = self.norm2(x)
        # 3. FeedForward + 残差连接 + LayerNorm
        new_x = self.feed_forward(x)
        x = x + self.dropout(new_x)
        x = self.norm3(x)

        return x


class DecoderLayer_RoPE_MoE(nn.Module):
    """解码器层 - 支持 RoPE + MoE"""
    def __init__(
        self,
        d_model,
        d_ff,
        n_heads,
        attn_type='mha',  # 'mha', 'mqa', 'gqa'
        n_kv_heads=None,  # 用于 GQA
        num_experts=4,
        top_k=2,
        dropout=0.1,
        activation='gelu',
        capacity_factor=1.0,
        aux_loss_coeff=0.01
    ):
        super(DecoderLayer_RoPE_MoE, self).__init__()

        # 选择自注意力机制
        if attn_type == 'mha':
            self.self_attention = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        elif attn_type == 'mqa':
            self.self_attention = MultiQueryAttention(d_model, n_heads, dropout=dropout)
        elif attn_type == 'gqa':
            self.self_attention = GroupedQueryAttention(d_model, n_heads, n_kv_heads, dropout=dropout)
        else:
            raise ValueError(f"Unknown attention type: {attn_type}")

        # 选择交叉注意力机制（通常与自注意力类型相同）
        if attn_type == 'mha':
            self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        elif attn_type == 'mqa':
            self.cross_attention = MultiQueryAttention(d_model, n_heads, dropout=dropout)
        elif attn_type == 'gqa':
            self.cross_attention = GroupedQueryAttention(d_model, n_heads, n_kv_heads, dropout=dropout)

        # 使用 MoE 前馈网络
        self.feed_forward = SparseMoEFeedForward(
            d_model, d_ff, num_experts, top_k, dropout, activation, capacity_factor, aux_loss_coeff
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # RoPE 位置编码
        self.rope = RotaryPositionEmbedding(d_model // n_heads)

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        """
        Args:
            x: [batch_size, dec_seq_len, d_model]
            cross: [batch_size, enc_seq_len, d_model]
            x_mask: decoder self-attention mask
            cross_mask: cross-attention mask
        Returns:
            x: [batch_size, dec_seq_len, d_model]
            aux_loss: MoE 辅助损失
        """
        batch_size, dec_seq_len, _ = x.shape

        # 生成 RoPE
        cos, sin = self.rope(dec_seq_len, device=x.device)

        # 1. Self-Attention with RoPE
        new_x = self._attention_with_rope(x, x, x, x_mask, cos, sin)
        x = x + self.dropout(new_x)
        x = self.norm1(x)

        # 2. Cross-Attention with RoPE (对Query应用RoPE)
        enc_seq_len = cross.shape[1]
        cos_enc, sin_enc = self.rope(enc_seq_len, device=cross.device)
        new_x = self._cross_attention_with_rope(x, cross, cross, cross_mask, cos_enc, sin_enc)
        x = x + self.dropout(new_x)
        x = self.norm2(x)

        # 3. MoE FeedForward
        new_x, aux_loss = self.feed_forward(x)
        x = x + self.dropout(new_x)
        x = self.norm3(x)

        return x, aux_loss

    def _attention_with_rope(self, query, key, value, attn_mask, cos, sin):
        """应用带RoPE的自注意力"""
        batch_size = query.size(0)
        d_k = self.self_attention.d_k if hasattr(self.self_attention, 'd_k') else (query.shape[-1] // self.self_attention.n_heads)
        d_v = self.self_attention.d_v if hasattr(self.self_attention, 'd_v') else d_k

        Q = self.self_attention.W_q(query)
        K = self.self_attention.W_k(key)
        V = self.self_attention.W_v(value)

        Q = Q.view(batch_size, -1, self.self_attention.n_heads, d_k).transpose(1, 2)

        # 处理MQA/GQA的K, V
        if hasattr(self.self_attention, 'n_kv_heads') and self.self_attention.n_kv_heads != self.self_attention.n_heads:
            # GQA/MQA: K, V 按照 n_kv_heads reshape
            kv_d_k = K.shape[-1] // self.self_attention.n_kv_heads
            kv_d_v = V.shape[-1] // self.self_attention.n_kv_heads
            K = K.view(batch_size, -1, self.self_attention.n_kv_heads, kv_d_k).transpose(1, 2)
            V = V.view(batch_size, -1, self.self_attention.n_kv_heads, kv_d_v).transpose(1, 2)

            K = apply_rotary_pos_emb(K, cos, sin)

            if isinstance(self.self_attention, GroupedQueryAttention):
                K = K.repeat_interleave(self.self_attention.n_groups, dim=1)
                V = V.repeat_interleave(self.self_attention.n_groups, dim=1)
            else:
                K = K.expand(-1, self.self_attention.n_heads, -1, -1)
                V = V.expand(-1, self.self_attention.n_heads, -1, -1)
        else:
            # MHA: K, V 按照 n_heads reshape
            K = K.view(batch_size, -1, self.self_attention.n_heads, d_k).transpose(1, 2)
            V = V.view(batch_size, -1, self.self_attention.n_heads, d_v).transpose(1, 2)
            K = apply_rotary_pos_emb(K, cos, sin)

        Q = apply_rotary_pos_emb(Q, cos, sin)

        if attn_mask is not None:
            if attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)

        x, _ = self.self_attention.attention(Q, K, V, attn_mask)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.self_attention.n_heads * d_v)
        x = self.self_attention.W_o(x)

        return x

    def _cross_attention_with_rope(self, query, key, value, attn_mask, cos, sin):
        """应用带RoPE的交叉注意力"""
        batch_size = query.size(0)
        d_k = self.cross_attention.d_k if hasattr(self.cross_attention, 'd_k') else (query.shape[-1] // self.cross_attention.n_heads)
        d_v = self.cross_attention.d_v if hasattr(self.cross_attention, 'd_v') else d_k

        Q = self.cross_attention.W_q(query)
        K = self.cross_attention.W_k(key)
        V = self.cross_attention.W_v(value)

        Q = Q.view(batch_size, -1, self.cross_attention.n_heads, d_k).transpose(1, 2)

        # 处理MQA/GQA的K, V
        if hasattr(self.cross_attention, 'n_kv_heads') and self.cross_attention.n_kv_heads != self.cross_attention.n_heads:
            # GQA/MQA: K, V 按照 n_kv_heads reshape
            # 注意：K/V 的输出维度已经是 d_k * n_kv_heads，而不是 d_model
            kv_d_k = d_k  # GQA/MQA 中，d_k 对每个 KV 头都是一样的
            kv_d_v = d_v if hasattr(self.cross_attention, 'd_v') else d_k

            # K 和 V 的形状：[batch_size, seq_len, d_k * n_kv_heads]
            # 需要 reshape 为：[batch_size, seq_len, n_kv_heads, d_k]
            K = K.view(batch_size, -1, self.cross_attention.n_kv_heads, kv_d_k).transpose(1, 2)
            V = V.view(batch_size, -1, self.cross_attention.n_kv_heads, kv_d_v).transpose(1, 2)

            K = apply_rotary_pos_emb(K, cos, sin)

            if isinstance(self.cross_attention, GroupedQueryAttention):
                K = K.repeat_interleave(self.cross_attention.n_groups, dim=1)
                V = V.repeat_interleave(self.cross_attention.n_groups, dim=1)
            else:  # MQA
                K = K.expand(-1, self.cross_attention.n_heads, -1, -1)
                V = V.expand(-1, self.cross_attention.n_heads, -1, -1)
        else:
            # MHA: K, V 按照 n_heads reshape
            K = K.view(batch_size, -1, self.cross_attention.n_heads, d_k).transpose(1, 2)
            V = V.view(batch_size, -1, self.cross_attention.n_heads, d_v).transpose(1, 2)
            K = apply_rotary_pos_emb(K, cos, sin)

        Q = apply_rotary_pos_emb(Q, cos, sin)

        if attn_mask is not None:
            if attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)

        x, _ = self.cross_attention.attention(Q, K, V, attn_mask)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.cross_attention.n_heads * d_v)
        x = self.cross_attention.W_o(x)

        return x

# ============================= Encoder & Decoder =============================

class Encoder(nn.Module):
    """编码器"""
    def __init__(self, encoder_layers, norm_layer = None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(encoder_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        for layer in self.layers:
            x, attn = layer(x, attn_mask)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class Encoder_RoPE_MoE(nn.Module):
    """编码器 - 支持 RoPE + MoE"""
    def __init__(self, encoder_layers, norm_layer=None):
        super(Encoder_RoPE_MoE, self).__init__()
        self.layers = nn.ModuleList(encoder_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        total_aux_loss = 0.0

        for layer in self.layers:
            x, attn, aux_loss = layer(x, attn_mask)
            attns.append(attn)
            total_aux_loss += aux_loss

        if self.norm is not None:
            x = self.norm(x)

        return x, attns, total_aux_loss


class Decoder(nn.Module):
    """解码器"""
    def __init__(self, decoder_layers, norm_layer = None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(decoder_layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask, cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x


class Decoder_RoPE_MoE(nn.Module):
    """解码器 - 支持 RoPE + MoE"""
    def __init__(self, decoder_layers, norm_layer=None):
        super(Decoder_RoPE_MoE, self).__init__()
        self.layers = nn.ModuleList(decoder_layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        total_aux_loss = 0.0

        for layer in self.layers:
            x, aux_loss = layer(x, cross, x_mask, cross_mask)
            total_aux_loss += aux_loss

        if self.norm is not None:
            x = self.norm(x)

        return x, total_aux_loss

# ============================= Main Model =============================

class Transformer_base(nn.Module):
    """完整的Transformer模型"""

    def __init__(
        self,
        enc_in,      # Encoder输入维度（特征数）
        dec_in,      # Decoder输入维度
        c_out,       # 输出维度
        d_model = 128,
        n_heads = 4,
        e_layers = 2,  # Encoder层数
        d_layers = 1,  # Decoder层数
        d_ff = 256,    # 前馈网络隐藏层维度
        dropout = 0.0,
        activation = 'gelu'
    ):
        super(Transformer_base, self).__init__()

        # 嵌入层
        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, dropout)

        # Encoder
        encoder_layers = [
            EncoderLayer(d_model, d_ff, n_heads, dropout, activation)
            for _ in range(e_layers)
        ]
        self.encoder = Encoder(encoder_layers, norm_layer=nn.LayerNorm(d_model))

        # Decoder
        decoder_layers = [
            DecoderLayer(d_model, d_ff, n_heads, dropout, activation)
            for _ in range(d_layers)
        ]
        self.decoder = Decoder(decoder_layers, norm_layer=nn.LayerNorm(d_model))

        # 输出投影层
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_dec, enc_self_mask = None, dec_self_mask = None, dec_enc_mask = None):
        # 1. 嵌入
        enc_out = self.enc_embedding(x_enc)
        dec_out = self.dec_embedding(x_dec)
        # 2. Encoder
        enc_out, _ = self.encoder(enc_out, attn_mask=enc_self_mask) # [batch_size, enc_seq_len, d_model]
        # 3. Decoder
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask) # [batch_size, dec_seq_len, d_model]
        # 4. 输出投影
        output = self.projection(dec_out) # [batch_size, dec_seq_len, c_out]

        return enc_out, dec_out, output


class Transformer_MHA_RoPE_MoE(nn.Module):
    """Transformer with MHA + RoPE + MoE"""

    def __init__(
        self,
        enc_in,      # Encoder输入维度（特征数）
        dec_in,      # Decoder输入维度
        c_out,       # 输出维度
        d_model = 128,
        n_heads = 4,
        e_layers = 2,  # Encoder层数
        d_layers = 1,  # Decoder层数
        d_ff = 256,    # 前馈网络隐藏层维度
        num_experts = 4,  # MoE专家数
        top_k = 2,        # Top-K专家选择
        dropout = 0.0,
        activation = 'gelu',
        capacity_factor = 1.0,
        aux_loss_coeff = 0.01
    ):
        super(Transformer_MHA_RoPE_MoE, self).__init__()

        # 嵌入层（仅包含值嵌入，不包含位置编码，因为使用RoPE）
        self.enc_embedding = TokenEmbedding(enc_in, d_model)
        self.dec_embedding = TokenEmbedding(dec_in, d_model)
        self.dropout = nn.Dropout(dropout)

        # Encoder - MHA + RoPE + MoE
        encoder_layers = [
            EncoderLayer_RoPE_MoE(
                d_model, d_ff, n_heads,
                attn_type='mha',
                num_experts=num_experts,
                top_k=top_k,
                dropout=dropout,
                activation=activation,
                capacity_factor=capacity_factor,
                aux_loss_coeff=aux_loss_coeff
            )
            for _ in range(e_layers)
        ]
        self.encoder = Encoder_RoPE_MoE(encoder_layers, norm_layer=nn.LayerNorm(d_model))

        # Decoder - MHA + RoPE + MoE
        decoder_layers = [
            DecoderLayer_RoPE_MoE(
                d_model, d_ff, n_heads,
                attn_type='mha',
                num_experts=num_experts,
                top_k=top_k,
                dropout=dropout,
                activation=activation,
                capacity_factor=capacity_factor,
                aux_loss_coeff=aux_loss_coeff
            )
            for _ in range(d_layers)
        ]
        self.decoder = Decoder_RoPE_MoE(decoder_layers, norm_layer=nn.LayerNorm(d_model))

        # 输出投影层
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, return_aux_loss=False):
        # 1. 嵌入（仅值嵌入）
        enc_out = self.enc_embedding(x_enc)
        dec_out = self.dec_embedding(x_dec)
        enc_out = self.dropout(enc_out)
        dec_out = self.dropout(dec_out)

        # 2. Encoder
        enc_out, _, enc_aux_loss = self.encoder(enc_out, attn_mask=enc_self_mask)
        # 3. Decoder
        dec_out, dec_aux_loss = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        # 4. 输出投影
        output = self.projection(dec_out)
        # 5. 合并MoE辅助损失
        total_aux_loss = enc_aux_loss + dec_aux_loss

        if return_aux_loss:
            return enc_out, dec_out, output, total_aux_loss
        else:
            return enc_out, dec_out, output


class Transformer_MQA_RoPE_MoE(nn.Module):
    """Transformer with MQA + RoPE + MoE"""

    def __init__(
        self,
        enc_in,
        dec_in,
        c_out,
        d_model = 128,
        n_heads = 4,
        e_layers = 2,
        d_layers = 1,
        d_ff = 256,
        num_experts = 4,
        top_k = 2,
        dropout = 0.0,
        activation = 'gelu',
        capacity_factor = 1.0,
        aux_loss_coeff = 0.01
    ):
        super(Transformer_MQA_RoPE_MoE, self).__init__()

        # 嵌入层
        self.enc_embedding = TokenEmbedding(enc_in, d_model)
        self.dec_embedding = TokenEmbedding(dec_in, d_model)
        self.dropout = nn.Dropout(dropout)

        # Encoder - MQA + RoPE + MoE
        encoder_layers = [
            EncoderLayer_RoPE_MoE(
                d_model, d_ff, n_heads,
                attn_type='mqa',
                num_experts=num_experts,
                top_k=top_k,
                dropout=dropout,
                activation=activation,
                capacity_factor=capacity_factor,
                aux_loss_coeff=aux_loss_coeff
            )
            for _ in range(e_layers)
        ]
        self.encoder = Encoder_RoPE_MoE(encoder_layers, norm_layer=nn.LayerNorm(d_model))

        # Decoder - MQA + RoPE + MoE
        decoder_layers = [
            DecoderLayer_RoPE_MoE(
                d_model, d_ff, n_heads,
                attn_type='mqa',
                num_experts=num_experts,
                top_k=top_k,
                dropout=dropout,
                activation=activation,
                capacity_factor=capacity_factor,
                aux_loss_coeff=aux_loss_coeff
            )
            for _ in range(d_layers)
        ]
        self.decoder = Decoder_RoPE_MoE(decoder_layers, norm_layer=nn.LayerNorm(d_model))

        # 输出投影层
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, return_aux_loss=False):
        # 1. 嵌入
        enc_out = self.enc_embedding(x_enc)
        dec_out = self.dec_embedding(x_dec)
        enc_out = self.dropout(enc_out)
        dec_out = self.dropout(dec_out)

        # 2. Encoder
        enc_out, _, enc_aux_loss = self.encoder(enc_out, attn_mask=enc_self_mask)
        # 3. Decoder
        dec_out, dec_aux_loss = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        # 4. 输出投影
        output = self.projection(dec_out)
        # 5. 合并MoE辅助损失
        total_aux_loss = enc_aux_loss + dec_aux_loss

        if return_aux_loss:
            return enc_out, dec_out, output, total_aux_loss
        else:
            return enc_out, dec_out, output


class Transformer_GQA_RoPE_MoE(nn.Module):
    """Transformer with GQA + RoPE + MoE"""

    def __init__(
        self,
        enc_in,
        dec_in,
        c_out,
        d_model = 128,
        n_heads = 4,
        n_kv_heads = None,  # KV头数，默认为n_heads的1/4
        e_layers = 2,
        d_layers = 1,
        d_ff = 256,
        num_experts = 4,
        top_k = 2,
        dropout = 0.0,
        activation = 'gelu',
        capacity_factor = 1.0,
        aux_loss_coeff = 0.01
    ):
        super(Transformer_GQA_RoPE_MoE, self).__init__()

        if n_kv_heads is None:
            n_kv_heads = max(1, n_heads // 4)

        # 嵌入层
        self.enc_embedding = TokenEmbedding(enc_in, d_model)
        self.dec_embedding = TokenEmbedding(dec_in, d_model)
        self.dropout = nn.Dropout(dropout)

        # Encoder - GQA + RoPE + MoE
        encoder_layers = [
            EncoderLayer_RoPE_MoE(
                d_model, d_ff, n_heads,
                attn_type='gqa',
                n_kv_heads=n_kv_heads,
                num_experts=num_experts,
                top_k=top_k,
                dropout=dropout,
                activation=activation,
                capacity_factor=capacity_factor,
                aux_loss_coeff=aux_loss_coeff
            )
            for _ in range(e_layers)
        ]
        self.encoder = Encoder_RoPE_MoE(encoder_layers, norm_layer=nn.LayerNorm(d_model))

        # Decoder - GQA + RoPE + MoE
        decoder_layers = [
            DecoderLayer_RoPE_MoE(
                d_model, d_ff, n_heads,
                attn_type='gqa',
                n_kv_heads=n_kv_heads,
                num_experts=num_experts,
                top_k=top_k,
                dropout=dropout,
                activation=activation,
                capacity_factor=capacity_factor,
                aux_loss_coeff=aux_loss_coeff
            )
            for _ in range(d_layers)
        ]
        self.decoder = Decoder_RoPE_MoE(decoder_layers, norm_layer=nn.LayerNorm(d_model))

        # 输出投影层
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, return_aux_loss=False):
        # 1. 嵌入
        enc_out = self.enc_embedding(x_enc)
        dec_out = self.dec_embedding(x_dec)
        enc_out = self.dropout(enc_out)
        dec_out = self.dropout(dec_out)
        # 2. Encoder
        enc_out, _, enc_aux_loss = self.encoder(enc_out, attn_mask=enc_self_mask)
        # 3. Decoder
        dec_out, dec_aux_loss = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        # 4. 输出投影
        output = self.projection(dec_out)
        # 5. 合并MoE辅助损失
        total_aux_loss = enc_aux_loss + dec_aux_loss

        if return_aux_loss:
            return enc_out, dec_out, output, total_aux_loss
        else:
            return enc_out, dec_out, output
