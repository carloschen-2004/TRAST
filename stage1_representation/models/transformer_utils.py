"""
Transformer 通用工具模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================= Embedding =============================

class PositionalEmbedding(nn.Module):
    """位置编码"""

    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # 创建位置编码矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        # 计算位置编码
        position = torch.arange(0, max_len).float().unsqueeze(1)  # [max_len, 1]
        div_term = torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        div_term = div_term.exp()
        # 偶数维用sin，奇数维用cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 添加batch维度并注册为buffer
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)] # [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_model]


class TokenEmbedding(nn.Module):
    """值嵌入"""
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        # 使用1D卷积进行特征提取
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=3, padding=padding, padding_mode='circular')
        # Kaiming初始化
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # Conv1d需要输入格式为 [batch, channels, length]
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x # [batch_size, seq_len, d_model]


class DataEmbedding(nn.Module):
    """数据嵌入：值嵌入 + 位置嵌入"""
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x) # [batch_size, seq_len, d_model]


# ============================= RoPE Positional Embedding =============================

class RotaryPositionEmbedding(nn.Module):
    """旋转位置编码 (RoPE)"""

    def __init__(self, d_model, max_len=5000, base=10000):
        super(RotaryPositionEmbedding, self).__init__()
        self.d_model = d_model
        self.base = base
        # 生成频率
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model)) # 计算旋转的角度步长theta
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, seq_len, device=None):
        """生成旋转位置编码
        Args:
            seq_len: 序列长度
            device: 设备
        Returns:
            freqs: [seq_len, d_model]
        """
        if device is None:
            device = self.inv_freq.device

        t = torch.arange(seq_len, device=device).type_as(self.inv_freq) # 位置m
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)  # 外积得到 [seq_len, d_model/2] m * theta
        emb = torch.cat((freqs, freqs), dim=-1)  # [seq_len, d_model]
        cos = emb.cos()
        sin = emb.sin()

        return cos, sin


def apply_rotary_pos_emb(x, cos, sin):
    """应用旋转位置编码到查询或键
    Args:
        x: [batch_size, n_heads, seq_len, d_k]
        cos: [seq_len, d_model]
        sin: [seq_len, d_model]
    Returns:
        x: 应用旋转位置编码后的张量
    """
    d_k = x.shape[-1]
    seq_len = x.shape[2]

    # cos/sin: [seq_len, d_k] -> [1, 1, seq_len, d_k]
    # 这样可以正确广播到 [batch, n_heads, seq_len, d_k]
    cos = cos[:seq_len, :d_k].view(1, 1, -1, d_k)
    sin = sin[:seq_len, :d_k].view(1, 1, -1, d_k)

    # 应用旋转公式 - 需要分割cos和sin以匹配x1和x2的维度
    x1, x2 = x[..., :d_k//2], x[..., d_k//2:]
    cos1, cos2 = cos[..., :d_k//2], cos[..., d_k//2:]
    sin1, sin2 = sin[..., :d_k//2], sin[..., d_k//2:]

    x_rotated = torch.cat([
        x1 * cos2 - x2 * sin2,
        x1 * sin2 + x2 * cos2
    ], dim=-1)

    return x_rotated


# ============================= Attention =============================

class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, queries, keys, values, attn_mask=None):
        # 计算注意力分数：Q * K^T / sqrt(d_k)
        d_k = queries.shape[-1]
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(d_k)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)
        # Softmax归一化
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention) # [batch_size, query_len, key_len]
        # 加权求和：Attention * V
        output = torch.matmul(attention, values) # [batch_size, query_len, d_v]

        return output, attention


class MultiHeadAttention(nn.Module):
    """多头注意力"""
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        d_k = d_k or (d_model // n_heads)
        d_v = d_v or (d_model // n_heads)

        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v

        # Q, K, V投影层
        self.W_q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_k = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_v = nn.Linear(d_model, d_v * n_heads, bias=False)
        # 输出投影层
        self.W_o = nn.Linear(d_v * n_heads, d_model, bias=False)

        self.attention = ScaledDotProductAttention(attn_dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None):
        batch_size = query.size(0)
        # 1. 线性投影并分成多个头
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, n_heads * d_k]
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        # 2. 重塑为多头格式
        # [batch_size, seq_len, n_heads * d_k] -> [batch_size, seq_len, n_heads, d_k]
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        # 3. 应用缩放点积注意力
        # 如果attn_mask是2D的 [batch_size, query_len, key_len]，需要扩展维度
        if attn_mask is not None:
            if attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)  # [batch_size, 1, query_len, key_len]

        x, attn = self.attention(Q, K, V, attn_mask)
        # 4. 拼接多个头
        # [batch_size, n_heads, query_len, d_v] -> [batch_size, query_len, n_heads * d_v]
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        # 5. 输出投影
        output = self.W_o(x)

        return output, attn


class MultiQueryAttention(nn.Module):
    """多查询注意力 (MQA)"""

    def __init__(self, d_model, n_heads, d_k=None, d_v=None, dropout=0.1):
        super(MultiQueryAttention, self).__init__()
        d_k = d_k or (d_model // n_heads)
        d_v = d_v or (d_model // n_heads)

        self.n_heads = n_heads
        self.n_kv_heads = 1  # MQA: 只有1个KV头
        self.d_k = d_k
        self.d_v = d_v

        # Q: 多个头
        self.W_q = nn.Linear(d_model, d_k * n_heads, bias=False)
        # K, V: 只有一个头（所有查询头共享）
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, d_v, bias=False)
        # 输出投影
        self.W_o = nn.Linear(d_v * n_heads, d_model, bias=False)

        self.attention = ScaledDotProductAttention(attn_dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None):
        batch_size = query.size(0)
        # 1. 线性投影
        Q = self.W_q(query)  # [batch_size, query_len, n_heads * d_k]
        K = self.W_k(key)    # [batch_size, key_len, d_k]
        V = self.W_v(value)  # [batch_size, value_len, d_v]

        # 2. 重塑 Q 为多头格式
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) # [batch_size, n_heads, query_len, d_k]

        # 3. 扩展 K 和 V 到多个头（广播）
        K = K.unsqueeze(1).expand(-1, self.n_heads, -1, -1) # [batch_size, n_heads, key_len, d_k]
        V = V.unsqueeze(1).expand(-1, self.n_heads, -1, -1) # [batch_size, n_heads, value_len, d_v]

        # 4. 应用注意力
        if attn_mask is not None:
            if attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)

        x, attn = self.attention(Q, K, V, attn_mask)

        # 5. 拼接多头
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v) # [batch_size, query_len, n_heads * d_v]

        # 6. 输出投影
        output = self.W_o(x)

        return output, attn


class GroupedQueryAttention(nn.Module):
    """分组查询注意力 (GQA)"""

    def __init__(self, d_model, n_heads, n_kv_heads=None, d_k=None, d_v=None, dropout=0.1):
        super(GroupedQueryAttention, self).__init__()
        d_k = d_k or (d_model // n_heads)
        d_v = d_v or (d_model // n_heads)

        # n_kv_heads: Key-Value 头的数量，默认为 1（即 MQA）
        # 如果 n_kv_heads = n_heads，则等价于 MHA
        if n_kv_heads is None:
            n_kv_heads = max(1, n_heads // 4)  # 默认 KV 头数为 Q 头数的 1/4

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_k = d_k
        self.d_v = d_v
        self.n_groups = n_heads // n_kv_heads  # 每组包含的查询头数

        # Q: 多个头
        self.W_q = nn.Linear(d_model, d_k * n_heads, bias=False) # n_heads
        # K, V: 较少的头
        self.W_k = nn.Linear(d_model, d_k * n_kv_heads, bias=False) # n_kv_heads
        self.W_v = nn.Linear(d_model, d_v * n_kv_heads, bias=False) # n_kv_heads
        # 输出投影
        self.W_o = nn.Linear(d_v * n_heads, d_model, bias=False)

        self.attention = ScaledDotProductAttention(attn_dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None):
        batch_size = query.size(0)
        # 1. 线性投影
        Q = self.W_q(query)  # [batch_size, query_len, n_heads * d_k]
        K = self.W_k(key)    # [batch_size, key_len, n_kv_heads * d_k]
        V = self.W_v(value)  # [batch_size, value_len, n_kv_heads * d_v]

        # 2. 重塑为多头格式
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) # [batch_size, n_heads, query_len, d_k]
        K = K.view(batch_size, -1, self.n_kv_heads, self.d_k).transpose(1, 2) # [batch_size, n_kv_heads, key_len, d_k]
        V = V.view(batch_size, -1, self.n_kv_heads, self.d_v).transpose(1, 2) # [batch_size, n_kv_heads, value_len, d_v]

        # 3. 扩展 K 和 V 以匹配 Q 的头数
        # 每个共享头重复 n_groups 次
        K = K.repeat_interleave(self.n_groups, dim=1) # [batch_size, n_heads, key_len, d_k]
        V = V.repeat_interleave(self.n_groups, dim=1) # [batch_size, n_heads, value_len, d_v]
        
        # 4. 应用注意力
        if attn_mask is not None:
            if attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)

        x, attn = self.attention(Q, K, V, attn_mask)
        
        # 5. 拼接多头
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v) # [batch_size, query_len, n_heads * d_v]

        # 6. 输出投影
        output = self.W_o(x)

        return output, attn


# ============================= FeedForward =============================

class PositionwiseFeedForward(nn.Module):
    """逐位置前馈网络"""

    def __init__(self, d_model, d_ff, dropout=0.1, activation='relu'):
        super(PositionwiseFeedForward, self).__init__()
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x):
        # Conv1d需要 [batch, channels, length] 格式
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = x.transpose(1, 2)
        return x


class MoEFeedForward(nn.Module):
    """混合专家前馈网络 (MoE-FeedForward)"""

    def __init__(self, d_model, d_ff, num_experts=4, top_k=2, dropout=0.1,
                 activation='gelu', expert_dropout=0.1, capacity_factor=1.0):
        super(MoEFeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.capacity_factor = capacity_factor
        self.expert_dropout = expert_dropout

        # 门控网络 (Router)
        self.gate = nn.Linear(d_model, num_experts, bias=False)

        # 专家网络（每个专家是一个小型 FFN，使用 Conv1d 实现）
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, d_ff, kernel_size=1),
                nn.GELU() if activation == 'gelu' else nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(d_ff, d_model, kernel_size=1),
                nn.Dropout(dropout)
            )
            for _ in range(num_experts)
        ])

        # 负载均衡损失系数
        self.aux_loss_coeff = 0.01

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            output: [batch_size, seq_len, d_model]
            aux_loss: 辅助损失（用于负载均衡）
        """
        batch_size, seq_len, d_model = x.shape
        total_tokens = batch_size * seq_len
        x_flat = x.view(-1, d_model)  # [total_tokens, d_model]

        # 1. 门控网络计算路由权重
        gate_logits = self.gate(x_flat)
        gate_probs = F.softmax(gate_logits, dim=-1)

        # 2. 选择 Top-K 专家
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        # 归一化权重（保证每个 token 的 Top-K 权重和为 1）
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # 3. 计算每个专家的容量
        capacity = int((total_tokens * self.top_k) / self.num_experts * self.capacity_factor)

        # 4. 初始化输出
        output = torch.zeros_like(x_flat)
        aux_loss = 0.0

        # 5. 为每个专家分配并处理 token（修复后的核心部分）
        for expert_id in range(self.num_experts):
            # 找出所有选择该专家的 token（任意一个 top-k 位置）
            expert_mask = (top_k_indices == expert_id).any(dim=-1)  # [total_tokens]
            indices = torch.where(expert_mask)[0]

            if len(indices) == 0:
                continue

            # 容量限制（防止某个专家过载）
            if len(indices) > capacity:
                indices = indices[:capacity]

            # 获取专家输入
            expert_input = x_flat[indices] # [num_selected, d_model]

            # Conv1d 需要 [batch, channels, length] 格式，这里 length=1
            expert_input_conv = expert_input.unsqueeze(-1)    # [num_selected, d_model, 1]
            expert_output_conv = self.experts[expert_id](expert_input_conv)
            expert_output = expert_output_conv.squeeze(-1)    # [num_selected, d_model]

            # 计算该专家对这些 token 的加权权重（考虑 Top-K 中可能重复出现）
            weights = torch.zeros(len(indices), device=x.device)
            for k in range(self.top_k):
                mask_k = (top_k_indices[indices, k] == expert_id)
                weights[mask_k] += top_k_probs[indices, k][mask_k]

            # 加权累加到输出
            output[indices] += weights.unsqueeze(1) * expert_output

        # 6. 计算辅助损失（负载均衡损失）
        expert_counts = torch.zeros(self.num_experts, device=x.device)
        for expert_id in range(self.num_experts):
            mask = (top_k_indices == expert_id).any(dim=-1)
            expert_counts[expert_id] = mask.sum().float()

        expert_freq = expert_counts / total_tokens
        avg_gate_probs = gate_probs.mean(dim=0)

        # 标准负载均衡损失
        aux_loss = self.num_experts * (expert_freq * avg_gate_probs).sum() * self.aux_loss_coeff

        # 重塑回原始形状
        output = output.view(batch_size, seq_len, d_model)

        return output, aux_loss


class SparseMoEFeedForward(nn.Module):
    """稀疏混合专家前馈网络 (Sparse MoE)
    使用更高效的实现方式
    """

    def __init__(self, d_model, d_ff, num_experts=4, top_k=2, dropout=0.1, activation='gelu', capacity_factor=1.0, aux_loss_coeff=0.01):
        super(SparseMoEFeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.capacity_factor = capacity_factor
        self.aux_loss_coeff = aux_loss_coeff

        # 门控网络
        self.gate = nn.Linear(d_model, num_experts, bias=False)

        # 专家网络权重（合并为一个大矩阵）
        self.w_gate = nn.Parameter(torch.randn(d_model, d_ff * num_experts))
        self.w_down = nn.Parameter(torch.randn(d_ff * num_experts, d_model))

        # 初始化
        nn.init.normal_(self.w_gate, std=0.02)
        nn.init.normal_(self.w_down, std=0.02)

        self.activation = F.gelu if activation == 'gelu' else F.relu
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            x: [batch_size, seq_len, d_model]
            aux_loss: 辅助损失
        """
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)

        # 门控
        gate_logits = self.gate(x_flat)
        gate_probs = F.softmax(gate_logits, dim=-1)

        # Top-K 选择
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # 初始化输出
        output = torch.zeros_like(x_flat)

        # 为每个专家计算输出
        for expert_id in range(self.num_experts):
            # 找到使用该专家的 token
            mask = (top_k_indices == expert_id).any(dim=-1)
            indices = torch.where(mask)[0]

            if len(indices) == 0:
                continue

            # 获取输入
            expert_input = x_flat[indices]  # [num_tokens, d_model]

            # 计算专家输出
            # 上投影: [num_tokens, d_model] @ [d_model, d_ff] = [num_tokens, d_ff]
            mid = self.activation(torch.matmul(expert_input, self.w_gate[:, expert_id * self.d_ff:(expert_id + 1) * self.d_ff]))
            mid = self.dropout(mid)

            # 下投影: [num_tokens, d_ff] @ [d_ff, d_model] = [num_tokens, d_model]
            expert_output = torch.matmul(mid, self.w_down[expert_id * self.d_ff:(expert_id + 1) * self.d_ff, :])

            # 加权求和
            weights = top_k_probs[mask]
            # 对于每个 token，累加其选择的专家权重
            expert_mask = (top_k_indices[mask] == expert_id)
            weighted_output = expert_output * weights[expert_mask].unsqueeze(-1)

            output[indices] += weighted_output

        # 计算辅助损失
        expert_counts = torch.zeros(self.num_experts, device=x.device)
        for expert_id in range(self.num_experts):
            mask = (top_k_indices == expert_id).any(dim=-1)
            expert_counts[expert_id] = mask.sum()

        expert_freq = expert_counts / (batch_size * seq_len)
        avg_gate_probs = gate_probs.mean(dim=0)
        aux_loss = (expert_freq * avg_gate_probs).sum() * self.aux_loss_coeff

        output = output.view(batch_size, seq_len, d_model) # [batch_size, seq_len, d_model]

        return output, aux_loss
