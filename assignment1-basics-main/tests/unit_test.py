from jaxtyping import Float, Bool
import torch
from torch import Tensor
from einops import einsum, rearrange
import numpy as np

def save_4d_tensor(tensor, filename):
    # 保存形状信息
    shape = tensor.shape
    with open(filename, 'w') as f:
        # 第一行保存形状
        f.write(' '.join(map(str, shape)) + '\n')
        # 保存展平的数据
        flattened = tensor.flatten().numpy()
        for value in flattened:
            f.write(f'{value:.6f}\n')

class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, d_v, device=None, dtype=None):
        super().__init__()
        self.d_v = d_v
        
    def forward(self, Q: Float[Tensor, " ... queries d_k"],
                K: Float[Tensor, " ... keys d_k"],
                V: Float[Tensor, " ... values d_v"],
                mask: Bool[Tensor, " ... queries keys"] | None = None) -> torch.Tensor:
        d_k = self.d_v
        print(Q.shape, K.shape)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
        save_4d_tensor(scores, "scores_.txt")
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        attn_weights = torch.softmax(scores, dim=-1)
        output = einsum(attn_weights, V, "... queries keys, ... keys d_v -> ... queries d_v")
        return output
    
class CausalMultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scaled_dot_product_attention = ScaledDotProductAttention(d_model)
        
    def forward(self, q_proj_weight: Float[Tensor, " d_k d_in"],
                k_proj_weight: Float[Tensor, " d_k d_in"],
                v_proj_weight: Float[Tensor, " d_v d_in"],
                o_proj_weight: Float[Tensor, " d_model d_v"],
                in_features: Float[Tensor, " ... sequence_length d_in"],) -> Float[Tensor, " ... sequence_length d_out"]:
        q = einsum(in_features, q_proj_weight, "... sequence_length d_in, d_k d_in -> ... sequence_length d_k")
        k = einsum(in_features, k_proj_weight, "... sequence_length d_in, d_k d_in -> ... sequence_length d_k")
        v = einsum(in_features, v_proj_weight, "... sequence_length d_in, d_v d_in -> ... sequence_length d_v")
        
        q = rearrange(q, " ... sequence_length (num_heads head_dim) ->  ... sequence_length num_heads head_dim", num_heads=self.num_heads)
        k = rearrange(k, " ... sequence_length (num_heads head_dim) ->  ... sequence_length num_heads head_dim", num_heads=self.num_heads)
        v = rearrange(v, " ... sequence_length (num_heads head_dim) ->  ... sequence_length num_heads head_dim", num_heads=self.num_heads)
        
        q = rearrange(q, "... seq heads head_dim -> ... heads seq head_dim")
        k = rearrange(k, "... seq heads head_dim -> ... heads seq head_dim")
        v = rearrange(v, "... seq heads head_dim -> ... heads seq head_dim")
        
        seq_len = q.shape[-2]
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=in_features.device), diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0)
        
        out = self.scaled_dot_product_attention(q, k, v, mask)
        print(out.shape)
        
        out = rearrange(out, "...  num_heads sequence_length head_dim -> ... sequence_length (num_heads head_dim)")
        out = einsum(out, o_proj_weight, "... sequence_length d_v, d_model d_v -> ... sequence_length d_model")
        return out
    
class CausalMultiHeadAttention_(torch.nn.Module):
    """
    CausalMultiHeadAttention 是因果多头注意力，它通过将输入的稠密向量与输入的稠密向量进行点积来得到输出。
    每个头的公式都是：
    out = softmax(QK^T / sqrt(d_k))V
    Args:
        d_model (int): 输入的维度，也就是d_model
        n_heads (int): 头的数量
    input:
        x: (batch_size, seq_len, d_model) 输入的稠密向量
        wq: (d_model, d_k) 查询的权重
        wk: (d_model, d_k) 键的权重
        wv: (d_model, d_v) 值的权重
        wo: (d_model, d_model) 输出的权重
    output:
        out: (batch_size, seq_len, d_model) 输出的稠密向量
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attention_ = ScaledDotProductAttention(d_model)

    def attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None):
        d_k = self.d_model
        print(Q.shape, K.shape)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k))
        save_4d_tensor(scores, "scores.txt")
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9) #如果mask为0，则将对应位置的score设置为-1e9
        attn_weights = torch.softmax(scores, dim=-1) #对key这一维度进行softmax归一化
        return torch.matmul(attn_weights, V)
    
    def forward(self, wq, wk, wv, wo, x)->torch.Tensor:
        batch_size, seq_len, d_model = x.shape

        q = x @ wq.T # (batch_size, seq_len, d_model) @ (d_model, d_k) -> (batch_size, seq_len, d_k)
        k = x @ wk.T # (batch_size, seq_len, d_model) @ (d_model, d_k) -> (batch_size, seq_len, d_k)
        v = x @ wv.T # (batch_size, seq_len, d_model) @ (d_model, d_v) -> (batch_size, seq_len, d_v)
        
        q_ = einsum(x, wq, "... sequence_length d_in, d_k d_in -> ... sequence_length d_k")
        k_ = einsum(x, wk, "... sequence_length d_in, d_k d_in -> ... sequence_length d_k")
        v_ = einsum(x, wv, "... sequence_length d_in, d_v d_in -> ... sequence_length d_v")
        print(q_.shape)

        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim) #view会优先切分最后一个维度，这和内存有关。
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        q_ = rearrange(q_, " ... sequence_length (num_heads head_dim) ->  ... num_heads sequence_length head_dim", num_heads=self.n_heads)
        k_ = rearrange(k_, " ... sequence_length (num_heads head_dim) ->  ... num_heads sequence_length head_dim", num_heads=self.n_heads)
        v_ = rearrange(v_, " ... sequence_length (num_heads head_dim) ->  ... num_heads sequence_length head_dim", num_heads=self.n_heads)
        
        print(q_.shape)

        #现在的形状是(batch_size, n_heads, seq_len, head_dim)
        # 创建mask，用于防止当前位置的token看到未来的token。
        mask = torch.triu(torch.ones(seq_len, seq_len,dtype=torch.bool,device=x.device), diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0) # (1, 1, seq_len, seq_len)

        out1 = self.attention(q, k, v, mask)
        out2 = self.attention_(q_, k_, v_, mask)
        
        out1 = out1.transpose(1, 2)
        out1 = out1.contiguous().view(batch_size, seq_len, d_model)
        out1 = out1 @ wo.T
        out2 = rearrange(out2, "...  num_heads sequence_length head_dim -> ... sequence_length (num_heads head_dim)")
        out2 = einsum(out2, wo, "... sequence_length d_v, d_model d_v -> ... sequence_length d_model")
        
        print(out1 == out2)
        return
    
if __name__ == '__main__':
    wq = torch.randn(64, 64)
    wk = torch.randn(64, 64)
    wv = torch.randn(64, 64)
    wo = torch.randn(64, 64)
    # a = CausalMultiHeadAttention(64, 4)
    b = CausalMultiHeadAttention_(64, 4)
    x = torch.randn(4, 12, 64)
    # o1 = a(wq, wk, wv, wo, x)
    o2 = b(wq, wk, wv, wo, x)