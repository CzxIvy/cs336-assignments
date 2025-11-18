from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from collections import defaultdict
import time
import heapq
import pickle
from einops import rearrange, einsum
from collections.abc import Callable, Iterable 
from typing import Optional
import math
import numpy as np

import sys
import pathlib
project_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from cs336_basics.pretokenization_example import *

class LinearModule(torch.nn.Module):
    def __init__(self, d_in: int, d_out: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        
        self.weight = torch.nn.Parameter(torch.empty((d_out, d_in), device=device, dtype=dtype))
        torch.nn.init.normal_(self.weight, mean=0.0, std=0.02)
        
    def forward(self, in_features: Float[Tensor, " ... d_in"]) -> torch.Tensor:
        return einsum(in_features, self.weight, "... d_in, d_out d_in -> ... d_out")
        

def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    linear = LinearModule(d_in, d_out)
    linear.weight.data = weights
    return linear(in_features)

    raise NotImplementedError


class EmbeddingModule(torch.nn.Module):
    def __init__(self, vocab_size: int, d_model: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.weight = torch.nn.Parameter(torch.empty((vocab_size, d_model), device=device, dtype=dtype))
        torch.nn.init.normal_(self.weight, mean=0.0, std=0.02)
        
    def forward(self, token_ids: Int[Tensor, " ..."]) -> torch.Tensor:
        return self.weight[token_ids]

def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """
    embedding = EmbeddingModule(vocab_size, d_model)
    embedding.weight.data = weights
    return embedding(token_ids)

    raise NotImplementedError

class SwiGLUModule(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        self.w1 = LinearModule(d_model, d_ff, device=device, dtype=dtype)
        torch.nn.init.normal_(self.w1.weight, mean=0.0, std=0.02)
        
        self.w2 = LinearModule(d_ff, d_model, device=device, dtype=dtype)
        torch.nn.init.normal_(self.w2.weight, mean=0.0, std=0.02)
        
        self.w3 = LinearModule(d_model, d_ff, device=device, dtype=dtype)
        torch.nn.init.normal_(self.w3.weight, mean=0.0, std=0.02)
        
        self.silu = SiLUModule()
        
    def forward(self, in_features: Float[Tensor, " ... d_model"]) -> torch.Tensor:
        x1 = self.silu(self.w1(in_features))
        x2 = self.w3(in_features)
        return self.w2(x1 * x2)

def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    
    swiglu = SwiGLUModule(d_model, d_ff)
    swiglu.w1.weight.data = w1_weight
    swiglu.w2.weight.data = w2_weight
    swiglu.w3.weight.data = w3_weight
    return swiglu(in_features)
    raise NotImplementedError


class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, device=None, dtype=None):
        super().__init__()
        
    def forward(self, Q: Float[Tensor, " ... queries d_k"],
                K: Float[Tensor, " ... keys d_k"],
                V: Float[Tensor, " ... values d_v"],
                mask: Bool[Tensor, " ... queries keys"] | None = None) -> torch.Tensor:
        d_k = Q.shape[-1]
        scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / (d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        attn_weights = softmax(scores, dim=-1)
        assert K.shape[-2] == V.shape[-2], "K length should equal to V length"
        output = einsum(attn_weights, V, "... queries keys, ... keys d_v -> ... queries d_v")
        return output

def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    scaled_dot_product_attention = ScaledDotProductAttention()
    return scaled_dot_product_attention(Q, K, V, mask)
    raise NotImplementedError

class CausalMultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scaled_dot_product_attention = ScaledDotProductAttention()
        
    def forward(self, q_proj_weight: Float[Tensor, " d_k d_in"],
                k_proj_weight: Float[Tensor, " d_k d_in"],
                v_proj_weight: Float[Tensor, " d_v d_in"],
                o_proj_weight: Float[Tensor, " d_model d_v"],
                in_features: Float[Tensor, " ... sequence_length d_in"],) -> Float[Tensor, " ... sequence_length d_out"]:
        q = einsum(in_features, q_proj_weight, "... sequence_length d_in, d_k d_in -> ... sequence_length d_k")
        k = einsum(in_features, k_proj_weight, "... sequence_length d_in, d_k d_in -> ... sequence_length d_k")
        v = einsum(in_features, v_proj_weight, "... sequence_length d_in, d_v d_in -> ... sequence_length d_v")
        
        q = rearrange(q, " ... sequence_length (num_heads head_dim) ->  ... num_heads sequence_length head_dim", num_heads=self.num_heads)
        k = rearrange(k, " ... sequence_length (num_heads head_dim) ->  ... num_heads sequence_length head_dim", num_heads=self.num_heads)
        v = rearrange(v, " ... sequence_length (num_heads head_dim) ->  ... num_heads sequence_length head_dim", num_heads=self.num_heads)
        
        seq_len = q.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
        mask = mask.unsqueeze(0).unsqueeze(0)
        
        out = self.scaled_dot_product_attention(q, k, v, mask)
        out = rearrange(out, "...  num_heads sequence_length head_dim -> ... sequence_length (num_heads head_dim)")
        out = einsum(out, o_proj_weight, "... sequence_length d_v, d_model d_v -> ... sequence_length d_model")
        return out

def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    
    causal_mha = CausalMultiHeadAttention(d_model, num_heads)
    out = causal_mha(q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight, in_features)
    return out
    raise NotImplementedError


class CausalMultiHeadAttentionWithRoPE(CausalMultiHeadAttention):
    def __init__(self, d_model, num_heads, max_seq_len, theta, device=None):
        super().__init__(d_model, num_heads)
        self.rope = RotaryPositionEmbedding(theta, self.head_dim, max_seq_len)
        
    def forward(self, q_proj_weight: Float[Tensor, " d_k d_in"],
                k_proj_weight: Float[Tensor, " d_k d_in"],
                v_proj_weight: Float[Tensor, " d_v d_in"],
                o_proj_weight: Float[Tensor, " d_model d_v"],
                in_features: Float[Tensor, " ... sequence_length d_in"],
                token_positions: Int[Tensor, " ... sequence_length"] | None = None,) -> Float[Tensor, " ... sequence_length d_out"]:
        q = einsum(in_features, q_proj_weight, "... sequence_length d_in, d_k d_in -> ... sequence_length d_k")
        k = einsum(in_features, k_proj_weight, "... sequence_length d_in, d_k d_in -> ... sequence_length d_k")
        v = einsum(in_features, v_proj_weight, "... sequence_length d_in, d_v d_in -> ... sequence_length d_v")
        
        q = rearrange(q, " ... sequence_length (num_heads head_dim) ->  ... num_heads sequence_length head_dim", num_heads=self.num_heads)
        k = rearrange(k, " ... sequence_length (num_heads head_dim) ->  ... num_heads sequence_length head_dim", num_heads=self.num_heads)
        v = rearrange(v, " ... sequence_length (num_heads head_dim) ->  ... num_heads sequence_length head_dim", num_heads=self.num_heads)
        
        if token_positions is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        
        seq_len = q.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
        mask = mask.unsqueeze(0).unsqueeze(0)
        
        out = self.scaled_dot_product_attention(q, k, v, mask)
        out = rearrange(out, "...  num_heads sequence_length head_dim -> ... sequence_length (num_heads head_dim)")
        out = einsum(out, o_proj_weight, "... sequence_length d_v, d_model d_v -> ... sequence_length d_model")
        return out       
        

def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    
    causal_mhar = CausalMultiHeadAttentionWithRoPE(d_model, num_heads, max_seq_len, theta)
    out = causal_mhar(q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight, in_features, token_positions)
    return out
    raise NotImplementedError

class RotaryPositionEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        
        freq = 1.0 / (self.theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        pos = torch.arange(0, self.max_seq_len, device=device)
        emd = einsum(pos, freq, "sequence_length, d_k_2 -> sequence_length d_k_2")
        
        self.register_buffer("cos_cached", torch.cos(emd))
        self.register_buffer("sin_cached", torch.sin(emd))
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_cached[token_positions].unsqueeze(0)  # Shape: (1, seq_len, d_k/2)
        sin = self.sin_cached[token_positions].unsqueeze(0)
        
        x1 = x[..., 0::2]  # Shape: (batch_size, seq_len, d_k/2)
        x2 = x[..., 1::2]
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        out = torch.stack((out1, out2), dim=-1).flatten(-2)
        return out


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    
    rope = RotaryPositionEmbedding(theta, d_k, max_seq_len, device=in_query_or_key.device)
    return rope(in_query_or_key, token_positions)
    raise NotImplementedError


class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, weights, device=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.weights = weights
        
        self.rmsnorm1 = RMSNormModule(d_model, device=device)
        self.rmsnorm1.weight.data = weights['ln1.weight']
        
        self.causal_mhar = CausalMultiHeadAttentionWithRoPE(d_model, num_heads, max_seq_len, theta, device=device)
        
        self.rmsnorm2 = RMSNormModule(d_model, device=device)
        self.rmsnorm2.weight.data = weights['ln2.weight']
        
        self.ffn = SwiGLUModule(d_model, d_ff, device=device)
        self.ffn.w1.weight.data = weights['ffn.w1.weight']
        self.ffn.w2.weight.data = weights['ffn.w2.weight']
        self.ffn.w3.weight.data = weights['ffn.w3.weight']
        
    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        seq_len = in_features.shape[-2]
        token_positions = torch.arange(seq_len)
        
        rms_norm1_out = self.rmsnorm1(in_features)
        causal_mhar_out = self.causal_mhar(self.weights['attn.q_proj.weight'],
                                           self.weights['attn.k_proj.weight'],
                                           self.weights['attn.v_proj.weight'],
                                           self.weights['attn.output_proj.weight'],
                                           rms_norm1_out, token_positions)
        add1_out = in_features + causal_mhar_out
        rms_norm2_out = self.rmsnorm2(add1_out)
        ffn_out = self.ffn(rms_norm2_out)
        add2_out = add1_out + ffn_out
        return add2_out
        
def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    
    transformer_block = TransformerBlock(d_model, num_heads, d_ff, max_seq_len, theta, weights, in_features.device)
    out = transformer_block(in_features)
    return out
    raise NotImplementedError

class TransformerLM(torch.nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, weights):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.weights = weights
        self.context_length = context_length
        
        self.embedding = EmbeddingModule(vocab_size, d_model)
        self.embedding.weight.data = weights['token_embeddings.weight']
        
        # 使用ModuleList
        self.blks = torch.nn.ModuleList()
        for i in range(num_layers):
            sub_weights = {
                'attn.q_proj.weight': weights[f'layers.{i}.attn.q_proj.weight'],
                'attn.k_proj.weight': weights[f'layers.{i}.attn.k_proj.weight'],
                'attn.v_proj.weight': weights[f'layers.{i}.attn.v_proj.weight'],
                'attn.output_proj.weight': weights[f'layers.{i}.attn.output_proj.weight'],
                'ln1.weight': weights[f'layers.{i}.ln1.weight'], 
                'ffn.w1.weight': weights[f'layers.{i}.ffn.w1.weight'],
                'ffn.w2.weight': weights[f'layers.{i}.ffn.w2.weight'],
                'ffn.w3.weight': weights[f'layers.{i}.ffn.w3.weight'],
                'ln2.weight': weights[f'layers.{i}.ln2.weight']
            }
            self.blks.append(TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, sub_weights))
        
        self.norm = RMSNormModule(d_model)
        self.norm.weight.data = weights['ln_final.weight']
        
        self.linear = LinearModule(d_model, vocab_size)
        self.linear.weight.data = weights['lm_head.weight']
        
    def forward(self, in_indices):
        embedding_out = self.embedding(in_indices)
        x = embedding_out
        for blk in self.blks:
            x = blk(x)
        norm_out = self.norm(x)
        lm_head_out = self.linear(norm_out)
        return lm_head_out
    
    @torch.no_grad
    def generate(self, x: torch.tensor, max_new_tokens: int, temperature: float=1.0, top_p: int | None=None, eos_token_id: int | None=None):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        original_sequence_length = x.shape(-1)
        for _ in range(max_new_tokens):
            x = x[:, -self.context_length:] if x.shape(-1) > self.context_length else x
            logits = self.forward(x)
            next_token_logits = logits[:, -1]
            temperature_scaled_logits = next_token_logits / temperature
            probs = softmax(temperature_scaled_logits, dim=-1)
            if top_p:
                sorted_probs, idx = torch.sort(probs, dim=-1, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                mask = cumulative_probs > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = False
                sorted_probs[mask] = 0
                
                sorted_probs.div_(sorted_probs.sum(dim=-1, keepdim=True))
                
                next_token_idx = torch.multinomial(sorted_probs, 1)
                next_token_id = torch.gather(idx, dim=-1, index=next_token_idx)
            else:
                next_token_id = torch.multinomial(probs, 1)
                
            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                break
            x = torch.cat((x, next_token_id), dim=-1)
            
        new_token_ids = x[:, original_sequence_length:]
        return new_token_ids

def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    
    transformer_lm = TransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, weights)
    out = transformer_lm(in_indices)
    return out
    raise NotImplementedError


class RMSNormModule(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        
    def forward(self, in_features: Float[Tensor, " ... d_model"]) -> torch.Tensor:
        in_dtype = in_features.dtype
        in_features = in_features.to(torch.float32)
        norm_x = in_features.norm(p=2, dim=-1, keepdim=True)
        rms_x = norm_x / (self.d_model ** 0.5)
        return ((self.weight * in_features) / (rms_x + self.eps)).to(in_dtype)
        

def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    
    rmsnorm = RMSNormModule(d_model, eps)
    rmsnorm.weight.data = weights
    return rmsnorm(in_features)
    raise NotImplementedError


class SiLUModule(torch.nn.Module):
    def __init__(self, device=None, dtype=None):
        super().__init__()
        
    def forward(self, in_features: Float[Tensor, " ..."]) -> torch.Tensor:
        return in_features * torch.sigmoid(in_features)

def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    
    silu = SiLUModule()
    return silu(in_features)
    raise NotImplementedError

class DataLoader:
    def __init__(self, dataset: npt.NDArray, batch_size: int, context_length: int, device: str):
        self.dataset = dataset
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device
        
    def get_batch_data(self):
        start_idxs = np.random.randint(
            low=0,
            high=len(self.dataset)-self.context_length,
            size=(self.batch_size,)
        )
        inputs = np.stack([self.dataset[start_idx:start_idx+self.context_length] for start_idx in start_idxs])
        labels = np.stack([self.dataset[start_idx+1:start_idx+self.context_length+1] for start_idx in start_idxs])
        return (torch.from_numpy(inputs).to(self.device), torch.from_numpy(labels).to(self.device))
            
def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    
    data_loader = DataLoader(dataset, batch_size, context_length, device)
    out = data_loader.get_batch_data()
    return out
    raise NotImplementedError

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    exp_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True).values)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)

def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    
    return softmax(in_features, dim=dim)
    raise NotImplementedError

def cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    inputs = inputs - torch.max(inputs, dim=-1, keepdim=True).values
    log_exp_sum = torch.log(torch.sum(torch.exp(inputs), dim=-1))
    log_p = inputs.gather(1, targets.unsqueeze(-1)).squeeze(-1) - log_exp_sum
    l = torch.mean(-log_p)
    return l

def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    
    return cross_entropy(inputs, targets)
    raise NotImplementedError

class GradientClip:
    def __init__(self, parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps=1e-6):
        self.parameters = parameters
        self.max_l2_norm = max_l2_norm
        self.eps = eps
    
    def __call__(self):
        grads = [p.grad for p in self.parameters if p.grad is not None]
        all_grads = torch.cat([grad.flatten() for grad in grads])
        l2_norm = torch.norm(all_grads, 2)
        if l2_norm >= self.max_l2_norm:
            clip_coeff = self.max_l2_norm/(l2_norm+self.eps)
            for grad in grads:
                grad *= clip_coeff
                # grad.mul_(clip_coeff)

def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    gradient_clip = GradientClip(parameters, max_l2_norm)
    gradient_clip()

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if betas[0] < 0 or betas[0] > 1:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if betas[1] < 0 or betas[1] > 1:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if eps < 0:
            raise ValueError(f"Invalid eps: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure() 
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state['t'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                    
                grad = p.grad.data
                t = state['t'] + 1
                m = betas[0]*state['m'] + (1-betas[0])*grad
                v = betas[1]*state['v'] + (1-betas[1])*(grad**2)
                beta0_t = (1-betas[1]**t)**0.5
                beta1_t = (1-betas[0]**t)
                lr_t = lr*beta0_t/beta1_t
                
                p.data -= lr_t*m/(v**0.5+eps)
                p.data *= (1-lr*weight_decay)
                
                state['t'] = t
                state['m'] = m
                state['v'] = v
        return loss

def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """

    return AdamW
    raise NotImplementedError

class LrCosineSchedule:
    def __init__(self, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        self.warmup_iters = warmup_iters
        self.cosine_cycle_iters = cosine_cycle_iters
    
    def __call__(self, it):
        if it < self.warmup_iters:
            lr_t = self.max_learning_rate * it / self.warmup_iters
        elif self.warmup_iters <= it <= self.cosine_cycle_iters:
            inter_data = math.cos(math.pi*(it-self.warmup_iters)/(self.cosine_cycle_iters-self.warmup_iters))
            lr_t = self.min_learning_rate + 0.5*(1 + inter_data)*(self.max_learning_rate-self.min_learning_rate)
        else:
            lr_t = self.min_learning_rate
        return lr_t

def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    
    lr_cosine_schedule = LrCosineSchedule(max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)
    out = lr_cosine_schedule(it)
    return out
    raise NotImplementedError


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    
    checkpoint = {
        'model_checkpoint': model.state_dict(),
        'optimizer_checkpoint': optimizer.state_dict(),
        'iteration': iteration
    }
    if isinstance(out, (str, os.PathLike)):
        with open(out, 'wb') as f:
            torch.save(checkpoint, f)
    else:
        torch.save(checkpoint, out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    
    if isinstance(src, (str, os.PathLike)):
        with open(src, 'rb') as f:
            checkpoint = torch.load(f)
    else:
        checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_checkpoint'])
    optimizer.load_state_dict(checkpoint['optimizer_checkpoint'])
    iteration = checkpoint['iteration']
    return iteration
    raise NotImplementedError


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []
        
        for token in self.special_tokens:
            if token.encode('utf-8') not in self.vocab.values():
                self.vocab[len(self.vocab)] = token.encode('utf-8')
        
        self.byte_to_id = {v: k for k, v in vocab.items()}
        
        # Create a merge dictionary for quick lookup
        self.merge_dict = {}
        for i, (token1, token2) in enumerate(merges):
            self.merge_dict[(token1, token2)] = (token1 + token2, i)
        
    def from_files(cls, vocab_path: str, merges_path: str, special_tokens: list[str] | None = None) -> Tokenizer:
        vocab = pickle.load(vocab_path)
        merges = pickle.load(merges_path)
        return Tokenizer(vocab, merges, special_tokens)
        
    def _split_into_bytes(self, text: str) -> list[bytes]:
        bytes_tuple_list = []
        for sub_text in regex.finditer(PAT, text):
            bytes_tuple_list.append(to_bytes_tuple(sub_text.group(0)))
        return bytes_tuple_list
        
    def _pre_tokenize(self, text: str) -> list[bytes | tuple[bytes, ...]]:
        pre_tokens = []
        if not self.special_tokens:
            return self._split_into_bytes(text)
        
        # 拆分文本，识别特殊token
        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        special_tokens_pattern = '|'.join(map(regex.escape, sorted_special_tokens))
        # 使用捕获组，这样分隔符也会包含在结果中
        pattern_with_capture = f'({special_tokens_pattern})'
        for sub_text in regex.split(pattern_with_capture, text):
            if sub_text in self.special_tokens:
                pre_tokens.append(sub_text.encode('utf-8'))
            else:
                pre_tokens.extend(self._split_into_bytes(sub_text))
        return pre_tokens
        
    def encode(self, text: str) -> list[int]:
        token_ids = []
        pre_tokens = self._pre_tokenize(text)
        for pre_token in pre_tokens:
            if isinstance(pre_token, bytes):
                # Special token
                token_ids.append(self.byte_to_id[pre_token])
                continue
            
            # BPE encoding for byte sequences
            byte_seq = list(pre_token)
            all_matched = False
            while not all_matched:
                matches = []
                for i in range(len(byte_seq) - 1):
                    pair = (byte_seq[i], byte_seq[i + 1])
                    if pair in self.merge_dict:
                        merged_token, merge_index = self.merge_dict[pair]
                        matches.append((i, merged_token, merge_index))
                if not matches:
                    all_matched = True
                    continue
                # Choose the match with the smallest merge index
                best_match = min(matches, key=lambda x: x[2])
                i, merged_token, _ = best_match
                byte_seq = byte_seq[:i] + [merged_token] + byte_seq[i + 2:]
            for token in byte_seq:  
                token_ids.append(self.byte_to_id[token])
        return token_ids
        raise NotImplementedError
    
    def encode_iterable(self, texts: Iterable[str]) -> Iterable[list[int]]:
        for text in texts:
            for token_id in self.encode(text):
                yield token_id
    
    def decode(self, token_ids: list[int]) -> str:
        bytes_list = [self.vocab[token_id] for token_id in token_ids]
        return b''.join(bytes_list).decode('utf-8', errors='replace')
        raise NotImplementedError

def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return Tokenizer(vocab, merges, special_tokens)
    raise NotImplementedError

class BPEPairElement:
    def __init__(self, pair: tuple[bytes, bytes], count:int):
        self.pair = pair
        self.count = count
        self.valid = True  # for lazy deletion in priority queue
        
    def __lt__(self, other: BPEPairElement) -> bool:
        if self.count != other.count:
            return self.count > other.count  # max-heap based on count
        return self.pair > other.pair  # tie-breaker based on lex order

class BPEPriorityQueue:
    def __init__(self):
        self.heap = []
        self.entry_finder = {}  # mapping of tasks to entries
        
    def push(self, pair: tuple[bytes, bytes], count: int):
        if count <= 0:
            raise ValueError("Count must be positive")
        
        if pair in self.entry_finder:
            self.entry_finder[pair].valid = False
            
        element = BPEPairElement(pair, count)
        heapq.heappush(self.heap, element)
        self.entry_finder[pair] = element
        
    def pop(self) -> BPEPairElement:
        while self.heap:
            element = heapq.heappop(self.heap)
            if element.valid:
                del self.entry_finder[element.pair]
                return element
        raise KeyError("pop from an empty priority queue")
        
    def is_empty(self) -> bool:
        return len(self.heap) == 0

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    pre_token_start_time = time.time()
    pre_token_counts  = parallel_process_file(input_path, special_tokens) # pre_token_counts is a dict[Tuple[bytes], int]
    pre_token_end_time = time.time()
    print(f"Pre-tokenization time: {pre_token_end_time - pre_token_start_time} seconds")
   
    vocab = {i: bytes([i]) for i in range(256)}
    
    # handle special tokens
    current_index = 256
    for token in special_tokens:
        if token.encode('utf-8') not in vocab.values():
            vocab[current_index] = token.encode('utf-8')
            current_index += 1
            
    merges = []
    
    bpe_start_time = time.time()
    
    # Count frequency of all pairs
    pairs = defaultdict(int)
    for pre_token, count in pre_token_counts.items():
        for i in range(len(pre_token) - 1):
            pair = (pre_token[i], pre_token[i + 1])
            pairs[pair] += count
    pairs_queue = BPEPriorityQueue()
    for pair, count in pairs.items():
        pairs_queue.push(pair, count)

    while len(vocab) < vocab_size:
        if not pairs:
            break  # No pairs to merge
        
        # Find the most frequent pair
        if pairs_queue.is_empty():
            break  # No pairs to merge
        element = pairs_queue.pop()
        best_pair = element.pair
        max_count = element.count
        if max_count < 1:
            break  # No more pairs to merge
        
        # best_pair, max_count = heapq.nlargest(
        #     1, pairs.items(), key=lambda item: (item[1], item[0])
        # )[0]
        # if max_count < 1:
        #     break  # No more pairs to merge
        # best_pair = max(pairs, key = lambda k: (pairs[k], k)) # slower method（实际差不多）

        merges.append(best_pair)
        vocab[current_index] = (best_pair[0] + best_pair[1])
        current_index += 1
        del pairs[best_pair]
        
        # Update pre_token_counts
        pre_token_changes = []
        pair_changes = set()
        for pre_token, count in pre_token_counts.items():
            merge_indices = [i for i in range(len(pre_token) - 1) if (pre_token[i], pre_token[i + 1]) == best_pair]
            if not merge_indices:
                continue
            else:
                new_token = []
                i = 0
                while i < len(pre_token):
                    if i < len(pre_token) - 1 and (pre_token[i], pre_token[i + 1]) == best_pair:
                        new_token.append(best_pair[0] + best_pair[1])
                        # Update pairs counts
                        if i > 0:
                            left_pair = (pre_token[i - 1], best_pair[0] + best_pair[1])
                            pairs[left_pair] = pairs.get(left_pair, 0) + count
                            pair_changes.add(left_pair)
                            old_left_pair = (pre_token[i - 1], pre_token[i])
                            pairs[old_left_pair] -= count
                            pair_changes.add(old_left_pair)
                            if pairs[old_left_pair] <= 0:
                                del pairs[old_left_pair]
                        if i < len(pre_token) - 2:
                            right_pair = (best_pair[0] + best_pair[1], pre_token[i + 2])
                            pairs[right_pair] = pairs.get(right_pair, 0) + count
                            pair_changes.add(right_pair)
                            old_right_pair = (pre_token[i + 1], pre_token[i + 2])
                            pairs[old_right_pair] -= count
                            pair_changes.add(old_right_pair)
                            if pairs[old_right_pair] <= 0:
                                del pairs[old_right_pair]
                        i += 2
                        
                    else:
                        new_token.append(pre_token[i])
                        i += 1
                pre_token_changes.append((pre_token, tuple(new_token), count))
        
        for old_token, new_token, count in pre_token_changes:
            pre_token_counts[new_token] = pre_token_counts.get(new_token, 0) + count
            del pre_token_counts[old_token]
            
        for pair in pair_changes:
            if pair in pairs:
                pairs_queue.push(pair, pairs[pair])
            else:
                # Remove from queue if count is zero
                if pair in pairs_queue.entry_finder:
                    pairs_queue.entry_finder[pair].valid = False
                        
    bpe_end_time = time.time()
    print(f"BPE training time: {bpe_end_time - bpe_start_time} seconds")
    
    return vocab, merges
    raise NotImplementedError

