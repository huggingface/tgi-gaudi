import torch
import itertools
from text_generation_server.models.globals import (
    ATTENTION,
    BLOCK_SIZE,
)
from text_generation_server.layers.attention import Seqlen
from typing import Optional, List
from vllm_hpu_extension import cache_ops, ops
from vllm_hpu_extension.utils import (Matmul, ModuleFusedSDPA, Softmax,
                                      VLLMKVCache)
try:
    from habana_frameworks.torch.hpex.kernels import FusedSDPA
except ImportError:
    print("Not using HPU fused scaled dot-product attention kernel.")
    FusedSDPA = None

SUPPORTS_WINDOWING = False
PREFILL_IN_KV_CACHE = False
def fetch_from_cache(cache, blocks):
    return cache.index_select(0, blocks)

def attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    seqlen: Seqlen,
    block_tables: torch.Tensor,
    softmax_scale,
    window_size_left=-1,
    causal=True,
    softcap: Optional[float] = None,
):
    attn_output = FusedSDPA.apply(
        q, key_cache, value_cache, None, 0.0, causal, None
    )
    
    return attn_output
   

def reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slots: torch.Tensor,
):
    if ATTENTION in {"flashdecoding", "flashinfer"}:
        shape = key_cache.shape
        key_cache.view(-1, shape[-2], shape[-1])[slots] = key
        value_cache.view(-1, shape[-2], shape[-1])[slots] = value
    else:
        cache_ops.reshape_and_cache(
            key, value, key_cache, value_cache, slots, "auto", 1.0
        )

def paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    kv_head_mapping: torch.Tensor,
    softmax_scale: float,
    block_tables: torch.Tensor,
    seqlen: Seqlen,
    max_s: int,
    softcap: Optional[float] = None,
):
    
    batch_size, seq_len, hidden_size = query.shape
    blocks_used = [len(bt) for bt in block_tables if bt]
    block_list = []
    block_scales = []
    for i, bt in enumerate(block_tables):
        block_list.extend(bt)
        blocks_in_group = len(bt)
        if blocks_in_group > 0:
            scale = 1.0 / blocks_in_group
            block_scales.extend([scale] * blocks_in_group)

    block_mapping_nested: List[List[int]] = [
        [i] * b_u for i, b_u in enumerate(blocks_used)
    ]
    block_mapping: List[int] = list(
        itertools.chain.from_iterable(block_mapping_nested))
    block_list = torch.tensor(block_list,
                                dtype=torch.int,
                                device="hpu")
    block_mapping = torch.tensor(block_mapping,
                                    dtype=torch.long,
                                    device="hpu")
    block_scales = torch.tensor(block_scales,
                                    dtype=torch.bfloat16,
                                    device="hpu")
    output = ops.flat_pa(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_list=block_list,
        block_mapping=block_mapping,
        block_bias=None,
        block_scales=block_scales,
        block_groups=None,
        scale=softmax_scale,
        matmul_qk_op=Matmul(),
        matmul_av_op=Matmul(),
        batch2block_matmul_op=Matmul(),
        block2batch_matmul_op=Matmul(),
        keys_fetch_func=fetch_from_cache,
        values_fetch_func=fetch_from_cache)
        # Reshape the output tensor.
    return output.view(batch_size, seq_len, hidden_size)
    
    # out = torch.empty_like(query)
    # ipex.llm.modules.PagedAttention.single_query_cached_kv_attention(
    #     out,
    #     query,
    #     key_cache,
    #     value_cache,
    #     kv_head_mapping,
    #     softmax_scale,
    #     block_tables,
    #     seqlen.input_lengths,
    #     BLOCK_SIZE,
    #     max_s,
    #     None,
    # )
    # return out


__all__ = [
    "PREFILL_IN_KV_CACHE",
    "SUPPORTS_WINDOWING",
    "attention",
    "paged_attention",
    "reshape_and_cache",
]
