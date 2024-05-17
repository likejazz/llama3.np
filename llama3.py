from __future__ import annotations

import math
import sys
import time
from typing import TypeVar, Generic, Optional

import numpy as np

from config import ModelArgs
from tokenizer import Tokenizer
from utils import load_parameters

Shape = TypeVar("Shape")


class Array(np.ndarray, Generic[Shape]): ...


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def silu(x):
    return x * (1 / (1 + np.exp(-x)))


def apply_rotary_emb(xq: Array["B, L or 1, QHN,  HD"], xk: Array["B, L or 1, KVHN, HD"],
                     freqs_cos: Array["L or 1, HD//2"], freqs_sin: Array["L or 1, HD//2"]):
    xqri: Array["B, L or 1, QHN,  HD//2, 2"] = xq.reshape(xq.shape[:-1] + (-1, 2))
    xkri: Array["B, L or 1, KVHN, HD//2, 2"] = xk.reshape(xk.shape[:-1] + (-1, 2))

    # Reshape `xq` and `xk` to match the complex representation.
    xq_r, xq_i = np.split(xqri, 2, axis=-1)
    xq_r: Array["B, L or 1, QHN,  HD//2"] = xq_r.squeeze(-1)
    xq_i: Array["B, L or 1, QHN,  HD//2"] = xq_i.squeeze(-1)

    xk_r, xk_i = np.split(xkri, 2, axis=-1)
    xk_r: Array["B, L or 1, KVHN, HD//2"] = xk_r.squeeze(-1)
    xk_i: Array["B, L or 1, KVHN, HD//2"] = xk_i.squeeze(-1)

    # Reshape `freqs_cos` and `freqs_sin` for broadcasting.
    freqs_cos: Array["B, L or 1, 1, HD//2"] = np.expand_dims(freqs_cos, axis=(0, 2))
    freqs_sin: Array["B, L or 1, 1, HD//2"] = np.expand_dims(freqs_sin, axis=(0, 2))

    # Apply rotation using real numbers.
    xq_out_r: Array["B, L or 1, QHN,  HD//2"] = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i: Array["B, L or 1, QHN,  HD//2"] = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r: Array["B, L or 1, KVHN, HD//2"] = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i: Array["B, L or 1, KVHN, HD//2"] = xk_r * freqs_sin + xk_i * freqs_cos

    # Flatten last two dimensions.
    xq_out: Array["B, L or 1, QHN,  HD//2, 2"] = np.stack([xq_out_r, xq_out_i], axis=-1)
    xk_out: Array["B, L or 1, KVHN, HD//2, 2"] = np.stack([xk_out_r, xk_out_i], axis=-1)
    xq_out: Array["B, L or 1, QHN,  HD"] = xq_out.reshape(xq_out.shape[:-2] + (-1,))
    xk_out: Array["B, L or 1, KVHN, HD"] = xk_out.reshape(xk_out.shape[:-2] + (-1,))

    return xq_out, xk_out


def repeat_kv(x: Array["B, L, KVHN, HD"], n_rep: int):
    if n_rep == 1:
        return x
    z: Array["B, L, QHN, HD"] = np.repeat(x, n_rep, axis=2)
    return z


class FeedForward:
    def __init__(self, up_weight: Array["FD, D"], gate_weight: Array["FD, D"], down_weight: Array["D, FD"]):
        self.up_weight = up_weight.T
        self.gate_weight = gate_weight.T
        self.down_weight = down_weight.T

    def __call__(self, x: Array["B, L or 1, D"]):
        # FD = 2 * 4 * D / 3
        swish: Array["B, L or 1, FD"] = silu(x @ self.gate_weight)
        x_V: Array["B, L or 1, FD"] = x @ self.up_weight
        x: Array["B, L or 1, FD"] = swish * x_V
        x: Array["B, L or 1, D"] = x @ self.down_weight
        return x


class RMSNorm:
    def __init__(self, weight: Array["H"], eps: float):
        self.weight = weight
        self.eps = eps

    def __call__(self, x: Array["B, L or 1, D"]):
        z: Array["B, L or 1, 1"] = (x ** 2).mean(-1, keepdims=True) + self.eps
        z: Array["B, L or 1, D"] = x / np.sqrt(z)
        return z * self.weight


class Attention:
    def __init__(self, q_weight: Array["D, D"], k_weight: Array["D, D"], v_weight: Array["D, D"],
                 o_weight: Array["D, D"], args: ModelArgs):
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.q_weight = q_weight.T
        self.k_weight = k_weight.T
        self.v_weight = v_weight.T
        self.o_weight = o_weight.T

        self.cache_k = np.zeros((args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, self.head_dim))
        self.cache_v = np.zeros((args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, self.head_dim))

    def __call__(self, x: Array["B, L or 1, D"], start_pos: int, mask: Optional[Array["L, L"]],
                 freqs_cos: Array["L or 1, HD//2"], freqs_sin: Array["L or 1, HD//2"]):
        B, L, _ = x.shape

        # QKV
        xq: Array["B, L or 1, D"] = x @ self.q_weight
        xk: Array["B, L or 1, D"] = x @ self.k_weight
        xv: Array["B, L or 1, D"] = x @ self.v_weight

        xq: Array["B, L or 1, QHN,  HD"] = xq.reshape(B, L, self.n_local_heads, self.head_dim)
        xk: Array["B, L or 1, KVHN, HD"] = xk.reshape(B, L, self.n_local_kv_heads, self.head_dim)
        xv: Array["B, L or 1, KVHN, HD"] = xv.reshape(B, L, self.n_local_kv_heads, self.head_dim)

        # RoPE #2
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # KV Cache
        self.cache_k[:B, start_pos: start_pos + L] = xk
        self.cache_v[:B, start_pos: start_pos + L] = xv
        ks: Array["B, L, KVHN, HD"] = self.cache_k[:B, : start_pos + L]
        vs: Array["B, L, KVHN, HD"] = self.cache_v[:B, : start_pos + L]

        # GQA
        xk: Array["B, L, HN, HD"] = repeat_kv(ks, self.n_rep)
        xv: Array["B, L, HN, HD"] = repeat_kv(vs, self.n_rep)

        # ["B, L, HN, HD"] -> ["B, HN, L, HD"]
        xq: Array["B, HN, L or 1, HD"] = xq.transpose(0, 2, 1, 3)
        xk: Array["B, HN, L, HD"] = xk.transpose(0, 2, 1, 3)
        xv: Array["B, HN, L, HD"] = xv.transpose(0, 2, 1, 3)

        # Scaled Dot-Product Attention
        # ["B, HN, L or 1, HD"] @ ["B, HN, HD, L"] -> ["B, HN, L or 1, L"]
        attention: Array["B, HN, L or 1, L"] = xq @ xk.transpose(0, 1, 3, 2) / math.sqrt(self.head_dim)
        # `mask` is used only once at the beginning.
        if mask is not None:
            attention = attention + mask[None, None, :, :]
        attention = softmax(attention)
        output: Array["B, HN, L or 1, HD"] = attention @ xv

        # ["B, HN, L or 1, HD"] -> ["B, L or 1, D"]
        output: Array["B, L or 1, D"] = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        output: Array["B, L or 1, D"] = output @ self.o_weight

        return output


class TransformerBlock:
    def __init__(self, weight: dict, layer_id: int, args: ModelArgs):
        self.attention = Attention(
            weight.get(f"model.layers.{layer_id}.self_attn.q_proj.weight"),
            weight.get(f"model.layers.{layer_id}.self_attn.k_proj.weight"),
            weight.get(f"model.layers.{layer_id}.self_attn.v_proj.weight"),
            weight.get(f"model.layers.{layer_id}.self_attn.o_proj.weight"),
            args
        )
        self.feed_forward = FeedForward(
            weight.get(f"model.layers.{layer_id}.mlp.up_proj.weight"),
            weight.get(f"model.layers.{layer_id}.mlp.gate_proj.weight"),
            weight.get(f"model.layers.{layer_id}.mlp.down_proj.weight"),
        )
        self.input_layernorm = RMSNorm(
            weight.get(f"model.layers.{layer_id}.input_layernorm.weight"),
            eps=args.norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            weight.get(f"model.layers.{layer_id}.post_attention_layernorm.weight"),
            eps=args.norm_eps
        )

    def __call__(self, x: Array["B, L or 1, D"], start_pos: int, mask: Array["L, L"],
                 freqs_cos: Array["L or 1, HD//2"], freqs_sin: Array["L or 1, HD//2"]):
        # RMSNorm
        norm_x: Array["B, L or 1, D"] = self.input_layernorm(x)
        # Masked Multi-Head Attention
        h1: Array["B, L or 1, D"] = self.attention(norm_x, start_pos, mask, freqs_cos, freqs_sin)
        z = x + h1

        # RMSNorm
        norm_z = self.post_attention_layernorm(z)
        # Feed Forward + SwiGLU
        h2: Array["B, L or 1, D"] = self.feed_forward(norm_z)
        out = z + h2

        return out


class Llama:
    def __init__(self, model_path: str, args: ModelArgs):
        self.args = args

        weight = load_parameters(model_path)
        self.tok_embedding: Array["VS, D"] = weight.get("model.embed_tokens.weight")

        # RoPE #1
        base = 10000
        head_dim = args.dim // args.n_heads
        inv_freq: Array["HD//2"] = 1.0 / (base ** (np.arange(0, head_dim, 2)[: (head_dim // 2)] / head_dim))
        t: Array["M"] = np.arange(args.max_seq_len)
        freqs: Array["M, HD//2"] = np.outer(t, inv_freq)
        self.freqs_cos: Array["M, HD//2"] = np.cos(freqs)
        self.freqs_sin: Array["M, HD//2"] = np.sin(freqs)

        self.layers = []
        for layer_id in range(args.n_layers):
            self.layers.append(TransformerBlock(weight, layer_id, args))

        self.norm = RMSNorm(weight.get("model.norm.weight"), eps=args.norm_eps)
        self.lm_head_weight: Array["D, VS"] = weight.get("lm_head.weight").T

        del weight

    def __call__(self, input_ids: Array["B, L"], start_pos: int):
        _, L = input_ids.shape
        h: Array["B, L or 1, D"] = self.tok_embedding[input_ids]
        # ["M, HD//2"] -> ["L or 1, HD//2"]
        freqs_cos: Array["L or 1, HD//2"] = self.freqs_cos[start_pos: start_pos + L]
        freqs_sin: Array["L or 1, HD//2"] = self.freqs_sin[start_pos: start_pos + L]

        # `mask` is generated only once at the beginning.
        mask: Array["L, L"] = None
        if L > 1:
            mask = np.full((L, L), float("-inf"))
            mask = np.triu(mask, k=1)
            mask = np.concatenate([np.zeros((L, start_pos)), mask], axis=1)

        # Transformer Layers
        for i, layer in enumerate(self.layers):
            h: Array["B, L or 1, D"] = layer(h, start_pos, mask, freqs_cos, freqs_sin)

        # RMSNorm
        h: Array["B, L or 1, D"] = self.norm(h)
        # Only forward the output from the last position.
        # ["B, 1, VS"] = ["B, 1(L), D"] @ ["D, VS"]
        logit: Array["B, 1, VS"] = h[:, [-1], :] @ self.lm_head_weight
        return logit

    def generate(self, input_ids: Array["B, L"], max_new_tokens: int):
        _, L = input_ids.shape
        for i, curr_pos in enumerate(range(L, max_new_tokens)):
            if i == 0:  # Prefill Phase
                inputs = input_ids
                pos = 0
            else:  # Decode Phase
                inputs = next_id
                pos = curr_pos
            logits: Array["B, 1, VS"] = self(inputs, pos)
            next_id = logits[:, -1, :].argmax(-1, keepdims=True)
            yield next_id


if __name__ == '__main__':
    args = ModelArgs()

    tokenizer = Tokenizer("./tokenizer.model.np")
    model = Llama("./stories15M.model.npz", args)

    if len(sys.argv) == 1:
        prompt = "I have a dream"
    else:
        prompt = sys.argv[1]

    print(f"\n{prompt}", end="")
    input_ids = np.array([tokenizer.encode(prompt)])
    start = time.time()
    _, L = input_ids.shape
    for id in model.generate(input_ids, args.max_new_tokens):
        L += 1
        output_id = id[0].tolist()
        if output_id[-1] in [tokenizer.eos_id, tokenizer.bos_id]:
            break
        print(tokenizer.decode(output_id), end="")
        sys.stdout.flush()
    elapsed = time.time() - start
    print(f"\n\nToken count: {L}, elapsed: {elapsed:.2f}s, {round(L / elapsed)} tokens/s")
