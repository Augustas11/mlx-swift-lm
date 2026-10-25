"""Tiny DeepSeek-V4 parity fixture from the mlx-vlm 0.7.10 reference.

Builds a 4-layer `deepseek_v4` (compress ratios 0, 4, 128, 4; one hash layer;
an 8-token window) with seeded random weights, quantizes it like the
V4-Flash-0731 MXFP4 artifact (routed experts mxfp4, attention and shared
experts mxfp8, gates/compressors/indexer weights unquantized), writes it in
the checkpoint layout the artifact uses, reloads it through the reference
sanitize. The oracles (`oracle.json`) come from `make_oracle.py` and a port
of DeepSeek's reference, not from mlx-vlm.

Outputs, next to this file: `config.json` and `model.safetensors`.

Run with the reference venv: python make_fixture.py
"""

import json
import os

import mlx.core as mx

mx.set_default_device(mx.cpu)

import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from mlx_vlm.models.deepseek_v4.config import ModelConfig
from mlx_vlm.models.deepseek_v4.deepseek_v4 import Model

HERE = os.path.dirname(os.path.abspath(__file__))
MXFP4 = {"group_size": 32, "bits": 4, "mode": "mxfp4"}
MXFP8 = {"group_size": 32, "bits": 8, "mode": "mxfp8"}

CONFIG = {
    "model_type": "deepseek_v4",
    "vocab_size": 128,
    "hidden_size": 64,
    "moe_intermediate_size": 32,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 1,
    "n_shared_experts": 1,
    "n_routed_experts": 8,
    "num_experts_per_tok": 2,
    "routed_scaling_factor": 1.5,
    "q_lora_rank": 32,
    "qk_rope_head_dim": 16,
    "head_dim": 64,
    "norm_topk_prob": True,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000,
    "compress_rope_theta": 160000,
    "rope_scaling": {
        "type": "yarn",
        "factor": 16,
        "original_max_position_embeddings": 65536,
        "beta_fast": 32,
        "beta_slow": 1,
    },
    "scoring_func": "sqrtsoftplus",
    "topk_method": "noaux_tc",
    "compress_ratios": [0, 4, 128, 4, 0],
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "hc_eps": 1e-6,
    "num_hash_layers": 1,
    "swiglu_limit": 10.0,
    "sliding_window": 8,
    "o_groups": 2,
    "o_lora_rank": 32,
    "index_n_heads": 2,
    "index_head_dim": 32,
    "index_topk": 4,
    "index_block": 4,
    "index_keep": 2,
    "tie_word_embeddings": False,
    "bos_token_id": 0,
    "eos_token_id": 1,
}

# (index_block, index_keep): HISA as configured, then the flat indexer.
INDEXER_MODES = {"hisa": (4, 2), "flat": (0, 0)}


def layer_quant(path):
    if ".ffn.switch_mlp." in path:
        return MXFP4
    if ".ffn.shared_experts." in path or ".attn.w" in path or ".attn.indexer.wq" in path:
        return MXFP8
    return None


def to_checkpoint(params, n_experts, o_groups):
    """Reference module paths -> the artifact's checkpoint keys."""
    out = {}
    top = {
        "model.embed_tokens.weight": "embed.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "head.weight",
        "model.hc_head.fn": "hc_head_fn",
        "model.hc_head.base": "hc_head_base",
        "model.hc_head.scale": "hc_head_scale",
    }
    for k, v in params.items():
        if k in top:
            out[top[k]] = v
            continue
        k = k.removeprefix("model.")
        for sub in ("attn", "ffn"):
            for p in ("fn", "base", "scale"):
                k = k.replace(f".{sub}_hc.{p}", f".hc_{sub}_{p}")
        k = k.replace(".ffn.gate.e_score_correction_bias", ".ffn.gate.bias")
        for new, old in (("gate_proj", "w1"), ("down_proj", "w2"), ("up_proj", "w3")):
            k = k.replace(f".shared_experts.{new}.", f".shared_experts.{old}.")
        if ".ffn.switch_mlp." in k:
            prefix, rest = k.split(".ffn.switch_mlp.")
            proj, suffix = rest.split(".")
            name = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}[proj]
            for e in range(n_experts):
                out[f"{prefix}.ffn.experts.{e}.{name}.{suffix}"] = v[e]
            continue
        if ".attn.wo_a." in k and v.ndim == 3:
            v = v.reshape(o_groups * v.shape[1], -1)
        out[k] = v
    return out


def build(indexer_mode, weights=None):
    cfg = dict(CONFIG)
    cfg["index_block"], cfg["index_keep"] = INDEXER_MODES[indexer_mode]
    model = Model(ModelConfig.from_dict(cfg))
    lm = model.language_model
    quant = {}
    for path, module in tree_flatten(lm.leaf_modules(), is_leaf=nn.Module.is_module):
        q = layer_quant(path)
        if q is not None and hasattr(module, "to_quantized"):
            quant[path] = q
    if weights is None:
        mx.random.seed(11)
        flat = tree_flatten(lm.parameters())
        fresh = []
        for k, v in flat:
            if k.endswith("tid2eid"):
                # Distinct experts per token, as in the V4-Flash checkpoint
                # (a repeated expert has no defined reference semantics).
                # The committed fixture's tid2eid was redrawn this way after
                # generation: numpy default_rng(11), argsort of uniform noise.
                rows, picks = v.shape
                noise = np.random.default_rng(11).random((rows, cfg["n_routed_experts"]))
                fresh.append((k, mx.array(np.argsort(noise, axis=1)[:, :picks].astype(np.int32))))
                continue
            noise = mx.random.normal(v.shape)
            if k.endswith("norm.weight"):
                value = 1 + 0.1 * noise
            elif k.endswith("e_score_correction_bias"):
                value = 0.05 * noise
            elif k.endswith(".scale"):
                value = 0.5 + 0.1 * noise
            else:
                value = 0.15 * noise
            keep_f32 = not lm.cast_predicate(k) or k.endswith(".ape")
            fresh.append((k, value if keep_f32 else value.astype(mx.bfloat16)))
        lm.load_weights(fresh)
        nn.quantize(
            lm,
            class_predicate=lambda p, m: quant.get(p, False),
        )
        mx.eval(lm.parameters())
        return model, quant
    nn.quantize(lm, class_predicate=lambda p, m: quant.get(p, False))
    lm.load_weights(list(lm.sanitize(weights).items()))
    mx.eval(lm.parameters())
    return model, quant


def main():
    model, quant = build("hisa")
    params = dict(tree_flatten(model.language_model.parameters()))
    checkpoint = to_checkpoint(params, CONFIG["n_routed_experts"], CONFIG["o_groups"])
    mx.save_safetensors(os.path.join(HERE, "model.safetensors"), checkpoint)

    quant_cfg = dict(MXFP4)
    for path, q in quant.items():
        layer_path = path.removeprefix("model.").replace(".switch_mlp.", ".switch_mlp.")
        for new, old in (("gate_proj", "w1"), ("down_proj", "w2"), ("up_proj", "w3")):
            layer_path = layer_path.replace(f".shared_experts.{new}", f".shared_experts.{old}")
        if ".switch_mlp." not in layer_path:
            quant_cfg[layer_path] = q
    config = dict(CONFIG, quantization=quant_cfg)
    with open(os.path.join(HERE, "config.json"), "w") as f:
        json.dump(config, f, indent=1, sort_keys=True)

    # oracle.json: python make_oracle.py --reference /path/to/deepseek_v4.py

if __name__ == "__main__":
    main()
