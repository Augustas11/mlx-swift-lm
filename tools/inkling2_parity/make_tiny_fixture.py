"""Writes the Inkling 2 Mini tiny parity fixtures under Tests/MLXLMTests/Resources/inkling2/.

Random float32 weights for an 8-layer `inkling2_loop` config whose core (layers 2-5, blocks of
two layers) runs three times: 16 layer visits, local and global layers inside and outside the
core, 4 routed experts with top-2 routing, an 8-token window and a 6-token global band, both
shorter than the 24-token prompt. The oracle is mlx-vlm's own `LanguageModel`
(Blaizzy/mlx-vlm@f1618d6d, which contains "Add Inkling 2 Mini model." 111ac668).

Three fixtures, each a directory with config.json, model.safetensors and oracle.json:

- tiny-source: the source checkpoint layout (`model.llm.*`): NVFP4 experts with a per-expert
  amax on layers 0-6, MXFP8 on layer 7, dense attention. mlx-vlm reads it through its own
  `Model.sanitize` and `hf_quant_config.json`, as it reads thinkingmachines/Inkling-2-Mini.
- tiny-oq: the same weights in the oQ layout (oMLX's runtime names, as in
  mlx-community/Inkling-2-Mini-oQ4e-mtp). Same oracle.
- tiny-oq-affine: oQ layout only, other weights: affine 8-bit attention, embedding and head,
  NVFP4 experts on layer 0 and affine 4-bit experts on layers 1-7. mlx-vlm runs the same
  quantized bytes (its separate projections hold the fused projection's rows).

Run on the Studio, from the fork root, with mlx-vlm at f1618d6d (a clone or an installed copy):

    python3 -I tools/inkling2_parity/make_tiny_fixture.py --mlx-vlm /path/to/mlx-vlm

The oracle runs on the CPU by default (`--device gpu` to change); the Swift test runs on the
CPU unless INKLING2_PARITY_DEVICE=gpu. Never run this on the M5.
"""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

# `python3 -I` leaves the script's directory off sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from oq_adapter import from_oq, load_reference  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "Tests/MLXLMTests/Resources/inkling2"

H = 128
LAYERS = 8
CORE_START, CORE_END = 2, 6
LOCAL = [0, 2, 4, 5, 6]
KERNEL = 4
RANK = 8
EXPERTS = 4
INTERMEDIATE = 64
TEXT = {
    "model_type": "inkling2_loop_text",
    "hidden_size": H,
    "num_hidden_layers": LAYERS,
    "vocab_size": 256,
    "unpadded_vocab_size": 250,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "swa_num_attention_heads": 4,
    "swa_num_key_value_heads": 2,
    "swa_head_dim": 32,
    "sliding_window_size": 8,
    "local_layer_ids": LOCAL,
    "d_rel": 8,
    "rel_extent": 6,
    "log_scaling_n_floor": 4,
    "log_scaling_alpha": 0.1,
    "rms_norm_eps": 1e-6,
    "use_value_norm_no_gamma": True,
    "qk_l2_clip_threshold": 3.0,
    "rel_l2_clip_threshold": 2.0,
    "sconv_kernel_size": KERNEL,
    "n_layers_per_attn_res_block": 2,
    "attn_res_normalize_values": True,
    "n_routed_experts": EXPERTS,
    "num_experts_per_tok": 2,
    "n_shared_experts": 0,
    "intermediate_size": INTERMEDIATE,
    "mlp_gate_softcap": 4.0,
    "mlp_up_softcap": 25.0,
    "route_scale": 2.0,
    "use_gate_bias": True,
    "final_logit_softcapping": 30.0,
    "loop_config": {
        "core_start": CORE_START,
        "core_end": CORE_END,
        "num_passes": 3,
        "gate_rank": RANK,
        "is_varloop": False,
    },
}
PROMPT_LENGTH = 24
DECODE = 8
PREFILL_STEPS = [1, 5, 23]
# ModelOpt's NVFP4 tensor-scale denominator (E4M3 max x E2M1 max). With it the oQ
# `global_scale` equals the amax, and `global_scale / 2688` equals mlx-vlm's
# `weight_scale_2 = amax / denominator` bit for bit.
NVFP4_DENOMINATOR = 6.0 * 448.0
FORMATS = {
    "nvfp4": {"group_size": 16, "bits": 4, "mode": "nvfp4"},
    "mxfp8": {"group_size": 32, "bits": 8, "mode": "mxfp8"},
    "affine4": {"group_size": 64, "bits": 4, "mode": "affine"},
    "affine8": {"group_size": 64, "bits": 8, "mode": "affine"},
}
CONTROLLERS = ("attn", "k", "v", "mlp")
SOURCE_ATTENTION = {
    "wq_du": "q_proj",
    "wk_dv": "k_proj",
    "wv_dv": "v_proj",
    "wr_du": "r_proj",
    "wo_ud": "o_proj",
    "gate_attn": "gate_proj",
    "attn_sink_proj": "sink_proj",
}


def sliding(layer):
    return layer in LOCAL


def core(layer):
    return CORE_START <= layer < CORE_END


def head_shapes(layer):
    t = TEXT
    if sliding(layer):
        return t["swa_num_attention_heads"], t["swa_num_key_value_heads"], t["swa_head_dim"]
    return t["num_attention_heads"], t["num_key_value_heads"], t["head_dim"]


# MARK: random weights in mlx-vlm's module names


def make_vlm_weights(mx, expert_formats, quantize_dense):
    """mlx-vlm `LanguageModel` parameters (names relative to it) and each NVFP4 layer's amax."""
    w, quant, amax = {}, {}, {}

    def normal(*shape, std):
        return (mx.random.normal(shape) * std).astype(mx.float32)

    def norm(size):
        return 1 + normal(size, std=0.1)

    def quantize(path, dense, fmt):
        spec = FORMATS[fmt]
        out = mx.quantize(dense, group_size=spec["group_size"], bits=spec["bits"], mode=spec["mode"])
        w[path + ".weight"], w[path + ".scales"] = out[0], out[1]
        if len(out) > 2 and out[2] is not None:
            w[path + ".biases"] = out[2]
        quant[path] = dict(spec)

    def linear(path, rows, columns, std):
        dense = normal(rows, columns, std=std)
        if quantize_dense:
            quantize(path, dense, "affine8")
        else:
            w[path + ".weight"] = dense

    vocab = TEXT["vocab_size"]
    embed = normal(vocab, H, std=1.0)
    if quantize_dense:
        quantize("model.embed_tokens", embed, "affine8")
    else:
        w["model.embed_tokens.weight"] = embed
    w["model.embed_norm.weight"] = norm(H)
    w["model.norm.weight"] = norm(H)
    w["model.final_input.query"] = normal(H, std=0.5)
    w["model.message_queries"] = normal(CORE_END - CORE_START, 4, H, std=0.5)
    linear("lm_head", vocab, H, std=0.15)

    d_rel = TEXT["d_rel"]
    for i in range(LAYERS):
        p = f"model.layers.{i}."
        a = p + "self_attn."
        heads, kv_heads, head_dim = head_shapes(i)
        q, kv = heads * head_dim, kv_heads * head_dim
        std = H**-0.5
        linear(a + "q_proj", q, H, std)
        linear(a + "k_proj", kv, H, std)
        linear(a + "v_proj", kv, H, std)
        linear(a + "r_proj", heads * d_rel, H, std)
        linear(a + "gate_proj", q, H, std)
        linear(a + "sink_proj", heads, H, std)
        linear(a + "o_proj", H, q, q**-0.5)
        w[a + "q_scale"] = 1 + normal(heads, head_dim, std=0.1)
        w[a + "rel_scale"] = 1 + normal(heads, d_rel, std=0.1)
        extent = TEXT["sliding_window_size"] if sliding(i) else TEXT["rel_extent"]
        w[a + "rel_proj"] = normal(d_rel, extent, std=0.5)
        for conv in ("k_sconv", "v_sconv"):
            if sliding(i):
                w[a + conv + ".conv.weight"] = normal(kv, KERNEL, 1, std=0.3)
            else:
                w[a + conv + ".static_weight"] = normal(KERNEL, kv, std=0.3)
                linear(a + conv + ".weight_proj", kv_heads * KERNEL, H, std)
        if sliding(i):
            w[p + "input_layernorm.weight"] = norm(H)
        else:
            for n in ("q", "k", "v"):
                w[p + f"{n}_input_layernorm.weight"] = norm(H)
        w[p + "post_attention_layernorm.weight"] = norm(H)
        w[p + "attn_sconv.conv.weight"] = normal(H, KERNEL, 1, std=0.3)
        w[p + "mlp_sconv.conv.weight"] = normal(H, KERNEL, 1, std=0.3)
        if i > 0:
            w[p + "attention_input.query"] = (
                normal(H, std=0.5) if sliding(i) else normal(3, H, std=0.5)
            )
        w[p + "mlp_input.query"] = normal(H, std=0.5)
        w[p + "mlp.gate_weight"] = normal(EXPERTS, H, std=0.2)
        w[p + "mlp.e_score_correction_bias"] = normal(EXPERTS, std=0.1)
        fmt = expert_formats[i]
        s = p + "mlp.switch_mlp."
        for name, rows, columns in (
            ("gate_proj", INTERMEDIATE, H),
            ("up_proj", INTERMEDIATE, H),
            ("down_proj", H, INTERMEDIATE),
        ):
            quantize(s + name, normal(EXPERTS, rows, columns, std=columns**-0.5), fmt)
        if fmt == "nvfp4":
            layer_amax = {}
            for which, names in (("w13", ("gate_proj", "up_proj")), ("w2", ("down_proj",))):
                value = NVFP4_DENOMINATOR * (0.75 + 0.5 * mx.random.uniform(shape=(EXPERTS,)))
                layer_amax[which] = value.astype(mx.float32)
                scale = layer_amax[which] / mx.array([NVFP4_DENOMINATOR], dtype=mx.float32)
                for name in names:
                    w[s + name + ".weight_scale_2"] = scale
            amax[i] = layer_amax
        if core(i):
            for m in CONTROLLERS:
                c = p + f"controllers.{m}."
                w[c + "down_weight"] = normal(RANK, 2 * H, std=0.1)
                w[c + "forget_weight"] = normal(H, RANK, std=0.3)
                w[c + "gain_weight"] = normal(H, RANK, std=0.3)
    return w, quant, amax


# MARK: layouts


def interleave(mx, gate, up):
    """Rows of `gate` on even and `up` on odd output rows: [E, 2I, X]."""
    e, i, x = gate.shape
    return mx.stack([gate, up], axis=2).reshape(e, 2 * i, x)


def linear_parts(w, path):
    return {s: w[f"{path}.{s}"] for s in ("weight", "scales", "biases") if f"{path}.{s}" in w}


def to_source(mx, w, amax):
    """The source checkpoint (`model.llm.*`) holding `w`; dense attention only."""
    out = {}
    llm = "model.llm."
    out[llm + "embed.weight"] = w["model.embed_tokens.weight"]
    out[llm + "embed_norm.weight"] = w["model.embed_norm.weight"]
    out[llm + "norm.weight"] = w["model.norm.weight"]
    out[llm + "unembed.weight"] = w["lm_head.weight"]
    out[llm + "attn_res.output_query"] = w["model.final_input.query"]
    mq = w["model.message_queries"]
    count = mq.shape[0]
    out[llm + "middle_cycle_cross_pass_queries"] = mx.stack([mq[:, 0], mq[:, 3]], axis=1).reshape(
        2 * count, H
    )
    out[llm + "middle_cycle_cross_pass_kv_queries"] = mq[:, 1:3].reshape(2 * count, H)
    for i in range(LAYERS):
        v = f"model.layers.{i}."
        a = v + "self_attn."
        p = llm + f"layers.{i}."
        for source, name in SOURCE_ATTENTION.items():
            out[p + f"attn.{source}.weight"] = w[a + name + ".weight"]
        out[p + "attn.q_l2_clip.scale"] = w[a + "q_scale"]
        out[p + "attn.rel_l2_clip.scale"] = w[a + "rel_scale"]
        out[p + "attn.rel_logits_proj.proj"] = w[a + "rel_proj"]
        for conv in ("k_sconv", "v_sconv"):
            if sliding(i):
                out[p + f"attn.{conv}.weight"] = w[a + conv + ".conv.weight"].transpose(0, 2, 1)
            else:
                out[p + f"attn.{conv}.static_w"] = w[a + conv + ".static_weight"]
                out[p + f"attn.{conv}.weight_proj"] = w[a + conv + ".weight_proj.weight"]
        if sliding(i):
            out[p + "attn_norm.weight"] = w[v + "input_layernorm.weight"]
        else:
            for n in ("q", "k", "v"):
                out[p + f"{n}_input_norm.weight"] = w[v + f"{n}_input_layernorm.weight"]
        out[p + "mlp_norm.weight"] = w[v + "post_attention_layernorm.weight"]
        out[p + "attn_sconv.weight"] = w[v + "attn_sconv.conv.weight"].transpose(0, 2, 1)
        out[p + "mlp_sconv.weight"] = w[v + "mlp_sconv.conv.weight"].transpose(0, 2, 1)
        out[p + "mlp.gate.weight"] = w[v + "mlp.gate_weight"]
        out[p + "mlp.gate.bias"] = w[v + "mlp.e_score_correction_bias"]
        s = v + "mlp.switch_mlp."
        gate, up, down = (linear_parts(w, s + n) for n in ("gate_proj", "up_proj", "down_proj"))
        assert "biases" not in gate, "the source layout has no affine experts"
        e = p + "mlp.experts."
        out[e + "w13_weight"] = interleave(mx, gate["weight"], up["weight"]).view(mx.uint8)
        out[e + "w13_weight_scale"] = interleave(mx, gate["scales"], up["scales"])
        out[e + "w2_weight"] = down["weight"].view(mx.uint8)
        out[e + "w2_weight_scale"] = down["scales"]
        if i in amax:
            for which in ("w13", "w2"):
                out[e + f"{which}_weight_amax"] = amax[i][which]
                out[e + f"{which}_weight_outer_scale_denominator"] = mx.array(
                    [NVFP4_DENOMINATOR], dtype=mx.float32
                )
        if i > 0:
            out[llm + f"attn_res.attn_queries.{i - 1}"] = w[v + "attention_input.query"]
        out[llm + f"attn_res.mlp_queries.{i}"] = w[v + "mlp_input.query"]
        if core(i):
            for m in CONTROLLERS:
                for kind in ("down", "forget", "gain"):
                    out[p + f"controllers.{m}.{kind}_weight"] = w[v + f"controllers.{m}.{kind}_weight"]
    return out


def to_oq(mx, w, quant, amax):
    """The oQ layout holding `w`, and its per-layer quantization map (oQ module names)."""
    out, qmap = {}, {}

    def put(path, parts, source_paths):
        for suffix, value in parts.items():
            out[f"{path}.{suffix}"] = value
        specs = [quant.get(p) for p in source_paths]
        assert all(spec == specs[0] for spec in specs), f"{path}: mixed formats {specs}"
        if specs[0] is not None:
            qmap[path] = specs[0]

    def fuse(paths):
        parts = [linear_parts(w, p) for p in paths]
        assert all(set(p) == set(parts[0]) for p in parts), paths
        return {s: mx.concatenate([p[s] for p in parts], axis=0) for s in parts[0]}

    put("model.embed", linear_parts(w, "model.embed_tokens"), ["model.embed_tokens"])
    put("lm_head", linear_parts(w, "lm_head"), ["lm_head"])
    out["model.embed_norm.weight"] = w["model.embed_norm.weight"]
    out["model.norm.weight"] = w["model.norm.weight"]
    out["model.final_input.query"] = w["model.final_input.query"].reshape(1, H)
    for i in range(LAYERS):
        v = f"model.layers.{i}."
        a = v + "self_attn."
        o = f"model.layers.{i}."
        if sliding(i):
            names = [a + n for n in ("q_proj", "k_proj", "v_proj", "r_proj", "gate_proj", "sink_proj")]
            put(o + "attn.in_proj", fuse(names), names)
            out[o + "attn.k_conv"] = w[a + "k_sconv.conv.weight"].reshape(-1, KERNEL)
            out[o + "attn.v_conv"] = w[a + "v_sconv.conv.weight"].reshape(-1, KERNEL)
            out[o + "attn_norm.weight"] = w[v + "input_layernorm.weight"]
        else:
            names = [a + n for n in ("q_proj", "r_proj", "gate_proj", "sink_proj")]
            put(o + "attn.q_proj", fuse(names), names)
            for n in ("k", "v"):
                names = [a + f"{n}_proj", a + f"{n}_sconv.weight_proj"]
                put(o + f"attn.{n}_proj", fuse(names), names)
                out[o + f"attn.{n}_static"] = w[a + f"{n}_sconv.static_weight"]
            for n in ("q", "k", "v"):
                out[o + f"{n}_input_norm.weight"] = w[v + f"{n}_input_layernorm.weight"]
        put(o + "attn.o_proj", linear_parts(w, a + "o_proj"), [a + "o_proj"])
        out[o + "attn.q_scale"] = w[a + "q_scale"]
        out[o + "attn.r_scale"] = w[a + "rel_scale"]
        out[o + "attn.rel_proj"] = w[a + "rel_proj"]
        out[o + "attn_conv"] = w[v + "attn_sconv.conv.weight"].reshape(H, KERNEL)
        out[o + "mlp_conv"] = w[v + "mlp_sconv.conv.weight"].reshape(H, KERNEL)
        out[o + "mlp_norm.weight"] = w[v + "post_attention_layernorm.weight"]
        out[o + "mlp.router_weight"] = w[v + "mlp.gate_weight"]
        out[o + "mlp.router_bias"] = w[v + "mlp.e_score_correction_bias"]
        s = v + "mlp.switch_mlp."
        gate, up, down = (linear_parts(w, s + n) for n in ("gate_proj", "up_proj", "down_proj"))
        put(
            o + "mlp.experts.w13",
            {k: interleave(mx, gate[k], up[k]) for k in gate},
            [s + "gate_proj", s + "up_proj"],
        )
        put(o + "mlp.experts.w2", down, [s + "down_proj"])
        if i in amax:
            # amax x (2688 / denominator), which is the amax itself here.
            out[o + "mlp.experts.w13.global_scale"] = amax[i]["w13"]
            out[o + "mlp.experts.w2.global_scale"] = amax[i]["w2"]
        if i > 0:
            out[o + "attn_input.query"] = w[v + "attention_input.query"].reshape(-1, H)
        out[o + "mlp_input.query"] = w[v + "mlp_input.query"].reshape(-1, H)
        if core(i):
            downs, gains = [], []
            for m in CONTROLLERS:
                c = v + f"controllers.{m}."
                downs.append(w[c + "down_weight"].reshape(RANK, 2, H).transpose(1, 2, 0))
                gains.append(
                    mx.concatenate([w[c + "gain_weight"].T, w[c + "forget_weight"].T], axis=1)
                )
            out[o + "loop_down"] = mx.stack(downs)
            out[o + "loop_gain_forget"] = mx.stack(gains)
            out[o + "loop_message.query"] = w["model.message_queries"][i - CORE_START]
    return out, qmap


# MARK: the oracle


def check_oq_round_trip(mx, oq, quantization, weights, quant, config):
    """`oq_adapter.from_oq` (which `dump_real_oracle.py --layout oq` uses on the real oQ build)
    must give back the mlx-vlm weights and quantization `to_oq` started from."""
    from mlx_vlm.models.inkling.inkling import _split_gate_up

    mapped, mapped_quant = from_oq(mx, oq, config, quantization, _split_gate_up)
    assert set(mapped) == set(weights), sorted(set(mapped) ^ set(weights))
    for key, value in weights.items():
        assert mapped[key].shape == value.shape, (key, mapped[key].shape, value.shape)
        assert mapped[key].dtype == value.dtype, (key, mapped[key].dtype, value.dtype)
        assert mx.array_equal(mapped[key], value).item(), key
    assert mapped_quant == quant, sorted(
        k for k in set(mapped_quant) | set(quant) if mapped_quant.get(k) != quant.get(k)
    )


def run_oracle(mx, model, prompt):
    full = model(mx.array([prompt])).logits[0].astype(mx.float32)
    mx.eval(full)
    cases = []
    for step in PREFILL_STEPS:
        cache = model.make_cache()
        rest = list(prompt)
        while len(rest) > 1:
            n = min(step, len(rest) - 1)
            mx.eval(model(mx.array([rest[:n]]), cache=cache).logits)
            rest = rest[n:]
        tokens, rows = [], []
        inputs = rest
        for _ in range(DECODE):
            row = model(mx.array([inputs]), cache=cache).logits[0, -1].astype(mx.float32)
            mx.eval(row)
            token = int(mx.argmax(row).item())
            tokens.append(token)
            rows.append(row.tolist())
            inputs = [token]
        cases.append({"prefill_step": step, "greedy": tokens, "logits": rows})

    # The reference's own consistency: the cached last-prompt logits match the uncached
    # forward, and no chunking changes the greedy tokens.
    for case in cases:
        difference = mx.abs(mx.array(case["logits"][0]) - full[-1]).max().item()
        assert difference < 1e-3, f"step {case['prefill_step']}: cached vs uncached {difference}"
        assert case["greedy"] == cases[0]["greedy"], "chunking changed the greedy tokens"
    return {"prompt": prompt, "full_logits": full.tolist(), "cases": cases}


def hf_quant_config(formats):
    """The source checkpoint's `hf_quant_config.json` shape (`quant_method` "inkling2")."""
    transforms = {"nvfp4": "nvfp4_nearest", "mxfp8": "mxfp8_nearest"}
    groups = []
    for fmt, transform in transforms.items():
        tensors = [
            f"model.llm.layers.{i}.mlp.experts.{which}_weight"
            for i in range(LAYERS)
            if formats[i] == fmt
            for which in ("w13", "w2")
        ]
        if tensors:
            groups.append({"weight_transform": transform, "tensors": tensors})
    groups.append({"weight_transform": "cast", "tensors": ["model.llm.unembed.weight"]})
    return {"quant_method": "inkling2", "model": {"parameter_groups": groups}}


def write(mx, directory, config, weights, oracle, sidecar=None):
    directory.mkdir(parents=True, exist_ok=True)
    if sidecar is not None:
        (directory / "hf_quant_config.json").write_text(json.dumps(sidecar, indent=1) + "\n")
    (directory / "config.json").write_text(json.dumps(config, indent=1) + "\n")
    mx.save_safetensors(str(directory / "model.safetensors"), weights, metadata={"format": "mlx"})
    (directory / "oracle.json").write_text(json.dumps(oracle) + "\n")
    print(f"wrote {directory}: {len(weights)} tensors, greedy {oracle['cases'][0]['greedy']}")


def reference_revision(path):
    if path is None:
        return None
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mlx-vlm", type=Path, help="mlx-vlm checkout at f1618d6d (else installed)")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.mlx_vlm is not None:
        sys.path.insert(0, str(args.mlx_vlm.resolve()))

    import mlx.core as mx
    import mlx.nn as nn
    from mlx_vlm.models.inkling2.config import TextConfig
    from mlx_vlm.models.inkling2.inkling2 import Model
    from mlx_vlm.models.inkling2.language import LanguageModel

    mx.set_default_device(mx.cpu if args.device == "cpu" else mx.gpu)
    reference = {
        "mlx": mx.__version__,
        "mlx_vlm": reference_revision(args.mlx_vlm),
        "device": args.device,
    }
    print("reference", reference)
    prompt = random.Random(2467).choices(range(TEXT["unpadded_vocab_size"]), k=PROMPT_LENGTH)
    top = {"model_type": "inkling2_loop", "eos_token_id": 1, "text_config": TEXT}

    # Source and oQ layouts of one set of weights: one oracle.
    mx.random.seed(2467)
    formats = ["nvfp4"] * (LAYERS - 1) + ["mxfp8"]
    weights, quant, amax = make_vlm_weights(mx, formats, quantize_dense=False)
    source = to_source(mx, weights, amax)
    # mlx-vlm maps the source layout itself; its result must be the weights we drew.
    stub = SimpleNamespace(config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=H)))
    mapped = {
        k.removeprefix("language_model."): value for k, value in Model.sanitize(stub, source).items()
    }
    assert set(mapped) == set(weights), sorted(set(mapped) ^ set(weights))
    for key, value in weights.items():
        assert mapped[key].shape == value.shape and mx.array_equal(mapped[key], value).item(), key
    # mlx-vlm reads the sidecar into the expert formats we drew.
    sidecar = hf_quant_config(formats)
    declared = Model.quantization_from_metadata(stub, sidecar)
    for path, spec in quant.items():
        if ".switch_mlp." in path:
            assert declared["language_model." + path] == spec, (path, declared.get(path), spec)
    config = TextConfig.from_dict(TEXT)
    model = load_reference(mx, nn, LanguageModel, config, mapped, quant)
    oracle = run_oracle(mx, model, prompt)
    oracle["reference"] = reference
    write(mx, args.out / "tiny-source", top, source, oracle, sidecar)
    oq, qmap = to_oq(mx, weights, quant, amax)
    quantization = {"group_size": 64, "bits": 4, "mode": "affine", **qmap}
    check_oq_round_trip(mx, oq, quantization, weights, quant, config)
    write(mx, args.out / "tiny-oq", {**top, "quantization": quantization}, oq, oracle)

    # Quantized attention, embedding and head; affine experts.
    mx.random.seed(24670)
    formats = ["nvfp4"] + ["affine4"] * (LAYERS - 1)
    weights, quant, amax = make_vlm_weights(mx, formats, quantize_dense=True)
    model = load_reference(mx, nn, LanguageModel, config, weights, quant)
    oracle = run_oracle(mx, model, prompt)
    oracle["reference"] = reference
    oq, qmap = to_oq(mx, weights, quant, amax)
    quantization = {"group_size": 64, "bits": 8, "mode": "affine", **qmap}
    check_oq_round_trip(mx, oq, quantization, weights, quant, config)
    write(mx, args.out / "tiny-oq-affine", {**top, "quantization": quantization}, oq, oracle)


if __name__ == "__main__":
    main()
