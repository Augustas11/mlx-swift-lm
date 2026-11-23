"""mlx-vlm (Blaizzy/mlx-vlm@f1618d6d) on the oQ layout of Inkling 2 Mini.

mlx-vlm's `Model.sanitize` maps only the source checkpoint (`model.llm.*`). The oQ layout
(oMLX's runtime names, as in mlx-community/Inkling-2-Mini-oQ4e-mtp) holds the same model in
other tensors: fused attention projections, interleaved `w13` experts, the NVFP4 tensor scale as
`global_scale` (amax rescaled to 2688), stacked controller and message tensors. `from_oq` is the
exact inverse of `make_tiny_fixture.to_oq`: it renames, splits fused rows, de-interleaves `w13`,
reshapes and transposes, and turns `global_scale` into mlx-vlm's `weight_scale_2` with the
expression the Swift port uses (`global_scale` in float32 / 2688). No value is requantized:
quantization groups run along the input dimension, so a row slice of an affine-quantized fused
projection is the separate projection's quantized rows.

Used by `make_tiny_fixture.py` (which checks the round trip `from_oq(to_oq(w)) == w` on both oQ
fixtures) and by `dump_real_oracle.py --layout oq`. Never import this on the M5.
"""

NVFP4_DENOMINATOR = 6.0 * 448.0
CONTROLLERS = ("attn", "k", "v", "mlp")
DROPPED_PREFIXES = ("vision_tower.", "model.visual.", "model.mtp.", "mtp.")


def is_text_key(key):
    return not key.startswith(DROPPED_PREFIXES)


def load_reference(mx, nn, language_model_cls, text_config, weights, quant):
    """mlx-vlm's `LanguageModel` holding `weights` (its own module names), quantized per `quant`
    (module path -> {group_size, bits, mode}) as `load_model` quantizes a checkpoint."""
    from mlx_vlm.quantization.nvfp4 import replace_scaled_quantized_linears

    model = language_model_cls(text_config)

    def predicate(path, module):
        return dict(quant[path]) if path in quant else False

    nn.quantize(model, group_size=64, bits=4, class_predicate=predicate)
    replace_scaled_quantized_linears(model, weights)
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model


def _parts(weights, path):
    return {s: weights[f"{path}.{s}"] for s in ("weight", "scales", "biases") if f"{path}.{s}" in weights}


def _split_rows(parts, widths):
    """Row slices of a fused projection's tensors, one dict per output width."""
    assert sum(widths) == parts["weight"].shape[0], (widths, parts["weight"].shape)
    out, start = [], 0
    for width in widths:
        out.append({s: v[start : start + width] for s, v in parts.items()})
        start += width
    return out


def from_oq(mx, oq, config, quantization, split_gate_up):
    """mlx-vlm `LanguageModel` weights and per-module quantization for oQ-layout text tensors.

    - `oq`: oQ tensors (vision and draft tensors already dropped);
    - `config`: mlx-vlm's `TextConfig`;
    - `quantization`: the oQ `config.json` `quantization` map (oQ module paths plus the default
      `group_size` / `bits` / `mode`), or None for an unquantized checkpoint;
    - `split_gate_up`: mlx-vlm's `_split_gate_up` (contiguous gate and up halves).
    """
    quantization = quantization or {}
    default = {k: quantization[k] for k in ("group_size", "bits", "mode") if k in quantization}
    w, quant = {}, {}
    used = set()
    h = config.hidden_size
    kernel = config.sconv_kernel_size
    loop = config.loop_config

    def take(key):
        used.add(key)
        return oq[key]

    def spec(path):
        if f"{path}.scales" not in oq:
            return None
        value = quantization.get(path, default)
        assert isinstance(value, dict) and value, f"{path}: quantized tensors but no format"
        return {k: value[k] for k in ("group_size", "bits", "mode")}

    def put(path, parts, source):
        """`parts` of oQ module `source` under mlx-vlm module `path`."""
        for suffix, value in parts.items():
            w[f"{path}.{suffix}"] = value
        fmt = spec(source)
        if fmt is not None:
            quant[path] = fmt

    def module(source):
        parts = _parts(oq, source)
        used.update(f"{source}.{s}" for s in parts)
        return parts

    put("model.embed_tokens", module("model.embed"), "model.embed")
    put("lm_head", module("lm_head"), "lm_head")
    w["model.embed_norm.weight"] = take("model.embed_norm.weight")
    w["model.norm.weight"] = take("model.norm.weight")
    w["model.final_input.query"] = take("model.final_input.query").reshape(h)
    messages = []
    for i in range(config.num_hidden_layers):
        o = f"model.layers.{i}."
        v = f"model.layers.{i}."
        a = v + "self_attn."
        sliding = config.layer_is_sliding(i)
        heads = config.swa_num_attention_heads if sliding else config.num_attention_heads
        kv_heads = config.swa_num_key_value_heads if sliding else config.num_key_value_heads
        head_dim = config.swa_head_dim if sliding else config.head_dim
        q, kv, r = heads * head_dim, kv_heads * head_dim, heads * config.d_rel
        if sliding:
            names = ("q_proj", "k_proj", "v_proj", "r_proj", "gate_proj", "sink_proj")
            for name, part in zip(names, _split_rows(module(o + "attn.in_proj"), (q, kv, kv, r, q, heads))):
                put(a + name, part, o + "attn.in_proj")
            for n in ("k", "v"):
                w[a + f"{n}_sconv.conv.weight"] = take(o + f"attn.{n}_conv").reshape(kv, kernel, 1)
            w[v + "input_layernorm.weight"] = take(o + "attn_norm.weight")
        else:
            names = ("q_proj", "r_proj", "gate_proj", "sink_proj")
            for name, part in zip(names, _split_rows(module(o + "attn.q_proj"), (q, r, q, heads))):
                put(a + name, part, o + "attn.q_proj")
            for n in ("k", "v"):
                source = o + f"attn.{n}_proj"
                main, taps = _split_rows(module(source), (kv, kv_heads * kernel))
                put(a + f"{n}_proj", main, source)
                put(a + f"{n}_sconv.weight_proj", taps, source)
                w[a + f"{n}_sconv.static_weight"] = take(o + f"attn.{n}_static")
            for n in ("q", "k", "v"):
                w[v + f"{n}_input_layernorm.weight"] = take(o + f"{n}_input_norm.weight")
        put(a + "o_proj", module(o + "attn.o_proj"), o + "attn.o_proj")
        w[a + "q_scale"] = take(o + "attn.q_scale")
        w[a + "rel_scale"] = take(o + "attn.r_scale")
        w[a + "rel_proj"] = take(o + "attn.rel_proj")
        w[v + "attn_sconv.conv.weight"] = take(o + "attn_conv").reshape(h, kernel, 1)
        w[v + "mlp_sconv.conv.weight"] = take(o + "mlp_conv").reshape(h, kernel, 1)
        w[v + "post_attention_layernorm.weight"] = take(o + "mlp_norm.weight")
        w[v + "mlp.gate_weight"] = take(o + "mlp.router_weight")
        w[v + "mlp.e_score_correction_bias"] = take(o + "mlp.router_bias")
        s = v + "mlp.switch_mlp."
        w13 = o + "mlp.experts.w13"
        gate, up = {}, {}
        for suffix, value in module(w13).items():
            gate[suffix], up[suffix] = split_gate_up(value)
        put(s + "gate_proj", gate, w13)
        put(s + "up_proj", up, w13)
        put(s + "down_proj", module(o + "mlp.experts.w2"), o + "mlp.experts.w2")
        for which, names in (("w13", ("gate_proj", "up_proj")), ("w2", ("down_proj",))):
            key = o + f"mlp.experts.{which}.global_scale"
            if key in oq:
                # The Swift port's `convertOQ` expression, so both sides use one multiplier.
                scale = take(key).astype(mx.float32) / NVFP4_DENOMINATOR
                for name in names:
                    w[s + name + ".weight_scale_2"] = scale
        if i > 0:
            query = take(o + "attn_input.query")
            w[v + "attention_input.query"] = query.reshape(h) if query.shape[0] == 1 else query
        w[v + "mlp_input.query"] = take(o + "mlp_input.query").reshape(h)
        if loop is not None and loop.core_start <= i < loop.core_end:
            downs = take(o + "loop_down")
            gains = take(o + "loop_gain_forget")
            rank = downs.shape[-1]
            for index, m in enumerate(CONTROLLERS):
                c = v + f"controllers.{m}."
                # loop_down[m] = down.reshape(rank, 2, H).transpose(1, 2, 0).
                w[c + "down_weight"] = downs[index].transpose(2, 0, 1).reshape(rank, 2 * h)
                # loop_gain_forget[m] = concat(gain.T, forget.T, axis=1).
                w[c + "gain_weight"] = gains[index][:, :h].T
                w[c + "forget_weight"] = gains[index][:, h:].T
            messages.append(take(o + "loop_message.query"))
    if messages:
        w["model.message_queries"] = mx.stack(messages)
    unused = sorted(set(oq) - used)
    assert not unused, f"unmapped oQ tensors: {unused[:8]}"
    return w, quant
