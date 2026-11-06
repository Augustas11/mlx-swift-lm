# Compiled KV-cache decoding

Compile single-token decode steps for compatible text models, either through
``TokenIterator`` with ``GenerateParameters/compiledDecode`` or directly with
``CompiledDecodeSession``.

## Why this is a separate cache

``KVCacheSimple`` is optimized for eager generation. It grows storage in chunks
and returns only the written prefix, so attention work tracks the actual context
length. Its write position is a Swift `Int`, however, and dynamic slices derived
from that value become constants when MLX traces a compiled function.

``FixedCapacityKVCache`` has the opposite tradeoff. It allocates its final shape
before compilation, carries position as an `MLXArray`, scatters new rows using
tensor indices, and masks unwritten capacity. This makes the graph reusable and
correct across decode steps, but attention spans the configured capacity even
when only part of it has been written. Keeping the implementations separate
preserves the efficient default for eager generation.

## Supported models

Only models conforming to ``FixedCapacityKVCacheProviding`` may use this cache.
The capability means the model has been audited and parity-tested so every
position-dependent forward operation remains in the MLX graph.

The supported implementations are:

- `Qwen2Model`, including Qwen 2 and Qwen 2.5 text checkpoints.
- `LlamaModel`, including Llama and the Mistral text checkpoints backed by that
  implementation.

Do not substitute this cache into vision-language, sliding-window, recurrent,
hybrid, or model-specific cache layouts. Unsupported models continue using their
standard eager cache.

## Serial generation

Set ``GenerateParameters/compiledDecode`` to compile ``TokenIterator``'s decode
steps. The iterator keeps the caller's cache: prefill and the first decode step
run eagerly into the ordinary ``KVCacheSimple`` layers, then the iterator adopts
those buffers into ``FixedCapacityKVCache`` instances and compiles one step.
After each compiled step it points every ``KVCacheSimple`` back at the
fixed-capacity buffers with the new offset (no copy), so at every token boundary
the caller's layers are ordinary ``KVCacheSimple`` caches. Reusing them for a
later turn, trimming them, or serializing them with `savePromptCache(url:cache:metadata:state:)`
works exactly as after eager decode.

Capacity grows in blocks of 256 positions, bounded by the remaining
`maxTokens`. Opening a block re-adopts the layers into larger caches and builds
a new compiled step; the previous one is released, which erases its traces.

The compiled path is skipped, and decode stays eager, unless every layer is
exactly ``KVCacheSimple`` at batch 1 and one shared offset, the model carries no
per-call ``LMOutput/State``, the input is plain text, and the request has no
pending dynamic cache conversion (for example `kvBits`). If the layers are
replaced or trimmed while generating, the iterator adopts them again.

## CompiledDecodeSession

Size capacity for the prompt and the complete generation. The session performs
prefill before creating the compiled closure and rejects decode after capacity:

```swift
guard let cacheProvider = model as? any FixedCapacityKVCacheProviding else {
    // Fall back to the normal eager generation path.
    return
}

let maxNewTokens = 256
let capacity = promptTokens.dim(1) + maxNewTokens
let session = try CompiledDecodeSession(
    model: cacheProvider, prompt: promptTokens, capacity: capacity)

var token = nextToken(from: session.prefillLogits)
emit(token)
for _ in 1 ..< maxNewTokens {
    let logits = try session.step(token.reshaped(1, 1))
    token = nextToken(from: logits)
    emit(token)
}
```

``CompiledDecodeSession`` owns the host token count because a compiled graph
cannot raise a Swift error from a tensor-valued position without synchronizing
it to the CPU. Advanced callers can construct ``FixedCapacityKVCache`` directly,
but then they own prefill order and capacity admission: check
`offset + inputTokenCount <= maxTokens` on the host before each model invocation,
outside the compiled closure.

Both paths declare the model's weights as compile inputs (a snapshot of the
module's arrays taken when the step is built), so `update(parameters:)` is not
captured as a constant. Finish the session or generation before changing the
module tree (loading, fusing, or unloading adapters, or quantizing), then start
a new one. A session's existing KV state is not recomputed when weights change.
