# Inkling 2 Mini decode: Ollama levers ported (issue 2467)

Branch `macprovider/2467-ollama-kernels`, from `5c0ccd8`. Nothing here has been built or run
yet. Every lever below needs a Studio build, the parity gate and the A/B.

## Where the time goes

Our decode is 24.7 / 23.8 / 21.2 tok/s at 1K / 8K / 32K (oQ4e, compiled path). mlx-vlm decodes
at 29.0 / 27.9 / 24.7. Rule 15 needs 95% of mlx-vlm: 27.6 / 26.5 / 23.5.

Decode is bound by dispatches, not by bandwidth:

- **Dispatches.** One token runs about 5000 GPU dispatches over 48 layer visits: 40 local, and 8 global (layers 5, 11, 17, 23 run twice in the loop).
- **Ceiling.** A token reads about 3.8 GB, so memory bandwidth allows about 210 tok/s. We decode at 12% of that.
- **Native weights.** The native NVFP4/MXFP8 source checkpoint decodes at the same speed as oQ4e (24.59 / 23.62 / 21.10).
- **Host time.** Building the graph takes about 6 ms per token. It is hidden behind 36 to 43 ms of GPU wait.

Phase shares come from the 1K profile of the plain path, with an `eval` barrier after every phase. Read them as relative shares, not absolute times. The phases are:

| Phase | Share | Calls per token |
|---|---|---|
| attention | 21% | 48 |
| shortConvolutions | 20% | 192 |
| messages | 18% | 122 |
| norms | 14% | 152 |
| residualMix | 13% | 97 |
| experts | 7.5% | 48 |
| router | 5.8% | 48 |

## Levers

In the commit column, "—" means the lever was not ported. In the "Exact?" column, "yes" means bit-equal to the existing path. "Tokens" means the reduction order changes, so the gate is greedy token parity.

| Lever (`INKLING2_FAST_OFF` name) | Done before? | Phase it targets | Est. dispatches cut per token | Exact? | Commit |
|---|---|---|---|---|---|
| Fused residual mix, plus the input RMS norm (`mix`) | no | residualMix 13% + norms 14% | ~880 | tokens | 195b578 |
| Fused message controllers, 2 kernels per launch (`ctrl`) | no | messages 18% | ~770 | tokens | 2d3a8b5 |
| Fused attention inputs: clips, value norm, log scaling, layouts (`inputs`) | no | attention 21% | ~530 | tokens | b9955fe |
| Short convolution with the residual add; dynamic K/V conv in one kernel (`convadd`) | partly (`convstate`) | shortConvolutions 20% | ~170 | yes | 337db8d |
| One quantized product per (projection, input) (`oneproj`) | no | attention projections | ~200 | yes* | 47bd807 |
| Shared router top-k kernel (`router`) | no | router 5.8% | ~50–100 | yes | 5e33b34 |
| Ring read as stored, sink as the SDPA sink (`sinkring`) | no | attention 21% plus the 1K→32K slope | ~144, plus the ring copies | tokens | 716c094 |
| NVFP4/MXFP8 expert gathers with activation quantization | — | experts 7.5% | — | — | not ported: native weights only, W4A4 semantics, epilogue M5+ only |
| BF16 short products and head | — | head 0.5% | — | — | not ported: native weights only; our head is 8-bit |
| 8-bit draft and 4-bit draft head; block-draft acceptance carry | — | (drafting) | — | — | for the DSpark drafter port, not this decode |
| Early submit | — | host 6 ms | — | — | not ported: host time is already hidden behind GPU time |

The ported levers together cut about 2700 of the 5000 dispatches.

\* `oneproj` is exact if MLX's quantized matrix-vector product computes each output row from its own weight rows alone. `testFastPathLeversAreBitExact` checks this.

**Native NVFP4 recommendation: keep serving oQ4e.**

- oQ4e already stores the experts of layers 0–19 as NVFP4.
- The native checkpoint measured no faster.
- Native needs more memory, because attention and the head are BF16. That is a risk on the 24 GB tier.
- Once dispatch overhead is gone, bytes per token decide the speed, and the 8-bit attention reads fewer bytes than BF16.
- The only native-only levers are the W4A4 expert kernels. They use different activation semantics from the mlx-vlm oracle, so they would not pass token parity against it.

## Studio A/B plan

Window `scratchpad/windows/window-2467-ref.sh` setup:

1. Set `WT=/Users/augstar/mlx-swift-lm-2467-ollama` and `BRANCH=macprovider/2467-ollama-kernels`.
2. Push the branch first. The script requires that HEAD equals the pushed branch.
3. Run with `PORT_ONLY=1 SKIP_DOWNLOAD=1` and `INKLING2_BENCH_CONTEXTS=1024,8192,32768`.

Steps:

1. **Gate, before any timing.** Run the real greedy parity test against the mlx-vlm oracle: 64/64 tokens at 1K and 8K, for both source and oQ, on the compiled path. Then run `testFastPathLeversAreBitExact`, where exact levers must be bit-equal and the others must give the same tokens. A failing lever is switched off by name and its commit is reverted.
2. **All new levers against the current path.** In the control/candidate/control run below, `compiled~mix~ctrl~inputs~convadd~oneproj~router~sinkring` is the compiled path of `5c0ccd8`:
   `BENCH_PATHS=compiled~mix~ctrl~inputs~convadd~oneproj~router~sinkring,compiled,compiled~mix~ctrl~inputs~convadd~oneproj~router~sinkring`
3. **One lever off at a time,** each run between two all-on controls. Keep a lever only if all-on beats both neighbouring runs with that lever off:
   `BENCH_PATHS=compiled,compiled~mix,compiled,compiled~ctrl,compiled,compiled~inputs,compiled,compiled~convadd,compiled,compiled~oneproj,compiled,compiled~router,compiled,compiled~sinkring,compiled`
4. **Report.** For each context, report decode tok/s, prefill tok/s, and the sync build/eval ms. Report each as a share of mlx-vlm (95% target) and of the 210 tok/s ceiling. Add each lever's result to the Known levers table.
