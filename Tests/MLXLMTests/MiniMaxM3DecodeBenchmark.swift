import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

/// One M3.1-Flash-sized attention layer at decode, by component. Skipped
/// unless `MINIMAX_M3_BENCH` is set; prints milliseconds per call.
final class MiniMaxM3DecodeBenchmark: XCTestCase {

    private let json = """
        {
            "model_type": "minimax_m3", "vocab_size": 64, "hidden_size": 6144,
            "intermediate_size": 3072, "dense_intermediate_size": 12288,
            "shared_intermediate_size": 3072, "num_hidden_layers": 1,
            "num_attention_heads": 64, "num_key_value_heads": 4, "head_dim": 128,
            "partial_rotary_factor": 0.5, "rope_theta": 10000000, "rms_norm_eps": 1e-6,
            "use_gemma_norm": true, "use_qk_norm": true, "qk_norm_type": "per_head",
            "hidden_act": "swigluoai", "swiglu_alpha": 1.702, "swiglu_limit": 7.0,
            "num_local_experts": 128, "num_experts_per_tok": 4, "n_shared_experts": 1,
            "scoring_func": "sigmoid", "use_routing_bias": true,
            "routed_scaling_factor": 2.0, "moe_layer_freq": [1],
            "pack_shared_expert": false, "tie_word_embeddings": false,
            "sparse_attention_config": {
                "use_sparse_attention": true, "sparse_index_dim": 128,
                "sparse_num_index_heads": 4, "sparse_topk_blocks": 16,
                "sparse_block_size": 128, "sparse_score_type": "max",
                "sparse_init_block": 0, "sparse_local_block": 1,
                "sparse_attention_freq": [1], "sparse_disable_index_value": [1]
            }
        }
        """

    private func time(_ label: String, iterations: Int = 30, _ body: () -> MLXArray) {
        for _ in 0 ..< 3 { eval(body()) }
        let start = Date()
        for _ in 0 ..< iterations { eval(body()) }
        let ms = Date().timeIntervalSince(start) * 1000 / Double(iterations)
        print(String(format: "MM3BENCH %@ %.3f ms", label, ms))
    }

    /// One oQ3e-quantized MoE decoder layer standing in for each of 60
    /// layers (own cache each), decoded with async evaluation like
    /// `TokenIterator`: milliseconds per token for the layer stack.
    func testDecodeLayerStackPipelined() throws {
        guard ProcessInfo.processInfo.environment["MINIMAX_M3_BENCH"] != nil else {
            throw XCTSkip("MINIMAX_M3_BENCH is not set")
        }
        let config = try JSONDecoder().decode(MiniMaxM3Configuration.self, from: Data(json.utf8))
        let layer = MiniMaxM3DecoderLayer(config, layer: 0)
        let parameters = layer.parameters().flattened().map { key, value in
            let noise = 0.02 * MLXRandom.normal(value.shape)
            let keepFloat32 = key.hasSuffix("block_sparse_moe.gate.weight")
                || key.hasSuffix("e_score_correction_bias")
            return (key, keepFloat32 ? noise : noise.asType(.bfloat16))
        }
        layer.update(parameters: ModuleParameters.unflattened(parameters))
        quantize(model: layer) { path, module in
            if path.hasSuffix("block_sparse_moe.gate") { return nil }
            guard module is Linear || module is SwitchLinear else { return nil }
            if path.contains("index_") || path.contains("shared_experts") { return (32, 8, .mxfp8) }
            if path.contains("self_attn.") { return (64, 4, .affine) }
            return (64, 3, .affine)
        }
        for norm in layer.norms { _ = norm.prepare() }
        eval(layer)

        for total in [1024, 8192, 32768] {
            let caches = (0 ..< 60).map { _ in MiniMaxM3Cache() }
            for cache in caches {
                cache.state = [
                    MLXRandom.normal([1, 4, total, 128]).asType(.bfloat16),
                    MLXRandom.normal([1, 4, total, 128]).asType(.bfloat16),
                    MLXRandom.normal([1, 1, total, 128]).asType(.bfloat16),
                ]
                eval(cache.innerState())
            }
            let x = (0.5 * MLXRandom.normal([1, 1, 6144])).asType(.bfloat16)
            func token() -> MLXArray {
                var h = x
                for cache in caches { h = layer(h, mask: .none, cache: cache) }
                return h
            }
            eval(token())
            let steps = 24
            let start = Date()
            var last = token()
            asyncEval([last] + caches.flatMap { $0.state })
            for _ in 1 ..< steps {
                last = token()
                asyncEval([last] + caches.flatMap { $0.state })
            }
            eval(last)
            let ms = Date().timeIntervalSince(start) * 1000 / Double(steps)
            print(String(format: "MM3BENCH T=%d stack60-token %.2f ms (%.1f tok/s)", total, ms, 1000 / ms))
        }
    }

    func testDecodeAttentionComponents() throws {
        guard ProcessInfo.processInfo.environment["MINIMAX_M3_BENCH"] != nil else {
            throw XCTSkip("MINIMAX_M3_BENCH is not set")
        }
        let config = try JSONDecoder().decode(MiniMaxM3Configuration.self, from: Data(json.utf8))
        let attention = MiniMaxM3Attention(config, layer: 0)
        let parameters = attention.parameters().flattened().map { key, value in
            (key, (0.02 * MLXRandom.normal(value.shape)).asType(.bfloat16))
        }
        attention.update(parameters: ModuleParameters.unflattened(parameters))
        for norm in attention.norms { _ = norm.prepare() }
        eval(attention)

        for total in [1024, 8192, 32768] {
            let cache = MiniMaxM3Cache()
            cache.state = [
                MLXRandom.normal([1, 4, total, 128]).asType(.bfloat16),
                MLXRandom.normal([1, 4, total, 128]).asType(.bfloat16),
                MLXRandom.normal([1, 1, total, 128]).asType(.bfloat16),
            ]
            eval(cache.innerState())
            let x = (0.5 * MLXRandom.normal([1, 1, 6144])).asType(.bfloat16)
            time("T=\(total) layer-step") {
                let out = attention(x, mask: .none, cache: cache)
                return out
            }

            let q = MLXRandom.normal([1, 64, 1, 128]).asType(.bfloat16)
            let k = MLXRandom.normal([1, 4, total, 128]).asType(.bfloat16)
            let v = MLXRandom.normal([1, 4, total, 128]).asType(.bfloat16)
            let iq = MLXRandom.normal([1, 4, 1, 128]).asType(.bfloat16)
            let ik = MLXRandom.normal([1, 1, total, 128]).asType(.bfloat16)
            eval(q, k, v, iq, ik)
            let qStart = total - 1
            time("T=\(total) sdpa-nomask") {
                MLXFast.scaledDotProductAttention(
                    queries: q, keys: k, values: v, scale: 0.088, mask: .none)
            }
            guard total > 2048 else { continue }
            time("T=\(total) selectBlocks") {
                attention.selectBlocks(queries: iq, keys: ik, qStart: qStart, sorted: false)
            }
            let blocks = attention.selectBlocks(queries: iq, keys: ik, qStart: qStart, sorted: false)
            eval(blocks)
            time("T=\(total) blockMask") {
                attention.blockMask(blocks, total: total, qStart: qStart)
            }
            let mask = attention.blockMask(blocks, total: total, qStart: qStart)
            eval(mask)
            time("T=\(total) sdpa-mask") {
                MLXFast.scaledDotProductAttention(
                    queries: q, keys: k, values: v, scale: 0.088, mask: .array(mask))
            }
            let sortedBlocks = attention.selectBlocks(
                queries: iq, keys: ik, qStart: qStart, sorted: true)
            let positions = MLXArray([Int32(qStart)]).reshaped(1, 1)
            eval(sortedBlocks)
            time("T=\(total) sparse-kernel") {
                miniMaxM3SparseAttention(
                    queries: q, keys: k, values: v, blockIndices: sortedBlocks,
                    queryPositions: positions, scale: 0.088, blockSize: 128)!
            }
        }
    }
}
