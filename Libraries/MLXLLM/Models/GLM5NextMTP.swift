//
//  GLM5NextMTP.swift
//  mlx-swift-lm
//
//  GLM-5-Next (GLM-5.3-Flash) multi-token-prediction drafter. The checkpoint's
//  nextn layer (`GLM5NextMTPLayer`, loaded with the target) fuses the next
//  token's embedding with the target's pre-final-norm hidden state:
//  `eh_proj([enorm(embed(token)); hnorm(h)])`, then one plain pre-norm decoder
//  block (sparse attention with its own indexer, MoE; no hyper-connections),
//  then the `shared_head` norm and the target's lm_head. Feed and block follow
//  oMLX v0.7.1rc1 (`omlx/patches/mlx_vlm_mtp/glm5_next_vlm_runtime.py`,
//  https://github.com/jundot/omlx, Apache License 2.0): the trunk's raw hidden
//  for the first fold, the head's own output for chained steps.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// MTP drafter for GLM-5-Next, depth 1...7 (block size up to 8): the single
/// nextn layer drafts recursively, as oMLX does. Each round folds the
/// committed tokens (accepted drafts plus the final token) into the head's
/// cache with the target's hidden states, takes the first draft from that
/// forward, and chains the rest on the head's own output hidden; chained
/// entries are tentative and every commit trims them all (exact trim), so the
/// cache holds committed entries only. It owns no parameters: every call
/// resolves the target's `mtp[0]` layer, embeddings and lm_head. Its
/// per-stream state is one `GLM5NextAttentionCache`.
public final class GLM5NextMTPDraftModel: Module, MTPPackedStatefulDrafterModel {
    public let configuration: GLM5NextConfiguration
    /// A verify carries at most 8 columns (the batched-exactness cap).
    public let maximumBlockSize: Int? = 8
    public let requiresSharedTargetKV = false
    public let requiresPromptPrefill = true
    public let requiresGreedySampling = true

    /// Drafter-cache tokens a round leaves tentative: the chained drafts.
    public static let maxTrim = 7

    public init(_ configuration: GLM5NextConfiguration) {
        self.configuration = configuration
        super.init()
    }

    public func makeState(parameters: GenerateParameters?) -> MTPDrafterState {
        MTPDrafterState(cache: [
            GLM5NextAttentionCache(kpool: configuration.indexKpool, maxTrim: Self.maxTrim)
        ])
    }

    // MARK: - Head

    private struct Head {
        let layer: GLM5NextMTPLayer
        let text: GLM5NextTextModel

        /// The block's output before the shared-head norm, `[B, L, D]`.
        func forward(
            tokens: MLXArray, hidden: MLXArray, cache: KVCache, rowLengths: [Int]? = nil
        ) -> MLXArray {
            let embedded = layer.enorm(text.model.embedTokens(tokens))
            var x = glm5NextProject(
                layer.ehProj, concatenated([embedded, layer.hnorm(hidden)], axis: -1))
            let block = layer.block
            let (attention, _) = block.selfAttn(
                block.inputLayerNorm(x), cache: cache, previousTopk: [], rowLengths: rowLengths)
            x = x + attention
            return x + (block.mlp as! UnaryLayer)(block.postAttentionLayerNorm(x))
        }

        /// One proposal per row from the last column of `raw`, `[B, 1]`.
        func sample(_ raw: MLXArray, sampler: any LogitSampler) -> MLXArray {
            let logits = text.logits(layer.norm(raw))
            return sampler.sample(logits: logits[0..., -1, 0...]).reshaped(raw.dim(0), 1)
        }
    }

    private func head(_ target: any LanguageModel) -> Head {
        let text: GLM5NextTextModel
        if let model = target as? GLM5NextModel {
            text = model.languageModel
        } else if let model = target as? GLM5NextTextModel {
            text = model
        } else {
            fatalError("GLM5NextMTPDraftModel requires a GLM5Next target, got \(type(of: target))")
        }
        guard let layer = text.mtp.first as? GLM5NextMTPLayer else {
            fatalError("GLM5NextMTPDraftModel: the target was loaded without MTP weights")
        }
        return Head(layer: layer, text: text)
    }

    // MARK: - Serial

    public func prepareDrafterState(
        target: any LanguageModel,
        promptTokens: MLXArray,
        targetHidden: MLXArray,
        firstBonus: MLXArray,
        positionDeltas _: MLXArray?,
        state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) {
        let head = head(target)
        let prompt = normalizedMTPTokenBatch(promptTokens)
        guard prompt.dim(-1) > 0 else { return }
        let shifted = concatenated([prompt[0..., 1...], normalizedMTPColumn(firstBonus)], axis: 1)
        let raw = head.forward(
            tokens: shifted, hidden: targetHidden[0..., ..<shifted.dim(1), 0...],
            cache: state.cache[0])
        state.nextPosition = shifted.dim(1)
        state.seedHidden = raw[0..., (-1)..., 0...]
        state.seedToken = head.sample(state.seedHidden!, sampler: sampler)
        state.proposalAppended = 0
    }

    public func draftBlock(
        target: any LanguageModel,
        lastToken: MLXArray,
        lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)],
        positionDeltas: MLXArray?,
        queryOffset: Int,
        blockSize: Int,
        sampler: any LogitSampler
    ) -> MLXArray {
        var state = makeState(parameters: nil)
        return draftBlock(
            target: target, lastToken: lastToken, lastHidden: lastHidden, sharedKV: sharedKV,
            positionDeltas: positionDeltas, queryOffset: queryOffset, blockSize: blockSize,
            state: &state, sampler: sampler)
    }

    /// `blockSize - 1` proposals: the seed computed while committing the
    /// previous round (or a first chained step without one), then chained
    /// steps each feeding the head's own output hidden and the previous draft
    /// back; chained entries stay tentative in the drafter cache until the
    /// commit trims them.
    public func draftBlock(
        target: any LanguageModel,
        lastToken: MLXArray,
        lastHidden: MLXArray,
        sharedKV _: [String: (MLXArray, MLXArray)],
        positionDeltas _: MLXArray?,
        queryOffset _: Int,
        blockSize: Int,
        state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) -> MLXArray {
        let head = head(target)
        var proposals = [MLXArray]()
        var token: MLXArray
        var hidden: MLXArray
        if let seed = state.seedToken, let seedHidden = state.seedHidden {
            (token, hidden) = (seed, seedHidden)
            proposals.append(seed)
        } else {
            (token, hidden) = (normalizedMTPColumn(lastToken), lastHidden)
        }
        state.seedToken = nil
        state.seedHidden = nil
        var appended = 0
        while proposals.count < blockSize - 1 {
            hidden = head.forward(tokens: token, hidden: hidden, cache: state.cache[0])
            appended += 1
            token = head.sample(hidden, sampler: sampler)
            proposals.append(token)
        }
        state.proposalAppended = appended
        state.nextPosition += appended
        return concatenated(proposals, axis: 1)
    }

    /// Trim every tentative chained entry (exact), then fold the accepted
    /// proposals and the final token, each with the target hidden of the
    /// verification column before it, and seed the next proposal from the
    /// last (oMLX's committed-only head history).
    public func commitDrafterState(
        target: any LanguageModel,
        targetHidden: MLXArray,
        draftTokens: MLXArray,
        acceptedCount: Int,
        finalToken: MLXArray,
        positionDeltas _: MLXArray?,
        state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) {
        let keepAppended = 0
        let trim = state.proposalAppended
        if trim > 0 {
            let trimmed = state.cache[0].trim(trim)
            precondition(trimmed == trim, "GLM5Next MTP cache trimmed \(trimmed) of \(trim)")
            state.nextPosition -= trim
        }
        var tokens = [MLXArray]()
        for index in keepAppended ..< acceptedCount {
            tokens.append(draftTokens[0..., index ..< (index + 1)])
        }
        tokens.append(normalizedMTPColumn(finalToken))
        let head = head(target)
        let raw = head.forward(
            tokens: concatenated(tokens, axis: 1),
            hidden: targetHidden[0..., keepAppended ..< (acceptedCount + 1), 0...],
            cache: state.cache[0])
        state.nextPosition += tokens.count
        state.seedHidden = raw[0..., (-1)..., 0...]
        state.seedToken = head.sample(state.seedHidden!, sampler: sampler)
        state.proposalAppended = 0
    }

    // MARK: - Packed

    /// Every row's commit in one head forward: the rows' columns right-padded
    /// to the widest, each row attending only its own columns against its
    /// own (copied, trimmed) cache, then each row's next proposal from its
    /// last valid column. Input states are not mutated; nothing is evaluated.
    public func advanceAndProposePacked(
        target: any LanguageModel,
        rows: [MTPPackedDrafterAdvanceRow],
        sampler: any LogitSampler
    ) throws -> MTPPackedDrafterAdvanceResult {
        guard !rows.isEmpty else {
            return MTPPackedDrafterAdvanceResult(
                states: [], proposals: MLXArray.zeros([0, 1], dtype: .int32))
        }
        var caches = [GLM5NextAttentionCache]()
        var columnTokens = [[Int32]]()
        var columnHidden = [MLXArray]()
        for (index, row) in rows.enumerated() {
            let hidden = row.targetHidden
            guard hidden.ndim == 3, hidden.dim(0) == 1, hidden.dim(1) > 0 else {
                throw MTPPackedDrafterError.invalidTargetHidden(row: index, shape: hidden.shape)
            }
            let acceptedCount = row.acceptedTokens.count
            guard acceptedCount < hidden.dim(1) else {
                throw MTPPackedDrafterError.invalidAcceptedCount(
                    row: index, acceptedCount: acceptedCount, inputCount: hidden.dim(1))
            }
            guard row.state.cache.count == 1,
                let cache = row.state.cache[0] as? GLM5NextAttentionCache,
                row.state.proposalAppended >= 0
            else { throw MTPPackedDrafterError.incompatibleState(row: index) }
            guard cache.offset == row.state.nextPosition else {
                throw MTPPackedDrafterError.positionMismatch(
                    row: index, nextPosition: row.state.nextPosition, cacheOffset: cache.offset)
            }
            let keepAppended = 0
            let trim = row.state.proposalAppended
            let copy = cache.copy() as! GLM5NextAttentionCache
            guard copy.trim(trim) == trim else {
                throw MTPPackedDrafterError.incompatibleState(row: index)
            }
            caches.append(copy)
            columnTokens.append(
                row.acceptedTokens[keepAppended...].map(Int32.init) + [Int32(row.finalToken)])
            columnHidden.append(hidden[0..., keepAppended ..< (acceptedCount + 1), 0...])
        }

        let counts = columnTokens.map(\.count)
        let width = counts.max()!
        let hiddenSize = columnHidden[0].dim(2)
        let tokens = MLXArray(
            columnTokens.flatMap { $0 + Array(repeating: Int32(0), count: width - $0.count) },
            [rows.count, width])
        let hidden = concatenated(
            columnHidden.map { h -> MLXArray in
                let pad = width - h.dim(1)
                return pad == 0
                    ? h
                    : concatenated(
                        [h, MLXArray.zeros([1, pad, hiddenSize], dtype: h.dtype)], axis: 1)
            }, axis: 0)

        let head = head(target)
        let raw = head.forward(
            tokens: tokens, hidden: hidden,
            cache: GLM5NextLatentCacheBatch(rowCaches: caches), rowLengths: counts)
        let seedHidden = counts.enumerated().map { row, count in
            raw[row ..< (row + 1), (count - 1) ..< count, 0...]
        }
        let proposals = head.sample(concatenated(seedHidden, axis: 0), sampler: sampler)
        let states = rows.indices.map { row in
            MTPDrafterState(
                cache: [caches[row]],
                nextPosition: caches[row].offset,
                seedToken: proposals[row ..< (row + 1), 0...],
                seedHidden: seedHidden[row],
                proposalAppended: 0)
        }
        return MTPPackedDrafterAdvanceResult(states: states, proposals: proposals)
    }

    /// Continue every row's chain from its seed to `depth` proposals with one
    /// head forward per step for all rows (each row attending its own copied
    /// cache): the packed form of the chained `draftBlock`, bitwise equal to
    /// it per row. Returns the advanced states (chained entries tentative) and
    /// the proposals `[rows, depth]`. Input states are not mutated.
    public func chainPacked(
        target: any LanguageModel, states: [MTPDrafterState], depth: Int,
        sampler: any LogitSampler
    ) throws -> (states: [MTPDrafterState], proposals: MLXArray) {
        precondition(depth >= 1 && depth <= Self.maxTrim + 1)
        var caches = [GLM5NextAttentionCache]()
        var tokens = [MLXArray]()
        var hidden = [MLXArray]()
        for (index, state) in states.enumerated() {
            guard state.cache.count == 1, let cache = state.cache[0] as? GLM5NextAttentionCache,
                let seed = state.seedToken, let seedHidden = state.seedHidden,
                state.proposalAppended == 0
            else { throw MTPPackedDrafterError.incompatibleState(row: index) }
            caches.append(cache.copy() as! GLM5NextAttentionCache)
            tokens.append(seed)
            hidden.append(seedHidden)
        }
        let head = head(target)
        var token = concatenated(tokens, axis: 0)
        var h = concatenated(hidden, axis: 0)
        var proposals = [token]
        for _ in 1 ..< depth {
            h = head.forward(
                tokens: token, hidden: h, cache: GLM5NextLatentCacheBatch(rowCaches: caches))
            token = head.sample(h, sampler: sampler)
            proposals.append(token)
        }
        let advanced = states.indices.map { row in
            MTPDrafterState(
                cache: [caches[row]], nextPosition: caches[row].offset, seedToken: nil,
                seedHidden: nil, proposalAppended: depth - 1)
        }
        return (advanced, concatenated(proposals, axis: 1))
    }

    /// The head's weights load with the target (`mtp.*`); the drafter keeps none.
    public func sanitize(weights _: [String: MLXArray]) -> [String: MLXArray] { [:] }
}

/// Registers the GLM-5-Next MTP drafter for `glm5_next` and `glm5_next_text`
/// configurations. Call once before loading the drafter through
/// `MTPDrafterModelFactory`.
public enum GLM5NextMTPRegistration {
    public static func register() async {
        for modelType in ["glm5_next", "glm5_next_text"] {
            await MTPDrafterTypeRegistry.shared.registerModelType(
                modelType,
                creator: { data in
                    let config = try JSONDecoder.json5().decode(
                        GLM5NextConfiguration.self, from: data)
                    try config.validateModelConfiguration()
                    return GLM5NextMTPDraftModel(config)
                }
            )
        }
    }
}
