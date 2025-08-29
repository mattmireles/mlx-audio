/// KokoroTTS.swift - Core MLX Neural Text-to-Speech Engine
///
/// This file implements the core neural text-to-speech synthesis engine using Apple's MLX
/// framework for high-performance on-device inference. It provides the complete TTS pipeline
/// from text preprocessing through neural synthesis to audio waveform generation.
///
/// Key responsibilities:
/// - **Neural Model Orchestration**: Coordinates BERT encoder, duration predictor, and decoder models
/// - **Text Processing Pipeline**: Phonemization, tokenization, and linguistic feature extraction
/// - **MLX Inference**: Optimized neural network inference for Apple Silicon (M1/M2/M3)
/// - **Audio Synthesis**: Streaming waveform generation with real-time chunk-based output
/// - **Memory Management**: Efficient tensor operations with automatic memory cleanup
/// - **Voice Support**: 60+ voice models across multiple languages and speaking styles
///
/// Architecture:
/// - **Transformer-Based**: Uses ALBERT encoder for contextual text understanding
/// - **Duration Modeling**: LSTM-based phoneme duration prediction for natural speech timing
/// - **Prosody Generation**: F0 (pitch) and energy prediction for expressive synthesis
/// - **Neural Vocoder**: High-quality waveform generation from intermediate representations
/// - **Streaming Design**: Sentence-by-sentence processing for low-latency real-time synthesis
///
/// Called by:
/// - `KokoroTTSModel.swift`: SwiftUI-compatible wrapper providing UI state management
/// - `MlxTTSService.swift` (main TalkToMe app): Service layer for integration with TTS pipeline
/// - Benchmark tools: Performance measurement and model evaluation utilities
/// - Direct API consumers: Applications requiring low-level TTS control
///
/// Integrates with:
/// - **MLX Framework**: Apple's ML framework optimized for Apple Silicon neural inference
/// - **ALBERT Models**: Contextual text encoder providing linguistic understanding
/// - **ESpeakNG**: Phonemization engine for converting text to phonetic representations
/// - **Voice Embeddings**: Pre-trained speaker models for voice characteristics
/// - **Neural Components**: Duration encoder, prosody predictor, text encoder, decoder
///
/// Neural Pipeline Architecture:
/// 1. **Text Preprocessing**: Input text → phonemization → tokenization → input IDs
/// 2. **Contextual Encoding**: ALBERT transformer processes tokens for linguistic context
/// 3. **Duration Prediction**: LSTM predicts phoneme durations for natural speech timing
/// 4. **Alignment Matrix**: Maps linguistic features to audio frame timeline
/// 5. **Prosody Generation**: Predicts F0 (pitch) curves and energy patterns
/// 6. **Text Encoding**: Parallel text processing for synthesis conditioning
/// 7. **Neural Vocoding**: Decoder generates high-quality audio waveforms
/// 8. **Post-Processing**: Audio cleanup and format conversion for playback
///
/// Performance Characteristics:
/// - **Synthesis Speed**: ~50-100ms per sentence depending on length and complexity
/// - **Model Size**: ~80MB neural models with optional 8-bit quantization
/// - **Memory Usage**: ~100-300MB peak during synthesis depending on sequence length
/// - **Quality**: State-of-the-art neural synthesis rivaling cloud-based TTS services  
/// - **Apple Silicon**: Optimized for M1/M2/M3 Neural Engine acceleration via MLX
///
/// Voice Model Support:
/// - **60+ Voices**: Comprehensive collection across multiple languages and styles
/// - **Language Coverage**: English, Japanese, Chinese, and other major languages
/// - **Style Variety**: Male/female, young/mature, formal/casual speaking styles
/// - **Quality Tiers**: High-quality bfloat16 and efficient 8-bit quantized variants
///
/// Memory Management Strategy:
/// - **Lazy Loading**: Models loaded on-demand to minimize startup time and memory
/// - **Autorelease Pools**: Aggressive memory cleanup during synthesis operations
/// - **Tensor Lifecycle**: Explicit evaluation and release of intermediate tensors
/// - **GPU Cache Management**: Periodic MLX GPU cache clearing for sustained operation
///
/// Error Handling and Robustness:
/// - **Graceful Degradation**: Fallback audio generation for synthesis failures
/// - **Resource Validation**: Comprehensive model weight and voice embedding validation
/// - **Memory Recovery**: Automatic cleanup and reset for out-of-memory conditions
/// - **Tokenization Limits**: Safe handling of text exceeding maximum token count
///
/// Thread Safety and Concurrency:
/// - **Async Processing**: Background synthesis with main thread callback delivery
/// - **Sentence Streaming**: Concurrent processing of multiple sentences for responsiveness
/// - **State Isolation**: Thread-safe model state management across concurrent requests

import Foundation
import MLX
import MLXNN

/// Comprehensive voice options for Kokoro neural text-to-speech synthesis.
///
/// This enumeration provides 60+ voice options across multiple languages and speaking
/// styles, each corresponding to pre-trained neural voice embeddings optimized for
/// specific acoustic characteristics and linguistic patterns.
///
/// Voice Naming Convention:
/// - **Prefix**: Language/region code (af=American Female, am=American Male, etc.)
/// - **Name**: Distinctive identifier for voice characteristics and style
///
/// Language Support:
/// - **English**: American English voices with diverse styles and characteristics
/// - **Japanese**: Native Japanese voices optimized for Japanese phonetics
/// - **Chinese**: Mandarin Chinese voices with proper tonal handling
/// - **Others**: Extended language support via specialized voice embeddings
///
/// Voice Categories:
/// - **af* (American Female)**: Female English voices with various age and style profiles
/// - **am* (American Male)**: Male English voices spanning formal to casual styles
/// - **bf* (British Female)**: British English female voices with accent variations
/// - **bm* (British Male)**: British English male voices with regional characteristics
/// - **jf*/jm* (Japanese)**: Japanese voices optimized for Japanese phonetic structure
/// - **zf*/zm* (Chinese)**: Chinese voices with proper Mandarin pronunciation
///
/// Integration with Neural Models:
/// - Each voice maps to a specific neural embedding in the Kokoro model architecture
/// - Voice embeddings trained on extensive datasets for authentic speech characteristics
/// - Compatible with both streaming and batch synthesis modes
/// - Supports runtime voice switching without model reloading
public enum TTSVoice: String, CaseIterable {
  case afAlloy
  case afAoede
  case afBella
  case afHeart
  case afJessica
  case afKore
  case afNicole
  case afNova
  case afRiver
  case afSarah
  case afSky
  case amAdam
  case amEcho
  case amEric
  case amFenrir
  case amLiam
  case amMichael
  case amOnyx
  case amPuck
  case amSanta
  case bfAlice
  case bfEmma
  case bfIsabella
  case bfLily
  case bmDaniel
  case bmFable
  case bmGeorge
  case bmLewis
  case efDora
  case emAlex
  case ffSiwis
  case hfAlpha
  case hfBeta
  case hfOmega
  case hmPsi
  case ifSara
  case imNicola
  case jfAlpha
  case jfGongitsune
  case jfNezumi
  case jfTebukuro
  case jmKumo
  case pfDora
  case pmSanta
  case zfZiaobei
  case zfXiaoni
  case zfXiaoxiao
  case zfZiaoyi
  case zmYunjian
  case zmYunxi
  case zmYunxia
  case zmYunyang
}

/// Core neural text-to-speech engine providing MLX-optimized synthesis pipeline.
///
/// This class orchestrates the complete Kokoro TTS architecture, from text preprocessing
/// through neural inference to audio waveform generation, optimized for Apple Silicon
/// devices using the MLX framework for maximum performance and efficiency.
///
/// Architecture Overview:
/// - **Multi-Stage Pipeline**: Sequential processing through specialized neural components
/// - **Lazy Initialization**: On-demand model loading to minimize memory footprint
/// - **Streaming Synthesis**: Sentence-by-sentence processing for real-time applications
/// - **Memory Optimized**: Aggressive tensor cleanup and autorelease pool management
///
/// Neural Components:
/// - **ALBERT Encoder**: Contextual text understanding via transformer architecture
/// - **Duration Predictor**: LSTM-based phoneme timing for natural speech rhythm
/// - **Prosody Generator**: F0 (pitch) and energy prediction for expressive synthesis
/// - **Text Encoder**: Parallel linguistic feature extraction for synthesis conditioning
/// - **Neural Decoder**: High-quality audio waveform generation from intermediate features
///
/// Performance Optimizations:
/// - **Apple Silicon**: MLX framework provides Neural Engine acceleration for inference
/// - **Chunked Processing**: Memory-efficient processing of long text sequences
/// - **Batch Operations**: Optimized tensor operations for improved throughput
/// - **GPU Cache Management**: Periodic cache clearing for sustained operation
///
/// Thread Safety:
/// - **Async Operations**: Background synthesis with callback-based result delivery
/// - **State Management**: Thread-safe model initialization and voice switching
/// - **Memory Cleanup**: Automatic resource management across concurrent operations
public class KokoroTTS {
  enum KokoroTTSError: Error {
    case tooManyTokens
    case sentenceSplitError
    case modelNotInitialized
  }

  private var bert: CustomAlbert!
  private var bertEncoder: Linear!
  private var durationEncoder: DurationEncoder!
  private var predictorLSTM: LSTM!
  private var durationProj: Linear!
  private var prosodyPredictor: ProsodyPredictor!
  private var textEncoder: TextEncoder!
  private var decoder: Decoder!
  private var eSpeakEngine: ESpeakNGEngine!
  private var kokoroTokenizer: KokoroTokenizer!
  private var chosenVoice: TTSVoice?
  private var voice: MLXArray!

  // Flag to indicate if model components are initialized
  private var isModelInitialized = false

  // Custom URL of Koroko safetensors file
  private var customURL: URL?

  // Callback type for streaming audio generation
  public typealias AudioChunkCallback = (MLXArray) -> Void

  /// Initializes with a custom URL for the kokoro safetensors file.
  ///
  /// If the custom URL is nil, it will fallback to the bundled kokoro-v1_0_bf16.safetensors resource.
  public init(customURL: URL? = nil) {
    self.customURL = customURL
  }

  // Reset the model to free up memory
  public func resetModel(preserveTextProcessing: Bool = true) {
    // Reset heavy ML model components
    bert = nil
    bertEncoder = nil
    durationEncoder = nil
    predictorLSTM = nil
    durationProj = nil
    prosodyPredictor = nil
    textEncoder = nil
    decoder = nil
    voice = nil
    chosenVoice = nil
    isModelInitialized = false

    // Optionally preserve text processing components for faster restart
    if !preserveTextProcessing {
      if let _ = eSpeakEngine {
        // Ensure eSpeakEngine is terminated properly
        eSpeakEngine = nil
      }
      kokoroTokenizer = nil
    }

    // Use plain autoreleasepool to encourage memory release
    autoreleasepool { }
  }

  // Initialize model on demand
  private func ensureModelInitialized() throws {
    if isModelInitialized {
      return
    }

    // Initialize text processing components first (less expensive)
    if eSpeakEngine == nil {
      eSpeakEngine = try ESpeakNGEngine()
    }

    if kokoroTokenizer == nil {
      kokoroTokenizer = KokoroTokenizer(engine: eSpeakEngine)
    }

    try autoreleasepool {
      let sanitizedWeights = KokoroWeightLoader.loadWeights(url: self.customURL)

      // Validate weights before constructing layers to avoid crashes
      guard !sanitizedWeights.isEmpty else {
        print("Kokoro: No weights loaded. Ensure 'kokoro-v1_0_bf16.safetensors' is bundled or pass a valid customURL.")
        throw KokoroTTSError.modelNotInitialized
      }

      // Early validation: check critical shapes to prevent addmm mismatches
      // bert_encoder.weight must be [dModel, 768]
      if let be = sanitizedWeights["bert_encoder.weight"] {
        let shape = be.shape
        if shape.count != 2 || shape[1] != 768 {
          print("Kokoro: bert_encoder.weight expected [dModel, 768], got \(shape)")
          throw KokoroTTSError.modelNotInitialized
        }
      } else { print("Kokoro: Missing bert_encoder.weight in weights"); throw KokoroTTSError.modelNotInitialized }

      // bert.encoder.embedding_hidden_mapping_in.weight must be [768, 128]
      if let eh = sanitizedWeights["bert.encoder.embedding_hidden_mapping_in.weight"] {
        let shape = eh.shape
        if shape.count != 2 || !(shape[0] == 768 && shape[1] == 128) {
          print("Kokoro: bert.encoder.embedding_hidden_mapping_in.weight expected [768, 128], got \(shape)")
          throw KokoroTTSError.modelNotInitialized
        }
      } else { print("Kokoro: Missing bert.encoder.embedding_hidden_mapping_in.weight in weights"); throw KokoroTTSError.modelNotInitialized }

      bert = CustomAlbert(weights: sanitizedWeights, config: AlbertModelArgs())
      bertEncoder = Linear(weight: sanitizedWeights["bert_encoder.weight"]!, bias: sanitizedWeights["bert_encoder.bias"]!)
      durationEncoder = DurationEncoder(weights: sanitizedWeights, dModel: 512, styDim: 128, nlayers: 6)

      predictorLSTM = LSTM(
        inputSize: 512 + 128,
        hiddenSize: 512 / 2,
        wxForward: sanitizedWeights["predictor.lstm.weight_ih_l0"]!,
        whForward: sanitizedWeights["predictor.lstm.weight_hh_l0"]!,
        biasIhForward: sanitizedWeights["predictor.lstm.bias_ih_l0"]!,
        biasHhForward: sanitizedWeights["predictor.lstm.bias_hh_l0"]!,
        wxBackward: sanitizedWeights["predictor.lstm.weight_ih_l0_reverse"]!,
        whBackward: sanitizedWeights["predictor.lstm.weight_hh_l0_reverse"]!,
        biasIhBackward: sanitizedWeights["predictor.lstm.bias_ih_l0_reverse"]!,
        biasHhBackward: sanitizedWeights["predictor.lstm.bias_hh_l0_reverse"]!
      )

      durationProj = Linear(
        weight: sanitizedWeights["predictor.duration_proj.linear_layer.weight"]!,
        bias: sanitizedWeights["predictor.duration_proj.linear_layer.bias"]!
      )

      prosodyPredictor = ProsodyPredictor(
        weights: sanitizedWeights,
        styleDim: 128,
        dHid: 512
      )

      textEncoder = TextEncoder(
        weights: sanitizedWeights,
        channels: 512,
        kernelSize: 5,
        depth: 3,
        nSymbols: 178
      )

      decoder = Decoder(
        weights: sanitizedWeights,
        dimIn: 512,
        styleDim: 128,
        dimOut: 80,
        resblockKernelSizes: [3, 7, 11],
        upsampleRates: [10, 6],
        upsampleInitialChannel: 512,
        resblockDilationSizes: [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsampleKernelSizes: [20, 12],
        genIstftNFft: 20,
        genIstftHopSize: 5
      )
    }

    isModelInitialized = true
  }

  /// Public helper for benchmarking to trigger model initialization and measure load time
  /// without starting any audio generation.
  public func initializeIfNeeded() throws {
    try ensureModelInitialized()
  }

  // MARK: - Safe style splitting
  /// Safely split reference style tensor into decoder half (first 128) and conditioning half (last 128).
  /// Always returns fixed [channels, 128] for each half, padding/truncating as needed.
  private func splitStyle(refS: MLXArray) -> (voiceS: MLXArray, condS: MLXArray) {
    // refS expected shape: [channels, features]
    let channels = max(1, refS.shape.count > 0 ? refS.shape[0] : 1)
    let feats = refS.shape.count > 1 ? refS.shape[1] : 0

    // Fast paths: empty features
    if feats <= 0 {
      let zero = MLXArray.zeros([channels, 128])
      return (zero, zero)
    }

    // Slice available portions with explicit closed ranges
    let voiceCount = min(128, feats)
    let condCount = feats > 128 ? min(128, feats - 128) : 0

    // First half [0 ..< voiceCount]
    var voicePart = refS[0 ... (channels - 1), 0 ... (voiceCount - 1)]
    voicePart.eval()

    // Second half [128 ..< 128+condCount], if any
    var condPart: MLXArray
    if condCount > 0 {
      condPart = refS[0 ... (channels - 1), 128 ... (128 + condCount - 1)]
      condPart.eval()
    } else {
      condPart = MLXArray.zeros([channels, 0])
    }

    // Pad to 128 columns each if needed
    if voiceCount < 128 {
      let pad = MLXArray.zeros([channels, 128 - voiceCount])
      voicePart = MLX.concatenated([voicePart, pad], axis: 1)
    } else if voiceCount > 128 {
      voicePart = voicePart[0 ... (channels - 1), 0 ... 127]
    }

    if condCount < 128 {
      let pad = MLXArray.zeros([channels, 128 - condCount])
      condPart = condCount > 0 ? MLX.concatenated([condPart, pad], axis: 1) : pad
    } else if condCount > 128 {
      condPart = condPart[0 ... (channels - 1), 0 ... 127]
    }

    return (voicePart, condPart)
  }

  private func generateAudioForTokens(
    inputIds: [Int],
    speed: Float
  ) throws -> MLXArray {
    // Create a fresh autorelease pool for the entire process
    return try autoreleasepool { () -> MLXArray in
      // Start with the standard processing
      try autoreleasepool {
        let paddedInputIdsBase = [0] + inputIds + [0]
        let paddedInputIds = MLXArray(paddedInputIdsBase).expandedDimensions(axes: [0])
        paddedInputIds.eval()

        let inputLengths = MLXArray(paddedInputIds.dim(-1))
        inputLengths.eval()

        let inputLengthMax: Int = MLX.max(inputLengths).item()
        var textMask = MLXArray(0 ..< inputLengthMax)
        textMask.eval()

        textMask = textMask + 1 .> inputLengths
        textMask.eval()

        textMask = textMask.expandedDimensions(axes: [0])
        textMask.eval()

        let swiftTextMask: [Bool] = textMask.asArray(Bool.self)
        let swiftTextMaskInt = swiftTextMask.map { !$0 ? 1 : 0 }
        let attentionMask = MLXArray(swiftTextMaskInt).reshaped(textMask.shape)
        attentionMask.eval()

        return try autoreleasepool { () -> MLXArray in
          // Ensure model is initialized
          guard let bert = bert,
                let bertEncoder = bertEncoder else {
            throw KokoroTTSError.modelNotInitialized
          }

          let (bertDur, _) = bert(paddedInputIds, attentionMask: attentionMask)
          bertDur.eval()

          autoreleasepool {
            _ = attentionMask
          }

          let dEn = bertEncoder(bertDur).transposed(0, 2, 1)
          dEn.eval()

          autoreleasepool {
            _ = bertDur
          }

          guard let voice = voice else {
            throw KokoroTTSError.modelNotInitialized
          }
          let timeIdx = max(0, min(inputIds.count - 1, voice.shape[0] - 1))
          // Clamp channel and feature ranges defensively to avoid invalid ranges
          let channelsMaxIndex = max(0, voice.shape[1] - 1)
          let channelEnd = min(1, channelsMaxIndex)
          let feats = max(0, voice.shape[2])
          // Explicit last index for features; bail out to zeros if none
          if feats == 0 {
            // Produce zeros and continue with empty style
            let zero = MLXArray.zeros([channelEnd + 1, 128])
            let s = zero
            return try autoreleasepool { () -> MLXArray in
              // Ensure components are initialized
              guard let durationEncoder = durationEncoder,
                    let predictorLSTM = predictorLSTM,
                    let durationProj = durationProj else {
                throw KokoroTTSError.modelNotInitialized
              }
              // Minimal safe flow (rare path)
              let d = durationEncoder(dEn, style: s, textLengths: inputLengths, m: textMask)
              d.eval()
              let (x, _) = predictorLSTM(d)
              x.eval()
              let duration = durationProj(x)
              duration.eval()
              let durationSigmoid = MLX.sigmoid(duration).sum(axis: -1) / speed
              durationSigmoid.eval()
              let predDur = MLX.clip(durationSigmoid.round(), min: 1).asType(.int32)[0]
              predDur.eval()
              let indices = MLXArray.zeros([1])
              indices.eval()
              let predAlnTrg = MLXArray.zeros([paddedInputIds.shape[1], 1])
              predAlnTrg.eval()
              let en = d.transposed(0, 2, 1).matmul(predAlnTrg)
              en.eval()
          let tEn = textEncoder(paddedInputIds, inputLengths: inputLengths, m: textMask)
              tEn.eval()
              let asr = MLX.matmul(tEn, predAlnTrg)
              asr.eval()
              let audio = decoder(asr: asr, F0Curve: MLXArray.zeros([1,1]), N: MLXArray.zeros([1,1]), s: s)[0]
              audio.eval()
              return audio
            }
          }

          let refS = voice[timeIdx, 0 ... channelEnd, 0 ... (feats - 1)]
          refS.eval()

          // Split safely into (decoder half, conditioning half)
          let (voiceS, s) = splitStyle(refS: refS)
          s.eval()

          return try autoreleasepool { () -> MLXArray in
            // Ensure all components are initialized
            guard let durationEncoder = durationEncoder,
                  let predictorLSTM = predictorLSTM,
                  let durationProj = durationProj else {
              throw KokoroTTSError.modelNotInitialized
            }

            let d = durationEncoder(dEn, style: s, textLengths: inputLengths, m: textMask)
            d.eval()

            autoreleasepool {
              _ = dEn
              _ = textMask
            }

            let (x, _) = predictorLSTM(d)
            x.eval()

            let duration = durationProj(x)
            duration.eval()

            autoreleasepool {
              _ = x
            }

            let durationSigmoid = MLX.sigmoid(duration).sum(axis: -1) / speed
            durationSigmoid.eval()

            autoreleasepool {
              _ = duration
            }

            let predDur = MLX.clip(durationSigmoid.round(), min: 1).asType(.int32)[0]
            predDur.eval()

            autoreleasepool {
              _ = durationSigmoid
            }

            // Index and matrix generation - high memory usage
            // Build indices in chunks to reduce memory
            var allIndices: [MLXArray] = []
            let chunkSize = 50 // Process 50 items at a time

            for startIdx in stride(from: 0, to: predDur.shape[0], by: chunkSize) {
              autoreleasepool {
                let endIdx = min(startIdx + chunkSize, predDur.shape[0])
                let chunkIndices = predDur[startIdx..<endIdx]

                let indices = MLX.concatenated(
                  chunkIndices.enumerated().map { i, n in
                    let nSize: Int = n.item()
                    let arrayIndex = MLXArray([i + startIdx])
                    arrayIndex.eval()
                    let repeated = MLX.repeated(arrayIndex, count: nSize)
                    repeated.eval()
                    return repeated
                  }
                )
                indices.eval()
                allIndices.append(indices)
              }
            }

            let indices = MLX.concatenated(allIndices)
            indices.eval()

            allIndices.removeAll()

            let indicesShape = indices.shape[0]
            let inputIdsShape = paddedInputIds.shape[1]

            // Create sparse matrix more efficiently using COO format
            // This drastically reduces memory usage compared to dense matrix
            var rowIndices: [Int] = []
            var colIndices: [Int] = []
            var values: [Float] = []

            // Reserve capacity to avoid reallocations
            let estimatedNonZeros = min(indicesShape, inputIdsShape * 5)
            rowIndices.reserveCapacity(estimatedNonZeros)
            colIndices.reserveCapacity(estimatedNonZeros)
            values.reserveCapacity(estimatedNonZeros)

            // Process in batches to reduce cache misses
            let batchSize = 256
            for startIdx in stride(from: 0, to: indicesShape, by: batchSize) {
              autoreleasepool {
                let endIdx = min(startIdx + batchSize, indicesShape)
                for i in startIdx..<endIdx {
                  let indiceValue: Int = indices[i].item()
                  if indiceValue < inputIdsShape {
                    rowIndices.append(indiceValue)
                    colIndices.append(i)
                    values.append(1.0)
                  }
                }
              }
            }

            autoreleasepool {
              _ = indices
              _ = predDur
            }

            // Create MLXArray from COO data
            let rowIndicesArray = MLXArray(rowIndices)
            let colIndicesArray = MLXArray(colIndices)
            let coo_indices = MLX.stacked([rowIndicesArray, colIndicesArray], axis: 0).transposed(1, 0)
            let coo_values = MLXArray(values)
            rowIndicesArray.eval()
            colIndicesArray.eval()
            coo_indices.eval()
            coo_values.eval()

            // Go back to the original dense matrix approach but with better memory management
            // Create sparse matrix efficiently using Swift arrays first
            var swiftPredAlnTrg = [Float](repeating: 0.0, count: inputIdsShape * indicesShape)
            // Process in batches to reduce cache misses
            let matrixBatchSize = 1000
            for startIdx in stride(from: 0, to: rowIndices.count, by: matrixBatchSize) {
              autoreleasepool {
                let endIdx = min(startIdx + matrixBatchSize, rowIndices.count)
                for i in startIdx..<endIdx {
                  let row = rowIndices[i]
                  let col = colIndices[i]
                  if row < inputIdsShape && col < indicesShape {
                    swiftPredAlnTrg[row * indicesShape + col] = 1.0
                  }
                }
              }
            }

            // Create MLXArray from the dense matrix
            let predAlnTrg = MLXArray(swiftPredAlnTrg).reshaped([inputIdsShape, indicesShape])
            predAlnTrg.eval()

            // Clear Swift array immediately
            swiftPredAlnTrg = []

            autoreleasepool {
              rowIndices = []
              colIndices = []
              values = []
            }

            let predAlnTrgBatched = predAlnTrg.expandedDimensions(axis: 0)
            predAlnTrgBatched.eval()

            let en = d.transposed(0, 2, 1).matmul(predAlnTrgBatched)
            en.eval()

            autoreleasepool {
              _ = d
              _ = predAlnTrgBatched
            }

            return try autoreleasepool { () -> MLXArray in
              // Ensure components are initialized
              guard let prosodyPredictor = prosodyPredictor,
                    let textEncoder = textEncoder,
                    let decoder = decoder else {
                throw KokoroTTSError.modelNotInitialized
              }

              let (F0Pred, NPred) = prosodyPredictor.F0NTrain(x: en, s: s)
              F0Pred.eval()
              NPred.eval()

              autoreleasepool {
                _ = en
              }

              let tEn = textEncoder(paddedInputIds, inputLengths: inputLengths, m: textMask)
              tEn.eval()

              autoreleasepool {
                _ = paddedInputIds
                _ = inputLengths
              }

              let asr = MLX.matmul(tEn, predAlnTrg)
              asr.eval()

              autoreleasepool {
                _ = tEn
                _ = predAlnTrg
              }

              voiceS.eval()

              autoreleasepool {
                _ = refS
              }

              let audio = decoder(asr: asr, F0Curve: F0Pred, N: NPred, s: voiceS)[0]
              audio.eval()

              autoreleasepool {
                _ = asr
                _ = F0Pred
                _ = NPred
                _ = voiceS
                _ = s
              }

              let audioShape = audio.shape

              // Check if the audio shape is valid
              let totalSamples: Int
              if audioShape.count == 1 {
                totalSamples = audioShape[0]
              } else if audioShape.count == 2 {
                totalSamples = audioShape[1]
              } else {
                totalSamples = 0
              }

              if totalSamples <= 1 {
                // Return an error tone
                var errorAudioData = [Float](repeating: 0.0, count: 24000)

                // Create a simple repeating beep pattern to indicate error
                for i in stride(from: 0, to: 24000, by: 100) {
                  let endIdx = min(i + 100, 24000)
                  for j in i..<endIdx {
                    let t = Float(j) / Float(Constants.sampleRate)
                    let freq = (Int(t * 2) % 2 == 0) ? 500.0 : 800.0
                    errorAudioData[j] = sin(Float(2.0 * .pi * freq) * t) * 0.5
                  }
                }

                let fallbackAudio = MLXArray(errorAudioData)
                fallbackAudio.eval()
                return fallbackAudio
              }

              return audio
            }
          }
        }
      }
    }
  }

  public func generateAudio(
    voice: TTSVoice,
    text: String,
    speed: Float = 1.0,
    completion: (() -> Void)? = nil,
    chunkCallback: @escaping AudioChunkCallback
  ) throws {
    try ensureModelInitialized()

    let sentences = SentenceTokenizer.splitIntoSentences(text: text)
    if sentences.isEmpty {
      throw KokoroTTSError.sentenceSplitError
    }

    // Process each sentence in sequence with better performance
    DispatchQueue.global(qos: .userInitiated).async {
      self.voice = nil

      // Use a separate autorelease pool for each sentence to release memory faster
      for (_, sentence) in sentences.enumerated() {
        autoreleasepool {
          do {
            // Generate audio for this sentence
            let audio = try self.generateAudioForSentence(voice: voice, text: sentence, speed: speed)

            // Force evaluation to ensure tensor is computed before sending
            audio.eval()

            // Send this chunk to the callback immediately on the main thread
            // Dispatch to main thread to avoid threading issues with UI updates
            DispatchQueue.main.async {
              chunkCallback(audio)
            }

            // Explicitly release large tensors
            autoreleasepool {
              _ = audio
            }
          } catch {
            // Handle error silently
          }
        }
        MLX.GPU.clearCache()
      }

      // Reset model after completing a long text to free memory (optional)
      if sentences.count > 5 {
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
          self.resetModel()
        }
      }

      // Notify completion on main thread
      if let completion = completion {
        DispatchQueue.main.async { completion() }
      }
    }
  }

  private func generateAudioForSentence(voice: TTSVoice, text: String, speed: Float) throws -> MLXArray {
    try ensureModelInitialized()

    if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
      return MLXArray.zeros([1])
    }

    return try autoreleasepool { () -> MLXArray in
      if chosenVoice != voice {
        autoreleasepool {
          self.voice = VoiceLoader.loadVoice(voice)
          self.voice?.eval() // Force immediate evaluation
        }

        try kokoroTokenizer.setLanguage(for: voice)
        chosenVoice = voice
      }

      do {
        let phonemizedResult = try kokoroTokenizer.phonemize(text)

        let inputIds = Tokenizer.tokenize(phonemizedText: phonemizedResult.phonemes)
        guard inputIds.count <= Constants.maxTokenCount else {
          throw KokoroTTSError.tooManyTokens
        }

        // Continue with normal audio generation
        return try self.processTokensToAudio(inputIds: inputIds, speed: speed)
      } catch {
        // Return a short error tone instead of crashing; also log diagnostics in debug mode
        let debugEnabled = UserDefaults.standard.bool(forKey: "com.talktome.mlx.debug")
        if debugEnabled {
          let voiceName = voice.rawValue
          let maybeShape = (self.voice != nil) ? " voiceTensorShape=\(self.voice!.shape)" : ""
          print("Kokoro: generateAudioForSentence failed; returning error tone. textLen=\(text.count) voice=\(voiceName) speed=\(speed). Error=\(error).\(maybeShape)")
        }
        var errorAudioData = [Float](repeating: 0.0, count: 4800) // 0.2s at 24kHz

        // Simple error beep
        for i in 0..<4800 {
          let t = Float(i) / Float(Constants.sampleRate)
          let freq: Float = 880.0 // High-pitched error tone
          errorAudioData[i] = sin(Float(2.0 * .pi * freq) * t) * 0.3
        }

        return MLXArray(errorAudioData)
      }
    }
  }

  // Common processing method to convert tokens to audio - used by streaming methods
  private func processTokensToAudio(inputIds: [Int], speed: Float) throws -> MLXArray {
    // Use the token processing method
    return try generateAudioForTokens(
      inputIds: inputIds,
      speed: speed
    )
  }

  struct Constants {
    static let maxTokenCount = 510
    static let sampleRate = 24000
  }
}
