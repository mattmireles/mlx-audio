//
//  Kokoro-tts-lib
//
import Foundation
import MLX
import MLXNN

// Utility class for loading and preprocessing the weights for the model
class KokoroWeightLoader {
    private init() {}
    
    /// Debug toggle key to print model selection and critical tensor shapes
    private static let debugDefaultsKey = "com.talktome.mlx.debug"
    private static var didPrintDebugOnce: Bool = false
    
    static func loadWeights(url: URL? = nil) -> [String: MLXArray] {
        // Resolve the model URL, preferring a provided custom URL
        let resolvedURL: URL? = {
            if let explicitURL = url { return explicitURL }
            #if SWIFT_PACKAGE
            return Bundle.module.url(forResource: "kokoro-v1_0_bf16", withExtension: "safetensors")
            #else
            return Bundle.main.url(forResource: "kokoro-v1_0_bf16", withExtension: "safetensors")
            #endif
        }()
        
        var modelURLOpt = resolvedURL
        // Fallback: look directly in the app bundle resource path (in case not indexed)
        if modelURLOpt == nil, let resPath = Bundle.main.resourcePath {
            let fallbackPath = (resPath as NSString).appendingPathComponent("kokoro-v1_0_bf16.safetensors")
            if FileManager.default.fileExists(atPath: fallbackPath) {
                modelURLOpt = URL(fileURLWithPath: fallbackPath)
            }
        }
        guard let modelURL = modelURLOpt else {
            print("Kokoro: Weights file 'kokoro-v1_0_bf16.safetensors' not found in bundle. Provide a customURL to KokoroTTS or add the file to package resources.")
            return [:]
        }
        
        do {
            let weights = try MLX.loadArrays(url: modelURL)
            var sanitizedWeights: [String: MLXArray] = [:]
            let debugEnabled = UserDefaults.standard.bool(forKey: debugDefaultsKey)
            if debugEnabled && !Self.didPrintDebugOnce {
                print("Kokoro: Using weights file -> \(modelURL.lastPathComponent)")
            }
            
            for (key, value) in weights {
                if key.hasPrefix("bert") {
                    if key.contains("position_ids") {
                        continue
                    }
                    sanitizedWeights[key] = value
                } else if key.hasPrefix("predictor") {
                    if key.contains("F0_proj.weight") {
                        sanitizedWeights[key] = value.transposed(0, 2, 1)
                    } else if key.contains("N_proj.weight") {
                        sanitizedWeights[key] = value.transposed(0, 2, 1)
                    } else if key.contains("weight_v") {
                        if checkArrayShape(arr: value) {
                            sanitizedWeights[key] = value
                        } else {
                            sanitizedWeights[key] = value.transposed(0, 2, 1)
                        }
                    } else {
                        sanitizedWeights[key] = value
                    }
                } else if key.hasPrefix("text_encoder") {
                    if key.contains("weight_v") {
                        if checkArrayShape(arr: value) {
                            sanitizedWeights[key] = value
                        } else {
                            sanitizedWeights[key] = value.transposed(0, 2, 1)
                        }
                    } else {
                        sanitizedWeights[key] = value
                    }
                } else if key.hasPrefix("decoder") {
                    if key.contains("noise_convs"), key.hasSuffix(".weight") {
                        sanitizedWeights[key] = value.transposed(0, 2, 1)
                    } else if key.contains("weight_v") {
                        if checkArrayShape(arr: value) {
                            sanitizedWeights[key] = value
                        } else {
                            sanitizedWeights[key] = value.transposed(0, 2, 1)
                        }
                    } else {
                        sanitizedWeights[key] = value
                    }
                }
            }

            // Key aliasing to handle naming variants across checkpoints
            func alias(_ target: String, from candidates: [String]) {
                if sanitizedWeights[target] == nil {
                    for c in candidates {
                        if let v = sanitizedWeights[c] {
                            sanitizedWeights[target] = v
                            break
                        }
                    }
                }
            }

            // Alias predictor.lstm keys (forward/backward)
            alias("predictor.lstm.weight_ih_l0", from: [
                "predictor.lstm.Wx_forward", "predictor.lstm.weight_ih_forward"
            ])
            alias("predictor.lstm.weight_hh_l0", from: [
                "predictor.lstm.Wh_forward", "predictor.lstm.weight_hh_forward"
            ])
            alias("predictor.lstm.bias_ih_l0", from: [
                "predictor.lstm.bias_ih_forward"
            ])
            alias("predictor.lstm.bias_hh_l0", from: [
                "predictor.lstm.bias_hh_forward"
            ])
            alias("predictor.lstm.weight_ih_l0_reverse", from: [
                "predictor.lstm.Wx_backward", "predictor.lstm.weight_ih_backward"
            ])
            alias("predictor.lstm.weight_hh_l0_reverse", from: [
                "predictor.lstm.Wh_backward", "predictor.lstm.weight_hh_backward"
            ])
            alias("predictor.lstm.bias_ih_l0_reverse", from: [
                "predictor.lstm.bias_ih_backward"
            ])
            alias("predictor.lstm.bias_hh_l0_reverse", from: [
                "predictor.lstm.bias_hh_backward"
            ])

            // Alias duration encoder text encoder lstm keys for each layer
            for i in 0..<6 {
                let base = "predictor.text_encoder.lstms.\(i)"
                alias("\(base).weight_ih_l0", from: ["\(base).Wx_forward", "\(base).weight_ih_forward"]) 
                alias("\(base).weight_hh_l0", from: ["\(base).Wh_forward", "\(base).weight_hh_forward"]) 
                alias("\(base).bias_ih_l0", from: ["\(base).bias_ih_forward"]) 
                alias("\(base).bias_hh_l0", from: ["\(base).bias_hh_forward"]) 
                alias("\(base).weight_ih_l0_reverse", from: ["\(base).Wx_backward", "\(base).weight_ih_backward"]) 
                alias("\(base).weight_hh_l0_reverse", from: ["\(base).Wh_backward", "\(base).weight_hh_backward"]) 
                alias("\(base).bias_ih_l0_reverse", from: ["\(base).bias_ih_backward"]) 
                alias("\(base).bias_hh_l0_reverse", from: ["\(base).bias_hh_backward"]) 
            }

            // Alias shared LSTM keys
            alias("predictor.shared.weight_ih_l0", from: ["predictor.shared.Wx_forward", "predictor.shared.weight_ih_forward"]) 
            alias("predictor.shared.weight_hh_l0", from: ["predictor.shared.Wh_forward", "predictor.shared.weight_hh_forward"]) 
            alias("predictor.shared.bias_ih_l0", from: ["predictor.shared.bias_ih_forward"]) 
            alias("predictor.shared.bias_hh_l0", from: ["predictor.shared.bias_hh_forward"]) 
            alias("predictor.shared.weight_ih_l0_reverse", from: ["predictor.shared.Wx_backward", "predictor.shared.weight_ih_backward"]) 
            alias("predictor.shared.weight_hh_l0_reverse", from: ["predictor.shared.Wh_backward", "predictor.shared.weight_hh_backward"]) 
            alias("predictor.shared.bias_ih_l0_reverse", from: ["predictor.shared.bias_ih_backward"]) 
            alias("predictor.shared.bias_hh_l0_reverse", from: ["predictor.shared.bias_hh_backward"]) 

            // Alias conv1x1 naming variants
            alias("predictor.F0_proj.weight", from: ["predictor.F0_proj.linear_layer.weight"]) 
            alias("predictor.F0_proj.bias", from: ["predictor.F0_proj.linear_layer.bias"]) 
            alias("predictor.N_proj.weight", from: ["predictor.N_proj.linear_layer.weight"]) 
            alias("predictor.N_proj.bias", from: ["predictor.N_proj.linear_layer.bias"]) 

            // Alias text_encoder layernorm gamma/beta vs weight/bias
            for i in 0..<3 {
                alias("text_encoder.cnn.\(i).1.gamma", from: ["text_encoder.cnn.\(i).1.weight"]) 
                alias("text_encoder.cnn.\(i).1.beta", from: ["text_encoder.cnn.\(i).1.bias"]) 
            }

            // Alias text_encoder LSTM keys
            alias("text_encoder.lstm.weight_ih_l0", from: ["text_encoder.lstm.Wx_forward", "text_encoder.lstm.weight_ih_forward"]) 
            alias("text_encoder.lstm.weight_hh_l0", from: ["text_encoder.lstm.Wh_forward", "text_encoder.lstm.weight_hh_forward"]) 
            alias("text_encoder.lstm.bias_ih_l0", from: ["text_encoder.lstm.bias_ih_forward"]) 
            alias("text_encoder.lstm.bias_hh_l0", from: ["text_encoder.lstm.bias_hh_forward"]) 
            alias("text_encoder.lstm.weight_ih_l0_reverse", from: ["text_encoder.lstm.Wx_backward", "text_encoder.lstm.weight_ih_backward"]) 
            alias("text_encoder.lstm.weight_hh_l0_reverse", from: ["text_encoder.lstm.Wh_backward", "text_encoder.lstm.weight_hh_backward"]) 
            alias("text_encoder.lstm.bias_ih_l0_reverse", from: ["text_encoder.lstm.bias_ih_backward"]) 
            alias("text_encoder.lstm.bias_hh_l0_reverse", from: ["text_encoder.lstm.bias_hh_backward"]) 
            
            // --- Critical linear weight orientation validation/fix ---
            // Expected core dimensions from architecture
            let hiddenSize = 768  // AlbertModelArgs.hiddenSize default
            let embeddingSize = 128  // ALBERT embedding size expected by embedding_hidden_mapping_in

            // bert_encoder: maps hidden (768) -> dModel; Linear expects [out, in] = [dModel, hidden]
            if let w = sanitizedWeights["bert_encoder.weight"] {
              if w.shape.count != 2 {
                print("Kokoro: Invalid bert_encoder.weight rank: \(w.shape)"); return [:]
              }
              if w.shape[1] == hiddenSize {
                // OK: already [dModel, 768]
              } else if w.shape[0] == hiddenSize {
                sanitizedWeights["bert_encoder.weight"] = w.transposed(1, 0)
                if debugEnabled && !Self.didPrintDebugOnce {
                  print("Kokoro: Transposed bert_encoder.weight to [dModel, \(hiddenSize)]")
                }
              } else {
                print("Kokoro: Unexpected bert_encoder.weight shape: \(w.shape). Need second dim == \(hiddenSize)"); return [:]
              }
            } else { print("Kokoro: Missing bert_encoder.weight"); return [:] }

            // embedding_hidden_mapping_in: maps 128 -> 768; Linear expects [out, in] = [768, 128]
            if let w = sanitizedWeights["bert.encoder.embedding_hidden_mapping_in.weight"] {
              if w.shape.count != 2 {
                print("Kokoro: Invalid rank for bert.encoder.embedding_hidden_mapping_in.weight: \(w.shape). Expected rank-2")
                return [:]
              }
              if w.shape[0] == hiddenSize && w.shape[1] == embeddingSize {
                // OK
              } else if w.shape[0] == embeddingSize && w.shape[1] == hiddenSize {
                sanitizedWeights["bert.encoder.embedding_hidden_mapping_in.weight"] = w.transposed(1, 0)
                if debugEnabled && !Self.didPrintDebugOnce {
                  print("Kokoro: Transposed bert.encoder.embedding_hidden_mapping_in.weight to [\(hiddenSize), \(embeddingSize)]")
                }
              } else {
                print("Kokoro: Unexpected shape for bert.encoder.embedding_hidden_mapping_in.weight: \(w.shape). Expected [\(hiddenSize), \(embeddingSize)] or [\(embeddingSize), \(hiddenSize)].")
                return [:]
              }
            } else { print("Kokoro: Missing bert.encoder.embedding_hidden_mapping_in.weight in weights"); return [:] }

            // Pooler remains square [hidden, hidden] → leave as-is

            // Optional one-time debug dump of critical shapes
            if debugEnabled && !Self.didPrintDebugOnce {
                func shapeString(_ key: String) -> String {
                    if let a = sanitizedWeights[key] { return "\(a.shape)" } else { return "<missing>" }
                }
                let keysToReport = [
                    "bert_encoder.weight",
                    "bert_encoder.bias",
                    "bert.pooler.weight",
                    "bert.pooler.bias",
                    "bert.encoder.embedding_hidden_mapping_in.weight",
                    "bert.encoder.embedding_hidden_mapping_in.bias"
                ]
                print("Kokoro: Critical parameter shapes →")
                for k in keysToReport {
                    print("  - \(k): \(shapeString(k))")
                }
                // Print a sample of all available parameter keys and shapes (first 40)
                let allKeys = sanitizedWeights.keys.sorted()
                let sampleCount = min(40, allKeys.count)
                if sampleCount > 0 {
                    print("Kokoro: Sample of parameter shapes (\(sampleCount)/\(allKeys.count)) →")
                    for i in 0..<sampleCount {
                        let k = allKeys[i]
                        if let arr = sanitizedWeights[k] {
                            print("  - \(k): \(arr.shape)")
                        }
                    }
                }
                Self.didPrintDebugOnce = true
            }

            return sanitizedWeights
        } catch {
            print("Kokoro: Error loading weights: \(error)")
            return [:]
        }
    }
    
    private static func checkArrayShape(arr: MLXArray) -> Bool {
        guard arr.shape.count != 3 else { return false }
        
        let outChannels = arr.shape[0]
        let kH = arr.shape[1]
        let kW = arr.shape[2]
        
        return (outChannels >= kH) && (outChannels >= kW) && (kH == kW)
    }
}
