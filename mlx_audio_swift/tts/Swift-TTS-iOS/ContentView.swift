//
//  ContentView.swift
//   Swift-TTS-iOS
//
//  Created by Sachin Desai on 5/17/25.
//

import SwiftUI
import MLX
import Darwin
import Darwin.Mach

struct ContentView: View {
    @State private var speed = 1.0
    @State public var text = ""
    @State private var showAlert = false
  
    @FocusState private var isTextEditorFocused: Bool
    @ObservedObject var viewModel: KokoroTTSModel
    @StateObject private var speakerModel = SpeakerViewModel()
    @State private var showBenchmark = false
    
    var body: some View {
        NavigationStack {
            ZStack {
                backgroundView
                
                ScrollView(showsIndicators: false) {
                    VStack(spacing: 16) {
                        // Model selector (bf16 vs 8-bit)
                        modelSelectorView
                        VStack(alignment: .leading, spacing: 8) {
                            HStack(spacing: 12) {
                                compactSpeakerView(
                                    selectedSpeakerId: $speakerModel.selectedSpeakerId,
                                    title: "Speaker"
                                )
                                .frame(maxWidth: .infinity)
                            }
                        }
                        
                        speedControlView
                        textInputView
                        
                        actionButtonsView
                    }
                    .padding([.horizontal, .bottom])
                }
                .toolbar {
                    ToolbarItem(placement: .principal) {
                        VStack(spacing: 0) {
                            HStack {
                                Text("Kokoro")
                                    .font(.title)
                            }
                            Text("\(viewModel.currentModelDescription) • TTFA: \(viewModel.audioGenerationTime > 0 ? String(format: "%.2f", viewModel.audioGenerationTime) : "--")s")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        }
                    }
                    ToolbarItem(placement: .topBarTrailing) {
                        Button("Benchmark") { showBenchmark = true }
                    }
                }
                .scrollContentBackground(.hidden)
                .alert("Empty Text", isPresented: $showAlert) {
                    Button("OK", role: .cancel) { }
                } message: {
                    Text("Please enter some text before generating audio.")
                }
                .contentShape(Rectangle())
                .onTapGesture {
                    if isTextEditorFocused {
                        dismissKeyboard()
                        isTextEditorFocused = false
                    }
                }
            }
        }
        .sheet(isPresented: $showBenchmark) {
            BenchmarkView(ttsModel: viewModel)
        }
        // Sync viewModel.generationInProgress to speakerModel.isGenerating
        .onChange(of: viewModel.generationInProgress) { _, newValue in
            speakerModel.isGenerating = newValue
        }
    }
    
    private var backgroundView: some View {
        Color(.systemBackground)
            .ignoresSafeArea()
    }
    
    private func compactSpeakerView(selectedSpeakerId: Binding<Int>, title: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.subheadline)
                .foregroundStyle(.secondary)
            
            Menu {
                ForEach(speakerModel.speakers) { speaker in
                    Button(action: {
                        selectedSpeakerId.wrappedValue = speaker.id
                    }) {
                        HStack {
                            Text("\(speaker.flag) \(speaker.displayName)")
                            if selectedSpeakerId.wrappedValue == speaker.id {
                                Image(systemName: "checkmark")
                            }
                        }
                    }
                }
            } label: {
                HStack {
                    if let speaker = speakerModel.getSpeaker(id: selectedSpeakerId.wrappedValue) {
                        Text(speaker.flag)
                        Text(speaker.displayName)
                            .lineLimit(1)
                            .foregroundStyle(.primary)
                    }
                    Spacer()
                    Image(systemName: "chevron.down")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding(8)
                .background(
                    RoundedRectangle(cornerRadius: 8)
                        .fill(Color(.tertiarySystemBackground))
                )
            }
            .disabled(viewModel.generationInProgress)
        }
    }
    
    private var speedControlView: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Speed")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                Spacer()
                Text(String(format: "%.1fx", speed))
                    .font(.subheadline)
                    .bold()
            }
            
            Slider(value: $speed, in: 0.5...2.0, step: 0.1)
                .tint(.accentColor)
                .disabled(viewModel.generationInProgress)
        }
    }
    
    private var textInputView: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Text Input")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
            
            ZStack(alignment: .topLeading) {
                TextEditor(text: $text)
                    .font(.body)
                    .frame(minHeight: 120)
                    .scrollContentBackground(.hidden)
                    .padding()
                    .background(
                        RoundedRectangle(cornerRadius: 12)
                            .fill(Color(.tertiarySystemBackground))
                    )
                    .focused($isTextEditorFocused)
                    .disabled(viewModel.generationInProgress)
                    .onTapGesture {
                        // Explicitly focus the text editor when tapped
                        if !isTextEditorFocused && !viewModel.generationInProgress {
                            isTextEditorFocused = true
                        }
                    }
                
                if text.isEmpty {
                    Text("Enter your text here...")
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 20)
                        .padding(.top, 25)
                        .allowsHitTesting(false)
                }
            }
        }
    }
    
    private var actionButtonsView: some View {
        HStack(spacing: 12) {
            // generatge button
            Button {
                if isTextEditorFocused {
                    dismissKeyboard()
                    isTextEditorFocused = false
                }
                
                // Prepare text and speaker
                let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
                let speaker = speakerModel.getPrimarySpeaker().first!
                
                // Set memory constraints for MLX and start generation
                MLX.GPU.set(cacheLimit: 20 * 1024 * 1024)
                viewModel.say(t, TTSVoice.fromIdentifier(speaker.name) ?? .afHeart, speed: Float(speed))
            } label: {
                HStack {
                    if viewModel.generationInProgress {
                        ProgressView()
                            .controlSize(.small)
                        Text("Generating...")
                    } else {
                        Text("Generate")
                    }
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 10)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.regular)
            .frame(maxWidth: .infinity, minHeight: 44)
            .disabled(viewModel.generationInProgress || text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            
            // Stop button
            Button {
                viewModel.stopPlayback()
            } label: {
                HStack {
                    Image(systemName: "stop.fill")
                    Text("Stop")
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 10)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.regular)
            .frame(maxWidth: .infinity, minHeight: 44)
            .tint(.red)
            .disabled(!viewModel.isAudioPlaying)
        }
    }

    // MARK: - Model selector
    @State private var selectedModelIndex: Int = 0
    private let modelOptions: [(title: String, resource: String)] = [
        ("Kokoro 82M (bf16)", "kokoro-v1_0_bf16"),
        ("Kokoro 82M (8-bit)", "kokoro-v1_0_8bit")
    ]

    private var modelSelectorView: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Model")
                .font(.subheadline)
                .foregroundStyle(.secondary)
            Picker("Model", selection: $selectedModelIndex) {
                ForEach(0..<modelOptions.count, id: \.self) { i in
                    Text(modelOptions[i].title).tag(i)
                }
            }
            .pickerStyle(.segmented)
            .onChange(of: selectedModelIndex) { _, newValue in
                let resName = modelOptions[newValue].resource
                let url = Bundle.main.url(forResource: resName, withExtension: "safetensors")
                viewModel.switchModel(weightsURL: url)
                // Warm reset of MLX caches between model switches
                MLX.GPU.clearCache()
            }
        }
    }
}

// MARK: - Benchmarking types and view (embedded)

struct BenchmarkResult: Codable {
    let modelName: String
    let modelDescription: String
    let text: String
    let voice: String
    let modelLoadTime: Double
    let inferenceTime: Double
    let totalTime: Double
    let timeToFirstAudio: Double
    let perSentenceInference: [Double]
    let audioDuration: Double
    let realTimeFactor: Double
    let samplesPerSec: Double
    let peakMemoryGB: Double
    let sampleRate: Int
}

final class Stopwatch {
    private var startTime: DispatchTime?
    func start() { startTime = DispatchTime.now() }
    func elapsed() -> Double {
        guard let s = startTime else { return 0 }
        let ns = DispatchTime.now().uptimeNanoseconds &- s.uptimeNanoseconds
        return Double(ns) / 1_000_000_000.0
    }
}

final class MemorySampler {
    private var timer: Timer?
    private(set) var peakGB: Double = 0

    func start(intervalSec: TimeInterval = 0.2) {
        stop()
        peakGB = 0
        DispatchQueue.main.async {
            self.timer = Timer.scheduledTimer(withTimeInterval: intervalSec, repeats: true) { _ in
                let rss = MemorySampler.currentResidentBytes()
                self.peakGB = max(self.peakGB, Double(rss) / (1024.0 * 1024.0 * 1024.0))
            }
            if let t = self.timer { RunLoop.current.add(t, forMode: .common) }
        }
    }

    func stop() { timer?.invalidate(); timer = nil }

    private static func currentResidentBytes() -> UInt64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        let kr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? info.resident_size : 0
    }
}

final class AudioCollector {
    private var totalSamples: Int = 0
    private let sampleRate: Int
    init(sampleRate: Int = 24_000) { self.sampleRate = sampleRate }
    func append(_ audio: MLXArray) {
        let shape = audio.shape
        if shape.count == 1 { totalSamples += shape[0] }
        else if shape.count == 2 { totalSamples += shape[1] }
    }
    func samples() -> Int { totalSamples }
    func durationSeconds() -> Double { Double(totalSamples) / Double(sampleRate) }
}

struct BenchmarkView: View {
    @ObservedObject var ttsModel: KokoroTTSModel

    @State private var text: String = "The quick brown fox jumps over the lazy dog. This is a second sentence."
    @State private var voice: TTSVoice = .afHeart
    @State private var warmup: Bool = true
    @State private var running: Bool = false
    @State private var results: [BenchmarkResult] = []

    var body: some View {
        NavigationStack {
            List {
                Section("Input") {
                    TextEditor(text: $text).frame(minHeight: 120)
                    Picker("Voice", selection: $voice) {
                        ForEach(TTSVoice.allCases, id: \.self) { v in
                            Text(v.rawValue).tag(v)
                        }
                    }
                    Toggle("Warm-up", isOn: $warmup)
                    Button(running ? "Running…" : "Run Benchmark") { runOnce() }
                        .disabled(running || text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }

                if !results.isEmpty {
                    Section("Results") {
                        ForEach(Array(results.enumerated()), id: \.offset) { _, r in
                            VStack(alignment: .leading, spacing: 4) {
                                Text("\(r.modelName) • \(r.voice)")
                                Text(String(format: "TTFA %.2fs  |  Load %.2fs  |  Inference %.2fs", r.timeToFirstAudio, r.modelLoadTime, r.inferenceTime))
                                    .font(.caption).foregroundStyle(.secondary)
                                Text(String(format: "RTF %.2f  |  mem %.2f GB", r.realTimeFactor, r.peakMemoryGB))
                                    .font(.caption).foregroundStyle(.secondary)
                            }
                        }
                    }
                }
            }
            .navigationTitle("Benchmark")
        }
    }

    private func runOnce() {
        running = true

        let mem = MemorySampler(); mem.start()
        let swLoad = Stopwatch()
        let swInfer = Stopwatch()
        let collector = AudioCollector(sampleRate: 24_000)
        var perSentence: [Double] = []
        var firstChunkTime: Double? = nil
        var lastChunkTime: Double? = nil

        // Optional warm-up
        if warmup {
            do { try ttsModel.withEngine { engine in try engine.initializeIfNeeded() } } catch {}
            do { try ttsModel.withEngine { engine in try engine.generateAudio(voice: voice, text: "Hi.", completion: nil, chunkCallback: { _ in }) } } catch {}
            MLX.GPU.clearCache()
        }

        // Model load timing
        swLoad.start()
        do { try ttsModel.withEngine { engine in try engine.initializeIfNeeded() } } catch {}
        let modelLoad = swLoad.elapsed()

        // Inference timing
        swInfer.start()
        do {
            try ttsModel.withEngine { engine in
                try engine.generateAudio(voice: voice, text: text, completion: {
                    // Completion on main thread
                    let inference = swInfer.elapsed()
                    let audioDur = collector.durationSeconds()
                    let rtf = audioDur > 0 ? inference / audioDur : 0
                    let sps = inference > 0 ? Double(collector.samples()) / inference : 0
                    let peak = mem.peakGB
                    mem.stop()

                    let result = BenchmarkResult(
                        modelName: viewModel.currentModelResource,
                        modelDescription: viewModel.currentModelDescription,
                        text: text,
                        voice: voice.rawValue,
                        modelLoadTime: modelLoad,
                        inferenceTime: inference,
                        totalTime: modelLoad + inference,
                        timeToFirstAudio: firstChunkTime ?? 0,
                        perSentenceInference: perSentence,
                        audioDuration: audioDur,
                        realTimeFactor: rtf,
                        samplesPerSec: sps,
                        peakMemoryGB: peak,
                        sampleRate: 24_000
                    )
                    persist(results: [result])
                    results.insert(result, at: 0)
                    running = false
                }, chunkCallback: { audio in
                    let t = swInfer.elapsed()
                    if firstChunkTime == nil { firstChunkTime = t }
                    if let last = lastChunkTime { perSentence.append(t - last) }
                    else { perSentence.append(t) }
                    lastChunkTime = t
                    collector.append(audio)
                    ttsModel.enqueueAudioChunk(audio)
                })
            }
        } catch {
            mem.stop(); running = false
        }
    }

    private func persist(results: [BenchmarkResult]) {
        let fm = FileManager.default
        do {
            let docs = try fm.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
            let dir = docs.appendingPathComponent("benchmark_results", isDirectory: true)
            try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
            let ts = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
            let file = dir.appendingPathComponent("benchmark_results_\(ts).json")
            let data = try JSONEncoder().encode(results)
            try data.write(to: file)
        } catch {
            // ignore persistence errors in UI
        }
    }
}

struct Speaker: Identifiable {
    let id: Int
    let name: String

    var flag: String {
        if name.lowercased() == "none" {
            return "⚪️" // Empty/None speaker icon
        }

        guard name.count >= 2 else { return "🏳️" }
        let country = name.prefix(1)

        // Determine country flag
        let countryFlag: String
        switch country {
        case "a": countryFlag = "🇺🇸" // USA
        case "b": countryFlag = "🇬🇧" // British
        case "e": countryFlag = "🇪🇸" // Spain
        case "f": countryFlag = "🇫🇷" // French
        case "h": countryFlag = "🇮🇳" // Hindi
        case "i": countryFlag = "🇮🇹" // Italian
        case "j": countryFlag = "🇯🇵" // Japanese
        case "p": countryFlag = "🇧🇷" // Brazil
        case "z": countryFlag = "🇨🇳" // Chinese
        default: countryFlag = "🏳️"
        }

        return countryFlag
    }

    var displayName: String {
        if name.lowercased() == "none" {
            return "None" // Special case for None option
        }

        guard name.count >= 2 else { return name }
        let cleanName = name.dropFirst(3).capitalized
        return "\(cleanName)"
    }
}

class SpeakerViewModel: ObservableObject {
    @Published var selectedSpeakerId: Int = 0
    @Published var selectedSpeakerId2: Int = -1
    @Published var isGenerating: Bool = false

    let speakers: [Speaker] = [
        Speaker(id: -1, name: "none"),
        Speaker(id: 0, name: "af_alloy"),
        Speaker(id: 1, name: "af_aoede"),
        Speaker(id: 2, name: "af_bella"),
        Speaker(id: 3, name: "af_heart"),
        Speaker(id: 4, name: "af_jessica"),
        Speaker(id: 5, name: "af_kore"),
        Speaker(id: 6, name: "af_nicole"),
        Speaker(id: 7, name: "af_nova"),
        Speaker(id: 8, name: "af_river"),
        Speaker(id: 9, name: "af_sarah"),
        Speaker(id: 10, name: "af_sky"),
        Speaker(id: 11, name: "am_adam"),
        Speaker(id: 12, name: "am_echo"),
        Speaker(id: 13, name: "am_eric"),
        Speaker(id: 14, name: "am_fenrir"),
        Speaker(id: 15, name: "am_liam"),
        Speaker(id: 16, name: "am_michael"),
        Speaker(id: 17, name: "am_onyx"),
        Speaker(id: 18, name: "am_puck"),
        Speaker(id: 19, name: "am_santa"),
        Speaker(id: 20, name: "bf_alice"),
        Speaker(id: 21, name: "bf_emma"),
        Speaker(id: 22, name: "bf_isabella"),
        Speaker(id: 23, name: "bf_lily"),
        Speaker(id: 24, name: "bm_daniel"),
        Speaker(id: 25, name: "bm_fable"),
        Speaker(id: 26, name: "bm_george"),
        Speaker(id: 27, name: "bm_lewis"),
        Speaker(id: 28, name: "ef_dora"),
        Speaker(id: 29, name: "em_alex"),
        Speaker(id: 30, name: "ff_siwis"),
        Speaker(id: 31, name: "hf_alpha"),
        Speaker(id: 32, name: "hf_beta"),
        Speaker(id: 33, name: "hm_omega"),
        Speaker(id: 34, name: "hm_psi"),
        Speaker(id: 35, name: "if_sara"),
        Speaker(id: 36, name: "im_nicola"),
        Speaker(id: 37, name: "jf_alpha"),
        Speaker(id: 38, name: "jf_gongitsune"),
        Speaker(id: 39, name: "jf_nezumi"),
        Speaker(id: 40, name: "jf_tebukuro"),
        Speaker(id: 41, name: "jm_kumo"),
        Speaker(id: 42, name: "pf_dora"),
        Speaker(id: 43, name: "pm_alex"),
        Speaker(id: 44, name: "pm_santa"),
        Speaker(id: 45, name: "zf_xiaobei"),
        Speaker(id: 46, name: "zf_xiaoni"),
        Speaker(id: 47, name: "zf_xiaoxiao"),
        Speaker(id: 48, name: "zf_xiaoyi"),
        Speaker(id: 49, name: "zm_yunjian"),
        Speaker(id: 50, name: "zm_yunxi"),
        Speaker(id: 51, name: "zm_yunxia"),
        Speaker(id: 52, name: "zm_yunyang"),
    ]
    
   func getPrimarySpeaker() -> [Speaker] {
        speakers.filter { $0.id == selectedSpeakerId }
    }
    
    func getSecondarySpeaker() -> [Speaker] {
        speakers.filter { $0.id == selectedSpeakerId2 }
    }

    func getSpeaker(id: Int) -> Speaker? {
        speakers.first { $0.id == id }
    }
}

extension View {
    func dismissKeyboard() {
        UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder),
                                        to: nil,
                                        from: nil,
                                        for: nil)
    }
}

#Preview {
  ContentView(viewModel: KokoroTTSModel())
}
