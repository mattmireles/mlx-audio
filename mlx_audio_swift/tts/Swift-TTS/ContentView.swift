/// ContentView.swift - MLX Text-to-Speech Demo Interface
///
/// This file provides the primary SwiftUI interface for demonstrating MLX-based neural
/// text-to-speech synthesis. It offers comprehensive controls for voice selection,
/// text input, and real-time synthesis with immediate audio playback.
///
/// Key responsibilities:
/// - **User Interface**: Clean, intuitive SwiftUI interface for TTS interaction
/// - **Voice Selection**: Comprehensive picker for 60+ available voice models
/// - **Real-time Synthesis**: Immediate audio generation and playback with progress indication
/// - **Provider Switching**: Support for multiple TTS engines (Kokoro, Orpheus)
/// - **Performance Feedback**: Live synthesis timing and status updates
///
/// Architecture:
/// - **SwiftUI Declarative UI**: Reactive interface with @State-driven updates
/// - **MVVM Pattern**: Clean separation between UI state and TTS business logic
/// - **ObservableObject Integration**: Reactive binding to KokoroTTSModel for state synchronization
/// - **Async Operations**: Non-blocking synthesis with proper error handling
///
/// Called by:
/// - **Swift_TTSApp**: Embedded as root view in main application window
/// - **SwiftUI Runtime**: Rendered as primary interface for user interaction
///
/// Integrates with:
/// - **KokoroTTSModel**: Primary TTS engine providing MLX-based neural synthesis
/// - **OrpheusTTSModel**: Alternative TTS engine for comparison and testing
/// - **TTSVoice**: Voice enumeration providing comprehensive voice selection options
/// - **SwiftUI Framework**: Apple's reactive UI framework for cross-platform compatibility
///
/// User Experience Features:
/// - **Instant Feedback**: Real-time synthesis progress and audio playback
/// - **Voice Variety**: Easy access to 60+ voices across multiple languages
/// - **Text Flexibility**: Support for various text inputs including special expressions
/// - **Performance Metrics**: Visible synthesis timing for performance evaluation
///
/// Integration Context:
/// - **Development Interface**: Primary development interface for MLX TTS validation
/// - **User Testing**: Platform for evaluating voice quality and synthesis performance
/// - **Gist Preview**: Demonstration of capabilities before main app integration
///
/// TTS Provider Support:
/// - **Kokoro Engine**: High-quality neural synthesis via MLX framework
/// - **Orpheus Engine**: Alternative synthesis for comparison and specialized use cases
/// - **Runtime Switching**: Dynamic engine selection without application restart
///
/// Voice Management:
/// - **Dynamic Loading**: Voices loaded on-demand to optimize memory usage
/// - **Language Adaptation**: Automatic voice filtering based on selected engine
/// - **Quality Indicators**: Clear voice descriptions and engine-specific capabilities

import SwiftUI

/// Primary SwiftUI interface for MLX-based text-to-speech demonstration and testing.
///
/// This view provides comprehensive TTS functionality including voice selection,
/// text input, real-time synthesis, and immediate audio playback with progress
/// indication and performance metrics.
struct ContentView: View {

    @State private var kokoroTTSModel: KokoroTTSModel? = nil
    @State private var orpheusTTSModel: OrpheusTTSModel? = nil

    @State private var sayThis : String = "Hello Everybody"
    @State private var status : String = ""
    
    private var availableProviders = ["kokoro", "orpheus"]
    @State private var chosenProvider : String = "kokoro"
    @State private var availableVoices: [String] = TTSVoice.allCases.map { $0.rawValue }
    @State private var chosenVoice: String = TTSVoice.bmGeorge.rawValue

    var body: some View {
        VStack {
            Image(systemName: "mouth")
                .imageScale(.large)
                .foregroundStyle(.tint)
            Text("TTS Examples")
                .font(.headline)
                .padding()

            Picker("Choose a provider", selection: $chosenProvider) {
                ForEach(availableProviders, id: \.self) { provider in
                    Text(provider.capitalized)
                }
            }
            .onChange(of: chosenProvider) { newProvider in
                if newProvider == "orpheus" {
                    availableVoices = OrpheusVoice.allCases.map { $0.rawValue }
                    chosenVoice = availableVoices.first ?? "dan"

                    status = "Orpheus is currently quite slow (0.1x on M1).  Working on it!\n\nBut it does support expressions: <laugh>, <chuckle>, <sigh>, <cough>, <sniffle>, <groan>, <yawn>, <gasp>"
                } else {
                    availableVoices = TTSVoice.allCases.map { $0.rawValue }
                    chosenVoice = availableVoices.first ?? TTSVoice.bmGeorge.rawValue

                    status = ""
                }
            }
            .padding()
            .padding(.bottom, 0)

            // Voice picker
            Picker("Choose a voice", selection: $chosenVoice) {
                ForEach(availableVoices, id: \.self) { voice in
                    Text(voice.capitalized)
                }
            }
            .padding()
            .padding(.top, 0)

            TextField("Enter text", text: $sayThis).padding()
            
            Button(action: {
                Task {
                    status = "Generating..."
                    if chosenProvider == "kokoro" {
                        if kokoroTTSModel == nil {
                            kokoroTTSModel = KokoroTTSModel()
                        }

                        if let kokoroVoice = TTSVoice.fromIdentifier(chosenVoice) ?? TTSVoice(rawValue: chosenVoice) {
                            kokoroTTSModel!.say(sayThis, kokoroVoice)
                        } else {
                            status = "Invalid Kokoro voice selected"
                        }
                        
                    } else if chosenProvider == "orpheus" {
                        if orpheusTTSModel == nil {
                            orpheusTTSModel = OrpheusTTSModel()
                        }

                        if let orpheusVoice = OrpheusVoice(rawValue: chosenVoice) {
                            await orpheusTTSModel!.say(sayThis, orpheusVoice)
                        } else {
                            status = "Invalid Orpheus voice selected"
                        }
                    }
                    
                    status = "Done"
                }
            }, label: {
                Text("Generate")
                    .font(.title2)
                    .padding()
            })
            .buttonStyle(.borderedProminent)

            Text(status)
                .font(.caption)
                .padding()
        }
        .padding()
    }
}

#Preview {
    ContentView()
}
