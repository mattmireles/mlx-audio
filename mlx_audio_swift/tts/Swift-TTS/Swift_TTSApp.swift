/// Swift_TTSApp.swift - MLX Text-to-Speech Demo Application Entry Point
///
/// This file provides the main application entry point for the MLX-based text-to-speech
/// demonstration application. It serves as a standalone SwiftUI app showcasing Kokoro
/// neural TTS capabilities while also functioning as a testbed for Gist integration.
///
/// Key responsibilities:
/// - **SwiftUI App Lifecycle**: Standard SwiftUI application initialization and management
/// - **TTS Demo Interface**: Provides user interface for testing MLX-based synthesis
/// - **Integration Testing**: Serves as development platform for Gist MLX integration
/// - **Performance Validation**: Allows benchmarking of MLX synthesis performance
///
/// Architecture:
/// - **SwiftUI App Structure**: Clean separation between app shell and content interface
/// - **Single Window**: Simple single-window application focused on TTS functionality
/// - **Minimal Dependencies**: Lightweight app structure for maximum performance
///
/// Called by:
/// - **iOS/macOS Runtime**: System launches this app as main entry point
/// - **Development Environment**: Xcode runs this for MLX TTS testing and development
///
/// Integrates with:
/// - **ContentView**: Main SwiftUI interface providing TTS controls and voice selection
/// - **KokoroTTSModel**: Core TTS engine integration via ContentView
/// - **SwiftUI Framework**: Apple's declarative UI framework for cross-platform compatibility
///
/// Integration Context:
/// - **Gist Bridge**: This app validates MLX components before Gist integration
/// - **Development Tool**: Enables rapid iteration on MLX synthesis improvements
/// - **Performance Baseline**: Provides performance comparison against CoreML implementation
///
/// Use Cases:
/// - **MLX Development**: Primary development environment for MLX TTS improvements
/// - **Voice Testing**: Interface for evaluating voice quality across all available models
/// - **Performance Analysis**: Benchmarking platform for synthesis speed and quality metrics
/// - **Integration Preview**: Preview of MLX capabilities before Gist deployment

import SwiftUI

/// Main SwiftUI application providing MLX text-to-speech demonstration and testing interface.
///
/// This app serves dual purposes as both a standalone TTS demonstration and a development
/// platform for validating MLX synthesis integration with the main Gist application.
@main
struct Swift_TTSApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
