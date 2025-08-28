//
//  Swift_TTS_iOSApp.swift
//  Swift-TTS-iOS
//
//  Created by Sachin Desai on 5/20/25.
//

import SwiftUI

@main
struct Swift_TTS_iOSApp: App {
    var body: some Scene {
        WindowGroup {
            // Default to bf16 checkpoint; allow switching in UI
            let weightsURL = Bundle.main.url(forResource: "kokoro-v1_0_bf16", withExtension: "safetensors")
            ContentView(viewModel: KokoroTTSModel(weightsURL: weightsURL))
        }
    }
}
