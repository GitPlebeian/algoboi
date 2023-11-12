//
//  PlaybackController.swift
//  stockInterface
//
//  Created by Jackson Tubbs on 11/12/23.
//

import Foundation

class PlaybackController {
    
    static let shared = PlaybackController()
    
    var currentPlaybackModel: MLPlayback?
    
    func loadPlaybackFromFile() {
        guard let data = SharedFileManager.shared.getDataFromPlaybackFile("playback.json") else {
            TerminalManager.shared.currentTerminal?.addText("playback file does not exist or there was an error getting the data", type: .error)
            return
        }
        let jsonDecoder = JSONDecoder()
        do {
            let playbackModel = try jsonDecoder.decode(MLPlayback.self, from: data)
            self.currentPlaybackModel = playbackModel
            TerminalManager.shared.currentTerminal?.addText("Success loading playback model\nEpisodes: \(playbackModel.episodeCount)")
        } catch {
            TerminalManager.shared.currentTerminal?.addText("Unable to convert data to MLPlayback model", type: .error)
        }
    }
}
