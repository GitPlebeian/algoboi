//
//  PlaybackController.swift
//  stockInterface
//
//  Created by Jackson Tubbs on 11/12/23.
//

import Cocoa

class PlaybackController {
    
    static let shared = PlaybackController()
    
    var currentPlaybackModel: MLPlayback?
    
    var playbackIndex = 0
    
    var playbackBars: [[(Int, NSColor)]] = []
    
    func addToPlaybackIndex(_ amount: Int) {
        guard let model = currentPlaybackModel else {return}
        playbackIndex += amount
        if playbackIndex < 0 {playbackIndex = 0}
        if playbackIndex >= model.episodeCount {playbackIndex = model.episodeCount - 1}
        chartCurrentPlaybackIndex()
    }
    
    func loadPlaybackFromFile() {
        guard let data = SharedFileManager.shared.getDataFromPlaybackFile("playback.json") else {
            TerminalManager.shared.addText("playback file does not exist or there was an error getting the data", type: .error)
            return
        }
        let jsonDecoder = JSONDecoder()
        do {
            let playbackModel = try jsonDecoder.decode(MLPlayback.self, from: data)
            self.currentPlaybackModel = playbackModel
            setPlaybackBarsForModel(playbackModel)
            TerminalManager.shared.addText("Success loading playback model\nEpisodes: \(playbackModel.episodeCount)")
            chartCurrentPlaybackIndex()
        } catch {
            TerminalManager.shared.addText("Unable to convert data to MLPlayback model", type: .error)
        }
    }
    
    private func setPlaybackBarsForModel(_ model: MLPlayback) {
        playbackBars = []
        for i in 0..<model.episodeCount {
            var bars: [(Int, NSColor)] = []
            for e in model.purchaseIndexs[i] {
                bars.append((e + StockCalculations.StartAtElement - 1, CGColor.ChartPurchasedLines.NSColor()))
            }
            for e in model.sellIndexs[i] {
                bars.append((e + StockCalculations.StartAtElement - 1, CGColor.ChartSoldLines.NSColor()))
            }
            playbackBars.append(bars)
        }
    }
    
    private func chartCurrentPlaybackIndex() {
        guard let model = self.currentPlaybackModel else {return}
//        guard let chartedStockAggregate = ChartManager.shared.currentAggregate else {return}
//        if model.length != chartedStockAggregate.candles.count {return}
        if playbackIndex < 0 {
            TerminalManager.shared.addText("Playback index < 0", type: .error)
            return
        }
        if playbackIndex >= model.episodeCount {
            TerminalManager.shared.addText("Playback index: \(playbackIndex) >= episode count: \(model.episodeCount)", type: .error)
            return
        }
        TerminalManager.shared.addText("Charting Episode: \(playbackIndex) / \(model.episodeCount)")
        ChartManager.shared.currentStockView?.setColoredFullHeightBars(bars: playbackBars[playbackIndex])
    }
}
