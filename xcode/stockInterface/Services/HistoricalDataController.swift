//
//  HistoricalDataController.swift
//  stockInterface
//
//  Created by CHONK on 11/20/23.
//

import Foundation

class HistoricalDataController {
    
    static let shared = HistoricalDataController()
    
    var totalCount: Int = 0
    var stocksToDownload: [TickerNameModel] = []
    var downloadedStocks: [String] = []
    
    var shouldDownload = false
    
    // MARK: Public
    
    func reset() {
        self.stocksToDownload = AllTickersController.shared.getAllTickers()
        self.totalCount = self.stocksToDownload.count
        self.downloadedStocks = []
        
        SharedFileManager.shared.writeCodableToFileNameInShared(codable: stocksToDownload, fileName: "/historicalData/00toDownload.json")
        SharedFileManager.shared.writeCodableToFileNameInShared(codable: downloadedStocks, fileName: "/historicalData/00downloaded.json")
    }
    
    func startDownload() {
        if shouldDownload == true {return}
        shouldDownload = true
        downloadAndSaveStock()
    }
    
    func stopDownload() {
        shouldDownload = false
    }
    
    func downloadAndSaveStock() {
        if shouldDownload == false {
            let percentageComplete = (Float(downloadedStocks.count) / Float(totalCount))
            TerminalManager.shared.addText("Stopped Downloading: \((percentageComplete * 100).toRoundedString(precision: 1))%")
            return
        }
        guard let nextStockToDownload = stocksToDownload.popLast() else {
            TerminalManager.shared.addText("Completed Download")
            shouldDownload = false
            return
        }
        
        TickerDownload.shared.getAlpacaStock(ticker: nextStockToDownload.symbol, year: 30) { message, aggregate in
            DispatchQueue.main.async {
                guard let aggregate = aggregate else {
                    TerminalManager.shared.addText(message, type: .error)
                    return
                }
                var newAggregate = aggregate
                newAggregate.name = nextStockToDownload.name
                SharedFileManager.shared.writeCodableToFileNameInShared(codable: newAggregate, fileName: "/historicalData/\(newAggregate.symbol).json")
                self.downloadedStocks.append(newAggregate.symbol)
                SharedFileManager.shared.writeCodableToFileNameInShared(codable: self.downloadedStocks, fileName: "/historicalData/00downloaded.json")
                SharedFileManager.shared.writeCodableToFileNameInShared(codable: self.stocksToDownload, fileName: "/historicalData/00toDownload.json")
                let percentageComplete = (Float(self.downloadedStocks.count) / Float(self.totalCount))
                TerminalManager.shared.addText("\((percentageComplete * 100).toRoundedString(precision: 1))% | \(newAggregate.symbol): ")
                self.downloadAndSaveStock()
            }
        }
        
    }
}
