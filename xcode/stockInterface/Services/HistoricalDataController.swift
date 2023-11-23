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
    
    func loadFromDisk() {
        guard let toDownloadData = SharedFileManager.shared.getDataFromFile("/historicalData/00toDownload.json") else {
            TerminalManager.shared.addText("Could load load 00toDownload.json", type: .error)
            return
        }
        guard let downloadedData = SharedFileManager.shared.getDataFromFile("/historicalData/00downloaded.json") else {
            TerminalManager.shared.addText("Could load load 00downloaded.json", type: .error)
            return
        }
        do {
            let jsonDecoder = JSONDecoder()
            let toDownload = try jsonDecoder.decode([TickerNameModel].self, from: toDownloadData)
            let downloaded = try jsonDecoder.decode([String].self, from: downloadedData)
            self.stocksToDownload = toDownload
            self.downloadedStocks = downloaded
            self.totalCount = stocksToDownload.count + downloadedStocks.count
            
            let percentageComplete = (Float(downloadedStocks.count) / Float(totalCount))
            TerminalManager.shared.addText("\((percentageComplete * 100).toRoundedString(precision: 1))% Downloaded\n\(downloaded.count) Downloaded\n\(toDownload.count) Needs to be downloaded")
        } catch let e {
            TerminalManager.shared.addText("Could not decode data for toDownload / downloaded: \(e)")
        }
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
    
    func calculateIndicatorData() {
        let fileNames = downloadedStocks
//        var results = [Float]()
//        let resultsQueue = DispatchQueue(label: "stockCalculationQueue", attributes: .concurrent)
        let group = DispatchGroup()

        
        let jsonDecoder = JSONDecoder()
        
        fileNames.forEach { fileName in
            group.enter() // Enter the group for each file

            DispatchQueue.global().async {
                
                if let data = SharedFileManager.shared.getDataFromFile("/historicalData/\(fileName).json") {
                    let aggregate = try! jsonDecoder.decode(StockAggregate.self, from: data)
                    
                    if let indicatorData = StockCalculations.GetIndicatorData(aggregate: aggregate) {
                        SharedFileManager.shared.writeCodableToFileNameInShared(codable: indicatorData, fileName: "/indicatorData/\(aggregate.symbol).json")
                    }
                    
//                    resultsQueue.async(flags: .barrier) {
//                        results.append(result)
//                        group.leave() // Leave the group when done
//                    }
                    group.leave()
                } else {
                    group.leave()
                }
            }
        }
        group.notify(queue: DispatchQueue.main) {
            TerminalManager.shared.addText("All Files Proccessed")
        }
    }
}
