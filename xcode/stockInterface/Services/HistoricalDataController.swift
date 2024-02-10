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
        if BacktestController.shared.spyAggregate == nil {BacktestController.shared.loadSPYAggregate()}
        let fileNames = downloadedStocks
        let group = DispatchGroup()
        
        let jsonDecoder = JSONDecoder()
        
        fileNames.forEach { fileName in
            group.enter() // Enter the group for each file

            DispatchQueue.global().async {
                
                if let data = SharedFileManager.shared.getDataFromFile("/historicalData/\(fileName).json") {
                    let aggregate = try! jsonDecoder.decode(StockAggregate.self, from: data)
                    if var indicatorData = StockCalculations.GetIndicatorData(aggregate: aggregate) {
                        var startDateFound = false
                        let startDate = aggregate.candles.first!.timestamp.stripDateToDayMonthYearAndAddOneDay()
                        for (i, candle) in BacktestController.shared.spyAggregate.candles.enumerated() {
                            if startDate == candle.timestamp.stripDateToDayMonthYearAndAddOneDay() {
                                startDateFound = true
                                indicatorData.backtestingOffset = i
                            }
                        }
                        if startDateFound == false {fatalError()}
                        SharedFileManager.shared.writeCodableToFileNameInShared(codable: indicatorData, fileName: "/indicatorData/\(aggregate.symbol).json")
                    }
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
    
    func generateAndSaveGeneralAggregate() {
        
        var aggregates: [StockAggregate] = []
        // Load All Aggregates
        let jsonDecoder = JSONDecoder()
        downloadedStocks.forEach { fileName in
            if let data = SharedFileManager.shared.getDataFromFile("/historicalData/\(fileName).json") {
                let aggregate = try! jsonDecoder.decode(StockAggregate.self, from: data)
                aggregates.append(aggregate)
            }
        }
        TerminalManager.shared.addText("Loaded All Aggregates: \(aggregates.count)")
        
        let startingCandle = Candle(volume: 1, volumeWeighted: 1, timestamp: Date(), transactionCount: 1, open: 100, close: 100, high: 110, low: 90)
        
        var candles: [Candle] = [startingCandle]
        var candlesCount: Int = 0
        var averagePercentageGain: Float = 0
        var index = 1
        
        var aggregateHadACandle = true
        while aggregateHadACandle {
            candlesCount = 0
            averagePercentageGain = 0
            for aggregate in aggregates {
                aggregateHadACandle = false
                if index >= aggregate.candles.count {
                    continue
                }
                aggregateHadACandle = true
                let percentageGain = (aggregate.candles[index].close - aggregate.candles[index - 1].close) / aggregate.candles[index - 1].close
                if percentageGain > 0.18 && index == 1352 {
                    print("\(percentageGain) \(aggregate.symbol) \(aggregate.candles[index].timestamp) \(aggregate.name)")
                }
                candlesCount += 1
                averagePercentageGain += percentageGain
            }
            if aggregateHadACandle == false {break}
            let percentGain = averagePercentageGain / Float(candlesCount)
            if index == 1352 {
                
                print("Index: \(index) % Gain: \(percentGain) Candle Count\(candlesCount)")
            }
            let newClose = candles[index - 1].close * (percentGain + 1)
            let candle = Candle(volume: 1, volumeWeighted: 1, timestamp: Date(), transactionCount: 1, open: candles[index - 1].close, close: newClose, high: newClose, low: newClose)
            candles.append(candle)
            index += 1
        }
        let generalAggregate = StockAggregate(symbol: "GA", candles: candles, name: "General Aggregate")
        let indicatorData = StockCalculations.GetIndicatorData(aggregate: generalAggregate)!
        
        SharedFileManager.shared.writeCodableToFileNameInShared(codable: indicatorData, fileName: "/indicatorData/\(generalAggregate.symbol).json")
        SharedFileManager.shared.writeCodableToFileNameInShared(codable: generalAggregate, fileName: "/historicalData/\(generalAggregate.symbol).json")
        
    }
}
