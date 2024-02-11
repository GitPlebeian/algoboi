//
//  BacktestController.swift
//  stockInterface
//
//  Created by CHONK on 2/6/24.
//

import Cocoa

class BacktestController {
    
    static let shared = BacktestController()
    
    private var portfolioValue:           Float = 1000
    private var buyAndHoldPortfolioValue: Float = 1000
    
    var portfolioValueAmounts: [Float] = []
    var buyAndHoldPortfolioAmounts: [Float] = []
    
    // All Backtest
    
    var scores: [[BKTestScoreModel]] = []
    var allAggregatesAndIndicatorData: [(StockAggregate, IndicatorData)] = []
    
    var backtestIndex = 0
    var shouldContinueBacktesting = false
    var cashPositions: [(Int, Float)] = []
    
    var isRunningFullBacktest: Bool = false
    
    init() {
        self.backtestIndex = StockCalculations.StartAtElement - 1
    }
    
    func backtestCurrentChartedStock() {
        
        guard let aggregate = ChartManager.shared.currentAggregate else {
            TerminalManager.shared.addText("No current aggregate", type: .error)
            return
        }
        guard let indicatorData = ChartManager.shared.indicatorData else {
            TerminalManager.shared.addText("No current indicatorData", type: .error)
            return
        }
        isRunningFullBacktest = false
        ChartManager.shared.currentStockView?.mouseDelegate = self
        
        let startingPortfolioValues: Float = 1000
        
        portfolioValue = startingPortfolioValues
        buyAndHoldPortfolioValue = startingPortfolioValues
        
        let startingIndex = StockCalculations.StartAtElement - 1
        
        if startingIndex >= aggregate.candles.count {
            TerminalManager.shared.addText("Starting index: \(startingIndex) is greater than or equal to the length of the aggregate.candles", type: .error)
            return
        }
        
        var buyBars: [(Int, NSColor)] = []
        portfolioValueAmounts      = .init(repeating: startingPortfolioValues, count: aggregate.candles.count)
        buyAndHoldPortfolioAmounts = .init(repeating: startingPortfolioValues, count: aggregate.candles.count)
        predictedAmounts = .init(repeating: startingPortfolioValues, count: aggregate.candles.count)
        
        var boughtPrice: Float? = nil
        
        for i in startingIndex..<aggregate.candles.count {
            guard let prediction = MLPredictor1.shared.makePrediction(indicatorData: indicatorData, index: i, candlesToTarget: 1) else {
                TerminalManager.shared.addText("No prediciton available....", type: .error)
                return
            }
            
            if boughtPrice != nil {
                let endingPrice = aggregate.candles[i].close
                let changeMultiplyer = (endingPrice - boughtPrice!) / boughtPrice! + 1
                portfolioValue *= changeMultiplyer
                boughtPrice = nil
            }
            if prediction > 0 { // Buy
                if i != aggregate.candles.count - 1 {
                    boughtPrice = aggregate.candles[i].close
                }
                buyBars.append((i, CGColor.ChartPurchasedLines.NSColor()))
            }
            
            if i != startingIndex {
                let endingPrice = aggregate.candles[i].close
                let startingPrice = aggregate.candles[i - 1].close
                let changeMultiplyer = (endingPrice - startingPrice) / startingPrice + 1
                buyAndHoldPortfolioValue *= changeMultiplyer
            }
            predictedAmounts[i] = prediction
            portfolioValueAmounts[i] = portfolioValue
            buyAndHoldPortfolioAmounts[i] = buyAndHoldPortfolioValue
        }
        
        ChartManager.shared.currentStockView?.setColoredFullHeightBars(bars: buyBars)
    }
    
    func startFullBacktest() {
        shouldContinueBacktesting = true
        if allAggregatesAndIndicatorData.count == 0 {
            loadAllStocksFromShared()
        } else {
            if backtestIndex == StockCalculations.StartAtElement - 1 {
                let startingValue: Float = 1000
                portfolioValueAmounts          = .init(repeating: startingValue, count: SPYController.shared.spyAggregate.candles.count)
                buyAndHoldPortfolioAmounts     = .init(repeating: startingValue, count: SPYController.shared.spyAggregate.candles.count)
                portfolioValue = startingValue
                buyAndHoldPortfolioValue = startingValue
            }
            guard let indicatorData = SharedFileManager.shared.getDataFromFile("/indicatorData/VOO.json") else {
                TerminalManager.shared.addText("Indicator File does not exist", type: .error)
                return
            }
            do {
                let jsonDecoder = JSONDecoder()
                let indicator = try jsonDecoder.decode(IndicatorData.self, from: indicatorData)
                
                let auxSets = StockCalculations.GetAuxSetsForAggregate(aggregate: SPYController.shared.spyAggregate)
                
                ChartManager.shared.chartStock(SPYController.shared.spyAggregate)
                ChartManager.shared.setIndicatorData(indicator)
                ChartManager.shared.setAuxSets(auxSets: auxSets)
                ChartManager.shared.currentStockView?.mouseDelegate = self
                self.isRunningFullBacktest = true
            } catch let e {
                TerminalManager.shared.addText("Unable to unwrap data into aggregate: \(e)", type: .error)
            }
            backtestIndex(index: backtestIndex)
        }
    }
    
    func stopFullBacktest() {
        shouldContinueBacktesting = false
    }
    
    func backtestIndex(index: Int) {
        DispatchQueue.global().async {
            if self.shouldContinueBacktesting == false {return}
            print("Index: \(index)")
            // Sell All Current Positions
            for (i, e) in self.cashPositions.enumerated() {
                let startingPrice = self.allAggregatesAndIndicatorData[e.0].0.candles[index - 1].close
                let endingPrice   = self.allAggregatesAndIndicatorData[e.0].0.candles[index].close
                let changeMultiplyer = (endingPrice - startingPrice) / startingPrice + 1
                self.cashPositions[i].1 *= changeMultiplyer
            }
            if self.cashPositions.count != 0 {
                self.portfolioValue = self.cashPositions.map{$0.1}.sum()
            }
            self.cashPositions = []
            
            var predictions: [(Int, Float)] = []
            for (i, e) in self.allAggregatesAndIndicatorData.enumerated() {
                if StrategyTester.shared.isStockGood(index: index - e.1.backtestingOffset, aggregate: e.0, indicator: e.1) == false {
                    predictions.append((i, -999))
                    continue
                }
                if index - e.1.backtestingOffset >= e.0.candles.count {
                    predictions.append((i, -999))
                    continue
                }
                if e.0.candles[index - e.1.backtestingOffset].timestamp.stripDateToDayMonthYearAndAddOneDay() != SPYController.shared.spyAggregate.candles[index].timestamp.stripDateToDayMonthYearAndAddOneDay() {
                    print("Date Gap Found: \(e.0.symbol)")
                    predictions.append((i, -999))
                    continue
                }
                let prediction = MLPredictor1.shared.makePrediction(indicatorData: e.1, index: index - e.1.backtestingOffset, candlesToTarget: 1)!
                predictions.append((i, prediction))
            }
            
            let indexesToBuy = StrategyTester.shared.getIndexsToBuy(stocks: self.allAggregatesAndIndicatorData, scores: predictions)
            if indexesToBuy.count > 0 {
                let divider = Float(indexesToBuy.count)
                for e in indexesToBuy {
                    self.cashPositions.append((e, self.portfolioValue / divider))
                }
            }
            
            // SPY Stuff
            if index != StockCalculations.StartAtElement - 1 {
                let endingPrice = SPYController.shared.spyAggregate.candles[index].close
                let startingPrice = SPYController.shared.spyAggregate.candles[index - 1].close
                let changeMultiplyer = (endingPrice - startingPrice) / startingPrice + 1
                self.buyAndHoldPortfolioValue *= changeMultiplyer
            }
            
            self.portfolioValueAmounts[index] = self.portfolioValue
            self.buyAndHoldPortfolioAmounts[index] = self.buyAndHoldPortfolioValue
            
            self.backtestIndex += 1
            self.backtestIndex(index: self.backtestIndex)
        }
    }
    
    func loadAllStocksFromShared() {
        let decoder = JSONDecoder()
        TerminalManager.shared.addText("Getting All Stocks From Libary")
        allAggregatesAndIndicatorData = []
        let allStocks = AllTickersController.shared.getAllTickers()
        
        var stockDataArray = Array(repeating: nil as (StockAggregate, IndicatorData)?, count: allStocks.count)
        
        let queue = DispatchQueue(label: "stockLoadingQueue", attributes: .concurrent)
        let group = DispatchGroup()
        
        let start = DispatchTime.now()
        
        for (index, stock) in allStocks.enumerated() {
            if stock.symbol == "VOO" {continue}
            group.enter()
            queue.async {
                guard let aggregateData = SharedFileManager.shared.getDataFromFile("/historicalData/\(stock.symbol).json") else {
                    group.leave()
                    return
                }
                guard let indicatorData = SharedFileManager.shared.getDataFromFile("/indicatorData/\(stock.symbol).json") else {
                    group.leave()
                    return
                }
                do {
                    let aggregate = try decoder.decode(StockAggregate.self, from: aggregateData)
                    let indicator = try decoder.decode(IndicatorData.self, from: indicatorData)
                    stockDataArray[index] = (aggregate, indicator)
                    group.leave()
                } catch {
                    //                    fatalError("Could not decode for ticker: \(stock.name)")
                    print("Could not decode for ticker: \(stock.name) \(stock.symbol)")
                    group.leave()
                }
            }
        }
        group.notify(queue: .main) {
            for data in stockDataArray {
                if data != nil {
                    self.allAggregatesAndIndicatorData.append(data!)
                }
            }
            let end = DispatchTime.now()
            let seconds = Float((end.uptimeNanoseconds - start.uptimeNanoseconds) / 10000000)
            TerminalManager.shared.addText("Loaded: Data in \((seconds / 100).toRoundedString(precision: 2)) seconds")
            self.startFullBacktest()
        }
    }
    
//    func backtestAllStocks() {
//        
//        if let fileNames = SharedFileManager.shared.getFileNamesFromFolder(path: "scores/") {
//            if fileNames.contains("scores.json") {
//                TerminalManager.shared.addText("Scores file already exists. Loading File.")
//                loadScores()
//                chartAndRunBacktest()
//                return
//            }
//        }
//        let decoder = JSONDecoder()
//        TerminalManager.shared.addText("Getting All Stocks From Libary")
//        let allStocks = AllTickersController.shared.getAllTickers()
//        
//        var stockDataArray = Array(repeating: nil as (StockAggregate, IndicatorData)?, count: allStocks.count)
//        
//        let queue = DispatchQueue(label: "stockLoadingQueue", attributes: .concurrent)
//        let group = DispatchGroup()
//        
//        let start = DispatchTime.now()
//        
//        for (index, stock) in allStocks.enumerated() {
//            group.enter()
//            queue.async {
//                guard let aggregateData = SharedFileManager.shared.getDataFromFile("/historicalData/\(stock.symbol).json") else {
//                    group.leave()
//                    return
//                }
//                guard let indicatorData = SharedFileManager.shared.getDataFromFile("/indicatorData/\(stock.symbol).json") else {
//                    group.leave()
//                    return
//                }
//                do {
//                    let aggregate = try decoder.decode(StockAggregate.self, from: aggregateData)
//                    let indicator = try decoder.decode(IndicatorData.self, from: indicatorData)
//                    stockDataArray[index] = (aggregate, indicator)
//                    group.leave()
//                } catch {
//                    //                    fatalError("Could not decode for ticker: \(stock.name)")
//                    print("Could not decode for ticker: \(stock.name) \(stock.symbol)")
//                    group.leave()
//                }
//            }
//        }
//        group.notify(queue: .main) {
//            
//            for data in stockDataArray {
//                if data != nil {
//                    self.allAggregatesAndIndicatorData.append(data!)
//                }
//            }
//            
//            let end = DispatchTime.now()
//            let seconds = Float((end.uptimeNanoseconds - start.uptimeNanoseconds) / 1000000000)
//            TerminalManager.shared.addText("Loaded: Data in \(seconds.toRoundedString(precision: 2)) seconds")
//            self.calculateBacktestingScores()
//        }
//    }
    
//    func loadScores() {
//        guard let scoresData = SharedFileManager.shared.getDataFromFile("/scores/scores.json") else {
//            TerminalManager.shared.addText("Unable to load scores model as it does not yet exist")
//            return
//        }
//        let decoder = JSONDecoder()
//        do {
//            self.scores = try decoder.decode([[BKTestScoreModel]].self, from: scoresData)
//            TerminalManager.shared.addText("Loaded Scores: \(scores.count)")
//        } catch {
//            TerminalManager.shared.addText("Unable to decode scores", type: .error)
//            return
//        }
//    }
//    
//    private func calculateBacktestingScores() {
//        if SPYController.shared.spyAggregate == nil {loadSPYAggregate()}
//        if SPYController.shared.spyAggregate == nil {
//            TerminalManager.shared.addText("VOO data not found. Cancelling backtest", type: .error)
//            return
//        }
//        scores = Array(repeating: [], count: SPYController.shared.spyAggregate.candles.count)
//        //        for candleIndex in 0..<SPYController.shared.spyAggregate.candles.count {
//        //            addScoresForCandleIndex(index: candleIndex)
//        //        }
//        let queue = DispatchQueue(label: "scoreAssignmentQueue", attributes: .concurrent)
//        let group = DispatchGroup()
//        
//        group.enter()
//        queue.async {
//            var completedCount: Int = 0
//            for e in self.allAggregatesAndIndicatorData {
//                for (i, candle) in e.0.candles.enumerated() {
//                    guard let score = MLPredictor1.shared.makePrediction(indicatorData: e.1, index: i, candlesToTarget: 1) else {
//                        fatalError()
//                    }
//                    let model = BKTestScoreModel(predictedPercentageGain: score, currentClosingPrice: candle.close, ticker: e.0.symbol, companyName: e.0.name)
//                    self.scores[i + e.1.backtestingOffset].append(model)
//                }
//                completedCount += 1
//                print("Completed: \(completedCount) Out Of: \(self.allAggregatesAndIndicatorData.count)")
//                //            group.enter()
//                //            queue.async {
//                //                group.leave()
//                //            }
//            }
//            SharedFileManager.shared.writeCodableToFileNameInShared(codable: self.scores, fileName: "scores/scores.json")
//            group.leave()
//        }
//        group.notify(queue: .main) {
//            TerminalManager.shared.addText("Completed Score Assignments")
//            self.chartAndRunBacktest()
//        }
//    }
    
    
//    private func chartAndRunBacktest() {
//        let startingPortfolioValues: Float = 1000
//        let startingIndex = StockCalculations.StartAtElement - 1
//        portfolioValueAmounts      = .init(repeating: startingPortfolioValues, count: SPYController.shared.spyAggregate.candles.count)
//        buyAndHoldPortfolioAmounts = .init(repeating: startingPortfolioValues, count: SPYController.shared.spyAggregate.candles.count)
//        portfolioValue = startingPortfolioValues
//        buyAndHoldPortfolioValue = startingPortfolioValues
//        
//        var boughtModels: [BKTestScoreModel] = []
//        var cashPositions: [Float] = []
//        print("Starting At Index: \(startingIndex)")
//        for i in startingIndex..<SPYController.shared.spyAggregate.candles.count {
//            
//            let scores = scores[i]
//            
//            for (boughtIndex, boughtModel) in boughtModels.enumerated() {
//                var foundE = false
//                for e in scores {
//                    if e.ticker == boughtModel.ticker {
//                        foundE = true
//                        let endingPrice = e.currentClosingPrice
//                        let changeMultiplyer = (endingPrice - boughtModel.currentClosingPrice) / boughtModel.currentClosingPrice + 1
////                        print("Selling for :\(changeMultiplyer)")
//                        cashPositions[boughtIndex] *= changeMultiplyer
//                        break
//                    }
//                }
//                if foundE == false {fatalError()}
//            }
//            if cashPositions.count != 0 {
//                portfolioValue = cashPositions.sum()
//            }
//            
//            let modelsToBuy = StrategyTester.shared.getIndexesToBuy(scores)
//            let divider = Float(modelsToBuy.count)
//            cashPositions = Array(repeating: portfolioValue / divider, count: modelsToBuy.count)
//            boughtModels = modelsToBuy
//            
//            // SPY Stuff
//            if i != startingIndex {
//                let endingPrice = SPYController.shared.spyAggregate.candles[i].close
//                let startingPrice = SPYController.shared.spyAggregate.candles[i - 1].close
//                let changeMultiplyer = (endingPrice - startingPrice) / startingPrice + 1
//                buyAndHoldPortfolioValue *= changeMultiplyer
//            }
//            
//            portfolioValueAmounts[i] = portfolioValue
//            buyAndHoldPortfolioAmounts[i] = buyAndHoldPortfolioValue
//        }
//        
//        guard let indicatorData = SharedFileManager.shared.getDataFromFile("/indicatorData/VOO.json") else {
//            TerminalManager.shared.addText("Indicator File does not exist", type: .error)
//            return
//        }
//        do {
//            let jsonDecoder = JSONDecoder()
//            let indicator = try jsonDecoder.decode(IndicatorData.self, from: indicatorData)
//            
//            let auxSets = StockCalculations.GetAuxSetsForAggregate(aggregate: SPYController.shared.spyAggregate)
//            
//            ChartManager.shared.chartStock(SPYController.shared.spyAggregate)
//            ChartManager.shared.setIndicatorData(indicator)
//            ChartManager.shared.setAuxSets(auxSets: auxSets)
//            ChartManager.shared.currentStockView?.mouseDelegate = self
//            self.isRunningFullBacktest = true
//        } catch let e {
//            TerminalManager.shared.addText("Unable to unwrap data into aggregate: \(e)", type: .error)
//        }
//    }
}
extension BacktestController: StockViewMouseDelegate {
    func candleHoverChanged(index: Int) {
        if index < 0 || index >= portfolioValueAmounts.count {return}
        
        if isRunningFullBacktest {
            LabelValueController.shared.setLabelValue(index: 0, label: "Buy Hold Portfolio $", value: buyAndHoldPortfolioAmounts[index].toRoundedString(precision: 2))
            LabelValueController.shared.setLabelValue(index: 3, label: "Strategy", value: portfolioValueAmounts[index].toRoundedString(precision: 2))
            LabelValueController.shared.setLabelValue(index: 6, label: "Index", value: "\(index)")
        } else {
            LabelValueController.shared.setLabelValue(index: 0, label: "Portfolio $", value: portfolioValueAmounts[index].toRoundedString(precision: 2))
            LabelValueController.shared.setLabelValue(index: 3, label: "Buy Hold Portfolio $", value: buyAndHoldPortfolioAmounts[index].toRoundedString(precision: 2))
            LabelValueController.shared.setLabelValue(index: 6, label: "Prediction", value: (predictedAmounts[index] * 100).toRoundedString(precision: 2))
            LabelValueController.shared.setLabelValue(index: 9, label: "Index", value: "\(index)")
        }
        
    }
}
