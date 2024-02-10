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
    var predictedAmounts: [Float] = []
    
    // All Backtest
    
    var scores: [[BKTestScoreModel]] = []
    var spyAggregate: StockAggregate!
    var allAggregatesAndIndicatorData: [(StockAggregate, IndicatorData)] = []
    
    init() {
        loadSPYAggregate()
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

    func backtestAllStocks() {
        let decoder = JSONDecoder()
        TerminalManager.shared.addText("Getting All Stocks From Libary")
        let allStocks = AllTickersController.shared.getAllTickers()
        
        var stockDataArray = Array(repeating: nil as (StockAggregate, IndicatorData)?, count: allStocks.count)
        
        let queue = DispatchQueue(label: "stockLoadingQueue", attributes: .concurrent)
        let group = DispatchGroup()
        
        let start = DispatchTime.now()
        
        for (index, stock) in allStocks.enumerated() {
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
                    fatalError("Could not decode for ticker: \(stock.name)")
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
            let seconds = Float((end.uptimeNanoseconds - start.uptimeNanoseconds) / 1000000000)
            TerminalManager.shared.addText("Loaded: Data in \(seconds.toRoundedString(precision: 2)) seconds")
            self.runBacktest()
        }
    }
    
    func loadSPYAggregate() {
        guard let spyAggregateData = SharedFileManager.shared.getDataFromFile("/historicalData/VOO.json") else {
            TerminalManager.shared.addText("VOO aggregate File does not exist", type: .error)
            return
        }
        let decoder = JSONDecoder()
        do {
            spyAggregate = try decoder.decode(StockAggregate.self, from: spyAggregateData)
        } catch {
            TerminalManager.shared.addText("Unable to decode VOO", type: .error)
            return
        }
    }
    
    private func runBacktest() {
        if spyAggregate == nil {loadSPYAggregate()}
        if spyAggregate == nil {
            TerminalManager.shared.addText("VOO data not found. Cancelling backtest", type: .error)
            return
        }
        scores = []
        for candleIndex in 0..<spyAggregate.candles.count {
            print("SPY Index: \(candleIndex)")
            addScoresForCandleIndex(index: candleIndex)
        }
    }
    
    private func addScoresForCandleIndex(index: Int = 0) {
        let indexDate = spyAggregate.candles[index].timestamp.stripDateToDayMonthYearAndAddOneDay()
        scores.append([])
        for e in allAggregatesAndIndicatorData {
            let start = DispatchTime.now()
            for (i, candle) in e.0.candles.enumerated() {
                if candle.timestamp.stripDateToDayMonthYearAndAddOneDay() == indexDate {
//                    print("Index date match for: \(e.0.name)")
                    guard let score = MLPredictor1.shared.makePrediction(indicatorData: e.1, index: i, candlesToTarget: 1) else {
                        fatalError()
                    }
                    let end = DispatchTime.now()
                    let diff = (end.uptimeNanoseconds - start.uptimeNanoseconds)
                    print("Diff: \(diff)")
                    let scoreModel = BKTestScoreModel(predictedPercentageGain: score,
                                                      currentClosingPrice: candle.close,
                                                      ticker: e.0.symbol,
                                                      companyName: e.0.name)
                    scores[index].append(scoreModel)
                }
            }
        }
        
    }
}

extension BacktestController: StockViewMouseDelegate {
    func candleHoverChanged(index: Int) {
        if index < 0 || index >= portfolioValueAmounts.count {return}
        LabelValueController.shared.setLabelValue(index: 0, label: "Portfolio $", value: portfolioValueAmounts[index].toRoundedString(precision: 2))
        LabelValueController.shared.setLabelValue(index: 3, label: "Buy Hold Portfolio $", value: buyAndHoldPortfolioAmounts[index].toRoundedString(precision: 2))
        LabelValueController.shared.setLabelValue(index: 6, label: "Prediction", value: (predictedAmounts[index] * 100).toRoundedString(precision: 2))
        LabelValueController.shared.setLabelValue(index: 9, label: "Index", value: "\(index)")
    }
}
