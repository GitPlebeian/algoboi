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
        let getAllStockGroup = DispatchGroup()
        TerminalManager.shared.addText("Getting All Stocks From Libary")
        let allStocks = AllTickersController.shared.getAllTickers()
        for stock in allStocks {
            DispatchQueue.global().async(group: getAllStockGroup) {
                guard let aggregateData = SharedFileManager.shared.getDataFromFile("/historicalData/\(stock.symbol).json") else {
                    return
                }
                guard let indicatorData = SharedFileManager.shared.getDataFromFile("/indicatorData/\(stock.symbol).json") else {
                    return
                }
                do {
                    let aggregate = try decoder.decode(StockAggregate.self, from: aggregateData)
                    let indicator = try decoder.decode(IndicatorData.self, from: indicatorData)
                    self.allAggregatesAndIndicatorData.append((aggregate, indicator))
                } catch {
                    fatalError("Could not decode for ticker: \(stock.name)")
                }
            }
        }
        getAllStockGroup.notify(queue: .main) {
            TerminalManager.shared.addText("Loaded: \(self.allAggregatesAndIndicatorData.count) aggregates")
        }
        
//        var scores: [[BKTestScoreModel]] = []
        
        
        
//        addScoresForCandleIndex()
        
//        guard let aggregate = ChartManager.shared.currentAggregate else {
//            TerminalManager.shared.addText("No current aggregate", type: .error)
//            return
//        }
//        guard let indicatorData = ChartManager.shared.indicatorData else {
//            TerminalManager.shared.addText("No current indicatorData", type: .error)
//            return
//        }
        
    }
    
    private func addScoresForCandleIndex(index: Int = 0) {
        let indexDate = spyAggregate.candles[index].timestamp.stripDateToDayMonthYearAndAddOneDay()
        
        
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
