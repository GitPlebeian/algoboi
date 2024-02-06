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
        
        var startingIndex = StockCalculations.StartAtElement - 1
        
        if startingIndex >= aggregate.candles.count {
            TerminalManager.shared.addText("Starting index: \(startingIndex) is greater than or equal to the length of the aggregate.candles", type: .error)
            return
        }
        
        var buyBars: [(Int, NSColor)] = []
        portfolioValueAmounts      = .init(repeating: startingPortfolioValues, count: aggregate.candles.count)
        buyAndHoldPortfolioAmounts = .init(repeating: startingPortfolioValues, count: aggregate.candles.count)
        
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
            
            portfolioValueAmounts[i] = portfolioValue
            buyAndHoldPortfolioAmounts[i] = buyAndHoldPortfolioValue
        }
        
        ChartManager.shared.currentStockView?.setColoredFullHeightBars(bars: buyBars)
    }
}

extension BacktestController: StockViewMouseDelegate {
    func candleHoverChanged(index: Int) {
        if index < 0 || index >= portfolioValueAmounts.count {return}
        LabelValueController.shared.setLabelValue(index: 0, label: "Portfolio $", value: portfolioValueAmounts[index].toRoundedString(precision: 2))
        LabelValueController.shared.setLabelValue(index: 3, label: "Buy Hold Portfolio $", value: buyAndHoldPortfolioAmounts[index].toRoundedString(precision: 2))
        LabelValueController.shared.setLabelValue(index: 6, label: "Index", value: "\(index)")
    }
}
