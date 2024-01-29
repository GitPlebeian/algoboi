//
//  MLDatasetGenerator.swift
//  stockInterface
//
//  Created by CHONK on 11/19/23.
//

import Foundation

class MLDatasetGenerator {
    
    static let shared = MLDatasetGenerator()
    
    // MARK: Properties
    
    private let tradeHealth: Float = 0.1
    private let tradeDecay:  Float = 0.1 / 4
    
    private var currentAggregate: StockAggregate?
    private var index: Int = 0
    
    // MARK: Get
    
    func getOuputForCurrentIndex() -> MLDatasetOutput? {
        guard let aggregate = currentAggregate else {return nil}
        let output = calculateOutputForIndex(index: self.index, aggregate: aggregate)
        return output
    }
    
    // MARK: Set
    
    func setAggregateForGeneration(aggregate: StockAggregate) {
        index = 0
        currentAggregate = aggregate
    }
    
    // MARK: Public
    
    func incrementIndex(by: Int) {
        guard let aggregate = self.currentAggregate else {
            
            return
        }
        index += by
        if index >= aggregate.candles.count {
            index = aggregate.candles.count - 1
        } else if index < 0 {
            index = 0
        }
    }
    
//    func calculateMLDataPointForAggregate(aggregate: StockAggregate) -> [MLDatasetInputOutput] {
//        var results: [MLDatasetInputOutput] = []
//        for i in 0..<aggregate.candles.count
//    }
    
    // MARK: Private
    
    func calculateOutputForIndex(index: Int, aggregate: StockAggregate) -> MLDatasetOutput? {
        
        let startingPrice = aggregate.candles[index].close
        var tradeHealth = self.tradeHealth
        var currentIndex = index
        
        var recordPercentageGainPerTrade:      Float = -.infinity
        var recordCandleCount: Int   = 0
        
//        print()
//        print()
//        
//        print("Starting Price: \(startingPrice)")
        
        while tradeHealth > 0 {
//            print("Starting Health: \(tradeHealth)")
            currentIndex += 1
            if currentIndex >= aggregate.candles.count {return nil}
            
            let totalPercentageChange = (aggregate.candles[currentIndex].close - startingPrice) / startingPrice
            let candlesPassed = currentIndex - index
            let averagePercentageChangePerCandle = totalPercentageChange / Float(candlesPassed)
//            print("Candles Passed: \(candlesPassed) Total % Change: \(totalPercentageChange) AVG: \(averagePercentageChangePerCandle)")
            if averagePercentageChangePerCandle >= 0.025 && averagePercentageChangePerCandle > recordPercentageGainPerTrade && totalPercentageChange > 0.1 && candlesPassed >= 2 && totalPercentageChange <= 0.3 && candlesPassed <= 10 {
                recordPercentageGainPerTrade = averagePercentageChangePerCandle
                recordCandleCount = candlesPassed
//                print("Setting Record")
            }
            
            let percentageDiffFromPerviousDay = (aggregate.candles[currentIndex].close - aggregate.candles[currentIndex - 1].close) / aggregate.candles[currentIndex - 1].close
//            print("Percentage % From Previous Day: \(percentageDiffFromPerviousDay)")
            tradeHealth += percentageDiffFromPerviousDay
//            print("Starting Health After 1: \(tradeHealth)")
            tradeHealth -= self.tradeDecay
//            print("Starting Health After 2: \(tradeHealth)")
        }
        if recordCandleCount == 0 {return nil}
        return MLDatasetOutput(percentagePerCandle: recordPercentageGainPerTrade,
                                    candlesToTarget:     recordCandleCount)
    }
    
    func calculateOutputsForIndex(index: Int, aggregate: StockAggregate) -> [MLDatasetOutput] {
        
        let outputsPerCandle: Int = 5
        
        var outputs: [MLDatasetOutput] = []
        
        for i in 1...outputsPerCandle {
            if i + index >= aggregate.candles.count {break}
            let currentPrice = aggregate.candles[index].close
            let candlesToTarget = i
            let totalPercentageChange = (aggregate.candles[index + 1].close - currentPrice) / currentPrice
            outputs.append(MLDatasetOutput(percentagePerCandle: totalPercentageChange, candlesToTarget: candlesToTarget))
        }
        return outputs
    }
}

extension MLDatasetGenerator: StockViewMouseDelegate {
    func candleHoverChanged(index: Int) {
        guard let aggregate = self.currentAggregate else {
            return
        }
        self.index = index
        if self.index >= aggregate.candles.count {
            self.index = aggregate.candles.count - 1
        } else if self.index < 0 {
            self.index = 0
        }
        guard let model = calculateOutputForIndex(index: self.index, aggregate: aggregate) else {
            print("NIL")
            return
        }
        print("\n% Per Candle: \(model.percentagePerCandle)\nOver \(model.candlesToTarget) Candles\nTotal \(model.percentagePerCandle * Float(model.candlesToTarget))")
    }
}
