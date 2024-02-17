//
//  StrategyTester.swift
//  stockInterface
//
//  Created by CHONK on 2/9/24.
//

import Foundation

class StrategyTester {
    
    static let shared = StrategyTester()
    
    func getIndexesToBuy(_ scores: [BKTestScoreModel]) -> [BKTestScoreModel] {
        let sorted = scores.sorted { a, b in
            a.predictedPercentageGain > b.predictedPercentageGain
        }
        
        var returnValues: [BKTestScoreModel] = []
        
        for i in 0..<10 {
            returnValues.append(sorted[i])
        }
        
        return returnValues
    }
    
    func getIndexsToBuy(stocks: [(StockAggregate, IndicatorData)], scores: [(Int, Float)]) -> [Int] {
        let sortedScores = scores.sorted { a, b in
            a.1 > b.1
        }
//        return [sortedScores[0].0, sortedScores[1].0, sortedScores[2].0]
        var result: [Int] = []
        for i in 0..<2 {
            result.append(sortedScores[i].0)
        }
        return result
    }
    
    func isStockGood(index: Int, aggregate: StockAggregate, indicator: IndicatorData) -> Bool {
        if index - indicator.backtestingOffset < 0 {return false}
        if indicator.isBadIndex[index - indicator.backtestingOffset] == true { return false}
        return true
    }
}
