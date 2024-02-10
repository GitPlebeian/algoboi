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
}
