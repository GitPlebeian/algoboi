//
//  StockCalculations.swift
//  stockInterface
//
//  Created by CHONK on 10/17/23.
//

import Foundation

class StockCalculations {
    
    static let StartAtElement: Int = 25
    
    static func GetEMASFor(for values: [Float], period: Int) -> [Float] {
        guard !values.isEmpty, period > 0 else {
            return []
        }

        var emas: [Float] = []
        var previousEMA = values[0]  // Start with the first data point
        emas.append(previousEMA)

        let multiplier: Float = 2.0 / Float(period + 1)

        for t in 1..<values.count {
            let currentEMA = (values[t] - previousEMA) * multiplier + previousEMA
            emas.append(currentEMA)
            previousEMA = currentEMA
        }

        return emas
    }
    
    static func GetAngleBetweenTwoPoints(start: Float, end: Float) -> Float {
        let dx: Float = 1.0
        let dy = end - start
        return atan2f(dy, dx)
    }
    
    static func GetAngleBetweenTwoPoints(arr: [Float], multiplier: Float = 1.0) -> [Float] {
        var results: [Float] = []
        for i in 0..<arr.count {
            if i == 0 {
                results.append(GetAngleBetweenTwoPoints(start: arr[i], end: arr[i]) * multiplier)
                continue
            }
            results.append(GetAngleBetweenTwoPoints(start: arr[i - 1], end: arr[i]) * multiplier)
        }
        return results
    }
    
    static func ConvertStockAggregateToMLTrainingData(_ aggregate: StockAggregate) -> MLTrainingData {
        let closes = aggregate.candles.map { $0.close }
        let ema9 = GetEMASFor(for: closes, period: 9)
        let ema25 = GetEMASFor(for: closes, period: 25)
        let ema9Slopes = GetAngleBetweenTwoPoints(arr: ema9)
        let ema25Slopes = GetAngleBetweenTwoPoints(arr: ema25)
        return MLTrainingData(closes: closes, slopeOf9DayEMA: ema9Slopes, slopeOf25DayEMA: ema25Slopes)
    }
}
