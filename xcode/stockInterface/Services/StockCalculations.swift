//
//  StockCalculations.swift
//  stockInterface
//
//  Created by CHONK on 10/17/23.
//

import Foundation

class StockCalculations {
    
    static let StartAtElement: Int = 25
    

    static func Normalize(_ inputArray: [Float]) -> [Float] {

        // Calculate the mean
        let sum = inputArray.reduce(0.0, +)
        let mean = sum / Float(inputArray.count)

        // Calculate the standard deviation
        let squaredDifferences = inputArray.map { pow($0 - mean, 2) }
        let variance = squaredDifferences.reduce(0.0, +) / Float(inputArray.count)
        let standardDeviation = sqrt(variance)

        // Normalize the array
        let normalizedArray = inputArray.map { ($0 - mean) / standardDeviation }

        return normalizedArray
    }
    
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
    
    static func GenerateNetZeroRandomAggregate(length: Int) -> StockAggregate {
        let ticker = "Random"
        var candles: [Candle] = []
        
        var previousClose: Float = 1000
        let volatility: Float = 5 // Adjust this for more/less volatility
        let trendProbability: Float = 0.3 // Probability of a trend starting
        
        for _ in 1...length {
            let randomChange = Float.random(in: -volatility...volatility)
            let isTrending = Float.random(in: 0...1) < trendProbability
            let trendFactor = isTrending ? (Float.random(in: -2...2) * volatility) : 0
            
            let open = previousClose
            let close = open + randomChange + trendFactor
            let high = max(open, close) + Float.random(in: 0...volatility)
            let low = min(open, close) - Float.random(in: 0...volatility)
            previousClose = close
            let candle = Candle(volume: 2, volumeWeighted: 2, timestamp: Date(), transactionCount: 32, open: open, close: close, high: high, low: low)
            candles.append(candle)
        }
        
        return StockAggregate(symbol: ticker, candles: candles)
    }
}
