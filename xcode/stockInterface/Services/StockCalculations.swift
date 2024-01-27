//
//  StockCalculations.swift
//  stockInterface
//
//  Created by CHONK on 10/17/23.
//

import Cocoa

class StockCalculations {
    
    static let StartAtElement: Int = 200
    
    static func GetIndicatorData(aggregate: StockAggregate) -> IndicatorData? {
        if aggregate.candles.count < StartAtElement {return nil}
        let closes = aggregate.candles.map { $0.close }
        let volumes = aggregate.candles.map { $0.volume }
        let sma200 = GetSMAS(for: closes, period: 200)
        let sma50  = GetSMAS(for: closes, period: 50)
        let ema14  = GetEMAS(for: closes, period: 14)
        let ema28  = GetEMAS(for: closes, period: 28)
        let volume5 = GetVolumsFromAverage(volumes: volumes, average: closes, period: 5)
        
        let result = IndicatorData(ticker: aggregate.symbol, sma200: sma200, sma50: sma50, ema14: ema14, ema28: ema28, volumeIndicator: volume5)
        return result
    }
    
    static func GetAuxSetsForAggregate(aggregate: StockAggregate) -> [StockViewAuxGraphProperties] {
        
        // INFORMATION: The first aux view in the array will be at the bottom of the stock chart FYI
        
        let volumes = aggregate.candles.map { Float($0.volume) }
        let closes  = aggregate.candles.map { $0.close }
        
        var periodC = GetPercentageChangeFromMovingAverageNotIncluding(volumes, period: 30, useSMA: true)
        periodC = ScalePositiveNumbersFromZeroToMaxOne(arr: periodC, scalingFactor: 1, maxNum: 3)
        let periodCBars = GetAuxGraphBarsForMinusOneToOne(periodC)
        let periodCProperties = StockViewAuxGraphProperties(height: 100, bars: periodCBars)

        // Impulse MACD
        
        let impulseMACDLines = GetImpulseMACD(arr: closes)
        let macdLine = StockViewAuxGraphLines(yValues: impulseMACDLines.0.map{ CGFloat($0) }, color: CGColor.barRed.NSColor())
        let macdSignalLine = StockViewAuxGraphLines(yValues: impulseMACDLines.1.map{ CGFloat($0) }, color: CGColor.barGreen.NSColor())
        let macdProperties = StockViewAuxGraphProperties(height: 120, lines: [macdLine, macdSignalLine])
        return [periodCProperties, macdProperties]
    }
    
    static func GetAuxGraphBarsForMinusOneToOne(_ arr: [Float]) -> [StockViewAuxGraphBars] {
        var results: [StockViewAuxGraphBars] = []
        for e in arr {
            var color: NSColor
            var y: CGFloat
            var height: CGFloat
            if e <= 0 {
                color = CGColor.barRed.NSColor()
                y = 0.5 - (0.5 * CGFloat(abs(e)))
                height = 0.5 * CGFloat(abs(e))
            } else {
                color = CGColor.barGreen.NSColor()
                y = 0.5
                height = 0.5 * CGFloat(e)
            }
            results.append(StockViewAuxGraphBars(y: y, height: height, color: color))
        }
        return results
    }
    
    static func ScalePositiveNumbersFromZeroToMaxOne(arr: [Float], scalingFactor: Float = 1, maxNum: Float) -> [Float] {
        var scaledArray: [Float] = []
        
        // Iterate through the input array
        for num in arr {
            if num > 0 {
                // Scale the positive numbers using the logarithm
                let scaledValue = log(1 + (num / maxNum) * scalingFactor)
                
                // Ensure the scaled value is between 0 and 1
                let scaledClamped = max(0, min(1, scaledValue))
                
                // Append the scaled value to the result array
                scaledArray.append(scaledClamped)
            } else {
                // Negative numbers remain unchanged
                scaledArray.append(num)
            }
        }
        
        return scaledArray
    }
    
    
    static func GetImpulseMACD(arr: [Float]) -> ([Float], [Float]) {
        let ema12 = GetEMAS(for: arr, period: 12)
        let ema26 = GetEMAS(for: arr, period: 26)
        var macdLine = zip(ema12, ema26).map(-)
        var signalLine = GetEMAS(for: macdLine, period: 9)
//        macdLine = NormalizeFromMinusOneToOne(array: macdLine)
//        signalLine = NormalizeFromMinusOneToOne(array: signalLine)
        return (macdLine, signalLine)
    }
    
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
    
    static func NormalizeFromMinusOneToOne(array: [Float]) -> [Float] {
        guard let max = array.max(), let min = array.min() else { return [] }
        return array.map { 2 * ($0 - min) / (max - min) - 1 }
    }
    
    static func GetVolumsFromAverage(volumes: [Int64], average: [Float], period: Int) -> [Float] {
        return []
    }
    
    static func GetEMAS(for values: [Float], period: Int) -> [Float] {
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
    
    static func GetSMAS(for values: [Float], period: Int) -> [Float] {
        var smas: [Float] = []
        for endingIndex in 0..<values.count {
            var startingIndex = endingIndex - period + 1
            var divideBy = period
            if startingIndex < 0 {
                divideBy += startingIndex
                startingIndex = 0
            }
            
            let sum = values[startingIndex...endingIndex].reduce(0, +)
            let average = sum / Float(divideBy)
            smas.append(average)
        }
        return smas
    }
    
    static func GetPercentageChangeFromMovingAverageNotIncluding(_ values: [Float], period: Int, useSMA: Bool) -> [Float] {
        var movingAverages: [Float] = []
        if useSMA {
            movingAverages = GetSMAS(for: values, period: period)
        } else {
            movingAverages = GetEMAS(for: values, period: period)
        }
        var results: [Float] = [0]
        for index in 1..<values.count {
            let value = (values[index] - movingAverages[index - 1]) / movingAverages[index - 1]
            results.append(value)
        }
        return results
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
        let ema9 = GetEMAS(for: closes, period: 9)
        let ema25 = GetEMAS(for: closes, period: 25)
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
