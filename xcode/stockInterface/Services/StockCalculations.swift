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
        let high = aggregate.candles.map { $0.high }
        let low = aggregate.candles.map { $0.low }
        let open = aggregate.candles.map { $0.open }
        let volumes = aggregate.candles.map { $0.volume }
        let volumesFloat = aggregate.candles.map{ Float($0.volume) }
        let timestamps = aggregate.candles.map {$0.timestamp}
        let sma200 = GetSMAS(for: closes, period: 200)
        let sma50  = GetSMAS(for: closes, period: 50)
        let ema14  = GetEMAS(for: closes, period: 14)
        let ema28  = GetEMAS(for: closes, period: 28)
        let volumesTimesClose = GetVolumsFromAverage(volumes: volumes, average: closes, period: 10)
        let volumeTCAverage = GetEMAS(for: volumesTimesClose, period: 20)
        let volumesChange = GetPercentageChangeFromMovingAverageNotIncluding(volumesFloat, period: 30, useSMA: true)
        let percentageChanges = GetPercentageChange(closes)
        let standardDeviationForPercentageChange1 = StandardDeviation(values: percentageChanges, period: 5)
        let standardDeviationForPercentageChange2 = StandardDeviation(values: percentageChanges, period: 30)
        let stdDifference = Minus(values1: standardDeviationForPercentageChange1, values2: standardDeviationForPercentageChange2)
        let averageVolume = GetSMAS(for: volumesFloat, period: 20)
        
        let macdData = GetImpulseMACD(arr: closes)
        let macdDifference = macdData.0.minus(compare: macdData.1)
        
        // MACD Red is 0 | MACD Green is 1
        
        let htl = GetPercentageChange(starting: low, ending: high)
        let hto = GetPercentageChange(starting: open, ending: high)
        let htc = GetPercentageChange(starting: closes, ending: high)
        let ltc = GetPercentageChange(starting: low, ending: closes)
        let lto = GetPercentageChange(starting: low, ending: open)
        let cto = GetPercentageChange(starting: open, ending: closes)
        
        // Calculations for Machine Learning dataset inputs
        
        let ema9 = GetEMAS(for: closes, period: 9)
        let ema9Slopes = GetAngleBetweenTwoPoints(arr: ema9)
        let ema25 = GetEMAS(for: closes, period: 25)
        let ema25Slopes = GetAngleBetweenTwoPoints(arr: ema25)
        let sma200Slopes = GetAngleBetweenTwoPoints(arr: sma200)
        let sma50Slopes = GetAngleBetweenTwoPoints(arr: sma50)
        let macdGreenSlopes = GetAngleBetweenTwoPoints(arr: macdData.1)
        let macdRedSlopes = GetAngleBetweenTwoPoints(arr: macdData.0)
        
        var result = IndicatorData(ticker: aggregate.symbol,
                                   timestamps: timestamps,
                                   sma200: sma200,
                                   sma50: sma50,
                                   ema14: ema14,
                                   ema28: ema28,
                                   volumesTimesClose: volumesTimesClose, 
                                   volumeTCAverage: volumeTCAverage,
                                   volumeChange: volumesChange,
                                   averageVolume: averageVolume,
                                   macdDifference: macdDifference,
                                   macdGreen: macdData.1,
                                   macdRed: macdData.0, 
                                   percentageChange: percentageChanges,
                                   standardDeviationForPercentageChange1: standardDeviationForPercentageChange1,
                                   standardDeviationForPercentageChange2: standardDeviationForPercentageChange2, stdDifference: stdDifference,
                                   htl: htl,
                                   hto: hto,
                                   htc: htc,
                                   ltc: ltc,
                                   lto: lto,
                                   cto: cto,
                                   isBadIndex: .init(repeating: false, count: closes.count), slopesOf9DayEMA: ema9Slopes,
                                   slopesOf25DayEMA: ema25Slopes,
                                   slopesOf50DaySMA: sma50Slopes,
                                   slopesOf200DaySMA: sma200Slopes,
                                   macdGreenLineLevels: macdData.1,
                                   macdGreenLineSlopes: macdGreenSlopes,
                                   macdRedLineSlopes: macdRedSlopes,
                                   macdDifferences: macdDifference)
        
        CalculateBadIndeces(&result, aggregate)
        
        return result
    }
    
    static func CalculateBadIndeces(_ model: inout IndicatorData, _ aggregate: StockAggregate) {
        for i in 0..<aggregate.candles.count {
            if aggregate.candles[i].close < 3 || aggregate.candles[i].close > 1000 {
                // Price
                model.isBadIndex[i] = true
                continue
            }
            if model.volumeTCAverage[i] < 10_000_000 {
                // Market Cap
                model.isBadIndex[i] = true
                continue
            }
            if model.volumesTimesClose[i] < 10_000_000 {
                model.isBadIndex[i] = true
                continue
            }
            if model.averageVolume[i] < 10_000 {
                model.isBadIndex[i] = true
                continue
            }
            if aggregate.candles[i].volume < 10_000 {
                model.isBadIndex[i] = true
                continue
            }
        }
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
        let macdLine = zip(ema12, ema26).map(-)
        let signalLine = GetEMAS(for: macdLine, period: 9)
        return (macdLine, signalLine)
    }
    
    static func GetMACDPercentageDifference(closes: [Float]) -> [Float] {
        let macdData = GetImpulseMACD(arr: closes)
        return macdData.1.percentageChanges(compare: macdData.0)
    }
    
    static func CalculateTrend(values: [Float], periods: [Int]) -> Float {
        var stdDevs: [Float] = []
        
        // Calculate standard deviations for each period
        for period in periods.sorted() {
            let stdDev = StandardDeviation(values: values, period: period).last ?? 0
            stdDevs.append(stdDev)
        }
        
        // Assume periods are sorted and the first one is the shortest
        guard let shortestPeriodStdDev = stdDevs.first else { return 0 }
        
        // Calculate the average of the longer periods' standard deviations
        let longerPeriodsStdDev = stdDevs.dropFirst() // Removes the shortest period std dev
        let averageLongerPeriodsStdDev = longerPeriodsStdDev.reduce(0, +) / Float(longerPeriodsStdDev.count)
        
        // The trend is the difference between the shortest period std dev and the average of the longer periods
        let trend = shortestPeriodStdDev - averageLongerPeriodsStdDev
        
        return trend
    }
    
    static func StandardDeviation(values: [Float], period: Int) -> [Float] {
        guard period > 0 else { return [] }
        var stdDevArray: [Float] = []
        
        for i in 0..<values.count {
            let start = max(0, i - period + 1)
            let end = i + 1
            let subset = Array(values[start..<end])
            let stdDev = StandardDeviation(subset)
            stdDevArray.append(stdDev)
        }
        
        return stdDevArray
    }
    
    static func StandardDeviation(_ values: [Float]) -> Float {
        let mean = values.reduce(0, +) / Float(values.count)
        let variance = values.reduce(0) { $0 + pow($1 - mean, 2) } / Float(values.count)
        return sqrt(variance)
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
        var values: [Float] = []
        
        for i in 0..<volumes.count {
            let start = max(0, i - period + 1)
            let end = i + 1
            let subset = Array(volumes[start..<end])
            let closeSubset = Array(average[start..<end])
            var total: Int64 = 0
            for e in subset {
                total += e
            }
            total /= Int64(period)
            values.append(Float(total) * closeSubset.avg())
        }
        return values
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
            if value.isNaN || value.isInfinite {
                results.append(0)
                continue
            }
            results.append(value)
        }
        return results
    }
    
    static func GetPercentageChange(starting: [Float], ending: [Float]) -> [Float] {
        var results: [Float] = []
        for i in 0..<starting.count {
            results.append((ending[i] - starting[i]) / starting[i])
        }
        return results
    }
    
    static func GetPercentageChange(_ arr: [Float]) -> [Float] {
        var results: [Float] = [0]

        for i in 1..<arr.count {
            results.append((arr[i] - arr[i - 1]) / arr[i - 1])
        }
        
        return results
    }
    
    static func Minus(values1: [Float], values2: [Float]) -> [Float] {
        var value: [Float] = []
        for i in 0..<values1.count {
            value.append(values1[i] - values2[i])
        }
        return value
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
    
//    static func ConvertStockAggregateToMLTrainingData(_ aggregate: StockAggregate) -> MLTrainingData {
//        let closes = aggregate.candles.map { $0.close }
//        let ema9 = GetEMAS(for: closes, period: 9)
//        let ema25 = GetEMAS(for: closes, period: 25)
//        let ema9Slopes = GetAngleBetweenTwoPoints(arr: ema9)
//        let ema25Slopes = GetAngleBetweenTwoPoints(arr: ema25)
//        return MLTrainingData(closes: closes, slopeOf9DayEMA: ema9Slopes, slopeOf25DayEMA: ema25Slopes)
//    }
    
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
