//
//  IndicatorData.swift
//  stockInterface
//
//  Created by CHONK on 11/20/23.
//

import Foundation

struct IndicatorData: Codable {
    let ticker: String
    var length: Int {
        return sma200.count
    }
    let timestamps:        [Date]
    let sma200:            [Float]
    let sma50:             [Float]
    let ema14:             [Float]
    let ema28:             [Float]
    let volumesTimesClose: [Float]
    let volumeTCAverage:   [Float]
    let volumeChange:      [Float]
    let averageVolume:     [Float]
    let macdDifference:    [Float]
    let macdGreen:         [Float]
    let macdRed:           [Float]
    let percentageChange:  [Float]
    let standardDeviationForPercentageChange1: [Float]
    let standardDeviationForPercentageChange2: [Float]
    let stdDifference: [Float]
    
    let htl: [Float]
    let hto: [Float]
    let htc: [Float]
    let ltc: [Float]
    let lto: [Float]
    let cto: [Float]
    
    var backtestingOffset: Int = 0
    var gapInDate:         Bool = false
    
    var isBadIndex: [Bool] = []
    
    
    // Anything else below is used for the machine learning dataset inputs
    let slopesOf9DayEMA:     [Float]
    let slopesOf25DayEMA:    [Float]
    let slopesOf50DaySMA:    [Float]
    let slopesOf200DaySMA:   [Float]
    let macdGreenLineLevels: [Float]
    let macdGreenLineSlopes: [Float]
    let macdRedLineSlopes:   [Float]
    let macdDifferences:     [Float]
}
