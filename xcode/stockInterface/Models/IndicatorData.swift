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
    let volumeIndicator:   [Float]
    let macdDifference:    [Float]
    let macdGreen:         [Float]
    let macdRed:           [Float]
    let percentageChange:  [Float]
    var backtestingOffset: Int = 0
    var gapInDate:         Bool = false
    // Predicted Values
    var predicted1CandleOut: [Float] = []
    var predicted2CandleOut: [Float] = []
    var predicted3CandleOut: [Float] = []
    var predicted4CandleOut: [Float] = []
    var predicted5CandleOut: [Float] = []
    // Actual Value
    var actualPercentagePerCandle: [Float] = []
    var actualCandlesToTarget: [Float] = []
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
