//
//  Calculate.swift
//  stockInterface
//
//  Created by CHONK on 9/29/23.
//

import Cocoa

extension StockView {
    
    func updateAllPropertyValues() {
        setVisibleCandleCount()
        setStartingCandleIndex()
        setMissingCandleMarginSpace()
        getMinMaxRange()
    }
    
    func getCandleIndexForMousePositionInView(point: NSPoint) -> Int? {
        var x =  point.x
        x -= xPositionOffset
        let candle = Int(x / candleWidth)
        return candle + startingCandleIndex
    }
    
    func setVisibleCandleToMax() {
        guard let stockAggregate = stockAggregate else {return}
        endingCandleIndex = stockAggregate.candles.count - 1
    }
    
    func setMinimumCandleWidth() {
        guard let stockAggregate = stockAggregate else {return}
        minimumCandleWidth = bounds.width / CGFloat(stockAggregate.candles.count)
    }
    
    func setVisibleCandleCount() {
        visibleCandles = Int(bounds.width / candleWidth)
    }
    
    func setStartingCandleIndex() {
        startingCandleIndex = endingCandleIndex - visibleCandles + 1
    }
    
    func setMissingCandleMarginSpace() {
        lastCandleMissingSpace = bounds.width / candleWidth - CGFloat(visibleCandles)
        lastCandleMissingSpace *= candleWidth
    }
    
    func getMinMaxRange() {
        guard let stockAggregate = stockAggregate else {return}
        let candlesCount = stockAggregate.candles.count
        
        maxValue = -.infinity
        minValue = .infinity
        
        for index in startingCandleIndex...endingCandleIndex {
            if index < 0 {continue}
            if index >= candlesCount {break}
            let candle = stockAggregate.candles[index]
            if candle.low < minValue {minValue = candle.low}
            if candle.high > maxValue {maxValue = candle.high}
        }
        range = abs(maxValue - minValue)
    }
    
    func updateValuesForScaleChanged(scale: CGFloat) {
        if pinchGesture.state == .began {
            previousCandleWidth = candleWidth
        }
        if scale + 1 < 0.05 {return}
        
        candleWidth = previousCandleWidth * (scale + 1)
        if candleWidth < minimumCandleWidth {candleWidth = minimumCandleWidth}
//        updateForCurrentMousePosition()
        setNeedsDisplay(bounds)
    }
}
