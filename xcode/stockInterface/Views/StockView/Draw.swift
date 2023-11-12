//
//  Draw.swift
//  stockInterface
//
//  Created by CHONK on 9/29/23.
//

import Cocoa

extension StockView {
    
    override func draw(_ dirtyRect: NSRect) {
        super.draw(dirtyRect)
        if stockAggregate == nil {return}
        guard let context = NSGraphicsContext.current?.cgContext else { return }
        context.setShouldAntialias(false)
        updateAllPropertyValues()
        calculateDrawingValues()
        drawHoveringCandleLine()
        drawGreenRedBars()
        drawCandles()
        drawBordlineCandles()
    }
    
    private func calculateDrawingValues() {
        xPositionOffset = lastCandleMissingSpace / 2 + panDistance
    }
    
    func drawGreenRedBars() {
        guard let hoveringCandleIndex = currentHoveringCandle else {
            return
        }
        
        for e in greenBarIndices {
            if e < 0 || e >= stockAggregate!.candles.count {
                continue
            }
            if e < startingCandleIndex || e > endingCandleIndex {
                return
            }
        }
        
        if hoveringCandleIndex < 0 || hoveringCandleIndex >= stockAggregate!.candles.count {
            return
        }
        if hoveringCandleIndex < startingCandleIndex || hoveringCandleIndex > endingCandleIndex {
            return
        }

        var xPosition: CGFloat = getCandleXPositionInViewForCandleIndex(candleIndex: hoveringCandleIndex)
        xPosition *= candleWidth
        let candleBodyPath = NSBezierPath(rect: NSRect(x: xPosition + xPositionOffset,
                                                       y: 0,
                                                       width: candleWidth,
                                                       height: bounds.height))
        let color: NSColor = .init(displayP3Red: 1, green: 1, blue: 1, alpha: 0.15)
        color.setFill()
        candleBodyPath.fill()
    }
    
    func drawCandles() {
        
        for index in 0..<visibleCandles {
            let candleToIndex = index + startingCandleIndex
            if candleToIndex < 0 || candleToIndex >= stockAggregate!.candles.count {continue}
            let candle = stockAggregate!.candles[index + startingCandleIndex]
            var candleBodyYPos: Float
            if candle.open > candle.close {
                candleBodyYPos = 1 - ((maxValue - candle.close) / range)
            } else {
                candleBodyYPos = 1 - ((maxValue - candle.open) / range)
            }
            let candleBodyX = CGFloat(index) * candleWidth
            let candleBodyHeight = bounds.height / (CGFloat(range) / (abs(CGFloat(candle.open - candle.close))))
            let candleBodyPath = NSBezierPath(rect: NSRect(x: candleBodyX + xPositionOffset,
                                                           y: CGFloat(candleBodyYPos) * bounds.height,
                                                           width: candleWidth,
                                                           height: candleBodyHeight))
            
            let candleLineXPos = (CGFloat(index) * (candleWidth)) + candleWidth / 2
            let candleLineYPos = 1 - ((maxValue - candle.low) / range)
            let candleLinePath = NSBezierPath()
            candleLinePath.move(to: CGPoint(x: candleLineXPos + xPositionOffset,
                                            y: (CGFloat(candleLineYPos) * bounds.height)))
            candleLinePath.line(to: CGPoint(x: candleLineXPos + xPositionOffset,
                                               y: (CGFloat(candleLineYPos) * bounds.height) + (bounds.height / (CGFloat(range) / (abs(CGFloat(candle.high - candle.low)))))))
            var candleColor: NSColor
            if candle.open > candle.close {
                candleColor = .init(red: 1, green: 0, blue: 0, alpha: 1)
            } else {
                candleColor = .init(red: 0, green: 1, blue: 0, alpha: 1)
            }
            
            candleColor.setFill()
            candleColor.setStroke()
            candleLinePath.lineWidth = 1
            candleLinePath.stroke()
            candleBodyPath.lineWidth = 0
            candleBodyPath.fill()
        }
        
    }
    
    func drawBordlineCandles() {
        drawSpecificCandle(candleToIndex: startingCandleIndex - 1, positionIndex: -1)
        drawSpecificCandle(candleToIndex: startingCandleIndex - 2, positionIndex: -2)
        drawSpecificCandle(candleToIndex: endingCandleIndex + 1, positionIndex: visibleCandles)
        drawSpecificCandle(candleToIndex: endingCandleIndex + 2, positionIndex: visibleCandles + 1)
    }
    
    func drawSpecificCandle(candleToIndex: Int, positionIndex: Int) {
        if candleToIndex < 0 || candleToIndex >= stockAggregate!.candles.count {return}
        let candle = stockAggregate!.candles[candleToIndex]
        var candleBodyYPos: Float
        if candle.open > candle.close {
            candleBodyYPos = 1 - ((maxValue - candle.close) / range)
        } else {
            candleBodyYPos = 1 - ((maxValue - candle.open) / range)
        }
        let candleBodyX = CGFloat(positionIndex) * candleWidth
        let candleBodyHeight = bounds.height / (CGFloat(range) / (abs(CGFloat(candle.open - candle.close))))
        let candleBodyPath = NSBezierPath(rect: NSRect(x: candleBodyX + xPositionOffset,
                                                       y: CGFloat(candleBodyYPos) * bounds.height,
                                                       width: candleWidth,
                                                       height: candleBodyHeight))
        
        let candleLineXPos = (CGFloat(positionIndex) * (candleWidth)) + candleWidth / 2
        let candleLineYPos = 1 - ((maxValue - candle.low) / range)
        let candleLinePath = NSBezierPath()
        candleLinePath.move(to: CGPoint(x: candleLineXPos + xPositionOffset,
                                        y: (CGFloat(candleLineYPos) * bounds.height)))
        candleLinePath.line(to: CGPoint(x: candleLineXPos + xPositionOffset,
                                           y: (CGFloat(candleLineYPos) * bounds.height) + (bounds.height / (CGFloat(range) / (abs(CGFloat(candle.high - candle.low)))))))
        var candleColor: NSColor
        if candle.open > candle.close {
            candleColor = .init(red: 0.7, green: 0, blue: 0, alpha: 1)
        } else {
            candleColor = .init(red: 0, green: 0.7, blue: 0, alpha: 1)
        }
        
        candleColor.setFill()
        candleColor.setStroke()
        candleLinePath.lineWidth = 1
        candleLinePath.stroke()
        candleBodyPath.lineWidth = 0
        candleBodyPath.fill()
    }
    
    func drawHoveringCandleLine() {
        guard let hoveringCandleIndex = currentHoveringCandle else {
            return
        }
        if hoveringCandleIndex < 0 || hoveringCandleIndex >= stockAggregate!.candles.count {
            return
        }
        if hoveringCandleIndex < startingCandleIndex || hoveringCandleIndex > endingCandleIndex {
            return
        }

        var xPosition: CGFloat = getCandleXPositionInViewForCandleIndex(candleIndex: hoveringCandleIndex)
        xPosition *= candleWidth
        let candleBodyPath = NSBezierPath(rect: NSRect(x: xPosition + xPositionOffset,
                                                       y: 0,
                                                       width: candleWidth,
                                                       height: bounds.height))
        let color: NSColor = .init(displayP3Red: 1, green: 1, blue: 1, alpha: 0.15)
        color.setFill()
        candleBodyPath.fill()
    }
    
    private func getCandleXPositionInViewForCandleIndex(candleIndex index: Int) -> CGFloat {
        // Get Candle Count From Starting Candle
        let candleDiff = index - startingCandleIndex
        return CGFloat(candleDiff)
    }
}
