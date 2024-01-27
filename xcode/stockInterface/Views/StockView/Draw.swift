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
        drawLines()
        
        drawAuxBlackSquareOverlay()
        
        drawAuxViews()
    }
    
    private func drawAuxViews() {
        
        var heightStartingPoint: CGFloat = 0
        
        for auxGraph in self.auxViews {
            drawAuxBackground(auxHeight: auxGraph.height, startingHeight: heightStartingPoint)
            drawAuxBars(auxHeight: auxGraph.height, bars: auxGraph.bars, startingHeight: heightStartingPoint)
            drawAuxLines(auxHeight: auxGraph.height, lines: auxGraph.lines, startingHeight: heightStartingPoint)
            heightStartingPoint += auxGraph.height + Style.ChartDividerWidth
        }
    }
    
    private func drawAuxBackground(auxHeight: CGFloat, startingHeight: CGFloat) {
        
        let path = NSBezierPath(rect: NSRect(x: 0, y: startingHeight + auxHeight, width: bounds.width, height: Style.ChartDividerWidth))
        CGColor.ChartDividerColor.NSColor().setFill()
        path.fill()
    }
    
    private func drawAuxBars(auxHeight: CGFloat, bars: [StockViewAuxGraphBars], startingHeight: CGFloat) {
        if bars.count == 0 {return}
        for i in -2..<visibleCandles + 2 {
            let dataIndex = i + startingCandleIndex
            if dataIndex < 0 || dataIndex >= bars.count {continue}
            let yPos = startingHeight + auxHeight * bars[dataIndex].y
            let xPos = CGFloat(i) * candleWidth + xPositionOffset
            let height = auxHeight * bars[dataIndex].height
            let path = NSBezierPath(rect: NSRect(x: xPos, y: yPos, width: candleWidth, height: height))
            bars[dataIndex].color.setFill()
            path.fill()
        }
    }
    
    private func drawAuxLines(auxHeight: CGFloat, lines: [StockViewAuxGraphLines], startingHeight: CGFloat) {
        if lines.count == 0 {return}
        print(startingCandleIndex)
        var normalizedValues: [Float] = []
        
        // Visible Normalize | Expensive
        for i in -2..<visibleCandles + 2 {
            let dataIndex = i + startingCandleIndex
            for line in lines {
                if dataIndex < 0 || dataIndex >= line.yValues.count {continue}
                normalizedValues.append(Float(line.yValues[dataIndex]))
            }
        }
        normalizedValues = StockCalculations.NormalizeFromMinusOneToOne(array: normalizedValues)
        
        let dataValues = normalizedValues.map{ CGFloat($0)}
        
        for (lineIndex, line) in lines.enumerated() {
            let linePath = NSBezierPath()
            var didStartPath = false
            
            
            for i in 0..<dataValues.count / lines.count {
                let dataIndex = i * lines.count
                if dataIndex < 0 || dataIndex >= dataValues.count {continue}
                
                var yPos = dataValues[dataIndex + lineIndex]
                yPos = auxHeight / 2 + auxHeight / 2 * yPos
                
                var xPos: CGFloat
                if  startingCandleIndex < 0 {
                    xPos = CGFloat(i - startingCandleIndex) * candleWidth + xPositionOffset + candleWidth / 2
                } else {
                    xPos = CGFloat(i - 2) * candleWidth + xPositionOffset + candleWidth / 2
                }
                if didStartPath == false {
                    didStartPath = true
                    linePath.move(to: NSPoint(x: xPos, y: startingHeight + yPos))
                } else {
                    linePath.line(to: NSPoint(x: xPos, y: startingHeight + yPos))
                }
            }
            line.color.setStroke()
            linePath.lineWidth = 2
            linePath.stroke()
        }
    }
    
    private func drawAuxBlackSquareOverlay() {
        let path = NSBezierPath(rect: NSRect(x: 0,
                                             y: 0,
                                             width: bounds.width,
                                             height: self.stockViewAuxYOffset))
        let color = NSColor.black
        color.setFill()
        path.fill()
    }
    
    private func calculateDrawingValues() {
        xPositionOffset = lastCandleMissingSpace / 2 + panDistance
    }
    
    func drawGreenRedBars() {

        for e in coloredFullHeightBars {
            if e.0 < 0 || e.0 >= stockAggregate!.candles.count {
                continue
            }
            if e.0 < startingCandleIndex || e.0 > endingCandleIndex {
                continue
            }
            var xPosition: CGFloat = getCandleXPositionInViewForCandleIndex(candleIndex: e.0)
            xPosition *= candleWidth
            let candleBodyPath = NSBezierPath(rect: NSRect(x: xPosition + xPositionOffset,
                                                           y: self.stockViewAuxYOffset,
                                                           width: candleWidth,
                                                           height: self.stockViewHeight))
            let color = e.1
            color.setFill()
            candleBodyPath.fill()
        }
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
            let candleBodyHeight = self.stockViewHeight / (CGFloat(range) / (abs(CGFloat(candle.open - candle.close))))
            let candleBodyPath = NSBezierPath(rect: NSRect(x: candleBodyX + xPositionOffset,
                                                           y: CGFloat(candleBodyYPos) * self.stockViewHeight + self.stockViewAuxYOffset,
                                                           width: candleWidth,
                                                           height: candleBodyHeight))
            
            let candleLineXPos = (CGFloat(index) * (candleWidth)) + candleWidth / 2
            let candleLineYPos = 1 - ((maxValue - candle.low) / range)
            let candleLinePath = NSBezierPath()
            candleLinePath.move(to: CGPoint(x: candleLineXPos + xPositionOffset,
                                            y: (CGFloat(candleLineYPos) * self.stockViewHeight) + self.stockViewAuxYOffset))
            candleLinePath.line(to: CGPoint(x: candleLineXPos + xPositionOffset,
                                               y: self.stockViewAuxYOffset + (CGFloat(candleLineYPos) * self.stockViewHeight) + (self.stockViewHeight / (CGFloat(range) / (abs(CGFloat(candle.high - candle.low)))))))
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
        let candleBodyHeight = self.stockViewHeight / (CGFloat(range) / (abs(CGFloat(candle.open - candle.close))))
        let candleBodyPath = NSBezierPath(rect: NSRect(x: candleBodyX + xPositionOffset,
                                                       y: CGFloat(candleBodyYPos) * self.stockViewHeight + self.stockViewAuxYOffset,
                                                       width: candleWidth,
                                                       height: candleBodyHeight))
        
        let candleLineXPos = (CGFloat(positionIndex) * (candleWidth)) + candleWidth / 2
        let candleLineYPos = 1 - ((maxValue - candle.low) / range)
        let candleLinePath = NSBezierPath()
        candleLinePath.move(to: CGPoint(x: candleLineXPos + xPositionOffset,
                                        y: (CGFloat(candleLineYPos) * self.stockViewHeight) + self.stockViewAuxYOffset))
        candleLinePath.line(to: CGPoint(x: candleLineXPos + xPositionOffset,
                                           y: self.stockViewAuxYOffset + (CGFloat(candleLineYPos) * self.stockViewHeight) + (self.stockViewHeight / (CGFloat(range) / (abs(CGFloat(candle.high - candle.low)))))))
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
                                                       y: self.stockViewAuxYOffset,
                                                       width: candleWidth,
                                                       height: self.stockViewHeight))
        let color: NSColor = .init(displayP3Red: 1, green: 1, blue: 1, alpha: 0.15)
        color.setFill()
        candleBodyPath.fill()
    }
    
    private func drawLines() {
        for line in coloredLines {
            let linePath = NSBezierPath()
            var didStartPath = false
            for i in -2..<visibleCandles + 2 {
                let dataIndex = i + startingCandleIndex
                if dataIndex < 0 || dataIndex >= line.0.count {continue}
                let yPos = 1 - ((maxValue - line.0[dataIndex]) / range)
                let xPos = CGFloat(i) * candleWidth + xPositionOffset + candleWidth / 2
                if didStartPath == false {
                    didStartPath = true
                    linePath.move(to: NSPoint(x: xPos, y: self.stockViewAuxYOffset + CGFloat(yPos) * self.stockViewHeight))
                } else {
                    linePath.line(to: NSPoint(x: xPos, y: self.stockViewAuxYOffset + CGFloat(yPos) * self.stockViewHeight))
                }
            }
            line.1.setStroke()
            linePath.lineWidth = 2
            linePath.stroke()
        }
    }
    
    private func getCandleXPositionInViewForCandleIndex(candleIndex index: Int) -> CGFloat {
        // Get Candle Count From Starting Candle
        let candleDiff = index - startingCandleIndex
        return CGFloat(candleDiff)
    }
}
