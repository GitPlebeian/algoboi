//
//  StockView.swift
//  stockInterface
//
//  Created by CHONK on 9/29/23.
//

import Cocoa

class StockView: NSView {
    
    // MARK: Subviews
    
    // MARK: Properties
    
    var stockAggregate: StockAggregate? = nil {
        didSet {
            setVisibleCandleToMax()
            setMinimumCandleWidth()
            setNeedsDisplay(bounds)
        }
    }
    let margin: CGFloat = 30
    
    
    var minimumCandleWidth: CGFloat!
    var candleWidth: CGFloat = 80
    
    
    var maxValue:              Float               = -.infinity
    var minValue:              Float               = .infinity
    var range:                 Float = 0
    var endingCandleIndex: Int = 0
    var startingCandleIndex: Int = 0
    var visibleCandles: Int = 0
    var lastCandleMissingSpace: CGFloat = 0
    
    // Mouse Over Candle
    var currentCandleMountOver: Int? = nil
    
    // Mouse Hovering
    var trackingArea: NSTrackingArea!
    // Candle Hovering
    var currentHoveringCandle: Int?
    var currentHoveringCandleXPosition: CGFloat?
    
    // MARK: Additonal View Items
    
    var coloredFullHeightBars: [(Int, NSColor)] = [(5, CGColor.ChartPurchasedLines.NSColor())]
    
    // MARK: Gestures
    
    // Pinch - Zooming
    var pinchGesture:  NSMagnificationGestureRecognizer!
    var previousScale: CGFloat  = 0
    var currentScale:  CGFloat  = 0
    var maxZoom:       CGFloat  = 5
    var minZoom:       CGFloat  = 0.5
    var previousCandleWidth: CGFloat!
    
    // Pan - Scrolling
    var panDistance:           CGFloat = 0
    
    // Mouse Position
    var lastMousePosition: NSPoint?
    
    // MARK: Drawing Calculations
    
    var xPositionOffset: CGFloat = 0
    
    // MARK: Init
    
    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        previousCandleWidth = candleWidth
        setupView()
    }
    
    required init?(coder: NSCoder) {
        super.init(coder: coder)
    }
    
    // MARK: Layout
    
    override func layout() {
        super.layout()

    }
    
    // MARK: Public
    
    func clearAllAdditionalItems() {
        coloredFullHeightBars = []
        setNeedsDisplay(bounds)
    }
    
    func setColoredFullHeightBars(bars: [(Int, NSColor)]) {
        coloredFullHeightBars = bars
        setNeedsDisplay(bounds)
    }
    
    // MARK: Tracking Area
    
    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        if let existingTrackingArea = self.trackingArea {
            self.removeTrackingArea(existingTrackingArea)
        }
        // Create a new tracking area
        let options: NSTrackingArea.Options = [.activeAlways, .mouseMoved, .activeInKeyWindow, .mouseEnteredAndExited]
        self.trackingArea = NSTrackingArea(rect: self.bounds, options: options, owner: self, userInfo: nil)
        self.addTrackingArea(self.trackingArea)
    }
    
    // MARK: Actions
    
    // Pan Left and Right
    override func scrollWheel(with event: NSEvent) {
        guard let stockAggregate = stockAggregate else {return}
        super.scrollWheel(with: event)
        
        let deltaX = event.scrollingDeltaX
        panDistance += deltaX
        
        if panDistance >= candleWidth {
            let candleScrollAmount = Int(panDistance / candleWidth)
            if endingCandleIndex - candleScrollAmount <= 0 {
                endingCandleIndex = 0
                panDistance = 0
                updateForCurrentMousePosition()
                setNeedsDisplay(bounds)
                return
            }
            panDistance = panDistance.truncatingRemainder(dividingBy: candleWidth)
            endingCandleIndex -= candleScrollAmount
        } else if panDistance <= -candleWidth {
            let candleScrollAmount = Int(abs(panDistance) / candleWidth)
            if (endingCandleIndex + candleScrollAmount) - visibleCandles + 1 >= stockAggregate.candles.count {
                endingCandleIndex = stockAggregate.candles.count + visibleCandles - 2
                panDistance = 0
                updateForCurrentMousePosition()
                setNeedsDisplay(bounds)
                return
            }
            panDistance = panDistance.truncatingRemainder(dividingBy: candleWidth)
            endingCandleIndex += candleScrollAmount
        }
        if endingCandleIndex == stockAggregate.candles.count + visibleCandles - 2 && panDistance < 0 {
            panDistance = 0
            return
        } else if endingCandleIndex == 0 && panDistance > 0 {
            panDistance = 0
            return
        }
        updateForCurrentMousePosition()
        setNeedsDisplay(bounds)
    }
    
    // Zooming into chart
    @objc private func pinched() {
        let scale = pinchGesture.magnification * 0.5
        updateValuesForScaleChanged(scale: scale)
    }
    
    // Mouse Moved Over View
    
    override func mouseMoved(with event: NSEvent) {
        if stockAggregate == nil {return}
        lastMousePosition = self.convert(event.locationInWindow, from: nil)
        updateForCurrentMousePosition()
    }
    
    override func mouseExited(with event: NSEvent) {
        lastMousePosition = nil
        currentHoveringCandle = nil
        setNeedsDisplay(bounds)
    }
    
    func updateForCurrentMousePosition() {
        guard let lastMousePosition = lastMousePosition else {return}
        let newHoveringCandle = getCandleIndexForMousePositionInView(point: lastMousePosition)
        guard let newHoveringCandle = newHoveringCandle else {return}
        if let currentHoveringCandle = self.currentHoveringCandle {
            if newHoveringCandle != currentHoveringCandle {
                setNeedsDisplay(bounds)
            }
        } else {
            setNeedsDisplay(bounds)
        }
        self.currentHoveringCandle = newHoveringCandle
        var x = CGFloat(newHoveringCandle - startingCandleIndex)
        x *= candleWidth
        x += xPositionOffset
        self.currentHoveringCandleXPosition = x
    }
    
    // MARK: Setup View
    
    private func setupView() {
        self.wantsLayer = true
//        self.layer?.backgroundColor = .CelledBackground
        self.layer?.masksToBounds = true
        pinchGesture = NSMagnificationGestureRecognizer(target: self, action: #selector(pinched))
        self.addGestureRecognizer(pinchGesture)
        
    }
}
