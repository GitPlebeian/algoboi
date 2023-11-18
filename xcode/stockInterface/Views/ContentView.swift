//
//  ContentView.swift
//  stockInterface
//
//  Created by CHONK on 9/29/23.
//

import Cocoa
import ModelIO

class ContentView: NSView {
    
    // MARK: Properties
    
    // MARK: Subviews

    weak var stockView:          StockView!
    weak var terminalView:       ScrollingTerminalView!
    weak var frostedGlassEffect: NSVisualEffectView!
    
    // MARK: Init
    
    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        setupView()
    }
    
    required init?(coder: NSCoder) {
        super.init(coder: coder)
    }
    
    // MARK: Layout
    
    override func layout() {
        super.layout()
        frostedGlassEffect.frame = frame
//        stockView.frame = frame
    }
    
    override func keyDown(with event: NSEvent) {
        if event.keyCode == 123 || event.keyCode == 124 || event.keyCode == 125 || event.keyCode == 126 {return}
        super.keyDown(with: event)
    }
    
    // MARK: Setup Views
    
    private func setupView() {
        
        let frostedGlassEffect = NSVisualEffectView()
        frostedGlassEffect.appearance = .currentDrawing()
        frostedGlassEffect.blendingMode = .behindWindow
        frostedGlassEffect.state = .active
        addSubview(frostedGlassEffect, positioned: .below, relativeTo: nil)
        self.frostedGlassEffect = frostedGlassEffect
        
        let terminalView = ScrollingTerminalView()
        terminalView.translatesAutoresizingMaskIntoConstraints = false
        addSubview(terminalView)
        NSLayoutConstraint.activate([
            terminalView.topAnchor.constraint(equalTo: topAnchor),
            terminalView.trailingAnchor.constraint(equalTo: trailingAnchor),
            terminalView.bottomAnchor.constraint(equalTo: bottomAnchor),
            terminalView.widthAnchor.constraint(equalToConstant: 300)
        ])
        self.terminalView = terminalView
        
        let stockView = StockView()
        stockView.translatesAutoresizingMaskIntoConstraints = false
        addSubview(stockView)
        NSLayoutConstraint.activate([
            stockView.topAnchor.constraint(equalTo: topAnchor, constant: 0),
            stockView.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 0),
            stockView.trailingAnchor.constraint(equalTo: terminalView.leadingAnchor),
            stockView.bottomAnchor.constraint(equalTo: bottomAnchor, constant: 0)
            
//            stockView.centerXAnchor.constraint(equalTo: centerXAnchor),
//            stockView.centerYAnchor.constraint(equalTo: centerYAnchor),
//            stockView.widthAnchor.constraint(equalTo: widthAnchor, multiplier: 0.7),
//            stockView.heightAnchor.constraint(equalTo: heightAnchor, multiplier: 0.7)
        ])
        self.stockView = stockView
        ChartManager.shared.currentStockView = self.stockView
    }
}
