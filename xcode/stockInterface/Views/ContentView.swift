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
    weak var textValueView:      TextValueView!
    weak var terminalView:       ScrollingTerminalView!
//    weak var frostedGlassEffect: NSVisualEffectView!
    
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
//        frostedGlassEffect.frame = frame
//        stockView.frame = frame
    }
    
    override func keyDown(with event: NSEvent) {
        if event.keyCode == 123 || event.keyCode == 124 || event.keyCode == 125 || event.keyCode == 126 {return}
        super.keyDown(with: event)
    }
    
    // MARK: Setup Views
    
    private func setupView() {
        
        wantsLayer = true
        layer?.backgroundColor = .black
//        let frostedGlassEffect = NSVisualEffectView()
//        frostedGlassEffect.appearance = .currentDrawing()
//        frostedGlassEffect.blendingMode = .behindWindow
//        frostedGlassEffect.state = .active
//        addSubview(frostedGlassEffect, positioned: .below, relativeTo: nil)
//        self.frostedGlassEffect = frostedGlassEffect
        
        let textValueView = TextValueView()
        textValueView.translatesAutoresizingMaskIntoConstraints = false
        addSubview(textValueView)
        NSLayoutConstraint.activate([
            textValueView.topAnchor.constraint(equalTo: topAnchor),
            textValueView.trailingAnchor.constraint(equalTo: trailingAnchor),
            textValueView.widthAnchor.constraint(equalToConstant: 300),
            textValueView.heightAnchor.constraint(equalToConstant: 250)
        ])
        self.textValueView = textValueView
        LabelValueController.shared.setView(textValueView)
        
        let textValueViewBottomDivider = NSView()
        textValueViewBottomDivider.wantsLayer = true
        textValueViewBottomDivider.layer?.backgroundColor = .ChartDividerColor
        textValueViewBottomDivider.translatesAutoresizingMaskIntoConstraints = false
        addSubview(textValueViewBottomDivider)
        NSLayoutConstraint.activate([
            textValueViewBottomDivider.topAnchor.constraint(equalTo: textValueView.bottomAnchor),
            textValueViewBottomDivider.trailingAnchor.constraint(equalTo: textValueView.trailingAnchor),
            textValueViewBottomDivider.leadingAnchor.constraint(equalTo: textValueView.leadingAnchor),
            textValueViewBottomDivider.heightAnchor.constraint(equalToConstant: Style.ChartDividerWidth)
        ])
        
        let terminalView = ScrollingTerminalView()
        terminalView.translatesAutoresizingMaskIntoConstraints = false
        addSubview(terminalView)
        NSLayoutConstraint.activate([
            terminalView.topAnchor.constraint(equalTo: textValueViewBottomDivider.bottomAnchor),
            terminalView.trailingAnchor.constraint(equalTo: trailingAnchor),
            terminalView.bottomAnchor.constraint(equalTo: bottomAnchor),
            terminalView.widthAnchor.constraint(equalToConstant: 300)
        ])
        self.terminalView = terminalView
        
        let dividerView = NSView()
        dividerView.wantsLayer = true
        dividerView.layer?.backgroundColor = .ChartDividerColor
        dividerView.translatesAutoresizingMaskIntoConstraints = false
        addSubview(dividerView)
        NSLayoutConstraint.activate([
            dividerView.trailingAnchor.constraint(equalTo: textValueView.leadingAnchor),
            dividerView.topAnchor.constraint(equalTo: topAnchor),
            dividerView.bottomAnchor.constraint(equalTo: bottomAnchor),
            dividerView.widthAnchor.constraint(equalToConstant: Style.ChartDividerWidth)
        ])
        
        let stockView = StockView()
        stockView.translatesAutoresizingMaskIntoConstraints = false
        addSubview(stockView)
        NSLayoutConstraint.activate([
            stockView.topAnchor.constraint(equalTo: topAnchor, constant: 0),
            stockView.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 0),
            stockView.trailingAnchor.constraint(equalTo: dividerView.leadingAnchor),
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
