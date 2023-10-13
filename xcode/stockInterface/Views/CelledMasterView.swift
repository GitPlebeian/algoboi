//
//  CelledMasterView.swift
//  stockInterface
//
//  Created by Jackson Tubbs on 10/13/23.
//

import Cocoa

class CelledMasterView: NSView {

    // MARK: Properties
    
    // MARK: Subviews

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
        LayoutManager.shared.didUpdateMasterViewFrame(newFrame: frame)
    }
    
    // MARK: Setup Views
    
    private func setupView() {
//        
//        wantsLayer = true
//        layer?.backgroundColor = .white
//
        
        let frostedGlassEffect = NSVisualEffectView()
        frostedGlassEffect.appearance = .currentDrawing()
        frostedGlassEffect.blendingMode = .behindWindow
        frostedGlassEffect.state = .active
        addSubview(frostedGlassEffect, positioned: .below, relativeTo: nil)
        self.frostedGlassEffect = frostedGlassEffect
        
//        let stockView = StockView()
//        stockView.translatesAutoresizingMaskIntoConstraints = false
//        addSubview(stockView)
//        NSLayoutConstraint.activate([
////            stockView.topAnchor.constraint(equalTo: topAnchor, constant: 20),
////            stockView.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 20),
////            stockView.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -20),
////            stockView.bottomAnchor.constraint(equalTo: bottomAnchor, constant: -20)
//            
//            stockView.centerXAnchor.constraint(equalTo: centerXAnchor),
//            stockView.centerYAnchor.constraint(equalTo: centerYAnchor),
//            stockView.widthAnchor.constraint(equalTo: widthAnchor, multiplier: 0.7),
//            stockView.heightAnchor.constraint(equalTo: heightAnchor, multiplier: 0.7)
//        ])
//        self.stockView = stockView
    }
}