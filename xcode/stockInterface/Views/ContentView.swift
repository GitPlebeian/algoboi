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
    weak var frostedGlassEffect: NSVisualEffectView!
    
    // MARK: Init
    
    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        setupView()
//        TickerDownload.shared.getAlpacaStock(ticker: "AAPL", year: 2) { messageReturn, model in
//            DispatchQueue.main.async {
//                guard let stock = model else {return}
//
//                self.stockView.stockAggregate = stock
//            }
//        }
        TickerDownload.shared.getAlpacaStockMinuteInterval(ticker: "fngr", date: (10,3,2023)) { _, stockAggregate in
            DispatchQueue.main.async {
                guard let stockAggregate = stockAggregate else {return}
                self.stockView.stockAggregate = stockAggregate
                
//                FileManager.shared.writeStockAggregateToSharedFoler(entity: stockAggregate)
                SharedFileManager.shared.writeStockAggregateToTestFile(stockAggregate)
            }
        }
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
    
    // MARK: Setup Views
    
    private func setupView() {
        let frostedGlassEffect = NSVisualEffectView()
        frostedGlassEffect.appearance = .currentDrawing()
        frostedGlassEffect.blendingMode = .behindWindow
        frostedGlassEffect.state = .active
        addSubview(frostedGlassEffect, positioned: .below, relativeTo: nil)
        self.frostedGlassEffect = frostedGlassEffect
        
        let stockView = StockView()
        stockView.translatesAutoresizingMaskIntoConstraints = false
        addSubview(stockView)
        NSLayoutConstraint.activate([
//            stockView.topAnchor.constraint(equalTo: topAnchor, constant: 20),
//            stockView.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 20),
//            stockView.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -20),
//            stockView.bottomAnchor.constraint(equalTo: bottomAnchor, constant: -20)
            
            stockView.centerXAnchor.constraint(equalTo: centerXAnchor),
            stockView.centerYAnchor.constraint(equalTo: centerYAnchor),
            stockView.widthAnchor.constraint(equalTo: widthAnchor, multiplier: 0.7),
            stockView.heightAnchor.constraint(equalTo: heightAnchor, multiplier: 0.7)
        ])
        self.stockView = stockView
    }
}
