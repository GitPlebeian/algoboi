//
//  StockViewAuxGraphProperties.swift
//  stockInterface
//
//  Created by CHONK on 11/26/23.
//

import Cocoa

struct StockViewAuxGraphProperties {
    let height: CGFloat
    let bars: [StockViewAuxGraphBars]
}

struct StockViewAuxGraphBars {
    let y: CGFloat
    let height: CGFloat
    let color: NSColor
}
