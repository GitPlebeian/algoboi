//
//  StockViewAuxGraphProperties.swift
//  stockInterface
//
//  Created by CHONK on 11/26/23.
//

import Cocoa

struct StockViewAuxGraphProperties {
    let height: CGFloat
    var bars: [StockViewAuxGraphBars] = []
    var lines: [StockViewAuxGraphLines] = []
}

struct StockViewAuxGraphBars {
    let y: CGFloat
    let height: CGFloat
    let color: NSColor
}

struct StockViewAuxGraphLines {
    let yValues: [CGFloat]
    let color: NSColor
}
