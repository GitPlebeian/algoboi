//
//  CelledView.swift
//  stockInterface
//
//  Created by Jackson Tubbs on 10/13/23.
//

import Cocoa

class CelledView: NSView {

    // MARK: Init
    
    init(x: Int, y: Int, width: Int, height: Int) {
        super.init(frame: NSRect(x: Style.CellWidth * CGFloat(x),
                                 y: Style.CellHeight * CGFloat(y),
                                 width: Style.CellWidth * CGFloat(width),
                                 height: Style.CellHeight * CGFloat(height)))
        setupView()

    }
    
    required init?(coder: NSCoder) {
        super.init(coder: coder)
    }
    
    // Setup View
    
    private func setupView() {
        wantsLayer = true
        layer?.backgroundColor = .white
    }
}
