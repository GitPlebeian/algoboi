//
//  CircleView.swift
//  stockInterface
//
//  Created by CHONK on 9/29/23.
//

import Cocoa

class CircleView: NSView {
    
    init() {
        super.init(frame: .zero)
        self.wantsLayer = true
        self.layer?.backgroundColor = .white
    }
    
    required init?(coder: NSCoder) {
        super.init(coder: coder)
    }
    
    override func draw(_ dirtyRect: NSRect) {
        super.draw(dirtyRect)
        
        let diameter = min(self.bounds.width * 0.5, self.bounds.height * 0.5)

        let circleRect = NSRect(x: bounds.width / 2 - (diameter / 2),
                                y: bounds.height / 2 - (diameter / 2),
                                width: diameter,
                                height: diameter)
        
        // Draw the Circle
        let circlePath = NSBezierPath(ovalIn: circleRect)
        NSColor.red.setFill()
        circlePath.fill()
    }
}
