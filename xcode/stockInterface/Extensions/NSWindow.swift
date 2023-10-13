//
//  NSWindow.swift
//  stockInterface
//
//  Created by CHONK on 10/2/23.
//

import Cocoa

extension NSWindow {
    var titlebarHeight: CGFloat {
        frame.height - contentRect(forFrameRect: frame).height
    }
}
