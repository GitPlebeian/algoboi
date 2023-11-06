//
//  Text.swift
//  stockInterface
//
//  Created by CHONK on 11/6/23.
//

import Cocoa

class Text {
    
    static func HeightForString(string: String, width: CGFloat, font: NSFont) -> CGFloat {
        let textStorage = NSTextStorage(string: string)
        let textContainer = NSTextContainer(containerSize: NSSize(width: width, height: CGFloat.greatestFiniteMagnitude))
        let layoutManager = NSLayoutManager()
        
        layoutManager.addTextContainer(textContainer)
        textStorage.addLayoutManager(layoutManager)
        
        textStorage.addAttribute(.font, value: font, range: NSRange(location: 0, length: textStorage.length))
        textContainer.lineFragmentPadding = 0
        
        layoutManager.glyphRange(for: textContainer)
        let rect = layoutManager.usedRect(for: textContainer)
        
        return ceil(rect.height)
    }
}
