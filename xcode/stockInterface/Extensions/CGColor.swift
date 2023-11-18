//
//  CGColor.swift
//  stockInterface
//
//  Created by CHONK on 10/16/23.
//

import Cocoa

extension CGColor {
    
    static func FromHex(_ hex: String) -> CGColor {
        let r, g, b, a: CGFloat
        
//        let start = hex.index(hex.startIndex, offsetBy: 1)
//        let hexColor = String(hex[start...])
        let scanner = Scanner(string: hex)
        var hexNumber: UInt64 = 0
        
        if hex.count == 8 {
            if scanner.scanHexInt64(&hexNumber) {
                r = CGFloat((hexNumber & 0xff000000) >> 24) / 255
                g = CGFloat((hexNumber & 0x00ff0000) >> 16) / 255
                b = CGFloat((hexNumber & 0x0000ff00) >> 8) / 255
                a = CGFloat(hexNumber & 0x000000ff) / 255
                return CGColor.init(red: r, green: g, blue: b, alpha: a)
            }
        } else {
            if scanner.scanHexInt64(&hexNumber) {
                r = CGFloat((hexNumber & 0xff0000) >> 16) / 255
                g = CGFloat((hexNumber & 0x00ff00) >> 8) / 255
                b = CGFloat(hexNumber & 0x0000ff) / 255
                return CGColor.init(red: r, green: g, blue: b, alpha: 1)
            }
        }
        fatalError("Invalid Hex")
    }
    
    func NSColor() -> NSColor { return AppKit.NSColor(cgColor: self)!}
    
    static let CelledBackground    = FromHex("030303")
    static let InputBackground     = FromHex("131313")
    static let TerminalUserInput   = FromHex("002b70")
    static let TerminalError       = FromHex("700707")
    static let ChartPurchasedLines = FromHex("32a87930")
    static let ChartSoldLines      = FromHex("c4553330")
    
}
