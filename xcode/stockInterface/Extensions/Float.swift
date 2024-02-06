//
//  Float.swift
//  stockInterface
//
//  Created by CHONK on 11/20/23.
//

import Foundation

extension Float {
    
//    func toRoundedString(precision: Int) -> String {
//        let multiplier = pow(10, Float(precision))
//        let roundedValue = roundf(self * multiplier) / multiplier
//        return String(format: "%.\(precision)f", roundedValue)
//        
//    }
    
    func toRoundedString(precision: Int) -> String {
        if self < 1000 {
            let multiplier = pow(10, Float(precision))
            let roundedValue = roundf(self * multiplier) / multiplier
            return String(format: "%.\(precision)f", roundedValue)
        }
        
        let roundedValue = floorf(self)
        
        // NumberFormatter to format the number with thousand separators.
        let numberFormatter = NumberFormatter()
        numberFormatter.numberStyle = .decimal // Use decimal style to get commas.
        numberFormatter.minimumFractionDigits = 0
        numberFormatter.maximumFractionDigits = 0
        
        // Convert the rounded float to NSNumber and format it using NumberFormatter.
        guard let formattedString = numberFormatter.string(from: NSNumber(value: roundedValue)) else {
            return "\(roundedValue)" // Fallback to the rounded value without formatting in case of an error.
        }
        
        return formattedString
    }
}
