//
//  Float.swift
//  stockInterface
//
//  Created by CHONK on 11/20/23.
//

import Foundation

extension Float {
    
    func toRoundedString(precision: Int) -> String {
        let multiplier = pow(10, Float(precision))
        let roundedValue = roundf(self * multiplier) / multiplier
        return String(format: "%.\(precision)f", roundedValue)
        
    }
}
