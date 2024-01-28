//
//  MLDatasetInputOutput1.swift
//  stockInterface
//
//  Created by CHONK on 1/27/24.
//

import Foundation

struct MLDatasetInputOutput1: Codable {
    
    var slopeOf9DayEMA: Float
    var slopeOf25DayEMA: Float
    var slopeOf50DaySMA: Float
    var slopeOf200DaySMA: Float
    var macdBaseLineLevel:   Float
    var macdBaseLineSlope:   Float
    var macdSignalLineSlope: Float
    var macdDifference:      Float
    
    init(closes: [Float], slopeOf9DayEMA: [Float], slopeOf25DayEMA: [Float]) {
        self.closes = closes
        self.slopeOf9DayEMA = StockCalculations.Normalize(slopeOf9DayEMA)
        self.slopeOf25DayEMA = StockCalculations.Normalize(slopeOf25DayEMA)
    }
    
    func toNestedArray() -> [[Float]] {
        // Use Mirror to reflect the properties of the struct
        let mirror = Mirror(reflecting: self)
        
        // Create an array to hold the final nested array
        var result = [[Float]]()
        
        // Get the first value from each property to determine the count
        guard let firstProperty = mirror.children.first?.value as? [Float],
              !firstProperty.isEmpty else {
            return result
        }
        
        for i in 0..<firstProperty.count {
            var currentRow: [Float] = []
            
            // Iterate over each property and append the i'th value to currentRow
            for case let (_, value) in mirror.children {
                if let arrayValue = value as? [Float], arrayValue.indices.contains(i) {
                    currentRow.append(arrayValue[i])
                }
            }
            
            // Append the currentRow to the result
            result.append(currentRow)
        }
        return Array(result[(StockCalculations.StartAtElement - 1)...])
    }
    
}
