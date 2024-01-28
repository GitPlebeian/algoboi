//
//  MLDataset1.swift
//  stockInterface
//
//  Created by CHONK on 1/27/24.
//

import Foundation


struct MLDatasetInputOutputCombined1: Codable {
    var input: MLDatasetOutput
    var output: MLDatasetInput1
    
    static func write(dataSets: [MLDatasetInputOutputCombined1]) {
        let encoder = JSONEncoder()
        if let jsonData = try? encoder.encode(dataSets),
           let jsonString = String(data: jsonData, encoding: .utf8) {
            var url = FileManager.default.urls(for: .desktopDirectory, in: .userDomainMask).first!
            url = url.appendingPathComponent("/algoboi/shared/datasets/set1.json")
            do {
                try jsonString.write(to: url, atomically: true, encoding: .utf8)
            } catch {
                print("Failed to write JSON data: \(error.localizedDescription)")
            }
        }
    }
}

struct MLDatasetInput1: Codable {
    var slopeOf9DayEMA:     Float
    var slopeOf25DayEMA:    Float
    var slopeOf50DaySMA:    Float
    var slopeOf200DaySMA:   Float
    var macdGreenLineLevel: Float
    var macdGreenLineSlope: Float
    var macdRedLineSlope:   Float
    var macdDifference:     Float
}
