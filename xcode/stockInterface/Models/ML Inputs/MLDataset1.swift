//
//  MLDataset1.swift
//  stockInterface
//
//  Created by CHONK on 1/27/24.
//

import Foundation


struct MLDatasetInputOutputCombined1: Codable {
    let input: MLDatasetInput1
    let output: MLDatasetOutput
    
//    init(i: MLDatasetOutput, o: MLDatasetInput1) {
//        self.input = i
//        self.output = o
//    }
    
    static func Write(dataSets: [MLDatasetInputOutputCombined1]) {
        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        if let jsonData = try? encoder.encode(dataSets),
           let jsonString = String(data: jsonData, encoding: .utf8) {
            var url = FileManager.default.urls(for: .desktopDirectory, in: .userDomainMask).first!
            url = url.appendingPathComponent("/algoboi/shared/datasets/set1.json")
            do {
                try jsonString.write(to: url, atomically: false, encoding: .utf8)
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
    
    init(indicatorData: IndicatorData, index i: Int) {
        self.slopeOf9DayEMA = indicatorData.slopesOf9DayEMA[i]
        self.slopeOf25DayEMA = indicatorData.slopesOf25DayEMA[i]
        self.slopeOf50DaySMA = indicatorData.slopesOf50DaySMA[i]
        self.slopeOf200DaySMA = indicatorData.slopesOf200DaySMA[i]
        self.macdGreenLineLevel = indicatorData.macdGreenLineLevels[i]
        self.macdGreenLineSlope = indicatorData.macdGreenLineSlopes[i]
        self.macdRedLineSlope = indicatorData.macdRedLineSlopes[i]
        self.macdDifference = indicatorData.macdDifferences[i]
    }
}
