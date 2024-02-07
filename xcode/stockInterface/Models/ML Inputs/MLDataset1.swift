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
        
        let dataset10Percent = SampleArray(dataSets, percentageToReturn: 0.1)
        let dataset1Percent  = SampleArray(dataSets, percentageToReturn: 0.01)
        
        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        if let jsonData = try? encoder.encode(dataSets),
           let jsonString = String(data: jsonData, encoding: .utf8) {
            var url = FileManager.default.urls(for: .desktopDirectory, in: .userDomainMask).first!
            url = url.appendingPathComponent("/algoboi/shared/datasets/set1FullSet.json")
            do {
                try jsonString.write(to: url, atomically: false, encoding: .utf8)
            } catch {
                print("Failed to write JSON data: \(error.localizedDescription)")
            }
        }
        if let jsonData = try? encoder.encode(dataset10Percent),
           let jsonString = String(data: jsonData, encoding: .utf8) {
            var url = FileManager.default.urls(for: .desktopDirectory, in: .userDomainMask).first!
            url = url.appendingPathComponent("/algoboi/shared/datasets/set1TenPercent.json")
            do {
                try jsonString.write(to: url, atomically: false, encoding: .utf8)
            } catch {
                print("Failed to write JSON data: \(error.localizedDescription)")
            }
        }
        if let jsonData = try? encoder.encode(dataset1Percent),
           let jsonString = String(data: jsonData, encoding: .utf8) {
            var url = FileManager.default.urls(for: .desktopDirectory, in: .userDomainMask).first!
            url = url.appendingPathComponent("/algoboi/shared/datasets/set1OnePercent.json")
            do {
                try jsonString.write(to: url, atomically: false, encoding: .utf8)
            } catch {
                print("Failed to write JSON data: \(error.localizedDescription)")
            }
        }
    }
    
    
    private static func SampleArray(_ inputArray: [MLDatasetInputOutputCombined1], percentageToReturn: Float) -> [MLDatasetInputOutputCombined1] {
        guard percentageToReturn > 0 && percentageToReturn <= 1 else {
            fatalError("Percentage should be between 0 and 1")
        }
        
        let sampleSize = Int(Float(inputArray.count) * percentageToReturn)
        var sampledArray = [MLDatasetInputOutputCombined1]()
        var sampledIndexes = Set<Int>()
        
        while sampledIndexes.count < sampleSize {
            let randomIndex = Int.random(in: 0..<inputArray.count)
            if !sampledIndexes.contains(randomIndex) {
                sampledIndexes.insert(randomIndex)
                sampledArray.append(inputArray[randomIndex])
            }
        }
        
        return sampledArray
    }
}

struct MLDatasetInput1: Codable {
    var candlesToTarget:    Float
    var slopeOf9DayEMA:     Float
    var slopeOf25DayEMA:    Float
    var slopeOf50DaySMA:    Float
    var slopeOf200DaySMA:   Float
    var macdGreenLineLevel: Float
    var macdGreenLineSlope: Float
    var macdRedLineSlope:   Float
    var macdDifference:     Float
    
    init(indicatorData: IndicatorData, index i: Int, candlesToTarget: Float) {
        self.candlesToTarget = candlesToTarget
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
