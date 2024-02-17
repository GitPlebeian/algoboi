//
//  MLPredictor2.swift
//  stockInterface
//
//  Created by CHONK on 2/11/24.
//

import CoreML

class MLPredictor2 {
    
    static let shared = MLPredictor2()
    
    private let model: ForcastingModel2
    private let scalingParametersType: Int = 2
    private var normalizingValues: [[Float]] = []

    init() {
        guard let loadedModel = try? ForcastingModel2(configuration: MLModelConfiguration()) else {
            fatalError("Couldn't load the model")
        }
        model = loadedModel
        
        if let stringData = SharedFileManager.shared.getDataFromPythonFile("bot3/ForcastingModel\(scalingParametersType)ScalingParameters.txt") {
            guard let content = String(data: stringData, encoding: .utf8) else {
                fatalError()
            }
            // Split the content into lines
            let lines = content.components(separatedBy: "\n")
            
            // Create a two-dimensional array to store the numbers
            var twoDimensionalArray: [[Float]] = []
            
            for line in lines {
                if let parsedValues = parseLine(line) {
                    twoDimensionalArray.append([parsedValues.0, parsedValues.1])
                }
            }
            self.normalizingValues = twoDimensionalArray
        } else {
            fatalError()
        }
    }
    
    private func parseLine(_ line: String) -> (Float, Float)? {
        let components = line.split(separator: ",")
        guard components.count == 2,
              let firstNumber = Float(components[0]),
              let secondNumber = Float(components[1]) else {
            return nil
        }
        return (firstNumber, secondNumber)
    }
    
    private func normalizeValue(originalValue: Float, mean: Float, std: Float) -> Float {
        return (originalValue - mean) / std
    }

    
    func makePrediction(indicatorData: IndicatorData, index: Int, candlesToTarget: Int) -> Float? {
        guard let inputArray = try? MLMultiArray(shape: [1, 20], dataType: .float32) else {
            print("Failed to create input array")
            return nil
        }
        
        inputArray[0] = candlesToTarget as NSNumber
        inputArray[1] = indicatorData.cto[index] as NSNumber
        inputArray[2] = indicatorData.htc[index] as NSNumber
        inputArray[3] = indicatorData.htl[index] as NSNumber
        inputArray[4] = indicatorData.hto[index] as NSNumber
        inputArray[5] = indicatorData.ltc[index] as NSNumber
        inputArray[6] = indicatorData.lto[index] as NSNumber
        inputArray[7] = indicatorData.macdDifferences[index] as NSNumber
        inputArray[8] = indicatorData.macdGreenLineLevels[index] as NSNumber
        inputArray[9] = indicatorData.macdGreenLineSlopes[index] as NSNumber
        inputArray[10] = indicatorData.macdRedLineSlopes[index] as NSNumber
        inputArray[11] = indicatorData.percentageChange[index] as NSNumber
        inputArray[12] = indicatorData.slopesOf9DayEMA[index] as NSNumber
        inputArray[13] = indicatorData.slopesOf25DayEMA[index] as NSNumber
        inputArray[14] = indicatorData.slopesOf50DaySMA[index] as NSNumber
        inputArray[15] = indicatorData.slopesOf200DaySMA[index] as NSNumber
        inputArray[16] = indicatorData.standardDeviationForPercentageChange1[index] as NSNumber
        inputArray[17] = indicatorData.standardDeviationForPercentageChange2[index] as NSNumber
        inputArray[18] = indicatorData.stdDifference[index] as NSNumber
        inputArray[19] = indicatorData.volumeChange[index] as NSNumber
        
        DispatchQueue.concurrentPerform(iterations: inputArray.count) { i in
            let old = Float(truncating: inputArray[i])
            inputArray[i] = normalizeValue(originalValue: old, mean: normalizingValues[i][0], std: normalizingValues[i][1]) as NSNumber
        }
        
        // Load and Normalize
        
        
        
        
        guard let output = try? model.prediction(input: ForcastingModel2Input(x: inputArray)) else {
            print("Prediction failed")
            return nil
        }
        
//        let firstValue = output.output[0].floatValue
//        let secondValue = output.output[1].floatValue
//        let firstValue = output.
        return output.linear_3[0].floatValue
    }
}
