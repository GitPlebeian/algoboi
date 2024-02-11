//
//  MLPredictor1.swift
//  stockInterface
//
//  Created by CHONK on 1/28/24.
//

import CoreML

class MLPredictor1 {
    
    static let shared = MLPredictor1()
    
    private let model: ForcastingModel1
    private var normalizingValues: [[Float]] = []

    init() {
        guard let loadedModel = try? ForcastingModel1(configuration: MLModelConfiguration()) else {
            fatalError("Couldn't load the model")
        }
        model = loadedModel
        
        if let stringData = SharedFileManager.shared.getDataFromPythonFile("bot3/ForcastingModel1ScalingParameters.txt") {
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
        guard let inputArray = try? MLMultiArray(shape: [1, 9], dataType: .float32) else {
            print("Failed to create input array")
            return nil
        }
        
        inputArray[0] = candlesToTarget as NSNumber
        inputArray[1] = indicatorData.macdDifferences[index] as NSNumber
        inputArray[2] = indicatorData.macdGreenLineLevels[index] as NSNumber
        inputArray[3] = indicatorData.macdGreenLineSlopes[index] as NSNumber
        inputArray[4] = indicatorData.macdRedLineSlopes[index] as NSNumber
        inputArray[5] = indicatorData.slopesOf9DayEMA[index] as NSNumber
        inputArray[6] = indicatorData.slopesOf25DayEMA[index] as NSNumber
        inputArray[7] = indicatorData.slopesOf50DaySMA[index] as NSNumber
        inputArray[8] = indicatorData.slopesOf200DaySMA[index] as NSNumber
        
        // * Percentage Change From Average Volume
        // * Volume * Close Market Cap Average
        //  STD Period 1
        // STD Period 2
        // STD Period Difference
        // Percentage Change
        // Hight Clow Low Open ratio
        
        DispatchQueue.concurrentPerform(iterations: inputArray.count) { i in
            let old = Float(truncating: inputArray[i])
            inputArray[i] = normalizeValue(originalValue: old, mean: normalizingValues[i][0], std: normalizingValues[i][1]) as NSNumber
        }
        
        // Load and Normalize
        
        
        
        
        guard let output = try? model.prediction(input: ForcastingModel1Input(x: inputArray)) else {
            print("Prediction failed")
            return nil
        }
        
//        let firstValue = output.output[0].floatValue
//        let secondValue = output.output[1].floatValue
//        let firstValue = output.
        return output.linear_2[0].floatValue
    }
}
