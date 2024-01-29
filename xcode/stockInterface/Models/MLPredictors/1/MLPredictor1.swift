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

    init() {
        guard let loadedModel = try? ForcastingModel1(configuration: MLModelConfiguration()) else {
            fatalError("Couldn't load the model")
        }
        model = loadedModel
    }
    
    func makePrediction(indicatorData: IndicatorData, index: Int) -> [Float]? {
        guard let inputArray = try? MLMultiArray(shape: [1, 8], dataType: .float32) else {
            print("Failed to create input array")
            return nil
        }
        print(inputArray)
        
        inputArray[0] = indicatorData.macdDifferences[index] as NSNumber
        inputArray[1] = indicatorData.macdGreenLineLevels[index] as NSNumber
        inputArray[2] = indicatorData.macdGreenLineSlopes[index] as NSNumber
        inputArray[3] = indicatorData.macdRedLineSlopes[index] as NSNumber
        inputArray[4] = indicatorData.slopesOf9DayEMA[index] as NSNumber
        inputArray[5] = indicatorData.slopesOf25DayEMA[index] as NSNumber
        inputArray[6] = indicatorData.slopesOf50DaySMA[index] as NSNumber
        inputArray[7] = indicatorData.slopesOf200DaySMA[index] as NSNumber
        
        print(inputArray)
        
        guard let output = try? model.prediction(input: ForcastingModel1Input(x: inputArray)) else {
            print("Prediction failed")
            return nil
        }
        
//        let firstValue = output.output[0].floatValue
//        let secondValue = output.output[1].floatValue
//        let firstValue = output.
        return [output.linear_2[0].floatValue, output.linear_2[1].floatValue]
    }
}
