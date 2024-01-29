//
//  GenerateEntireDatasetCommand.swift
//  stockInterface
//
//  Created by CHONK on 1/28/24.
//

import Foundation

class GenerateEntireDatasetCommand: Command {
    
    var name: String { "g" }

    func execute(with arguments: [String]) {

        var dataSets: [MLDatasetInputOutputCombined1] = []
        var droppedNoIndicatorDataCount: Int = 0
        var droppedNoEnoughLengthCount: Int = 0
        
        let allTickers = AllTickersController.shared.getAllTickers()
        
        for tickerElement in allTickers {
            let ticker = tickerElement.symbol
            guard let aggregateData = SharedFileManager.shared.getDataFromFile("/historicalData/\(ticker).json") else {
                TerminalManager.shared.addText("Aggregate File does not exist", type: .error)
                return
            }
            guard let indicatorData = SharedFileManager.shared.getDataFromFile("/indicatorData/\(ticker).json") else {
//                TerminalManager.shared.addText("Indicator File does not exist", type: .error)
                droppedNoIndicatorDataCount += 1
                continue
            }
            
            do {
                let jsonDecoder = JSONDecoder()
                let aggregate = try jsonDecoder.decode(StockAggregate.self, from: aggregateData)
                let indicator = try jsonDecoder.decode(IndicatorData.self, from: indicatorData)
                
                if aggregate.candles.count < StockCalculations.StartAtElement {
//                    TerminalManager.shared.addText("Aggregate length is not equal too or longer than \(StockCalculations.StartAtElement)")
                    droppedNoEnoughLengthCount += 1
                    continue
                }
                
//                for i in (StockCalculations.StartAtElement - 1)..<aggregate.candles.count {
//                    if let datasetOutput = MLDatasetGenerator.shared.calculateOutputForIndex(index: i, aggregate: aggregate) {
//                        let datasetInput = MLDatasetInput1(indicatorData: indicator, index: i)
//                        let inputOutput = MLDatasetInputOutputCombined1(input: datasetInput, output: datasetOutput)
//                        dataSets.append(inputOutput)
//                    }
//                }
                
            } catch let e {
                TerminalManager.shared.addText("Unable to unwrap data into aggregate: \(e)", type: .error)
            }
        }
        MLDatasetInputOutputCombined1.Write(dataSets: dataSets)
        TerminalManager.shared.addText("Saved \(dataSets.count) training data", type: .normal)
        TerminalManager.shared.addText("Dropped \(droppedNoIndicatorDataCount) becuase no indicator data", type: .normal)
        TerminalManager.shared.addText("Dropped \(droppedNoEnoughLengthCount) becuase not long enough", type: .normal)
    }
}
