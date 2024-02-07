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
        let dataSetsLock = NSLock()
        
        var droppedNoIndicatorDataCount: Int = 0
        var droppedNoEnoughLengthCount: Int = 0
        
        let allTickers = AllTickersController.shared.getAllTickers()
        
        let group = DispatchGroup()

        let jsonDecoder = JSONDecoder()
        
        allTickers.forEach { tickerElement in
            group.enter() // Enter the group for each file

            DispatchQueue.global().async {
                let ticker = tickerElement.symbol
                guard let aggregateData = SharedFileManager.shared.getDataFromFile("/historicalData/\(ticker).json") else {
//                    TerminalManager.shared.addText("Aggregate File does not exist", type: .error)
                    print("Aggregate file does not exists")
                    return
                }
                guard let indicatorData = SharedFileManager.shared.getDataFromFile("/indicatorData/\(ticker).json") else {
    //                TerminalManager.shared.addText("Indicator File does not exist", type: .error)
                    droppedNoIndicatorDataCount += 1
                    group.leave()
                    return
                }
                
                do {
//                    let jsonDecoder = JSONDecoder()
                    let aggregate = try jsonDecoder.decode(StockAggregate.self, from: aggregateData)
                    let indicator = try jsonDecoder.decode(IndicatorData.self, from: indicatorData)
                    
                    if aggregate.candles.count < StockCalculations.StartAtElement {
    //                    TerminalManager.shared.addText("Aggregate length is not equal too or longer than \(StockCalculations.StartAtElement)")
                        droppedNoEnoughLengthCount += 1
                        group.leave()
                        return
                    }
                    var threadLocalDataset: [MLDatasetInputOutputCombined1] = []
                    for i in (StockCalculations.StartAtElement - 1)..<aggregate.candles.count {
                        if let datasetOutput = MLDatasetGenerator.shared.calculateTotalPercentageChangeForXChandlesToTarget(index: i, aggregate: aggregate, candlesToTarget: 1) {
                            let datasetInput = MLDatasetInput1(indicatorData: indicator, index: i, candlesToTarget: Float(1))
                            threadLocalDataset.append(MLDatasetInputOutputCombined1(input: datasetInput, output: datasetOutput))
                            
                        } else {break}
                    }
                    dataSetsLock.lock()
                    dataSets.append(contentsOf: threadLocalDataset)
                    dataSetsLock.unlock()
                } catch let e {
                    fatalError(e.localizedDescription)
                }
                group.leave()
            }
        }
        group.notify(queue: DispatchQueue.main) {
//            TerminalManager.shared.addText("All Files Proccessed")
            print("DONE")
            MLDatasetInputOutputCombined1.Write(dataSets: dataSets)
            TerminalManager.shared.addText("Saved \(dataSets.count) training data", type: .normal)
            TerminalManager.shared.addText("Dropped \(droppedNoIndicatorDataCount) becuase no indicator data", type: .normal)
            TerminalManager.shared.addText("Dropped \(droppedNoEnoughLengthCount) becuase not long enough", type: .normal)
        }
        
    }
}
