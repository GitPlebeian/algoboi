//
//  SaveCommand.swift
//  stockInterface
//
//  Created by Jackson Tubbs on 11/12/23.
//

import Foundation

class SaveCommand: Command {
    
    var name: String { "save" }

    func execute(with arguments: [String]) {

        guard let aggregate = ChartManager.shared.currentAggregate else {
            TerminalManager.shared.addText("No charted stock. Please run the \"chart\" command", type: .error)
            return
        }
        guard let indicatorData = ChartManager.shared.indicatorData else {
            TerminalManager.shared.addText("No indicator data for the current charted stock.", type: .error)
            return
        }
        
        if aggregate.candles.count < StockCalculations.StartAtElement {
            TerminalManager.shared.addText("Aggregate length is not equal too or longer than \(StockCalculations.StartAtElement)", type: .error)
            return
        }
        
        var dataSets: [MLDatasetInputOutputCombined1] = []
        
        for i in (StockCalculations.StartAtElement - 1)..<aggregate.candles.count {
//            if let datasetOutputs = MLDatasetGenerator.shared.calculateOutputsForIndex(index: i, aggregate: aggregate) {
//                let datasetInput = MLDatasetInput1(indicatorData: indicatorData, index: i)
//                let inputOutput = MLDatasetInputOutputCombined1(input: datasetInput, output: datasetOutput)
//                dataSets.append(inputOutput)
//            }
            let datasetOutputs = MLDatasetGenerator.shared.calculateOutputsForIndex(index: i, aggregate: aggregate)
            let datasetInput = MLDatasetInput1(indicatorData: indicatorData, index: i)
            for datasetOutput in datasetOutputs {
                dataSets.append(MLDatasetInputOutputCombined1(input: datasetInput, output: datasetOutput))
            }
        }
        
        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        if let jsonData = try? encoder.encode(dataSets),
           let jsonString = String(data: jsonData, encoding: .utf8) {
            var url = FileManager.default.urls(for: .desktopDirectory, in: .userDomainMask).first!
            url = url.appendingPathComponent("/algoboi/shared/datasets/AAASingleSet.json")
            do {
                try jsonString.write(to: url, atomically: false, encoding: .utf8)
            } catch {
                print("Failed to write JSON data: \(error.localizedDescription)")
            }
        }
        
        TerminalManager.shared.addText("Saved:\n\(aggregate.symbol)\n\(aggregate.name)\nDatasets: \(dataSets.count)")
    }
}
