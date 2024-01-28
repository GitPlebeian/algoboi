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
            TerminalManager.shared.addText("Aggregate length is not equal too or longer than \(StockCalculations.StartAtElement)")
            return
        }
        
        var dataSets: [MLDatasetInputOutputCombined1] = []
        
        for i in (StockCalculations.StartAtElement - 1)..<aggregate.candles.count {
            
        }
//        let convertedData = StockCalculations.ConvertStockAggregateToMLTrainingData(aggregate)
//        SharedFileManager.shared.writeMLTrainingDataToFile(convertedData)
    }
}
