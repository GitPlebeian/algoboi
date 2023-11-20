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
        let convertedData = StockCalculations.ConvertStockAggregateToMLTrainingData(aggregate)
        SharedFileManager.shared.writeMLTrainingDataToFile(convertedData)
    }
}
