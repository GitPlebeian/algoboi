//
//  InputOutputGenerationCommand.swift
//  stockInterface
//
//  Created by CHONK on 11/19/23.
//

import Foundation

class InputOutputGenerationCommand: Command {
    var name: String { "run" }

    func execute(with arguments: [String]) {
        if arguments.count != 1 {
            TerminalManager.shared.currentTerminal?.addText("Invalid arguement", type: .error)
            return
        }
        
        let ticker = arguments[0]
        
        TickerDownload.shared.getAlpacaStock(ticker: ticker.uppercased(), year: 4) { messageReturn, stockAggregate in
            DispatchQueue.main.async {
                guard let stockAggregate = stockAggregate else {
                    TerminalManager.shared.currentTerminal?.addText(messageReturn, type: .error)
                    return
                }
                MLDatasetGenerator.shared.setAggregateForGeneration(aggregate: stockAggregate)
                ChartManager.shared.chartStock(stockAggregate)
                ChartManager.shared.currentStockView?.mouseDelegate = MLDatasetGenerator.shared
                TerminalManager.shared.currentTerminal?.addText("Charting: \(ticker.uppercased())", type: .normal)
            }
        }
    }
}
