//
//  ChartCommand.swift
//  stockInterface
//
//  Created by Jackson Tubbs on 11/11/23.
//

import Foundation

class ChartCommand: Command {
    var name: String { "chart" }

    func execute(with arguments: [String]) {
        if arguments.count != 1 {
            TerminalManager.shared.currentTerminal?.addText("Arguement count must be of length 1. IE \"chart aapl\"", type: .error)
        }
        let ticker = arguments[0]
        TickerDownload.shared.getAlpacaStock(ticker: ticker.uppercased(), year: 4) { messageReturn, stockAggregate in
            DispatchQueue.main.async {
                guard let stockAggregate = stockAggregate else {return}
//                self.stockView.stockAggregate = stockAggregate
                
                //                SharedFileManager.shared.writeMLTrainingDataToFile(StockCalculations.ConvertStockAggregateToMLTrainingData(stockAggregate))
            }
        }
    }
}
