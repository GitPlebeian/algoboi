//
//  ChartAllTimeCommand.swift
//  stockInterface
//
//  Created by CHONK on 11/20/23.
//

import Foundation

class ChartAllTimeCommand: Command {
    var name: String { "chartAllTime" }

    func execute(with arguments: [String]) {
        if arguments.count != 1 {
            TerminalManager.shared.addText("Arguement count must be of length 1. IE \"chartAllTime aapl\"", type: .error)
            return
        }
        let ticker = arguments[0]
        TickerDownload.shared.getAlpacaStock(ticker: ticker.uppercased(), year: 30) { messageReturn, stockAggregate in
            DispatchQueue.main.async {
                guard let stockAggregate = stockAggregate else {
                    TerminalManager.shared.addText(messageReturn, type: .error)
                    return
                }
                ChartManager.shared.chartStock(stockAggregate)
                TerminalManager.shared.addText("Charting: \(ticker.uppercased())", type: .normal)
            }
        }
    }
}
