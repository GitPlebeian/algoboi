//
//  ChartLocalCommand.swift
//  stockInterface
//
//  Created by CHONK on 11/20/23.
//

import Cocoa

class ChartLocalCommand: Command {
    var name: String { "c" }

    func execute(with arguments: [String]) {
        if arguments.count != 1 {
            TerminalManager.shared.addText("Arguement count must be of length 1. IE \"chartLocal aapl\"", type: .error)
            return
        }
        let ticker = arguments[0].uppercased()
        guard let aggregateData = SharedFileManager.shared.getDataFromFile("/historicalData/\(ticker).json") else {
            TerminalManager.shared.addText("Aggregate File does not exist", type: .error)
            return
        }
        guard let indicatorData = SharedFileManager.shared.getDataFromFile("/indicatorData/\(ticker).json") else {
            TerminalManager.shared.addText("Indicator File does not exist", type: .error)
            return
        }
            
        do {
            let jsonDecoder = JSONDecoder()
            let aggregate = try jsonDecoder.decode(StockAggregate.self, from: aggregateData)
            let indicator = try jsonDecoder.decode(IndicatorData.self, from: indicatorData)
            ChartManager.shared.chartStock(aggregate)
            ChartManager.shared.setIndicatorData(indicator)
        } catch let e {
            TerminalManager.shared.addText("Unable to unwrap data into aggregate: \(e)", type: .error)
        }
        
    }
}

extension ChartLocalCommand: StockViewMouseDelegate {
    func candleHoverChanged(index: Int) {
        
    }
}
