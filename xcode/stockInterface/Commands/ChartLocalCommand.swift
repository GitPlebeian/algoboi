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

        let ticker = arguments[0].uppercased()
        guard let aggregateData = SharedFileManager.shared.getDataFromFile("/historicalData/\(ticker).json") else {
            TerminalManager.shared.addText("Aggregate File does not exist", type: .error)
            return
        }
        
        var aggregate: StockAggregate!
        var indicator: IndicatorData!
        do {
            let jsonDecoder = JSONDecoder()
            aggregate = try jsonDecoder.decode(StockAggregate.self, from: aggregateData)
            
            if arguments.contains("i") {
                if let i = StockCalculations.GetIndicatorData(aggregate: aggregate) {
                    indicator = i
                } else {
                    TerminalManager.shared.addText("Stock length less than 200", type: .error)
                    return
                }
            } else {
                if let id = SharedFileManager.shared.getDataFromFile("/indicatorData/\(ticker).json") {
                    indicator = try jsonDecoder.decode(IndicatorData.self, from: id)
                } else if let i = StockCalculations.GetIndicatorData(aggregate: aggregate) {
                    indicator = i
                } else {
                    TerminalManager.shared.addText("Stock length less than 200", type: .error)
                    return
                }
            }
            
            
        } catch {
            if let i = StockCalculations.GetIndicatorData(aggregate: aggregate) {
                indicator = i
            } else {
                TerminalManager.shared.addText("Stock length less than 200", type: .error)
                return
            }
        }
        let auxSets = StockCalculations.GetAuxSetsForAggregate(aggregate: aggregate)
        
        ChartManager.shared.chartStock(aggregate)
        ChartManager.shared.setIndicatorData(indicator)
        ChartManager.shared.setAuxSets(auxSets: auxSets)
        
    }
}

extension ChartLocalCommand: StockViewMouseDelegate {
    func candleHoverChanged(index: Int) {
        
    }
}
