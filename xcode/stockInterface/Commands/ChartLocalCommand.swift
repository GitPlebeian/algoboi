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
            var indicator = try jsonDecoder.decode(IndicatorData.self, from: indicatorData)
            
            let auxSets = StockCalculations.GetAuxSetsForAggregate(aggregate: aggregate)
            
            // Predictions
            for i in 0..<aggregate.candles.count {
                var predictions: [Float] = []
                for j in 1...5 {
                    if let prediction = MLPredictor1.shared.makePrediction(indicatorData: indicator, index: i, candlesToTarget: j) {
                        predictions.append(prediction)
                    } else {
                        predictions.append(999)
                    }
                }
                indicator.predicted1CandleOut.append(predictions[0] * 100)
                indicator.predicted2CandleOut.append(predictions[1] * 100)
                indicator.predicted3CandleOut.append(predictions[2] * 100)
                indicator.predicted4CandleOut.append(predictions[3] * 100)
                indicator.predicted5CandleOut.append(predictions[4] * 100)
                
            }
            
            ChartManager.shared.chartStock(aggregate)
            ChartManager.shared.setIndicatorData(indicator)
            ChartManager.shared.setAuxSets(auxSets: auxSets)
        } catch let e {
            TerminalManager.shared.addText("Unable to unwrap data into aggregate: \(e)", type: .error)
        }
        
    }
}

extension ChartLocalCommand: StockViewMouseDelegate {
    func candleHoverChanged(index: Int) {
        
    }
}
