//
//  ChartRandomCommand.swift
//  stockInterface
//
//  Created by Jackson Tubbs on 11/18/23.
//

import Foundation

class ChartRandomCommand: Command {
    var name: String { "chartRandom" }

    func execute(with arguments: [String]) {
        let aggregate = StockCalculations.GenerateNetZeroRandomAggregate(length: 2000)
        ChartManager.shared.chartStock(aggregate)
    }
}
