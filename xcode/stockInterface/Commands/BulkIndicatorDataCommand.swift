//
//  CalculateAllStocksCommand.swift
//  stockInterface
//
//  Created by CHONK on 11/20/23.
//

import Foundation

class BulkIndicatorDataCommand: Command {
    var name: String { "ci" }

    func execute(with arguments: [String]) {
        HistoricalDataController.shared.calculateIndicatorData()
    }
}
