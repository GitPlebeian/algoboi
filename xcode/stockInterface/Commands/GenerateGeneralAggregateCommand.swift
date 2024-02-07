//
//  GenerateGeneralAggregateCommand.swift
//  stockInterface
//
//  Created by CHONK on 2/6/24.
//

import Foundation

class GenerateGeneralAggregateCommand: Command {
    
    var name: String { "ga" }

    func execute(with arguments: [String]) {

        HistoricalDataController.shared.generateAndSaveGeneralAggregate()
    }
}
