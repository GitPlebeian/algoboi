//
//  SPYController.swift
//  stockInterface
//
//  Created by CHONK on 2/10/24.
//

import Foundation

class SPYController {
    
    static let shared = SPYController()
    
    var spyAggregate: StockAggregate!
    var spyIndicator: IndicatorData!
    
    func loadSPYAggregate() {
        guard let spyAggregateData = SharedFileManager.shared.getDataFromFile("/historicalData/VOO.json") else {
            TerminalManager.shared.addText("VOO aggregate File does not exist", type: .error)
            return
        }
        let decoder = JSONDecoder()
        do {
            spyAggregate = try decoder.decode(StockAggregate.self, from: spyAggregateData)
        } catch {
            TerminalManager.shared.addText("Unable to decode VOO", type: .error)
            return
        }
        if let indicatorData = SharedFileManager.shared.getDataFromFile("/indicatorData/VOO.json") {
            do {
                self.spyIndicator = try decoder.decode(IndicatorData.self, from: indicatorData)
            } catch {
                self.spyIndicator = StockCalculations.GetIndicatorData(aggregate: spyAggregate)!
                SharedFileManager.shared.writeCodableToFileNameInShared(codable: spyIndicator, fileName: "/indicatorData/VOO.json")
            }
        } else {
            self.spyIndicator = StockCalculations.GetIndicatorData(aggregate: spyAggregate)!
            SharedFileManager.shared.writeCodableToFileNameInShared(codable: spyIndicator, fileName: "/indicatorData/VOO.json")
        }
    }
}
