//
//  FileManager.swift
//  stockInterface
//
//  Created by CHONK on 10/16/23.
//

import Foundation

class FileManager {
    
    static let shared = FileManager()
    
    func writeStockAggregateToSharedFoler(entity: StockAggregate) {
        
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        
        var jsonData: Data
        do {
            jsonData = try encoder.encode(entity)
        } catch let e {
            print("Faile to encode entity: \(e)")
        }
    }
}
