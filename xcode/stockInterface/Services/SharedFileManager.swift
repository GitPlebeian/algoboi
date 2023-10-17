//
//  SharedFileManager.swift
//  stockInterface
//
//  Created by Jackson Tubbs on 10/16/23.
//

import Foundation

class SharedFileManager {
    
    static let shared = SharedFileManager()
    
    func writeDataToFileName(data: Data, fileName: String) {
        
        var url = FileManager.default.urls(for: .desktopDirectory, in: .userDomainMask).first!
        url = url.appendingPathComponent("/algoboi/shared/\(fileName).json")
        
        do {
            try data.write(to: url)
        } catch let e {
            fatalError("Error writing file to file name: \(e)")
        }
    }
    
    func writeStockAggregateToTestFile(_ model: StockAggregate) {
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        
        do {
            let data = try encoder.encode(model)
            writeDataToFileName(data: data, fileName: "bob")
        } catch let e {
            fatalError("Error encoding stock Aggregate: \(e)")
        }
        
//        let data = encoder.json
    }
}
