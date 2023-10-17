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
        let projectDirectory = FileManager.default.currentDirectoryPath
        let directoryAbove = URL(fileURLWithPath: projectDirectory).deletingLastPathComponent()
        let fileURL = directoryAbove.appendingPathComponent("\(fileName).json")
        do {
            try data.write(to: fileURL)
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
