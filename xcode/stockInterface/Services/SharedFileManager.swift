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
        url = url.appendingPathComponent("/algoboi/shared/\(fileName)")
        
        do {
            try data.write(to: url)
        } catch let e {
            fatalError("Error writing file to file name: \(e)")
        }
    }
    
    func writeMLTrainingDataToFile(_ model: MLTrainingData) {
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        
        do {
            let data = try encoder.encode(model.toNestedArray())
            writeDataToFileName(data: data, fileName: "test.json")
        } catch let e {
            fatalError("Error encoding stock Aggregate: \(e)")
        }
        
//        let data = encoder.json
    }
}
