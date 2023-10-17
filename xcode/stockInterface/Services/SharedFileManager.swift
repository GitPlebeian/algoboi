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
        
//        guard let resourcePath = Bundle.main.resourcePath else {
//            print("Could not get resource path.")
//            return
//        }
//        
//        // 2. Construct a URL for the "algoboi" folder.
//        // Assuming the structure is: algoboi > Xcode > YourApp.app
//        let algoboiFolderURL = URL(fileURLWithPath: resourcePath).deletingLastPathComponent().deletingLastPathComponent()
//        
//        let fileURL = algoboiFolderURL.appendingPathComponent(fileName)
        
        let desktopURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        print(desktopURL)
//
//        let projectDirectory = FileManager.default.currentDirectoryPath
//        let directoryAbove = URL(fileURLWithPath: projectDirectory).deletingLastPathComponent()
//        let fileURL = directoryAbove.appendingPathComponent("\(fileName).json")
//        do {
//            try data.write(to: fileURL)
//        } catch let e {
//            fatalError("Error writing file to file name: \(e)")
//        }
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
