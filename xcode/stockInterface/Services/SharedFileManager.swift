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
    }
    
    func getFileNamesFromPlaybackFolder() -> [String]? {
        do {
            var url = FileManager.default.urls(for: .desktopDirectory, in: .userDomainMask).first!
            url = url.appendingPathComponent("algoboi/shared/")
            var fileNames = try FileManager.default.contentsOfDirectory(atPath: "/Users/\(NSFullUserName())/Desktop/algoboi/shared/playback")
            fileNames = fileNames.filter { $0.hasSuffix(".json") }
            return fileNames
        } catch {
            print("Error retrieving file names: \(error)")
            return nil
        }
    }
    
    func getDataFromPlaybackFile(_ fileName: String) -> Data? {
        let urlString = "/Users/\(NSFullUserName())/Desktop/algoboi/shared/playback/\(fileName)"
        let fileURL = URL(string: urlString)!
        print(fileURL)
        
        do {
            let data = try Data(contentsOf: fileURL)
            return data
        } catch let e {
            print(e)
        }
        return nil
    }
}
