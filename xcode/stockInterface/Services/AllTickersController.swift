//
//  AllTickersController.swift
//  stockInterface
//
//  Created by CHONK on 11/20/23.
//

import Foundation

class AllTickersController {
    
    static let shared = AllTickersController()

    private var allTickers: [TickerNameModel] = []
    
    // MARK: Public
    
    func resetAllTickers() {
        allTickers = []
    }
    
    func appendTickers(_ arr: [TickerNameModel]) {
        allTickers.append(contentsOf: arr)
    }
    
    func insertTickerAtFront (_ ticker: TickerNameModel) {
        allTickers.insert(ticker, at: 0)
    }
    
    func getCount() -> Int {return allTickers.count}
    
    func getAllTickers() -> [TickerNameModel] {return allTickers}
    
    func saveToDisk() {
        SharedFileManager.shared.writeCodableToFileNameInShared(codable: allTickers, fileName: "allTicker.json")
    }
    
    func loadFromDisk() {
        guard let data = SharedFileManager.shared.getDataFromFile("allTicker.json") else {
            TerminalManager.shared.addText("Could not load All Tickers file because it does not exists. Run \"getAllTickers\" to download data", type: .error)
            return
        }
        do {
            let jsonDecoder = JSONDecoder()
            let array = try jsonDecoder.decode([TickerNameModel].self, from: data)
            self.allTickers = array
            TerminalManager.shared.addText("Loaded \(array.count) All Tickers From Disk")
        } catch let e {
            TerminalManager.shared.addText("Error decoding allTickers file: \(e)", type: .error)
        }
    }
}
