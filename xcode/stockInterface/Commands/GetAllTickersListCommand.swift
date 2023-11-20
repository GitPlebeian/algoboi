//
//  GetAllTickersListCommand.swift
//  stockInterface
//
//  Created by CHONK on 11/20/23.
//

import Foundation

class GetAllTickersListCommand: Command {
    var name: String { "getAllTickers" }

    var isCurrentlyDownloading = false
    
    lazy var processTickerArray: ([TickerNameModel]?, Bool) -> Bool = { tickerArray, isDone in
        guard let arr = tickerArray else {
            return false
        }
        DispatchQueue.main.async {
            if isDone {
                TerminalManager.shared.addText("Done")
                self.isCurrentlyDownloading = false
                AllTickersController.shared.saveToDisk()
                return
            }
            AllTickersController.shared.appendTickers(arr)
            TerminalManager.shared.addText("Saved Tickers: \(AllTickersController.shared.getCount())")
        }
        if isDone {
            return false
        }
        return true
    }
    
    func execute(with arguments: [String]) {
        if isCurrentlyDownloading == true {
            TerminalManager.shared.addText("Download already in progress", type: .error)
            return
        }
        isCurrentlyDownloading = true
        AllTickersController.shared.resetAllTickers()
        TickerDownload.shared.getEveryTicker(handler: processTickerArray)
    }
}
