//
//  DownloadEveryStock.swift
//  stockInterface
//
//  Created by CHONK on 11/20/23.
//

import Foundation

class DownloadEveryStockCommand: Command {
    var name: String { "bulk" }

    func execute(with arguments: [String]) {
        if arguments.count != 1 {
            TerminalManager.shared.addText("Arguement count must be of length 1. IE \"bulk start,stop,resume\"", type: .error)
            return
        }
        let argument = arguments[0]
        
        switch argument {
        case "start": HistoricalDataController.shared.startDownload()
        case "stop": HistoricalDataController.shared.stopDownload()
        case "reset": HistoricalDataController.shared.reset()
        default:
            TerminalManager.shared.addText("Invalid argument IE \"bulk start,stop,resume\"", type: .error)
        }
    }
}
