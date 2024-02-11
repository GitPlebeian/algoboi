//
//  BacktestCurrentCommand.swift
//  stockInterface
//
//  Created by CHONK on 2/6/24.
//

import Foundation

class BacktestCurrentCommand: Command {
    
    var name: String { "bc" }

    func execute(with arguments: [String]) {

        if ChartManager.shared.currentlyChartingStock() == false {
            TerminalManager.shared.addText("You are currently not charting a stock. use the \"c\" command.", type: .error)
            return
        }
        
        TerminalManager.shared.addText("Backtesting Current")
        
        BacktestController.shared.backtestCurrentChartedStock()
    }
}

class BacktestAllCommand: Command {
    
    var name: String { "b" }

    func execute(with arguments: [String]) {
        if arguments.count != 1 {
            TerminalManager.shared.addText("Arguement count must be of length 1. IE \"b start,stop", type: .error)
            return
        }
        let argument = arguments[0]
        switch argument {
        case "start": BacktestController.shared.startFullBacktest()
        case "stop": BacktestController.shared.stopFullBacktest()
        default:
            TerminalManager.shared.addText("Invalid arguement", type: .error)
        }
//        BacktestController.shared.backtestAllStocks()
    }
}
