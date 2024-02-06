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
