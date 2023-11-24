//
//  HelpCommand.swift
//  stockInterface
//
//  Created by Jackson Tubbs on 11/11/23.
//

import Foundation

class HelpCommand: Command {
    var name: String { "help" }

    func execute(with arguments: [String]) {
        TerminalManager.shared.addText("chart xxx: Chart A Ticker Online", type: .normal)
        TerminalManager.shared.addText("c xxx: Chart A Ticker and indicator from saved indicator and aggregate data", type: .normal)
        TerminalManager.shared.addText("\"getAllTickers\" Download list of every ticker", type: .normal)
        TerminalManager.shared.addText("\"bulk start,stop,reset\": bulk downloading commands", type: .normal)
        TerminalManager.shared.addText("\"ci\" calculate and save indicator data", type: .normal)
    }
}
