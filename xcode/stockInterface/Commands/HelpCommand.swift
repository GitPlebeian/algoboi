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
        TerminalManager.shared.addText("Help: Lists Commands\nChart XXXX: Chart A Ticker", type: .normal)
    }
}
