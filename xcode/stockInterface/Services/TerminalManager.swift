//
//  TerminalManager.swift
//  stockInterface
//
//  Created by Jackson Tubbs on 11/11/23.
//

import Foundation

class TerminalManager {
    
    static let shared = TerminalManager()
    
    private var commands: [String: Command] = [:]
    
    weak var currentTerminal: ScrollingTerminalView?
    
    private init() {
        registerCommand(HelpCommand())
    }

    func registerCommand(_ command: Command) {
        commands[command.name] = command
    }
    
    func enterCommand(command: String) {
        currentTerminal?.addText(command, type: .userInput)
        let components = command.split(separator: " ").map(String.init)
        guard let commandName = components.first, let command = commands[commandName] else {
            currentTerminal?.addText("Invalid Command", type: .error)
            return
        }
        
        let arguments = Array(components.dropFirst())
        command.execute(with: arguments)
    }
}
